//! Byte-instance collator: `prompt ++ target ++ EOS`, padded to batch max.
//!
//! Byte 0 is PAD/EOS (task alphabets are printable ASCII, never 0x00).
//! The loss mask scores target bytes + the EOS terminator only; prompt bytes
//! and pads are masked out. Greedy eval decodes until EOS.

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use crate::tasks::Instance;

/// Reserved byte ids. Task alphabets are printable ASCII (plus `\n` as a
/// structural separator), so these never occur in task content.
/// PAD (0x00) is batch padding: blocked from attention, never scored.
/// EOS (0x01) is the sequence terminator and the only scored control byte.
///
/// The split matters. Sharing one id floods inputs with pads while half
/// the supervised targets are the same id, teaching the model that id 0
/// is likely everywhere (EOS-first greedy decode).
pub const PAD: u8 = 0;
pub const EOS: u8 = 1;

/// Padded-length buckets. Every batch/decoded sequence is padded to a bucket
/// edge so fused-JIT shapes take only a handful of values per run: without
/// this, each unique T recompiles the whole fused graph (~13ms/load, 92% of
/// CUDA API time in the profile). Lengths stay exact; masks stay correct.
pub const BUCKET_EDGES: &[usize] = &[64, 128, 256, 512];

pub fn bucket_len(n: usize) -> usize {
    for edge in BUCKET_EDGES {
        if *edge >= n {
            return *edge;
        }
    }
    panic!("sequence length exceeds top bucket");
}

/// [`bucket_len`] that also refuses to exceed a model's `max_seq_len`.
///
/// The training path already asserts the *real* length against
/// `max_seq_len`, but that check misses the padding: a row of real length
/// 200 buckets to 256, and eval decode adds `max_new` on top, so a
/// `max_new`-extended row can bucket past `max_seq_len` even when every real
/// length is legal. RoPE is sized to `max_seq_len`, so the result is an
/// opaque broadcast panic from deep inside attention rather than a message
/// naming the knob that is wrong.
pub fn bucket_len_within(n: usize, max_seq_len: usize, what: &str) -> usize {
    let b = bucket_len(n);
    assert!(
        b <= max_seq_len,
        "{what}: padded length {b} (bucket for a {n}-length sequence) exceeds \
         model.max_seq_len {max_seq_len}. Raise max_seq_len to at least the top \
         bucket edge, or lower eval_max_new / shorten the prompts."
    );
    b
}

pub struct Collated<B: Backend> {
    /// `[B, T]` padded with PAD.
    pub tokens: Tensor<B, 2, Int>,
    /// `[B, T]` padded with PAD.
    pub targets: Tensor<B, 2, Int>,
    /// `[B, T]` float: 1.0 on target bytes + EOS, else 0.0.
    pub loss_mask: Tensor<B, 2>,
    /// Real (unpadded) lengths for the attention pad mask.
    pub lengths: Vec<usize>,
    /// Prompt lengths: target region of row r is `[prompt_lens[r], lengths[r])`.
    pub prompt_lens: Vec<usize>,
}

/// Collate prebuilt sequences (prompt ++ target ++ EOS each) with explicit
/// prompt lengths. [`collate`] is the Instance-based frontend.
pub fn collate_seqs<B: Backend>(
    seqs: Vec<Vec<u8>>,
    prompt_lens: Vec<usize>,
    device: &B::Device,
) -> Collated<B> {
    assert!(!seqs.is_empty());
    assert_eq!(seqs.len(), prompt_lens.len());
    let t_raw = seqs.iter().map(|s| s.len()).max().unwrap_or(1);
    let t = bucket_len(t_raw);
    let b = seqs.len();

    let mut tok = Vec::with_capacity(b * t);
    let mut tgt = Vec::with_capacity(b * t);
    let mut mask = Vec::with_capacity(b * t);
    let mut lengths = Vec::with_capacity(b);
    for (seq, prompt_len) in seqs.iter().zip(prompt_lens.iter()) {
        lengths.push(seq.len());
        for i in 0..t {
            // Standard shifted causal LM: input[p] = seq[p-1] (input[0] =
            // PAD), so logits[p] predicts seq[p] from strictly earlier
            // bytes. Scoring position p against seq[p] with unshifted inputs
            // leaks the label into its own key set (causal attention sees
            // key p) and trains index-selection instead of induction.
            let byte = if i < seq.len() { seq[i] } else { PAD };
            let input = if i == 0 || i > seq.len() {
                PAD
            } else {
                seq[i - 1]
            };
            tok.push(input as i64);
            tgt.push(byte as i64);
            let scored = i >= *prompt_len && i < seq.len();
            mask.push(if scored { 1.0f32 } else { 0.0f32 });
        }
    }
    Collated {
        tokens: Tensor::from_data(TensorData::new(tok, [b, t]), device),
        targets: Tensor::from_data(TensorData::new(tgt, [b, t]), device),
        loss_mask: Tensor::from_data(TensorData::new(mask, [b, t]), device),
        lengths,
        prompt_lens,
    }
}

/// Collate instances into one padded causal-LM batch.
pub fn collate<B: Backend>(instances: &[Instance], device: &B::Device) -> Collated<B> {
    let seqs: Vec<Vec<u8>> = instances
        .iter()
        .map(|inst| {
            let mut s = inst.prompt.clone();
            s.extend_from_slice(&inst.target);
            s.push(EOS);
            s
        })
        .collect();
    let prompt_lens: Vec<usize> = instances.iter().map(|inst| inst.prompt.len()).collect();
    collate_seqs(seqs, prompt_lens, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::demo::InstanceInfo;
    use crate::test_backend::{TestBackend, test_device};

    fn inst(prompt: &[u8], target: &[u8]) -> Instance {
        Instance {
            prompt: prompt.to_vec(),
            target: target.to_vec(),
            info: InstanceInfo::test_info(),
        }
    }

    #[test]
    fn collate_masks_prompt_and_pad() {
        let device = test_device();
        let batch = collate::<TestBackend>(&[inst(b"ab->", b"cd"), inst(b"abc->", b"d")], &device);
        // Bucketed to edge 64; real lengths recorded exactly.
        assert_eq!(batch.tokens.dims(), [2, 64]);
        assert_eq!(batch.lengths, vec![7, 7]);
        let m = batch
            .loss_mask
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        // Row 0: prompt "ab->" (4) masked, "cd"+EOS scored, rest pad.
        assert_eq!(&m[..8], &[0., 0., 0., 0., 1., 1., 1., 0.]);
        // Row 1 starts at offset 64: prompt "abc->" (5) masked, "d"+EOS scored.
        assert_eq!(&m[64..72], &[0., 0., 0., 0., 0., 1., 1., 0.]);
    }

    #[test]
    fn collate_shifts_inputs_for_causal_lm() {
        // tokens[p] = seq[p-1] (tokens[0] = PAD): no scored position can
        // attend its own label byte. Targets stay unshifted.
        use burn::tensor::{DType, TensorData};
        let device = test_device();
        let batch = collate::<TestBackend>(&[inst(b"ab", b"cd")], &device);
        fn ints(data: &TensorData) -> Vec<i64> {
            match data.dtype {
                DType::I64 => data.as_slice::<i64>().unwrap().to_vec(),
                DType::I32 => data
                    .as_slice::<i32>()
                    .unwrap()
                    .iter()
                    .map(|v| *v as i64)
                    .collect(),
                d => panic!("{d:?}"),
            }
        }
        let tok = ints(&batch.tokens.into_data());
        let tgt = ints(&batch.targets.into_data());
        // seq = [a b c d EOS]=[97 98 99 100 1]; tokens shifted right by 1.
        assert_eq!(&tok[..6], &[0, 97, 98, 99, 100, 1]);
        assert_eq!(&tgt[..6], &[97, 98, 99, 100, 1, 0]);
    }

    #[test]
    fn bucket_edges_canonicalize() {
        assert_eq!(bucket_len(1), 64);
        assert_eq!(bucket_len(64), 64);
        assert_eq!(bucket_len(65), 128);
        assert_eq!(bucket_len(512), 512);
    }

    /// The guard that names the knob instead of letting RoPE fail with an
    /// opaque broadcast error. Found by running a 20-step multi-stage eval
    /// with `max_seq_len: 256` against prompts that bucket to 512.
    #[test]
    fn bucket_len_within_names_the_knob_when_padded_length_overflows() {
        // Legal: the padded length fits.
        assert_eq!(bucket_len_within(200, 256, "t"), 256);
        assert_eq!(bucket_len_within(10, 512, "t"), 64);
        // Illegal: real length 200 is legal on its own, but pads to 256 and
        // that is still fine, so use a case where the bucket itself overflows.
        let msg = std::panic::catch_unwind(|| bucket_len_within(300, 256, "eval decode"))
            .expect_err("300 buckets to 512 > 256, must be rejected");
        let msg = msg
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| msg.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(msg.contains("eval decode"), "context missing: {msg}");
        assert!(msg.contains("max_seq_len"), "knob not named: {msg}");
        assert!(msg.contains("512"), "padded length not reported: {msg}");
    }

    /// The specific gap the training assert misses: a real length under the
    /// limit whose *padded* length, or whose `max_new`-extended length, is
    /// not. Real 250 + max_new 8 = 258, which buckets to 512.
    #[test]
    fn guard_catches_the_case_the_training_assert_misses() {
        let threw =
            std::panic::catch_unwind(|| bucket_len_within(250 + 8, 256, "eval decode")).is_err();
        assert!(
            threw,
            "the max_new overflow that the training assert cannot see went unreported"
        );
    }
}
