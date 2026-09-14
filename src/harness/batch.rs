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
            let byte = if i < seq.len() { seq[i] } else { PAD };
            tok.push(byte as i64);
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
        let batch = collate::<TestBackend>(
            &[inst(b"ab->", b"cd"), inst(b"abc->", b"d")],
            &device,
        );
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
    fn bucket_edges_canonicalize() {
        assert_eq!(bucket_len(1), 64);
        assert_eq!(bucket_len(64), 64);
        assert_eq!(bucket_len(65), 128);
        assert_eq!(bucket_len(512), 512);
    }
}
