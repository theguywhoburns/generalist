//! Byte-instance collator: `prompt ++ target ++ EOS`, padded to batch max.
//!
//! Byte 0 is PAD/EOS (task alphabets are printable ASCII, never 0x00).
//! The loss mask scores target bytes + the EOS terminator only; prompt bytes
//! and pads are masked out. Greedy eval decodes until EOS.

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use crate::tasks::Instance;

pub const EOS: u8 = 0;

pub struct Collated<B: Backend> {
    /// `[B, T]` padded with EOS(0).
    pub tokens: Tensor<B, 2, Int>,
    /// `[B, T]` padded with EOS(0).
    pub targets: Tensor<B, 2, Int>,
    /// `[B, T]` float: 1.0 on target bytes + EOS, else 0.0.
    pub loss_mask: Tensor<B, 2>,
    /// Real (unpadded) lengths for the attention pad mask.
    pub lengths: Vec<usize>,
}

/// Collate instances into one padded causal-LM batch.
pub fn collate<B: Backend>(instances: &[Instance], device: &B::Device) -> Collated<B> {
    assert!(!instances.is_empty());
    let seqs: Vec<Vec<u8>> = instances
        .iter()
        .map(|inst| {
            let mut s = inst.prompt.clone();
            s.extend_from_slice(&inst.target);
            s.push(EOS);
            s
        })
        .collect();
    let t = seqs.iter().map(|s| s.len()).max().unwrap_or(1);
    let b = seqs.len();

    let mut tok = Vec::with_capacity(b * t);
    let mut tgt = Vec::with_capacity(b * t);
    let mut mask = Vec::with_capacity(b * t);
    let mut lengths = Vec::with_capacity(b);
    for (seq, inst) in seqs.iter().zip(instances.iter()) {
        let prompt_len = inst.prompt.len();
        lengths.push(seq.len());
        for i in 0..t {
            let byte = if i < seq.len() { seq[i] } else { EOS };
            tok.push(byte as i64);
            tgt.push(byte as i64);
            let scored = i >= prompt_len && i < seq.len();
            mask.push(if scored { 1.0f32 } else { 0.0f32 });
        }
    }
    Collated {
        tokens: Tensor::from_data(TensorData::new(tok, [b, t]), device),
        targets: Tensor::from_data(TensorData::new(tgt, [b, t]), device),
        loss_mask: Tensor::from_data(TensorData::new(mask, [b, t]), device),
        lengths,
    }
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
        assert_eq!(batch.tokens.dims(), [2, 7]);
        assert_eq!(batch.lengths, vec![7, 7]);
        let m = batch
            .loss_mask
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        // Row 0: prompt "ab->" (4) masked, "cd"+EOS scored.
        assert_eq!(&m[..7], &[0., 0., 0., 0., 1., 1., 1.]);
        // Row 1: prompt "abc->" (5) masked, "d"+EOS scored.
        assert_eq!(&m[7..], &[0., 0., 0., 0., 0., 1., 1.]);
    }
}
