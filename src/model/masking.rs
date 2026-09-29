//! Attention masks: key-padding construction plus the additive bias views.
//! Attention receives semantic masks from here instead of assembling them.

use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};

/// Key-padding mask `[B, T]` (`true` = pad, blocked from attention).
/// Bytes 0x00 (PAD) and 0x01 (EOS) are reserved; task alphabets never
/// contain them, so pad positions are unambiguous.
/// `lens.lower_equal(pos)` is true exactly on pads (`len <= pos`); position
/// 0 stays open so a fully-padded row never softmaxes over an empty set
/// (NaN would poison shared-weight gradients).
pub fn pad_mask<B: Backend>(lengths: &[usize], t: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
    let b = lengths.len();
    let pos = Tensor::<B, 1, Int>::arange(0..t as i64, device)
        .unsqueeze_dim::<2>(0)
        .repeat_dim(0, b);
    let data: Vec<i64> = lengths.iter().map(|l| *l as i64).collect();
    let lens = Tensor::<B, 1, Int>::from_data(TensorData::new(data, [b]), device)
        .unsqueeze_dim::<2>(1)
        .repeat_dim(1, t);
    let first = Tensor::<B, 1, Int>::arange(0..t as i64, device)
        .lower_equal_elem(0)
        .unsqueeze_dim::<2>(0)
        .repeat_dim(0, b);
    lens.lower_equal(pos).bool_and(first.bool_not())
}

/// Additive causal bias `[1, 1, T, T]`: 0 where `key <= query`, `-1e30`
/// where blocked (exactly 0 after exp; finite, so no NaN).
pub fn causal_bias<B: Backend>(t: usize, device: &B::Device) -> Tensor<B, 4> {
    let qi = Tensor::<B, 1, Int>::arange(0..t as i64, device).unsqueeze_dim::<2>(1);
    let kj = Tensor::<B, 1, Int>::arange(0..t as i64, device).unsqueeze_dim::<2>(0);
    kj.lower_equal(qi)
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0)
        .float()
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .mul_scalar(-1e30)
}

/// Additive key-padding bias `[B, 1, 1, T]` from a [`pad_mask`]: 0 on real
/// keys, `-1e30` on pads. Adds with [`causal_bias`] (both blocked -> -2e30,
/// still exactly 0 after exp).
pub fn key_bias<B: Backend>(pad: Tensor<B, 2, Bool>) -> Tensor<B, 4> {
    pad.bool_not()
        .unsqueeze_dim::<3>(1)
        .unsqueeze_dim::<4>(2)
        .float()
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .mul_scalar(-1e30)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_backend::{TestBackend, test_device};

    #[test]
    fn pad_mask_values_are_correct() {
        // true = pad (blocked). Row len 0 keeps position 0 open (NaN guard).
        let device = test_device();
        let m = pad_mask::<TestBackend>(&[4, 0, 2], 4, &device);
        let v = m.float().into_data().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(
            v,
            vec![
                0., 0., 0., 0., //
                0., 1., 1., 1., //
                0., 0., 1., 1.,
            ]
        );
    }
}
