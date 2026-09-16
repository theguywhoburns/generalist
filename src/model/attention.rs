use burn::{
    module::Module,
    nn::{Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig},
    tensor::{Bool, Int, Tensor, backend::Backend},
};

/// Query-row tile for the chunked attention forward ([`MultiHeadAttention`]).
/// 64 divides every `T` bucket ({64, 128, 256, 512}), so bucketed batches
/// split evenly; any remainder is handled by a smaller final chunk.
pub const ATTN_QUERY_CHUNK: usize = 64;

/// Unfused causal MHA with RoPE. Unfused Q/K/V keeps one `d x d` input
/// covariance per matrix, which is what the Newton-Muon right-preconditioner
/// expects (no packed-QKV block handling needed).
#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    pub q: Linear<B>,
    pub k: Linear<B>,
    pub v: Linear<B>,
    pub o: Linear<B>,
    pub rope: RotaryEncoding<B>,
    pub use_rope: bool,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        use_rope: bool,
        device: &B::Device,
    ) -> Self {
        assert_eq!(d_model, n_heads * head_dim);
        let proj = || {
            LinearConfig::new(d_model, d_model)
                .with_bias(false)
                .init(device)
        };
        Self {
            q: proj(),
            k: proj(),
            v: proj(),
            o: proj(),
            rope: RotaryEncodingConfig::new(max_seq_len, head_dim).init(device),
            use_rope,
            n_heads,
            head_dim,
        }
    }

    fn split_heads(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, t, _] = x.dims();
        x.reshape([b, t, self.n_heads, self.head_dim])
            .swap_dims(1, 2)
    }

    /// Input/output `[B, T, D]`. No padding mask (use `forward_masked` for
    /// padded batches).
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_masked(x, None)
    }

    /// Like [`Self::forward`], with an optional key-padding mask `[B, T]`
    /// (`true` = pad position, blocked from attention).
    ///
    /// Implemented as explicit scored math (matmul + additive `-1e30` bias +
    /// softmax), NOT the fused attention op: as of burn 0.21 the fused CUDA
    /// path ignores explicit bool masks when `is_causal` is set (proven by
    /// `padding_masks_pad_rows` failing on CUDA while mask values verify
    /// correct). Explicit math is portable and auditable; query chunking below
    /// keeps the transient `[B, H, C, T]` tiles small at 1M scale.
    ///
    /// Memory design (no full-`T²` live set — see module docs in
    /// `newton_muon.rs` for the analogous documented-deviation style):
    /// the query axis is tiled in blocks of [`ATTN_QUERY_CHUNK`] rows, so no
    /// `[B, H, T, T]` f32 buffer is ever fully live at once: each chunk only
    /// materializes `[B, H, C, T]` tiles (scores, bias add-out, softmax
    /// shifted/exp/probs) plus chunk-sized backward recompute tiles, i.e.
    /// `O(C*T)` transient instead of `O(T²)`. Softmax is still applied over
    /// the FULL key axis per chunk, so the math is identical to the unfused
    /// reference (same op order per row; only cross-chunk grad-accumulation
    /// summation order in `dK`/`dV` can differ at `~1e-7` level).
    ///
    /// Why tiling instead of a fused cubecl kernel with a custom backward
    /// (which would also cut the autodiff tape, not just transients)? First,
    /// `burn::tensor::module::attention` does NOT help training memory:
    /// burn 0.21's `Autodiff` impl of the attention module op lowers to
    /// `attention_fallback` (plain tracked matmul/mask/softmax/matmul),
    /// so the full `T²` tape is kept anyway; the flash kernel only fires
    /// on the inner (inference) backend. Second, a hand-written fused kernel
    /// with recompute-backward needs a custom `Backward` impl, which lives
    /// in the `burn-autodiff` crate — not a direct dependency and not
    /// re-exported through `burn`, so it cannot be implemented here without
    /// adding a dependency (forbidden) nor reached via `burn::cubecl`
    /// (which only re-exports the core `cubecl` crate, not `burn-cubecl`'s
    /// tensor/runtime types).
    /// Tiling is therefore the strongest tape-neutral, portable reduction
    /// available with the public `burn` API: it removes every full-`T²`
    /// TRANSIENT (forward tiles and, more importantly, the backward
    /// `dProbs`/`dScores`/recomputed-exp tiles, which otherwise peak several
    /// `T²` buffers deep in the unrolled loop nest). The taped per-row
    /// intermediates (`shifted`, `exp` input for its backward, `probs` —
    /// one `T²` trio per block-step, chunk-summed) are unchanged.
    /// No dropout exists; nothing to fuse there.
    pub fn forward_masked(
        &self,
        x: Tensor<B, 3>,
        key_pad: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [b, t, _] = x.dims();
        let device = x.device();
        let q = self.split_heads(self.q.forward(x.clone()));
        let q = if self.use_rope { self.rope.forward(q) } else { q };
        let k = self.split_heads(self.k.forward(x.clone()));
        let k = if self.use_rope { self.rope.forward(k) } else { k };
        let v = self.split_heads(self.v.forward(x));

        // scores[b,h,i,j] = q[b,h,i] . k[b,h,j] / sqrt(head_dim)
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let k_t = k.swap_dims(2, 3);

        // Additive -1e30 bias, broadcastable: causal [1,1,T,T] plus an
        // optional key-padding [B,1,1,T]. Blocked-by-either -> -1e30
        // (both -> -2e30, still exactly 0 after exp), unblocked -> 0:
        // identical softmax to the old expanded [B,H,T,T] keep-mask (~2x
        // scores-sized tensors: the old full-size bools plus full-size
        // float bias were the largest per-step tape entries and OOMed
        // batch 6 on the first T=512 micro).
        let qi = Tensor::<B, 1, Int>::arange(0..t as i64, &device).unsqueeze_dim::<2>(1);
        let kj = Tensor::<B, 1, Int>::arange(0..t as i64, &device).unsqueeze_dim::<2>(0);
        let causal_bias = kj
            .lower_equal(qi)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0)
            .float()
            .mul_scalar(-1.0)
            .add_scalar(1.0)
            .mul_scalar(-1e30);
        let key_bias = key_pad.map(|pad| {
            pad.bool_not()
                .unsqueeze_dim::<3>(1)
                .unsqueeze_dim::<4>(2)
                .float()
                .mul_scalar(-1.0)
                .add_scalar(1.0)
                .mul_scalar(-1e30)
        });

        // Query-chunked scored attention: per chunk the math is exactly the
        // unfused reference above (same bias values, same softmax over the
        // full key axis), only `[B,H,C,T]` tiles are ever live.
        // Blocked -> -1e30 attends to nothing (exp underflows to exactly 0;
        // finite so a fully-blocked row can't NaN — and pos0 is always kept
        // by the pad_mask guard anyway).
        let mut outs = Vec::new();
        let mut start = 0;
        while start < t {
            let c = (t - start).min(ATTN_QUERY_CHUNK);
            let scores = q
                .clone()
                .narrow(2, start, c)
                .matmul(k_t.clone())
                .mul_scalar(scale);
            let scores = scores + causal_bias.clone().narrow(2, start, c);
            let scores = match key_bias.clone() {
                None => scores,
                Some(bias) => scores + bias,
            };
            let probs = burn::tensor::activation::softmax(scores, 3);
            outs.push(probs.matmul(v.clone()));
            start += c;
        }
        let y = if outs.len() == 1 {
            outs.pop().expect("one chunk")
        } else {
            Tensor::cat(outs, 2)
        };

        let y = y
            .swap_dims(1, 2)
            .reshape([b, t, self.n_heads * self.head_dim]);
        self.o.forward(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::pad_mask;
    use crate::test_backend::{TestBackend, test_device};
    use burn::tensor::TensorData;

    fn test_mha() -> MultiHeadAttention<TestBackend> {
        // Production head geometry: H=4, head_dim=64.
        MultiHeadAttention::new(256, 4, 64, 512, true, &test_device())
    }

    /// Deterministic `O(1)`-magnitude input (no RNG dependence).
    fn test_input(b: usize, t: usize) -> Tensor<TestBackend, 3> {
        let vals: Vec<f32> = (0..b * t * 256)
            .map(|i| (i as f32 * 0.731 + 0.37).sin() * 0.8)
            .collect();
        Tensor::from_data(TensorData::new(vals, [b, t, 256]), &test_device())
    }

    /// Pre-chunking reference: full-`T²` explicit scored math, the exact old
    /// `forward_masked` body, sharing the module weights.
    fn reference_forward(
        mha: &MultiHeadAttention<TestBackend>,
        x: Tensor<TestBackend, 3>,
        key_pad: Option<Tensor<TestBackend, 2, Bool>>,
    ) -> Tensor<TestBackend, 3> {
        let [b, t, _] = x.dims();
        let device = x.device();
        let split = |x: Tensor<TestBackend, 3>| {
            let [b, t, _] = x.dims();
            x.reshape([b, t, mha.n_heads, mha.head_dim])
                .swap_dims(1, 2)
        };
        let q = split(mha.q.forward(x.clone()));
        let q = if mha.use_rope { mha.rope.forward(q) } else { q };
        let k = split(mha.k.forward(x.clone()));
        let k = if mha.use_rope { mha.rope.forward(k) } else { k };
        let v = split(mha.v.forward(x));
        let scale = 1.0 / (mha.head_dim as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);
        let qi = Tensor::<TestBackend, 1, Int>::arange(0..t as i64, &device).unsqueeze_dim::<2>(1);
        let kj = Tensor::<TestBackend, 1, Int>::arange(0..t as i64, &device).unsqueeze_dim::<2>(0);
        let causal_bias = kj
            .lower_equal(qi)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0)
            .float()
            .mul_scalar(-1.0)
            .add_scalar(1.0)
            .mul_scalar(-1e30);
        let scores = scores + causal_bias;
        let scores = match key_pad {
            None => scores,
            Some(pad) => {
                scores + pad
                    .bool_not()
                    .unsqueeze_dim::<3>(1)
                    .unsqueeze_dim::<4>(2)
                    .float()
                    .mul_scalar(-1.0)
                    .add_scalar(1.0)
                    .mul_scalar(-1e30)
            }
        };
        let probs = burn::tensor::activation::softmax(scores, 3);
        let y = probs.matmul(v);
        let y = y
            .swap_dims(1, 2)
            .reshape([b, t, mha.n_heads * mha.head_dim]);
        mha.o.forward(y)
    }

    fn max_abs_diff(a: Tensor<TestBackend, 3>, c: Tensor<TestBackend, 3>) -> f32 {
        let [b, t, d] = a.dims();
        let av = a.into_data().as_slice::<f32>().unwrap().to_vec();
        let cv = c.into_data().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(av.len(), b * t * d);
        assert_eq!(cv.len(), b * t * d);
        av.iter()
            .zip(cv.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn assert_finite(vals: &[f32], what: &str) {
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "{what} has non-finite entries"
        );
    }

    /// Epsilon justification: f32 has ~6e-8 relative precision; the QK dot
    /// accumulates `head_dim=64` products of `O(1)` values, so per-element
    /// rounding is `~1e-6` abs worst case, and chunked-vs-full matmul tiling
    /// may reorder the summation. `1e-5` is ~10x that bound.
    const FWD_TOL: f32 = 1e-5;
    /// Grad epsilon: on top of forward rounding, `dK`/`dV` accumulate one
    /// small matmul per query chunk (2 chunks at T=128), adding one more
    /// rounding step per element. `1e-4` is generous; observed is ~1e-6.
    const GRAD_TOL: f32 = 1e-4;

    #[test]
    fn chunked_matches_reference_no_pad() {
        for (b, t) in [(1usize, 64usize), (2, 128), (1, 96)] {
            let mha = test_mha();
            let x = test_input(b, t);
            let got = mha.forward_masked(x.clone(), None);
            let want = reference_forward(&mha, x, None);
            let diff = max_abs_diff(got.clone(), want);
            let vals = got.into_data().as_slice::<f32>().unwrap().to_vec();
            assert_finite(&vals, "chunked output");
            assert!(
                diff <= FWD_TOL,
                "B={b} T={t}: max abs fwd diff {diff} > {FWD_TOL}"
            );
        }
    }

    #[test]
    fn chunked_matches_reference_padded() {
        // Heavy padding with the production pos0-keep guard: every query row
        // keeps >= 1 open key, so no row softmaxes over an empty set.
        let device = test_device();
        for (lengths, t) in [(vec![64usize], 64usize), (vec![100, 37], 128)] {
            let b = lengths.len();
            let mha = test_mha();
            let x = test_input(b, t);
            let pad = pad_mask::<TestBackend>(&lengths, t, &device);
            let got = mha.forward_masked(x.clone(), Some(pad.clone()));
            let want = reference_forward(&mha, x, Some(pad));
            let diff = max_abs_diff(got.clone(), want);
            let vals = got.into_data().as_slice::<f32>().unwrap().to_vec();
            assert_finite(&vals, "chunked padded output");
            assert!(
                diff <= FWD_TOL,
                "lengths={lengths:?} T={t}: max abs fwd diff {diff} > {FWD_TOL}"
            );
        }
    }

    #[test]
    fn padding_edge_rows_stay_finite_and_match() {
        // Degenerate pad (bypasses the `pad_mask` pos0 guard on purpose):
        // row 0 keeps key 0 only, so query row 1 sees keys {0,1} both
        // pad-blocked. Scores there are all `-1e30..-2e30` (finite), the row
        // max is `-1e30`, and softmax stays finite (uniform over the
        // causally-open blocked keys, exact 0 past the causal edge) — the
        // chunked path must reproduce the reference bit-pattern here.
        let device = test_device();
        let (b, t) = (2usize, 64usize);
        let mha = test_mha();
        let x = test_input(b, t);
        let mut pad = vec![false; b * t];
        pad.iter_mut().take(t).skip(1).for_each(|p| *p = true);
        let pad = Tensor::<TestBackend, 2, Bool>::from_data(TensorData::new(pad, [b, t]), &device);
        let got = mha.forward_masked(x.clone(), Some(pad.clone()));
        let want = reference_forward(&mha, x, Some(pad));
        let gvals = got
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        assert_finite(&gvals, "degenerate-pad output");
        let diff = max_abs_diff(got, want);
        assert!(diff <= FWD_TOL, "degenerate pad diff {diff} > {FWD_TOL}");
    }

    #[test]
    fn nope_matches_reference_and_differs_from_rope() {
        let device = test_device();
        let (b, t) = (2usize, 96usize);
        let x = test_input(b, t);
        let pad = pad_mask::<TestBackend>(&[96, 41], t, &device);
        // Same weights, RoPE on vs off: build once, flip the flag.
        let mut mha = test_mha();
        let got_rope = mha.forward_masked(x.clone(), Some(pad.clone()));
        mha.use_rope = false;
        let got_nope = mha.forward_masked(x.clone(), Some(pad.clone()));
        let want_nope = reference_forward(&mha, x, Some(pad));
        let diff = max_abs_diff(got_nope.clone(), want_nope);
        // Position-sensitive input: RoPE must actually change the output.
        assert!(
            max_abs_diff(got_rope, got_nope.clone()) > 1e-3,
            "RoPE/NoPE unexpectedly identical on position-sensitive input"
        );
        let vals = got_nope.into_data().as_slice::<f32>().unwrap().to_vec();
        assert_finite(&vals, "nope output");
        assert!(diff <= FWD_TOL, "nope vs reference diff {diff} > {FWD_TOL}");
    }

    #[test]
    fn chunked_input_grads_match_reference() {
        for (b, t, lengths) in [
            (1usize, 64usize, vec![64usize]),
            (2usize, 128usize, vec![128, 41]),
        ] {
            let device = test_device();
            let mha = test_mha();
            let pad = pad_mask::<TestBackend>(&lengths, t, &device);

            let xc = test_input(b, t).require_grad();
            let loss_c = mha
                .forward_masked(xc.clone(), Some(pad.clone()))
                .sum();
            let grads_c = loss_c.backward();
            let gc = xc
                .grad(&grads_c)
                .expect("chunked input grad")
                .into_data()
                .as_slice::<f32>()
                .unwrap()
                .to_vec();

            let xr = test_input(b, t).require_grad();
            let loss_r = reference_forward(&mha, xr.clone(), Some(pad)).sum();
            let grads_r = loss_r.backward();
            let gr = xr
                .grad(&grads_r)
                .expect("reference input grad")
                .into_data()
                .as_slice::<f32>()
                .unwrap()
                .to_vec();

            assert_finite(&gc, "chunked input grad");
            assert_finite(&gr, "reference input grad");
            let diff = gc
                .iter()
                .zip(gr.iter())
                .map(|(a, c)| (a - c).abs())
                .fold(0.0f32, f32::max);
            assert!(
                diff <= GRAD_TOL,
                "B={b} T={t}: max abs input-grad diff {diff} > {GRAD_TOL}"
            );
        }
    }
}
