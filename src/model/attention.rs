use burn::{
    module::Module,
    nn::{Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig},
    tensor::{Bool, Int, Tensor, backend::Backend},
};

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
    pub n_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
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
    /// correct). Explicit math is portable and auditable; at 1M scale the
    /// transient `[B, H, T, T]` scores are small.
    pub fn forward_masked(
        &self,
        x: Tensor<B, 3>,
        key_pad: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [b, t, _] = x.dims();
        let device = x.device();
        let q = self.rope.forward(self.split_heads(self.q.forward(x.clone())));
        let k = self.rope.forward(self.split_heads(self.k.forward(x.clone())));
        let v = self.split_heads(self.v.forward(x));

        // scores[b,h,i,j] = q[b,h,i] . k[b,h,j] / sqrt(head_dim)
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);

        // keep[b,h,i,j] = causal (j <= i) AND key not pad.
        let qi = Tensor::<B, 1, Int>::arange(0..t as i64, &device)
            .unsqueeze_dim::<2>(1)
            .repeat_dim(1, t);
        let kj = Tensor::<B, 1, Int>::arange(0..t as i64, &device)
            .unsqueeze_dim::<2>(0)
            .repeat_dim(0, t);
        let causal = kj
            .lower_equal(qi)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0)
            .repeat_dim(0, b)
            .repeat_dim(1, self.n_heads);
        let keep = match key_pad {
            None => causal,
            Some(pad) => {
                let keys_kept = pad
                    .bool_not()
                    .unsqueeze_dim::<3>(1)
                    .unsqueeze_dim::<4>(2)
                    .repeat_dim(1, self.n_heads)
                    .repeat_dim(2, t);
                causal.bool_and(keys_kept)
            }
        };
        // Blocked -> -1e30 attends to nothing (exp underflows to exactly 0;
        // finite so a fully-blocked row can't NaN — and pos0 is always kept
        // by the pad_mask guard anyway).
        let bias = keep.float().mul_scalar(-1.0).add_scalar(1.0).mul_scalar(-1e30);
        let probs = burn::tensor::activation::softmax(scores + bias, 3);
        let y = probs.matmul(v);

        let y = y
            .swap_dims(1, 2)
            .reshape([b, t, self.n_heads * self.head_dim]);
        self.o.forward(y)
    }
}
