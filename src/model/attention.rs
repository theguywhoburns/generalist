use burn::{
    module::Module,
    nn::{Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig},
    tensor::{Tensor, backend::Backend, module::attention, ops::AttentionModuleOptions},
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
    /// (`true` = pad position, blocked from attention). Combined with the
    /// causal mask by the backend.
    pub fn forward_masked(
        &self,
        x: Tensor<B, 3>,
        key_pad: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let [b, t, _] = x.dims();
        let q = self.rope.forward(self.split_heads(self.q.forward(x.clone())));
        let k = self.rope.forward(self.split_heads(self.k.forward(x.clone())));
        let v = self.split_heads(self.v.forward(x));
        let mask_4d = key_pad.map(|kp| {
            kp.unsqueeze_dim::<3>(1)
                .unsqueeze_dim::<4>(2)
                .repeat_dim(1, self.n_heads)
                .repeat_dim(2, t)
        });
        let y = attention(
            q,
            k,
            v,
            mask_4d,
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: true,
            },
        );
        let y = y
            .swap_dims(1, 2)
            .reshape([b, t, self.n_heads * self.head_dim]);
        self.o.forward(y)
    }
}
