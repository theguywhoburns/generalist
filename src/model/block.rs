use burn::{
    module::Module,
    nn::{RmsNorm, RmsNormConfig},
    tensor::{Tensor, backend::Backend},
};

use super::{attention::MultiHeadAttention, config::LoopedConfig, mlp::SwiGluMlp};

/// The single weight-tied block. Instantiated once, applied up to
/// `max_loops` times. Physical layers: 1. Virtual depth: loop count.
#[derive(Module, Debug)]
pub struct LoopedBlock<B: Backend> {
    pub attn: MultiHeadAttention<B>,
    pub mlp: SwiGluMlp<B>,
    pub norm1: RmsNorm<B>,
    pub norm2: RmsNorm<B>,
    pub scale: f64,
}

impl<B: Backend> LoopedBlock<B> {
    pub fn new(config: &LoopedConfig, device: &B::Device) -> Self {
        Self {
            attn: MultiHeadAttention::new(
                config.d_model,
                config.n_heads,
                config.head_dim,
                config.max_seq_len,
                device,
            ),
            mlp: SwiGluMlp::new(config.d_model, config.ffn_hidden, device),
            norm1: RmsNormConfig::new(config.d_model).init(device),
            norm2: RmsNormConfig::new(config.d_model).init(device),
            scale: config.residual_scale(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_masked(x, None)
    }

    pub fn forward_masked(
        &self,
        x: Tensor<B, 3>,
        key_pad: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let h = self.norm1.forward(x.clone());
        let x = x + self.attn.forward_masked(h, key_pad.clone()).mul_scalar(self.scale);
        let h = self.norm2.forward(x.clone());
        x + self.mlp.forward(h).mul_scalar(self.scale)
    }
}
