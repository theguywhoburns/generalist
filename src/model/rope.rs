use burn::{
    module::Module,
    nn::{RmsNorm, RmsNormConfig},
    tensor::{Bool, Tensor, backend::Backend},
};

use super::{attention::MultiHeadAttention, config::LoopedConfig, mlp::SwiGluMlp};

/// Inputs to each matrix group, exposed for Newton-Muon input statistics.
pub struct BlockInputs<B: Backend> {
    pub attn_in: Tensor<B, 3>,
    pub mlp_in: Tensor<B, 3>,
    pub hidden: Tensor<B, 3>,
}

/// One encoder layer, burn-`TransformerEncoderLayer` style: unfused causal
/// MHA with RoPE, RMSNorms, SwiGLU MLP. Deliberately halt-free and
/// optimizer-blind: halting lives in [`LoopedStage`](super::transformer::LoopedStage),
/// parameter routing in the model's `grad_specs`/`muon_ids` (rank-based).
/// A decoder cross-attention variant belongs here once a task produces a
/// memory stream; until then it would be dead code.
#[derive(Module, Debug)]
pub struct RopeTransformer<B: Backend> {
    pub attn: MultiHeadAttention<B>,
    pub mlp: SwiGluMlp<B>,
    pub norm1: RmsNorm<B>,
    pub norm2: RmsNorm<B>,
    pub scale: f64,
}

impl<B: Backend> RopeTransformer<B> {
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
        key_pad: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        self.forward_split(x, key_pad).0
    }

    /// Forward pass that also returns per-group inputs for statistics.
    /// Same math as [`Self::forward_masked`], no duplicated computation.
    pub fn forward_split(
        &self,
        x: Tensor<B, 3>,
        key_pad: Option<Tensor<B, 2, Bool>>,
    ) -> (Tensor<B, 3>, BlockInputs<B>) {
        let attn_in = self.norm1.forward(x.clone());
        let x = x + self
            .attn
            .forward_masked(attn_in.clone(), key_pad)
            .mul_scalar(self.scale);
        let mlp_in = self.norm2.forward(x.clone());
        let hidden = self.mlp.hidden(mlp_in.clone());
        let y = x + self.mlp.down.forward(hidden.clone()).mul_scalar(self.scale);
        (
            y,
            BlockInputs {
                attn_in,
                mlp_in,
                hidden,
            },
        )
    }
}
