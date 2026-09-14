use burn::{
    module::{Initializer, Module, Param, ParamId},
    nn::{Linear, LinearConfig},
    tensor::{Tensor, activation::sigmoid, backend::Backend},
};

/// Scalar per-token halting head. Zero weights + `bias_init` bias gives a
/// uniform starting halt probability of `sigmoid(bias_init)` per step:
/// with the `-3` deep start that is ~4.7%, so training begins near max depth
/// and the ponder pressure shortens it (avoids the shallow-halt trap).
#[derive(Module, Debug)]
pub struct HaltingHead<B: Backend> {
    pub head: Linear<B>,
}

impl<B: Backend> HaltingHead<B> {
    pub fn new(d_model: usize, bias_init: f64, device: &B::Device) -> Self {
        let mut head = LinearConfig::new(d_model, 1)
            .with_bias(true)
            .with_initializer(Initializer::Zeros)
            .init(device);
        // NOTE: `Param::initialized` inherits `require_grad` from the value,
        // so the constant tensor must be marked (a plain constant freezes).
        head.bias = Some(Param::initialized(
            ParamId::new(),
            Tensor::ones([1], device)
                .mul_scalar(bias_init)
                .require_grad(),
        ));
        Self { head }
    }

    /// Halt probability per token. Input `[B, T, D]` -> `[B, T]`.
    pub fn probs(&self, h: Tensor<B, 3>) -> Tensor<B, 2> {
        sigmoid(self.head.forward(h).squeeze_dim::<2>(2))
    }
}
