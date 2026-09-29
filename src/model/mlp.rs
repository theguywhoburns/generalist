use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Tensor, activation::silu, backend::Backend},
};

/// SwiGLU MLP: `down(silu(gate(x)) * up(x))`. Separate gate/up/down linears
/// rather than one fused gate_up matrix: keeps each a plain `Linear`, so
/// parameter classification, Muon routing, and per-matrix shape-based LR
/// adjustment all apply uniformly with no packed-matrix special cases.
#[derive(Module, Debug)]
pub struct SwiGluMlp<B: Backend> {
    pub gate: Linear<B>,
    pub up: Linear<B>,
    pub down: Linear<B>,
}

impl<B: Backend> SwiGluMlp<B> {
    pub fn new(d_model: usize, hidden: usize, device: &B::Device) -> Self {
        let proj = |d_in, d_out| LinearConfig::new(d_in, d_out).with_bias(false).init(device);
        Self {
            gate: proj(d_model, hidden),
            up: proj(d_model, hidden),
            down: proj(hidden, d_model),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = silu(self.gate.forward(x.clone())) * self.up.forward(x);
        self.down.forward(h)
    }

    /// The SwiGLU hidden state, i.e. the down-projection input.
    pub fn hidden(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        silu(self.gate.forward(x.clone())) * self.up.forward(x)
    }
}
