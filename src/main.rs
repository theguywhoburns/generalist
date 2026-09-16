//! CPU demo for the 1M looped character transformer base.

use burn::{
    backend::{Autodiff, NdArray},
    tensor::{Int, Tensor},
};
use generalist::{
    model::{LoopedConfig, LoopedTransformer, StopMode},
    optim::NewtonMuonConfig,
};

type B = Autodiff<NdArray>;

fn main() {
    let device = Default::default();
    let config = LoopedConfig::base_1m();
    println!("Option-A param budget: {}", config.param_count());

    let model = LoopedTransformer::<B>::new(&config, &device);
    println!("muon params: {}", model.muon_ids().len());

    let tokens: Tensor<B, 2, Int> = Tensor::zeros([2, 32], &device);
    let out = model.forward(tokens, &config, StopMode::Fixed { loops: 4 }, &[32, 32], None);
    println!(
        "fixed logits: {:?}, steps: {}",
        out.logits.dims(),
        out.steps_used
    );

    let nm_cfg = NewtonMuonConfig::new();
    let _muon = nm_cfg.muon_config();
    println!("newton-muon ready (refresh every {})", nm_cfg.refresh_every);
}
