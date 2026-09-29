//! CPU demo for the 1M looped character transformer base.

use burn::{
    backend::{Autodiff, NdArray},
    tensor::{Int, Tensor},
};
use generalist::{
    model::{LoopedConfig, LoopedTransformer, StopMode},
    optim::MuonTuning,
};

type B = Autodiff<NdArray>;

fn main() {
    let device = Default::default();
    let config = LoopedConfig::base_1m();
    println!("Option-A param budget: {}", config.param_count());

    let model = LoopedTransformer::<B>::new(&config, &device);
    println!("muon params: {}", model.muon_ids().len());

    let tokens: Tensor<B, 2, Int> = Tensor::zeros([2, 32], &device);
    let out = model.forward(
        tokens,
        &config,
        StopMode::Fixed { loops: 4 },
        &[32, 32],
        None,
    );
    println!(
        "fixed logits: {:?}, steps: {}",
        out.logits.dims(),
        out.steps_used
    );

    let tuning = MuonTuning::new();
    let _muon = tuning.to_muon_config();
    println!(
        "stock muon ready (ns_steps {}, adjust {:?})",
        tuning.ns_steps, tuning.adjust_lr_fn
    );
}
