//! CPU demo for the 1M looped character transformer base.
//!
//! Runs on whichever backend is compiled in, preferring the CPU. The binary
//! used to name `burn::backend::NdArray` unconditionally, which broke any
//! build without the `ndarray` feature. Selecting the backend by `cfg` keeps
//! every feature combination buildable.

#[cfg(all(feature = "cuda", not(feature = "ndarray")))]
use burn::backend::Cuda;
#[cfg(feature = "ndarray")]
use burn::backend::NdArray;

use burn::{
    backend::Autodiff,
    tensor::{Int, Tensor},
};
use generalist::{
    model::{LoopedConfig, LoopedTransformer, StopMode},
    optim::MuonTuning,
};

#[cfg(feature = "ndarray")]
type B = Autodiff<NdArray>;
#[cfg(all(feature = "cuda", not(feature = "ndarray")))]
type B = Autodiff<Cuda>;

fn main() {
    let device = Default::default();
    let config = LoopedConfig::base_1m();
    println!("Option-A param budget: {}", config.param_count());
    // `type_name`, not `Backend::name`: the latter is not in scope for every
    // backend combination this file compiles under.
    println!("backend: {}", std::any::type_name::<B>());

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
