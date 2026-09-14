//! NCU micro-benchmark: one collated batch (B=8, bucketed T) forward_act +
//! backward + one Muon NS step on CUDA, then exit. Mirrors
//! `configs/profile.json` (base_1m model, parity pool, batch 8).
//! Step 0 = warmup (autotune/fusion caches), step 1 = measured steady state.

use burn::backend::{Autodiff, Cuda, cuda::CudaDevice};
use generalist::{
    harness::{Experiment, generate},
    model::LoopedConfig,
    optim::NewtonMuonConfig,
    tasks::{HarnessRng, TaskRegistry, Track},
    train::{TrainConfig, Trainer},
};

type B = Autodiff<Cuda>;

fn main() {
    let device = CudaDevice::default();
    // Same model as configs/profile.json.
    let model_cfg = LoopedConfig::base_1m();
    println!("params: {}", model_cfg.param_count());
    // Same optim as configs/profile.json.
    let nm_cfg = NewtonMuonConfig::new()
        .with_precond_ewma(0.95)
        .with_refresh_every(32)
        .with_ridge_mult(0.2)
        .with_eps(1e-8)
        .with_init_diag(1e-3)
        .with_ns_steps(5);
    let train_cfg = TrainConfig::new()
        .with_lr_muon(2e-3)
        .with_lr_adamw(3e-4)
        .with_batch_size(8)
        .with_seed(0);

    // Tiny inline pool: parity only, matches profile.json protocol defaults.
    let exp = Experiment {
        tasks: vec!["parity".to_string()],
        tracks: vec![Track::A, Track::B],
        per_cell: 32,
        seeds: vec![0],
        protocol: Default::default(),
    };
    let registry = TaskRegistry::builtin();
    let mut pool = generate(&exp, &registry);
    println!("pool: {} instances", pool.len());
    pool.sort_by_key(|i| i.prompt.len() + i.target.len());

    let mut rng = HarnessRng::new(0x9E37_79B9_7F4A_7C15);
    let mut trainer = Trainer::<B>::new(&model_cfg, &nm_cfg, &train_cfg, &device, None);

    // Warmup: autotune + fusion caches outside the profile window.
    // Reuse the SAME banded window for warmup + measured so the T bucket
    // (and fused-kernel shapes) are identical: steady-state profiling.
    let batch = Trainer::<B>::sample_banded_batch(&mut rng, &pool, 8);
    let tmax: usize = batch.iter().map(|i| i.prompt.len() + i.target.len() + 1).max().unwrap_or(0);
    println!("warmup batch=8 t_raw_max={tmax}");
    let info = trainer.train_step(&batch);
    println!("warmup done loss {:.4} loops {} halt {:.2}", info.loss, info.steps_used, info.mean_halt);

    // Measured steady-state steps: same instances, same bucket.
    // Two of them back-to-back: identical optimizer-gap signatures in both
    // prove per-step repeated host overhead (vs one-time compile, which
    // would appear only before the first measured step).
    for m in 1..=2 {
        println!("NCU_MEASURE_START{m} batch=8 t_raw_max={tmax}");
        let info2 = trainer.train_step(&batch);
        println!(
            "NCU_MEASURE_END{m} loss {:.4} ponder {:.2} loops {} halt {:.2}",
            info2.loss, info2.ponder, info2.steps_used, info2.mean_halt
        );
    }
}
