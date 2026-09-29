//! LR sweep: `cargo run --example lr_sweep -- configs/<base>.json [gpu]`
//!
//! The Muon learning rate is the one knob that has never been measured, and
//! the previous value was inherited from a run whose optimizer scaled
//! gradients by a data-dependent factor — so it does not transfer. This runs
//! a short run per candidate and reports held-out accuracy for each.
//!
//! It is a sweep, not a tuner: the objective is **held-out accuracy**, never
//! training loss. A lower LR trades training loss for generalization on these
//! tasks, so picking by loss would select exactly the wrong end of the range.
//!
//! ```text
//! cargo run --release --example lr_sweep -- configs/stage0-fixed.json gpu
//! ```
//!
//! Each candidate writes to its own `ckpt_dir` so runs cannot clobber each
//! other, and each is a full `run_stage` — same code path, same holdout
//! split, same everything except the LR under test.

use generalist::{
    harness::{RunConfig, load_run},
    train::run_stage,
};

fn main() {
    generalist::fail_fast::install();
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .map(|s| s.as_str())
        .expect("usage: cargo run --example lr_sweep -- configs/<base>.json [gpu] [lrs...]");
    let gpu = args.iter().any(|a| a == "gpu");
    // Trailing numeric args are the candidate LRs; otherwise a default ladder.
    let candidates: Vec<f64> = args
        .iter()
        .skip(2)
        .filter_map(|a| a.parse::<f64>().ok())
        .collect();
    let candidates = if candidates.is_empty() {
        vec![1e-4, 3e-4, 1e-3, 3e-3, 5e-3, 1e-2]
    } else {
        candidates
    };

    let base =
        load_run(std::path::Path::new(path)).unwrap_or_else(|e| panic!("load base manifest:\n{e}"));

    println!("base: {}", base.summary());
    println!("{} candidates: {candidates:?}\n", candidates.len());

    if gpu {
        #[cfg(feature = "cuda")]
        {
            use burn::backend::{Autodiff, Cuda, cuda::CudaDevice};
            let device = CudaDevice::default();
            println!("backend: CUDA (fused)");
            for lr in &candidates {
                one(&base, *lr, path, |r| {
                    run_stage::<Autodiff<Cuda>>(&r, &device, None, &[])
                });
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            panic!("gpu requested but the `cuda` feature is off");
        }
    } else {
        let device = Default::default();
        println!("backend: NdArray (CPU)");
        for lr in &candidates {
            one(&base, *lr, path, |r| {
                run_stage::<burn::backend::Autodiff<burn::backend::NdArray>>(&r, &device, None, &[])
            });
        }
    }

    println!("\nsweep done; per-candidate logs are in each run's ckpt_dir");
}

/// Run one candidate LR. `ckpt_dir` is suffixed with the LR so candidates
/// cannot overwrite each other's checkpoints or metrics.
fn one<F>(base: &RunConfig, lr: f64, base_path: &str, run: F)
where
    F: FnOnce(RunConfig) -> generalist::train::StageOutcome,
{
    use generalist::optim::LrConfig;
    let mut cfg = base.clone();
    cfg.train.lr_muon = LrConfig::Constant { lr };
    // A sweep that also swept the schedule would be measuring two things.
    cfg.train.lr_adamw = base.train.lr_adamw.clone();
    cfg.train.ckpt_dir = format!("{}-lr{:e}", base.train.ckpt_dir, lr);
    let tag = format!("{}/run.jsonl", cfg.train.ckpt_dir);
    println!("=== lr_muon = {lr:e} -> {tag} ===");
    run(cfg);
    if let Some(acc) = held_out_accuracy(&tag) {
        println!("--> lr_muon {lr:e}: held-out acc {acc:.3}\n");
    } else {
        println!("--> lr_muon {lr:e}: no final-pool record found\n");
    }
    let _ = base_path;
}

/// Pull the held-out accuracy out of a run's metrics log.
///
/// Reads the `final-trained` records — the whole-split final eval — and skips
/// the `-shuffled`, `final-control*` and `shuffle-control` rows, which either
/// use a different stage order or carry no `correct` field at all. The `self`
/// rows are mid-run evals on a smaller split, so they are skipped too.
fn held_out_accuracy(path: &str) -> Option<f64> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut hit = 0usize;
    let mut total = 0usize;
    for line in text.lines() {
        if !line.contains("\"eval\":\"final-trained\"") {
            continue;
        }
        total += 1;
        if line.contains("\"correct\":true") {
            hit += 1;
        }
    }
    (total > 0).then(|| hit as f64 / total as f64)
}
