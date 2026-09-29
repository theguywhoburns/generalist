//! LR sweep over learning rates and seeds.
//!
//! `cargo run --release --example lr_sweep -- configs/<base>.json [gpu] [lrs...] [--seeds=a,b,c]`
//!
//! The Muon learning rate is the one knob that was never measured — the old
//! value was inherited from a run whose optimizer scaled gradients by a
//! data-dependent factor, so it did not transfer. This measures it.
//!
//! Two axes, because one is not enough:
//!
//! - **LR.** Each candidate is a full `run_stage`: same code path, same
//!   holdout split, same everything except the LR.
//! - **Seed.** A single-seed comparison cannot separate a real difference from
//!   run-to-run variance. On a ~96-instance eval, one flipped instance is
//!   ~0.01 accuracy, which is the same order as the effect being measured.
//!   Without `--seeds`, the summary says so rather than presenting a ranking.
//!
//! The reported objective is **held-out accuracy**, never training loss: a
//! lower LR trades training loss for generalization here, so selecting by loss
//! would pick the wrong end of the range.
//!
//! ```text
//! cargo run --release --example lr_sweep -- configs/stage0-fixed.json gpu 2e-3 5e-3 --seeds=0,1,2
//! ```

use generalist::{
    harness::{RunConfig, load_run},
    optim::LrConfig,
    train::run_stage,
};

fn main() {
    generalist::fail_fast::install();
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).expect(
        "usage: cargo run --example lr_sweep -- configs/<base>.json [gpu] [lrs...] [--seeds=a,b,c]",
    );
    let gpu = args.iter().any(|a| a == "gpu");
    // Bare numeric args are candidate LRs; otherwise a default ladder.
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
    let seeds: Vec<u64> = match args.iter().find_map(|a| a.strip_prefix("--seeds=")) {
        Some(s) => s
            .split(',')
            .filter_map(|p| p.trim().parse::<u64>().ok())
            .collect(),
        None => vec![0],
    };
    assert!(!seeds.is_empty(), "--seeds= yielded no valid seeds");

    let base =
        load_run(std::path::Path::new(path)).unwrap_or_else(|e| panic!("load base manifest:\n{e}"));

    println!("base: {}", base.summary());
    println!("lrs: {candidates:?}  seeds: {seeds:?}\n");

    // Collected per (lr, seed) so the summary reports spread, not a point
    // estimate dressed up as a ranking.
    let mut results: Vec<(f64, u64, f64)> = Vec::new();

    if gpu {
        #[cfg(feature = "cuda")]
        {
            use burn::backend::{Autodiff, Cuda, cuda::CudaDevice};
            let device = CudaDevice::default();
            println!("backend: CUDA (fused)");
            for lr in &candidates {
                for seed in &seeds {
                    if let Some(acc) = one(&base, *lr, *seed, |r| {
                        run_stage::<Autodiff<Cuda>>(&r, &device, None, &[])
                    }) {
                        results.push((*lr, *seed, acc));
                    }
                }
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
            for seed in &seeds {
                if let Some(acc) = one(&base, *lr, *seed, |r| {
                    run_stage::<burn::backend::Autodiff<burn::backend::NdArray>>(
                        &r,
                        &device,
                        None,
                        &[],
                    )
                }) {
                    results.push((*lr, *seed, acc));
                }
            }
        }
    }

    summarize(&results);
    println!("\nsweep done; per-candidate logs are in each run's ckpt_dir");
}

/// Run one (lr, seed) candidate. `ckpt_dir` is suffixed with both so runs
/// cannot overwrite each other's checkpoints or metrics.
fn one<F>(base: &RunConfig, lr: f64, seed: u64, run: F) -> Option<f64>
where
    F: FnOnce(RunConfig) -> generalist::train::StageOutcome,
{
    let mut cfg = base.clone();
    // Constant, not the base's schedule: a sweep that also varied the decay
    // shape would be measuring two things at once.
    cfg.train.lr_muon = LrConfig::Constant { lr };
    cfg.train.seed = seed;
    cfg.train.ckpt_dir = format!("{}-lr{:e}-s{seed}", base.train.ckpt_dir, lr);
    let tag = format!("{}/run.jsonl", cfg.train.ckpt_dir);
    println!("=== lr_muon = {lr:e}  seed = {seed} -> {tag} ===");
    run(cfg);
    let acc = held_out_accuracy(&tag);
    match acc {
        Some(a) => {
            println!("--> lr {lr:e} seed {seed}: held-out acc {a:.3}\n");
            Some(a)
        }
        None => {
            println!("--> lr {lr:e} seed {seed}: no final-trained record found\n");
            None
        }
    }
}

/// Report mean and spread per LR. The spread is the point: without it a
/// difference smaller than the seed-to-seed range reads as a result.
fn summarize(results: &[(f64, u64, f64)]) {
    if results.is_empty() {
        println!("\nno results");
        return;
    }
    println!("\n=== held-out accuracy by lr_muon ===");
    println!(
        "{:>10} {:>3} {:>7} {:>7} {:>7}  {}",
        "lr", "n", "mean", "min", "max", "per-seed"
    );
    let mut lrs: Vec<f64> = results.iter().map(|(lr, _, _)| *lr).collect();
    lrs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    lrs.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    for lr in lrs {
        let vals: Vec<f64> = results
            .iter()
            .filter(|(l, _, _)| (*l - lr).abs() < 1e-12)
            .map(|(_, _, a)| *a)
            .collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let per: Vec<String> = results
            .iter()
            .filter(|(l, _, _)| (*l - lr).abs() < 1e-12)
            .map(|(_, s, a)| format!("s{s}={a:.3}"))
            .collect();
        println!(
            "{lr:>10.1e} {:>3} {mean:>7.3} {min:>7.3} {max:>7.3}  {}",
            vals.len(),
            per.join(" ")
        );
    }
    // With one seed the table looks like a clean ranking. Say so, because that
    // is the trap.
    if results.iter().all(|r| r.1 == results[0].1) {
        let p = results[0].2;
        println!(
            "\nNOTE: ONE seed per LR. Binomial SE at n=96, p={p:.2} is ~{:.3}, and \
             seed-to-seed variance is larger still. Treat the ranking as \
             provisional; rerun with --seeds=0,1,2 before believing any gap.",
            (p * (1.0 - p) / 96.0).sqrt()
        );
    }
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
