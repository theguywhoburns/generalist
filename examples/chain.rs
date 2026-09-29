//! Chained experiments: `cargo run --example chain -- configs/chain.json [gpu]`
//! Runs ordered experiments whose checkpoints feed later ones (`$prev` or
//! explicit paths), re-evaluating every earlier experiment's split after
//! each run (forgetting checks).

use generalist::{
    harness::{ExperimentPlan, load_chain},
    tasks::Instance,
    train::{eval_split, run_stage},
};

fn main() {
    generalist::fail_fast::install();
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .map(|s| s.as_str())
        .expect("usage: cargo run --example chain -- configs/chain.json [gpu]");
    let gpu = args.iter().any(|a| a == "gpu");
    // Every manifest in the plan is resolved, merged and validated up front:
    // a typo in stage 4 must not surface after stage 3 has trained.
    let (plan, runs) =
        load_chain(std::path::Path::new(path)).unwrap_or_else(|e| panic!("load chain:\n{e}"));
    if gpu {
        #[cfg(feature = "cuda")]
        {
            use burn::backend::{Autodiff, Cuda, cuda::CudaDevice};
            println!("backend: CUDA (fused)");
            run_chain::<Autodiff<Cuda>>(plan, runs, &CudaDevice::default());
        }
        #[cfg(not(feature = "cuda"))]
        {
            panic!("gpu requested but the `cuda` feature is off");
        }
    } else {
        println!("backend: NdArray (CPU)");
        run_chain::<burn::backend::Autodiff<burn::backend::NdArray>>(
            plan,
            runs,
            &Default::default(),
        );
    }
}

fn run_chain<B: burn::tensor::backend::AutodiffBackend>(
    plan: ExperimentPlan,
    runs: Vec<(String, generalist::harness::RunConfig)>,
    device: &B::Device,
) {
    // Retained eval pools for forgetting checks: (experiment name, split).
    let mut retained: Vec<(String, Vec<Instance>)> = vec![];
    let mut prev_ckpt: Option<String> = None;

    for ((name, run), exp) in runs.into_iter().zip(&plan.experiments) {
        println!("=== experiment {} ({}) ===", name, exp.manifest);
        let init: Option<String> = match exp.init_from.as_deref() {
            None => None,
            Some(ExperimentPlan::PREV) => prev_ckpt.clone(),
            Some(p) => Some(p.to_string()),
        };
        let pool = generalist::harness::generate(
            &run.experiment,
            &generalist::tasks::TaskRegistry::builtin(),
        );
        // Retained splits must come from the HELD-OUT side of this stage's own
        // split. `run_stage` applies the identical partition internally, so
        // the set computed here is exactly the set it evaluates against.
        let (_, held) =
            generalist::train::partition_holdout(&pool, run.train.eval_holdout, run.train.seed);
        if held.is_empty() {
            eprintln!(
                "warning: {}: eval_holdout is 0, so this stage's retained split is \
                 in-distribution and its forgetting evals do not measure generalization",
                exp.name
            );
        }
        let eval_here = eval_split(&held, 16);
        let outcome = run_stage::<B>(
            &run,
            device,
            init.as_deref().map(std::path::Path::new),
            &retained,
        );
        retained.push((name, eval_here));
        prev_ckpt = Some(outcome.last_ckpt);
    }
    println!("chain done; {} experiments", plan.experiments.len());
}
