//! Chained experiments: `cargo run --example chain -- configs/chain.json [gpu]`
//! Runs ordered experiments whose checkpoints feed later ones (`$prev` or
//! explicit paths), re-evaluating every earlier experiment's split after
//! each run (forgetting checks).

use generalist::{
    harness::{ExperimentPlan, RunConfig, generate},
    tasks::{Instance, TaskRegistry},
    train::{eval_split, run_stage},
};

fn main() {
    generalist::fail_fast::install();
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).expect(
        "usage: cargo run --example chain -- configs/chain.json [gpu]",
    );
    let gpu = args.iter().any(|a| a == "gpu");
    let plan = ExperimentPlan::load_json(std::path::Path::new(path)).expect("load plan");
    if gpu {
        #[cfg(feature = "cuda")]
        {
            use burn::backend::{Autodiff, Cuda, cuda::CudaDevice};
            println!("backend: CUDA (fused)");
            run_chain::<Autodiff<Cuda>>(plan, &CudaDevice::default());
        }
        #[cfg(not(feature = "cuda"))]
        {
            panic!("gpu requested but the `cuda` feature is off");
        }
    } else {
        println!("backend: NdArray (CPU)");
        run_chain::<burn::backend::Autodiff<burn::backend::NdArray>>(
            plan,
            &Default::default(),
        );
    }
}

fn run_chain<B: burn::tensor::backend::AutodiffBackend>(plan: ExperimentPlan, device: &B::Device) {
    let registry = TaskRegistry::builtin();
    // Retained eval pools for forgetting checks: (experiment name, split).
    let mut retained: Vec<(String, Vec<Instance>)> = vec![];
    let mut prev_ckpt: Option<String> = None;

    for exp in &plan.experiments {
        println!("=== experiment {} ({}) ===", exp.name, exp.manifest);
        let run = RunConfig::load_json(std::path::Path::new(&exp.manifest))
            .unwrap_or_else(|e| panic!("load {}: {e}", exp.manifest));
        let init: Option<String> = match exp.init_from.as_deref() {
            None => None,
            Some("$prev") => prev_ckpt.clone(),
            Some(p) => Some(p.to_string()),
        };
        let pool = generate(&run.experiment, &registry);
        let eval_here = eval_split(&pool, 16);
        let outcome = run_stage::<B>(
            &run,
            device,
            init.as_deref().map(std::path::Path::new),
            &retained,
        );
        retained.push((exp.name.clone(), eval_here));
        prev_ckpt = Some(outcome.last_ckpt);
    }
    println!("chain done; {} experiments", plan.experiments.len());
}
