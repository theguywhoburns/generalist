//! Dispatch dry-run: `cargo run --example run -- configs/<run>.json`
//! Loads and validates the manifest, generates the pool, prints cell counts
//! and one sample. No training, no GPU. The manifest path is required.

use generalist::{
    harness::{RunConfig, generate, load_run},
    tasks::TaskRegistry,
};

fn main() {
    generalist::fail_fast::install();
    let path = std::env::args()
        .nth(1)
        .expect("usage: cargo run --example run -- configs/<run>.json");
    // `extends` is resolved, merged and validated here; a bad manifest fails
    // with every problem listed, before any pool or model exists.
    let run = load_run(std::path::Path::new(&path))
        .unwrap_or_else(|e| panic!("load run manifest:\n{e}"));
    println!("{} | batch {} | steps {}", run.summary(), run.train.batch_size, run.train.steps);
    println!("muon lr: {:?} | adamw lr: {:?}", run.train.lr_muon, run.train.lr_adamw);
    let registry = TaskRegistry::builtin();
    let pool = generate(&run.experiment, &registry);
    println!("pool instances: {}", pool.len());
    let mut cells = std::collections::BTreeMap::new();
    for inst in &pool {
        *cells
            .entry((inst.info.task, format!("{:?}", inst.info.track), inst.info.k))
            .or_insert(0usize) += 1;
    }
    for ((task, track, k), n) in &cells {
        println!("  {task} track={track} k={k}: {n}");
    }
    if let Some(first) = pool.first() {
        println!("--- sample prompt ---");
        print!("{}", String::from_utf8_lossy(&first.prompt));
        println!("--- sample target ---");
        println!("{}", String::from_utf8_lossy(&first.target));
    }
    // Keep the type in the signature honest: RunConfig is what we loaded.
    let _ = RunConfig::smoke();
}
