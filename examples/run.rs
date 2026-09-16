//! Dispatch dry-run: `cargo run --example run -- configs/<run>.json`
//! Loads the manifest, generates the pool, prints cell counts and one sample.
//! No training, no GPU. The manifest path is required.

use generalist::{
    harness::{RunConfig, generate},
    tasks::TaskRegistry,
};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: cargo run --example run -- configs/<run>.json");
    let run =
        RunConfig::load_json(std::path::Path::new(&path)).expect("load run manifest");
    println!(
        "model params: {} | muon lr: {} | adamw lr: {} | batch: {} | steps: {}",
        run.model.param_count(),
        run.train.lr_muon,
        run.train.lr_adamw,
        run.train.batch_size,
        run.train.steps,
    );
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
}
