//! Is the oracle-vs-induction comparison confounded by prompt length?
//!
//! The oracle header (`MAP a->c b->a c->b`, 20 chars) is prepended to every
//! prompt. Longer prompts push instances into higher length buckets, which
//! changes the B×T row trim and therefore the micro-batch the model actually
//! sees — so "rule told vs rule inferred" is confounded with "longer context"
//! unless the length distributions are checked.
//!
//! Prints the prompt-length distribution for both rungs at the same settings.

use generalist::{
    harness::{Experiment, generate},
    tasks::TaskRegistry,
};

fn main() {
    let registry = TaskRegistry::builtin();
    for task in ["subst-fst", "subst-fst-oracle"] {
        let cfg = Experiment {
            tasks: vec![task.to_string()],
            per_cell: 200,
            seeds: vec![0],
            protocol: generalist::tasks::DemoProtocol {
                k_set: vec![1, 2, 3, 5],
                k0_rate: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let pool = generate(&cfg, &registry);
        let mut lens: Vec<usize> = pool
            .iter()
            .map(|i| i.prompt.len() + i.target.len() + 1)
            .collect();
        lens.sort_unstable();
        let bucket = |lo: usize, hi: usize| lens.iter().filter(|l| **l >= lo && **l < hi).count();
        let mean = lens.iter().sum::<usize>() as f64 / lens.len() as f64;
        println!("{task}");
        println!(
            "  n={}  mean len {:.1}  min {}  max {}",
            lens.len(),
            mean,
            lens[0],
            lens[lens.len() - 1]
        );
        // Same bucket edges the sampler uses.
        for (lo, hi) in [(0, 64), (64, 128), (128, 256), (256, 512), (512, 100000)] {
            let n = bucket(lo, hi);
            println!(
                "    len [{lo:>3}, {hi:>6}): {n:>4}  {:>5.1}%",
                100.0 * n as f64 / lens.len() as f64
            );
        }
    }
}
