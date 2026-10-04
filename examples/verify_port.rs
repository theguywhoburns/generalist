//! Verify a ported manifest matches the original scratch manifest field-for-field.
//!
//! The research manifests were reconstructed from throwaway files in /tmp after
//! the fact. "It loads" is not enough: a silently dropped `k_set` or `eval_holdout`
//! would reproduce a headline number on the wrong cell. This compares the fully
//! resolved configs.

use generalist::harness::{RunConfig, load_run};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut bad = 0;
    for pair in args.chunks(2) {
        let (ported, original) = (&pair[0], &pair[1]);
        let a = load_run(std::path::Path::new(ported)).expect("ported manifest");
        let b = load_run(std::path::Path::new(original)).expect("original manifest");
        // `ckpt_dir` is intentionally different (repo-local vs /tmp scratch).
        let mut diffs: Vec<String> = Vec::new();
        let mut cmp = |name: &str, x: serde_json::Value, y: serde_json::Value| {
            if x != y {
                diffs.push(format!("{name}: ported={x} original={y}"));
            }
        };
        cmp("model", canon(&a.model), canon(&b.model));
        cmp("optim", canon(&a.optim), canon(&b.optim));
        cmp("stop", canon(&a.stop), canon(&b.stop));
        // `experiment` is compared with one documented exception rather than
        // field-by-field, because `k0_rate` is provably inert when `k_set` has
        // no positive entry: `sample_k` returns 0 either because the k0_rate
        // roll fired or because there is nothing positive to draw, so
        // k_set [0] means k == 0 always. The ported manifests write
        // `k0_rate: 1.0` because it is the honest spelling of "always
        // zero-shot", which reads as a typo next to the original's inherited
        // 0.15.
        if k_is_always_zero(&a) && k_is_always_zero(&b) {
            let mut ea = a.experiment.clone();
            let mut eb = b.experiment.clone();
            ea.protocol.k0_rate = 0.0;
            eb.protocol.k0_rate = 0.0;
            cmp(
                "experiment (k0_rate ignored: k_set has no positive entry)",
                canon(&ea),
                canon(&eb),
            );
        } else {
            cmp("experiment", canon(&a.experiment), canon(&b.experiment));
        }
        // One call per field: they have different types, so a shared array
        // would need boxing behind a trait object for nothing.
        cmp("steps", canon(&a.train.steps), canon(&b.train.steps));
        cmp(
            "batch_size",
            canon(&a.train.batch_size),
            canon(&b.train.batch_size),
        );
        cmp(
            "auto_batch",
            canon(&a.train.auto_batch),
            canon(&b.train.auto_batch),
        );
        cmp(
            "auto_batch_max",
            canon(&a.train.auto_batch_max),
            canon(&b.train.auto_batch_max),
        );
        cmp(
            "eval_holdout",
            canon(&a.train.eval_holdout),
            canon(&b.train.eval_holdout),
        );
        cmp(
            "eval_max_new",
            canon(&a.train.eval_max_new),
            canon(&b.train.eval_max_new),
        );
        cmp(
            "eval_every",
            canon(&a.train.eval_every),
            canon(&b.train.eval_every),
        );
        cmp(
            "shuffle_eval",
            canon(&a.train.shuffle_eval),
            canon(&b.train.shuffle_eval),
        );
        cmp(
            "eval_in_distribution",
            canon(&a.train.eval_in_distribution),
            canon(&b.train.eval_in_distribution),
        );
        cmp(
            "indist_eval_per_cell",
            canon(&a.train.indist_eval_per_cell),
            canon(&b.train.indist_eval_per_cell),
        );
        cmp(
            "max_train_instances",
            canon(&a.train.max_train_instances),
            canon(&b.train.max_train_instances),
        );
        cmp("lr_muon", canon(&a.train.lr_muon), canon(&b.train.lr_muon));
        cmp(
            "lr_adamw",
            canon(&a.train.lr_adamw),
            canon(&b.train.lr_adamw),
        );
        cmp(
            "dump_samples",
            canon(&a.train.dump_samples),
            canon(&b.train.dump_samples),
        );
        if diffs.is_empty() {
            println!("OK    {ported} == {original}");
        } else {
            bad += 1;
            println!("DIFF  {ported} vs {original}");
            for d in &diffs {
                println!("        {d}");
            }
        }
    }
    if bad > 0 {
        std::process::exit(1);
    }
}

fn canon<T: serde::Serialize>(v: &T) -> serde_json::Value {
    serde_json::to_value(v).expect("serialize")
}

/// True when this config can only ever generate k = 0 instances, which makes
/// `k0_rate` inert. Mirrors `DemoProtocol::sample_k`: k is 0 if the k0_rate
/// roll fires OR if `k_set` holds no positive entry.
fn k_is_always_zero(cfg: &RunConfig) -> bool {
    cfg.experiment.protocol.k_set.iter().all(|k| *k == 0)
}
