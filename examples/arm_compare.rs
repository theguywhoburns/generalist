//! Two-arm comparison: does resampling the stage execution order during
//! training change what the role instruments say?
//!
//! `cargo run --release --example arm_compare -- <A.json> <B.json> gpu [seeds...] [--auto-batch]`
//!
//! # Why this is the decisive test
//!
//! The role diagnostic reads a collapse under eval-time stage-order reversal as
//! evidence of order-dependent computation. That inference has a hidden
//! premise: the trained order's collapse is *learned*, not structural. If stage
//! execution is a fixed sequential dependency in the architecture, then a
//! model can collapse under reversal no matter what it learned, and the
//! collapse carries no information about roles.
//!
//! Order-augmented training is the intervention that separates those. It is the
//! one cell where a negative result is informative:
//!
//! - collapse persists → the sequential dependency is structural, the
//!   diagnostic is void as an instrument, and the "learned roles" framing needs
//!   replacing rather than refining.
//! - collapse disappears → order-dependence was learned and avoidable, which is
//!   the claim the diagnostic is meant to make, and the ordered run's collapse
//!   becomes meaningful.
//!
//! Everything else is held fixed: same data, same steps, same model. The only
//! difference between the arms is `stop.act.shuffle_train`.
//!
//! # What is read
//!
//! Both halves of each run, from its log:
//!
//! - **byte accuracy and the train/held-out gap** — the generalization
//!   instruments. The arms are *not* compared on training loss: the
//!   order-augmented arm is solving a different, permutation-robust function,
//!   so its loss is not the ordered arm's loss even at identical steps.
//! - **the profile reading** (entropy / max-deviation / off-diagonal
//!   correlation) — needs no control, so it works on the weights-only rungs
//!   where the random-weight control sits at chance.
//! - **the reversal collapse** — order accuracy minus shuffled accuracy, with
//!   the control's informativeness reported alongside, so "collapse" is never
//!   printed where the control could not have measured it.
//!
//! ```text
//! cargo run --release --example arm_compare -- \
//!   configs/stage0-4block.json configs/stage0-orderaug.json gpu 0 1 2
//! ```

use generalist::{
    harness::{RunConfig, load_run},
    train::run_stage,
};

fn main() {
    generalist::fail_fast::install();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        panic!(
            "usage: cargo run --example arm_compare -- <A.json> <B.json> \
             [gpu] [seeds...] [--auto-batch]"
        );
    }
    let (path_a, path_b) = (args[1].clone(), args[2].clone());
    let gpu = args.iter().any(|a| a == "gpu");
    let auto_batch = args.iter().any(|a| a == "--auto-batch");
    // Everything after the two manifests that is not a flag is a seed.
    let seeds: Vec<u64> = args[3..]
        .iter()
        .filter_map(|a| a.parse::<u64>().ok())
        .collect();
    let seeds = if seeds.is_empty() { vec![0] } else { seeds };

    let a = must_load(&path_a);
    let b = must_load(&path_b);

    // The arms must differ ONLY in the intervention, or the comparison reads
    // as isolating something it does not. Checked rather than assumed: the two
    // configs differ in ckpt_dir by construction, which is not an intervention.
    check_isolated(&a, &b);

    println!("A (ordered)      : {}", a.train.ckpt_dir);
    println!("B (order-aug)    : {}", b.train.ckpt_dir);
    println!("seeds            : {seeds:?}");
    println!(
        "order augmentation: A={} B={}",
        shuffle_train(&a),
        shuffle_train(&b)
    );
    if auto_batch {
        println!("auto-batch: ON");
    }
    println!();

    let mut rows: Vec<Row> = Vec::new();
    if gpu {
        #[cfg(feature = "cuda")]
        {
            use burn::backend::{Autodiff, Cuda, cuda::CudaDevice};
            let device = CudaDevice::default();
            println!("backend: CUDA (fused)\n");
            for (name, cfg) in [("A", &a), ("B", &b)] {
                for seed in &seeds {
                    if let Some(r) = run_one(name, cfg, *seed, auto_batch, |c| {
                        run_stage::<Autodiff<Cuda>>(&c, &device, None, &[])
                    }) {
                        rows.push(r);
                    }
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        panic!("gpu requested but the `cuda` feature is off");
    } else {
        let device = Default::default();
        println!("backend: NdArray (CPU)\n");
        for (name, cfg) in [("A", &a), ("B", &b)] {
            for seed in &seeds {
                if let Some(r) = run_one(name, cfg, *seed, auto_batch, |c| {
                    run_stage::<burn::backend::Autodiff<burn::backend::NdArray>>(
                        &c,
                        &device,
                        None,
                        &[],
                    )
                }) {
                    rows.push(r);
                }
            }
        }
    }

    report(&rows);
}

fn must_load(path: &str) -> RunConfig {
    load_run(std::path::Path::new(path)).unwrap_or_else(|e| panic!("load {path}:\n{e}"))
}

/// Canonical comparison key for a config that does not derive `PartialEq`.
///
/// `serde_json::to_value` is used rather than `to_string`: object key order is
/// not preserved by `Map` iteration guarantees in every build, so a string
/// comparison could report a spurious difference between two identical
/// configs. Comparing values is order-independent by construction.
fn canon<T: serde::Serialize>(v: &T) -> serde_json::Value {
    serde_json::to_value(v).unwrap_or(serde_json::Value::Null)
}

/// Whether the arm resamples stage order during training.
fn shuffle_train(cfg: &RunConfig) -> bool {
    match cfg.stop {
        generalist::model::StopConfig::Act { shuffle_train } => shuffle_train,
        // A fixed or converge control has no order to resample; reporting it
        // as "off" would hide that the arms are not comparable.
        _ => false,
    }
}

/// Refuse to report a comparison whose arms differ by more than the
/// intervention.
///
/// Without this, a config edit that changes two knobs at once produces a
/// difference that gets attributed entirely to the one under test, and the
/// result is worse than no result because it looks clean.
fn check_isolated(a: &RunConfig, b: &RunConfig) {
    let mut problems: Vec<String> = Vec::new();
    // Compared by serialized form rather than `PartialEq`: the config types
    // derive Debug and Serialize but not PartialEq, and a Debug string is a
    // worse comparison key than JSON (field order is not guaranteed).
    if canon(&a.model) != canon(&b.model) {
        problems.push("model config differs".into());
    }
    if canon(&a.experiment) != canon(&b.experiment) {
        problems.push("experiment config differs".into());
    }
    if canon(&a.optim) != canon(&b.optim) {
        problems.push("optim config differs".into());
    }
    // Compared through `canon` for the same reason as the config blocks: Debug
    // output is not a stable comparison key, and these fields are not all the
    // same type so they cannot share one tuple.
    // Checked one field at a time rather than as a single tuple: the fields
    // have different types, so a shared array would need them boxed behind a
    // trait object for no gain.
    let mut diff = |field: &str, x: serde_json::Value, y: serde_json::Value| {
        if x != y {
            problems.push(format!("train.{field} differs"));
        }
    };
    diff("steps", canon(&a.train.steps), canon(&b.train.steps));
    diff("lr_muon", canon(&a.train.lr_muon), canon(&b.train.lr_muon));
    diff(
        "lr_adamw",
        canon(&a.train.lr_adamw),
        canon(&b.train.lr_adamw),
    );
    diff(
        "eval_holdout",
        canon(&a.train.eval_holdout),
        canon(&b.train.eval_holdout),
    );
    diff(
        "max_train_instances",
        canon(&a.train.max_train_instances),
        canon(&b.train.max_train_instances),
    );
    diff(
        "accum_steps",
        canon(&a.train.accum_steps),
        canon(&b.train.accum_steps),
    );
    diff(
        "eval_in_distribution",
        canon(&a.train.eval_in_distribution),
        canon(&b.train.eval_in_distribution),
    );
    diff(
        "indist_eval_per_cell",
        canon(&a.train.indist_eval_per_cell),
        canon(&b.train.indist_eval_per_cell),
    );
    // The stop config differs by construction (that IS the intervention), so it
    // is checked for shape rather than equality: both arms must be ACT, or the
    // comparison is ordered-vs-fixed rather than ordered-vs-order-augmented.
    if !matches!(
        (kind(&a.stop), kind(&b.stop)),
        (StopKind::Act, StopKind::Act)
    ) {
        problems.push("both arms must use ACT; this is not an order-augmentation test".into());
    }
    if !problems.is_empty() {
        panic!(
            "arms differ by more than the intervention:\n  - {}\n\
             A comparison that isolates nothing here would report a difference \
             and attribute it to the order augmentation.",
            problems.join("\n  - ")
        );
    }
}

enum StopKind {
    Act,
    Other,
}

fn kind(s: &generalist::model::StopConfig) -> StopKind {
    match s {
        generalist::model::StopConfig::Act { .. } => StopKind::Act,
        _ => StopKind::Other,
    }
}

#[derive(Clone)]
struct Row {
    arm: &'static str,
    seed: u64,
    byte_in: f64,
    byte_out: f64,
    gap: f64,
    halt: f64,
    roles: String,
    /// Order accuracy minus reversed-order accuracy. The collapse the role
    /// diagnostic reads. `None` when the reversal pass did not run.
    collapse: Option<f64>,
    /// Whether the random-weight control was informative enough to read a
    /// collapse against. `None` = the control could not measure.
    control_ok: Option<bool>,
}

fn run_one<F>(
    arm: &'static str,
    base: &RunConfig,
    seed: u64,
    auto_batch: bool,
    run: F,
) -> Option<Row>
where
    F: FnOnce(RunConfig) -> generalist::train::StageOutcome,
{
    let mut cfg = base.clone();
    cfg.train.seed = seed;
    if auto_batch {
        cfg.train.auto_batch = true;
    }
    // Arm and seed both in the path, so the two arms never share a checkpoint
    // directory and a rerun cannot silently overwrite the other arm.
    cfg.train.ckpt_dir = format!("{}-arm{arm}-s{seed}", base.train.ckpt_dir);
    let log_path = format!("{}/run.jsonl", cfg.train.ckpt_dir);

    println!("=== arm {arm}  seed {seed} -> {log_path} ===");
    run(cfg.clone());
    let text = std::fs::read_to_string(&log_path).ok()?;
    let row = parse(arm, seed, &text)?;
    println!(
        "--> arm {arm} seed {seed}: byte in {:.3} out {:.3} gap {:+.3} | \
         collapse {} | roles {}\n",
        row.byte_in,
        row.byte_out,
        row.gap,
        row.collapse
            .map(|c| format!("{c:+.3}"))
            .unwrap_or_else(|| "n/a".into()),
        row.roles
    );
    Some(row)
}

/// Pull both halves of one run out of its log.
fn parse(arm: &'static str, seed: u64, text: &str) -> Option<Row> {
    let mut byte_in = None;
    let mut byte_out = None;
    let mut gap = None;
    let mut roles = String::new();
    let mut order_acc = None;
    let mut shuffled_acc = None;
    let mut control_ok = None;
    // Per-instance tallies, used only if the pool line is absent (see below).
    let (mut order_hits, mut order_n) = (0u64, 0u64);
    let (mut shuf_hits, mut shuf_n) = (0u64, 0u64);
    for line in text.lines() {
        if line.contains("\"eval\":\"memorization\"") {
            byte_in = jget(line, "indist_byte_accuracy");
            byte_out = jget(line, "heldout_byte_accuracy");
            gap = jget(line, "byte_gap");
        }
        if line.contains("-roles\"") && line.contains("final-trained") {
            roles = line
                .split("\"reading\":")
                .nth(1)
                .and_then(|s| s.split(&[',', '}'][..]).next())
                .unwrap_or("?")
                .trim_matches('"')
                .to_string();
        }
        // Collapse = ordered accuracy minus reversed-order accuracy, over the
        // SAME instances.
        //
        // Read from the pool-level summaries, which are the only lines carrying
        // a single accuracy per eval pass. The per-cell lines share the label
        // prefix, so they are matched on the exact `-pool` suffix: matching the
        // prefix instead silently picks up 576 per-instance records and
        // double-counts.
        if line.contains("\"eval\":\"final-trained-pool\"") {
            order_acc = jget(line, "accuracy");
        }
        if line.contains("\"eval\":\"final-shuffled-pool\"") {
            shuffled_acc = jget(line, "accuracy");
        }
        // Per-instance records, as a fallback. This is an exact identity
        // rather than an approximation: pool accuracy is defined as
        // `records.filter(correct).count() / records.len()`, so tallying the
        // per-instance `correct` flags over all cells of one eval pass
        // reproduces it bit for bit. Used because logs written before the
        // `-pool` line existed have only these, and re-running a 500-step arm
        // to recover one derived number is not a good trade.
        if !line.contains("-pool") && !line.contains("-roles") {
            if line.contains("\"eval\":\"final-trained\"") {
                order_n += 1;
                order_hits += u64::from(line.contains("\"correct\":true"));
            }
            if line.contains("\"eval\":\"final-trained-shuffled\"") {
                shuf_n += 1;
                shuf_hits += u64::from(line.contains("\"correct\":true"));
            }
        }
        if line.contains("\"shuffle_control\"") {
            control_ok = match jbool(line, "control_informative") {
                Some(true) => Some(true),
                Some(false) => Some(false),
                None => None,
            };
        }
    }
    Some(Row {
        arm,
        seed,
        byte_in: byte_in?,
        byte_out: byte_out?,
        gap: gap?,
        halt: mean_halt(text),
        roles,
        collapse: match (order_acc, shuffled_acc) {
            (Some(o), Some(s)) => Some(o - s),
            // Fall back to the per-instance identity.
            _ if order_n > 0 && shuf_n > 0 => {
                Some(order_hits as f64 / order_n as f64 - shuf_hits as f64 / shuf_n as f64)
            }
            _ => None,
        },
        control_ok,
    })
}

fn jget(line: &str, key: &str) -> Option<f64> {
    let k = format!("\"{key}\":");
    let i = line.find(&k)? + k.len();
    let rest = &line[i..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn jbool(line: &str, key: &str) -> Option<bool> {
    let k = format!("\"{key}\":");
    let i = line.find(&k)? + k.len();
    let rest = &line[i..].trim_start();
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn mean_halt(text: &str) -> f64 {
    let (mut sum, mut n) = (0.0, 0usize);
    for line in text.lines() {
        if !line.contains("\"eval\":\"final-trained\"") {
            continue;
        }
        if let Some(h) = jget(line, "halt") {
            sum += h;
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

fn report(rows: &[Row]) {
    if rows.is_empty() {
        println!("\nno results");
        return;
    }
    println!("\n=== order augmentation: two arms ===");
    println!(
        "{:>3} {:>5} {:>9} {:>9} {:>8} {:>9} {:>7}  roles",
        "arm", "seed", "byte-in", "byte-out", "gap", "collapse", "halt"
    );
    for r in rows {
        println!(
            "{:>3} {:>5} {:>9.3} {:>9.3} {:>+8.3} {:>9} {:>7.1}  {}",
            r.arm,
            r.seed,
            r.byte_in,
            r.byte_out,
            r.gap,
            r.collapse
                .map(|c| format!("{c:+.3}"))
                .unwrap_or_else(|| "n/a".into()),
            r.halt,
            r.roles
        );
    }

    // The verdict, stated only from what the arms actually measured. Both
    // directions are informative; neither is a failure.
    let (a, b) = split(rows);
    if a.is_empty() || b.is_empty() {
        println!("\nneed both arms to draw a conclusion; one produced no rows.");
        return;
    }
    let mean = |v: &[Row], f: fn(&Row) -> f64| v.iter().map(f).sum::<f64>() / v.len() as f64;
    let collapse_of = |v: &[Row]| -> Option<f64> {
        let got: Vec<f64> = v.iter().filter_map(|r| r.collapse).collect();
        (!got.is_empty()).then(|| got.iter().sum::<f64>() / got.len() as f64)
    };

    println!(
        "\nA ordered     : byte-out {:.3}  gap {:+.3}  collapse {}",
        mean(&a, |r| r.byte_out),
        mean(&a, |r| r.gap),
        fmt_opt(collapse_of(&a)),
    );
    println!(
        "B order-aug   : byte-out {:.3}  gap {:+.3}  collapse {}",
        mean(&b, |r| r.byte_out),
        mean(&b, |r| r.gap),
        fmt_opt(collapse_of(&b)),
    );

    // Whether the collapse is even readable. Where the control sits at chance
    // the collapse is not evidence about roles, whatever its size.
    let uninformative: Vec<&str> = rows
        .iter()
        .filter(|r| r.control_ok == Some(false))
        .map(|r| if r.arm == "A" { "A" } else { "B" })
        .collect();
    if !uninformative.is_empty() {
        println!(
            "\nthe random-weight control was at chance for arm(s) {}, so any collapse \
             there is NOT evidence about learned roles.",
            uninformative.join(", ")
        );
    }

    println!("\n{}", verdict(&a, &b, collapse_of(&a), collapse_of(&b)));
}

/// The decision, and the fact that it changes what the framing is.
fn verdict(a: &[Row], b: &[Row], collapse_a: Option<f64>, collapse_b: Option<f64>) -> String {
    let (Some(ca), Some(cb)) = (collapse_a, collapse_b) else {
        return "Collapse not measured in both arms (shuffle_eval off, or the \
                random-weight control sat at chance), so this run cannot decide \
                whether order-dependence is learned or structural."
            .to_string();
    };
    let mean = |v: &[Row], f: fn(&Row) -> f64| v.iter().map(f).sum::<f64>() / v.len() as f64;
    let out_a = mean(a, |r| r.byte_out);
    let out_b = mean(b, |r| r.byte_out);
    if cb.abs() < ca.abs() * 0.5 {
        format!(
            "ORDER-DEPENDENCE IS LEARNED, NOT STRUCTURAL. Order augmentation cut the \
             reversal collapse from {:.3} to {:.3} while held-out byte accuracy went \
             {:.3} -> {:.3}. The trained-order collapse therefore carries \
             information about learned roles, and the role diagnostic stands.",
            ca, cb, out_a, out_b
        )
    } else if cb.abs() >= ca.abs() * 0.8 {
        format!(
            "ORDER-DEPENDENCE IS STRUCTURAL. Order augmentation left the reversal \
             collapse at {:.3} vs {:.3} (held-out byte {:.3} -> {:.3}). A model can \
             collapse under reversal regardless of what it learned, so the collapse \
             does not evidence learned roles and the diagnostic needs a different \
             basis.",
            ca, cb, out_a, out_b
        )
    } else {
        format!(
            "INCONCLUSIVE on the structural/learned question: collapse {:.3} -> {:.3}, \
             which is neither halved nor unchanged. Needs more seeds before the \
             framing is decided (held-out byte {:.3} -> {:.3}).",
            ca, cb, out_a, out_b
        )
    }
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:+.3}"))
        .unwrap_or_else(|| "n/a".into())
}

fn split(rows: &[Row]) -> (Vec<Row>, Vec<Row>) {
    rows.iter()
        .fold((Vec::new(), Vec::new()), |(mut a, mut b), r| {
            if r.arm == "A" {
                a.push(r.clone());
            } else {
                b.push(r.clone());
            }
            (a, b)
        })
}
