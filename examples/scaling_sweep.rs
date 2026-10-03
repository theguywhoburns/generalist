//! Scaling sweep: vary one axis of the model or data and report where
//! generalization starts to fail.
//!
//! `cargo run --release --example scaling_sweep -- configs/<base>.json <axis> [gpu] [values...] [--seeds=a,b]`
//!
//! # Axes
//!
//! - `data`   — `experiment.per_cell`. **This is the memorization-onset axis.**
//!   Every point reports held-out accuracy, a training-pool accuracy from the
//!   same run, and the gap between them. The onset is where the gap starts
//!   opening, not where held-out accuracy falls: a falling held-out number
//!   alone is ambiguous between "the task got harder" and "the model
//!   overfit", and only the divergence separates them.
//! - `depth`  — `stop.loops` under `fixed`, with ACT as a reference point.
//!   The compute-depth curve. Paired with the per-stage halt profile so a
//!   fixed-depth win and a degenerate-halting artifact stay distinguishable.
//! - `params` — `model.d_model` with the head geometry kept consistent.
//!
//! # What the objective is
//!
//! The reported number is the **generalization gap** (in-distribution minus
//! held-out), alongside both halves. Selecting a configuration by held-out
//! accuracy alone picks whatever memorizes most, because in-distribution
//! accuracy is a strictly easier target and the two diverge exactly where the
//! question is being asked.
//!
//! ```text
//! cargo run --release --example scaling_sweep -- configs/stage0-fixed.json data gpu 8 16 32 64 128
//! ```

use generalist::{
    harness::{RunConfig, load_run},
    model::StopConfig,
    train::run_stage,
};

fn main() {
    generalist::fail_fast::install();
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).expect(
        "usage: cargo run --example scaling_sweep -- configs/<base>.json <data|depth|params> [gpu] [values...] [--seeds=a,b]",
    );
    let axis = args
        .get(2)
        .map(|s| s.as_str())
        .expect("axis is required: data | depth | params");
    let gpu = args.iter().any(|a| a == "gpu");
    // Off by default: the tuner changes the effective batch (it cannot
    // subdivide below the micro-batch), which is a confound on a sweep whose
    // point is a single controlled comparison. Opt in explicitly.
    let auto_batch = args.iter().any(|a| a == "--auto-batch");

    let values: Vec<f64> = args
        .iter()
        .skip(3)
        .filter_map(|a| a.parse::<f64>().ok())
        .collect();
    let values = if values.is_empty() {
        default_values(axis).to_vec()
    } else {
        values
    };
    let seeds: Vec<u64> = match args.iter().find_map(|a| a.strip_prefix("--seeds=")) {
        Some(s) => s
            .split(',')
            .filter_map(|p| p.trim().parse::<u64>().ok())
            .collect(),
        None => vec![0],
    };

    let base =
        load_run(std::path::Path::new(path)).unwrap_or_else(|e| panic!("load base manifest:\n{e}"));
    println!("base: {}", base.summary());
    println!("axis: {axis}  values: {values:?}  seeds: {seeds:?}");
    if auto_batch {
        println!("auto-batch: ON (the effective batch may exceed the manifest's)");
    }
    println!();

    let mut rows: Vec<SweepRow> = Vec::new();

    if gpu {
        #[cfg(feature = "cuda")]
        {
            use burn::backend::{Autodiff, Cuda, cuda::CudaDevice};
            // Tuned runs resolve their own batch; the `apply` knobs below only
            // set the axis value.
            let device = CudaDevice::default();
            println!("backend: CUDA (fused)");
            for v in &values {
                for seed in &seeds {
                    // Tuned per point: the batch is a function of the card, and a sweep
                    // re-measuring it every run is the only way the rows stay
                    // comparable when the tuner picks different sizes.
                    let mut cfg = apply(&base, axis, *v, *seed);
                    if auto_batch {
                        cfg.train.auto_batch = true;
                    }
                    if let Some(r) = one(&cfg, axis, *v, *seed, |r| {
                        run_stage::<Autodiff<Cuda>>(&r, &device, None, &[])
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
        println!("backend: NdArray (CPU)");
        for v in &values {
            for seed in &seeds {
                let cfg = apply(&base, axis, *v, *seed);
                if let Some(r) = one(&cfg, axis, *v, *seed, |r| {
                    run_stage::<burn::backend::Autodiff<burn::backend::NdArray>>(
                        &r,
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

    report(&rows, axis);
}

fn default_values(axis: &str) -> &'static [f64] {
    match axis {
        "data" => &[8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
        // 16 exceeds the base's max_loops of 8, so it is raised with the value.
        "depth" => &[1.0, 2.0, 4.0, 8.0],
        "params" => &[128.0, 192.0, 256.0, 384.0],
        other => panic!("unknown axis {other}: expected data | depth | params"),
    }
}

/// Build the config for one (axis, value, seed) point.
fn apply(base: &RunConfig, axis: &str, v: f64, seed: u64) -> RunConfig {
    let mut cfg = base.clone();
    cfg.train.seed = seed;
    // Distinguishable per point so a rerun never silently clobbers a result.
    cfg.train.ckpt_dir = format!("{}-sweep-{axis}{:?}-s{seed}", base.train.ckpt_dir, v);
    match axis {
        "data" => {
            // Cap training data, NOT per_cell. `per_cell` would move the eval
            // set along with the training set, so every point would be scored
            // on a different split and the curve would not be comparable
            // across points. The cap is applied after the holdout, so the
            // held-out set is identical at every point on this axis.
            cfg.train.max_train_instances = v as usize;
        }
        "depth" => {
            let loops = v as usize;
            cfg.stop = StopConfig::Fixed { loops };
            // Fixed depth has no learned step count, so the ponder penalty is
            // a constant loss offset. Zero it, as the fixed4 manifest does.
            cfg.model.ponder_weight = 0.0;
            // The depth axis must be reachable: raise the loop ceiling to
            // match, or `max_loops` silently caps the point.
            if loops > cfg.model.max_loops {
                cfg.model.max_loops = loops;
            }
        }
        "params" => {
            // Keep head geometry consistent: hold head_dim fixed and derive
            // the head count, so the change is width, not head layout.
            cfg.model.d_model = v as usize;
            cfg.model.n_heads = (v as usize / 64).max(1);
            cfg.model.head_dim = 64;
        }
        other => panic!("unknown axis {other}"),
    }
    let problems = cfg.validate();
    assert!(
        problems.is_empty(),
        "generated config for {axis}={v} is invalid:\n  - {}",
        problems.join("\n  - ")
    );
    cfg
}

struct SweepRow {
    value: f64,
    seed: u64,
    indist: f64,
    heldout: f64,
    indist_byte: f64,
    heldout_byte: f64,
    byte_gap: f64,
    halt: f64,
    roles: String,
}

fn one<F>(cfg: &RunConfig, axis: &str, v: f64, seed: u64, run: F) -> Option<SweepRow>
where
    F: FnOnce(RunConfig) -> generalist::train::StageOutcome,
{
    let log_path = format!("{}/run.jsonl", cfg.train.ckpt_dir);
    println!("=== {axis} = {v:.0}  seed = {seed} -> {log_path} ===");
    run(cfg.clone());
    let text = std::fs::read_to_string(&log_path).ok()?;
    let row = parse(&text, v, seed)?;
    println!(
        "--> {axis}={v:.0} seed={seed}: byte in {:.3} out {:.3} gap {:+.3} | exact in {:.3} out {:.3} | halt {:.1} roles {}\n",
        row.indist_byte,
        row.heldout_byte,
        row.byte_gap,
        row.indist,
        row.heldout,
        row.halt,
        row.roles
    );
    Some(row)
}

/// Read the numbers the sweep is about out of one run's log.
///
/// Byte accuracy is read alongside exact-match because exact-match is floored
/// at 0.000 across entire depth points while training CE sits near 0.45 — the
/// model is getting most bytes right and still failing the conjunction. A
/// sweep that reports only the strict metric concludes "depth does nothing"
/// when the truth is "the strict metric cannot see this".
fn parse(text: &str, value: f64, seed: u64) -> Option<SweepRow> {
    let mut indist = None;
    let mut heldout = None;
    let mut indist_byte = None;
    let mut heldout_byte = None;
    let mut byte_gap = None;
    let mut roles = String::new();
    for line in text.lines() {
        if line.contains("\"eval\":\"memorization\"") {
            indist = jget(line, "indist_accuracy");
            heldout = jget(line, "heldout_accuracy");
            indist_byte = jget(line, "indist_byte_accuracy");
            heldout_byte = jget(line, "heldout_byte_accuracy");
            byte_gap = jget(line, "byte_gap");
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
    }
    Some(SweepRow {
        value,
        seed,
        indist: indist?,
        heldout: heldout?,
        indist_byte: indist_byte?,
        heldout_byte: heldout_byte?,
        byte_gap: byte_gap?,
        halt: mean_halt(text),
        roles,
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

fn mean_halt(text: &str) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
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

fn report(rows: &[SweepRow], axis: &str) {
    if rows.is_empty() {
        println!("\nno results");
        return;
    }
    println!("\n=== {axis} sweep: generalization vs memorization ===");
    println!(
        "byte accuracy is the load-bearing column; exact-match is floored wherever byte accuracy is high but < 1."
    );
    println!(
        "{:>10} {:>3} {:>9} {:>9} {:>8} | {:>8} {:>8} {:>7}  roles",
        "value", "n", "byte-in", "byte-out", "byte-gap", "ex-in", "ex-out", "halt"
    );
    let mut vals: Vec<f64> = rows.iter().map(|r| r.value).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    for v in vals {
        let g: Vec<&SweepRow> = rows.iter().filter(|r| (r.value - v).abs() < 1e-9).collect();
        let avg = |f: fn(&SweepRow) -> f64| g.iter().map(|r| f(r)).sum::<f64>() / g.len() as f64;
        let roles = g[0].roles.clone();
        println!(
            "{v:>10.0} {:>3} {:>9.3} {:>9.3} {:>+8.3} | {:>8.3} {:>8.3} {:>7.1}  {roles}",
            g.len(),
            avg(|r| r.indist_byte),
            avg(|r| r.heldout_byte),
            avg(|r| r.byte_gap),
            avg(|r| r.indist),
            avg(|r| r.heldout),
            avg(|r| r.halt),
        );
    }
    // Per-seed detail matters more than the means here: the gap is a
    // DIFFERENCE of two accuracies, so its variance is larger than either
    // half's, and a mean gap can hide a bimodal result.
    let distinct = rows
        .iter()
        .map(|r| r.value.to_bits())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    if rows.len() > distinct {
        println!("\nper-seed:");
        for r in rows {
            println!(
                "  {axis}={:.0} seed {}: byte in {:.3} out {:.3} gap {:+.3} | ex in {:.3} out {:.3} | halt {:.1}",
                r.value,
                r.seed,
                r.indist_byte,
                r.heldout_byte,
                r.byte_gap,
                r.indist,
                r.heldout,
                r.halt
            );
        }
    }
    // The onset claim, stated only if the data supports one. Read off the byte
    // gap, and stated as a direction rather than a threshold, because with
    // n=3 seeds no threshold is defensible.
    let first = rows.first().map(|r| r.byte_gap).unwrap_or(0.0);
    let last = rows.last().map(|r| r.byte_gap).unwrap_or(0.0);
    println!(
        "\nbyte gap moved {:+.3} -> {:+.3} across the axis. {}",
        first,
        last,
        if last - first > 0.05 {
            "Gap narrowing with the axis: more of the train-pool advantage \
             transfers, consistent with the model moving out of the pure-memorization regime."
        } else if first - last > 0.05 {
            "Gap WIDENING with the axis: the train-pool advantage is growing, \
             which is the signature of memorization onset along this axis."
        } else {
            "Flat byte gap: this axis is not where memorization begins. The \
             model is limited by data or capacity at every point, so the \
             question cannot be answered on this axis at these settings."
        }
    );
}
