//! Solve BOTH arms of the compute-matched depth-vs-width comparison.
//!
//! The depth axis measured held-out accuracy rising 0.134 -> 0.433 as loops
//! went 1 -> 8 at fixed parameters, which reads as "looping buys generalization".
//! But depth *is* compute, so it equally reads as "more arithmetic helps". The
//! two are separated by a pair at matched parameters AND matched block-steps:
//!
//! ```text
//!   looping : n_stages S, max_loops L  ->  S * L block-steps per token
//!   wide    : n_stages S*L, max_loops 1 -> S*L * 1 block-steps per token
//! ```
//!
//! Matching parameters across an 8x stage-count difference is the fiddly part,
//! and hand-matching got it 50% wrong on the first attempt. This solves it:
//!
//! - `d_model` must equal `n_heads * head_dim` or the config validator rejects
//!   it, so head_dim is searched rather than assumed.
//! - parameters scale roughly with d^2, so `d_model` alone steps past the
//!   target. `ffn_hidden` is a second, gentler knob; a binary search on it
//!   closes the gap.
//! - head *count* cannot be matched across the two arms — an 8x stage ratio
//!   forces an ~8x body-parameter ratio and no width choice removes that — so
//!   both arms are pinned to the same `head_dim` instead. Per-head geometry
//!   matched; head count is not, and the output says so.
//!
//! ```text
//! cargo run --example param_match -- 8 4
//! ```
//!
//! # The stages<->width frontier
//!
//! A third argument pins the parameter target instead of deriving it from the
//! looping arm's shape. That is what a frontier needs: holding parameters and
//! block-steps fixed while `stages` varies *is* the tradeoff being measured, and
//! the derived target moves with `stages`, so it can never hold parameters fixed.
//!
//! ```text
//! for pair in 8/2 4/4 2/8; do   # max_loops/n_stages, all = 16 block-steps
//!   cargo run --example param_match -- $loops $stages 492418
//! done
//! ```
//!
//! With a pinned target this also prints a `frontier` point with a freely
//! searched `head_dim`, which is the config to put in the manifest for that
//! point. Unlike the two-arm pair, a frontier point unavoidably varies head
//! count: trading stages for width at fixed parameters *is* trading heads for
//! stages. That is the axis, not a confound in it.
//!
//! # blocks_per_stage, the fourth argument
//!
//! `param_match -- <loops> <stages> <target> <blocks_per_stage>`. Block
//! applications per token is `n_stages * blocks_per_stage * max_loops`, and this
//! is **not** interchangeable with `n_stages`:
//!
//! - `param_count` scales the encoder stack by `n_stages * blocks_per_stage`, so
//!   it changes the parameter total;
//! - the halting gates scale by `n_stages` **alone**, so one stage holding four
//!   blocks has **one** gate, not four.
//!
//! It is therefore the knob that separates *how many distinct parameter sets* from
//! *how many gates watch them*. At `n_stages = 1, blocks_per_stage = 4,
//! max_loops = 4` the whole four-block stack is re-applied four times under a
//! single gate — "global ACT over the stack", per the `LoopedConfig` doc — against
//! `n_stages = 4, blocks_per_stage = 1` where each block carries its own gate.
//! Same 16 block applications, different halting structure and different grouping
//! of the repetition.

use generalist::model::LoopedConfig;

/// A model shape, kept as a struct rather than eight positional arguments.
///
/// Two reasons. Clippy caps this at seven, and — more usefully — `n_stages` and
/// `blocks_per_stage` are the two knobs this file exists to keep distinct (they
/// scale the block stack and the halting gates differently), so naming them at
/// every call site is cheaper than remembering which is which.
#[derive(Clone, Copy)]
struct Shape {
    n_stages: usize,
    max_loops: usize,
    blocks_per_stage: usize,
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    ffn_hidden: usize,
}

fn count(s: Shape) -> usize {
    LoopedConfig {
        n_stages: s.n_stages,
        max_loops: s.max_loops,
        blocks_per_stage: s.blocks_per_stage,
        d_model: s.d_model,
        n_heads: s.n_heads,
        head_dim: s.head_dim,
        ffn_hidden: s.ffn_hidden,
        ..LoopedConfig::new()
    }
    .param_count()
}

/// Best `(head_dim, n_heads, d_model, ffn_hidden)` reaching `target`, searching
/// `head_dim` only when `pinned_head_dim` is `None`.
fn solve(
    n_stages: usize,
    max_loops: usize,
    bps: usize,
    target: usize,
    pinned_head_dim: Option<usize>,
) -> (usize, usize, usize, usize, i64) {
    let dims: Vec<usize> = match pinned_head_dim {
        Some(h) => vec![h],
        None => vec![16, 32, 64, 128],
    };
    let mut best: Option<(usize, usize, usize, usize, i64)> = None;
    for hd in dims {
        for nh in 1..=32usize {
            let d = hd * nh;
            if !(64..=1024).contains(&d) {
                continue;
            }
            let at = |ffn: usize| {
                count(Shape {
                    n_stages,
                    max_loops,
                    blocks_per_stage: bps,
                    d_model: d,
                    n_heads: nh,
                    head_dim: hd,
                    ffn_hidden: ffn,
                })
            };
            let (mut lo, mut hi) = (8usize, d * 8);
            while lo < hi {
                let mid = (lo + hi) / 2;
                if at(mid) < target {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let mut ffn = lo;
            let mut bd = (at(ffn) as i64 - target as i64).abs();
            for cand in [ffn.saturating_sub(8), ffn, ffn + 8] {
                if cand < 8 {
                    continue;
                }
                let delta = (at(cand) as i64 - target as i64).abs();
                if delta < bd {
                    bd = delta;
                    ffn = cand;
                }
            }
            let delta = at(ffn) as i64 - target as i64;
            if best.is_none_or(|(_, _, _, _, b)| delta.abs() < b.abs()) {
                best = Some((hd, nh, d, ffn, delta));
            }
        }
    }
    best.expect("grid is non-empty")
}

fn show(label: &str, s: Shape) {
    println!(
        "  {label:<9} head_dim {:>3}  n_heads {:>2}  d_model {:>4}  \
         ffn_hidden {:>4}  n_stages {:>2}  max_loops {}  \
         blocks_per_stage {:>2}  = {} block-steps/token",
        s.head_dim,
        s.n_heads,
        s.d_model,
        s.ffn_hidden,
        s.n_stages,
        s.max_loops,
        s.blocks_per_stage,
        s.n_stages * s.blocks_per_stage * s.max_loops
    );
}

/// Build the shape the solver settled on, for printing and for recounting.
fn shape_of(
    n_stages: usize,
    max_loops: usize,
    bps: usize,
    hd: usize,
    nh: usize,
    d: usize,
    ffn: usize,
) -> Shape {
    Shape {
        n_stages,
        max_loops,
        blocks_per_stage: bps,
        d_model: d,
        n_heads: nh,
        head_dim: hd,
        ffn_hidden: ffn,
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let loops: usize = a.first().and_then(|s| s.parse().ok()).unwrap_or(8);
    let stages: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);

    // An explicit third argument pins the parameter target instead of deriving
    // it from the looping arm's own shape. That is what a stages<->width sweep
    // needs: the derived target moves with `stages`, so it can never hold
    // parameters fixed while the split varies.
    let explicit: Option<usize> = a.get(2).and_then(|s| s.parse().ok());
    // Fourth argument: encoder layers per stage. Block applications per token is
    // n_stages * blocks_per_stage * max_loops, and `param_count` scales the block
    // stack by blocks_per_stage while the halting gates scale by n_stages ALONE --
    // one stage holding four blocks has ONE gate, not four. So this argument
    // changes both the parameter total and the arithmetic, and it is not
    // interchangeable with n_stages.
    let bps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let target = explicit.unwrap_or_else(|| {
        count(Shape {
            n_stages: stages,
            max_loops: loops,
            blocks_per_stage: bps,
            d_model: 128,
            n_heads: 2,
            head_dim: 64,
            ffn_hidden: 384,
        })
    });
    println!(
        "compute-matched pair: {stages} stages x {loops} loops x {bps} blocks/stage \
         = {} block-steps/token",
        stages * bps * loops
    );
    match explicit {
        Some(t) => println!("target parameter count, pinned on the command line: {t}\n"),
        None => println!("target parameter count from the looping arm's shape: {target}\n"),
    }

    let (whd, wnh, wd, wffn, wdelta) = solve(stages * loops, 1, 1, target, None);
    let (lhd, lnh, ld, lffn, ldelta) = solve(stages, loops, bps, target, Some(whd));
    if explicit.is_some() {
        // Free head_dim on the frontier point itself. Pinning it to the wide
        // arm's head_dim would reintroduce the head-count confound the pinned
        // mode exists to avoid.
        let (fhd, fnh, fd, fffn, fdelta) = solve(stages, loops, bps, target, None);
        show("frontier", shape_of(stages, loops, bps, fhd, fnh, fd, fffn));
        println!("\n  frontier point param total:");
        println!(
            "    {} ({fdelta:+}), target {target}, off by {:.2}%",
            count(Shape {
                n_stages: stages,
                max_loops: loops,
                blocks_per_stage: bps,
                d_model: fd,
                n_heads: fnh,
                head_dim: fhd,
                ffn_hidden: fffn,
            }),
            100.0 * (fdelta as f64 / target as f64)
        );
    }
    println!("ARMS (search head_dim for the wide arm, then pin the looping arm to it):");
    show("looping", shape_of(stages, loops, bps, lhd, lnh, ld, lffn));
    show("wide", shape_of(stages * loops, 1, 1, whd, wnh, wd, wffn));

    println!("\nparam totals:");
    println!(
        "  looping {} ({ldelta:+})",
        count(Shape {
            n_stages: stages,
            max_loops: loops,
            blocks_per_stage: bps,
            d_model: ld,
            n_heads: lnh,
            head_dim: lhd,
            ffn_hidden: lffn,
        })
    );
    println!(
        "  wide    {} ({wdelta:+})",
        count(Shape {
            n_stages: stages * loops,
            max_loops: 1,
            blocks_per_stage: 1,
            d_model: wd,
            n_heads: wnh,
            head_dim: whd,
            ffn_hidden: wffn,
        })
    );
    let worst = ldelta.abs().max(wdelta.abs());
    if worst > target as i64 / 50 {
        println!("\nWARNING: worst arm is more than 2% off target.");
    } else {
        println!("\nboth arms within 2% of target.");
    }
    println!("\nhead_dim is matched ({whd}) across both arms; head COUNT is not, and");
    println!(
        "cannot be: an {}-x stage ratio forces an ~{}-x body-parameter ratio.",
        loops, loops
    );
    println!("If the wide arm wins, rule out the head-count difference first.");

    println!("\nSet in the two manifests:");
    println!(
        "  looping: \"model\": {{ \"d_model\": {ld}, \"n_heads\": {lnh}, \"head_dim\": {lhd}, \
         \"ffn_hidden\": {lffn}, \"n_stages\": {stages}, \"max_loops\": {loops}, \
         \"blocks_per_stage\": {bps} }}"
    );
    println!(
        "  wide:    \"model\": {{ \"d_model\": {wd}, \"n_heads\": {wnh}, \"head_dim\": {whd}, \
         \"ffn_hidden\": {wffn}, \"n_stages\": {}, \"max_loops\": 1 }}",
        stages * loops
    );
    println!("\nsweep `depth gpu {loops}` for looping and `depth gpu 1` for wide.");
}
