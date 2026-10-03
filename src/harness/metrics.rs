//! Metric records + hand-rolled JSONL (no new deps).
//! New metric = one function over `&[Record]`.

use std::fmt::Write as _;

use crate::tasks::Track;

/// One evaluated instance. Logged by whoever runs the model; aggregated by
/// [`summarize`].
#[derive(Debug, Clone)]
pub struct Record {
    pub task: String,
    pub track: Track,
    pub k: usize,
    pub correct: bool,
    /// Matching byte count between decoded output and target, and the target
    /// length, so [`Summary::byte_accuracy`] can be aggregated.
    ///
    /// Exact-match is a conjunction over every byte, so on a 20-byte target
    /// it reports 0.000 identically for "19 of 20 right" and "nothing right".
    /// At the scale these experiments run, the model sits firmly in the
    /// former state: a measured exact-match of 0.000 says the strict metric
    /// is floored, not that nothing was learned. Byte accuracy resolves the
    /// region exact-match cannot see.
    pub byte_hits: usize,
    pub byte_total: usize,
    /// Model output equalled the query **input** verbatim.
    ///
    /// Distinct from `copied`, which asks whether the output appeared among the
    /// demo *outputs*. The two separate the two ways a model can dodge a rule:
    /// replaying something it was shown versus echoing the question. For a
    /// transduction task whose rule is a non-identity permutation, echoing the
    /// input is guaranteed wrong, so a high value here names the failure
    /// precisely instead of leaving it to be guessed at from a chance-rate
    /// comparison.
    pub echoed_query: bool,
    /// Model output appeared verbatim among demo outputs.
    pub copied: bool,
    pub steps_used: usize,
    pub mean_halt: f32,
    /// Mean halt steps per block (where the compute happened).
    pub block_halt: Vec<f32>,
    /// The run's training seed.
    ///
    /// Carried per-record because a comparison across seeds is the whole
    /// point of a seed sweep, and a log where the seed lives only in the
    /// filename cannot be aggregated without re-joining on string parsing.
    pub seed: u64,
}

impl Record {
    pub fn to_json(&self) -> String {
        let track = match self.track {
            Track::A => "A",
            Track::B => "B",
        };
        let bh: Vec<String> = self.block_halt.iter().map(|v| format!("{v:.3}")).collect();
        format!(
            "{{\"task\":\"{}\",\"track\":\"{track}\",\"k\":{},\"correct\":{},\"byte_hits\":{},\"byte_total\":{},\"copied\":{},\"echoed_query\":{},\"steps\":{},\"halt\":{:.3},\"bh\":[{}],\"seed\":{}}}",
            self.task,
            self.k,
            self.correct,
            self.byte_hits,
            self.byte_total,
            self.copied,
            self.echoed_query,
            self.steps_used,
            self.mean_halt,
            bh.join(","),
            self.seed,
        )
    }
}

#[derive(Debug, Default)]
pub struct Summary {
    pub n: usize,
    pub accuracy: f64,
    /// Fraction of instances where the output echoed the query input verbatim.
    ///
    /// Read against `copy_rate` (which asks about demo outputs) this separates
    /// "replayed something shown" from "echoed the question", which is what
    /// distinguishes a model that ignores the rule from one that never learned
    /// to read it.
    pub query_echo_rate: f64,
    /// Micro-averaged over all target bytes: `sum(hits) / sum(total)`.
    ///
    /// Micro rather than per-instance mean because instance lengths vary, and
    /// the long targets are exactly the ones exact-match grades as all-or-
    /// nothing.
    pub byte_accuracy: f64,
    pub copy_rate: f64,
    pub mean_steps: f64,
    pub mean_halt: f64,
    /// Mean halt steps per block index (where the compute happened).
    pub mean_block_halt: Vec<f64>,
    /// Per-block means conditioned on correctness (specialization signal:
    /// does the model spend compute differently when it gets it right?).
    pub mean_block_halt_correct: Vec<f64>,
    pub mean_block_halt_wrong: Vec<f64>,
    /// Per-block p50 / p90 over records (means hide bimodal policies).
    pub p50_block_halt: Vec<f64>,
    pub p90_block_halt: Vec<f64>,
    /// Per-block std over records (zero std = point-mass policy; correlations
    /// against such columns are degenerate — see `block_halt_corr`).
    pub std_block_halt: Vec<f64>,
    /// Utilization shares: `mean_i / sum(mean)`. The computation topology —
    /// how the workload splits across blocks, comparable across cells.
    pub share_block_halt: Vec<f64>,
    /// Lower-triangle Pearson correlations between block-halt vectors
    /// (row i holds corr(i,0)..corr(i,i-1)): ~1.0 everywhere means the four
    /// gates are one global head in disguise; near-0 means differentiated.
    pub block_halt_corr: Vec<Vec<f64>>,
    /// Mean of the off-diagonal block-halt correlations, the scalar summary
    /// of the matrix above. ~1.0 => one global head; ~0 => independent gates.
    pub mean_offdiag_block_corr: f64,
    /// Normalized entropy of `share_block_halt`, in [0, 1].
    ///
    /// 1.0 = every stage spends exactly its share (one global head in
    /// disguise); lower = compute is concentrated on some stages. This is the
    /// null-profile test: it asks directly whether the stages differ, rather
    /// than inferring it from a reversal collapse.
    pub stage_halt_entropy: f64,
    /// `max_i |share_i - 1/n|`: the largest single-stage deviation from a
    /// uniform profile. 0 = uniform.
    pub share_max_deviation: f64,
}

/// Nearest-rank quantile over a sorted slice.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1;
    sorted[rank]
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n == 0 {
        return 0.0;
    }
    let (xs, ys) = (&xs[..n], &ys[..n]);
    let mx = xs.iter().sum::<f64>() / n as f64;
    let my = ys.iter().sum::<f64>() / n as f64;
    let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for (x, y) in xs.iter().zip(ys.iter()) {
        cov += (x - mx) * (y - my);
        vx += (x - mx) * (x - mx);
        vy += (y - my) * (y - my);
    }
    if vx <= 0.0 || vy <= 0.0 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// Aggregate records. Filter beforehand for per-(task, track, k) slices.
pub fn summarize(records: &[Record]) -> Summary {
    if records.is_empty() {
        return Summary::default();
    }
    let n = records.len() as f64;
    let width = records
        .iter()
        .map(|r| r.block_halt.len())
        .max()
        .unwrap_or(0);
    let col = |i: usize| -> Vec<f64> {
        records
            .iter()
            .filter_map(|r| r.block_halt.get(i).map(|v| *v as f64))
            .collect()
    };
    let mean = |v: &[f64]| -> f64 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    let correct_col = |i: usize| -> Vec<f64> {
        records
            .iter()
            .filter(|r| r.correct)
            .filter_map(|r| r.block_halt.get(i).map(|v| *v as f64))
            .collect()
    };
    let wrong_col = |i: usize| -> Vec<f64> {
        records
            .iter()
            .filter(|r| !r.correct)
            .filter_map(|r| r.block_halt.get(i).map(|v| *v as f64))
            .collect()
    };
    let mut mean_block_halt = vec![0.0; width];
    let mut mean_block_halt_correct = vec![0.0; width];
    let mut mean_block_halt_wrong = vec![0.0; width];
    let mut p50_block_halt = vec![0.0; width];
    let mut p90_block_halt = vec![0.0; width];
    let mut std_block_halt = vec![0.0; width];
    let mut share_block_halt = vec![0.0; width];
    for i in 0..width {
        let vals = col(i);
        mean_block_halt[i] = mean(&vals);
        mean_block_halt_correct[i] = mean(&correct_col(i));
        mean_block_halt_wrong[i] = mean(&wrong_col(i));
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        p50_block_halt[i] = quantile_sorted(&sorted, 0.5);
        p90_block_halt[i] = quantile_sorted(&sorted, 0.9);
        let m = mean_block_halt[i];
        std_block_halt[i] = if vals.is_empty() {
            0.0
        } else {
            (vals.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / vals.len() as f64).sqrt()
        };
    }
    let total: f64 = mean_block_halt.iter().sum();
    if total > 0.0 {
        for (sh, m) in share_block_halt.iter_mut().zip(mean_block_halt.iter()) {
            *sh = m / total;
        }
    }
    // Lower-triangle block correlations (diagnostic-only, no objective).
    let cols: Vec<Vec<f64>> = (0..width).map(col).collect();
    let mut block_halt_corr = vec![];
    let mut offdiag: Vec<f64> = Vec::new();
    for i in 0..width {
        let mut row = vec![];
        for j in 0..i {
            let c = pearson(&cols[i], &cols[j]);
            offdiag.push(c);
            row.push(c);
        }
        block_halt_corr.push(row);
    }
    let mean_offdiag_block_corr = mean(&offdiag);

    // Profile shape: how far the per-stage compute split is from uniform.
    // A uniform split is the signature of one global head spread over stages
    // that all compute the same thing.
    let stage_halt_entropy = normalized_entropy(&share_block_halt);
    let share_max_deviation = if width == 0 {
        0.0
    } else {
        let uniform = 1.0 / width as f64;
        share_block_halt
            .iter()
            .map(|s| (s - uniform).abs())
            .fold(0.0f64, f64::max)
    };
    Summary {
        n: records.len(),
        accuracy: records.iter().filter(|r| r.correct).count() as f64 / n,
        query_echo_rate: records.iter().filter(|r| r.echoed_query).count() as f64 / n,
        byte_accuracy: {
            let hits: usize = records.iter().map(|r| r.byte_hits).sum();
            let total: usize = records.iter().map(|r| r.byte_total).sum();
            if total == 0 {
                0.0
            } else {
                hits as f64 / total as f64
            }
        },
        copy_rate: records.iter().filter(|r| r.copied).count() as f64 / n,
        mean_steps: records.iter().map(|r| r.steps_used as f64).sum::<f64>() / n,
        mean_halt: records.iter().map(|r| r.mean_halt as f64).sum::<f64>() / n,
        mean_block_halt,
        mean_block_halt_correct,
        mean_block_halt_wrong,
        p50_block_halt,
        p90_block_halt,
        std_block_halt,
        share_block_halt,
        block_halt_corr,
        mean_offdiag_block_corr,
        stage_halt_entropy,
        share_max_deviation,
    }
}

/// Shannon entropy of a share vector, normalized to [0, 1] by log2(n).
///
/// Returns 0.0 for an empty or single-element vector: with one stage there is
/// no profile to be non-uniform about, and log2(1) = 0 would be a division by
/// zero.
fn normalized_entropy(shares: &[f64]) -> f64 {
    if shares.len() < 2 {
        return 0.0;
    }
    let h: f64 = shares
        .iter()
        .filter(|s| **s > 0.0)
        .map(|s| -s * s.log2())
        .sum();
    h / (shares.len() as f64).log2()
}

/// Append-only JSONL sink.
#[derive(Default)]
pub struct Jsonl {
    pub lines: Vec<String>,
}

impl Jsonl {
    pub fn push(&mut self, record: &Record) {
        self.lines.push(record.to_json());
    }

    pub fn text(&self) -> String {
        let mut s = String::new();
        for line in &self.lines {
            let _ = writeln!(s, "{line}");
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(correct: bool, copied: bool) -> Record {
        Record {
            task: "parity".to_string(),
            track: Track::B,
            k: 2,
            correct,
            byte_hits: usize::from(correct),
            byte_total: 1,
            echoed_query: false,
            copied,
            steps_used: 6,
            mean_halt: 4.5,
            block_halt: vec![1.0, 1.5, 1.0, 1.0],
            seed: 0,
        }
    }

    /// Byte accuracy is micro-averaged over bytes, and the two averaging
    /// schemes differ enough to matter: a per-instance mean would let one
    /// short, mostly-correct target outweigh a long, mostly-wrong one. This
    /// pins the choice, because exact-match is floored wherever byte accuracy
    /// is imperfect, so byte accuracy is what the sweeps actually read.
    #[test]
    fn byte_accuracy_is_micro_averaged_over_bytes() {
        let mut short = rec(false, false);
        short.byte_hits = 9;
        short.byte_total = 10; // 0.90 on a short target
        let mut long = rec(false, false);
        long.byte_hits = 2;
        long.byte_total = 100; // 0.02 on a long target
        let s = summarize(&[short, long]);
        // Micro: 11 hits / 110 bytes.
        assert!(
            (s.byte_accuracy - 0.1).abs() < 1e-9,
            "got {}",
            s.byte_accuracy
        );
        // A per-instance mean would have given (0.90 + 0.02) / 2 = 0.46, which
        // is a 4.6x overstatement and would invert any gap computed from it.
    }

    /// The whole reason byte accuracy exists: exact-match is blind to "mostly
    /// right", so a run can be simultaneously 0% exact and high byte accuracy.
    /// If this stops holding, the softer metric has stopped adding information.
    #[test]
    fn byte_accuracy_sees_what_exact_match_cannot() {
        let mut r = rec(false, false); // one instance, not exactly correct
        r.byte_hits = 19;
        r.byte_total = 20;
        let s = summarize(&[r]);
        assert_eq!(s.accuracy, 0.0, "exact-match is 0 as intended");
        assert!(
            s.byte_accuracy > 0.9,
            "byte accuracy must still read 0.95, got {}",
            s.byte_accuracy
        );
    }

    /// Zero-length targets must not divide by zero.
    #[test]
    fn byte_accuracy_is_zero_when_there_are_no_bytes() {
        let mut r = rec(false, false);
        r.byte_hits = 0;
        r.byte_total = 0;
        assert_eq!(summarize(&[r]).byte_accuracy, 0.0);
        // And the empty-set path stays total.
        assert_eq!(summarize(&[]).byte_accuracy, 0.0);
    }

    #[test]
    fn summary_math() {
        let rs = vec![
            rec(true, false),
            rec(false, true),
            rec(true, false),
            rec(true, false),
        ];
        let s = summarize(&rs);
        assert_eq!(s.n, 4);
        assert!((s.accuracy - 0.75).abs() < 1e-9);
        assert!((s.copy_rate - 0.25).abs() < 1e-9);
        assert!((s.mean_steps - 6.0).abs() < 1e-9);
    }

    #[test]
    fn summary_averages_block_halts() {
        let rs = vec![rec(true, false), rec(false, true)];
        let s = summarize(&rs);
        assert_eq!(s.mean_block_halt.len(), 4);
        assert!((s.mean_block_halt[1] - 1.5).abs() < 1e-9);
        assert!((s.mean_block_halt[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn summary_splits_block_halts_by_correctness() {
        let mut a = rec(true, false);
        a.block_halt = vec![1.0, 8.0];
        let mut b = rec(false, true);
        b.block_halt = vec![3.0, 2.0];
        let s = summarize(&[a, b]);
        assert!((s.mean_block_halt[0] - 2.0).abs() < 1e-9);
        assert!((s.mean_block_halt_correct[1] - 8.0).abs() < 1e-9);
        assert!((s.mean_block_halt_wrong[1] - 2.0).abs() < 1e-9);
        // p50 over {1.0, 3.0} (nearest-rank) is 1.0; p90 is 3.0.
        assert!((s.p50_block_halt[0] - 1.0).abs() < 1e-9);
        assert!((s.p90_block_halt[0] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn block_correlations_detect_shared_policy() {
        // Perfectly correlated gates: corr 1.0.
        let mut a = rec(true, false);
        a.block_halt = vec![1.0, 2.0, 3.0];
        let mut b = rec(false, false);
        b.block_halt = vec![2.0, 4.0, 1.0];
        let s = summarize(&[a.clone(), b]);
        assert_eq!(s.block_halt_corr.len(), 3);
        assert!((s.block_halt_corr[1][0] - 1.0).abs() < 1e-9);
        // Constant column has zero variance: corr reports 0.0, never NaN.
        let mut c = rec(true, false);
        c.block_halt = vec![5.0, 5.0, 5.0];
        let s2 = summarize(&[a, c]);
        assert!(s2.block_halt_corr[1][0].is_finite());
    }

    #[test]
    fn shares_sum_to_one_and_std_flags_point_mass() {
        let mut a = rec(true, false);
        a.block_halt = vec![2.0, 2.0];
        let mut b = rec(false, false);
        b.block_halt = vec![4.0, 2.0];
        let s = summarize(&[a, b]);
        // means [3.0, 2.0] -> shares [0.6, 0.4]
        assert!((s.share_block_halt[0] - 0.6).abs() < 1e-9);
        assert!((s.share_block_halt[1] - 0.4).abs() < 1e-9);
        assert!((s.share_block_halt.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        // constant column has zero std; varying column does not
        assert!((s.std_block_halt[1] - 0.0).abs() < 1e-9);
        assert!(s.std_block_halt[0] > 0.0);
    }

    #[test]
    fn json_is_one_line_per_record() {
        let mut j = Jsonl::default();
        j.push(&rec(true, false));
        j.push(&rec(false, false));
        assert_eq!(j.text().lines().count(), 2);
        assert!(j.text().contains("\"track\":\"B\""));
    }

    /// A seed sweep has to be aggregatable from the log alone. The seed
    /// therefore rides on every record rather than living only in the run's
    /// ckpt_dir name, which a multi-run aggregation cannot join on cheaply.
    #[test]
    fn records_carry_their_seed_into_json() {
        let mut r = rec(true, false);
        r.seed = 7;
        let text = r.to_json();
        assert!(text.contains("\"seed\":7"), "seed missing from {text}");
        // And it parses, so the field is not just textually present.
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(parsed["seed"], serde_json::json!(7));
        // Grouping by seed across records is then a plain filter.
        let mut j = Jsonl::default();
        j.push(&r);
        r.seed = 8;
        j.push(&r);
        let seeds: Vec<u64> = j
            .text()
            .lines()
            .map(|l| {
                serde_json::from_str::<serde_json::Value>(l).unwrap()["seed"]
                    .as_u64()
                    .unwrap()
            })
            .collect();
        assert_eq!(seeds, vec![7, 8]);
    }
}
