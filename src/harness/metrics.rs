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
    /// Model output appeared verbatim among demo outputs.
    pub copied: bool,
    pub steps_used: usize,
    pub mean_halt: f32,
}

impl Record {
    pub fn to_json(&self) -> String {
        let track = match self.track {
            Track::A => "A",
            Track::B => "B",
        };
        format!(
            "{{\"task\":\"{}\",\"track\":\"{track}\",\"k\":{},\"correct\":{},\"copied\":{},\"steps\":{},\"halt\":{:.3}}}",
            self.task, self.k, self.correct, self.copied, self.steps_used, self.mean_halt,
        )
    }
}

#[derive(Debug, Default)]
pub struct Summary {
    pub n: usize,
    pub accuracy: f64,
    pub copy_rate: f64,
    pub mean_steps: f64,
    pub mean_halt: f64,
}

/// Aggregate records. Filter beforehand for per-(task, track, k) slices.
pub fn summarize(records: &[Record]) -> Summary {
    if records.is_empty() {
        return Summary::default();
    }
    let n = records.len() as f64;
    Summary {
        n: records.len(),
        accuracy: records.iter().filter(|r| r.correct).count() as f64 / n,
        copy_rate: records.iter().filter(|r| r.copied).count() as f64 / n,
        mean_steps: records.iter().map(|r| r.steps_used as f64).sum::<f64>() / n,
        mean_halt: records.iter().map(|r| r.mean_halt as f64).sum::<f64>() / n,
    }
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
            copied,
            steps_used: 6,
            mean_halt: 4.5,
        }
    }

    #[test]
    fn summary_math() {
        let rs = vec![rec(true, false), rec(false, true), rec(true, false), rec(true, false)];
        let s = summarize(&rs);
        assert_eq!(s.n, 4);
        assert!((s.accuracy - 0.75).abs() < 1e-9);
        assert!((s.copy_rate - 0.25).abs() < 1e-9);
        assert!((s.mean_steps - 6.0).abs() < 1e-9);
    }

    #[test]
    fn json_is_one_line_per_record() {
        let mut j = Jsonl::default();
        j.push(&rec(true, false));
        j.push(&rec(false, false));
        assert_eq!(j.text().lines().count(), 2);
        assert!(j.text().contains("\"track\":\"B\""));
    }
}
