//! Copy / reverse / repeat transduction over character strings.
//!
//! Rule = (op, alphabet). Track A: {copy, reverse} over {a,b,c} (trained).
//! Track B: {copy, reverse, repeat} over {x,y,z} — novel op plus novel
//! symbols, defined only by prompt demos. Regime split: demo strings 2..8,
//! query strings 8..20.

use super::{Demo, HarnessRng, Query, Rule, Task, Track};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyOp {
    Copy,
    Reverse,
    Repeat,
}

pub struct CopyTask;

impl Task for CopyTask {
    fn name(&self) -> &'static str {
        "copy-rev-rep"
    }

    fn stage(&self) -> u8 {
        0
    }

    fn sample_rule(&self, rng: &mut HarnessRng, track: Track) -> Box<dyn Rule> {
        let (alphabet, ops): (&[u8], &[CopyOp]) = match track {
            Track::A => (b"abc", &[CopyOp::Copy, CopyOp::Reverse]),
            Track::B => (b"xyz", &[CopyOp::Copy, CopyOp::Reverse, CopyOp::Repeat]),
        };
        Box::new(CopyRule {
            op: *rng.pick(ops),
            alphabet: alphabet.to_vec(),
        })
    }
}

pub struct CopyRule {
    pub op: CopyOp,
    pub alphabet: Vec<u8>,
}

impl CopyRule {
    fn apply(&self, input: &str) -> String {
        match self.op {
            CopyOp::Copy => input.to_string(),
            CopyOp::Reverse => input.chars().rev().collect(),
            CopyOp::Repeat => format!("{input}{input}"),
        }
    }
}

impl Rule for CopyRule {
    fn render_demo(&self, rng: &mut HarnessRng) -> Demo {
        let input = rng.token_string(&self.alphabet, 2, 8);
        let output = self.apply(&input);
        Demo { input, output }
    }

    fn render_query(&self, rng: &mut HarnessRng) -> Query {
        // Repeat doubles length: keep queries shorter so targets stay small.
        let hi = match self.op {
            CopyOp::Repeat => 10,
            _ => 20,
        };
        let input = rng.token_string(&self.alphabet, 8, hi);
        let target = self.apply(&input);
        Query { input, target }
    }

    fn verify(&self, input: &str, output: &str) -> bool {
        output.trim() == self.apply(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ops_apply_and_verify() {
        let r = CopyRule { op: CopyOp::Reverse, alphabet: b"abc".to_vec() };
        assert_eq!(r.apply("abca"), "acba");
        assert!(r.verify("abca", "acba"));
        assert!(!r.verify("abca", "abca"));
        let r = CopyRule { op: CopyOp::Repeat, alphabet: b"xy".to_vec() };
        assert_eq!(r.apply("xy"), "xyxy");
    }

    #[test]
    fn sampled_rules_verify_own_queries() {
        let task = CopyTask;
        let mut rng = HarnessRng::new(51);
        for _ in 0..50 {
            for track in [Track::A, Track::B] {
                let rule = task.sample_rule(&mut rng, track);
                let q = rule.render_query(&mut rng);
                assert!(rule.verify(&q.input, &q.target), "track {track:?}");
            }
        }
    }

    #[test]
    fn demos_never_copy_query_target() {
        crate::tasks::demo::fuzz_no_demo_equals_target(&CopyTask, 302, 200);
    }
}
