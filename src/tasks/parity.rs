//! Parity: is the count of a target symbol odd?
//!
//! Regime split: demos use lengths 4..12, queries 16..32 (length held out).
//! Track split: A counts `1`s (trained), B counts `0`s (rule seen only via
//! prompt demos). Same procedure, different target — minimal unseen-rule probe.

use super::{Demo, HarnessRng, Query, Rule, Task, Track};

pub struct ParityTask;

impl Task for ParityTask {
    fn name(&self) -> &'static str {
        "parity"
    }

    fn stage(&self) -> u8 {
        0
    }

    fn sample_rule(&self, _rng: &mut HarnessRng, track: Track) -> Box<dyn Rule> {
        let symbol = match track {
            Track::A => b'1',
            Track::B => b'0',
        };
        Box::new(ParityRule { symbol })
    }
}

pub struct ParityRule {
    pub symbol: u8,
}

impl ParityRule {
    fn parity(&self, input: &str) -> String {
        let n = input.bytes().filter(|b| *b == self.symbol).count();
        if n % 2 == 1 { "1".to_string() } else { "0".to_string() }
    }
}

impl Rule for ParityRule {
    fn render_demo(&self, rng: &mut HarnessRng) -> Demo {
        let input = rng.token_string(b"01", 4, 12);
        let output = self.parity(&input);
        Demo { input, output }
    }

    fn render_query(&self, rng: &mut HarnessRng) -> Query {
        let input = rng.token_string(b"01", 16, 32);
        let target = self.parity(&input);
        Query { input, target }
    }

    fn verify(&self, input: &str, output: &str) -> bool {
        output.trim() == self.parity(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_values_and_regimes() {
        let rule = ParityRule { symbol: b'1' };
        assert_eq!(rule.parity("1011"), "1");
        assert_eq!(rule.parity("1010"), "0");
        let mut rng = HarnessRng::new(11);
        for _ in 0..50 {
            let d = rule.render_demo(&mut rng);
            assert!((4..12).contains(&d.input.len()));
            let q = rule.render_query(&mut rng);
            assert!((16..32).contains(&q.input.len()));
            assert!(rule.verify(&q.input, &q.target));
            assert!(!rule.verify(&q.input, if q.target == "1" { "0" } else { "1" }));
        }
    }

    #[test]
    fn tracks_use_different_symbols() {
        let task = ParityTask;
        let mut rng = HarnessRng::new(12);
        let a = task.sample_rule(&mut rng, Track::A);
        let b = task.sample_rule(&mut rng, Track::B);
        // "0000" has even 1s but even... use input distinguishing the rules:
        // counting 1s in "10" -> 1 (odd); counting 0s in "10" -> 1 (odd). Bad.
        // "110": ones=2 even -> "0"; zeros=1 odd -> "1". Distinguishes.
        assert!(a.verify("110", "0"));
        assert!(b.verify("110", "1"));
    }
}
