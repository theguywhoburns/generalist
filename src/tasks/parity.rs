//! Parity: is the count of a target symbol odd?
//!
//! Regime split: demos use lengths 4..12, queries 16..32 (length held out).
//! Track split: A counts `1`s (trained), B counts `0`s (rule seen only via
//! prompt demos). Same procedure, different target — minimal unseen-rule probe.

use super::{Demo, DemoPolicy, HarnessRng, Query, Rule, Task, Track};

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

    /// Binary outputs need balance, not exclusion: excluding the answer would
    /// leave every demo showing the opposite class, a trivial flip shortcut.
    fn demo_policy(&self) -> DemoPolicy {
        DemoPolicy::Balanced {
            classes: vec!["0".to_string(), "1".to_string()],
        }
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

    /// Mirror [`ParityTask::demo_policy`]: `build` enforces the rule-level
    /// policy, so both must agree (checked by the policy-table test).
    fn demo_policy(&self) -> DemoPolicy {
        DemoPolicy::Balanced {
            classes: vec!["0".to_string(), "1".to_string()],
        }
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

    #[test]
    fn demo_policy_is_balanced_binary() {
        use crate::tasks::{DemoPolicy, Track};
        use crate::tasks::{HarnessRng, Task};
        let task = ParityTask;
        match task.demo_policy() {
            DemoPolicy::Balanced { classes } => {
                assert_eq!(classes, vec!["0".to_string(), "1".to_string()]);
            }
            other => panic!("parity task should be Balanced, got {other:?}"),
        }
        let mut rng = HarnessRng::new(13);
        for track in [Track::A, Track::B] {
            let rule = task.sample_rule(&mut rng, track);
            match rule.demo_policy() {
                DemoPolicy::Balanced { classes } => {
                    assert_eq!(classes, vec!["0".to_string(), "1".to_string()]);
                }
                other => panic!("parity rule should be Balanced, got {other:?}"),
            }
        }
    }

    fn build_parity_k(
        task: &ParityTask,
        track: Track,
        k: usize,
        seed: u64,
        n: usize,
    ) -> Vec<crate::tasks::Instance> {
        use crate::tasks::{DemoProtocol, Task};
        let proto = DemoProtocol {
            k_set: vec![k],
            k0_rate: 0.0,
            corrupt_rate: 0.0,
            ..Default::default()
        };
        let mut rng = HarnessRng::new(seed);
        (0..n)
            .map(|_| {
                let rule = task.sample_rule(&mut rng, track);
                proto.build(task.name(), task.stage(), track, rule.as_ref(), &mut rng)
            })
            .collect()
    }

    #[test]
    fn balanced_demos_exact_counts() {
        use crate::tasks::Track;
        let task = ParityTask;
        // (k, (minority, majority)): exact per-instance class counts.
        for (k, lo, hi) in [(2usize, 1usize, 1usize), (3, 1, 2), (5, 2, 3), (8, 4, 4)] {
            for (ti, track) in [Track::A, Track::B].iter().enumerate() {
                let insts = build_parity_k(&task, *track, k, 100 + k as u64 * 10 + ti as u64, 200);
                let mut saw_zero_majority = false;
                let mut saw_one_majority = false;
                for inst in &insts {
                    assert_eq!(inst.info.demos.len(), k, "k={k} {track:?}");
                    let c0 = inst.info.demos.iter().filter(|d| d.output == "0").count();
                    let c1 = inst.info.demos.iter().filter(|d| d.output == "1").count();
                    assert_eq!(c0 + c1, k, "non-binary demo output at k={k} {track:?}");
                    assert_eq!(c0.min(c1), lo, "k={k} {track:?}: counts {c0}/{c1}");
                    assert_eq!(c0.max(c1), hi, "k={k} {track:?}: counts {c0}/{c1}");
                    saw_zero_majority |= c0 > c1;
                    saw_one_majority |= c1 > c0;
                }
                if lo != hi {
                    // Odd k must alternate which class holds the majority.
                    assert!(
                        saw_zero_majority && saw_one_majority,
                        "k={k} {track:?}: majority side never alternated"
                    );
                }
            }
        }
    }

    #[test]
    fn k_one_left_as_sampled() {
        use crate::tasks::Track;
        let task = ParityTask;
        // k=1 is untouched by balancing: demos stay binary, and (unlike
        // ExcludeAnswer) some instances still show the query target, proving
        // no exclusion was applied.
        for track in [Track::A, Track::B] {
            let insts = build_parity_k(&task, track, 1, 200, 200);
            let mut saw_match = false;
            for inst in &insts {
                assert_eq!(inst.info.demos.len(), 1);
                assert!(inst.info.demos[0].output == "0" || inst.info.demos[0].output == "1");
                saw_match |= inst.info.demos[0].output == inst.info.expected;
            }
            assert!(
                saw_match,
                "{track:?}: k=1 never matched target; exclusion suspected"
            );
        }
    }
}
