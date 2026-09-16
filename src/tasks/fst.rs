//! Substitution FST: character-level map applied left to right.
//!
//! Each rule is a random 1:1 char map (a tiny finite-state transducer).
//! Track split uses DISJOINT alphabets: A maps within {a,b,c}, B maps within
//! {x,y,z} — Track B tests the same procedure on never-seen symbols.
//! Regime split: demo strings 2..8 chars, query strings 8..20.

use super::{Demo, HarnessRng, Query, Rule, Task, Track};

pub struct SubstFstTask;

impl Task for SubstFstTask {
    fn name(&self) -> &'static str {
        "subst-fst"
    }

    fn stage(&self) -> u8 {
        0
    }

    fn sample_rule(&self, rng: &mut HarnessRng, track: Track) -> Box<dyn Rule> {
        let alphabet: &[u8] = match track {
            Track::A => b"abc",
            Track::B => b"xyz",
        };
        // Random permutation of the alphabet as the substitution map.
        let mut targets = alphabet.to_vec();
        rng.shuffle(&mut targets);
        // Reject the identity map: it teaches copying, not transduction.
        if targets == alphabet {
            targets.swap(0, 1);
        }
        Box::new(SubstRule {
            source: alphabet.to_vec(),
            map: targets,
        })
    }
}

pub struct SubstRule {
    pub source: Vec<u8>,
    pub map: Vec<u8>,
}

impl SubstRule {
    fn apply(&self, input: &str) -> String {
        input
            .bytes()
            .map(|b| {
                self.source
                    .iter()
                    .position(|s| *s == b)
                    .map(|i| self.map[i] as char)
                    .unwrap_or('?')
            })
            .collect()
    }
}

impl Rule for SubstRule {
    fn render_demo(&self, rng: &mut HarnessRng) -> Demo {
        let input = rng.token_string(&self.source, 2, 8);
        let output = self.apply(&input);
        Demo { input, output }
    }

    fn render_query(&self, rng: &mut HarnessRng) -> Query {
        let input = rng.token_string(&self.source, 8, 20);
        let target = self.apply(&input);
        Query { input, target }
    }

    fn verify(&self, input: &str, output: &str) -> bool {
        output.trim() == self.apply(input)
    }
}

/// Fixed-map variant: ONE constant substitution for every instance
/// (`a→b b→c c→a`), no header, memorizable purely in weights. Both tracks
/// sample the identical rule. Scores above chance (ln 3 ≈ 1.10) here mean
/// the substrate owns a transduction primitive; failure means no context
/// experiment matters until that exists. Bottom rung of the staircase.
pub struct SubstFstFixedTask;

/// The constant map. Fixed point-free (teaches transduction, not copying).
const FIXED_SOURCE: &[u8] = b"abc";
const FIXED_MAP: &[u8] = b"bca";

impl Task for SubstFstFixedTask {
    fn name(&self) -> &'static str {
        "subst-fst-fixed"
    }

    fn stage(&self) -> u8 {
        0
    }

    fn sample_rule(&self, _rng: &mut HarnessRng, _track: Track) -> Box<dyn Rule> {
        Box::new(SubstRule {
            source: FIXED_SOURCE.to_vec(),
            map: FIXED_MAP.to_vec(),
        })
    }
}

/// Oracle variant: identical latent rules and targets, but the prompt states
/// the substitution explicitly (`MAP a->c b->a c->b`). Demos, regimes,
/// scoring, and the anti-copy policy are unchanged, so oracle-vs-normal
/// accuracy deltas isolate rule *execution* from rule *induction*.
pub struct SubstFstOracleTask;

impl Task for SubstFstOracleTask {
    fn name(&self) -> &'static str {
        "subst-fst-oracle"
    }

    fn stage(&self) -> u8 {
        0
    }

    fn sample_rule(&self, rng: &mut HarnessRng, track: Track) -> Box<dyn Rule> {
        // Same distribution as `SubstFstTask` (pool seeds differ by task
        // name, so instances differ — only the distribution matches).
        let alphabet: &[u8] = match track {
            Track::A => b"abc",
            Track::B => b"xyz",
        };
        let mut targets = alphabet.to_vec();
        rng.shuffle(&mut targets);
        if targets == alphabet {
            targets.swap(0, 1);
        }
        Box::new(OracleSubstRule(SubstRule {
            source: alphabet.to_vec(),
            map: targets,
        }))
    }
}

/// `SubstRule` wrapper stating the map explicitly. All behavior delegates.
pub struct OracleSubstRule(pub SubstRule);

impl Rule for OracleSubstRule {
    fn render_demo(&self, rng: &mut HarnessRng) -> Demo {
        self.0.render_demo(rng)
    }

    fn render_query(&self, rng: &mut HarnessRng) -> Query {
        self.0.render_query(rng)
    }

    fn verify(&self, input: &str, output: &str) -> bool {
        self.0.verify(input, output)
    }

    fn oracle_header(&self) -> Option<String> {
        let pairs: Vec<String> = self
            .0
            .source
            .iter()
            .zip(self.0.map.iter())
            .map(|(s, m)| format!("{}->{}", *s as char, *m as char))
            .collect();
        Some(format!("MAP {}", pairs.join(" ")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn maps_apply_and_verify() {
        let task = SubstFstTask;
        let mut rng = HarnessRng::new(31);
        for _ in 0..50 {
            let rule = task.sample_rule(&mut rng, Track::A);
            let q = rule.render_query(&mut rng);
            assert!(rule.verify(&q.input, &q.target));
            assert!(!rule.verify(&q.input, &format!("{}q", q.target)));
        }
    }

    #[test]
    fn tracks_use_disjoint_alphabets() {
        let task = SubstFstTask;
        let mut rng = HarnessRng::new(32);
        let mut a_syms = HashSet::new();
        let mut b_syms = HashSet::new();
        for _ in 0..20 {
            let a = task.sample_rule(&mut rng, Track::A);
            let b = task.sample_rule(&mut rng, Track::B);
            let q = a.render_query(&mut rng);
            a_syms.extend(q.input.bytes());
            let q = b.render_query(&mut rng);
            b_syms.extend(q.input.bytes());
        }
        assert!(a_syms.is_disjoint(&b_syms), "alphabet leak between tracks");
    }

    #[test]
    fn demos_never_copy_query_target() {
        crate::tasks::demo::fuzz_no_demo_equals_target(&SubstFstTask, 301, 200);
    }

    #[test]
    fn oracle_states_map_and_keeps_targets() {
        use crate::tasks::DemoProtocol;
        let task = SubstFstOracleTask;
        let mut rng = HarnessRng::new(303);
        let rule = task.sample_rule(&mut rng, Track::A);
        let header = rule.oracle_header().expect("oracle states its map");
        assert!(header.starts_with("MAP "));
        // Header ships first in the built prompt; the target still verifies
        // against the same rule (header is context, never scored).
        let proto = DemoProtocol::default();
        let inst = proto.build(task.name(), task.stage(), Track::A, rule.as_ref(), &mut rng);
        let prompt = String::from_utf8(inst.prompt).unwrap();
        assert!(prompt.starts_with(&format!("{header}\n")), "header first");
        let target = String::from_utf8(inst.target).unwrap();
        assert!(rule.verify(&inst.info.query_input, &target));
    }

    #[test]
    fn oracle_demos_never_copy_query_target() {
        crate::tasks::demo::fuzz_no_demo_equals_target(&SubstFstOracleTask, 304, 200);
    }

    #[test]
    fn fixed_map_is_constant_across_tracks_and_calls() {
        let task = SubstFstFixedTask;
        let mut rng = HarnessRng::new(305);
        let a = task.sample_rule(&mut rng, Track::A);
        let b = task.sample_rule(&mut rng, Track::B);
        // Identical rule both tracks; no header (pure weight memorization).
        assert!(a.oracle_header().is_none());
        let qa = a.render_query(&mut rng);
        let qb = b.render_query(&mut rng);
        assert!(a.verify(&qa.input, &qa.target));
        assert!(b.verify(&qb.input, &qb.target));
        // Fixed point-free constant map: never the identity copy.
        assert_eq!(qa.target.len(), qa.input.len());
        assert_ne!(qa.target, qa.input);
    }

    #[test]
    fn fixed_demos_never_copy_query_target() {
        crate::tasks::demo::fuzz_no_demo_equals_target(&SubstFstFixedTask, 306, 200);
    }
}
