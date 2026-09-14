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
}
