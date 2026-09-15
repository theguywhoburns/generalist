//! Periodic continuation: extend a repeating motif.
//!
//! Rule = (motif, period). Query asks for the next `k` symbols.
//! Regime split: demo strings 6..16 chars, query strings 16..48.
//! Track split: A uses periods {2, 3} over {a,b,c}; B uses {4, 5} over
//! {x,y,z} (longer periods on unseen symbols).

use super::{Demo, HarnessRng, Query, Rule, Task, Track};

pub struct PeriodicTask;

impl Task for PeriodicTask {
    fn name(&self) -> &'static str {
        "periodic"
    }

    fn stage(&self) -> u8 {
        0
    }

    fn sample_rule(&self, rng: &mut HarnessRng, track: Track) -> Box<dyn Rule> {
        let (alphabet, periods): (&[u8], &[usize]) = match track {
            Track::A => (b"abc", &[2, 3]),
            Track::B => (b"xyz", &[4, 5]),
        };
        let period = *rng.pick(periods);
        let mut motif: Vec<u8> = (0..period).map(|_| *rng.pick(alphabet)).collect();
        if motif.iter().all(|c| *c == motif[0]) {
            let last = motif.len() - 1;
            motif[last] = *rng.pick(&alphabet.iter().filter(|c| **c != motif[0]).copied().collect::<Vec<_>>());
        }
        Box::new(PeriodicRule { motif })
    }
}

pub struct PeriodicRule {
    pub motif: Vec<u8>,
}

impl PeriodicRule {
    fn extend(&self, prefix_len: usize, out_len: usize) -> (String, String) {
        let input: String = (0..prefix_len)
            .map(|i| self.motif[i % self.motif.len()] as char)
            .collect();
        let output: String = (prefix_len..prefix_len + out_len)
            .map(|i| self.motif[i % self.motif.len()] as char)
            .collect();
        (input, output)
    }
}

impl Rule for PeriodicRule {
    fn render_demo(&self, rng: &mut HarnessRng) -> Demo {
        let (input, output) = self.extend(rng.range(6, 16), rng.range(1, 4));
        Demo { input, output }
    }

    fn render_query(&self, rng: &mut HarnessRng) -> Query {
        let (input, target) = self.extend(rng.range(16, 48), rng.range(1, 5));
        Query { input, target }
    }

    fn verify(&self, input: &str, output: &str) -> bool {
        let output = output.trim();
        let (_, expected) = self.extend(input.len(), output.len());
        output == expected && !output.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuation_follows_motif() {
        let task = PeriodicTask;
        let mut rng = HarnessRng::new(41);
        for _ in 0..50 {
            let rule = task.sample_rule(&mut rng, Track::A);
            let q = rule.render_query(&mut rng);
            assert!(rule.verify(&q.input, &q.target));
            assert!(!rule.verify(&q.input, "zzzzzzzz"));
        }
    }

    #[test]
    fn motifs_are_nontrivial() {
        let task = PeriodicTask;
        let mut rng = HarnessRng::new(42);
        for _ in 0..20 {
            let rule = task.sample_rule(&mut rng, Track::A);
            let q = rule.render_query(&mut rng);
            assert!(!q.target.is_empty());
        }
    }

    #[test]
    fn demos_never_copy_query_target() {
        crate::tasks::demo::fuzz_no_demo_equals_target(&PeriodicTask, 304, 200);
    }
}
