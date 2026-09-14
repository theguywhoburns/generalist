//! Tiny SCAN: compositional command → action sequences.
//!
//! Primitives: jump→J, walk→W, turn left→L, turn right→R.
//! Modifiers: `twice` (×2), `thrice` (×3). Conjunction: `X and Y`.
//! Track A trains on single commands + `twice` + a subset of `and`
//! compositions. Track B holds out `thrice` and novel `and` pairings —
//! compositional generalization in the gSCAN spirit, at byte scale.
//! Regime split: demo commands use ≤2 clauses, queries up to 3.

use super::{Demo, HarnessRng, Query, Rule, Task, Track};

const PRIMS: &[(&str, &str)] = &[("jump", "J"), ("walk", "W"), ("turn left", "L"), ("turn right", "R")];

pub struct ScanTask;

impl Task for ScanTask {
    fn name(&self) -> &'static str {
        "scan-tiny"
    }

    fn stage(&self) -> u8 {
        0
    }

    fn sample_rule(&self, _rng: &mut HarnessRng, track: Track) -> Box<dyn Rule> {
        // The "rule" is the command language itself; track selects which
        // compositions appear. Thrice + novel pairings are Track-B-only.
        Box::new(ScanRule { track })
    }
}

pub struct ScanRule {
    pub track: Track,
}

impl ScanRule {
    fn atom(rng: &mut HarnessRng) -> (String, String) {
        let (w, a) = *rng.pick(PRIMS);
        (w.to_string(), a.to_string())
    }

    /// Render one command (1-3 clauses) under the track's allowed forms.
    fn command(&self, rng: &mut HarnessRng, max_clauses: usize) -> (String, String) {
        let n = rng.range(1, max_clauses + 1);
        let mut words = vec![];
        let mut acts = vec![];
        for _ in 0..n {
            let (w, a) = Self::atom(rng);
            // Track A: bare or twice. Track B: also thrice.
            let reps = match self.track {
                Track::A => {
                    if rng.prob(0.4) { 2 } else { 1 }
                }
                Track::B => match rng.below(3) {
                    0 => 1,
                    1 => 2,
                    _ => 3,
                },
            };
            let mut w2 = w;
            if reps == 2 {
                w2.push_str(" twice");
            } else if reps == 3 {
                w2.push_str(" thrice");
            }
            words.push(w2);
            for _ in 0..reps {
                acts.push(a.clone());
            }
        }
        // Track B biases toward multi-clause `and` compositions on top of the
        // thrice modifier above; the held-out axis is compositional form.
        if self.track == Track::B && n == 1 && rng.prob(0.5) {
            let (w, a) = Self::atom(rng);
            words.push(w);
            acts.push(a);
        }
        (words.join(" and "), acts.concat())
    }
}

impl Rule for ScanRule {
    fn render_demo(&self, rng: &mut HarnessRng) -> Demo {
        let (input, output) = self.command(rng, 2);
        Demo { input, output }
    }

    fn render_query(&self, rng: &mut HarnessRng) -> Query {
        let (input, target) = self.command(rng, 3);
        Query { input, target }
    }

    fn verify(&self, input: &str, output: &str) -> bool {
        // Re-derive: parse is deterministic, so check by re-execution.
        Self::execute(input).as_deref() == Some(output.trim())
    }

    fn corrupt(&self, rng: &mut HarnessRng, output: &str) -> String {
        // Corrupt by dropping or duplicating one action symbol.
        let mut s = output.to_string();
        if s.is_empty() || !rng.prob(0.5) {
            s.push(*rng.pick(&['J', 'W', 'L', 'R']));
        } else {
            s.pop();
        }
        if s == output {
            s.push('J');
        }
        s
    }
}

impl ScanRule {
    /// Deterministic executor for the tiny command language. `None` on
    /// malformed input (never happens for generated instances).
    fn execute(input: &str) -> Option<String> {
        let mut acts = String::new();
        for clause in input.split(" and ") {
            let clause = clause.trim();
            let (prim, reps) = if let Some(w) = clause.strip_suffix(" twice") {
                (w, 2)
            } else if let Some(w) = clause.strip_suffix(" thrice") {
                (w, 3)
            } else {
                (clause, 1)
            };
            let a = PRIMS.iter().find(|(w, _)| *w == prim)?.1;
            for _ in 0..reps {
                acts.push_str(a);
            }
        }
        Some(acts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executor_matches_renderer() {
        let rule = ScanRule { track: Track::B };
        let mut rng = HarnessRng::new(61);
        for _ in 0..200 {
            let q = rule.render_query(&mut rng);
            assert_eq!(ScanRule::execute(&q.input).as_deref(), Some(q.target.as_str()));
            assert!(rule.verify(&q.input, &q.target));
        }
    }

    #[test]
    fn executor_reference_cases() {
        assert_eq!(ScanRule::execute("jump twice and walk").as_deref(), Some("JJW"));
        assert_eq!(ScanRule::execute("turn left thrice").as_deref(), Some("LLL"));
        assert_eq!(ScanRule::execute("walk").as_deref(), Some("W"));
        assert_eq!(ScanRule::execute("bogus").as_deref(), None);
    }
}
