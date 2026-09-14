//! Dyck-1 bracket completion: close the open stack.
//!
//! Input is a balanced-prefix over `()`, target is exactly the closers that
//! balance it. Regime split: demo prefixes 2..8 chars, query prefixes 8..24.
//! Track split: A caps nesting depth at 3, B allows 4..6 (deeper nesting
//! unseen in training).

use super::{Demo, HarnessRng, Query, Rule, Task, Track};

pub struct DyckTask;

impl Task for DyckTask {
    fn name(&self) -> &'static str {
        "dyck1"
    }

    fn stage(&self) -> u8 {
        0
    }

    fn sample_rule(&self, _rng: &mut HarnessRng, track: Track) -> Box<dyn Rule> {
        let max_depth = match track {
            Track::A => 3,
            Track::B => 6,
        };
        let min_depth = match track {
            Track::A => 1,
            Track::B => 4,
        };
        Box::new(DyckRule { max_depth, min_depth })
    }
}

pub struct DyckRule {
    pub max_depth: usize,
    pub min_depth: usize,
}

impl DyckRule {
    /// Random balanced string via depth-capped stack walk.
    fn balanced(&self, rng: &mut HarnessRng, lo: usize, hi: usize) -> String {
        loop {
            let mut s = String::new();
            let mut depth = 0usize;
            let target = rng.range(lo, hi);
            while s.len() < target {
                let can_open = depth < self.max_depth;
                let must_close = depth > 0 && (s.len() + depth >= target || !can_open);
                if depth == 0 || (can_open && !must_close && rng.prob(0.5)) {
                    s.push('(');
                    depth += 1;
                } else if depth > 0 {
                    s.push(')');
                    depth -= 1;
                } else {
                    break;
                }
            }
            while depth > 0 {
                s.push(')');
                depth -= 1;
            }
            if s.len() >= lo && self.depth_of(&s) >= self.min_depth {
                return s;
            }
        }
    }

    fn depth_of(&self, s: &str) -> usize {
        let mut depth = 0usize;
        let mut max = 0usize;
        for c in s.chars() {
            if c == '(' {
                depth += 1;
                max = max.max(depth);
            } else {
                depth = depth.saturating_sub(1);
            }
        }
        max
    }

    fn completion(input: &str) -> String {
        let mut depth = 0usize;
        for c in input.chars() {
            if c == '(' {
                depth += 1;
            } else {
                depth = depth.saturating_sub(1);
            }
        }
        ")".repeat(depth)
    }

    fn prefix(&self, rng: &mut HarnessRng, lo: usize, hi: usize) -> (String, String) {
        // Cut only where the open depth is > 0: guarantees a non-empty
        // completion (empty targets would poison training batches).
        let full = self.balanced(rng, lo + 2, hi + 4);
        let mut depth = 0usize;
        let mut cands = Vec::new();
        for (i, c) in full.char_indices() {
            if c == '(' {
                depth += 1;
            } else {
                depth = depth.saturating_sub(1);
            }
            if depth > 0 {
                cands.push(i + 1); // ASCII: char boundary
            }
        }
        // A non-empty balanced string always has depth > 0 somewhere.
        let cut = *rng.pick(&cands);
        let input = full[..cut].to_string();
        (input.clone(), Self::completion(&input))
    }
}

impl Rule for DyckRule {
    fn render_demo(&self, rng: &mut HarnessRng) -> Demo {
        let (input, output) = self.prefix(rng, 2, 8);
        Demo { input, output }
    }

    fn render_query(&self, rng: &mut HarnessRng) -> Query {
        let (input, target) = self.prefix(rng, 8, 24);
        Query { input, target }
    }

    fn verify(&self, input: &str, output: &str) -> bool {
        let output = output.trim();
        output.chars().all(|c| c == ')') && output == Self::completion(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_balanced(s: &str) -> bool {
        let mut depth = 0i32;
        for c in s.chars() {
            if c == '(' {
                depth += 1;
            } else if depth == 0 {
                return false;
            } else {
                depth -= 1;
            }
        }
        depth == 0
    }

    #[test]
    fn completions_balance() {
        let rule = DyckRule { max_depth: 3, min_depth: 1 };
        let mut rng = HarnessRng::new(21);
        for _ in 0..200 {
            let q = rule.render_query(&mut rng);
            let combined = format!("{}{}", q.input, q.target);
            assert!(is_balanced(&combined), "unbalanced: {combined:?}");
            assert!(rule.verify(&q.input, &q.target));
            assert!(!rule.verify(&q.input, &format!("{}(", q.target)));
        }
    }

    #[test]
    fn track_b_reaches_deeper() {
        let task = DyckTask;
        let mut rng = HarnessRng::new(22);
        let b = task.sample_rule(&mut rng, Track::B);
        let mut max_seen = 0;
        for _ in 0..200 {
            let q = b.render_query(&mut rng);
            let combined = format!("{}{}", q.input, q.target);
            let mut depth = 0;
            let mut peak = 0;
            for c in combined.chars() {
                if c == '(' {
                    depth += 1;
                    peak = peak.max(depth);
                } else {
                    depth -= 1;
                }
            }
            max_seen = max_seen.max(peak);
        }
        assert!(max_seen >= 4, "track B never exceeded depth 3");
    }
}
