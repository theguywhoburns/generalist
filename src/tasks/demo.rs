//! In-context demonstration protocol (README § "In-context demonstration
//! protocol"). Applied uniformly to every task: tasks define latent rules,
//! this module defines how demos ship with each instance.

use super::{Demo, HarnessRng, Instance, Rule, Track};

/// Per-instance metadata for metrics (accuracy-vs-k, copy-rate, leakage).
#[derive(Debug, Clone)]
pub struct InstanceInfo {
    pub task: &'static str,
    pub stage: u8,
    pub track: Track,
    pub k: usize,
    pub demos: Vec<Demo>,
    pub query_input: String,
    pub expected: String,
    pub corrupted_demo: bool,
    pub query_first: bool,
}

impl InstanceInfo {
    #[cfg(test)]
    pub fn test_info() -> Self {
        Self {
            task: "test",
            stage: 0,
            track: Track::A,
            k: 0,
            demos: vec![],
            query_input: String::new(),
            expected: String::new(),
            corrupted_demo: false,
            query_first: false,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DemoProtocol {
    /// k values sampled per instance.
    pub k_set: Vec<usize>,
    /// P(k = 0).
    pub k0_rate: f64,
    /// P(one corrupted demo | k > 0).
    pub corrupt_rate: f64,
    /// Separator pool for `input SEP output` pairs.
    pub seps: Vec<String>,
}

impl Default for DemoProtocol {
    fn default() -> Self {
        Self {
            k_set: vec![0, 1, 2, 3, 5, 8],
            k0_rate: 0.15,
            corrupt_rate: 0.05,
            seps: ["->", ":", "=", "|"].iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl DemoProtocol {
    fn sample_k(&self, rng: &mut HarnessRng) -> usize {
        if rng.prob(self.k0_rate) {
            return 0;
        }
        // Uniform over positive entries; 0 if the set has none (never spin).
        let n = self.k_set.iter().filter(|k| **k > 0).count();
        if n == 0 {
            return 0;
        }
        let mut i = rng.below(n);
        for k in &self.k_set {
            if *k > 0 {
                if i == 0 {
                    return *k;
                }
                i -= 1;
            }
        }
        unreachable!("counted {n} positive entries");
    }

    /// Build one instance: sample k, resample demos per instance from the
    /// rule, randomize layout, optionally corrupt one demo.
    pub fn build(
        &self,
        task: &'static str,
        stage: u8,
        track: Track,
        rule: &dyn Rule,
        rng: &mut HarnessRng,
    ) -> Instance {
        let k = self.sample_k(rng);
        let sep = rng.pick(&self.seps).clone();
        let mut demos: Vec<Demo> = (0..k).map(|_| rule.render_demo(rng)).collect();
        rng.shuffle(&mut demos);

        let corrupted_demo = k > 0 && rng.prob(self.corrupt_rate);
        if corrupted_demo {
            let i = rng.below(demos.len());
            demos[i].output = rule.corrupt(rng, &demos[i].output);
        }

        let query = rule.render_query(rng);
        let query_first = rng.prob(0.5) && k > 0;

        let mut prompt = String::new();
        let push_pair = |s: &mut String, input: &str, output: Option<&str>| {
            s.push_str(input);
            s.push_str(&sep);
            if let Some(o) = output {
                s.push_str(o);
            }
            s.push('\n');
        };
        if query_first {
            push_pair(&mut prompt, "Q ", None);
            prompt.push_str(&query.input);
            prompt.push_str(&sep);
            prompt.push('\n');
        }
        for d in &demos {
            push_pair(&mut prompt, &d.input, Some(&d.output));
        }
        if !query_first {
            push_pair(&mut prompt, &query.input, None);
        } else {
            prompt.push('A');
            prompt.push_str(&sep);
            prompt.push('\n');
        }

        Instance {
            prompt: prompt.into_bytes(),
            target: query.target.clone().into_bytes(),
            info: InstanceInfo {
                task,
                stage,
                track,
                k,
                demos,
                query_input: query.input,
                expected: query.target,
                corrupted_demo,
                query_first,
            },
        }
    }

    /// Copy-rate probe: fraction of model outputs that appear verbatim among
    /// demo outputs. High copy-rate + high accuracy = shortcut, not induction.
    pub fn copy_rate(demos: &[Demo], outputs: &[&str]) -> f64 {
        if outputs.is_empty() {
            return 0.0;
        }
        let hits = outputs
            .iter()
            .filter(|o| demos.iter().any(|d| d.output == **o))
            .count();
        hits as f64 / outputs.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::Query;

    struct ConstRule;
    impl Rule for ConstRule {
        fn render_demo(&self, rng: &mut HarnessRng) -> Demo {
            let n = rng.range(4, 12);
            Demo {
                input: "a".repeat(n),
                output: "b".to_string(),
            }
        }
        fn render_query(&self, rng: &mut HarnessRng) -> Query {
            Query {
                input: "a".repeat(rng.range(16, 32)),
                target: "b".to_string(),
            }
        }
        fn verify(&self, _input: &str, output: &str) -> bool {
            output == "b"
        }
    }

    #[test]
    fn k_zero_has_no_demos() {
        let p = DemoProtocol {
            k_set: vec![0],
            ..Default::default()
        };
        let mut rng = HarnessRng::new(1);
        let inst = p.build("t", 0, Track::A, &ConstRule, &mut rng);
        assert_eq!(inst.info.k, 0);
        assert!(inst.info.demos.is_empty());
        assert!(!inst.info.corrupted_demo);
    }

    #[test]
    fn layouts_use_pool_separators() {
        let p = DemoProtocol::default();
        let mut rng = HarnessRng::new(2);
        for _ in 0..50 {
            let inst = p.build("t", 0, Track::A, &ConstRule, &mut rng);
            let prompt = String::from_utf8(inst.prompt).unwrap();
            assert!(
                p.seps.iter().any(|s| prompt.contains(s.as_str())),
                "prompt lacks a known separator: {prompt:?}"
            );
        }
    }

    #[test]
    fn corruption_hits_outputs_only() {
        let p = DemoProtocol {
            corrupt_rate: 1.0,
            k_set: vec![3],
            ..Default::default()
        };
        let mut rng = HarnessRng::new(3);
        let mut saw = false;
        for _ in 0..20 {
            let inst = p.build("t", 0, Track::A, &ConstRule, &mut rng);
            if inst.info.corrupted_demo {
                saw = true;
                assert!(inst.info.demos.iter().any(|d| d.output != "b"));
            }
        }
        assert!(saw);
    }

    #[test]
    fn copy_rate_counts_verbatim() {
        let demos = vec![
            Demo { input: "x".into(), output: "aa".into() },
            Demo { input: "y".into(), output: "bb".into() },
        ];
        assert!((DemoProtocol::copy_rate(&demos, &["aa", "zz"]) - 0.5).abs() < 1e-9);
        assert_eq!(DemoProtocol::copy_rate(&demos, &[]), 0.0);
    }
}
