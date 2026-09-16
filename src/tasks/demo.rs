//! In-context demonstration protocol (README § "In-context demonstration
//! protocol"). Applied uniformly to every task: tasks define latent rules,
//! this module defines how demos ship with each instance.

use super::{Demo, DemoPolicy, HarnessRng, Instance, Rule, Track};

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
    /// Bounded resample tries per demo slot. Large enough that exclusion /
    /// balancing always succeeds on real tasks (output spaces are far from
    /// singleton); on exhaustion the last render is kept so build can never
    /// infinite-loop (same lesson as `sample_k`).
    const RESAMPLE_TRIES: usize = 32;
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

        // Demo-output policy gate (runs AFTER corruption on the final demos
        // shipped in the prompt, so the guarantee holds as stated):
        // - ExcludeAnswer: resample any demo whose output exactly equals the
        //   query target via fresh regime-correct `render_demo` calls.
        // - Balanced: repair demo classes to equal per-instance coverage.
        // All rng use is sequential, so identical seeds give identical
        // instances. Known interaction (see docs on `DemoPolicy`): if the 5%
        // corruption produced the target (ExcludeAnswer) or broke class
        // balance (Balanced), enforcement overwrites that corrupted demo,
        // microscopically lowering the effective corruption rate in exchange
        // for a hard anti-copy guarantee. Layout randomization (shuffle,
        // query-first, separator) only reorders text, never outputs, so it
        // cannot undermine the guarantee.
        match rule.demo_policy() {
            DemoPolicy::ExcludeAnswer => {
                for demo in demos.iter_mut() {
                    let mut tries = 0;
                    while demo.output == query.target && tries < Self::RESAMPLE_TRIES {
                        *demo = rule.render_demo(rng);
                        tries += 1;
                    }
                }
            }
            DemoPolicy::Balanced { classes } => {
                enforce_balanced(rule, rng, &mut demos, &classes);
            }
        }

        let mut prompt = String::new();
        // Oracle header first (if the rule states itself explicitly): part
        // of the unscored prompt context, never of the target.
        if let Some(header) = rule.oracle_header() {
            prompt.push_str(&header);
            prompt.push('\n');
        }
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
}

/// Desired per-class demo counts for [`DemoPolicy::Balanced`]: even k splits
/// evenly; odd k gives one side `ceil(k/2)`, picking the side deterministically
/// from `rng` so instances alternate which class holds the majority. k <= 1 is
/// left to the caller (k = 0 is vacuous, k = 1 stays as sampled).
fn balanced_counts(k: usize, rng: &mut HarnessRng) -> (usize, usize) {
    let base = k / 2;
    if k.is_multiple_of(2) {
        (base, base)
    } else if rng.below(2) == 0 {
        (base + 1, base)
    } else {
        (base, base + 1)
    }
}

/// Repair `demos` in place to exact `classes` coverage. Demos already showing
/// a still-needed class are kept (positions stay shuffled); over-quota or
/// non-class demos (e.g. a corrupted output) are resampled via fresh
/// regime-correct `render_demo` calls until they fill a remaining need.
/// Bounded tries per slot, keep-last on exhaustion: never loops forever.
fn enforce_balanced(rule: &dyn Rule, rng: &mut HarnessRng, demos: &mut [Demo], classes: &[String]) {
    let k = demos.len();
    if k <= 1 || classes.len() != 2 {
        return;
    }
    let (mut need0, mut need1) = balanced_counts(k, rng);
    for demo in demos.iter_mut() {
        if demo.output == classes[0] && need0 > 0 {
            need0 -= 1;
            continue;
        }
        if demo.output == classes[1] && need1 > 0 {
            need1 -= 1;
            continue;
        }
        let mut kept: Option<Demo> = None;
        for _ in 0..DemoProtocol::RESAMPLE_TRIES {
            let d = rule.render_demo(rng);
            let wants0 = d.output == classes[0] && need0 > 0;
            let wants1 = d.output == classes[1] && need1 > 0;
            if wants0 || wants1 {
                if wants0 {
                    need0 -= 1;
                } else {
                    need1 -= 1;
                }
                *demo = d;
                kept = None;
                break;
            }
            kept = Some(d);
        }
        if let Some(d) = kept {
            if d.output == classes[0] {
                need0 = need0.saturating_sub(1);
            } else if d.output == classes[1] {
                need1 = need1.saturating_sub(1);
            }
            *demo = d;
        }
    }
}

/// Shared anti-copy fuzz: build `n` instances per track with the default
/// protocol and assert no demo output equals the query target. Task test
/// modules call this instead of duplicating the loop.
#[cfg(test)]
pub(crate) fn fuzz_no_demo_equals_target(task: &dyn super::Task, seed: u64, n: usize) {
    let proto = DemoProtocol::default();
    let mut rng = HarnessRng::new(seed);
    for _ in 0..n {
        for track in [Track::A, Track::B] {
            let rule = task.sample_rule(&mut rng, track);
            let inst = proto.build(task.name(), task.stage(), track, rule.as_ref(), &mut rng);
            assert!(
                !inst
                    .info
                    .demos
                    .iter()
                    .any(|d| d.output == inst.info.expected),
                "{} {track:?}: a demo copies the query target {:?}",
                task.name(),
                inst.info.expected,
            );
        }
    }
}

impl DemoProtocol {
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

    #[test]
    fn balanced_counts_shapes() {
        let mut rng = HarnessRng::new(5);
        assert_eq!(balanced_counts(8, &mut rng), (4, 4));
        assert_eq!(balanced_counts(2, &mut rng), (1, 1));
        assert_eq!(balanced_counts(0, &mut rng), (0, 0));
        // Odd k: ceil/floor split, majority side varies across draws.
        let mut saw_first = false;
        let mut saw_second = false;
        for _ in 0..20 {
            let (a, b) = balanced_counts(5, &mut rng);
            assert_eq!(a + b, 5);
            assert_eq!(a.abs_diff(b), 1);
            saw_first |= a == 3;
            saw_second |= b == 3;
        }
        assert!(
            saw_first && saw_second,
            "odd-k majority side never alternated"
        );
        let (a, b) = balanced_counts(3, &mut rng);
        assert_eq!(a + b, 3);
        assert_eq!(a.abs_diff(b), 1);
    }

    /// Binary-output stub under the DEFAULT (ExcludeAnswer) policy. Demos
    /// naturally match the target ~50% of the time, so this fuzz only passes
    /// if enforcement actively strips copies (uses the default protocol with
    /// corruption on, proving the post-corruption guarantee too).
    struct BinaryRule;
    impl Rule for BinaryRule {
        fn render_demo(&self, rng: &mut HarnessRng) -> Demo {
            Demo {
                input: "x".to_string(),
                output: if rng.prob(0.5) {
                    "0".to_string()
                } else {
                    "1".to_string()
                },
            }
        }
        fn render_query(&self, rng: &mut HarnessRng) -> Query {
            Query {
                input: "y".to_string(),
                target: if rng.prob(0.5) {
                    "0".to_string()
                } else {
                    "1".to_string()
                },
            }
        }
        fn verify(&self, _input: &str, output: &str) -> bool {
            output == "0" || output == "1"
        }
    }

    #[test]
    fn exclusion_strips_copies() {
        let p = DemoProtocol {
            k_set: vec![8],
            k0_rate: 0.0,
            ..Default::default()
        };
        let mut rng = HarnessRng::new(6);
        for _ in 0..200 {
            let inst = p.build("t", 0, Track::A, &BinaryRule, &mut rng);
            assert_eq!(inst.info.demos.len(), 8);
            assert!(
                inst.info
                    .demos
                    .iter()
                    .all(|d| d.output != inst.info.expected),
                "a demo copies the target {:?}",
                inst.info.expected,
            );
        }
    }

    #[test]
    fn exclusion_terminates_on_singleton_output_space() {
        // ConstRule can only emit "b" == target: bounded tries must terminate
        // and keep the last render instead of looping forever.
        let p = DemoProtocol {
            k_set: vec![3],
            k0_rate: 0.0,
            corrupt_rate: 0.0,
            ..Default::default()
        };
        let mut rng = HarnessRng::new(9);
        for _ in 0..20 {
            let inst = p.build("t", 0, Track::A, &ConstRule, &mut rng);
            assert_eq!(inst.info.demos.len(), 3);
            assert!(inst.info.demos.iter().all(|d| d.output == "b"));
        }
    }

    #[test]
    fn build_is_deterministic_across_tasks() {
        use crate::tasks::TaskRegistry;
        let registry = TaskRegistry::builtin();
        let proto = DemoProtocol::default();
        for seed in [7u64, 12345] {
            for track in [Track::A, Track::B] {
                for name in registry.names() {
                    let task = registry.get(name).unwrap();
                    let mut ra = HarnessRng::new(seed);
                    let mut rb = HarnessRng::new(seed);
                    for _ in 0..5 {
                        let rule_a = task.sample_rule(&mut ra, track);
                        let rule_b = task.sample_rule(&mut rb, track);
                        let a =
                            proto.build(task.name(), task.stage(), track, rule_a.as_ref(), &mut ra);
                        let b =
                            proto.build(task.name(), task.stage(), track, rule_b.as_ref(), &mut rb);
                        assert_eq!(a.prompt, b.prompt, "{name} {track:?}");
                        assert_eq!(a.target, b.target, "{name} {track:?}");
                        assert_eq!(a.info.demos.len(), b.info.demos.len(), "{name} {track:?}");
                        for (da, db) in a.info.demos.iter().zip(b.info.demos.iter()) {
                            assert_eq!(da.input, db.input, "{name} {track:?}");
                            assert_eq!(da.output, db.output, "{name} {track:?}");
                        }
                    }
                }
            }
        }
    }
}
