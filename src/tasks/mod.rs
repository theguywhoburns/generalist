//! Task harness core.
//!
//! Adding a task = implement [`Task`] + [`Rule`], add one line to
//! [`TaskRegistry::builtin`]. Everything else (k-sampling, demo resampling,
//! regime separation, layout randomization, corruption, copy-rate) lives in
//! [`demo`] and applies uniformly.

pub mod demo;
pub mod dyck;
pub mod fst;
pub mod parity;
pub mod periodic;
pub mod copy;
pub mod rng;
pub mod scan;

pub use demo::{DemoProtocol, InstanceInfo};
pub use rng::HarnessRng;

/// Per-task demo-output policy enforced by [`demo::DemoProtocol::build`].
///
/// Copying demos must never score, but binary and large output spaces need
/// different treatment:
/// - [`DemoPolicy::ExcludeAnswer`] resamples any demo whose output exactly
///   equals the query target, so verbatim copying scores ~0%.
/// - [`DemoPolicy::Balanced`] (binary outputs only) instead forces demo
///   outputs to cover each class equally per instance. Rationale: for binary
///   outputs pure exclusion would make all demos uniform-opposite (every demo
///   shows the opposite of the answer), enabling a trivial flip-the-demos
///   shortcut. Balanced demos hold EVERY non-inductive strategy at exactly
///   chance (copy-random, copy-nearest, majority vote, flip-majority all sit
///   at 50%) while true induction scores 100%, so absolute accuracy
///   separates cleanly.
#[derive(Debug, Clone)]
pub enum DemoPolicy {
    ExcludeAnswer,
    Balanced { classes: Vec<String> },
}

/// Track A: rule seen in training, held-out instances.
/// Track B: rule never seen, defined only by prompt demos.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Track {
    A,
    B,
}

/// One rendered demonstration: latent rule applied to a demo-regime input.
#[derive(Debug, Clone)]
pub struct Demo {
    pub input: String,
    pub output: String,
}

/// Query in the query regime + its expected answer.
#[derive(Debug, Clone)]
pub struct Query {
    pub input: String,
    pub target: String,
}

/// A training/eval instance: byte prompt, byte target, metadata for metrics.
#[derive(Debug, Clone)]
pub struct Instance {
    pub prompt: Vec<u8>,
    pub target: Vec<u8>,
    pub info: InstanceInfo,
}

impl Instance {
    /// Byte values as i64 ids for `Tensor<_, _, Int>`.
    pub fn prompt_ids(&self) -> Vec<i64> {
        self.prompt.iter().map(|b| *b as i64).collect()
    }

    pub fn target_ids(&self) -> Vec<i64> {
        self.target.iter().map(|b| *b as i64).collect()
    }
}

/// A latent rule: renders demos/queries and judges outputs.
/// Demo and query regimes MUST differ (lengths/symbols) so models cannot
/// copy-match; each task documents its regime split.
pub trait Rule: Send + Sync {
    fn render_demo(&self, rng: &mut HarnessRng) -> Demo;
    fn render_query(&self, rng: &mut HarnessRng) -> Query;
    /// Judge a (query input, model output) pair.
    fn verify(&self, input: &str, output: &str) -> bool;

    /// Demo-output policy this rule's instances must satisfy. Default is
    /// [`DemoPolicy::ExcludeAnswer`]; only binary-output rules override this
    /// (parity). Mirrors [`Task::demo_policy`]: `build` only sees `&dyn Rule`,
    /// so the rule carries the policy to the enforcement point while the task
    /// declares the canonical per-task config. Both must agree.
    fn demo_policy(&self) -> DemoPolicy {
        DemoPolicy::ExcludeAnswer
    }

    /// Explicit statement of the latent rule, prepended to the prompt when
    /// `Some` (oracle tasks). Default `None`: demos alone carry the rule.
    /// The header is context, never scored — targets and verification are
    /// unchanged, so oracle-vs-normal deltas isolate rule *execution* from
    /// rule *induction*.
    fn oracle_header(&self) -> Option<String> {
        None
    }

    /// Produce a wrong output for the ~5% corrupted-demo instances.
    /// Default: flip one character to a different ASCII symbol.
    fn corrupt(&self, rng: &mut HarnessRng, output: &str) -> String {
        const POOL: &[u8] = b"01ab()[]+-*";
        let mut bytes = output.as_bytes().to_vec();
        if bytes.is_empty() {
            return "X".to_string();
        }
        let i = rng.below(bytes.len());
        let mut replacement = *rng.pick(POOL);
        let mut guard = 0;
        while replacement == bytes[i] && guard < 16 {
            replacement = *rng.pick(POOL);
            guard += 1;
        }
        bytes[i] = replacement;
        String::from_utf8(bytes).unwrap_or_else(|_| "X".to_string())
    }
}

/// A task family: samples latent rules per track.
pub trait Task: Send + Sync {
    fn name(&self) -> &'static str;
    fn stage(&self) -> u8;
    fn sample_rule(&self, rng: &mut HarnessRng, track: Track) -> Box<dyn Rule>;
    /// Canonical per-task demo-output policy. Default is
    /// [`DemoPolicy::ExcludeAnswer`]; only parity overrides this (binary
    /// outputs need [`DemoPolicy::Balanced`], see its docs). The sampled
    /// rules mirror this via [`Rule::demo_policy`], which is what `build`
    /// enforces.
    fn demo_policy(&self) -> DemoPolicy {
        DemoPolicy::ExcludeAnswer
    }
}

/// Central registry. One line per task in [`TaskRegistry::builtin`].
#[derive(Default)]
pub struct TaskRegistry {
    tasks: Vec<Box<dyn Task>>,
}

impl TaskRegistry {
    pub fn builtin() -> Self {
        let mut r = Self::default();
        r.register(Box::new(parity::ParityTask));
        r.register(Box::new(dyck::DyckTask));
        r.register(Box::new(fst::SubstFstTask));
        r.register(Box::new(fst::SubstFstOracleTask));
        r.register(Box::new(periodic::PeriodicTask));
        r.register(Box::new(copy::CopyTask));
        r.register(Box::new(scan::ScanTask));
        r
    }

    pub fn register(&mut self, task: Box<dyn Task>) {
        self.tasks.push(task);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Task> {
        self.tasks.iter().find(|t| t.name() == name).map(|t| t.as_ref())
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.tasks.iter().map(|t| t.name()).collect()
    }

    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_lists_builtin_tasks() {
        let r = TaskRegistry::builtin();
        assert_eq!(r.len(), 7);
        for name in [
            "parity",
            "dyck1",
            "subst-fst",
            "subst-fst-oracle",
            "periodic",
            "copy-rev-rep",
            "scan-tiny",
        ] {
            assert!(r.get(name).is_some(), "missing {name}");
        }
    }

    #[test]
    fn demo_policy_table() {
        // Per-task demo-output policy: parity is Balanced (binary outputs),
        // every other task is ExcludeAnswer (large output spaces).
        let r = TaskRegistry::builtin();
        for name in [
            "dyck1",
            "subst-fst",
            "subst-fst-oracle",
            "periodic",
            "copy-rev-rep",
            "scan-tiny",
        ] {
            assert!(
                matches!(
                    r.get(name).unwrap().demo_policy(),
                    DemoPolicy::ExcludeAnswer
                ),
                "{name} should be ExcludeAnswer"
            );
        }
        match r.get("parity").unwrap().demo_policy() {
            DemoPolicy::Balanced { classes } => {
                assert_eq!(classes, vec!["0".to_string(), "1".to_string()]);
            }
            other => panic!("parity should be Balanced, got {other:?}"),
        }
        // Sampled rules must mirror their task's policy (build enforces the
        // rule-level policy).
        let mut rng = HarnessRng::new(99);
        for name in r.names() {
            let task = r.get(name).unwrap();
            for track in [Track::A, Track::B] {
                let rule = task.sample_rule(&mut rng, track);
                let task_balanced = matches!(task.demo_policy(), DemoPolicy::Balanced { .. });
                let rule_balanced = matches!(rule.demo_policy(), DemoPolicy::Balanced { .. });
                assert_eq!(
                    task_balanced, rule_balanced,
                    "{name} {track:?} task/rule mismatch"
                );
            }
        }
    }

    #[test]
    fn instance_ids_roundtrip_bytes() {
        let inst = Instance {
            prompt: b"01->1".to_vec(),
            target: b"1".to_vec(),
            info: InstanceInfo::test_info(),
        };
        assert_eq!(inst.prompt_ids(), vec![48, 49, 45, 62, 49]);
        assert_eq!(inst.target_ids(), vec![49]);
    }
}
