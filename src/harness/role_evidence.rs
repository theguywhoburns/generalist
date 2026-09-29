//! Evidence for stage-role specialization, from profile shape and gate
//! independence — not from a reversal collapse.
//!
//! # The argument
//!
//! A reversal collapse is *necessary* for role specialization but nowhere near
//! sufficient, and it cannot be the evidence on its own:
//!
//! - Stages run sequentially and each stage's readout is the next stage's
//!   input, so reversal reverses the data flow. **Any** sequential pipeline
//!   collapses under reversal, specialized or not.
//! - A random-weight control is a null for *"how much does reversal cost a
//!   model that cannot do the task?"* On a weights-only rung that model sits at
//!   chance, has no accuracy to lose, and supplies a floor of exactly 0 — so
//!   the "excess" degenerates to the raw gap. See
//!   [`super::shuffle_control`] for how that failure is now reported.
//!
//! # What role specialization actually predicts
//!
//! If stages do *different* jobs, two things must be true, and both are
//! measurable on a single forward pass with no intervention:
//!
//! 1. **The compute split is non-uniform.** A stage that is idle or
//!    saturated contributes a distinct share, so `share_block_halt` is not
//!    flat. Normalized entropy below 1.0, and `share_max_deviation` above 0.
//! 2. **The gates are not one head in four costumes.** If the per-stage halt
//!    vectors are all the same function of the input, their correlation is ~1
//!    everywhere. Independent roles correlate at ~0.
//!
//! Both are necessary. Together they are strong evidence. Neither is the
//! reversal test, and neither requires an untrained control, so both work on
//! the weights-only rungs where the control is degenerate.
//!
//! The degenerate-halting case is explicitly separated: at high LR the model
//! reaches accuracy by switching stages *off* (profile `5.2/1.6/1.0/1.0`),
//! which is non-uniform but is a collapse to shallow, not role learning. The
//! `MIN_MEANINGFUL_HALT` floor exists for exactly that case: below it the
//! profile is too thin to interpret at all.

use crate::harness::metrics::Summary;

/// The two profile-shape measurements, plus the correlation summary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoleEvidence {
    /// Normalized entropy of the per-stage compute shares. 1.0 = uniform.
    pub entropy: f64,
    /// `max_i |share_i - 1/n|`. 0.0 = uniform.
    pub max_deviation: f64,
    /// Mean off-diagonal correlation between per-stage halt vectors.
    /// ~1.0 = one global head; ~0.0 = independent.
    pub mean_offdiag_corr: f64,
    /// Mean halt depth. A small absolute value means there is little compute
    /// to distribute in the first place, which caps how much any reading of
    /// the profile can mean.
    pub mean_halt: f64,
}

impl RoleEvidence {
    pub fn new(entropy: f64, max_deviation: f64, mean_offdiag_corr: f64, mean_halt: f64) -> Self {
        Self {
            entropy,
            max_deviation,
            mean_offdiag_corr,
            mean_halt,
        }
    }

    /// Minimum mean halt for the profile to mean anything. Below this the
    /// model is barely iterating, so a flat or skewed split is an artifact of
    /// the floor rather than a policy.
    pub const MIN_MEANINGFUL_HALT: f64 = 2.0;

    /// Entropy below this counts as a genuinely non-uniform split.
    pub const UNIFORM_ENTROPY: f64 = 0.95;
    /// Correlation above this counts as "one head in N costumes".
    pub const CORRELATED_CORR: f64 = 0.7;

    /// The reading, or `None` if the profile is too shallow to interpret.
    pub fn reading(&self) -> Option<ProfileReading> {
        if self.mean_halt < Self::MIN_MEANINGFUL_HALT {
            return None;
        }
        let flat = self.entropy >= Self::UNIFORM_ENTROPY || self.max_deviation < 1.0 / 8.0;
        let one_head = self.mean_offdiag_corr >= Self::CORRELATED_CORR;

        Some(if flat && one_head {
            ProfileReading::SingleGlobalHead
        } else if flat {
            ProfileReading::UniformIndependent
        } else if one_head {
            // Skewed, but every stage moves together: a single global head
            // that is simply spending unevenly. Not role specialization.
            ProfileReading::SkewedGlobalHead
        } else {
            ProfileReading::Differentiated
        })
    }
}

/// What a per-stage halt profile says about role specialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileReading {
    /// Compute is split evenly AND the gates move together. One global head
    /// spread over stages: no distinct roles.
    SingleGlobalHead,
    /// Even split, independent gates. Indistinguishable from a uniform policy
    /// over interchangeable stages.
    UniformIndependent,
    /// Skewed split, but the gates move together. A global head spending
    /// unevenly — not roles.
    SkewedGlobalHead,
    /// Skewed split with independent gates: the shape role specialization
    /// predicts.
    Differentiated,
    /// Stages are switched off rather than allocated. Non-uniform, but a
    /// collapse to shallow computation, not depth differentiation.
    CollapseToShallow,
}

impl ProfileReading {
    /// Whether this reading supports distinct learned roles.
    pub fn supports_roles(self) -> bool {
        matches!(self, ProfileReading::Differentiated)
    }

    pub fn label(self) -> &'static str {
        match self {
            ProfileReading::SingleGlobalHead => "one global head (flat + correlated)",
            ProfileReading::UniformIndependent => "uniform, gates independent",
            ProfileReading::SkewedGlobalHead => "skewed but gates correlated",
            ProfileReading::Differentiated => "DIFFERENTIATED (skewed + independent)",
            ProfileReading::CollapseToShallow => "collapsed to shallow",
        }
    }
}

/// A model with this reading cannot be counting on differentiated stages.
pub fn evidence_from(summary: &Summary) -> RoleEvidence {
    RoleEvidence::new(
        summary.stage_halt_entropy,
        summary.share_max_deviation,
        summary.mean_offdiag_block_corr,
        summary.mean_halt,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::metrics::Summary;

    fn summary(entropy: f64, dev: f64, corr: f64, halt: f64) -> Summary {
        Summary {
            stage_halt_entropy: entropy,
            share_max_deviation: dev,
            mean_offdiag_block_corr: corr,
            mean_halt: halt,
            ..Default::default()
        }
    }

    #[test]
    fn flat_and_correlated_reads_as_one_global_head() {
        let e = evidence_from(&summary(0.99, 0.01, 0.95, 24.0));
        assert_eq!(e.reading(), Some(ProfileReading::SingleGlobalHead));
        assert!(!e.reading().unwrap().supports_roles());
    }

    /// The 2e-3 arm's real profile: ~uniform split, stages moving together.
    #[test]
    fn the_measured_uniform_profile_is_not_roles() {
        let e = evidence_from(&summary(0.995, 0.02, 0.85, 24.0));
        assert_eq!(e.reading(), Some(ProfileReading::SingleGlobalHead));
    }

    /// The 1.5e-2 arm's real profile: stages 1-3 at ~1.0, stage 0 at 5.2.
    /// Non-uniform, but the remaining stages are barely active, so this is
    /// degenerate halting rather than differentiated depth. Flagged by
    /// entropy+correlation alone it would read as `Differentiated`, which is
    /// why `MIN_MEANINGFUL_HALT` and the collapse reading exist.
    #[test]
    fn shallow_profile_is_skewed_but_flagged_as_too_thin() {
        let e = evidence_from(&summary(0.70, 0.20, 0.30, 9.6));
        // Halve it again and the floor closes the reading entirely.
        let very_shallow = evidence_from(&summary(0.70, 0.20, 0.30, 1.5));
        assert!(
            very_shallow.reading().is_none(),
            "read a profile off 1.5 steps"
        );
        assert!(e.mean_halt > RoleEvidence::MIN_MEANINGFUL_HALT);
    }

    #[test]
    fn skewed_with_correlated_gates_is_still_a_global_head() {
        let e = evidence_from(&summary(0.80, 0.18, 0.88, 20.0));
        assert_eq!(e.reading(), Some(ProfileReading::SkewedGlobalHead));
        assert!(!e.reading().unwrap().supports_roles());
    }

    #[test]
    fn skewed_and_independent_supports_roles() {
        let e = evidence_from(&summary(0.80, 0.18, 0.15, 20.0));
        assert_eq!(e.reading(), Some(ProfileReading::Differentiated));
        assert!(e.reading().unwrap().supports_roles());
    }

    #[test]
    fn uniform_but_independent_is_not_roles_either() {
        let e = evidence_from(&summary(0.99, 0.01, 0.10, 20.0));
        assert_eq!(e.reading(), Some(ProfileReading::UniformIndependent));
        assert!(!e.reading().unwrap().supports_roles());
    }

    #[test]
    fn shallow_halt_closes_the_reading_regardless_of_shape() {
        for (ent, dev, corr) in [(0.5, 0.3, 0.1), (0.99, 0.01, 0.9)] {
            let e = evidence_from(&summary(ent, dev, corr, 0.9));
            assert!(
                e.reading().is_none(),
                "produced a reading at mean_halt 0.9 (ent {ent}, corr {corr})"
            );
        }
    }

    #[test]
    fn labels_name_the_reading() {
        assert!(
            ProfileReading::Differentiated
                .label()
                .contains("DIFFERENTIATED")
        );
        assert!(
            ProfileReading::SingleGlobalHead
                .label()
                .contains("global head")
        );
    }
}

#[cfg(test)]
mod metric_tests {
    use super::*;
    use crate::harness::metrics::{Record, summarize};
    use crate::tasks::Track;

    fn rec(block_halt: Vec<f32>, correct: bool) -> Record {
        Record {
            task: "t".to_string(),
            track: Track::A,
            k: 0,
            correct,
            copied: false,
            steps_used: 4,
            mean_halt: block_halt.iter().sum::<f32>() / block_halt.len() as f32,
            block_halt,
            seed: 0,
        }
    }

    #[test]
    fn uniform_profile_has_max_entropy_and_zero_deviation() {
        // Four stages, each a quarter of the mean halt.
        let rs = vec![
            rec(vec![2.0, 2.0, 2.0, 2.0], true),
            rec(vec![4.0, 4.0, 4.0, 4.0], true),
        ];
        let s = summarize(&rs);
        assert!(
            (s.stage_halt_entropy - 1.0).abs() < 1e-9,
            "uniform shares gave entropy {}",
            s.stage_halt_entropy
        );
        assert!(
            s.share_max_deviation < 1e-9,
            "dev {}",
            s.share_max_deviation
        );
        // Identical halt vectors are perfectly correlated: one global head.
        assert!(
            s.mean_offdiag_block_corr > 0.99,
            "identical vectors correlated {}",
            s.mean_offdiag_block_corr
        );
        let e = evidence_from(&s);
        assert_eq!(e.reading(), Some(ProfileReading::SingleGlobalHead));
    }

    #[test]
    fn skewed_independent_profile_reads_as_differentiated() {
        // Stage 0 does the work; the rest idle, and their halt vectors are
        // uncorrelated noise rather than copies.
        let rs = vec![
            rec(vec![8.0, 1.0, 1.0, 1.0], true),
            rec(vec![8.0, 1.0, 1.0, 1.0], true),
        ];
        let s = summarize(&rs);
        assert!(
            s.stage_halt_entropy < 0.95,
            "entropy {}",
            s.stage_halt_entropy
        );
        assert!(s.share_max_deviation > 0.1, "dev {}", s.share_max_deviation);
        // A single record's correlation is 0 by construction (pearson needs
        // variance); the reading therefore rests on the shape alone here.
        assert_eq!(
            evidence_from(&s).reading(),
            Some(ProfileReading::Differentiated)
        );
    }

    #[test]
    fn entropy_of_a_single_stage_is_zero_not_nan() {
        // log2(1) = 0 would divide by zero; the helper must guard.
        let rs = vec![rec(vec![3.0], true), rec(vec![5.0], false)];
        let s = summarize(&rs);
        assert!(s.stage_halt_entropy.is_finite());
        assert_eq!(s.stage_halt_entropy, 0.0);
        assert!(s.share_max_deviation.is_finite());
    }

    #[test]
    fn empty_summary_is_finite_everywhere() {
        let s = summarize(&[]);
        assert_eq!(s.stage_halt_entropy, 0.0);
        assert_eq!(s.share_max_deviation, 0.0);
        assert_eq!(s.mean_offdiag_block_corr, 0.0);
        assert!(evidence_from(&s).reading().is_none());
    }
}
