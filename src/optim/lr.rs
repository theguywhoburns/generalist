//! Manifest-selectable learning-rate schedules.
//!
//! [`TrainConfig`](crate::train::TrainConfig) carried plain `f64` learning
//! rates, so every run was constant-LR by construction. This module makes the
//! schedule a manifest choice, which matters most right now: the Muon LR is
//! newly untuned and wants a sweep, and a sweep is only cheap if the decay
//! shape is a manifest edit rather than a recompile.
//!
//! ```json
//! "lr_muon": { "kind": "constant", "lr": 0.005 }
//! "lr_muon": { "kind": "linear", "lr": 1e-4, "max_lr": 0.005, "warmup_steps": 200 }
//! "lr_muon": { "kind": "cosine", "lr": 0.005, "min_lr": 1e-4, "total_steps": 500 }
//! "lr_muon": { "kind": "step", "lr": 0.005, "step_size": 200, "gamma": 0.5 }
//! ```
//!
//! Schedulers delegate to burn's own (`LinearLrScheduler`,
//! `CosineAnnealingLrScheduler`, `StepLrScheduler`) rather than reimplementing
//! the curves. The one thing this layer owns is the *ordering* convention,
//! which is otherwise easy to get subtly wrong: burn's schedulers emit
//! `initial_lr` on the **first** call to `step()` and the final value on the
//! last, so [`LrPair::step`] advances once per optimizer step, not per
//! micro-batch. Advancing per micro-batch would silently stretch a
//! `warmup_steps: 200` ramp across `accum_steps` times as many updates.

use burn::lr_scheduler::{
    LrScheduler, cosine::CosineAnnealingLrSchedulerConfig, linear::LinearLrSchedulerConfig,
    step::StepLrSchedulerConfig,
};

/// The one method a run needs from a schedule: the rate for the next step.
///
/// burn's `LrScheduler` is not `dyn`-compatible (it has a generic associated
/// type, so a vtable cannot be built), and the record plumbing this codebase
/// does not use — saving scheduler state into checkpoints — is what forces
/// that. Rather than restructure burn, narrow to the single operation the
/// trainer actually performs. `LrPair` advances in lockstep and is not
/// serialized, so no state is lost.
pub(crate) trait LrCurve {
    /// Advance one step and return the effective rate.
    fn next(&mut self) -> f64;
}

/// Blanket impl: anything that is a burn `LrScheduler` is a curve.
impl<S: LrScheduler> LrCurve for S {
    fn next(&mut self) -> f64 {
        LrScheduler::step(self)
    }
}

/// How one optimizer's learning rate evolves over the run.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum LrConfig {
    /// Fixed rate for every step. The historical behavior.
    Constant {
        /// The rate. Must be > 0.
        lr: f64,
    },
    /// Linear ramp from `lr` to `max_lr` over `warmup_steps`, then held at
    /// `max_lr`. The usual shape when a new optimizer needs a warmup.
    Linear {
        /// Starting rate. Must be > 0.
        lr: f64,
        /// Peak rate reached after the ramp. Must be > 0.
        max_lr: f64,
        /// Steps to ramp. Must be >= 1.
        warmup_steps: usize,
    },
    /// Cosine decay from `lr` to `min_lr` over one cycle of `total_steps`.
    ///
    /// Note: burn's scheduler **restarts** at `lr` when the cycle completes
    /// rather than holding at `min_lr`. For a single-cycle run that is
    /// indistinguishable from a decay; for a run longer than `total_steps` it
    /// means periodic warm restarts, so set `total_steps >= steps` for a true
    /// decay-to-floor.
    Cosine {
        /// Starting rate. Must be > 0.
        lr: f64,
        /// Floor reached at the end of each cycle. Must be >= 0 and <= `lr`.
        min_lr: f64,
        /// Length of one cycle, in steps. Must be >= 1.
        total_steps: usize,
    },
    /// Multiply by `gamma` every `step_size` steps.
    Step {
        /// Starting rate. Must be > 0.
        lr: f64,
        /// Steps between decays. Must be >= 1.
        step_size: usize,
        /// Per-decay multiplier. Must be > 0 and <= 1.
        gamma: f64,
    },
}

impl LrConfig {
    /// Build the schedule. `Err` is a human-readable reason this config
    /// cannot produce one; the loader turns it into a `ConfigError` naming
    /// the manifest.
    ///
    pub(crate) fn build(&self) -> Result<Box<dyn LrCurve>, String> {
        Ok(match self {
            LrConfig::Constant { lr } => {
                if *lr <= 0.0 {
                    return Err(format!("constant lr must be > 0, got {lr}"));
                }
                Box::new(burn::lr_scheduler::constant::ConstantLr::from(*lr)) as Box<dyn LrCurve>
            }
            LrConfig::Linear {
                lr,
                max_lr,
                warmup_steps,
            } => {
                // burn's `Config` derive makes fields without a `#[config(default)]`
                // positional args on `new`, and defaulted ones into `with_*`.
                LinearLrSchedulerConfig::new(*lr, *max_lr, *warmup_steps)
                    .init()
                    .map(|s| Box::new(s) as Box<dyn LrCurve>)?
            }
            LrConfig::Cosine {
                lr,
                min_lr,
                total_steps,
            } => CosineAnnealingLrSchedulerConfig::new(*lr, *total_steps)
                .with_min_lr(*min_lr)
                .init()
                .map(|s| Box::new(s) as Box<dyn LrCurve>)?,
            LrConfig::Step {
                lr,
                step_size,
                gamma,
            } => StepLrSchedulerConfig::new(*lr, *step_size)
                .with_gamma(*gamma)
                .init()
                .map(|s| Box::new(s) as Box<dyn LrCurve>)?,
        })
    }

    /// Cross-field checks, independent of burn's own `init` guards, so the
    /// loader can report a bad manifest without instantiating a backend.
    pub fn validate(&self, which: &str) -> Vec<String> {
        let at = |m: String| format!("train.{which}: {m}");
        match self {
            LrConfig::Constant { lr } => {
                if *lr <= 0.0 {
                    vec![at(format!("constant lr must be > 0, got {lr}"))]
                } else {
                    vec![]
                }
            }
            LrConfig::Linear {
                lr,
                max_lr,
                warmup_steps,
            } => {
                let mut e = Vec::new();
                if *lr <= 0.0 {
                    e.push(at(format!("linear lr must be > 0, got {lr}")));
                }
                if *max_lr <= 0.0 {
                    e.push(at(format!("linear max_lr must be > 0, got {max_lr}")));
                }
                if *warmup_steps == 0 {
                    e.push(at("linear warmup_steps must be >= 1".to_string()));
                }
                e
            }
            LrConfig::Cosine {
                lr,
                min_lr,
                total_steps,
            } => {
                let mut e = Vec::new();
                if *lr <= 0.0 {
                    e.push(at(format!("cosine lr must be > 0, got {lr}")));
                }
                if *min_lr < 0.0 {
                    e.push(at(format!("cosine min_lr must be >= 0, got {min_lr}")));
                }
                if *min_lr > *lr {
                    e.push(at(format!(
                        "cosine min_lr {min_lr} must not exceed lr {lr}"
                    )));
                }
                if *total_steps == 0 {
                    e.push(at("cosine total_steps must be >= 1".to_string()));
                }
                e
            }
            LrConfig::Step {
                lr,
                step_size,
                gamma,
            } => {
                let mut e = Vec::new();
                if *lr <= 0.0 {
                    e.push(at(format!("step lr must be > 0, got {lr}")));
                }
                if *step_size == 0 {
                    e.push(at("step step_size must be >= 1".to_string()));
                }
                if *gamma <= 0.0 {
                    e.push(at(format!("step gamma must be > 0, got {gamma}")));
                }
                e
            }
        }
    }
}

/// The two learning-rate schedulers a run drives, advanced in lockstep.
pub struct LrPair {
    muon: Box<dyn LrCurve>,
    adamw: Box<dyn LrCurve>,
}

impl LrPair {
    /// Build both schedulers. `Err` names the offending field so the loader
    /// can attribute it to `lr_muon` or `lr_adamw`.
    pub fn new(muon: &LrConfig, adamw: &LrConfig) -> Result<Self, (String, String)> {
        let m = muon.build().map_err(|e| ("lr_muon".to_string(), e))?;
        let a = adamw.build().map_err(|e| ("lr_adamw".to_string(), e))?;
        Ok(Self { muon: m, adamw: a })
    }

    /// Advance both one optimizer step and return the pair of effective rates.
    ///
    /// Called exactly once per optimizer step — never per micro-batch.
    pub fn step(&mut self) -> (f64, f64) {
        (self.muon.next(), self.adamw.next())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(cfg: &LrConfig, n: usize) -> Vec<f64> {
        let mut s = cfg.build().expect("build");
        (0..n).map(|_| s.next()).collect()
    }

    #[test]
    fn constant_is_flat() {
        let c = LrConfig::Constant { lr: 0.005 };
        assert_eq!(seq(&c, 4), vec![0.005; 4]);
        assert!(c.validate("lr_muon").is_empty());
    }

    #[test]
    fn linear_ramps_up_then_holds() {
        let c = LrConfig::Linear {
            lr: 1e-4,
            max_lr: 5e-3,
            warmup_steps: 4,
        };
        let s = seq(&c, 6);
        // Strictly increasing over the ramp...
        assert!(s[0] < s[1] && s[1] < s[2] && s[2] < s[3], "no ramp: {s:?}");
        // ...then pinned at the peak, never overshooting it.
        assert!(s[4] >= s[3] - 1e-12 && s[5] <= 5e-3 + 1e-12, "{s:?}");
        assert!(c.validate("lr_muon").is_empty());
    }

    #[test]
    fn cosine_decays_within_a_cycle_and_restarts_after() {
        // burn's cosine annealer RESETS to `lr` at the end of each cycle; it
        // does not hold at `min_lr`. Pinned here so the behavior is a
        // documented choice rather than a surprise: for a single-cycle run it
        // reads as a decay, and for longer runs it is warm restarts.
        let c = LrConfig::Cosine {
            lr: 5e-3,
            min_lr: 1e-4,
            total_steps: 4,
        };
        // 5 values span one full 4-step cycle: it emits `lr` first, reaches
        // `min_lr` last, and restarts on the following call.
        let s = seq(&c, 7);
        let cycle = &s[..5];
        for w in cycle.windows(2) {
            assert!(w[1] <= w[0] + 1e-12, "cosine increased in-cycle: {s:?}");
        }
        assert!(
            (cycle[4] - 1e-4).abs() < 1e-9,
            "did not reach min_lr: {s:?}"
        );
        // The next step restarts at the peak.
        assert!((s[5] - 5e-3).abs() < 1e-9, "no restart at cycle end: {s:?}");
        assert!(c.validate("lr_muon").is_empty());
    }

    #[test]
    fn step_drops_by_gamma_each_interval() {
        let c = LrConfig::Step {
            lr: 0.01,
            step_size: 2,
            gamma: 0.5,
        };
        let s = seq(&c, 5);
        assert!(s[0] > s[2], "no drop after step_size: {s:?}");
        assert!(s[2] > s[4], "no second drop: {s:?}");
        // Monotone non-increasing, as a decay should be.
        for w in s.windows(2) {
            assert!(w[1] <= w[0] + 1e-12, "step increased: {s:?}");
        }
    }

    #[test]
    fn bad_configs_are_rejected_with_a_reason() {
        let cases: Vec<LrConfig> = vec![
            LrConfig::Constant { lr: 0.0 },
            LrConfig::Constant { lr: -1.0 },
            LrConfig::Linear {
                lr: 0.0,
                max_lr: 0.01,
                warmup_steps: 1,
            },
            LrConfig::Linear {
                lr: 0.01,
                max_lr: 0.01,
                warmup_steps: 0,
            },
            LrConfig::Cosine {
                lr: 0.01,
                min_lr: 0.02,
                total_steps: 4,
            },
            LrConfig::Cosine {
                lr: 0.01,
                min_lr: -1.0,
                total_steps: 4,
            },
            LrConfig::Cosine {
                lr: 0.01,
                min_lr: 0.0,
                total_steps: 0,
            },
            LrConfig::Step {
                lr: 0.01,
                step_size: 0,
                gamma: 0.5,
            },
            LrConfig::Step {
                lr: 0.01,
                step_size: 1,
                gamma: 0.0,
            },
        ];
        for c in &cases {
            let errs = c.validate("lr_muon");
            assert!(!errs.is_empty(), "accepted a bad config: {c:?}");
            // Every message names the field it came from, so the loader can
            // attribute it without a second lookup.
            assert!(errs[0].contains("train.lr_muon"), "unattributed: {errs:?}");
        }
        // Crossing into burn must also fail, with our own message rather than
        // a panic: the loader reports these instead of starting a run.
        //
        // `Step { gamma: 0.0 }` is the one case burn ACCEPTS (it only warns,
        // since a zero gamma is occasionally intentional), so it is excluded
        // here and covered by validate() alone above.
        let must_fail: Vec<LrConfig> = cases
            .iter()
            .filter(|c| !matches!(c, LrConfig::Step { gamma, .. } if *gamma == 0.0))
            .cloned()
            .collect();
        for c in &must_fail {
            assert!(c.build().is_err(), "burn accepted a bad config: {c:?}");
        }
    }

    #[test]
    fn pair_attributes_failures_to_the_right_field() {
        let ok = LrConfig::Constant { lr: 1e-3 };
        let bad = LrConfig::Constant { lr: 0.0 };
        // Match the field name out of the error rather than Debug on the pair,
        // which holds trait objects.
        let which = |r: Result<LrPair, (String, String)>| match r {
            Ok(_) => "ok".to_string(),
            Err((field, _)) => field,
        };
        assert_eq!(which(LrPair::new(&bad, &ok)), "lr_muon");
        assert_eq!(which(LrPair::new(&ok, &bad)), "lr_adamw");
        assert_eq!(which(LrPair::new(&ok, &ok)), "ok");
    }

    #[test]
    fn pair_advances_both_locks() {
        let mut p = match LrPair::new(
            &LrConfig::Constant { lr: 0.005 },
            &LrConfig::Constant { lr: 3e-4 },
        ) {
            Ok(p) => p,
            Err(e) => panic!("build: {e:?}"),
        };
        for _ in 0..3 {
            assert_eq!(p.step(), (0.005, 3e-4));
        }
    }
}
