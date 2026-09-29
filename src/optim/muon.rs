//! Stock Muon: burn's `Muon` on 2D hidden matrices, no preconditioner.
//!
//! Muon ([Keller Jordan](https://kellerjordan.github.io/posts/muon/)) runs
//! SGD-momentum and then replaces each 2D parameter's update with its nearest
//! orthogonal matrix, via a quintic Newton-Schulz iteration. This module
//! carries only the knobs worth exposing in a run manifest; the algorithm
//! itself is burn's [`MuonConfig`].
//!
//! ## What replaced what
//!
//! The Newton-Muon reimplementation that used to live here pre-conditioned
//! gradients by an inverse activation second moment, `G <- inv @ G`, with
//! `inv` derived from an EWMA of the layer's input covariance. It is gone,
//! deliberately:
//!
//! - burn 0.21's portable `Tensor` API exposes no Cholesky or dense-inverse
//!   op, so that implementation was a **diagonal (Jacobi)** approximation —
//!   `inv = diag(1 / (diag(cov) + ridge))`, a per-input-feature rescale rather
//!   than a Newton step. The "Newton" in the name was aspirational.
//! - the preconditioner scaled gradients by a factor that depends on the
//!   realized activation covariance, which made `lr_muon` a moving target:
//!   the deleted unit test measured `inv ~ 16.4` on a unit covariance, and the
//!   value on a trained net is not knowable from the manifest.
//!
//! Stock Muon is the honest baseline. Measure it, and only then add curvature
//! back on top of a known-good reference.
//!
//! ## Learning-rate adjustment
//!
//! Muon rescales the step by matrix shape so the update RMS is comparable
//! across shapes. Two policies exist, and which one is active materially
//! changes what `lr_muon` means:
//!
//! | shape (Burn `[d_in, d_out]`) | param | `Original` | `MatchRmsAdamW` |
//! |---|---|---|---|
//! | `[256, 256]` | q, k, v, o | 1.000 | 3.200 |
//! | `[256, 768]` | gate, up   | 1.000 | 5.543 |
//! | `[768, 256]` | down        | 1.732 | 5.543 |
//!
//! [`AdjustLrFn::Original`] is `sqrt(max(1, d_in / d_out))` — Keller Jordan's
//! published default, and the one that matches the weight-decay exemption in
//! the Muon paper (hidden matrices train undecayed). [`AdjustLrFn::MatchRmsAdamW`]
//! is `0.2 * sqrt(max(d_in, d_out))` — Moonshot's variant, whose purpose is to
//! let a learning rate tuned for AdamW be reused unchanged for Muon.
//!
//! This project keeps **separate** `lr_muon` and `lr_adamw` knobs, so that
//! reuse-the-AdamW-LR benefit is already captured structurally and the `0.2`
//! factor is only a scale constant. `Original` is therefore the default: it is
//! the literal stock behaviour, it needs no unexplained fudge factor, and its
//! flat 1.0 on the attention matrices means `lr_muon` reads as a single
//! meaningful step size. Flip the manifest to `MatchRmsAdamW` to compare.

use burn::{
    config::Config,
    optim::{
        AdjustLrFn, MuonConfig,
        decay::WeightDecayConfig,
        momentum::MomentumConfig,
    },
};

/// The Muon knobs a run manifest may set. Everything else — the quintic
/// Newton-Schulz coefficients `(3.4445, -4.7750, 2.0315)`, the Frobenius-norm
/// epsilon, and the orthogonalization itself — is burn's and deliberately not
/// re-exposed: every value here is one a run author would actually want to
/// sweep.
#[derive(Config, Debug)]
pub struct MuonTuning {
    /// Momentum factor on the (pre-orthogonalization) gradient.
    #[config(default = 0.95)]
    pub momentum: f64,
    /// Momentum dampening. The Muon recipe uses 0.0.
    #[config(default = 0.0)]
    pub dampening: f64,
    /// Nesterov momentum. The Muon recipe uses true.
    #[config(default = true)]
    pub nesterov: bool,
    /// Newton-Schulz iterations. 5 is the reference default; more iterations
    /// track the true orthogonalization more closely at linear cost.
    #[config(default = 5)]
    pub ns_steps: usize,
    /// Shape-based learning-rate adjustment. See the module docs for the
    /// per-shape multipliers in this model.
    #[config(default = "AdjustLrFn::Original")]
    pub adjust_lr_fn: AdjustLrFn,
    /// L2 weight decay. The Muon paper exempts hidden matrices, so the
    /// default is 0.0 (= burn's `None`). Note burn applies decay AFTER
    /// orthogonalization and at the *unadjusted* learning rate.
    #[config(default = 0.0)]
    pub weight_decay: f32,
}

impl MuonTuning {
    /// Build burn's optimizer from this tuning. `weight_decay = 0.0` maps to
    /// `None` so "no decay" is the literal default rather than a no-op decay.
    pub fn to_muon_config(&self) -> MuonConfig {
        let momentum = MomentumConfig {
            momentum: self.momentum,
            dampening: self.dampening,
            nesterov: self.nesterov,
        };
        // `then_some` is eager, but the payload is a single f32 copy — the
        // `then(|| ..)` closure that would defer it buys nothing here.
        let weight_decay = (self.weight_decay > 0.0)
            .then_some(WeightDecayConfig {
                penalty: self.weight_decay,
            });
        MuonConfig::new()
            .with_momentum(momentum)
            .with_ns_steps(self.ns_steps)
            .with_adjust_lr_fn(self.adjust_lr_fn)
            .with_weight_decay(weight_decay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_muon_recipe() {
        let t = MuonTuning::new();
        assert_eq!(t.momentum, 0.95);
        assert_eq!(t.dampening, 0.0);
        assert!(t.nesterov);
        assert_eq!(t.ns_steps, 5);
        // `Original`, the published default — not the Moonshot variant.
        assert_eq!(t.adjust_lr_fn, AdjustLrFn::Original);
        // Hidden matrices train undecayed.
        assert_eq!(t.weight_decay, 0.0);
    }

    #[test]
    fn tuning_builds_a_usable_muon_config() {
        // Exercise the whole builder path, including the 0.0 -> None mapping
        // and the momentum struct-literal.
        let _decayed = MuonTuning::new()
            .with_weight_decay(0.01)
            .with_ns_steps(7)
            .with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW)
            .to_muon_config();
        let _plain = MuonTuning::new().to_muon_config();
    }

    #[test]
    fn tuning_roundtrips_through_json() {
        // Manifests deserialize this type, so serde must survive the Config
        // derive (which supplies Serialize/Deserialize itself).
        let t = MuonTuning::new().with_ns_steps(3);
        let text = serde_json::to_string(&t).expect("serialize");
        let back: MuonTuning = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back.ns_steps, 3);
        assert_eq!(back.momentum, t.momentum);
        assert_eq!(back.adjust_lr_fn, t.adjust_lr_fn);
    }
}
