//! Optimizer selection: the manifest's swap point.
//!
//! [`OptimConfig`] is internally tagged, so a run manifest names its
//! optimizer in place:
//!
//! ```json
//! "optim": { "kind": "muon", "ns_steps": 5, "adjust_lr_fn": "Original" }
//! ```
//!
//! The block is a **complete** spec, not a patch: burn's `Config` derive emits
//! no `#[serde(default)]`, so every knob must be present or the manifest is
//! rejected with `missing field ...`. That matches `LoopedConfig` and
//! `TrainConfig` and keeps a checked-in manifest a full description of the
//! run. To obtain a block to edit rather than hand-writing one, round-trip a
//! config through [`crate::harness::RunConfig::save_json`].
//!
//! Swapping optimizers is a one-line manifest edit plus one match arm in
//! [`crate::train::Trainer::new`] and [`crate::train::Trainer::optimizer_step`].
//! No code changes elsewhere: the model classifies its own parameters
//! ([`crate::model::ParamKind`]) and the split into "hidden matrices" vs
//! "everything else" is the same partition for every optimizer family, since
//! embedding / LM head / norms / halt gates stay on AdamW by the Muon recipe
//! (and would stay on AdamW under any optimizer that has no business
//! orthogonalizing a 1-row embedding).

pub mod hybrid;
pub mod lr;
pub mod muon;

use serde::{Deserialize, Serialize};

pub use hybrid::{merge_grads, split_grads};
pub use lr::{LrConfig, LrPair};
pub use muon::MuonTuning;

/// The optimizer stack a run uses.
///
/// The variant selects the **2D hidden-matrix** optimizer only; the
/// remainder of the parameter set always rides AdamW (see the module docs).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum OptimConfig {
    /// Hybrid stock Muon + AdamW. 2D hidden matrices on Muon, everything
    /// else on AdamW.
    Muon(MuonTuning),
}

impl OptimConfig {
    /// Human-readable name, for log lines.
    pub fn name(&self) -> &'static str {
        match self {
            OptimConfig::Muon(_) => "muon",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::optim::AdjustLrFn;

    #[test]
    fn muon_variant_is_tagged_in_json() {
        // The manifest contract: `"kind": "muon"` selects the variant and the
        // payload fields sit beside it (internally tagged, not nested).
        let cfg = OptimConfig::Muon(MuonTuning::new().with_ns_steps(3));
        let text = serde_json::to_string(&cfg).expect("serialize");
        assert!(text.contains("\"kind\":\"muon\""), "missing tag: {text}");
        let back: OptimConfig = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back.name(), "muon");
        match back {
            OptimConfig::Muon(t) => assert_eq!(t.ns_steps, 3),
        }
    }

    #[test]
    fn both_lr_adjustments_roundtrip_through_a_manifest() {
        // The comparison the two options exist for must be expressible
        // purely as a manifest edit.
        for (lit, want) in [
            (r#""Original""#, AdjustLrFn::Original),
            (r#""MatchRmsAdamW""#, AdjustLrFn::MatchRmsAdamW),
        ] {
            let json = format!(
                r#"{{"kind":"muon","momentum":0.95,"dampening":0.0,"nesterov":true,"ns_steps":5,"adjust_lr_fn":{lit},"weight_decay":0.0}}"#
            );
            let cfg: OptimConfig = serde_json::from_str(&json).expect("deserialize");
            match cfg {
                OptimConfig::Muon(t) => assert_eq!(t.adjust_lr_fn, want),
            }
        }
    }

    #[test]
    fn partial_optim_block_is_rejected() {
        // burn's `Config` derive emits no `#[serde(default)]`, so a manifest
        // is a COMPLETE optimizer spec, not a patch. Same convention as
        // `LoopedConfig`/`TrainConfig`, and it means a checked-in manifest
        // fully determines the run. To get a block to edit, round-trip a
        // complete config through `RunConfig::save_json`.
        let err = serde_json::from_str::<OptimConfig>(r#"{"kind":"muon","ns_steps":3}"#);
        assert!(err.is_err(), "partial optim block was silently accepted");
    }

    #[test]
    fn unknown_kind_is_rejected() {
        // A typo must fail loudly rather than silently falling back.
        let err = serde_json::from_str::<OptimConfig>(r#"{"kind":"adamw"}"#);
        assert!(err.is_err(), "unknown optimizer kind was accepted");
    }
}
