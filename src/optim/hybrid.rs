//! Hybrid training glue: Muon/Newton-Muon for 2D hidden matrices,
//! AdamW for everything else (embedding, norms, halt head, LM head).
//!
//! Burn's `OptimizerAdaptor` skips parameters with no gradient in the passed
//! `GradientsParams`, so each adaptor steps only its own partition:
//! 1. [`split_grads`] partitions by the model's explicit id sets (dimension
//!    alone is insufficient: the embedding is also 2D but must use AdamW);
//! 2. [`precondition_grads`] applies the Newton-Muon right-preconditioner to
//!    the Muon partition before the Muon step.
//!
//! Training loop sketch (each adaptor steps only grads it receives):
//! ```ignore
//! let grads = GradientsParams::from_grads(backward, &model);
//! let (muon_grads, adamw_grads) = split_grads::<B>(&model.muon_ids(), grads);
//! nm.observe_stats(&stats);
//! nm.maybe_refresh();
//! let (muon_grads, leftover) = precondition_grads(&nm, &model.precond_roles(), muon_grads);
//! assert!(leftover.is_empty());
//! let model = muon_opt.step(lr_muon, model, muon_grads);
//! let model = adamw_opt.step(lr_adamw, model, adamw_grads);
//! ```

use std::collections::{HashMap, HashSet};

use burn::{
    module::ParamId,
    optim::GradientsParams,
    tensor::backend::AutodiffBackend,
};

use super::newton_muon::{NewtonMuon, PrecondInput};

/// Split into (muon_2d_grads, rest). Missing ids are skipped.
/// Ids are processed in sorted order so fused-kernel graphs are identical
/// across steps (HashMap/HashSet iteration order is nondeterministic and
/// would defeat the fusion cache).
pub fn split_grads<B: AutodiffBackend>(
    muon_ids: &HashSet<ParamId>,
    mut grads: GradientsParams,
) -> (GradientsParams, GradientsParams) {
    let mut muon = GradientsParams::new();
    let mut ids: Vec<ParamId> = muon_ids.iter().copied().collect();
    ids.sort();
    for id in ids {
        if let Some(g) = grads.remove::<B::InnerBackend, 2>(id) {
            muon.register::<B::InnerBackend, 2>(id, g);
        }
    }
    (muon, grads)
}

/// Apply `inv @ G` per role. Returns (preconditioned, leftover-without-role).
pub fn precondition_grads<B: AutodiffBackend>(
    precond: &NewtonMuon<B::InnerBackend>,
    roles: &HashMap<ParamId, PrecondInput>,
    mut grads: GradientsParams,
) -> (GradientsParams, GradientsParams) {
    let mut out: GradientsParams = GradientsParams::new();
    let mut ids: Vec<ParamId> = roles.keys().copied().collect();
    ids.sort();
    for id in ids {
        let role = roles[&id];
        if let Some(g) = grads.remove::<B::InnerBackend, 2>(id) {
            out.register::<B::InnerBackend, 2>(id, precond.precondition(role, g));
        }
    }
    (out, grads)
}

/// Merge `next` into `acc` (elementwise add per param in `specs`).
/// Gradient accumulation across micro-batches: both partitions are linear
/// in the loss (mean-reduced CE + ponder), so summed micro-grads of
/// 1/accum-scaled losses equal the full-batch grad.
pub fn merge_grads<B: AutodiffBackend>(
    specs: &[(String, ParamId, usize)],
    mut acc: GradientsParams,
    mut next: GradientsParams,
) -> GradientsParams {
    for (_, id, rank) in specs {
        match rank {
            2 => {
                let a = acc.remove::<B::InnerBackend, 2>(*id);
                let b = next.remove::<B::InnerBackend, 2>(*id);
                match (a, b) {
                    (Some(a), Some(b)) => acc.register::<B::InnerBackend, 2>(*id, a + b),
                    (Some(a), None) => acc.register::<B::InnerBackend, 2>(*id, a),
                    (None, Some(b)) => acc.register::<B::InnerBackend, 2>(*id, b),
                    (None, None) => {}
                }
            }
            _ => {
                let a = acc.remove::<B::InnerBackend, 1>(*id);
                let b = next.remove::<B::InnerBackend, 1>(*id);
                match (a, b) {
                    (Some(a), Some(b)) => acc.register::<B::InnerBackend, 1>(*id, a + b),
                    (Some(a), None) => acc.register::<B::InnerBackend, 1>(*id, a),
                    (None, Some(b)) => acc.register::<B::InnerBackend, 1>(*id, b),
                    (None, None) => {}
                }
            }
        }
    }
    acc
}
