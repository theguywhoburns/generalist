//! Gradient partitioning glue for the hybrid optimizer stack.
//!
//! Every optimizer variant in [`OptimConfig`](super::OptimConfig) wants the
//! same partition: 2D hidden matrices go to the shape-aware optimizer, and
//! embedding / LM head / norms / halt gates go to AdamW. That split is
//! therefore expressed once, here, against the model's own semantic
//! classification ([`crate::model::ParamKind`]) rather than per optimizer.
//!
//! Burn's `OptimizerAdaptor` skips parameters absent from the passed
//! `GradientsParams`, so each adaptor steps only its own partition:
//! 1. [`split_grads`] partitions by semantic class via the model's id set;
//! 2. each adaptor is handed only the grads it owns.
//!
//! Training loop sketch (each adaptor steps only grads it receives):
//! ```ignore
//! let grads = GradientsParams::from_grads(backward, &model);
//! let (hidden_grads, adamw_grads) = split_grads::<B>(&model.muon_ids(), grads);
//! let model = muon_opt.step(lr_muon, model, hidden_grads);
//! let model = adamw_opt.step(lr_adamw, model, adamw_grads);
//! ```

use std::collections::HashSet;

use burn::{module::ParamId, optim::GradientsParams, tensor::backend::AutodiffBackend};

/// Split into (hidden-matrix grads, everything-else grads). Missing ids are
/// skipped, so a parameter that received no gradient this step is simply not
/// stepped.
/// Ids are processed in sorted order so fused-kernel graphs are identical
/// across steps (HashMap/HashSet iteration order is nondeterministic and
/// would defeat the fusion cache).
pub fn split_grads<B: AutodiffBackend>(
    hidden_ids: &HashSet<ParamId>,
    mut grads: GradientsParams,
) -> (GradientsParams, GradientsParams) {
    let mut hidden = GradientsParams::new();
    let mut ids: Vec<ParamId> = hidden_ids.iter().copied().collect();
    ids.sort();
    for id in ids {
        if let Some(g) = grads.remove::<B::InnerBackend, 2>(id) {
            hidden.register::<B::InnerBackend, 2>(id, g);
        }
    }
    (hidden, grads)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_backend::{TestBackend, test_device};
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::{Distribution, Tensor};

    type IB = <TestBackend as AutodiffBackend>::InnerBackend;

    fn linear() -> Linear<TestBackend> {
        LinearConfig::new(4, 4)
            .with_bias(false)
            .init::<TestBackend>(&test_device())
    }

    fn grads_of(model: &Linear<TestBackend>) -> GradientsParams {
        let x = Tensor::<TestBackend, 2>::random([2, 4], Distribution::Default, &test_device());
        GradientsParams::from_grads(model.forward(x).sum().backward(), model)
    }

    #[test]
    fn split_routes_hidden_ids_out_and_leaves_the_rest() {
        let a = linear();
        let mut hidden = HashSet::new();
        hidden.insert(a.weight.id);
        let (mut hidden_g, rest_g) = split_grads::<TestBackend>(&hidden, grads_of(&a));
        // The listed id is consumed into the hidden partition...
        assert!(hidden_g.remove::<IB, 2>(a.weight.id).is_some());
        // ...and nothing is left behind in either bucket.
        assert_eq!(hidden_g.len(), 0, "hidden partition leaked extras");
        assert_eq!(rest_g.len(), 0, "rest partition should be empty");
    }

    #[test]
    fn unlisted_ids_fall_through_to_the_rest() {
        // The model in `grads_of` is not in `hidden`, so its grad must survive
        // in the second partition untouched. This is the embedding/head path.
        let a = linear();
        let (hidden_g, mut rest_g) = split_grads::<TestBackend>(&HashSet::new(), grads_of(&a));
        assert_eq!(hidden_g.len(), 0, "nothing should be routed as hidden");
        assert!(rest_g.remove::<IB, 2>(a.weight.id).is_some());
    }

    #[test]
    fn merge_sums_matching_specs_and_keeps_unmatched() {
        let a = linear();
        let b = linear();
        let specs = vec![
            ("a".to_string(), a.weight.id, 2usize),
            ("b".to_string(), b.weight.id, 2),
        ];
        // acc has `a` only; next has `b` only. Merged must hold both.
        let mut acc = GradientsParams::new();
        if let Some(g) = grads_of(&a).remove::<IB, 2>(a.weight.id) {
            acc.register::<IB, 2>(a.weight.id, g);
        }
        let mut next = GradientsParams::new();
        if let Some(g) = grads_of(&b).remove::<IB, 2>(b.weight.id) {
            next.register::<IB, 2>(b.weight.id, g);
        }
        let mut merged = merge_grads::<TestBackend>(&specs, acc, next);
        assert!(merged.remove::<IB, 2>(a.weight.id).is_some());
        assert!(merged.remove::<IB, 2>(b.weight.id).is_some());
    }
}
