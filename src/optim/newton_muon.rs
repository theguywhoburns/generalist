//! Newton-Muon reimplementation for Burn (paper: arXiv:2604.01472).
//!
//! Newton-Muon = standard Muon applied **after** right-preconditioning each
//! layer gradient with an inverse activation second moment:
//! `W <- W - eta * msgn(G (Z Z^T)^-1))`, where `Z` stacks the layer inputs.
//!
//! Mapping onto this codebase (Burn stores linears as `[d_in, d_out]`, the
//! transpose of the PyTorch `[d_out, d_in]` layout, and the symmetric inverse
//! commutes with the transpose):
//! - precondition in Burn layout is a **left** multiply: `G <- inv @ G`,
//!   with `inv = (E[x x^T])^-1` over the layer's input dim;
//! - unfused Q/K/V/O + gate/up/down linears mean one `d_model x d_model`
//!   covariance covers attention and MLP inputs, plus one
//!   `ffn_hidden x ffn_hidden` covariance for the down-projection input.
//!   This replaces the paper's packed-QKV sharing and 4-block-diagonal
//!   `c_proj` approximation without changing the algorithm;
//! - EWMA (`beta = 0.95`), refresh every 32 steps, ridge
//!   `0.2 * mean(diag) + eps`, same as the reference repo.
//!
//! v1 deviation (documented, swappable): the inverse is a **diagonal (Jacobi)**
//! approximation `inv = diag(1 / (diag(cov) + ridge))`, because Burn 0.21's
//! portable Tensor API exposes no Cholesky/dense-inverse op. The refresh
//! schedule, EWMA bookkeeping, and Muon delegation (Burn's `MuonConfig`,
//! Newton-Schulz-5 with `(3.4445, -4.7750, 2.0315)`) match the paper.
//! Replace [`PrecondGroup::refresh`] internals with a full inverse when a
//! backend linalg op is available; the surrounding protocol is unchanged.
//!
//! Verification against paper (2604.01472) + reference repo:
//! | item | paper/repo | this impl | status |
//! |---|---|---|
//! | update `msgn(G (ZZ^T)^-1)` | right-precondition, then NS | `inv @ G` (Burn `[in,out]` transpose of repo layout), then Burn Muon | match |
//! | momentum 0.95 + Nesterov, applied to preconditioned grad | repo `step()` preconditions before momentum buffer | precondition grads first, delegate to Burn Muon | match |
//! | EWMA beta 0.95, refresh k=32, init 1e-3 I | paper ablations center here | defaults identical | match |
//! | ridge `0.2 * mean(diag) + eps` | repo `precond_ridge_mult=0.2` | same formula | match |
//! | second-moment normalization | mean of per-batch means | global mean over all tokens seen | intentional simplification (fewer batch-size artifacts) |
//! | inverse | damped full Cholesky | diagonal Jacobi | documented deviation (no portable inverse op) |
//! | lr scale `sqrt(max fan)` | repo multiplies update | use [`NewtonMuonConfig::muon_config_matched`] + `5x` repo lr | parity via helper |
//! | weight decay | 0 in repo GPT runs | Burn Muon default (none) | match |

use burn::{
    config::Config,
    optim::{AdjustLrFn, MuonConfig},
    tensor::{Tensor, backend::Backend},
};

use crate::model::StepStats;

/// Which input covariance a hidden matrix is preconditioned with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecondInput {
    AttnIn,
    MlpIn,
    MlpHidden,
}

#[derive(Config, Debug)]
pub struct NewtonMuonConfig {
    #[config(default = 0.95)]
    pub precond_ewma: f64,
    #[config(default = 32)]
    pub refresh_every: usize,
    #[config(default = 0.2)]
    pub ridge_mult: f64,
    #[config(default = 1e-8)]
    pub eps: f64,
    #[config(default = 1e-3)]
    pub init_diag: f64,
    #[config(default = 5)]
    pub ns_steps: usize,
}

impl NewtonMuonConfig {
    /// Burn Muon to run after the right-preconditioner. Defaults already
    /// match the paper (momentum 0.95, Nesterov, NS-5 quintic coefficients).
    pub fn muon_config(&self) -> MuonConfig {
        MuonConfig::new().with_ns_steps(self.ns_steps)
    }

    /// Paper-parity Muon config. The reference repo scales the orthogonalized
    /// update by `sqrt(max fan-in/out)`; Burn's default `Original` adjustment
    /// (`sqrt(max(1, d_in/d_out))`) does not. Under `MatchRmsAdamW`
    /// (`0.2 * sqrt(max(d_in, d_out))`) the repo-to-Burn ratio is a uniform
    /// 5.0 for every matrix shape in this model (square 256, 256x768,
    /// 768x256), so pass `lr_burn = 5 * lr_repo` to the adaptor step.
    /// Repo reference: muon lr = `0.1 * base_lr` (4e-4 at base 4e-3, d=768);
    /// start here at `2e-3` under this config and tune from the loss curve.
    pub fn muon_config_matched(&self) -> MuonConfig {
        MuonConfig::new()
            .with_ns_steps(self.ns_steps)
            .with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW)
    }
}

/// True exactly on the diagonal. (Burn's tri-masks use fill convention,
/// so `diag_mask` marks the diagonal `false`; invert it.)
fn bool_eye<B: Backend>(d: usize, device: &B::Device) -> Tensor<B, 2, burn::tensor::Bool> {
    Tensor::<B, 2, burn::tensor::Bool>::diag_mask([d, d], 0, device).bool_not()
}

/// Running input-second-moment state for one input dim.
/// The inverse is diagonal by construction; stored as a vector and applied
/// by broadcast multiply (exact same math as a dense diag-matrix matmul,
/// minus the GEMM + launches).
pub struct PrecondGroup<B: Backend> {
    pub dim: usize,
    cov: Tensor<B, 2>,
    inv_diag: Tensor<B, 1>,
    accum: Tensor<B, 2>,
    count: f64,
}

impl<B: Backend> PrecondGroup<B> {
    pub fn new(dim: usize, init_diag: f64, device: &B::Device) -> Self {
        let eye = bool_eye(dim, device);
        let zeros = Tensor::<B, 2>::zeros([dim, dim], device);
        let diag_mat = Tensor::<B, 2>::ones([dim, dim], device).mul_scalar(init_diag);
        Self {
            dim,
            cov: zeros.clone().mask_where(eye, diag_mat),
            inv_diag: Tensor::<B, 1>::ones([dim], device).mul_scalar(1.0 / init_diag),
            accum: zeros,
            count: 0.0,
        }
    }

    /// Fold summed `x^T x` statistics from a training step.
    pub fn observe(&mut self, xtx_sum: Tensor<B, 2>, n: usize) {
        self.accum = self.accum.clone() + xtx_sum;
        self.count += n as f64;
    }

    /// EWMA update + diagonal inverse refresh. No-op until data is observed.
    pub fn refresh(&mut self, ewma: f64, ridge_mult: f64, eps: f64) {
        if self.count <= 0.0 {
            return;
        }
        let device = self.cov.device();
        let d = self.dim;
        let mean_new = self.accum.clone().div_scalar(self.count);
        self.cov = self.cov.clone().mul_scalar(ewma) + mean_new.mul_scalar(1.0 - ewma);

        let eye = bool_eye::<B>(d, &device);
        let zeros = Tensor::zeros([d, d], &device);
        // diag(cov) via masked row sums.
        let diag = self
            .cov
            .clone()
            .mask_where(eye.clone().bool_not(), zeros.clone())
            .sum_dim(1);
        let mean_diag = scalar_of(&diag.clone().sum().div_scalar(d as f64));
        let ridge = mean_diag * ridge_mult as f32 + eps as f32;
        self.inv_diag = diag.reshape([d]).add_scalar(ridge as f64).powf_scalar(-1.0);

        self.accum = Tensor::zeros([d, d], &device);
        self.count = 0.0;
    }

    /// Right-preconditioner in Burn layout: `G <- inv @ G`, with diagonal
    /// `inv` applied as a broadcast row-scale (identical result, no GEMM).
    pub fn precondition(&self, grad: Tensor<B, 2>) -> Tensor<B, 2> {
        grad * self.inv_diag.clone().unsqueeze_dim::<2>(1)
    }
}

/// Newton-Muon preconditioner state (lives on the training backend's inner
/// backend). Orthogonalization/momentum itself is delegated to Burn's Muon.
pub struct NewtonMuon<B: Backend> {
    pub attn_in: PrecondGroup<B>,
    pub mlp_in: PrecondGroup<B>,
    pub mlp_hidden: PrecondGroup<B>,
    pub step: usize,
    pub ewma: f64,
    pub refresh_every: usize,
    pub ridge_mult: f64,
    pub eps: f64,
}

impl<B: Backend> NewtonMuon<B> {
    pub fn new(
        d_model: usize,
        ffn_hidden: usize,
        config: &NewtonMuonConfig,
        device: &B::Device,
    ) -> Self {
        Self {
            attn_in: PrecondGroup::new(d_model, config.init_diag, device),
            mlp_in: PrecondGroup::new(d_model, config.init_diag, device),
            mlp_hidden: PrecondGroup::new(ffn_hidden, config.init_diag, device),
            step: 0,
            ewma: config.precond_ewma,
            refresh_every: config.refresh_every,
            ridge_mult: config.ridge_mult,
            eps: config.eps,
        }
    }

    pub fn group(&self, role: PrecondInput) -> &PrecondGroup<B> {
        match role {
            PrecondInput::AttnIn => &self.attn_in,
            PrecondInput::MlpIn => &self.mlp_in,
            PrecondInput::MlpHidden => &self.mlp_hidden,
        }
    }

    /// Fold one training step's input statistics.
    pub fn observe_stats(&mut self, stats: &StepStats<B>) {
        self.attn_in.observe(stats.attn_xtx.clone(), stats.attn_n);
        self.mlp_in.observe(stats.mlp_xtx.clone(), stats.mlp_n);
        self.mlp_hidden
            .observe(stats.down_xtx.clone(), stats.down_n);
    }

    /// Advance the step counter; refresh inverses every `refresh_every` steps.
    pub fn maybe_refresh(&mut self) {
        self.step += 1;
        if self.step.is_multiple_of(self.refresh_every) {
            self.attn_in.refresh(self.ewma, self.ridge_mult, self.eps);
            self.mlp_in.refresh(self.ewma, self.ridge_mult, self.eps);
            self.mlp_hidden
                .refresh(self.ewma, self.ridge_mult, self.eps);
        }
    }

    pub fn precondition(&self, role: PrecondInput, grad: Tensor<B, 2>) -> Tensor<B, 2> {
        self.group(role).precondition(grad)
    }
}

fn scalar_of<B: Backend>(t: &Tensor<B, 1>) -> f32 {
    t.clone().into_data().as_slice::<f32>().unwrap()[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_backend::{TestBackend, test_device};

    #[test]
    fn refresh_produces_finite_positive_inverse() {
        let device = test_device();
        let cfg = NewtonMuonConfig::new();
        let mut nm = NewtonMuon::<TestBackend>::new(8, 16, &cfg, &device);
        // Identity-ish inputs: cov -> I, inv -> ~I/(1+ridge).
        let eye_like = Tensor::<TestBackend, 2>::ones([32, 8], &device);
        let (xtx, n) = {
            let n = 32;
            (eye_like.clone().transpose().matmul(eye_like), n)
        };
        let stats = StepStats {
            attn_xtx: xtx,
            attn_n: n,
            mlp_xtx: Tensor::zeros([8, 8], &device),
            mlp_n: 0,
            down_xtx: Tensor::zeros([16, 16], &device),
            down_n: 0,
        };
        nm.observe_stats(&stats);
        for _ in 0..32 {
            nm.maybe_refresh();
        }
        let g = Tensor::<TestBackend, 2>::ones([8, 8], &device);
        let pg = nm.precondition(PrecondInput::AttnIn, g);
        let vals = pg.into_data().as_slice::<f32>().unwrap().to_vec();
        assert!(vals.iter().all(|v| v.is_finite() && *v > 0.0));
        // One EWMA step (beta=0.95) from init_diag=1e-3 toward all-ones XtX:
        // diag(cov) = 0.95*1e-3 + 0.05*1 = 0.05095,
        // inv = 1 / (0.05095 * 1.2) ~= 16.36, so inv @ ones ~= 16.36.
        assert!(
            vals.iter().all(|v| (v - 16.36).abs() < 1.0),
            "unexpected preconditioned values: {vals:?}"
        );
    }

    #[test]
    fn muon_config_passthrough() {
        let cfg = NewtonMuonConfig::new().with_ns_steps(7);
        let _muon = cfg.muon_config();
    }
}
