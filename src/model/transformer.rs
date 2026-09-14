use std::collections::{HashMap, HashSet};

use burn::{
    module::{Module, ParamId},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    tensor::{Bool, Int, Tensor, TensorData, backend::{AutodiffBackend, Backend}},
};

use super::{
    block::LoopedBlock,
    config::{LoopedConfig, StopMode},
    halting::HaltingHead,
};
use crate::optim::PrecondInput;

/// Graves ACT halting threshold: a token halts once cumulative halt mass
/// reaches `1 - ACT_EPS`.
const ACT_EPS: f64 = 0.01;

pub struct LoopOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    /// Mean Graves ponder `mean(N_t + R_t)` over tokens (in-graph for Act).
    pub ponder: Tensor<B, 1>,
    /// Loop iterations actually executed.
    pub steps_used: usize,
    /// Host-side mean halt step, for logging.
    pub mean_halt: f32,
}

/// Sufficient statistics for the Newton-Muon input covariances, summed over
/// loop iterations: `attn`/`mlp` see `[N, d_model]` inputs, `down` sees
/// `[N, ffn_hidden]`.
pub struct StepStats<B: Backend> {
    pub attn_xtx: Tensor<B, 2>,
    pub attn_n: usize,
    pub mlp_xtx: Tensor<B, 2>,
    pub mlp_n: usize,
    pub down_xtx: Tensor<B, 2>,
    pub down_n: usize,
}

fn xtx_sum<B: Backend>(x: Tensor<B, 3>) -> (Tensor<B, 2>, usize) {
    let [b, t, d] = x.dims();
    let n = b * t;
    let x2 = x.reshape([n, d]);
    (x2.clone().transpose().matmul(x2), n)
}

/// The 1M looped character transformer (Option A, untied head).
#[derive(Module, Debug)]
pub struct LoopedTransformer<B: Backend> {
    pub embed: Embedding<B>,
    pub block: LoopedBlock<B>,
    pub norm_f: RmsNorm<B>,
    pub halt: HaltingHead<B>,
    pub head: Linear<B>,
}

impl<B: Backend> LoopedTransformer<B> {
    pub fn new(config: &LoopedConfig, device: &B::Device) -> Self {
        config.assert_valid();
        Self {
            embed: EmbeddingConfig::new(config.vocab_size, config.d_model)
                .with_initializer(burn::module::Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                })
                .init(device),
            block: LoopedBlock::new(config, device),
            norm_f: RmsNormConfig::new(config.d_model).init(device),
            halt: HaltingHead::new(config.d_model, config.halt_bias_init, device),
            head: LinearConfig::new(config.d_model, config.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }

    fn logits(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.head.forward(self.norm_f.forward(x))
    }

    /// Dispatch on stop mode. `lengths` holds the real (unpadded) length of
    /// each batch row; pad positions are blocked from attention and excluded
    /// from ponder means and the loss (via the collator's mask).
    pub fn forward(
        &self,
        tokens: Tensor<B, 2, Int>,
        config: &LoopedConfig,
        mode: StopMode,
        lengths: &[usize],
    ) -> LoopOutput<B> {
        match mode {
            StopMode::Fixed { loops } => self.forward_fixed(tokens, lengths, loops),
            StopMode::Act => self.forward_act(tokens, config, false, lengths).0,
            StopMode::Converge => self.forward_converge(tokens, config, lengths),
        }
    }

    /// Fixed-depth loop. No halting head, zero ponder.
    pub fn forward_fixed(
        &self,
        tokens: Tensor<B, 2, Int>,
        lengths: &[usize],
        loops: usize,
    ) -> LoopOutput<B> {
        let device = tokens.device();
        let [_, t] = tokens.dims();
        let key_pad = pad_mask(lengths, t, &device);
        let mut x = self.embed.forward(tokens);
        for _ in 0..loops {
            x = self.block.forward_masked(x, Some(key_pad.clone()));
        }
        let logits = self.logits(x);
        LoopOutput {
            logits,
            ponder: Tensor::zeros([1], &device),
            steps_used: loops,
            mean_halt: loops as f32,
        }
    }

    /// Graves-style ACT loop with per-token halting, copy-through freezing of
    /// halted states, and remainder-weighted trajectory readout.
    /// Returns output plus input-covariance sufficient statistics when
    /// `collect_stats` is set (training path; use `forward_act_with_stats`
    /// on an autodiff backend to get detached inner-backend stats).
    pub fn forward_act(
        &self,
        tokens: Tensor<B, 2, Int>,
        config: &LoopedConfig,
        collect_stats: bool,
        lengths: &[usize],
    ) -> (LoopOutput<B>, Option<StepStats<B>>) {
        let device = tokens.device();
        let [b, t] = tokens.dims();
        let d = config.d_model;
        let key_pad = pad_mask(lengths, t, &device);
        let keep_f = key_pad.clone().bool_not().float();
        let keep_d = keep_f
            .clone()
            .unsqueeze_dim::<3>(2)
            .repeat_dim(2, d);
        let keep_h = keep_f
            .clone()
            .unsqueeze_dim::<3>(2)
            .repeat_dim(2, config.ffn_hidden);
        let real_n: usize = lengths.iter().sum();

        let mut x = self.embed.forward(tokens);
        let mut out = Tensor::zeros([b, t, d], &device);
        let mut cum = Tensor::zeros([b, t], &device);
        let mut ponder = Tensor::zeros([b, t], &device);
        let mut halt_step = Tensor::zeros([b, t], &device);
        let mut stats = collect_stats.then(|| StepStats {
            attn_xtx: Tensor::zeros([d, d], &device),
            attn_n: 0,
            mlp_xtx: Tensor::zeros([d, d], &device),
            mlp_n: 0,
            down_xtx: Tensor::zeros([config.ffn_hidden, config.ffn_hidden], &device),
            down_n: 0,
        });

        let max = config.max_loops;
        let mut steps_used = max;
        // Loop-invariant constants hoisted: reusing them across iterations
        // avoids re-allocating identical tensors 8x per forward (backend
        // buffers pool-reuse anyway; this kills the launches too).
        let zeros_bt = Tensor::<B, 2>::zeros([b, t], &device);
        let ones_bt = Tensor::<B, 2>::ones([b, t], &device);
        for s in 1..=max {
            // Running = unhalted AND real (pad rows never run, so the loop
            // can still early-exit on padded batches).
            let still: Tensor<B, 2, burn::tensor::Bool> = cum
                .clone()
                .lower_equal_elem(1.0 - ACT_EPS)
                .bool_and(key_pad.clone().bool_not());
            let still_f = still.clone().float();

            if collect_stats && let Some(s) = stats.as_mut() {
                let a_in = self.block.norm1.forward(x.clone());
                let m_pre = x.clone()
                    + self
                        .block
                        .attn
                        .forward_masked(a_in.clone(), Some(key_pad.clone()))
                        .mul_scalar(self.block.scale);
                let m_in = self.block.norm2.forward(m_pre);
                let h_down = self.block.mlp.hidden(m_in.clone());
                let (sum, _) = xtx_sum(a_in * keep_d.clone());
                s.attn_xtx = s.attn_xtx.clone() + sum;
                s.attn_n += real_n;
                let (sum, _) = xtx_sum(m_in * keep_d.clone());
                s.mlp_xtx = s.mlp_xtx.clone() + sum;
                s.mlp_n += real_n;
                let (sum, _) = xtx_sum(h_down * keep_h.clone());
                s.down_xtx = s.down_xtx.clone() + sum;
                s.down_n += real_n;
            }

            let x_new = self
                .block
                .forward_masked(x.clone(), Some(key_pad.clone()));
            // Freeze halted states so running tokens attend to stable keys/values.
            // Single select op; replaces ones/sub/mul/mul/add with identical math.
            let still3 = still.clone().unsqueeze_dim::<3>(2).repeat_dim(2, d);
            x = x.mask_where(still3, x_new);

            let p = self.halt.probs(x.clone());
            let p_run = zeros_bt.clone().mask_where(still.clone(), p);

            if s == max {
                // Force-halt everything still running: remainder weight.
                // Maxed-out tokens pay the full ponder price.
                let rem = zeros_bt.clone().mask_where(still.clone(), ones_bt.clone() - cum.clone());
                out = out + rem.clone().unsqueeze_dim::<3>(2) * x.clone();
                cum = cum + rem.clone();
                ponder = ponder + still_f.clone() + rem.clone();
                halt_step = halt_step + rem.mul_scalar(s as f64);
            } else {
                let cum_try = cum.clone() + p_run.clone();
                let halt_now = cum_try.clone().greater_equal_elem(1.0 - ACT_EPS);
                let rem = zeros_bt
                    .clone()
                    .mask_where(still.clone(), ones_bt.clone() - cum.clone());
                let w = p_run.mask_where(halt_now.clone(), rem);
                out = out + w.clone().unsqueeze_dim::<3>(2) * x.clone();
                cum = cum + w.clone();
                ponder = ponder + still_f.clone() + w.clone() * halt_now.float();
                halt_step = halt_step + (w * still_f.clone()).mul_scalar(s as f64);
            }

            // Break check every 2nd loop (+final): the host sync stalls the
            // pipeline, and the break only skips halted tail iterations.
            // Ponder/stats accounting is unaffected.
            let running: Tensor<B, 1> = (still_f * keep_f.clone()).sum();
            if (s == max || s % 2 == 0) && scalar_of(&running) == 0.0 {
                steps_used = s;
                break;
            }
        }

        // Means over real tokens only; pads never ran.
        let denom = keep_f.clone().sum().clamp_min(1.0);
        let ponder_mean = (ponder * keep_f.clone()).sum().div(denom.clone());
        let mean_halt = scalar_of(&(halt_step * keep_f).sum().div(denom));
        let logits = self.logits(out);
        (
            LoopOutput {
                logits,
                ponder: ponder_mean,
                steps_used,
                mean_halt,
            },
            stats,
        )
    }

    /// Latent-convergence loop (RD-VLA style): run until the relative state
    /// change stays below `conv_tol` for `conv_patience` steps. No halt head.
    /// Ponder is the constant step count (no gradient; CE carries training).
    pub fn forward_converge(
        &self,
        tokens: Tensor<B, 2, Int>,
        config: &LoopedConfig,
        lengths: &[usize],
    ) -> LoopOutput<B> {
        let device = tokens.device();
        let [_, t] = tokens.dims();
        let key_pad = pad_mask(lengths, t, &device);
        let mut x = self.embed.forward(tokens);
        let mut calm = 0usize;
        let mut steps_used = config.max_loops;
        for s in 1..=config.max_loops {
            let x_new = self
                .block
                .forward_masked(x.clone(), Some(key_pad.clone()));
            let num = scalar_of(
                &(x_new.clone() - x.clone())
                    .powf_scalar(2.0)
                    .mean(),
            );
            let den = scalar_of(&x.clone().powf_scalar(2.0).mean()) + 1e-8;
            x = x_new;
            if (num.sqrt() / den.sqrt()) < config.conv_tol as f32 {
                calm += 1;
                if calm >= config.conv_patience {
                    steps_used = s;
                    break;
                }
            } else {
                calm = 0;
            }
        }
        let logits = self.logits(x);
        LoopOutput {
            logits,
            ponder: Tensor::from_floats([steps_used as f32], &device),
            steps_used,
            mean_halt: steps_used as f32,
        }
    }

    /// 2D hidden-matrix ids routed to Muon/Newton-Muon. Everything else
    /// (embedding, norms, halt head, LM head) goes to AdamW.
    pub fn muon_ids(&self) -> HashSet<ParamId> {
        [
            self.block.attn.q.weight.id,
            self.block.attn.k.weight.id,
            self.block.attn.v.weight.id,
            self.block.attn.o.weight.id,
            self.block.mlp.gate.weight.id,
            self.block.mlp.up.weight.id,
            self.block.mlp.down.weight.id,
        ]
        .into_iter()
        .collect()
    }

    /// Every float param with name and rank: the canonical id set for grad
    /// partitioning, accumulation merging, and coverage tests.
    pub fn grad_specs(&self) -> Vec<(&'static str, ParamId, usize)> {
        vec![
            ("embed", self.embed.weight.id, 2),
            ("q", self.block.attn.q.weight.id, 2),
            ("k", self.block.attn.k.weight.id, 2),
            ("v", self.block.attn.v.weight.id, 2),
            ("o", self.block.attn.o.weight.id, 2),
            ("gate", self.block.mlp.gate.weight.id, 2),
            ("up", self.block.mlp.up.weight.id, 2),
            ("down", self.block.mlp.down.weight.id, 2),
            ("norm1", self.block.norm1.gamma.id, 1),
            ("norm2", self.block.norm2.gamma.id, 1),
            ("norm_f", self.norm_f.gamma.id, 1),
            ("halt_w", self.halt.head.weight.id, 2),
            ("halt_b", self.halt.head.bias.as_ref().unwrap().id, 1),
            ("head", self.head.weight.id, 2),
        ]
    }

    /// Newton-Muon input group per hidden matrix.
    pub fn precond_roles(&self) -> HashMap<ParamId, PrecondInput> {
        [
            (self.block.attn.q.weight.id, PrecondInput::AttnIn),
            (self.block.attn.k.weight.id, PrecondInput::AttnIn),
            (self.block.attn.v.weight.id, PrecondInput::AttnIn),
            (self.block.attn.o.weight.id, PrecondInput::AttnIn),
            (self.block.mlp.gate.weight.id, PrecondInput::MlpIn),
            (self.block.mlp.up.weight.id, PrecondInput::MlpIn),
            (self.block.mlp.down.weight.id, PrecondInput::MlpHidden),
        ]
        .into_iter()
        .collect()
    }
}

impl<B: AutodiffBackend> LoopedTransformer<B> {
    /// Training path: ACT forward plus detached inner-backend input stats for
    /// the Newton-Muon preconditioner.
    pub fn forward_act_with_stats(
        &self,
        tokens: Tensor<B, 2, Int>,
        config: &LoopedConfig,
        lengths: &[usize],
    ) -> (LoopOutput<B>, StepStats<B::InnerBackend>) {
        let (out, stats) = self.forward_act(tokens, config, true, lengths);
        let s = stats.expect("collect_stats=true must return stats");
        (
            out,
            StepStats {
                attn_xtx: s.attn_xtx.inner(),
                attn_n: s.attn_n,
                mlp_xtx: s.mlp_xtx.inner(),
                mlp_n: s.mlp_n,
                down_xtx: s.down_xtx.inner(),
                down_n: s.down_n,
            },
        )
    }
}

/// Key-padding mask `[B, T]` (`true` = pad, blocked from attention).
/// Bytes 0x00 (PAD) and 0x01 (EOS) are reserved; task alphabets never
/// contain them, so pad positions are unambiguous.
/// `lens.lower_equal(pos)` is true exactly on pads (`len <= pos`); position
/// 0 stays open so a fully-padded row never softmaxes over an empty set
/// (NaN would poison shared-weight gradients).
pub fn pad_mask<B: Backend>(lengths: &[usize], t: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
    let b = lengths.len();
    let pos = Tensor::<B, 1, Int>::arange(0..t as i64, device)
        .unsqueeze_dim::<2>(0)
        .repeat_dim(0, b);
    let data: Vec<i64> = lengths.iter().map(|l| *l as i64).collect();
    let lens = Tensor::<B, 1, Int>::from_data(TensorData::new(data, [b]), device)
        .unsqueeze_dim::<2>(1)
        .repeat_dim(1, t);
    let first = Tensor::<B, 1, Int>::arange(0..t as i64, device)
        .lower_equal_elem(0)
        .unsqueeze_dim::<2>(0)
        .repeat_dim(0, b);
    lens.lower_equal(pos).bool_and(first.bool_not())
}

/// Causal LM loss with ponder penalty: `CE + ponder_weight * ponder`.
/// `tok_mask` (1.0 = scored position) masks prompt bytes and pads, so only
/// target bytes train the LM head.
pub fn lm_loss<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    tok_mask: Tensor<B, 2>,
    ponder: Tensor<B, 1>,
    ponder_weight: f64,
) -> Tensor<B, 1> {
    let [b, t, v] = logits.dims();
    let n = b * t;
    let logp = burn::tensor::activation::log_softmax(logits.reshape([n, v]), 1);
    let nll = logp
        .gather(1, targets.reshape([n, 1]))
        .reshape([n])
        .mul_scalar(-1.0);
    let m = tok_mask.reshape([b * t]);
    let ce = (nll * m.clone()).sum().div(m.sum().clamp_min(1.0));
    ce + ponder.mul_scalar(ponder_weight)
}

/// Split CE into answer-byte vs EOS-byte means (diagnostic only: aggregate
/// CE hides the split — EOS slots learn in minutes, answer slots carry the
/// actual task signal). Returns `(ce_answer, ce_eos)` as host floats.
pub fn ce_split<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    tok_mask: Tensor<B, 2>,
) -> (f32, f32) {
    let [b, t, v] = logits.dims();
    let n = b * t;
    let logp = burn::tensor::activation::log_softmax(logits.reshape([n, v]), 1);
    let nll = logp
        .gather(1, targets.clone().reshape([n, 1]))
        .reshape([n])
        .mul_scalar(-1.0);
    let m = tok_mask.reshape([n]);
    let is_eos = targets
        .reshape([n])
        .equal_elem(crate::harness::EOS as i64)
        .float();
    let w_eos = m.clone() * is_eos;
    let w_ans = m - w_eos.clone();
    let ce_ans = (nll.clone() * w_ans.clone())
        .sum()
        .div(w_ans.sum().clamp_min(1.0));
    let ce_eos = (nll * w_eos.clone()).sum().div(w_eos.sum().clamp_min(1.0));
    (scalar_of(&ce_ans), scalar_of(&ce_eos))
}

fn scalar_of<B: Backend>(t: &Tensor<B, 1>) -> f32 {
    t.clone().into_data().as_slice::<f32>().unwrap()[0]
}

#[cfg(test)]
mod loss_tests {
    use super::super::*;
    use crate::test_backend::{TestBackend, test_device};
    use burn::nn::LinearConfig;
    use burn::optim::GradientsParams;
    use burn::tensor::{Int, Tensor, TensorData};

    fn grad_norm_of_head(
        b: usize,
        t: usize,
        v: usize,
        d: usize,
        n_scored_per_row: usize,
    ) -> f32 {
        let device = test_device();
        let lin = LinearConfig::new(d, v)
            .with_bias(false)
            .init::<TestBackend>(&device);
        let h = Tensor::<TestBackend, 3>::ones([b, t, d], &device);
        let logits = lin.forward(h);
        let mut tgt = vec![0i64; b * t];
        let mut msk = vec![0.0f32; b * t];
        for row in 0..b {
            for j in 0..n_scored_per_row {
                let pos = row * t + (t - 1 - j);
                tgt[pos] = (row + j) as i64 % v as i64;
                msk[pos] = 1.0;
            }
        }
        let targets =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(tgt, [b, t]), &device);
        let mask =
            Tensor::<TestBackend, 2>::from_data(TensorData::new(msk, [b, t]), &device);
        let ponder = Tensor::<TestBackend, 1>::zeros([1], &device);
        let loss = lm_loss(logits, targets, mask, ponder, 0.0);
        let mut gp = GradientsParams::from_grads(loss.backward(), &lin);
        let g = gp.remove::<burn::backend::NdArray, 2>(lin.weight.id).unwrap();
        g.abs().sum().into_data().as_slice::<f32>().unwrap()[0]
    }

    #[test]
    fn masked_ce_gives_nonzero_grads() {
        let device = test_device();
        let lin = LinearConfig::new(4, 4)
            .with_bias(false)
            .init::<TestBackend>(&device);
        // [B=2, T=3, D=4] constant input; all learning must come from CE.
        let h = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        let logits = lin.forward(h);
        let targets = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[2i64, 1, 0], [3, 0, 0]]),
            &device,
        );
        let mask = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0f32, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            &device,
        );
        let ponder = Tensor::<TestBackend, 1>::zeros([1], &device);
        let loss = lm_loss(logits, targets, mask, ponder, 0.0);
        let lv: f32 = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
        assert!(lv > 0.5 && lv < 3.0, "unexpected loss value {lv}");
        let mut gp = GradientsParams::from_grads(loss.backward(), &lin);
        assert_eq!(gp.len(), 1);
        let g = gp
            .remove::<burn::backend::NdArray, 2>(lin.weight.id)
            .unwrap();
        let n: f32 = g.abs().sum().into_data().as_slice::<f32>().unwrap()[0];
        assert!(n > 0.0, "lm_loss CE grad is zero in isolation");
    }

    #[test]
    fn masked_ce_scales_to_combined_sizes() {
        // Mirror the combined setting: B=4, T~100, V=256, ~2 scored/row.
        let n = grad_norm_of_head(4, 100, 256, 256, 2);
        assert!(n > 0.0, "lm_loss CE grad is zero at combined sizes");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LoopedConfig;
    use crate::test_backend::{TestBackend, test_device};
    use burn::module::Module;

    fn tiny_config() -> LoopedConfig {
        LoopedConfig::new()
            .with_vocab_size(256)
            .with_d_model(32)
            .with_n_heads(2)
            .with_head_dim(16)
            .with_ffn_hidden(64)
            .with_max_loops(4)
            .with_max_seq_len(16)
    }

    fn tokens() -> Tensor<TestBackend, 2, Int> {
        Tensor::zeros([2, 4], &test_device())
    }

    fn lengths() -> Vec<usize> {
        vec![4, 4]
    }

    fn full_mask() -> Tensor<TestBackend, 2> {
        Tensor::ones([2, 4], &test_device())
    }

    #[test]
    fn fixed_forward_shapes() {
        let cfg = tiny_config();
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &test_device());
        let out = model.forward_fixed(tokens(), &lengths(), 2);
        assert_eq!(out.logits.dims(), [2, 4, 256]);
        assert_eq!(out.steps_used, 2);
    }

    #[test]
    fn act_forward_ponder_bounded() {
        let cfg = tiny_config();
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &test_device());
        let (out, stats) = model.forward_act(tokens(), &cfg, true, &lengths());
        assert_eq!(out.logits.dims(), [2, 4, 256]);
        assert!(out.steps_used <= 4);
        let p = scalar_of(&out.ponder);
        assert!((0.0..=5.0).contains(&p), "ponder {p} out of range");
        let s = stats.unwrap();
        assert!(s.attn_n > 0 && s.down_n > 0);
        // graph is trainable
        let loss = lm_loss(out.logits, tokens(), full_mask(), out.ponder, cfg.ponder_weight);
        let _grads = loss.backward();
    }

    #[test]
    fn converge_terminates() {
        let cfg = tiny_config();
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &test_device());
        let out = model.forward_converge(tokens(), &cfg, &lengths());
        assert_eq!(out.logits.dims(), [2, 4, 256]);
        assert!(out.steps_used <= 4);
    }

    #[test]
    fn padding_masks_pad_rows() {
        let cfg = tiny_config();
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &test_device());
        let toks: Tensor<TestBackend, 2, Int> = Tensor::zeros([2, 4], &test_device());
        let full = model.forward_fixed(toks.clone(), &[4, 4], 2);
        let padded = model.forward_fixed(toks, &[4, 0], 2);
        let a = full.logits.into_data().as_slice::<f32>().unwrap().to_vec();
        let b = padded.logits.into_data().as_slice::<f32>().unwrap().to_vec();
        // First row identical, second row differs (pad blocked).
        assert_eq!(&a[..1024], &b[..1024]);
        assert_ne!(&a[1024..], &b[1024..]);
    }

    #[test]
    fn pad_mask_values_are_correct() {
        // true = pad (blocked). Row len 0 keeps position 0 open (NaN guard).
        let device = test_device();
        let m = pad_mask::<TestBackend>(&[4, 0, 2], 4, &device);
        let v = m.float().into_data().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(
            v,
            vec![
                0., 0., 0., 0., //
                0., 1., 1., 1., //
                0., 0., 1., 1.,
            ]
        );
    }

    #[test]
    fn muon_ids_cover_seven_matrices() {
        let cfg = tiny_config();
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &test_device());
        assert_eq!(model.muon_ids().len(), 7);
        assert_eq!(model.precond_roles().len(), 7);
    }

    #[test]
    fn full_size_param_count_matches_budget() {
        let cfg = LoopedConfig::base_1m();
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &test_device());
        assert_eq!(model.num_params(), 984_065);
        assert_eq!(model.num_params(), cfg.param_count());
    }
}
