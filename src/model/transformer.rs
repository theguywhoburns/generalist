use std::collections::HashSet;

use burn::{
    module::{Module, ParamId},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};

use super::{
    block::TransformerBlock,
    config::{LoopedConfig, StopMode},
    halting::HaltingHead,
    masking::pad_mask,
};

/// Semantic parameter class. The optimizer routes on (rank, kind) — never
/// on name strings — so new modules classify themselves once in
/// [`LoopedTransformer::param_specs`] and routing follows automatically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// Token embedding (2D but AdamW per the Muon paper recipe).
    Embedding,
    /// Attention Q/K/V/O matrices (Muon).
    Attention,
    /// MLP gate/up matrices (Muon).
    MlpIn,
    /// MLP down matrix (Muon).
    MlpDown,
    /// Normalization scales (AdamW).
    Norm,
    /// Halting-gate weights (AdamW: thin rows, recipe exclusion).
    Halt,
    /// Untied LM head (2D but AdamW per recipe).
    Output,
}

impl ParamKind {
    /// Muon-routed classes: 2D hidden matrices with curvature worth
    /// orthogonalizing. Everything else rides AdamW.
    pub fn muon_routed(&self) -> bool {
        matches!(
            self,
            ParamKind::Attention | ParamKind::MlpIn | ParamKind::MlpDown
        )
    }
}

/// One parameter's identity: autodiff id, tensor rank, semantic class, and
/// a display label (`q0`, `halt_w3`, ...) used only for logs and tests.
#[derive(Debug, Clone)]
pub struct ParamSpec {
    pub id: ParamId,
    pub rank: usize,
    pub kind: ParamKind,
    pub label: String,
}

impl ParamSpec {
    pub fn new(id: ParamId, rank: usize, kind: ParamKind, label: String) -> Self {
        Self {
            id,
            rank,
            kind,
            label,
        }
    }
}

/// Graves ACT halting threshold: a token halts once cumulative halt mass
/// reaches `1 - ACT_EPS`.
const ACT_EPS: f64 = 0.01;

pub struct LoopOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    /// Mean Graves ponder `mean(N_t + R_t)` over tokens (in-graph for Act).
    pub ponder: Tensor<B, 1>,
    /// Block applications actually executed (summed over blocks).
    pub steps_used: usize,
    /// Host-side mean halt step, for logging.
    pub mean_halt: f32,
    /// Host-side mean halt steps per block: where the compute happened.
    /// `mean_halt` is the sum over blocks of these entries.
    pub block_halts: Vec<f32>,
}

/// One looped stage: a stack of `blocks_per_stage` encoder layers sharing
/// a single halting gate. The stage iterates its whole stack to the gate's
/// fixed point, then hands its readout to the next stage.
/// `blocks_per_stage = 1` is per-block ACT (current); one stage holding all
/// blocks is global ACT over the stack (the ablation axis).
#[derive(Module, Debug)]
pub struct LoopedStage<B: Backend> {
    pub blocks: Vec<TransformerBlock<B>>,
    pub halt: HaltingHead<B>,
}

impl<B: Backend> LoopedStage<B> {
    pub fn new(config: &LoopedConfig, device: &B::Device) -> Self {
        assert!(
            config.blocks_per_stage >= 1,
            "blocks_per_stage must be >= 1"
        );
        Self {
            blocks: (0..config.blocks_per_stage)
                .map(|_| TransformerBlock::new(config, device))
                .collect(),
            halt: HaltingHead::new(config.d_model, config.halt_bias_init, device),
        }
    }

    /// One full pass through this stage's stack.
    fn iterate(&self, x: Tensor<B, 3>, key_pad: &Tensor<B, 2, burn::tensor::Bool>) -> Tensor<B, 3> {
        let mut x = x;
        for block in &self.blocks {
            x = block.forward_masked(x, Some(key_pad.clone()));
        }
        x
    }
}

/// The 1M looped character transformer (Option A, untied head).
#[derive(Module, Debug)]
pub struct LoopedTransformer<B: Backend> {
    pub embed: Embedding<B>,
    /// Sequential looped stages; each iterates to its own gate's fixed
    /// point before handing its readout on (learned asynchronous depth).
    pub stages: Vec<LoopedStage<B>>,
    pub norm_f: RmsNorm<B>,
    pub head: Linear<B>,
}

impl<B: Backend> LoopedTransformer<B> {
    pub fn new(config: &LoopedConfig, device: &B::Device) -> Self {
        config.assert_valid();
        assert!(config.n_stages >= 1, "n_stages must be >= 1");
        Self {
            embed: EmbeddingConfig::new(config.vocab_size, config.d_model)
                .with_initializer(burn::module::Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                })
                .init(device),
            stages: (0..config.n_stages)
                .map(|_| LoopedStage::new(config, device))
                .collect(),
            norm_f: RmsNormConfig::new(config.d_model).init(device),
            head: LinearConfig::new(config.d_model, config.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }

    /// One full pass through all stages (one loop iteration).
    fn iterate(&self, x: Tensor<B, 3>, key_pad: &Tensor<B, 2, burn::tensor::Bool>) -> Tensor<B, 3> {
        let mut x = x;
        for stage in &self.stages {
            x = stage.iterate(x, key_pad);
        }
        x
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
        order: Option<&[usize]>,
    ) -> LoopOutput<B> {
        match mode {
            StopMode::Fixed { loops } => self.forward_fixed(tokens, lengths, loops),
            StopMode::Act => self.forward_act(tokens, config, lengths, order),
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
            x = self.iterate(x, &key_pad);
        }
        let logits = self.logits(x);
        let n = self.stages.len();
        LoopOutput {
            logits,
            ponder: Tensor::zeros([1], &device),
            steps_used: loops,
            mean_halt: loops as f32,
            block_halts: vec![loops as f32; n],
        }
    }

    /// Graves-style ACT loop with per-token halting, copy-through freezing of
    /// halted states, and remainder-weighted trajectory readout.
    /// `order` permutes stage execution (eval-only role diagnostic; `None` =
    /// trained order). `block_halts[i]` always refers to stage `i`, wherever
    /// it ran.
    pub fn forward_act(
        &self,
        tokens: Tensor<B, 2, Int>,
        config: &LoopedConfig,
        lengths: &[usize],
        order: Option<&[usize]>,
    ) -> LoopOutput<B> {
        let device = tokens.device();
        let [b, t] = tokens.dims();
        let d = config.d_model;
        let key_pad = pad_mask(lengths, t, &device);
        let keep_f = key_pad.clone().bool_not().float();

        let mut x = self.embed.forward(tokens);

        let max = config.max_loops;
        let n = self.stages.len();
        // Execution order: trained order by default; a permutation for the
        // role diagnostic. Validated: same length, in-range indices.
        let exec: Vec<usize> = match order {
            None => (0..n).collect(),
            Some(o) => {
                assert_eq!(o.len(), n, "stage order length {} != n_stages {n}", o.len());
                assert!(o.iter().all(|&i| i < n), "stage order index out of range");
                o.to_vec()
            }
        };
        // Loop-invariant constants hoisted: reusing them across iterations
        // avoids re-allocating identical tensors per step (backend buffers
        // pool-reuse anyway; this kills the launches too).
        let zeros_bt = Tensor::<B, 2>::zeros([b, t], &device);
        let ones_bt = Tensor::<B, 2>::ones([b, t], &device);
        // Learned asynchronous depth: stages run SEQUENTIALLY, each iterating
        // its whole block stack to its gate's fixed point before handing its
        // readout to the next stage. Every token passes through every stage;
        // each stage decides its own per-token iteration count.
        // For one single-block stage this is exactly classic single-gate ACT.
        // Means over real tokens only; pads never ran.
        let denom = keep_f.clone().sum().clamp_min(1.0);
        let mut total_ponder = Tensor::<B, 2>::zeros([b, t], &device);
        let mut total_halt = Tensor::<B, 2>::zeros([b, t], &device);
        let mut steps_used = 0usize;
        let mut block_halts = vec![0.0f32; n];
        for &si in &exec {
            let stage = &self.stages[si];
            let mut out = Tensor::zeros([b, t, d], &device);
            let mut cum = Tensor::zeros([b, t], &device);
            let mut ponder = Tensor::zeros([b, t], &device);
            let mut halt_step = Tensor::zeros([b, t], &device);
            let mut used = max;
            for s in 1..=max {
                // Running = unhalted AND real (pad rows never run, so the
                // loop can still early-exit on padded batches).
                let still: Tensor<B, 2, burn::tensor::Bool> = cum
                    .clone()
                    .lower_equal_elem(1.0 - ACT_EPS)
                    .bool_and(key_pad.clone().bool_not());
                let still_f = still.clone().float();

                // Single pass through the stage stack: this IS the forward.
                let mut xs = x.clone();
                for block in &stage.blocks {
                    xs = block.forward_masked(xs, Some(key_pad.clone()));
                }
                let x_new_full = xs;
                // Freeze halted states so running tokens attend to stable keys/values.
                // Single select op; replaces ones/sub/mul/mul/add with identical math.
                let still3 = still.clone().unsqueeze_dim::<3>(2).repeat_dim(2, d);
                x = x.mask_where(still3, x_new_full);

                let p = stage.halt.probs(x.clone());
                let p_run = zeros_bt.clone().mask_where(still.clone(), p);

                if s == max {
                    // Force-halt everything still running: remainder weight.
                    // Maxed-out tokens pay the full ponder price.
                    let rem = zeros_bt
                        .clone()
                        .mask_where(still.clone(), ones_bt.clone() - cum.clone());
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

                // Break check every 2nd loop (+final): the host sync stalls
                // the pipeline, and the break only skips halted tail steps.
                // Ponder/stats accounting is unaffected.
                let still: Tensor<B, 2, burn::tensor::Bool> = cum
                    .clone()
                    .lower_equal_elem(1.0 - ACT_EPS)
                    .bool_and(key_pad.clone().bool_not());
                let running: Tensor<B, 1> = (still.float() * keep_f.clone()).sum();
                if (s == max || s % 2 == 0) && scalar_of(&running) == 0.0 {
                    used = s;
                    break;
                }
            }
            // This stage's readout seeds the next stage (in execution order).
            // `block_halts[si]` always refers to stage `si` (`bh[i]` = Bi).
            block_halts[si] = scalar_of(
                &(halt_step.clone() * keep_f.clone())
                    .sum()
                    .div(denom.clone()),
            );
            total_ponder = total_ponder + ponder;
            total_halt = total_halt + halt_step;
            steps_used += used;
            x = out;
        }

        let ponder_mean = (total_ponder * keep_f.clone()).sum().div(denom.clone());
        let mean_halt = scalar_of(&(total_halt * keep_f).sum().div(denom));
        let logits = self.logits(x);
        LoopOutput {
            logits,
            ponder: ponder_mean,
            steps_used,
            mean_halt,
            block_halts,
        }
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
            let x_new = self.iterate(x.clone(), &key_pad);
            let num = scalar_of(&(x_new.clone() - x.clone()).powf_scalar(2.0).mean());
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
            block_halts: vec![steps_used as f32; self.stages.len()],
        }
    }

    /// 2D hidden-matrix ids routed to Muon, derived from
    /// [`Self::param_specs`] by semantic class — not by name strings and not
    /// by enumeration — so new modules with classified 2D matrices need zero
    /// optimizer changes. Muon paper recipe: embedding, LM head, norms,
    /// and halt gates stay on AdamW.
    pub fn muon_ids(&self) -> HashSet<ParamId> {
        self.param_specs()
            .into_iter()
            .filter(|s| s.rank == 2 && s.kind.muon_routed())
            .map(|s| s.id)
            .collect()
    }

    /// Every float param with name and rank: the canonical id set for grad
    /// partitioning, accumulation merging, and coverage tests.
    /// Blocks are numbered flat across stages (`q0`, `q1`, ...); stage gates
    /// are `halt_w{s}`/`halt_b{s}` per stage. Labels are display-only; the
    /// optimizer routes on [`ParamSpec::kind`], never on these strings.
    pub fn grad_specs(&self) -> Vec<(String, ParamId, usize)> {
        self.param_specs()
            .into_iter()
            .map(|s| (s.label, s.id, s.rank))
            .collect()
    }

    /// Canonical parameter inventory: identity, rank, and semantic class.
    /// Built once here; routing, merging, and tests all derive from it.
    pub fn param_specs(&self) -> Vec<ParamSpec> {
        use ParamKind::*;
        let mut specs = vec![ParamSpec::new(
            self.embed.weight.id,
            2,
            Embedding,
            "embed".to_string(),
        )];
        let mut i = 0usize;
        for (s, stage) in self.stages.iter().enumerate() {
            for b in &stage.blocks {
                let attn = [
                    (format!("q{i}"), b.attn.q.weight.id),
                    (format!("k{i}"), b.attn.k.weight.id),
                    (format!("v{i}"), b.attn.v.weight.id),
                    (format!("o{i}"), b.attn.o.weight.id),
                ];
                for (label, id) in attn {
                    specs.push(ParamSpec::new(id, 2, Attention, label));
                }
                let mlp_in = [
                    (format!("gate{i}"), b.mlp.gate.weight.id),
                    (format!("up{i}"), b.mlp.up.weight.id),
                ];
                for (label, id) in mlp_in {
                    specs.push(ParamSpec::new(id, 2, MlpIn, label));
                }
                specs.push(ParamSpec::new(
                    b.mlp.down.weight.id,
                    2,
                    MlpDown,
                    format!("down{i}"),
                ));
                specs.push(ParamSpec::new(
                    b.norm1.gamma.id,
                    1,
                    Norm,
                    format!("norm1_{i}"),
                ));
                specs.push(ParamSpec::new(
                    b.norm2.gamma.id,
                    1,
                    Norm,
                    format!("norm2_{i}"),
                ));
                i += 1;
            }
            specs.push(ParamSpec::new(
                stage.halt.head.weight.id,
                2,
                Halt,
                format!("halt_w{s}"),
            ));
            specs.push(ParamSpec::new(
                stage.halt.head.bias.as_ref().unwrap().id,
                1,
                Halt,
                format!("halt_b{s}"),
            ));
        }
        specs.push(ParamSpec::new(
            self.norm_f.gamma.id,
            1,
            Norm,
            "norm_f".to_string(),
        ));
        specs.push(ParamSpec::new(
            self.head.weight.id,
            2,
            Output,
            "head".to_string(),
        ));
        specs
    }
}

/// Scored-position window `[B, Ts, V]` of a `[B, T, V]` logits tensor, plus
/// matching target/mask columns, where `Ts = hi - lo` spans the earliest
/// first-scored to the latest last-scored position across rows.
///
/// Training CE then runs on `[B, Ts, V]` instead of `[B, T, V]`: with
/// `vocab == d_model` the logits and their CE intermediates are the largest
/// training activations (~60% of the tape at T=512), and byte-LM masks score
/// only the target tail, so Ts is typically a small fraction of T. Bounds
/// come from one host readback of the B×T mask (cheap next to the GEMM it
/// avoids); gradients flow through the `slice` op to the full logits.
/// # Panics
/// Panics if the mask scores no positions (nothing to train on).
fn scored_slice<B: Backend>(
    logits: Tensor<B, 3>,
    targets: &Tensor<B, 2, Int>,
    mask: &Tensor<B, 2>,
) -> (Tensor<B, 3>, Tensor<B, 2, Int>, Tensor<B, 2>) {
    let [b, _t, v] = logits.dims();
    let (lo, hi) = scored_bounds(&mask.clone().into_data());
    let logits = logits.slice([0..b, lo..hi, 0..v]);
    let targets = targets.clone().slice([0..b, lo..hi]);
    let mask = mask.clone().slice([0..b, lo..hi]);
    (logits, targets, mask)
}

/// Host scan of a `[B, T]` mask for (min first-scored, max last-scored).
fn scored_bounds(data: &TensorData) -> (usize, usize) {
    let slice = data.as_slice::<f32>().unwrap();
    let [b, t] = [data.shape[0], data.shape[1]];
    let (mut lo, mut hi) = (t, 0);
    for row in slice.chunks_exact(t).take(b) {
        for (c, w) in row.iter().enumerate() {
            if *w > 0.0 {
                lo = lo.min(c);
                hi = hi.max(c + 1);
            }
        }
    }
    assert!(hi > lo, "loss mask scores no positions");
    (lo, hi)
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
    let v = logits.dims()[2];
    let (logits, targets, tok_mask) = scored_slice(logits, &targets, &tok_mask);
    let [b, ts, _] = logits.dims();
    let n = b * ts;
    let logp = burn::tensor::activation::log_softmax(logits.reshape([n, v]), 1);
    let nll = logp
        .gather(1, targets.reshape([n, 1]))
        .reshape([n])
        .mul_scalar(-1.0);
    let m = tok_mask.reshape([b * ts]);
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
    let v = logits.dims()[2];
    let (logits, targets, tok_mask) = scored_slice(logits, &targets, &tok_mask);
    let [b, ts, _] = logits.dims();
    let n = b * ts;
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
    use burn::tensor::backend::AutodiffBackend;
    use burn::tensor::{Int, Tensor, TensorData};

    type IB = <TestBackend as AutodiffBackend>::InnerBackend;

    fn grad_norm_of_head(b: usize, t: usize, v: usize, d: usize, n_scored_per_row: usize) -> f32 {
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
        let mask = Tensor::<TestBackend, 2>::from_data(TensorData::new(msk, [b, t]), &device);
        let ponder = Tensor::<TestBackend, 1>::zeros([1], &device);
        let loss = lm_loss(logits, targets, mask, ponder, 0.0);
        let mut gp = GradientsParams::from_grads(loss.backward(), &lin);
        let g = gp.remove::<IB, 2>(lin.weight.id).unwrap();
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
        let g = gp.remove::<IB, 2>(lin.weight.id).unwrap();
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
        let out = model.forward_act(tokens(), &cfg, &lengths(), None);
        assert_eq!(out.logits.dims(), [2, 4, 256]);
        assert!(out.steps_used <= 4);
        let p = scalar_of(&out.ponder);
        assert!((0.0..=5.0).contains(&p), "ponder {p} out of range");
        // graph is trainable
        let loss = lm_loss(
            out.logits,
            tokens(),
            full_mask(),
            out.ponder,
            cfg.ponder_weight,
        );
        let _grads = loss.backward();
    }

    #[test]
    fn per_stage_gates_are_independent() {
        let cfg = LoopedConfig::base_1m().with_n_stages(2);
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &test_device());
        assert_ne!(
            model.stages[0].halt.head.weight.id,
            model.stages[1].halt.head.weight.id
        );
        let out = model.forward_act(tokens(), &cfg, &lengths(), None);
        assert_eq!(out.block_halts.len(), 2);
        // Total halt is the sum over blocks (where-compute accounting).
        let sum: f32 = out.block_halts.iter().sum();
        assert!(
            (sum - out.mean_halt).abs() < 1e-3,
            "sum {sum} vs total {}",
            out.mean_halt
        );
        assert!(out.steps_used <= 16);
    }

    #[test]
    fn identity_order_matches_default_bitwise() {
        let cfg = LoopedConfig::base_1m().with_n_stages(2);
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &test_device());
        let a = model.forward_act(tokens(), &cfg, &lengths(), None);
        let b = model.forward_act(tokens(), &cfg, &lengths(), Some(&[0, 1]));
        let va = a.logits.into_data().as_slice::<f32>().unwrap().to_vec();
        let vb = b.logits.into_data().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(va, vb);
        assert_eq!(a.block_halts, b.block_halts);
    }

    #[test]
    fn reversed_order_runs_with_block_indexed_halts() {
        let cfg = LoopedConfig::base_1m().with_n_stages(2);
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &test_device());
        let out = model.forward_act(tokens(), &cfg, &lengths(), Some(&[1, 0]));
        assert_eq!(out.logits.dims(), [2, 4, 256]);
        // bh[i] still refers to block i, wherever it executed.
        assert_eq!(out.block_halts.len(), 2);
        let sum: f32 = out.block_halts.iter().sum();
        assert!((sum - out.mean_halt).abs() < 1e-3);
    }

    #[test]
    #[should_panic(expected = "stage order length")]
    fn short_order_panics() {
        let cfg = LoopedConfig::base_1m().with_n_stages(2);
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &test_device());
        let _ = model.forward_act(tokens(), &cfg, &lengths(), Some(&[0]));
    }

    #[test]
    fn grad_specs_cover_per_block_gates() {
        let cfg = LoopedConfig::base_1m().with_n_stages(2);
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &test_device());
        let specs = model.grad_specs();
        // embed + norm_f + head + 11 per single-block stage
        assert_eq!(specs.len(), 3 + 11 * 2);
        assert!(specs.iter().any(|(n, _, _)| n == "halt_w0"));
        assert!(specs.iter().any(|(n, _, _)| n == "halt_b1"));
    }

    #[test]
    fn one_stage_two_blocks_shares_a_gate() {
        // Global-ACT layout: 2 blocks, 1 gate, 1 bh entry, same 14 Muon ids.
        let cfg = LoopedConfig::base_1m().with_blocks_per_stage(2);
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &test_device());
        assert_eq!(model.stages.len(), 1);
        assert_eq!(model.stages[0].blocks.len(), 2);
        assert_eq!(model.muon_ids().len(), 14);
        assert_eq!(model.grad_specs().len(), 3 + 9 * 2 + 2);
        let out = model.forward_act(tokens(), &cfg, &lengths(), None);
        assert_eq!(out.block_halts.len(), 1);
        assert_eq!(out.logits.dims(), [2, 4, 256]);
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
        // DISTINCT token values per position: with identical bytes any
        // attention pattern yields identical outputs, which would make this
        // test vacuous (it must discriminate masking, not values).
        let device = test_device();
        let cfg = tiny_config();
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &device);
        let row: Tensor<TestBackend, 2, Int> = Tensor::from_data(
            burn::tensor::TensorData::from([[5i64, 6, 7, 8], [5, 6, 7, 8]]),
            &device,
        );
        let full = model.forward_fixed(row.clone(), &[4, 4], 2);
        let padded = model.forward_fixed(row, &[4, 1], 2);
        let a = full.logits.into_data().as_slice::<f32>().unwrap().to_vec();
        let b = padded
            .logits
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        // Row 0 identical across runs (no cross-batch leakage, deterministic).
        assert_eq!(&a[..1024], &b[..1024]);
        // Row 1 differs: full causal context vs pos0-only keys.
        assert_ne!(&a[1024..], &b[1024..]);
    }

    #[test]
    fn outputs_invariant_to_pad_bucket() {
        // Same row evaluated at different bucketed T must give identical
        // outputs at shared positions (pads contribute exactly nothing).
        // Train batches hit T=512 while eval sees small T: any dependence
        // here is a train/eval skew.
        use burn::tensor::TensorData;
        let device = test_device();
        let cfg = LoopedConfig::new()
            .with_vocab_size(256)
            .with_d_model(32)
            .with_n_heads(2)
            .with_head_dim(16)
            .with_ffn_hidden(64)
            .with_max_loops(4)
            .with_max_seq_len(128);
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &device);
        let row: Vec<i64> = (0..20).map(|i| 5 + (i % 20)).collect();
        let t64: Vec<i64> = row
            .iter()
            .cloned()
            .chain(std::iter::repeat(0))
            .take(64)
            .collect();
        let t128: Vec<i64> = row
            .iter()
            .cloned()
            .chain(std::iter::repeat(0))
            .take(128)
            .collect();
        let a = Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(t64, [1, 64]), &device);
        let b = Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(t128, [1, 128]), &device);
        let oa = model.forward_fixed(a, &[20], 2);
        let ob = model.forward_fixed(b, &[20], 2);
        let va = oa
            .logits
            .slice([0..1, 0..20, 0..256])
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let vb = ob
            .logits
            .slice([0..1, 0..20, 0..256])
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        assert_eq!(va.len(), vb.len());
        for (x, y) in va.iter().zip(vb.iter()) {
            assert!((x - y).abs() < 1e-3, "T-bucket dependence: {x} vs {y}");
        }
    }

    #[test]
    fn slice_selects_exact_values() {
        // Value-level (not shape-level): scored_slice and decode argmax
        // both depend on multi-dim slice returning the right elements.
        use burn::tensor::TensorData;
        let device = test_device();
        // [2, 4, 8] with value = 100*b + 10*t + v.
        let flat: Vec<f32> = (0..2)
            .flat_map(|b| {
                (0..4).flat_map(move |t| (0..8).map(move |v| (100 * b + 10 * t + v) as f32))
            })
            .collect();
        let x = Tensor::<TestBackend, 3>::from_data(TensorData::new(flat, [2, 4, 8]), &device);
        let s = x.slice([0..2, 1..3, 2..5]);
        assert_eq!(s.dims(), [2, 2, 3]);
        let v = s.into_data().as_slice::<f32>().unwrap().to_vec();
        let mut expected = vec![];
        for b in 0..2 {
            for t in 1..3 {
                for vv in 2..5 {
                    expected.push((100 * b + 10 * t + vv) as f32);
                }
            }
        }
        assert_eq!(v, expected);
        // Single-row single-position slice as used in greedy decode.
        let x2 = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                (0..2 * 4 * 8).map(|i| i as f32).collect::<Vec<_>>(),
                [2, 4, 8],
            ),
            &device,
        );
        let one = x2.slice([1..2, 3..4, 0..8]).reshape([8]);
        let got = one.into_data().as_slice::<f32>().unwrap().to_vec();
        let exp: Vec<f32> = (0..8).map(|i| (32 + 24 + i) as f32).collect();
        assert_eq!(got, exp);
    }

    #[test]
    fn muon_ids_cover_seven_matrices() {
        // One block: q,k,v,o,gate,up,down. Nothing else is hidden-matrix class.
        let cfg = tiny_config();
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &test_device());
        assert_eq!(model.muon_ids().len(), 7);
    }

    #[test]
    fn muon_routing_is_rank_based_with_recipe_exclusions() {
        // 2 stages: 14 hidden matrices on Muon; embed/head/halt/norm classes
        // on AdamW. Verified by class, never by name strings.
        use super::ParamKind;
        let cfg = LoopedConfig::base_1m().with_n_stages(2);
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &test_device());
        let muon = model.muon_ids();
        assert_eq!(muon.len(), 14);
        for s in model.param_specs() {
            let routed = muon.contains(&s.id);
            assert_eq!(
                routed,
                s.rank == 2 && s.kind.muon_routed(),
                "misrouted {} (rank {}, kind {:?})",
                s.label,
                s.rank,
                s.kind
            );
        }
        // Spot-check the recipe side directly on classes: embedding and LM
        // head are 2D but stay on AdamW, as do norms and the halt gates.
        assert!(!ParamKind::Embedding.muon_routed());
        assert!(!ParamKind::Output.muon_routed());
        assert!(!ParamKind::Halt.muon_routed());
        assert!(!ParamKind::Norm.muon_routed());
        assert!(ParamKind::Attention.muon_routed() && ParamKind::MlpDown.muon_routed());
    }

    #[test]
    fn full_size_param_count_matches_budget() {
        let cfg = LoopedConfig::base_1m();
        let model = LoopedTransformer::<TestBackend>::new(&cfg, &test_device());
        assert_eq!(model.num_params(), 984_065);
        assert_eq!(model.num_params(), cfg.param_count());
    }
}
