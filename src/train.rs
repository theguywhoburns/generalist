//! Training loop: constant LR, ACT + hybrid Muon/AdamW, greedy eval with
//! EOS stopping, MPK checkpoints, JSONL metrics.
//!
//! One step: collate -> `forward_act` -> masked LM loss -> backward ->
//! split (hidden matrices / rest) -> two adaptor steps. Run orchestration
//! (seeds, pools) is the caller's job; see `Experiment` for data dispatch.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::{
    config::Config,
    module::{AutodiffModule, Module, ParamId},
    optim::{AdamW, AdamWConfig, GradientsParams, Muon, Optimizer, adaptor::OptimizerAdaptor},
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{
        Int, Tensor, TensorData,
        backend::{AutodiffBackend, Backend},
    },
};

use crate::{
    harness::Record as MetricRecord,
    model::{LoopedConfig, LoopedTransformer, StopConfig, StopMode, masked_ce},
    optim::{LrConfig, LrPair, OptimConfig, merge_grads, split_grads},
    tasks::{HarnessRng, Instance},
};

#[derive(Config, Debug)]
pub struct TrainConfig {
    /// Learning rate for the 2D hidden matrices (the optimizer named by
    /// `OptimConfig`). A schedule, not a scalar: a flat rate is
    /// `{"kind": "constant", "lr": ...}`, and a warmup or decay is the same
    /// key with a different `kind`. See `src/optim/lr.rs`.
    #[config(default = "LrConfig::Constant { lr: 5e-3 }")]
    pub lr_muon: LrConfig,
    /// Same, for the AdamW partition (embed, head, norms, halt gates).
    #[config(default = "LrConfig::Constant { lr: 3e-4 }")]
    pub lr_adamw: LrConfig,
    /// Micro-batch size. With `auto_batch` this is the *starting* size the
    /// tuner grows from, and a floor it will not go below.
    #[config(default = 32)]
    pub batch_size: usize,
    /// Grow `batch_size` to fill the GPU instead of leaving it at whatever a
    /// manifest happened to hardcode.
    ///
    /// The tuner measures a batch it has already survived and extrapolates
    /// before growing, because a CUDA OOM aborts the process rather than
    /// raising a catchable error. It holds the *effective* batch
    /// (`batch_size * accum_steps`) fixed while it moves, so gradient noise —
    /// and therefore every number in a sweep — stays comparable across
    /// machines.
    ///
    /// The realized `batch_size` and `accum_steps` are logged at the top of the
    /// run. Without that line an auto-tuned run is not reproducible, since the
    /// batch is a function of the card it ran on.
    #[config(default = false)]
    pub auto_batch: bool,
    /// Largest micro-batch the tuner may select. Bounds the search so one
    /// enormous allocation cannot be authorized on a card with room for it.
    #[config(default = 512)]
    pub auto_batch_max: usize,
    /// Fraction of total device memory activations may occupy. The remainder
    /// absorbs fragmentation and the eval pass, which is sized separately and
    /// would otherwise OOM after tuning concluded.
    #[config(default = 0.75)]
    pub auto_batch_headroom: f64,
    #[config(default = 1000)]
    pub steps: usize,
    #[config(default = 50)]
    pub log_every: usize,
    #[config(default = 200)]
    pub eval_every: usize,
    /// Save a checkpoint every N steps (independent of eval).
    #[config(default = 500)]
    pub ckpt_every: usize,
    #[config(default = 64)]
    pub eval_max_new: usize,
    #[config(default = 0)]
    pub seed: u64,
    /// Micro-batches per optimizer step (effective batch =
    /// `batch_size` x `accum_steps`). Each micro loss is scaled by
    /// 1/accum_steps before backward, so merged grads equal full-batch grads.
    #[config(default = 1)]
    pub accum_steps: usize,
    /// Whole-process stall watchdog: kill the run if no step completes
    /// within this many seconds (catches hangs no panic hook can see).
    #[config(default = 900)]
    pub stuck_timeout_secs: u64,
    /// First step at which eval runs; `(step - start) % eval_every` after.
    /// Set high (e.g. >= total steps) to isolate training memory from eval.
    #[config(default = 0)]
    pub eval_start_step: usize,
    /// Scheduled-sampling thresholds: free-running target inputs K =
    /// number of entries with answer-CE below them. Default [1.0]: once
    /// answer error dips below 1.0, the first target input per row comes
    /// from the model's own argmax; more entries grow K progressively.
    #[config(default = "vec![1.0]")]
    pub free_schedule: Vec<f64>,
    #[config(default = "\"checkpoints\".to_string()")]
    pub ckpt_dir: String,
    /// Ponder warmup: effective ponder weight ramps linearly 0 -> base
    /// over this many steps (0 = off/legacy: full weight from step 0).
    /// Applies to the ponder term only, never the CE term.
    #[config(default = 0)]
    pub ponder_warmup_steps: usize,
    /// Repeat the final eval with reversed stage order ("final-shuffled"):
    /// the role diagnostic. Eval-only; training never permutes.
    #[config(default = true)]
    pub shuffle_eval: bool,
    /// Fraction of each (task, track) cell held out of training and used for
    /// eval. Every reported accuracy is measured on instances the model never
    /// saw.
    ///
    /// `0.0` restores the old in-distribution behavior and is only for
    /// reproducing a pre-holdout number; the loader warns when it sees it,
    /// because such a number does not measure generalization.
    #[config(default = 0.1)]
    pub eval_holdout: f64,
    /// Also evaluate a sample of the TRAINING pool, logged separately as
    /// `*-indist`.
    ///
    /// This is the second half of the memorization measurement. Generalization
    /// alone cannot say where memorization begins, because held-out accuracy
    /// saturating is ambiguous between "solved the task" and "stopped
    /// benefiting". The train/held-out **divergence** is the signal, and it
    /// needs both numbers from the same run.
    ///
    /// Kept small on purpose: it is a diagnostic, and eval decode is the
    /// slowest part of a run.
    #[config(default = true)]
    pub eval_in_distribution: bool,
    /// Instances sampled from the training pool for that diagnostic.
    #[config(default = 48)]
    pub indist_eval_per_cell: usize,
    /// Cap on training instances, applied AFTER the holdout split. `0` = no
    /// cap.
    ///
    /// This is the data-scaling knob, deliberately separate from
    /// `experiment.per_cell`. Varying `per_cell` moves the training pool AND
    /// the eval set together, so held-out accuracy at each point is measured
    /// on a different split and the curve is not comparable across points.
    /// Capping after the holdout keeps the eval set fixed, which is what a
    /// learning curve needs: the only thing that changes along the axis is
    /// how much the model trained on.
    ///
    /// The cap takes a deterministic stride over the length-sorted pool, so
    /// the subset spans the length range rather than being all-shortest.
    #[config(default = 0)]
    pub max_train_instances: usize,
}

pub struct StepInfo {
    pub loss: f32,
    pub ponder: f32,
    pub steps_used: usize,
    pub mean_halt: f32,
    /// CE on answer bytes (the task signal) vs EOS bytes (stop signal).
    pub ce_answer: f32,
    pub ce_eos: f32,
}

/// Top-bucket row cap: T=512 micros carry B×T² attention tape (saved
/// softmax trios across all unrolled block-steps) that peaks past 4GB at
/// batch 6 regardless of pool packing — every observed death lands on these
/// micros. Cap them to 2 rows; mean-reduced CE keeps the grads unbiased
/// (same caveat as the B×T trim below).
pub fn top_bucket_rows(t_pad: usize, rows: usize) -> usize {
    if t_pad >= 512 { rows.clamp(1, 2) } else { rows }
}

/// Activation budget for one training micro-batch, as max `batch × padded_T`.
/// The banded sampler hops length bands (bucket edges 64..512); at a fixed
/// batch size the top band explodes the autodiff tape — the `[B,H,T,T]`
/// attention trio scales as B×T² and dominates logits (~2.4GB vs ~2MB at
/// batch 6/T=512 over 32 unrolled block-steps), which OOMs small GPUs
/// (RTX 3050 4GB) on the first full-depth T=512 micro.
/// Long-band windows are trimmed so the tape size stays band-independent.
/// Mean-reduced CE keeps merged micro-grads a mean of micro-means either way.
const MICRO_BT_BUDGET: usize = 2048;

/// Rows the banded window may hold before trimming, derived from a B×T budget
/// at the shortest band.
///
/// The budget — not `batch_size` — is the knob that decides how many rows
/// actually reach the GPU, because the trim caps every band. A window larger
/// than this simply supplies more candidate rows to trim from, which is
/// harmless: the trim drops the longest, so the survivors are the same rows it
/// would have kept from a smaller window.
///
/// Sized so the window always exceeds what the shortest band can use, making
/// the B×T budget the single binding constraint. Otherwise the window would
/// silently cap the short bands while the tuner believed the budget was doing
/// the limiting.
pub fn window_for_budget(budget: usize, batch_size: usize) -> usize {
    let shortest = crate::harness::batch::BUCKET_EDGES[0].max(1);
    (budget / shortest).max(1).max(batch_size)
}

/// Same budget for eval decode chunks (B×T per fused forward). Eval prompts
/// bucket to 512 while training bands sit near 64, so a fixed row count that
/// fits a training band OOMs on an eval chunk of long prompts.
const EVAL_BT_BUDGET: usize = 8192;

/// Effective ponder weight under linear warmup: ramps 0 -> `base` over
/// `warmup_steps` steps (`warmup_steps == 0` = off/legacy: full `base`).
/// Pure function (testable). Applies to the ponder term only, never CE.
pub fn ponder_warmup_weight(base: f64, step: usize, warmup_steps: usize) -> f64 {
    if warmup_steps == 0 {
        return base;
    }
    let frac = (step as f64 / warmup_steps as f64).clamp(0.0, 1.0);
    base * frac
}

pub struct Trainer<B: AutodiffBackend> {
    pub model: LoopedTransformer<B>,
    /// Optimizer for the 2D hidden matrices, selected by [`OptimConfig`].
    muon: OptimizerAdaptor<Muon<B::InnerBackend>, LoopedTransformer<B>, B>,
    /// Optimizer for everything else (embed, head, norms, halt gates).
    adamw: OptimizerAdaptor<AdamW, LoopedTransformer<B>, B>,
    muon_ids: HashSet<ParamId>,
    config: LoopedConfig,
    /// How the loop decides depth. Manifest-selected, so a fixed-depth or
    /// converge control needs no recompile.
    stop: StopMode,
    /// Advanced once per optimizer step, never per micro-batch.
    lrs: LrPair,
    /// The run's training seed, stamped onto every eval record so a multi-seed
    /// comparison is aggregatable from the log alone.
    pub seed: u64,
    /// Resample the stage execution order every training step. The causal
    /// control for the role diagnostic: see [`StopConfig::default()`].
    shuffle_train: bool,
    /// Stage order used by the most recent training forward, so a log line can
    /// show what was actually executed. `None` under the trained order.
    pub last_stage_order: Option<Vec<usize>>,
    /// Monotonic count of order draws, so each shuffle is distinct rather than
    /// re-deriving the same permutation from the seed.
    order_counter: u64,
    device: B::Device,
    /// Current scheduled-sampling depth (see `free_schedule`).
    pub free_k: usize,
}

impl<B: AutodiffBackend> Trainer<B> {
    /// # Panics
    /// Panics if a learning-rate schedule is invalid. Manifests are checked
    /// by [`crate::harness::RunConfig::validate`] before a `Trainer` exists,
    /// so reaching here means a programmatic caller skipped that.
    pub fn new(
        model_config: &LoopedConfig,
        optim: &OptimConfig,
        stop: StopConfig,
        train: &TrainConfig,
        device: &B::Device,
        init_from: Option<&Path>,
    ) -> Self {
        B::seed(device, train.seed);
        let mut model = LoopedTransformer::<B>::new(model_config, device);
        if let Some(path) = init_from {
            // Stage chaining: start from the previous stage's weights.
            // Optimizer states restart fresh (documented).
            model = Self::load_checkpoint(model_config, path, device);
        }
        // Optimizer selection: one match arm per `OptimConfig` variant.
        let muon = match optim {
            OptimConfig::Muon(tuning) => tuning.to_muon_config().init(),
        };
        let lrs = LrPair::new(&train.lr_muon, &train.lr_adamw)
            .unwrap_or_else(|(field, e)| panic!("invalid {field}: {e}"));
        Self {
            muon,
            adamw: AdamWConfig::new().init(),
            muon_ids: model.muon_ids(),
            model,
            config: model_config.clone(),
            stop: stop.to_mode(),
            lrs,
            seed: train.seed,
            shuffle_train: stop.shuffle_training(),
            last_stage_order: None,
            order_counter: 0,
            device: device.clone(),
            free_k: 0,
        }
    }

    /// A random permutation of the stage indices, for order-augmented
    /// training. `HarnessRng::shuffle` is a Fisher-Yates, so the result is a
    /// uniform permutation over `n!` orders and reproducible from the seed.
    ///
    /// Guarded against `n < 2`: with one stage the only permutation is the
    /// identity, so returning `None` keeps the trained-order code path
    /// identical rather than passing a meaningless no-op order.
    pub fn sample_stage_order(&mut self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.model.stages.len()).collect();
        if order.len() < 2 {
            return order;
        }
        let mut rng = HarnessRng::new(
            self.seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(self.order_counter)
                .wrapping_add(0x1234_5678_9ABC_DEF0),
        );
        self.order_counter += 1;
        rng.shuffle(&mut order);
        order
    }

    /// Advance the free-running schedule from mean answer-CE.
    pub fn update_free_schedule(&mut self, ce_answer: f32, schedule: &[f64]) {
        self.free_k = schedule.iter().filter(|t| (ce_answer as f64) < **t).count();
    }

    /// Override the effective ponder weight (warmup ramp). CE term untouched.
    pub fn set_ponder_weight(&mut self, w: f64) {
        self.config.ponder_weight = w;
    }

    /// Sample a batch with replacement from the pool (deterministic in rng).
    pub fn sample_batch<'a>(
        rng: &mut HarnessRng,
        pool: &'a [Instance],
        n: usize,
    ) -> Vec<&'a Instance> {
        (0..n).map(|_| &pool[rng.below(pool.len())]).collect()
    }

    /// Sample a contiguous window from a length-sorted pool. Rows share
    /// near-identical lengths, so padded T (and fused-kernel shapes) stay
    /// stable across steps: fewer recompiles, less pool fragmentation.
    pub fn sample_banded_batch<'a>(
        rng: &mut HarnessRng,
        sorted_pool: &'a [Instance],
        n: usize,
    ) -> Vec<&'a Instance> {
        assert!(!sorted_pool.is_empty());
        let n = n.min(sorted_pool.len());
        let start = rng.below(sorted_pool.len() - n + 1);
        sorted_pool[start..start + n].iter().collect()
    }

    /// Forward + loss + backward for one micro-batch. `scale` multiplies the
    /// loss before backward (1/accum_steps for mean-equivalent accumulation).
    /// Reports unscaled loss/ponder for logging.
    pub fn forward_backward(
        &mut self,
        batch: &[&Instance],
        scale: f64,
        free_k: usize,
    ) -> (StepInfo, GradientsParams) {
        let base_seqs: Vec<Vec<u8>> = batch
            .iter()
            .map(|inst| {
                let mut s = inst.prompt.clone();
                s.extend_from_slice(&inst.target);
                s.push(crate::harness::EOS);
                s
            })
            .collect();
        let prompt_lens: Vec<usize> = batch.iter().map(|inst| inst.prompt.len()).collect();
        // Scheduled sampling: proposal pass on the inner backend (no tape),
        // then the first K target INPUTS come from the model's own argmax.
        // Targets and mask ALWAYS come from unpatched truth: scoring patched
        // positions against patched targets is circular (self-agreement
        // drives loss to 0 regardless of correctness).
        let truth =
            crate::harness::collate_seqs(base_seqs.clone(), prompt_lens.clone(), &self.device);
        let tokens = if free_k == 0 {
            truth.tokens.clone()
        } else {
            let tokens_inner = truth.tokens.clone().inner();
            let lens = truth.lengths.clone();
            let valid = self.model.valid();
            let logits = valid
                .forward(tokens_inner, &self.config, self.stop, &lens, None)
                .logits;
            let [b2, t2, _] = logits.dims();
            let pred = int_vec(&logits.argmax(2).reshape([b2 * t2]).into_data());
            let patched = patch_free_inputs(&base_seqs, &prompt_lens, &pred, t2, free_k);
            crate::harness::collate_seqs(patched, prompt_lens, &self.device).tokens
        };
        let col = crate::harness::Collated {
            tokens,
            targets: truth.targets,
            loss_mask: truth.loss_mask,
            lengths: truth.lengths,
            prompt_lens: truth.prompt_lens,
        };
        let t = col.lengths.iter().max().copied().unwrap_or(0);
        assert!(
            t <= self.config.max_seq_len,
            "batch length {t} exceeds max_seq_len {}: raise max_seq_len or shorten instances",
            self.config.max_seq_len,
        );
        // Order-augmented training: when enabled, resample the stage
        // execution order for THIS forward. The model already takes the order
        // as an argument, so the intervention is entirely in the trainer —
        // no model path changes. `stage_order` is read back after the
        // forward so the executed order is observable in logs.
        let scratch: Option<Vec<usize>> = if self.shuffle_train {
            Some(self.sample_stage_order())
        } else {
            None
        };
        let out =
            self.model
                .forward_act(col.tokens, &self.config, &col.lengths, scratch.as_deref());
        self.last_stage_order = scratch;
        // One log_softmax and one mask readback serve the loss and the
        // answer/EOS split. `ce_answer` drives the free_schedule threshold, so
        // it is computed every step rather than gated behind log_every.
        let ce = masked_ce(out.logits, col.targets, col.loss_mask);
        let (ce_answer, ce_eos) = ce.split();
        let loss = ce.with_ponder(out.ponder.clone(), self.config.ponder_weight);
        let info = StepInfo {
            loss: scalar_of(&loss),
            ponder: scalar_of(&out.ponder),
            steps_used: out.steps_used,
            mean_halt: out.mean_halt,
            ce_answer,
            ce_eos,
        };
        let scaled = loss.mul_scalar(scale);
        let grads = scaled.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        (info, grads)
    }

    /// Consume raw grads through the split and both adaptor steps. The
    /// hidden-matrix partition goes to the optimizer named by [`OptimConfig`]
    /// (Muon today); everything else rides AdamW. Each adaptor only receives
    /// the ids it owns, so neither touches the other's parameters.
    ///
    /// The LR schedulers advance exactly once here — one call per optimizer
    /// step. Advancing per micro-batch would stretch a `warmup_steps: 200`
    /// ramp across `accum_steps` times as many updates.
    pub fn optimizer_step(&mut self, grads: GradientsParams) {
        let (lr_muon, lr_adamw) = self.lrs.step();
        let (hidden_grads, adamw_grads) = split_grads::<B>(&self.muon_ids, grads);
        let model = self.model.clone();
        self.model = self.muon.step(lr_muon, model, hidden_grads);
        let model = self.model.clone();
        self.model = self.adamw.step(lr_adamw, model, adamw_grads);
    }

    pub fn train_step(&mut self, batch: &[&Instance]) -> StepInfo {
        let fk = self.free_k;
        let (info, grads) = self.forward_backward(batch, 1.0, fk);
        self.optimizer_step(grads);
        info
    }

    /// Greedy decode to EOS: exact-match, copy flag, loop stats per instance.
    /// Decodes on the inner (inference) backend: eval forwards must not
    /// register nodes on the global autodiff tape, which is reclaimed only
    /// by `backward()` — un-backwarded eval graphs pile up (~80MB/instance
    /// at 1M scale) and OOM both RAM and VRAM.
    pub fn evaluate(
        &self,
        instances: &[Instance],
        max_new: usize,
        order: Option<&[usize]>,
    ) -> Vec<MetricRecord> {
        self.evaluate_batched(instances, max_new, 8, order)
    }

    /// Batched greedy decode with per-row EOS masking. Rows decode in lockstep
    /// (one fused forward per step for the whole chunk instead of one per
    /// instance); finished rows freeze while the rest continue. Per-row
    /// accuracy/copy are exact; steps/halt use batch means (cells average
    /// them anyway).
    pub fn evaluate_batched(
        &self,
        instances: &[Instance],
        max_new: usize,
        chunk: usize,
        order: Option<&[usize]>,
    ) -> Vec<MetricRecord> {
        self.evaluate_batched_inner(instances, max_new, chunk, order, None)
    }

    /// [`Self::evaluate`] that reports liveness during decode.
    ///
    /// Needed because eval is slow enough to trip the stall watchdog on its
    /// own: the final eval runs four decode passes (trained order, shuffled,
    /// and the same two for the random-weight control), and on a 3.5M-param
    /// model that can exceed `stuck_timeout_secs` with no training step in
    /// between to ping. A slow-but-progressing eval is indistinguishable from
    /// a hang to a watchdog that only hears from the training loop.
    pub fn evaluate_watched(
        &self,
        instances: &[Instance],
        max_new: usize,
        order: Option<&[usize]>,
        progress: &dyn Fn(),
    ) -> Vec<MetricRecord> {
        self.evaluate_batched_inner(instances, max_new, 8, order, Some(progress))
    }

    fn evaluate_batched_inner(
        &self,
        instances: &[Instance],
        max_new: usize,
        chunk: usize,
        order: Option<&[usize]>,
        progress: Option<&dyn Fn()>,
    ) -> Vec<MetricRecord> {
        let model = self.model.valid();
        let mut out = Vec::with_capacity(instances.len());
        for group in instances.chunks(chunk.max(1)) {
            // Cap the sub-chunk by B×T, not row count alone: rows can decode
            // up to `max_seq_len` (prompt + max_new, frozen at the cap), so
            // budget with the padded worst case and never re-chunk mid-decode.
            let worst_row = group
                .iter()
                .map(|i| i.prompt_ids().len())
                .max()
                .unwrap_or(1)
                .saturating_add(max_new)
                .min(self.config.max_seq_len)
                .max(1);
            let t_worst = crate::harness::bucket_len_within(
                worst_row,
                self.config.max_seq_len,
                "eval chunk sizing",
            );
            let sub = (EVAL_BT_BUDGET / t_worst).clamp(1, group.len());
            for sub_group in group.chunks(sub) {
                out.extend(self.decode_chunk(&model, sub_group, max_new, order, progress));
            }
        }
        // Decode allocates a fresh shape family per chunk and per growing
        // decode step; the backend buffer pool retains them all. Hand them
        // back before training resumes (no-op on backends without pooling).
        B::memory_cleanup(&self.device);
        out
    }

    fn decode_chunk(
        &self,
        model: &LoopedTransformer<B::InnerBackend>,
        group: &[Instance],
        max_new: usize,
        order: Option<&[usize]>,
        progress: Option<&dyn Fn()>,
    ) -> Vec<MetricRecord> {
        let g = group.len();
        let mut ids: Vec<Vec<i64>> = group.iter().map(|i| i.prompt_ids()).collect();
        let mut out_bytes: Vec<Vec<u8>> = vec![vec![]; g];
        let mut active = vec![true; g];
        let mut steps_sum = 0usize;
        let mut halt_sum = 0.0f32;
        let mut block_sums: Vec<f32> = vec![];
        let mut n_decode = 0usize;
        for _ in 0..max_new {
            // Liveness ping before the forward: one decode step on a 3.5M
            // model at T=512 can take seconds, and the final eval runs up to
            // four passes with no training step between them to ping.
            if let Some(p) = progress {
                p();
            }
            for (i, row) in ids.iter().enumerate() {
                if active[i] && row.len() + 1 >= self.config.max_seq_len {
                    active[i] = false;
                }
            }
            if !active.iter().any(|a| *a) {
                break;
            }
            // Shifted inputs (training convention): row input is
            // [PAD] ++ generated bytes, so logits[len] predicts the next
            // byte. Lengths count the prepended PAD.
            let lens: Vec<usize> = ids.iter().map(|r| r.len() + 1).collect();
            let tmax = crate::harness::bucket_len_within(
                *lens.iter().max().unwrap_or(&1),
                self.config.max_seq_len,
                "eval decode",
            );
            let mut flat = Vec::with_capacity(g * tmax);
            for row in ids.iter() {
                for i in 0..tmax {
                    flat.push(if i == 0 {
                        crate::harness::PAD as i64
                    } else if i - 1 < row.len() {
                        row[i - 1]
                    } else {
                        crate::harness::PAD as i64
                    });
                }
            }
            let tokens = Tensor::<B::InnerBackend, 2, Int>::from_data(
                TensorData::new(flat, [g, tmax]),
                &self.device,
            );
            let res = model.forward(tokens, &self.config, self.stop, &lens, order);
            steps_sum += res.steps_used;
            halt_sum += res.mean_halt;
            if block_sums.len() < res.block_halts.len() {
                block_sums.resize(res.block_halts.len(), 0.0);
            }
            for (acc, v) in block_sums.iter_mut().zip(res.block_halts.iter()) {
                *acc += *v;
            }
            n_decode += 1;
            let v = self.config.vocab_size;
            let mut next_ids = vec![0i64; g];
            let mut got_eos = vec![false; g];
            // One argmax over the whole batch instead of one per row. Each
            // row's next byte is the argmax at ITS OWN last position, and the
            // rows finish at different times, so the positions are gathered
            // into a `[g, V]` tensor and the reduction runs once. The old
            // per-row `slice + argmax` was B kernel launches and B host
            // readbacks per decode step; at max_new=64 and chunk 8 that is
            // 512 launches per chunk, and each readback is a pipeline stall.
            //
            // Inactive rows index position 0 with a clamped value: they are
            // skipped below, so their argmax is never read.
            let mut positions: Vec<usize> = Vec::with_capacity(g);
            for (i, row) in ids.iter().enumerate() {
                let t = if active[i] { row.len() } else { 0 };
                positions.push(t.min(tmax.saturating_sub(1)));
            }
            // Flatten (row, time) into one axis and gather each row's own last
            // position with a single `select`. `take` would insert the index
            // axis into the shape ([B, g, V]) rather than removing it, so a
            // flat `select` on `[B*T, V]` is the shape-clean route: the result
            // is exactly `[g, V]`, one row per gathered position.
            let flat: Vec<i64> = positions
                .iter()
                .enumerate()
                .map(|(i, p)| (i * tmax + p) as i64)
                .collect();
            let index = Tensor::<B::InnerBackend, 1, Int>::from_data(
                TensorData::new(flat, [g]),
                &self.device,
            );
            let gathered = res.logits.clone().reshape([g * tmax, v]).select(0, index);
            let preds = int_vec(&gathered.argmax(1).into_data());
            for i in 0..g {
                if !active[i] {
                    continue;
                }
                let next_id = preds[i];
                if next_id == crate::harness::EOS as i64 {
                    got_eos[i] = true;
                } else {
                    next_ids[i] = next_id;
                }
            }
            for i in 0..g {
                if !active[i] {
                    continue;
                }
                if got_eos[i] {
                    active[i] = false;
                } else {
                    ids[i].push(next_ids[i]);
                    out_bytes[i].push(next_ids[i] as u8);
                    if out_bytes[i].len() >= max_new {
                        active[i] = false;
                    }
                }
            }
        }
        group
            .iter()
            .zip(out_bytes.iter())
            .map(|(inst, gbytes)| {
                let text = String::from_utf8_lossy(gbytes);
                let expected = String::from_utf8_lossy(&inst.target);
                // Counted over `chars`, not bytes: targets are single-token
                // symbols, so UTF-8 byte positions do not line up with the
                // units the model emits.
                let byte_hits = text
                    .chars()
                    .zip(expected.chars())
                    .filter(|(a, b)| a == b)
                    .count();
                MetricRecord {
                    task: inst.info.task.to_string(),
                    track: inst.info.track,
                    k: inst.info.k,
                    correct: text == expected,
                    byte_hits,
                    byte_total: expected.chars().count(),
                    copied: inst.info.demos.iter().any(|d| d.output == text),
                    steps_used: steps_sum / n_decode.max(1),
                    mean_halt: halt_sum / n_decode.max(1) as f32,
                    block_halt: block_sums
                        .iter()
                        .map(|s| s / n_decode.max(1) as f32)
                        .collect(),
                    seed: self.seed,
                }
            })
            .collect()
    }

    pub fn save_checkpoint(&self, path: &Path) {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        recorder
            .record(self.model.clone().into_record(), PathBuf::from(path))
            .expect("checkpoint save");
    }

    pub fn load_checkpoint(
        config: &LoopedConfig,
        path: &Path,
        device: &B::Device,
    ) -> LoopedTransformer<B> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(PathBuf::from(path), device)
            .expect("checkpoint load");
        LoopedTransformer::<B>::new(config, device).load_record(record)
    }
}

/// Split a pool into (train, held-out eval) with **no instance in both**.
///
/// Why this exists: the eval split used to be the first `per_cell` instances
/// of the training pool, so every reported accuracy was measured on data the
/// model had trained on. For procedural tasks each instance samples its own
/// latent rule, so an instance-level split is genuinely held out — the rules
/// differ even though the task family is the same. Track A is still
/// "task semantics seen in training" by design; Track B rules are unseen
/// either way.
///
/// Deterministic given `seed`, and the split is per (task, track) cell so a
/// cell that runs dry is visible rather than silently reshuffling the whole
/// pool. `eval_frac = 0.0` disables the holdout and restores the old
/// in-distribution behavior — only for reproducing an old number.
pub fn partition_holdout(
    pool: &[Instance],
    eval_frac: f64,
    seed: u64,
) -> (Vec<Instance>, Vec<Instance>) {
    assert!(
        (0.0..1.0).contains(&eval_frac),
        "eval_frac must be in [0, 1), got {eval_frac}"
    );
    if eval_frac == 0.0 {
        return (pool.to_vec(), Vec::new());
    }
    // Bucket by cell, then hash each instance to decide its side. Hashing (not
    // a sequential slice) keeps the eval set spread across the length range
    // rather than clustered at whatever `generate` happened to emit first.
    let mut train: Vec<Instance> = Vec::with_capacity(pool.len());
    let mut held: Vec<Instance> = Vec::new();
    for inst in pool {
        if is_held_out(inst, eval_frac, seed) {
            held.push(inst.clone());
        } else {
            train.push(inst.clone());
        }
    }
    (train, held)
}

/// Stable per-instance holdout decision. Uses the instance's own identity
/// (task, track, k, prompt bytes) rather than its pool index, so the same
/// instance lands on the same side regardless of generation order.
fn is_held_out(inst: &Instance, eval_frac: f64, seed: u64) -> bool {
    let mut h: u64 = seed ^ 0x9E37_79B9_7F4A_7C15;
    h = h
        .wrapping_mul(0x100_0000_01B3)
        .wrapping_add(inst.info.task.len() as u64);
    for b in inst.info.task.as_bytes() {
        h = h.wrapping_mul(0x100_0000_01B3).wrapping_add(*b as u64);
    }
    h = h.wrapping_mul(31).wrapping_add(inst.info.track as u64);
    h = h.wrapping_mul(31).wrapping_add(inst.info.k as u64);
    for b in &inst.prompt {
        h = h.wrapping_mul(0x100_0000_01B3).wrapping_add(*b as u64);
    }
    // Compare the top 53 bits against the fraction: exact for any f64 in [0,1)
    // and independent of platform f64 division.
    ((h >> 11) as f64) < eval_frac * (1u64 << 53) as f64
}

/// First `per_cell` instances per (task, track) of a pool. Deterministic, so
/// forgetting checks across stages compare like with like. Apply this to the
/// HELD-OUT side of a [`partition_holdout`] split, not to the training pool.
pub fn eval_split(pool: &[Instance], per_cell: usize) -> Vec<Instance> {
    let mut cells: std::collections::BTreeMap<(String, u8), Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, inst) in pool.iter().enumerate() {
        let key = (inst.info.task.to_string(), inst.info.track as u8);
        let cell = cells.entry(key).or_default();
        if cell.len() < per_cell {
            cell.push(i);
        }
    }
    cells
        .values()
        .flat_map(|v| v.iter().map(|i| pool[*i].clone()))
        .collect()
}

/// Final-eval passes: trained order always; reversed block order iff
/// `shuffle` (role diagnostic). Eval-only; training never permutes.
///
/// `model_tag` distinguishes the trained model from the random-weight control
/// in the log labels, so a sweep can never conflate the two.
pub fn final_eval_passes(
    n_stages: usize,
    shuffle: bool,
    model_tag: &str,
) -> Vec<(String, Option<Vec<usize>>)> {
    let mut passes = vec![(format!("final-{model_tag}"), None)];
    if shuffle {
        passes.push((
            format!("final-{model_tag}-shuffled"),
            Some((0..n_stages).rev().collect()),
        ));
    }
    passes
}

/// Random-weight control for the stage-order diagnostic.
///
/// Returns `(order_acc, shuffled_acc)` for a freshly-initialized model of the
/// same architecture and the same stop mode, on the same split.
///
/// Implemented as a second `Trainer` over new random weights rather than by
/// hand-rolling a decode path: the control must differ from the trained run in
/// exactly one respect, and reusing `Trainer::evaluate` guarantees it differs
/// in nothing else — same chunking, same stop mode, same bucketing.
fn random_weight_control<B: AutodiffBackend>(
    run: &crate::harness::RunConfig,
    set: &[Instance],
    device: &B::Device,
    log: &mut String,
    progress: &dyn Fn(),
) -> (f64, f64) {
    // Distinct seed so the control is uncorrelated with the trained model.
    B::seed(device, run.train.seed ^ 0xA5A5_5A5A_DEAD_BEEF);
    let control = Trainer::<B>::new(
        &run.model, &run.optim, run.stop, &run.train, device,
        None, // no checkpoint: untrained weights, by construction
    );
    let n_stages = control.model.stages.len();
    let mut accs = Vec::new();
    for (label, order) in final_eval_passes(n_stages, true, "control") {
        let records =
            control.evaluate_watched(set, run.train.eval_max_new, order.as_deref(), progress);
        let s = crate::harness::summarize(&records);
        println!(
            "  eval [{label}-pool]: acc {:.3} copy {:.3} halt {:.2} bh [{}] (n={}) [untrained control]",
            s.accuracy,
            s.copy_rate,
            s.mean_halt,
            fmt2(&s.mean_block_halt),
            s.n
        );
        log.push_str(&format!(
            "{{\"eval\":\"{label}\",\"random_weights\":true,\"accuracy\":{:.6},\"n\":{}}}\n",
            s.accuracy, s.n
        ));
        accs.push(s.accuracy);
    }
    B::memory_cleanup(device);
    (accs[0], accs[1])
}

/// Chance-level accuracy for the shuffle control's informativeness gate.
///
/// Taken as the **highest** per-task chance in the split, which is the
/// conservative choice: a gate that is too easy to pass would let a
/// barely-above-chance control masquerade as a real floor.
///
/// The score is the fraction of distinct byte symbols in the task's targets,
/// so a task emitting `a`/`b`/`c` has chance 1/3. This is a proxy, not an
/// exact uniform-over-outputs model — the Dyck and SCAN targets are not
/// uniform over the alphabet they draw from — so it is used only as a bar to
/// clear, never as a number to report.
fn chance_level(instances: &[Instance]) -> f64 {
    let mut per_task: std::collections::BTreeMap<&str, std::collections::BTreeSet<u8>> =
        std::collections::BTreeMap::new();
    for inst in instances {
        per_task
            .entry(inst.info.task)
            .or_default()
            .extend(inst.target.iter().copied());
    }
    per_task
        .values()
        .map(|symbols| 1.0 / symbols.len().max(1) as f64)
        .fold(0.0f64, f64::max)
}

/// Space-joined 2dp rendering for eval summary vectors.
fn fmt2(vals: &[f64]) -> String {
    vals.iter()
        .map(|v| format!("{v:.2}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Outcome of one stage: last checkpoint path for `$prev` chaining.
pub struct StageOutcome {
    pub last_ckpt: String,
    pub log_path: String,
}

/// VRAM used (MiB) via `nvidia-smi`. None on CPU-only machines or any
/// failure (missing binary, parse error): logging must never break those.
fn vram_mb() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut lines = text.lines();
    lines.next()?; // header row
    lines.next()?.split_whitespace().next()?.parse().ok()
}

/// Total device memory in MB, for the tuner's budget.
fn vram_total_mb() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut lines = text.lines();
    lines.next()?;
    lines.next()?.split_whitespace().next()?.parse().ok()
}

/// Grow the micro-batch to fill the GPU, holding the EFFECTIVE batch fixed.
///
/// The returned `accum_steps` compensates for the growth
/// (`effective = micro * accum`), because the effective batch is what sets
/// gradient noise. Varying it would make the batch a second confound on top of
/// whatever axis a sweep is varying.
///
/// Returns `None` when tuning is inapplicable (no CUDA, `auto_batch` off, or
/// no readable memory), in which case the caller keeps the manifest values.
///
/// # Why this cannot OOM the process
///
/// Every candidate is *measured* by actually running steps at it, and growth
/// is authorized by extrapolating from a size that already completed. A CUDA
/// OOM aborts rather than unwinding, so the limit can never be discovered by
/// hitting it — see [`crate::harness::batch_tuner`] for the reasoning.
fn auto_tune_batch<B: AutodiffBackend>(
    trainer: &mut Trainer<B>,
    run: &crate::harness::RunConfig,
    train_pool: &[Instance],
    device: &B::Device,
) -> Option<(usize, usize, usize)> {
    use crate::harness::batch_tuner::{Measurement, next_candidate, predicts_fit};
    if !run.train.auto_batch {
        return None;
    }
    // Gated on the instantiated backend rather than a `cfg` feature test: a build
    // can have both features compiled in and still run on the CPU backend, so
    // the gate has to be on what is actually being used.
    //
    // Only meaningful against a real memory budget. On ndarray there is no VRAM
    // to fill: `vram_mb` reports whatever the display is doing (a flat ~12MB),
    // and an unguarded tuner reads that flat line as unlimited headroom and
    // grows to the ceiling for free. Skip it rather than emit a batch number
    // that means nothing.
    if !std::any::type_name::<B>().contains("Cuda") {
        println!(
            "auto-batch: skipped, no device memory to fill on {}",
            std::any::type_name::<B>()
        );
        return None;
    }
    let total = vram_total_mb()?;
    let target_effective = run.train.batch_size.max(1) * run.train.accum_steps.max(1);
    // The ladder's axis is the B×T budget, NOT the sampling window.
    //
    // `batch_size` alone cannot use more VRAM: the trim already caps rows per
    // band, so a window of 512 at the default budget still delivers at most
    // `MICRO_BT_BUDGET / T` rows to the GPU. Growing the window past that point
    // measured as a flat memory line — the signature of a knob that has
    // stopped being connected to the thing it claims to control.
    let mut bt = MICRO_BT_BUDGET;
    // The ceiling is the B×T product, so a manifest can bound memory directly:
    // the top band (T=512) then admits `max_bt / 512` rows.
    let ceiling = run.train.auto_batch_max.max(bt);

    // Baseline before any activation exists: weights + optimizer state +
    // context. Subtracting it is what makes the extrapolation a statement
    // about activations rather than about a constant.
    let baseline = vram_mb().unwrap_or(0);

    println!(
        "auto-batch: tuning the B×T budget from {bt} (window {target_effective}), \
         {total}MB total, baseline {baseline}MB, ceiling {ceiling}"
    );

    // A non-monotonic reading means the previous measurement was not a
    // reliable peak, and every prediction extrapolated from it is wrong.
    //
    // This is not defensive padding: an allocator that keeps pages between
    // probes makes "memory fell when the budget doubled" routine, and without
    // this the ladder takes the lower reading as its slope and can authorize a
    // budget larger than any actually measured.
    // `last` is the most recent trustworthy measurement, and is the ONLY source
    // of the slope every prediction below is computed from.
    let mut last: Option<Measurement> = None;
    loop {
        // Measure by doing real work: the probe runs the same forward and
        // backward a training step would, at a window sized so the budget is
        // the only binding constraint. Nothing here is simulated, so a budget
        // that survives the probe will survive the step.
        let window = window_for_budget(bt, run.train.batch_size);
        let peak = measure_batch_peak::<B>(trainer, train_pool, window, bt, device);
        let m = Measurement {
            batch: bt,
            peak_mb: peak.saturating_sub(baseline),
        };
        println!(
            "  probe B×T {bt} (window {window}): +{}MB over baseline ({}MB used)",
            m.peak_mb, peak
        );
        // A non-monotonic reading means the previous measurement was not a
        // reliable peak, and every prediction extrapolated from it is wrong.
        if let Some(prev) = last.filter(|p| m.peak_mb < p.peak_mb) {
            println!(
                "  memory fell as the budget grew ({}MB -> {}MB): the allocator \
                 is not returning pages between probes, so peaks are not \
                 comparable and no further growth is predictable. \
                 Keeping the largest measured, B×T {}",
                prev.peak_mb, m.peak_mb, prev.batch
            );
            break;
        }
        // A measurement that never moved means nothing is constraining growth.
        //
        // The 0MB case is a probe that cannot see this process at all (a CPU
        // backend reads the display's flat figure). Extrapolating it gives
        // slope 0, so every candidate predicts as fitting and the ladder runs
        // to the ceiling on imagination alone. A flat reading ABOVE zero is
        // the allocator holding its high-water mark, which is equally
        // uninformative about what a bigger budget would cost.
        if last.is_some_and(|p| p.peak_mb == m.peak_mb) {
            if m.peak_mb == 0 {
                println!(
                    "  memory reads 0MB at every size — the probe cannot see \
                     this process's allocations, so no prediction is \
                     meaningful; stopping at B×T {bt}"
                );
            } else {
                println!(
                    "  memory flat at +{}MB as the budget grew: the allocator \
                     is holding a high-water mark, so this reading says \
                     nothing about what a bigger budget would cost. \
                     Stopping at B×T {bt} rather than guessing",
                    m.peak_mb
                );
            }
            break;
        }
        last = Some(m);
        let budget = crate::harness::batch_tuner::Budget::new(
            total,
            run.train.auto_batch_headroom,
            baseline,
        );
        match next_candidate(bt, ceiling) {
            // Refusal is monotone in candidate size (pinned by a test), so
            // stopping at the first refusal cannot skip a size that fits.
            Some(cand) if predicts_fit(&m, cand, &budget) => bt = cand,
            Some(cand) => {
                println!("  B×T {cand} predicted to exceed budget; stopping at {bt}");
                break;
            }
            None => break,
        }
    }
    // The largest budget actually MEASURED, which is the last one when the
    // ladder grew cleanly. A non-monotonic reading broke out with `last` still
    // holding the trustworthy one, and claiming the rejected budget here would
    // select a size whose memory was never observed.
    let chosen_bt = last.map(|m| m.batch).unwrap_or(bt);
    // The window is sized from the budget, and the micro-batch at the shortest
    // band is what the effective batch is computed from.
    let window = window_for_budget(chosen_bt, run.train.batch_size);
    let micro_at_shortest = (chosen_bt / crate::harness::batch::BUCKET_EDGES[0].max(1)).max(1);
    let accum = crate::harness::batch_tuner::accum_for(micro_at_shortest, target_effective);
    let realized = micro_at_shortest * accum;
    println!(
        "auto-batch: selected B×T {chosen_bt} -> window {window}, \
         short-band micro {micro_at_shortest} x {accum} = effective {realized} \
         (target {target_effective})"
    );
    if realized != target_effective {
        // Rounding up means the realized effective batch can exceed the target.
        // Say so rather than let it read as an exact match.
        println!(
            "  note: effective batch is {realized}, above the requested {target_effective} \
             (the tuner cannot subdivide a micro-batch)"
        );
    }
    Some((chosen_bt, window, accum))
}

/// High-water mark of device memory while running a few real steps at `batch`.
///
/// Polls around each step rather than only at the end: an allocation that
/// spikes during forward and is freed by the time the step returns would be
/// invisible to an end-only measurement, and that spike is exactly what
/// decides whether the next size fits.
fn measure_batch_peak<B: AutodiffBackend>(
    trainer: &mut Trainer<B>,
    train_pool: &[Instance],
    window: usize,
    bt_budget: usize,
    device: &B::Device,
) -> u64 {
    const PROBE_STEPS: usize = 3;
    let mut peak = vram_mb().unwrap_or(0);
    let mut rng = HarnessRng::new(0xA070_5EED_5EED_5EED ^ bt_budget as u64);
    for _ in 0..PROBE_STEPS {
        let sampled = Trainer::<B>::sample_banded_batch(&mut rng, train_pool, window);
        if sampled.is_empty() {
            break;
        }
        let t_raw = sampled
            .iter()
            .map(|i| i.prompt.len() + i.target.len() + 1)
            .max()
            .unwrap_or(1);
        let t_pad = crate::harness::bucket_len(t_raw);
        let micro = &sampled[..(bt_budget / t_pad).clamp(1, sampled.len())];
        let micro = &micro[..top_bucket_rows(t_pad, micro.len())];
        let (_info, grads) = trainer.forward_backward(micro, 1.0, 0);
        // The tape is held by `grads`, so the peak exists while it is alive.
        peak = peak.max(vram_mb().unwrap_or(0));
        // Release without stepping: the optimizer must not move, or the probe
        // would train the model and the run would start from a different point
        // than an untuned run.
        drop(grads);
        B::memory_cleanup(device);
        peak = peak.max(vram_mb().unwrap_or(0));
    }
    peak
}

/// Run one manifest end to end: pool -> train loop -> evals -> checkpoints.
/// Shared by single-manifest runs and curriculum stages. `init_from` chains
/// onto a previous stage's checkpoint (optimizer states restart fresh).
///
/// # Panics
/// Panics if `run` fails validation. Manifest loaders call
/// [`crate::harness::RunConfig::validate`] first, so by the time a `RunConfig`
/// reaches here it is already known good; a programmatic caller that skipped
/// that gets every problem listed at once rather than the first.
pub fn run_stage<B: AutodiffBackend>(
    run: &crate::harness::RunConfig,
    device: &B::Device,
    init_from: Option<&Path>,
    eval_extra: &[(String, Vec<Instance>)],
) -> StageOutcome {
    use crate::harness::{generate, summarize};
    use crate::tasks::TaskRegistry;

    // Cheap insurance for the programmatic path. The loader already ran it.
    let problems = run.validate();
    assert!(
        problems.is_empty(),
        "invalid run config:\n  - {}",
        problems.join("\n  - ")
    );

    std::fs::create_dir_all(&run.train.ckpt_dir).expect("ckpt dir");
    let registry = TaskRegistry::builtin();
    let pool = generate(&run.experiment, &registry);
    // Hold out eval instances BEFORE anything touches the pool, so no reported
    // accuracy is measured on data the model trained on.
    let (train_pool, held) = partition_holdout(&pool, run.train.eval_holdout, run.train.seed);
    let held_note = if held.is_empty() {
        "EVAL SET EMPTY: eval_holdout is 0, so every accuracy below is in-distribution".to_string()
    } else {
        format!("{} of them held out", held.len())
    };
    println!(
        "pool: {} instances ({} train, {})",
        pool.len(),
        train_pool.len(),
        held_note
    );
    let eval_set = eval_split(&held, 16);
    println!("eval: {} instances (held out)", eval_set.len());
    // Larger final-eval split (48/cell): bhC/bhW and correlations are noise
    // at n=16. Deterministic prefix-superset of `eval_set`.
    let final_set = eval_split(&held, 48);
    println!("final eval: {} instances (held out)", final_set.len());
    assert!(
        !train_pool.is_empty(),
        "train pool is empty: per_cell and eval_holdout leave nothing to train on"
    );
    // Length-band training pool: stable padded T across steps.
    let mut train_pool = train_pool;
    train_pool.sort_by_key(|i| i.prompt.len() + i.target.len());

    // Data-scaling cap, after the holdout split so the eval set is fixed.
    // A stride over the length-sorted pool keeps the subset representative of
    // the length range; taking a prefix would select only the shortest rows.
    if run.train.max_train_instances > 0 && train_pool.len() > run.train.max_train_instances {
        let cap = run.train.max_train_instances;
        let stride = (train_pool.len() / cap).max(1);
        let capped: Vec<Instance> = train_pool
            .iter()
            .step_by(stride)
            .take(cap)
            .cloned()
            .collect();
        println!(
            "data cap: {} -> {} training instances (stride {stride}, eval set unchanged)",
            train_pool.len(),
            capped.len()
        );
        train_pool = capped;
    }

    // The in-distribution half of the memorization measurement: a sample of
    // instances the model DID train on, evaluated with the same decoder and
    // the same limit as the held-out split.
    //
    // Taken AFTER the data cap, deliberately: it must measure what the model
    // actually saw. Sampled before, a capped run would score instances it
    // never trained on and report them as in-distribution — which would make
    // the gap collapse to zero at exactly the points where memorization
    // matters most.
    let indist_set = if run.train.eval_in_distribution {
        let s = eval_split(&train_pool, run.train.indist_eval_per_cell);
        println!(
            "in-dist eval: {} instances (from the {} the model trains on)",
            s.len(),
            train_pool.len()
        );
        s
    } else {
        Vec::new()
    };

    // `run.train` is passed straight through. A previous version rebuilt a
    // partial `TrainConfig` here, which was both redundant and a trap: the
    // Trainer read only `seed` and the two learning rates, so the other six
    // copied fields were dead, and any *new* field read off that local would
    // have silently taken its struct default rather than the manifest value.
    // The loop below already reads `run.train.*` directly, so the manifest is
    // the single source of truth for every knob.
    println!(
        "optimizer: {} | stop: {}",
        run.optim.name(),
        run.stop.kind_name()
    );
    let mut trainer = Trainer::<B>::new(
        &run.model, &run.optim, run.stop, &run.train, device, init_from,
    );
    // Auto-batch runs before the loop, on the real trainer, so a tuned run and
    // an untuned one differ only in the batch — which the accum compensation
    // then holds constant in effective terms.
    let tuned = auto_tune_batch::<B>(&mut trainer, run, &train_pool, device);
    // Resolved once, before the loop, and logged: the B×T budget is a function
    // of the card it ran on, so a run whose realized budget is not recorded is
    // not reproducible. Un-tuned runs keep the shipped constant.
    let (bt_budget, batch_size, accum_steps) = tuned.unwrap_or((
        MICRO_BT_BUDGET,
        run.train.batch_size,
        run.train.accum_steps.max(1),
    ));
    if tuned.is_some() {
        println!(
            "auto-batch: training with B×T {bt_budget}, window {batch_size}, \
             {accum_steps} micro-batches per step"
        );
    }
    let mut rng = HarnessRng::new(run.train.seed ^ 0x9E37_79B9_7F4A_7C15);
    let watchdog = crate::fail_fast::Watchdog::spawn(run.train.stuck_timeout_secs);
    // Ponder warmup base: ramped per step via the setter below.
    let base_ponder = run.model.ponder_weight;
    let warmup_steps = run.train.ponder_warmup_steps;

    let log_path = format!("{}/run.jsonl", run.train.ckpt_dir);
    let mut log = String::new();
    // Init snapshot doubles as the first chained checkpoint.
    let mut last_ckpt = format!("{}/step000000.mpk", run.train.ckpt_dir);
    trainer.save_checkpoint(Path::new(&last_ckpt));
    // Throughput clock: rate is measured over each log interval.
    let mut last_log = (Instant::now(), 0usize);

    for step in 0..run.train.steps {
        // Ponder warmup: effective weight ramps 0 -> base over
        // `warmup_steps`. CE term untouched (ramp only scales ponder).
        trainer.set_ponder_weight(ponder_warmup_weight(base_ponder, step, warmup_steps));
        // Gradient accumulation: `accum` micro-batches of 1/accum-scaled
        // losses merge into one mean-equivalent grad for a single step.
        let accum = accum_steps;
        let scale = 1.0 / accum as f64;
        let specs = trainer.model.grad_specs();
        let mut acc_grads = None;
        let (mut loss_sum, mut ponder_sum, mut halt_sum) = (0.0f32, 0.0f32, 0.0f32);
        let (mut ans_sum, mut eos_sum) = (0.0f32, 0.0f32);
        let mut steps_sum = 0usize;
        for _ in 0..accum {
            let window = Trainer::<B>::sample_banded_batch(&mut rng, &train_pool, batch_size);
            // B×T budget: trim the banded window (drop longest rows, they sit
            // at the window's end) so padded T never inflates the tape.
            let t_raw = window
                .iter()
                .map(|i| i.prompt.len() + i.target.len() + 1)
                .max()
                .unwrap_or(1);
            let t_pad = crate::harness::bucket_len(t_raw);
            let micro = &window[..(bt_budget / t_pad).clamp(1, window.len())];
            let micro = &micro[..top_bucket_rows(t_pad, micro.len())];
            let fk = trainer.free_k;
            let (info, grads) = trainer.forward_backward(micro, scale, fk);
            loss_sum += info.loss;
            ponder_sum += info.ponder;
            halt_sum += info.mean_halt;
            ans_sum += info.ce_answer;
            eos_sum += info.ce_eos;
            steps_sum += info.steps_used;
            acc_grads = Some(match acc_grads {
                None => grads,
                Some(a) => merge_grads::<B>(&specs, a, grads),
            });
            // Per-micro pool release. Each micro visits an independently
            // sampled band (up to 8 different T-bucket shape families per
            // optimizer step); the pool otherwise retains every family's
            // pages plus autotune scratch until the step ends, and the
            // within-step peak crosses 4GB (~3.75GB sustained, death on a
            // 15MB page). Allocator-only: same windows, same grads, no
            // math/dynamics change.
            B::memory_cleanup(device);
        }
        trainer.optimizer_step(acc_grads.expect("at least one micro-batch"));
        // Per-step pool release. The cubecl pool never hands pages back to
        // the driver on its own, so band-hopping (T buckets 64..512 x fused
        // shape families x autotune scratch) accumulates monotonically until
        // a fresh page no longer fits in 4GB (death: 61.77MB page at ~3.7GB
        // used, mid-train, before any eval). Releasing after every optimizer
        // step bounds retention to one step's families plus persistent state
        // (params/optimizer, ~100MB). Allocator-only: no math,
        // batch-composition, or dynamics change. Costs some re-alloc churn
        // versus the old never-release policy; correctness first.
        B::memory_cleanup(device);
        let n = accum as f32;
        let info = StepInfo {
            loss: loss_sum / n,
            ponder: ponder_sum / n,
            steps_used: steps_sum / accum,
            mean_halt: halt_sum / n,
            ce_answer: ans_sum / n,
            ce_eos: eos_sum / n,
        };
        trainer.update_free_schedule(info.ce_answer, &run.train.free_schedule);
        watchdog.ping_step(step);
        if step % run.train.log_every == 0 {
            let vram = match vram_mb() {
                Some(mb) => format!("vram {mb}MB"),
                None => "vram n/a".to_string(),
            };
            let now = Instant::now();
            let dt = now.duration_since(last_log.0).as_secs_f64().max(1e-6);
            let rate = (step - last_log.1) as f64 / dt;
            last_log = (now, step);
            // `loops` and `halt` are ACT quantities. Under fixed depth the
            // loop count is the config, and `ponder` is a constant offset
            // rather than a per-token cost, so printing all three every run
            // invites comparing numbers that no longer mean what they did.
            let depth = match run.stop {
                StopConfig::Act { .. } => format!(
                    "ponder {:.2} loops {} halt {:.2}",
                    info.ponder, info.steps_used, info.mean_halt
                ),
                StopConfig::Fixed { .. } => String::new(),
                StopConfig::Converge => format!("halt {:.2}", info.mean_halt),
            };
            println!(
                "step {step:>5} loss {:.4} (ans {:.3} eos {:.3}) {depth} free {} {vram} {:.2} st/s",
                info.loss, info.ce_answer, info.ce_eos, trainer.free_k, rate
            );
        }
        // Skip the in-loop save on the final step: the post-loop snapshot
        // writes the same `step{steps}.mpk` path moments later. With the
        // shipped manifests (steps 500, ckpt_every 100) that was a duplicate
        // write of a 3.5M-param record on every run. The final path is still
        // written exactly once, by the post-loop snapshot, so a chain's
        // `init_from: $prev` is unaffected.
        let is_final_step = step + 1 == run.train.steps;
        if !is_final_step && step % run.train.ckpt_every == 0 {
            last_ckpt = format!("{}/step{:06}.mpk", run.train.ckpt_dir, step);
            trainer.save_checkpoint(Path::new(&last_ckpt));
        }
        if step > 0
            && step >= run.train.eval_start_step
            && (step - run.train.eval_start_step).is_multiple_of(run.train.eval_every)
        {
            // Own split plus retained splits from earlier stages (forgetting).
            let mut evals: Vec<(&str, &Vec<Instance>)> = vec![("self", &eval_set)];
            for (name, pool) in eval_extra {
                evals.push((name, pool));
            }
            for (label, set) in evals {
                let records = trainer.evaluate(set, run.train.eval_max_new, None);
                let mut cells: std::collections::BTreeMap<
                    (String, String),
                    Vec<crate::harness::Record>,
                > = std::collections::BTreeMap::new();
                for r in records {
                    cells
                        .entry((r.task.clone(), format!("{:?}", r.track)))
                        .or_default()
                        .push(r);
                }
                for ((task, track), rs) in &cells {
                    let s = summarize(rs);
                    let vram = match vram_mb() {
                        Some(mb) => format!("vram {mb}MB"),
                        None => "vram n/a".to_string(),
                    };
                    println!(
                        "  eval [{label}] {task}/{track}: acc {:.2} byte {:.3} copy {:.2} halt {:.2} bh [{}] (n={}) {vram}",
                        s.accuracy,
                        s.byte_accuracy,
                        s.copy_rate,
                        s.mean_halt,
                        fmt2(&s.mean_block_halt),
                        s.n
                    );
                    watchdog.ping_step(step);
                    for r in rs.iter() {
                        let mut line = r.to_json();
                        line.pop(); // trailing `}`; append eval context
                        log.push_str(&format!("{line},\"eval\":\"{label}\",\"step\":{step}}}\n"));
                    }
                }
            }
        }
    }
    // Always snapshot the end of the stage for chaining.
    last_ckpt = format!("{}/step{:06}.mpk", run.train.ckpt_dir, run.train.steps);
    trainer.save_checkpoint(Path::new(&last_ckpt));
    // Final eval, always (independent of the eval_every cadence): one
    // authoritative read per run on the larger split, plus a block-reversed
    // repeat iff `shuffle_eval`.
    //
    // The reversal collapse is guaranteed by construction (stages are
    // sequential, so reversal reverses the data flow), so the same two passes
    // are ALSO run against an untrained model of the same architecture. Only
    // the trained gap in excess of that control floor is evidence of learned
    // role specialization — see `harness::shuffle_control`.
    let n_stages = trainer.model.stages.len();
    let mut trained_order: Option<(f64, f64)> = None;
    let mut trained_shuffled: Option<(f64, f64)> = None;
    for (label, order) in final_eval_passes(n_stages, run.train.shuffle_eval, "trained") {
        let records = trainer.evaluate_watched(
            &final_set,
            run.train.eval_max_new,
            order.as_deref(),
            &|| {
                watchdog.ping_step(run.train.steps);
            },
        );
        let mut cells: std::collections::BTreeMap<(String, String), Vec<crate::harness::Record>> =
            std::collections::BTreeMap::new();
        for r in &records {
            cells
                .entry((r.task.clone(), format!("{:?}", r.track)))
                .or_default()
                .push(r.clone());
        }
        for ((task, track), rs) in &cells {
            let s = summarize(rs);
            let cor: Vec<String> = s
                .block_halt_corr
                .iter()
                .flat_map(|row| row.iter().map(|v| format!("{v:.2}")))
                .collect();
            println!(
                "  eval [{label}] {task}/{track}: acc {:.2} byte {:.3} copy {:.2} halt {:.2} bh [{}] bhC [{}] bhW [{}] p90 [{}] sh [{}] cor [{}] (n={})",
                s.accuracy,
                s.byte_accuracy,
                s.copy_rate,
                s.mean_halt,
                fmt2(&s.mean_block_halt),
                fmt2(&s.mean_block_halt_correct),
                fmt2(&s.mean_block_halt_wrong),
                fmt2(&s.p90_block_halt),
                fmt2(&s.share_block_halt),
                cor.join(" "),
                s.n
            );
            for r in rs.iter() {
                let mut line = r.to_json();
                line.pop();
                log.push_str(&format!("{line},\"eval\":\"{label}\"}}\n"));
            }
        }
        // Pool-level summary: correlations and shares computed across the
        // whole final set, where per-16-cell slices are variance-starved. The
        // pool accuracy is also what the shuffle control is read against, so
        // it is captured rather than only printed.
        let pool_acc = {
            let s = summarize(&records);
            let pool_byte_acc = s.byte_accuracy;
            let cor: Vec<String> = s
                .block_halt_corr
                .iter()
                .flat_map(|row| row.iter().map(|v| format!("{v:.2}")))
                .collect();
            // Profile evidence: the reading that does NOT depend on a
            // reversal collapse or an untrained control, so it works on the
            // weights-only rungs where both are uninformative.
            let ev = crate::harness::evidence_from(&s);
            let reading = match ev.reading() {
                Some(r) => r.label().to_string(),
                None => format!(
                    "unreadable (mean_halt {:.2} below the {:.1}-step floor)",
                    ev.mean_halt,
                    crate::harness::RoleEvidence::MIN_MEANINGFUL_HALT
                ),
            };
            println!(
                "  eval [{label}-pool]: acc {:.2} byte {:.3} copy {:.2} halt {:.2} bh [{}] sh [{}] std [{}] cor [{}] (n={})",
                s.accuracy,
                s.byte_accuracy,
                s.copy_rate,
                s.mean_halt,
                fmt2(&s.mean_block_halt),
                fmt2(&s.share_block_halt),
                fmt2(&s.std_block_halt),
                cor.join(" "),
                s.n
            );
            println!(
                "  eval [{label}-roles]: {reading} | entropy {:.3} maxdev {:.3} mean_offdiag_corr {:.3}",
                s.stage_halt_entropy, s.share_max_deviation, s.mean_offdiag_block_corr
            );
            log.push_str(&format!(
                "{{\"eval\":\"{label}-roles\",\"entropy\":{:.6},\"max_deviation\":{:.6},\
                 \"mean_offdiag_corr\":{:.6},\"mean_halt\":{:.6},\"reading\":{},\
                 \"roles_supported\":{}}}\n",
                s.stage_halt_entropy,
                s.share_max_deviation,
                s.mean_offdiag_block_corr,
                s.mean_halt,
                serde_json::to_string(match ev.reading() {
                    Some(r) => r.label(),
                    None => "unreadable",
                })
                .expect("label serializes"),
                ev.reading().is_some_and(|r| r.supports_roles()),
            ));
            (s.accuracy, pool_byte_acc)
        };
        if order.is_none() {
            trained_order = Some(pool_acc);
        } else {
            trained_shuffled = Some(pool_acc);
        }
    }

    // Memorization onset: the train/held-out divergence, from the same run.
    //
    // Held-out accuracy on its own cannot locate the onset: it saturates
    // identically whether the model solved the task or merely stopped
    // benefiting from more data. The GAP between a training-pool sample and
    // the held-out set is the signal, and both halves must come from one run
    // to be comparable at all.
    if !indist_set.is_empty() {
        let records = trainer.evaluate_watched(&indist_set, run.train.eval_max_new, None, &|| {
            watchdog.ping_step(run.train.steps)
        });
        let ind = crate::harness::summarize(&records);
        let (held_acc, held_byte) = trained_order.unwrap_or((0.0, 0.0));
        let gap = held_acc - ind.accuracy;
        let byte_gap = held_byte - ind.byte_accuracy;
        println!(
            "  memorization: train-pool acc {:.3} byte {:.3} | held-out acc {:.3} byte {:.3} | gap {:+.3} / {:+.3} (n_train={} n_held={})",
            ind.accuracy,
            ind.byte_accuracy,
            held_acc,
            held_byte,
            gap,
            byte_gap,
            ind.n,
            final_set.len()
        );
        log.push_str(&format!(
            "{{\"eval\":\"memorization\",\"indist_accuracy\":{:.6},\"heldout_accuracy\":{:.6},\
             \"gap\":{:.6},\"indist_byte_accuracy\":{:.6},\"heldout_byte_accuracy\":{:.6},\
             \"byte_gap\":{:.6},\"n_indist\":{},\"n_heldout\":{},\"seed\":{}}}\n",
            ind.accuracy,
            held_acc,
            gap,
            ind.byte_accuracy,
            held_byte,
            byte_gap,
            ind.n,
            final_set.len(),
            run.train.seed
        ));
    }

    // The role diagnostic, read against its floor.
    if let (Some((t, _)), Some((ts, _))) = (trained_order, trained_shuffled) {
        let (c, cs) = random_weight_control::<B>(run, &final_set, device, &mut log, &|| {
            watchdog.ping_step(run.train.steps);
        });
        let ctrl = crate::harness::shuffle_control::ShuffleControl::new(t, ts, c, cs);
        // Chance level has to come from the task, not be hardcoded: the
        // control gate is only meaningful relative to how hard guessing is.
        // Taken as the highest per-task chance across the eval split, which is
        // the conservative (highest-bar) choice.
        let chance = chance_level(&final_set);
        println!("  shuffle control: {}", ctrl.summary(chance));
        log.push_str(&ctrl.to_json(chance));
        log.push('\n');
    }

    std::fs::write(&log_path, log).expect("write log");
    println!("wrote {log_path}; last ckpt {last_ckpt}");
    StageOutcome {
        last_ckpt,
        log_path,
    }
}

fn scalar_of<B: Backend>(t: &Tensor<B, 1>) -> f32 {
    t.clone().into_data().as_slice::<f32>().unwrap()[0]
}

/// Backend-agnostic Int readback (NdArray uses i64, CUDA uses i32).
/// Patch the first-K target inputs per row with proposal argmax ids.
/// Pure function (testable): prompt region never touched, lengths unchanged,
/// only `seq[pl+j]` for `j < min(K, target_len)` are replaced.
/// CRITICAL: patched sequences must feed TOKENS ONLY — targets always come
/// from the unpatched truth, or the loss scores the model against its own
/// output (self-agreement -> 0 regardless of correctness).
pub fn patch_free_inputs(
    base_seqs: &[Vec<u8>],
    prompt_lens: &[usize],
    pred_flat: &[i64],
    t: usize,
    free_k: usize,
) -> Vec<Vec<u8>> {
    base_seqs
        .iter()
        .enumerate()
        .map(|(r, s)| {
            let mut s = s.clone();
            let target_len = s.len().saturating_sub(prompt_lens[r]);
            for j in 0..free_k.min(target_len) {
                let p = prompt_lens[r] + j;
                if p < s.len() {
                    s[p] = pred_flat[r * t + p.min(s.len() - 1)] as u8;
                }
            }
            s
        })
        .collect()
}

/// Backend-agnostic Int readback (NdArray uses i64, CUDA uses i32). Always a
/// batched readback: the per-row `int_scalar` this replaced cost one host sync
/// per row per decode step.
fn int_vec(data: &TensorData) -> Vec<i64> {
    match data.dtype {
        burn::tensor::DType::I64 => data.as_slice::<i64>().unwrap().to_vec(),
        burn::tensor::DType::I32 => data
            .as_slice::<i32>()
            .unwrap()
            .iter()
            .map(|v| *v as i64)
            .collect(),
        d => panic!("unexpected int dtype {d:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        harness::{Experiment, generate},
        optim::{MuonTuning, OptimConfig},
        tasks::TaskRegistry,
        test_backend::{TestBackend, test_device},
    };
    use burn::tensor::backend::AutodiffBackend;

    type IB = <TestBackend as AutodiffBackend>::InnerBackend;

    fn tiny_trainer() -> Trainer<TestBackend> {
        let optim = OptimConfig::Muon(MuonTuning::new());
        let model_cfg = LoopedConfig::new()
            .with_vocab_size(256)
            .with_d_model(32)
            .with_n_heads(2)
            .with_head_dim(16)
            .with_ffn_hidden(64)
            .with_max_loops(2)
            .with_max_seq_len(256);
        let train_cfg = TrainConfig::new();
        Trainer::<TestBackend>::new(
            &model_cfg,
            &optim,
            crate::model::StopConfig::default(),
            &train_cfg,
            &test_device(),
            None,
        )
    }

    #[test]
    fn train_step_runs_and_eval_decodes() {
        let registry = TaskRegistry::builtin();
        let exp = Experiment {
            tasks: vec!["parity".to_string()],
            per_cell: 8,
            seeds: vec![0],
            ..Default::default()
        };
        let pool = generate(&exp, &registry);
        let mut trainer = tiny_trainer();
        let mut rng = HarnessRng::new(99);
        // A few steps must run without NaN and keep params finite.
        for _ in 0..3 {
            let batch = Trainer::<TestBackend>::sample_batch(&mut rng, &pool, 4);
            let info = trainer.train_step(&batch);
            assert!(info.loss.is_finite(), "loss went NaN/Inf");
            assert!(info.loss >= 0.0);
        }
        let eval_set: Vec<Instance> = pool.iter().take(4).cloned().collect();
        let records = trainer.evaluate(&eval_set, 8, None);
        assert_eq!(records.len(), 4);
        // Chunk size must not change verdicts on this set.
        let records2 = trainer.evaluate_batched(&eval_set, 8, 2, None);
        assert_eq!(records2.len(), 4);
        for (a, b) in records.iter().zip(records2.iter()) {
            assert_eq!(a.correct, b.correct);
            assert_eq!(a.task, b.task);
            assert_eq!(a.k, b.k);
        }
    }

    #[test]
    fn grads_flow_and_weights_move() {
        use crate::harness::collate;
        use burn::optim::GradientsParams;

        let registry = TaskRegistry::builtin();
        let exp = Experiment {
            tasks: vec!["parity".to_string()],
            per_cell: 8,
            seeds: vec![0],
            ..Default::default()
        };
        let pool = generate(&exp, &registry);
        let mut trainer = tiny_trainer();
        let batch: Vec<Instance> = pool.iter().take(4).cloned().collect();
        let col = collate(&batch, &test_device());
        let out = trainer
            .model
            .forward_act(col.tokens, &trainer.config, &col.lengths, None);
        let loss = masked_ce(out.logits, col.targets, col.loss_mask)
            .with_ponder(out.ponder, trainer.config.ponder_weight);
        // Every param must get a nonzero grad. An inverted pad mask zeroes
        // all but the halt grads, so this is the tripwire for that bug class
        // (pad_mask values are pinned in model tests too).
        let mut gp = GradientsParams::from_grads(loss.backward(), &trainer.model);
        let named = trainer.model.grad_specs();
        assert_eq!(named.len(), 14);
        let mut zero = vec![];
        for (name, id, rank) in &named {
            let n: f32 = match rank {
                2 => gp
                    .remove::<IB, 2>(*id)
                    .unwrap()
                    .abs()
                    .sum()
                    .into_data()
                    .as_slice::<f32>()
                    .unwrap()[0],
                _ => gp
                    .remove::<IB, 1>(*id)
                    .unwrap()
                    .abs()
                    .sum()
                    .into_data()
                    .as_slice::<f32>()
                    .unwrap()[0],
            };
            if n == 0.0 {
                zero.push(name.clone());
            }
        }
        assert!(zero.is_empty(), "params with zero grad: {zero:?}");
        // Weights must actually move after a step.
        let before = trainer.model.stages[0].blocks[0]
            .attn
            .q
            .weight
            .val()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let before_head = trainer
            .model
            .head
            .weight
            .val()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let refs: Vec<&Instance> = batch.iter().collect();
        trainer.train_step(&refs);
        let after = trainer.model.stages[0].blocks[0]
            .attn
            .q
            .weight
            .val()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let after_head = trainer
            .model
            .head
            .weight
            .val()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        assert_ne!(before, after, "muon param frozen after a step");
        assert_ne!(before_head, after_head, "adamw param frozen after a step");
    }

    /// A fixed-depth run must not report ACT quantities. The regression this
    /// guards: `forward_act` was hardcoded in two places, so a fixed-depth
    /// manifest would have silently trained with a learned halt.
    /// The manifest must be the single source of truth for training knobs.
    ///
    /// Regression guard for a removed footgun: `run_stage` used to rebuild a
    /// partial `TrainConfig` from `run.train`, copying nine fields of which
    /// the Trainer read three. Any field not copied took its struct default,
    /// so a new knob added to the manifest could be silently ignored. This
    /// pins the values the Trainer actually consumes.
    #[test]
    fn trainer_uses_the_manifests_own_values() {
        let mut cfg = TrainConfig::new();
        cfg.seed = 4242;
        cfg.lr_muon = LrConfig::Constant { lr: 0.011 };
        cfg.lr_adamw = LrConfig::Constant { lr: 0.022 };
        let model_cfg = LoopedConfig::new()
            .with_d_model(32)
            .with_n_heads(2)
            .with_head_dim(16)
            .with_ffn_hidden(64)
            .with_max_loops(2)
            .with_max_seq_len(256);
        let mut trainer = Trainer::<TestBackend>::new(
            &model_cfg,
            &OptimConfig::Muon(MuonTuning::new()),
            StopConfig::default(),
            &cfg,
            &test_device(),
            None,
        );
        // The rates came from `cfg`, not from TrainConfig's defaults.
        assert_eq!(trainer.lrs.step(), (0.011, 0.022));
    }

    /// A run whose learning rates do not match its manifest is a silent
    /// correctness problem, so the round-trip through the loader is checked:
    /// the numbers in a manifest must be the numbers the trainer applies.
    #[test]
    fn manifest_learning_rates_reach_the_trainer() {
        use crate::harness::{RunConfig, load_run};
        let base = RunConfig::smoke();
        let value = serde_json::to_value(&base).expect("serialize");
        let path = std::env::temp_dir().join("generalist-lr-through-loader.json");
        std::fs::write(&path, value.to_string()).expect("write");
        let cfg = load_run(&path).expect("load");
        std::fs::remove_file(&path).ok();

        // Deliberately the manifest's own model config, not a tiny stand-in:
        // this is a loader round-trip, so the model block is exercised too.
        let mut trainer = Trainer::<TestBackend>::new(
            &cfg.model,
            &cfg.optim,
            cfg.stop,
            &cfg.train,
            &test_device(),
            None,
        );
        let (muon, adamw) = trainer.lrs.step();
        let (want_muon, want_adamw) = match (&cfg.train.lr_muon, &cfg.train.lr_adamw) {
            (LrConfig::Constant { lr: m }, LrConfig::Constant { lr: a }) => (*m, *a),
            other => panic!("expected constant LRs in the smoke default, got {other:?}"),
        };
        assert_eq!((muon, adamw), (want_muon, want_adamw));
    }

    /// The load-bearing property of the holdout: no instance may appear in
    /// both the training pool and the eval pool. Before this, `eval_split`
    /// took a prefix of the training pool, so every reported accuracy was
    /// measured on data the model had trained on.
    #[test]
    fn holdout_partitions_without_overlap_and_keeps_both_sides() {
        let registry = TaskRegistry::builtin();
        let exp = Experiment {
            tasks: vec!["parity".to_string(), "subst-fst".to_string()],
            per_cell: 64,
            seeds: vec![0, 1],
            ..Default::default()
        };
        let pool = generate(&exp, &registry);
        let (train, held) = partition_holdout(&pool, 0.1, 7);
        assert_eq!(train.len() + held.len(), pool.len(), "instances lost");
        assert!(!train.is_empty() && !held.is_empty());
        // Disjoint by construction, checked here by content rather than by
        // trusting the partition: key on (task, track, k, prompt, target).
        let key = |i: &Instance| {
            format!(
                "{:?}/{:?}/k{}/{}/{}",
                i.info.task,
                i.info.track,
                i.info.k,
                String::from_utf8_lossy(&i.prompt),
                String::from_utf8_lossy(&i.target)
            )
        };
        let train_keys: std::collections::HashSet<String> = train.iter().map(key).collect();
        for inst in &held {
            assert!(
                !train_keys.contains(&key(inst)),
                "instance leaked into both sides: {:?}",
                String::from_utf8_lossy(&inst.prompt)
            );
        }
        // Roughly the requested fraction, allowing for hash variance.
        let frac = held.len() as f64 / pool.len() as f64;
        assert!((0.04..0.16).contains(&frac), "holdout fraction {frac}");
    }

    /// The holdout must be stable: same seed, same split, whatever the pool
    /// order. Two stages of a chain each call this with the same seed, and
    /// their forgetting evals are only comparable if they hold out the same
    /// instances.
    #[test]
    fn holdout_is_deterministic_and_order_independent() {
        let registry = TaskRegistry::builtin();
        let exp = Experiment {
            tasks: vec!["parity".to_string()],
            per_cell: 64,
            seeds: vec![0],
            ..Default::default()
        };
        let pool = generate(&exp, &registry);
        let (_, a) = partition_holdout(&pool, 0.1, 3);
        let (_, b) = partition_holdout(&pool, 0.1, 3);
        let key = |i: &Instance| String::from_utf8_lossy(&i.prompt).into_owned();
        let ka: std::collections::HashSet<String> = a.iter().map(key).collect();
        let kb: std::collections::HashSet<String> = b.iter().map(key).collect();
        assert_eq!(ka, kb, "same seed gave a different split");

        // Reversing the pool must not move any instance across the boundary.
        let mut rev = pool.clone();
        rev.reverse();
        let (_, c) = partition_holdout(&rev, 0.1, 3);
        let kc: std::collections::HashSet<String> = c.iter().map(key).collect();
        assert_eq!(ka, kc, "split depends on pool order");

        // A different seed is a different split (otherwise the seed is ignored).
        let (_, d) = partition_holdout(&pool, 0.1, 4);
        let kd: std::collections::HashSet<String> = d.iter().map(key).collect();
        assert_ne!(ka, kd, "seed had no effect on the split");
    }

    /// `eval_holdout: 0` must reproduce the old in-distribution behavior
    /// exactly — that is the only reason the escape hatch exists.
    #[test]
    fn zero_holdout_preserves_the_old_behavior() {
        let registry = TaskRegistry::builtin();
        let exp = Experiment {
            tasks: vec!["parity".to_string()],
            per_cell: 16,
            seeds: vec![0],
            ..Default::default()
        };
        let pool = generate(&exp, &registry);
        let (train, held) = partition_holdout(&pool, 0.0, 1);
        assert_eq!(train.len(), pool.len());
        assert!(held.is_empty());
    }

    /// A holdout of 1.0 would empty the training pool. The loader rejects it
    /// (see `RunConfig::validate`), and the partition function refuses it too
    /// rather than handing back an empty training set.
    #[test]
    fn full_holdout_is_rejected() {
        let registry = TaskRegistry::builtin();
        let exp = Experiment {
            tasks: vec!["parity".to_string()],
            per_cell: 8,
            seeds: vec![0],
            ..Default::default()
        };
        let pool = generate(&exp, &registry);
        let threw = std::panic::catch_unwind(|| partition_holdout(&pool, 1.0, 0)).is_err();
        assert!(threw, "eval_frac 1.0 was accepted");

        // And the loader reports it as a validation problem, not a panic.
        let mut cfg = crate::harness::RunConfig::smoke();
        cfg.train.eval_holdout = 1.0;
        let joined = cfg.validate().join("\n");
        assert!(
            joined.contains("eval_holdout must be in [0, 1)"),
            "{joined}"
        );
    }

    /// `eval_holdout: 0` is the escape hatch for reproducing a pre-holdout
    /// number. The loader must call that out as untrustworthy rather than
    /// letting an in-distribution accuracy pass for a generalization result.
    #[test]
    fn zero_holdout_is_flagged_as_in_distribution() {
        let mut cfg = crate::harness::RunConfig::smoke();
        cfg.train.eval_holdout = 0.0;
        let joined = cfg.validate().join("\n");
        assert!(joined.contains("in-distribution"), "{joined}");
    }

    #[test]
    fn stop_mode_actually_reaches_the_forward() {
        let trainer = tiny_trainer();
        // ACT: the step count is data-dependent and within the config's range.
        let act = trainer.model.forward_act(
            burn::tensor::Tensor::zeros([2, 4], &test_device()),
            &trainer.config,
            &[4, 4],
            None,
        );
        assert!(act.steps_used >= 1 && act.steps_used <= 4);

        // Fixed: the step count is exactly what was asked for, at every
        // iteration, because nothing consults the halt head.
        for loops in 1..=4 {
            let out = trainer.model.forward_fixed(
                burn::tensor::Tensor::zeros([2, 4], &test_device()),
                &[4, 4],
                loops,
            );
            assert_eq!(out.steps_used, loops);
            assert_eq!(out.mean_halt, loops as f32);
            assert_eq!(out.block_halts, vec![loops as f32]);
            // Zero ponder: a fixed run has no per-token step count to charge.
            assert_eq!(scalar_of(&out.ponder), 0.0);
        }
    }

    #[test]
    fn trainer_runs_under_each_stop_mode() {
        let registry = TaskRegistry::builtin();
        let exp = Experiment {
            tasks: vec!["parity".to_string()],
            per_cell: 8,
            seeds: vec![0],
            ..Default::default()
        };
        let pool = generate(&exp, &registry);
        for stop in [
            StopConfig::default(),
            StopConfig::Fixed { loops: 2 },
            StopConfig::Converge,
        ] {
            let model_cfg = LoopedConfig::new()
                .with_d_model(32)
                .with_n_heads(2)
                .with_head_dim(16)
                .with_ffn_hidden(64)
                .with_max_loops(2)
                .with_max_seq_len(256)
                // Ponder is meaningless without ACT; matches what the loader
                // requires of a real manifest.
                .with_ponder_weight(if matches!(stop, StopConfig::Act { .. }) {
                    1e-3
                } else {
                    0.0
                });
            let optim = OptimConfig::Muon(MuonTuning::new());
            let mut trainer = Trainer::<TestBackend>::new(
                &model_cfg,
                &optim,
                stop,
                &TrainConfig::new(),
                &test_device(),
                None,
            );
            let mut rng = HarnessRng::new(11);
            let batch = Trainer::<TestBackend>::sample_batch(&mut rng, &pool, 4);
            let info = trainer.train_step(&batch);
            assert!(info.loss.is_finite(), "loss NaN under {stop:?}");
            let records = trainer.evaluate(&pool[..4], 8, None);
            assert_eq!(records.len(), 4, "decode failed under {stop:?}");
        }
    }

    #[test]
    fn lr_schedule_advances_once_per_optimizer_step() {
        // Guards the accumulation trap: advancing per micro-batch would
        // stretch a warmup across accum_steps times as many updates.
        let model_cfg = LoopedConfig::new()
            .with_d_model(32)
            .with_n_heads(2)
            .with_head_dim(16)
            .with_ffn_hidden(64)
            .with_max_loops(2)
            .with_max_seq_len(256);
        let train_cfg = TrainConfig::new().with_lr_muon(LrConfig::Linear {
            lr: 1e-4,
            max_lr: 1e-2,
            warmup_steps: 10,
        });
        let mut trainer = Trainer::<TestBackend>::new(
            &model_cfg,
            &OptimConfig::Muon(MuonTuning::new()),
            StopConfig::default(),
            &train_cfg,
            &test_device(),
            None,
        );
        // burn's linear scheduler counts down from `warmup_steps + 1` and
        // returns `final - step_size * remaining`, so the first call yields
        // `lr` and the peak lands on call `warmup_steps + 1`. Walk the whole
        // ramp and assert that shape rather than a guessed off-by-one.
        // 12 calls: the peak lands on call 11, leaving a call to confirm the hold.
        let mut seen = vec![trainer.lrs.step().0];
        for _ in 0..11 {
            seen.push(trainer.lrs.step().0);
        }
        assert!(
            (seen[0] - 1e-4).abs() < 1e-12,
            "ramp did not start at lr: {seen:?}"
        );
        // Monotonic non-decreasing; the ramp is strictly rising until it
        // saturates at the peak, then flat.
        for w in seen.windows(2) {
            assert!(w[1] >= w[0], "ramp decreased: {seen:?}");
        }
        assert!(
            (seen[10] - 1e-2).abs() < 1e-9,
            "peak not reached on call warmup_steps+1: {seen:?}"
        );
        // And it holds at the peak thereafter.
        assert!(
            (seen[10] - seen[11]).abs() < 1e-12,
            "did not hold: {seen:?}"
        );
    }

    /// The run must end with the final-step checkpoint on disk, exactly once,
    /// and the path must be the one a chain's `init_from: $prev` picks up.
    /// The in-loop cadence skips the final step, so this is the only place
    /// that file is written.
    #[test]
    fn final_checkpoint_is_written_once_and_named_for_chaining() {
        // `run_stage` generates the pool itself, so this drives it through the
        // real entry point rather than assembling instances by hand.
        let dir = std::env::temp_dir().join("generalist-final-ckpt");
        let _ = std::fs::remove_dir_all(&dir);

        let mut run = crate::harness::RunConfig::smoke();
        run.experiment.per_cell = 8;
        run.experiment.seeds = vec![0];
        run.train.steps = 4;
        run.train.ckpt_every = 2; // divides steps, so the naive code double-writes
        run.train.eval_every = 100; // no mid-run eval
        run.train.eval_max_new = 2;
        run.train.ckpt_dir = dir.display().to_string();
        run.train.eval_holdout = 0.25;

        let outcome = run_stage::<TestBackend>(&run, &test_device(), None, &[]);
        let expected = format!("{}/step{:06}.mpk", dir.display(), run.train.steps);
        assert_eq!(outcome.last_ckpt, expected, "chaining path changed");
        assert!(
            std::path::Path::new(&expected).exists(),
            "no final checkpoint"
        );
        // The in-loop saves before it still happened.
        assert!(std::path::Path::new(&format!("{}/step000000.mpk", dir.display())).exists());
        assert!(std::path::Path::new(&format!("{}/step000002.mpk", dir.display())).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Eval decode must be invariant to chunk size, and chunk-independent
    /// per-row accuracy is what the batched argmax has to preserve.
    ///
    /// The gather replaced a per-row `slice + argmax + into_data` with one
    /// flat `select` plus one batched argmax. That is only sound if every row
    /// still reads ITS OWN last position, which a shape mix-up would hide:
    /// wrong values can still decode to plausible strings. So the batched
    /// path is checked against chunk size 1, where the old per-row structure
    /// was unambiguous.
    #[test]
    fn batched_decode_matches_single_row_decode() {
        let registry = TaskRegistry::builtin();
        let exp = Experiment {
            tasks: vec!["parity".to_string()],
            per_cell: 16,
            seeds: vec![0],
            ..Default::default()
        };
        let pool = generate(&exp, &registry);
        let eval: Vec<Instance> = pool.iter().take(8).cloned().collect();
        let trainer = tiny_trainer();

        // Chunk 1: each row decodes in its own forward.
        let one = trainer.evaluate_batched(&eval, 12, 1, None);
        // Chunk 8: all rows decode in lockstep through the batched gather.
        let all = trainer.evaluate_batched(&eval, 12, 8, None);

        assert_eq!(one.len(), eval.len());
        assert_eq!(all.len(), eval.len());
        for (a, b) in one.iter().zip(all.iter()) {
            assert_eq!(a.correct, b.correct, "correctness flipped with chunk size");
            assert_eq!(a.copied, b.copied);
            assert_eq!(a.k, b.k);
            // steps/halt are batch means and legitimately differ; per-row
            // verdict equality is the invariant.
        }
    }

    /// Rows that finish at different steps must each read their own last
    /// position, including a row that hits the `max_new` cap rather than EOS.
    /// Short and long targets in one chunk is exactly the case a positional
    /// mix-up would corrupt.
    #[test]
    fn decode_handles_mixed_row_lengths() {
        let registry = TaskRegistry::builtin();
        let exp = Experiment {
            tasks: vec!["subst-fst".to_string()],
            per_cell: 32,
            seeds: vec![0],
            ..Default::default()
        };
        let pool = generate(&exp, &registry);
        // Sort by target length so the chunk spans a wide range of finish times.
        let mut eval: Vec<Instance> = pool.iter().take(8).cloned().collect();
        eval.sort_by_key(|i| i.target.len());
        let trainer = tiny_trainer();
        let one = trainer.evaluate_batched(&eval, 6, 1, None);
        let all = trainer.evaluate_batched(&eval, 6, 8, None);
        for (a, b) in one.iter().zip(all.iter()) {
            assert_eq!(a.correct, b.correct);
            assert_eq!(a.copied, b.copied);
        }
    }

    /// The shuffle control's informativeness gate compares the control's
    /// accuracy against a chance level. That level has to come from the
    /// instances, not a constant, or the gate silently means different things
    /// on different tasks.
    #[test]
    fn chance_level_is_derived_from_target_alphabet() {
        use crate::tasks::{Instance, InstanceInfo, Track};
        let mk = |task: &'static str, target: &[u8]| Instance {
            prompt: b"q".to_vec(),
            target: target.to_vec(),
            info: InstanceInfo {
                task,
                stage: 0,
                track: Track::A,
                ..InstanceInfo::test_info()
            },
        };
        // A task over a 3-symbol alphabet: chance 1/3.
        let three = vec![
            mk("subst-fst-fixed", b"a"),
            mk("subst-fst-fixed", b"b"),
            mk("subst-fst-fixed", b"c"),
        ];
        assert!((chance_level(&three) - 1.0 / 3.0).abs() < 1e-9);
        // A target repeated across instances adds no new symbols.
        let repeated = vec![
            mk("subst-fst-fixed", b"a"),
            mk("subst-fst-fixed", b"a"),
            mk("subst-fst-fixed", b"c"),
        ];
        assert!((chance_level(&repeated) - 0.5).abs() < 1e-9);

        // Mixed tasks: the MAXIMUM is taken, so the bar is the hardest task's
        // chance rather than an average that could be gamed by easy tasks.
        let mixed = vec![
            mk("subst-fst-fixed", b"a"),
            mk("subst-fst-fixed", b"b"),
            mk("parity", b"0"),
            mk("parity", b"1"),
        ];
        assert!(
            (chance_level(&mixed) - 0.5).abs() < 1e-9,
            "expected the max per-task chance (0.5), got {}",
            chance_level(&mixed)
        );

        // A single-symbol target would divide by zero if unguarded.
        let one = vec![mk("degenerate", b"a")];
        assert_eq!(chance_level(&one), 1.0);

        // Empty split must not panic.
        assert_eq!(chance_level(&[]), 0.0);
    }

    /// End-to-end on the real degenerate case: the weights-only rung, where
    /// the untrained control scores 0 in both orders. The run must emit a
    /// `null` verdict rather than a confident `true`.
    #[test]
    fn degenerate_control_yields_unknown_not_a_false_verdict() {
        let dir = std::env::temp_dir().join("generalist-shuffle-degenerate");
        let _ = std::fs::remove_dir_all(&dir);
        let mut run = crate::harness::RunConfig::smoke();
        run.train.steps = 2;
        run.train.ckpt_every = 100;
        run.train.eval_every = 1000;
        run.train.eval_max_new = 2;
        run.train.ckpt_dir = dir.display().to_string();
        run.train.eval_holdout = 0.25;
        run.train.shuffle_eval = true;

        run_stage::<TestBackend>(&run, &test_device(), None, &[]);
        let log = std::fs::read_to_string(dir.join("run.jsonl")).expect("log written");
        let ctrl = log
            .lines()
            .find(|l| l.contains("\"shuffle-control\""))
            .expect("shuffle-control record present");
        let v: serde_json::Value = serde_json::from_str(ctrl).expect("valid JSON");
        // The smoke model is 1 stage, so both orders are identical and the
        // control sits at chance: the gate must close.
        assert_eq!(v["control_order"], serde_json::json!(0.0));
        assert_eq!(
            v["roles_supported"],
            serde_json::Value::Null,
            "degenerate control emitted a verdict: {ctrl}"
        );
        assert_eq!(v["control_informative"], serde_json::json!(false));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The order-augmented control must actually permute. The whole point is
    /// that the trained model sees different stage orders, so a shuffle that
    /// silently returns the identity would make the control a no-op while
    /// appearing to be evidence.
    #[test]
    fn order_augmentation_actually_permutes() {
        let model_cfg = LoopedConfig::new()
            .with_d_model(32)
            .with_n_heads(2)
            .with_head_dim(16)
            .with_ffn_hidden(64)
            .with_max_loops(2)
            .with_max_seq_len(256)
            .with_n_stages(4);
        let mut t = Trainer::<TestBackend>::new(
            &model_cfg,
            &OptimConfig::Muon(MuonTuning::new()),
            StopConfig::Act {
                shuffle_train: true,
            },
            &TrainConfig::new(),
            &test_device(),
            None,
        );
        let mut seen: Vec<Vec<usize>> = Vec::new();
        for _ in 0..40 {
            seen.push(t.sample_stage_order());
        }
        // Every draw is a permutation of 0..4.
        for o in &seen {
            let mut sorted = o.clone();
            sorted.sort();
            assert_eq!(sorted, vec![0, 1, 2, 3], "not a permutation: {o:?}");
        }
        // Not all identical: 40 draws of a 24-permutation space should
        // produce several distinct orders, including non-identity ones.
        let distinct: std::collections::HashSet<&Vec<usize>> = seen.iter().collect();
        assert!(distinct.len() > 5, "shuffle is near-constant: {distinct:?}");
        assert!(
            seen.iter().any(|o| o != &vec![0, 1, 2, 3]),
            "never drew a non-identity order: {seen:?}"
        );
    }

    /// With `shuffle_train` off, the training forward must use the trained
    /// order — the control changes exactly one thing.
    #[test]
    fn no_augmentation_means_no_permutation() {
        let model_cfg = LoopedConfig::new()
            .with_d_model(32)
            .with_n_heads(2)
            .with_head_dim(16)
            .with_ffn_hidden(64)
            .with_max_loops(2)
            .with_max_seq_len(256)
            .with_n_stages(3);
        let t = Trainer::<TestBackend>::new(
            &model_cfg,
            &OptimConfig::Muon(MuonTuning::new()),
            StopConfig::default(),
            &TrainConfig::new(),
            &test_device(),
            None,
        );
        assert!(!t.shuffle_train);
        assert!(t.last_stage_order.is_none());
        // And the flag is the only difference the trainer sees.
        assert!(!StopConfig::default().shuffle_training());
        assert!(
            StopConfig::Act {
                shuffle_train: true
            }
            .shuffle_training()
        );
    }

    /// `shuffle_train` on a one-stage model is a no-op permutation, so the
    /// loader must reject it rather than let a vacuous control look valid.
    #[test]
    fn shuffle_train_requires_multiple_stages() {
        let mut cfg = crate::harness::RunConfig::smoke(); // n_stages = 1
        cfg.stop = StopConfig::Act {
            shuffle_train: true,
        };
        let joined = cfg.validate().join("\n");
        assert!(
            joined.contains("shuffle_train is set but model.n_stages is 1"),
            "vacuous control was accepted: {joined}"
        );

        // With 2+ stages it validates.
        cfg.model.n_stages = 2;
        assert!(
            !cfg.validate().join("\n").contains("shuffle_train"),
            "rejected a legitimate control: {:?}",
            cfg.validate()
        );
    }

    /// Both stop modes must round-trip through JSON, and the new field must
    /// default so that an existing `{"kind": "act"}` manifest still means
    /// ordered training.
    #[test]
    fn act_variants_roundtrip_and_default() {
        // Compact form: no shuffle_train key.
        let compact: StopConfig =
            serde_json::from_str(r#"{"kind":"act"}"#).expect("compact act parses");
        assert_eq!(compact, StopConfig::default());
        assert!(!compact.shuffle_training());

        // Explicit form.
        let aug: StopConfig =
            serde_json::from_str(r#"{"kind":"act","shuffle_train":true}"#).expect("aug parses");
        assert!(aug.shuffle_training());
        assert!(matches!(
            aug.to_mode(),
            StopMode::Act {
                shuffle_train: true
            }
        ));

        // Serialization always writes the field, so a saved manifest is
        // explicit about which control it ran.
        let text = serde_json::to_string(&StopConfig::default()).expect("serialize");
        assert!(text.contains("shuffle_train"), "default omitted: {text}");

        // And a training run under each mode must produce finite losses.
        for stop in [
            StopConfig::default(),
            StopConfig::Act {
                shuffle_train: true,
            },
        ] {
            let model_cfg = LoopedConfig::new()
                .with_d_model(32)
                .with_n_heads(2)
                .with_head_dim(16)
                .with_ffn_hidden(64)
                .with_max_loops(2)
                .with_max_seq_len(256)
                .with_n_stages(3);
            let mut trainer = Trainer::<TestBackend>::new(
                &model_cfg,
                &OptimConfig::Muon(MuonTuning::new()),
                stop,
                &TrainConfig::new(),
                &test_device(),
                None,
            );
            let registry = TaskRegistry::builtin();
            let exp = Experiment {
                tasks: vec!["parity".to_string()],
                per_cell: 8,
                seeds: vec![0],
                ..Default::default()
            };
            let pool = generate(&exp, &registry);
            let mut rng = HarnessRng::new(23);
            let batch = Trainer::<TestBackend>::sample_batch(&mut rng, &pool, 4);
            let info = trainer.train_step(&batch);
            assert!(info.loss.is_finite(), "loss NaN under {stop:?}");
        }
    }

    #[test]
    fn banded_batches_are_contiguous_windows() {
        let registry = TaskRegistry::builtin();
        let exp = Experiment {
            tasks: vec!["parity".to_string()],
            per_cell: 16,
            seeds: vec![0],
            ..Default::default()
        };
        let mut pool = generate(&exp, &registry);
        pool.sort_by_key(|i| i.prompt.len() + i.target.len());
        let mut rng = HarnessRng::new(3);
        for _ in 0..20 {
            let batch = Trainer::<TestBackend>::sample_banded_batch(&mut rng, &pool, 8);
            assert_eq!(batch.len(), 8);
            // Contiguous in the sorted pool: lengths non-decreasing.
            let lens: Vec<usize> = batch
                .iter()
                .map(|i| i.prompt.len() + i.target.len())
                .collect();
            assert!(lens.windows(2).all(|w| w[0] <= w[1]));
            // Tight band: window span small relative to pool range.
            assert!(*lens.last().unwrap() - lens[0] <= 200);
        }
    }

    #[test]
    fn free_schedule_mapping() {
        let mut trainer = tiny_trainer();
        let sched = [1.0, 0.5, 0.2];
        trainer.update_free_schedule(1.5, &sched);
        assert_eq!(trainer.free_k, 0);
        trainer.update_free_schedule(0.9, &sched);
        assert_eq!(trainer.free_k, 1);
        trainer.update_free_schedule(0.4, &sched);
        assert_eq!(trainer.free_k, 2);
        trainer.update_free_schedule(0.1, &sched);
        assert_eq!(trainer.free_k, 3);
    }

    #[test]
    fn patch_free_inputs_only_touches_target_prefix() {
        // prompt [10,11,12] + target [20,21] (incl EOS slot at index 4).
        let base = vec![vec![10u8, 11, 12, 20, 21]];
        let prompt_lens = vec![3usize];
        // Proposal predicts its flat index everywhere.
        let pred: Vec<i64> = (0..10).collect();
        let out = patch_free_inputs(&base, &prompt_lens, &pred, 10, 2);
        assert_eq!(out, vec![vec![10u8, 11, 12, 3, 4]]);
        // K larger than the target only patches the target region.
        let out = patch_free_inputs(&base, &prompt_lens, &pred, 10, 99);
        assert_eq!(out, vec![vec![10u8, 11, 12, 3, 4]]);
        // K=0 is identity.
        let out = patch_free_inputs(&base, &prompt_lens, &pred, 10, 0);
        assert_eq!(out, base);
    }

    #[test]
    fn free_running_step_stays_finite() {
        let registry = TaskRegistry::builtin();
        let exp = Experiment {
            tasks: vec!["parity".to_string()],
            per_cell: 8,
            seeds: vec![0],
            ..Default::default()
        };
        let pool = generate(&exp, &registry);
        let mut trainer = tiny_trainer();
        trainer.free_k = 2;
        let mut rng = HarnessRng::new(5);
        let batch = Trainer::<TestBackend>::sample_batch(&mut rng, &pool, 4);
        // Exercises the proposal pass + patched inputs end to end.
        let before_free = trainer.free_k;
        let info = trainer.train_step(&batch);
        assert!(info.loss.is_finite());
        assert_eq!(before_free, 2);
    }

    #[test]
    fn checkpoint_roundtrip() {
        let mut trainer = tiny_trainer();
        let registry = TaskRegistry::builtin();
        let exp = Experiment {
            tasks: vec!["parity".to_string()],
            per_cell: 8,
            seeds: vec![0],
            ..Default::default()
        };
        let pool = generate(&exp, &registry);
        let mut rng = HarnessRng::new(7);
        let batch = Trainer::<TestBackend>::sample_batch(&mut rng, &pool, 4);
        trainer.train_step(&batch);
        let dir = std::env::temp_dir().join("generalist-ckpt-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.mpk");
        trainer.save_checkpoint(&path);
        assert!(path.exists());
        let _loaded =
            Trainer::<TestBackend>::load_checkpoint(&trainer.config, &path, &test_device());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn top_bucket_cap_only_clips_512() {
        assert_eq!(top_bucket_rows(512, 6), 2);
        assert_eq!(top_bucket_rows(512, 2), 2);
        assert_eq!(top_bucket_rows(512, 1), 1);
        assert_eq!(top_bucket_rows(256, 6), 6);
        assert_eq!(top_bucket_rows(64, 8), 8);
    }

    #[test]
    fn final_eval_passes_gate_shuffle() {
        let off = final_eval_passes(4, false, "trained");
        assert_eq!(off.len(), 1);
        assert_eq!(off[0].0, "final-trained");
        assert!(off[0].1.is_none());
        let on = final_eval_passes(4, true, "trained");
        assert_eq!(on.len(), 2);
        assert_eq!(on[1].0, "final-trained-shuffled");
        assert_eq!(on[1].1, Some(vec![3, 2, 1, 0]));
        // The tag keeps the trained run and the random-weight control from
        // colliding in a sweep's log.
        let ctl = final_eval_passes(4, true, "control");
        assert_eq!(ctl[0].0, "final-control");
        assert_eq!(ctl[1].0, "final-control-shuffled");
    }

    #[test]
    fn ponder_warmup_ramps_linearly() {
        let base = 0.001;
        let n = 100;
        // Off/legacy: full weight regardless of step.
        assert_eq!(ponder_warmup_weight(base, 0, 0), base);
        assert_eq!(ponder_warmup_weight(base, 57, 0), base);
        // Ramp: 0 at step 0, full at/after N, midpoint ~half.
        assert_eq!(ponder_warmup_weight(base, 0, n), 0.0);
        assert!((ponder_warmup_weight(base, 50, n) - base * 0.5).abs() < 1e-12);
        assert!((ponder_warmup_weight(base, n, n) - base).abs() < 1e-12);
        assert_eq!(ponder_warmup_weight(base, n + 10, n), base);
        // Setter threads through to the loss term (train_step keeps working).
        let mut trainer = tiny_trainer();
        trainer.set_ponder_weight(0.0);
        assert_eq!(trainer.config.ponder_weight, 0.0);
        trainer.set_ponder_weight(base);
        assert_eq!(trainer.config.ponder_weight, base);
    }
}
