//! Training loop v1: constant LR, ACT + Newton-Muon preconditioning,
//! greedy eval with EOS stopping, MPK checkpoints, JSONL metrics.
//!
//! One step: collate -> `forward_act_with_stats` -> masked LM loss ->
//! backward -> split (Muon 2D / AdamW rest) -> observe/refresh precond ->
//! `inv @ G` -> two adaptor steps. Run orchestration (seeds, pools) is the
//! caller's job; see `Experiment` for data dispatch.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use burn::{
    config::Config,
    module::{AutodiffModule, Module, ParamId},
    optim::{
        AdamW, AdamWConfig, GradientsParams, LearningRate, Muon, Optimizer,
        adaptor::OptimizerAdaptor,
    },
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{Int, Tensor, TensorData, backend::{AutodiffBackend, Backend}},
};

use crate::{
    harness::Record as MetricRecord,
    model::{LoopedConfig, LoopedTransformer, StopMode, lm_loss},
    optim::{NewtonMuon, NewtonMuonConfig, PrecondInput, merge_grads, precondition_grads, split_grads},
    tasks::{HarnessRng, Instance},
};

#[derive(Config, Debug)]
pub struct TrainConfig {
    /// Muon LR under `muon_config_matched` (5x repo scale).
    #[config(default = 2e-3)]
    pub lr_muon: f64,
    #[config(default = 3e-4)]
    pub lr_adamw: f64,
    #[config(default = 32)]
    pub batch_size: usize,
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
    /// Scheduled-sampling thresholds: free-running target inputs K =
    /// number of entries with answer-CE below them. Default [1.0]: once
    /// answer error dips below 1.0, the first target input per row comes
    /// from the model's own argmax; more entries grow K progressively.
    #[config(default = "vec![1.0]")]
    pub free_schedule: Vec<f64>,
    #[config(default = "\"checkpoints\".to_string()")]
    pub ckpt_dir: String,
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

/// Activation budget for one training micro-batch, as max `batch × padded_T`.
/// The banded sampler hops length bands (bucket edges 64..512); at a fixed
/// batch size the top band quadruples the autodiff tape (logits are the
/// largest activation: `vocab == d_model` here), which OOMs small GPUs
/// (RTX 3050 4GB) after pool fragmentation from earlier bands + eval.
/// Long-band windows are trimmed so the tape size stays band-independent
/// (T=512 trains at batch 8, T≤256 at the configured batch). Mean-reduced
/// CE keeps merged micro-grads a mean of micro-means either way.
const MICRO_BT_BUDGET: usize = 4096;

/// Same budget for eval decode chunks (B×T per fused forward). Eval prompts
/// bucket to 512 while training bands sit near 64, so a fixed row count that
/// fits a training band OOMs on an eval chunk of long prompts.
const EVAL_BT_BUDGET: usize = 8192;

pub struct Trainer<B: AutodiffBackend> {
    pub model: LoopedTransformer<B>,
    muon: OptimizerAdaptor<Muon<B::InnerBackend>, LoopedTransformer<B>, B>,
    adamw: OptimizerAdaptor<AdamW, LoopedTransformer<B>, B>,
    precond: NewtonMuon<B::InnerBackend>,
    muon_ids: HashSet<ParamId>,
    roles: HashMap<ParamId, PrecondInput>,
    config: LoopedConfig,
    lr_muon: LearningRate,
    lr_adamw: LearningRate,
    device: B::Device,
    /// Current scheduled-sampling depth (see `free_schedule`).
    pub free_k: usize,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(
        model_config: &LoopedConfig,
        nm_config: &NewtonMuonConfig,
        train: &TrainConfig,
        device: &B::Device,
        init_from: Option<&Path>,
    ) -> Self {
        B::seed(device, train.seed);
        let mut model = LoopedTransformer::<B>::new(model_config, device);
        if let Some(path) = init_from {
            // Stage chaining: start from the previous stage's weights.
            // Optimizer + preconditioner states restart fresh (documented).
            model = Self::load_checkpoint(model_config, path, device);
        }
        let precond = NewtonMuon::<B::InnerBackend>::new(
            model_config.d_model,
            model_config.ffn_hidden,
            nm_config,
            device,
        );
        Self {
            muon: nm_config.muon_config_matched().init(),
            adamw: AdamWConfig::new().init(),
            muon_ids: model.muon_ids(),
            roles: model.precond_roles(),
            model,
            precond,
            config: model_config.clone(),
            lr_muon: train.lr_muon,
            lr_adamw: train.lr_adamw,
            device: device.clone(),
            free_k: 0,
        }
    }

    /// Advance the free-running schedule from mean answer-CE.
    pub fn update_free_schedule(&mut self, ce_answer: f32, schedule: &[f64]) {
        self.free_k = schedule.iter().filter(|t| (ce_answer as f64) < **t).count();
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
        // then the first K target inputs come from the model's own argmax
        // instead of the teacher. Everything is still scored vs truth, so
        // the model learns to continue from its own outputs. K=0 (default)
        // is pure teacher forcing.
        let seqs = if free_k == 0 {
            base_seqs
        } else {
            let probe: crate::harness::Collated<B> = crate::harness::collate_seqs(
                base_seqs.clone(),
                prompt_lens.clone(),
                &self.device,
            );
            let tokens_inner = probe.tokens.clone().inner();
            let lens = probe.lengths.clone();
            let valid = self.model.valid();
            let logits = valid.forward(tokens_inner, &self.config, StopMode::Act, &lens).logits;
            let [b2, t2, _] = logits.dims();
            let pred = int_vec(
                &logits
                    .argmax(2)
                    .reshape([b2 * t2])
                    .into_data(),
            );
            base_seqs
                .into_iter()
                .enumerate()
                .map(|(r, mut s)| {
                    let pl = prompt_lens[r];
                    let target_len = lens[r] - pl;
                    for j in 0..free_k.min(target_len) {
                        let p = pl + j;
                        s[p] = pred[r * t2 + p.min(s.len() - 1)] as u8;
                    }
                    s
                })
                .collect()
        };
        let col = crate::harness::collate_seqs(seqs, prompt_lens, &self.device);
        let t = col.lengths.iter().max().copied().unwrap_or(0);
        assert!(
            t <= self.config.max_seq_len,
            "batch length {t} exceeds max_seq_len {}: raise max_seq_len or shorten instances",
            self.config.max_seq_len,
        );
        let (out, stats) = self
            .model
            .forward_act_with_stats(col.tokens, &self.config, &col.lengths);
        let (ce_answer, ce_eos) = crate::model::ce_split(
            out.logits.clone(),
            col.targets.clone(),
            col.loss_mask.clone(),
        );
        let loss = lm_loss(
            out.logits,
            col.targets,
            col.loss_mask,
            out.ponder.clone(),
            self.config.ponder_weight,
        );
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
        self.precond.observe_stats(&stats);
        (info, grads)
    }

    /// Consume raw grads through split + Newton-Muon precondition + both
    /// adaptor steps. Refreshes the preconditioner inverses once per call
    /// (i.e. once per optimizer step, not per micro-batch).
    pub fn optimizer_step(&mut self, grads: GradientsParams) {
        let (muon_grads, adamw_grads) = split_grads::<B>(&self.muon_ids, grads);
        self.precond.maybe_refresh();
        let (muon_grads, leftover) =
            precondition_grads::<B>(&self.precond, &self.roles, muon_grads);
        assert!(leftover.is_empty(), "muon grad without precond role");

        let model = self.model.clone();
        self.model = self.muon.step(self.lr_muon, model, muon_grads);
        let model = self.model.clone();
        self.model = self.adamw.step(self.lr_adamw, model, adamw_grads);
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
    pub fn evaluate(&self, instances: &[Instance], max_new: usize) -> Vec<MetricRecord> {
        self.evaluate_batched(instances, max_new, 8)
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
            let t_worst = crate::harness::bucket_len(worst_row);
            let sub = (EVAL_BT_BUDGET / t_worst).clamp(1, group.len());
            for sub_group in group.chunks(sub) {
                out.extend(self.decode_chunk(&model, sub_group, max_new));
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
    ) -> Vec<MetricRecord> {
        let g = group.len();
        let mut ids: Vec<Vec<i64>> = group.iter().map(|i| i.prompt_ids()).collect();
        let mut out_bytes: Vec<Vec<u8>> = vec![vec![]; g];
        let mut active = vec![true; g];
        let mut steps_sum = 0usize;
        let mut halt_sum = 0.0f32;
        let mut n_decode = 0usize;
        for _ in 0..max_new {
            for (i, row) in ids.iter().enumerate() {
                if active[i] && row.len() >= self.config.max_seq_len {
                    active[i] = false;
                }
            }
            if !active.iter().any(|a| *a) {
                break;
            }
            let lens: Vec<usize> = ids.iter().map(|r| r.len()).collect();
            let tmax = crate::harness::bucket_len(*lens.iter().max().unwrap_or(&1));
            let mut flat = Vec::with_capacity(g * tmax);
            for row in ids.iter() {
                for i in 0..tmax {
                    flat.push(if i < row.len() { row[i] } else { crate::harness::PAD as i64 });
                }
            }
            let tokens = Tensor::<B::InnerBackend, 2, Int>::from_data(
                TensorData::new(flat, [g, tmax]),
                &self.device,
            );
            let res = model.forward(tokens, &self.config, StopMode::Act, &lens);
            steps_sum += res.steps_used;
            halt_sum += res.mean_halt;
            n_decode += 1;
            let v = self.config.vocab_size;
            let mut next_ids = vec![0i64; g];
            let mut got_eos = vec![false; g];
            for (i, row) in ids.iter().enumerate() {
                if !active[i] {
                    continue;
                }
                let t = row.len();
                let next = res.logits.clone().slice([i..i + 1, t - 1..t, 0..v]).reshape([v]).argmax(0);
                let next_id = int_scalar(&next.into_data());
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
                MetricRecord {
                    task: inst.info.task.to_string(),
                    track: inst.info.track,
                    k: inst.info.k,
                    correct: text == expected,
                    copied: inst.info.demos.iter().any(|d| d.output == text),
                    steps_used: steps_sum / n_decode.max(1),
                    mean_halt: halt_sum / n_decode.max(1) as f32,
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

/// Fixed eval split: first 16 instances per (task, track). Deterministic
/// from the pool, so forgetting checks across stages compare like with like.
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

/// Outcome of one stage: last checkpoint path for `$prev` chaining.
pub struct StageOutcome {
    pub last_ckpt: String,
    pub log_path: String,
}

/// Run one manifest end to end: pool -> train loop -> evals -> checkpoints.
/// Shared by single-manifest runs and curriculum stages. `init_from` chains
/// onto a previous stage's checkpoint (optimizer states restart fresh).
pub fn run_stage<B: AutodiffBackend>(
    run: &crate::harness::RunConfig,
    device: &B::Device,
    init_from: Option<&Path>,
    eval_extra: &[(String, Vec<Instance>)],
) -> StageOutcome {
    use crate::harness::{generate, summarize};
    use crate::tasks::TaskRegistry;

    std::fs::create_dir_all(&run.train.ckpt_dir).expect("ckpt dir");
    let registry = TaskRegistry::builtin();
    let pool = generate(&run.experiment, &registry);
    println!("pool: {} instances", pool.len());
    let eval_set = eval_split(&pool, 16);
    println!("eval: {} instances", eval_set.len());
    // Length-band training pool: stable padded T across steps.
    let mut train_pool = pool;
    train_pool.sort_by_key(|i| i.prompt.len() + i.target.len());

    let nm = NewtonMuonConfig::new()
        .with_precond_ewma(run.optim.precond_ewma)
        .with_refresh_every(run.optim.refresh_every)
        .with_ridge_mult(run.optim.ridge_mult)
        .with_eps(run.optim.eps)
        .with_init_diag(run.optim.init_diag)
        .with_ns_steps(run.optim.ns_steps);
    let train_cfg = TrainConfig::new()
        .with_lr_muon(run.train.lr_muon)
        .with_lr_adamw(run.train.lr_adamw)
        .with_batch_size(run.train.batch_size)
        .with_steps(run.train.steps)
        .with_log_every(run.train.log_every)
        .with_eval_every(run.train.eval_every)
        .with_eval_max_new(run.train.eval_max_new)
        .with_seed(run.train.seed)
        .with_ckpt_dir(run.train.ckpt_dir.clone());
    let mut trainer = Trainer::<B>::new(&run.model, &nm, &train_cfg, device, init_from);
    let mut rng = HarnessRng::new(run.train.seed ^ 0x9E37_79B9_7F4A_7C15);
    let watchdog = crate::fail_fast::Watchdog::spawn(run.train.stuck_timeout_secs);

    let log_path = format!("{}/run.jsonl", run.train.ckpt_dir);
    let mut log = String::new();
    // Init snapshot doubles as the first chained checkpoint.
    let mut last_ckpt = format!("{}/step000000.mpk", run.train.ckpt_dir);
    trainer.save_checkpoint(Path::new(&last_ckpt));

    for step in 0..run.train.steps {
        // Gradient accumulation: `accum` micro-batches of 1/accum-scaled
        // losses merge into one mean-equivalent grad for a single step.
        let accum = run.train.accum_steps.max(1);
        let scale = 1.0 / accum as f64;
        let specs = trainer.model.grad_specs();
        let mut acc_grads = None;
        let (mut loss_sum, mut ponder_sum, mut halt_sum) = (0.0f32, 0.0f32, 0.0f32);
        let (mut ans_sum, mut eos_sum) = (0.0f32, 0.0f32);
        let mut steps_sum = 0usize;
        for _ in 0..accum {
            let window =
                Trainer::<B>::sample_banded_batch(&mut rng, &train_pool, run.train.batch_size);
            // B×T budget: trim the banded window (drop longest rows, they sit
            // at the window's end) so padded T never inflates the tape.
            let t_raw = window
                .iter()
                .map(|i| i.prompt.len() + i.target.len() + 1)
                .max()
                .unwrap_or(1);
            let t_pad = crate::harness::bucket_len(t_raw);
            let micro = &window[..(MICRO_BT_BUDGET / t_pad).clamp(1, window.len())];
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
        }
        trainer.optimizer_step(acc_grads.expect("at least one micro-batch"));
        // Hand the backend buffer pool back after every optimizer step.
        // Band hopping (64..512 padded T) + free-running proposal passes make
        // the pool's high-water mark the SUM of all shape families seen; on
        // small GPUs that ratchet alone OOMs the next new shape. Re-allocing
        // the working set each step costs a few ms of cudaMalloc against
        // ~second-scale steps. No-op on backends without pooling.
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
            println!(
                "step {step:>5} loss {:.4} (ans {:.3} eos {:.3}) ponder {:.2} loops {} halt {:.2} free {}",
                info.loss, info.ce_answer, info.ce_eos,
                info.ponder, info.steps_used, info.mean_halt, trainer.free_k
            );
        }
        if step % run.train.ckpt_every == 0 {
            last_ckpt = format!("{}/step{:06}.mpk", run.train.ckpt_dir, step);
            trainer.save_checkpoint(Path::new(&last_ckpt));
        }
        if step % run.train.eval_every == 0 {
            // Own split plus retained splits from earlier stages (forgetting).
            let mut evals: Vec<(&str, &Vec<Instance>)> = vec![("self", &eval_set)];
            for (name, pool) in eval_extra {
                evals.push((name, pool));
            }
            for (label, set) in evals {
                let records = trainer.evaluate(set, run.train.eval_max_new);
                let mut cells: std::collections::BTreeMap<(String, String), Vec<crate::harness::Record>> =
                    std::collections::BTreeMap::new();
                for r in records {
                    cells
                        .entry((r.task.clone(), format!("{:?}", r.track)))
                        .or_default()
                        .push(r);
                }
                for ((task, track), rs) in &cells {
                    let s = summarize(rs);
                    println!(
                        "  eval [{label}] {task}/{track}: acc {:.2} copy {:.2} halt {:.2} (n={})",
                        s.accuracy, s.copy_rate, s.mean_halt, s.n
                    );
                    watchdog.ping_step(step);
                    for r in rs.iter().take(1) {
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
    std::fs::write(&log_path, log).expect("write log");
    println!("wrote {log_path}; last ckpt {last_ckpt}");
    StageOutcome { last_ckpt, log_path }
}

fn scalar_of<B: Backend>(t: &Tensor<B, 1>) -> f32 {
    t.clone().into_data().as_slice::<f32>().unwrap()[0]
}

/// Backend-agnostic Int readback (NdArray uses i64, CUDA uses i32).
fn int_scalar(data: &TensorData) -> i64 {
    int_vec(data).into_iter().next().unwrap_or(-1)
}

fn int_vec(data: &TensorData) -> Vec<i64> {
    match data.dtype {
        burn::tensor::DType::I64 => data.as_slice::<i64>().unwrap().to_vec(),
        burn::tensor::DType::I32 => data.as_slice::<i32>().unwrap().iter().map(|v| *v as i64).collect(),
        d => panic!("unexpected int dtype {d:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        harness::{Experiment, generate},
        optim::NewtonMuonConfig,
        tasks::TaskRegistry,
        test_backend::{TestBackend, test_device},
    };

    fn tiny_trainer() -> Trainer<TestBackend> {
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
            &NewtonMuonConfig::new(),
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
        let records = trainer.evaluate(&eval_set, 8);
        assert_eq!(records.len(), 4);
        // Chunk size must not change verdicts on this set.
        let records2 = trainer.evaluate_batched(&eval_set, 8, 2);
        assert_eq!(records2.len(), 4);
        for (a, b) in records.iter().zip(records2.iter()) {
            assert_eq!(a.correct, b.correct);
            assert_eq!(a.task, b.task);
            assert_eq!(a.k, b.k);
        }
    }

    #[test]
    fn grads_flow_and_weights_move() {
        use burn::optim::GradientsParams;
        use crate::harness::collate;

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
        let (out, _) = trainer.model.forward_act_with_stats(
            col.tokens,
            &trainer.config,
            &col.lengths,
        );
        let loss = lm_loss(
            out.logits,
            col.targets,
            col.loss_mask,
            out.ponder,
            trainer.config.ponder_weight,
        );
        // Every param must get a nonzero grad. An inverted pad mask zeroes
        // all but the halt grads, so this is the tripwire for that bug class
        // (pad_mask values are pinned in model tests too).
        let mut gp = GradientsParams::from_grads(loss.backward(), &trainer.model);
        let named = trainer.model.grad_specs();
        assert_eq!(named.len(), 14);
        let mut zero = vec![];
        for (name, id, rank) in &named {
            let n: f32 = match rank {
                2 => gp.remove::<burn::backend::NdArray, 2>(*id).unwrap().abs().sum().into_data().as_slice::<f32>().unwrap()[0],
                _ => gp.remove::<burn::backend::NdArray, 1>(*id).unwrap().abs().sum().into_data().as_slice::<f32>().unwrap()[0],
            };
            if n == 0.0 {
                zero.push(*name);
            }
        }
        assert!(zero.is_empty(), "params with zero grad: {zero:?}");
        // Weights must actually move after a step.
        let before = trainer.model.block.attn.q.weight.val().into_data().as_slice::<f32>().unwrap().to_vec();
        let before_head = trainer.model.head.weight.val().into_data().as_slice::<f32>().unwrap().to_vec();
        let refs: Vec<&Instance> = batch.iter().collect();
        trainer.train_step(&refs);
        let after = trainer.model.block.attn.q.weight.val().into_data().as_slice::<f32>().unwrap().to_vec();
        let after_head = trainer.model.head.weight.val().into_data().as_slice::<f32>().unwrap().to_vec();
        assert_ne!(before, after, "muon param frozen after a step");
        assert_ne!(before_head, after_head, "adamw param frozen after a step");
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
            let lens: Vec<usize> =
                batch.iter().map(|i| i.prompt.len() + i.target.len()).collect();
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
        let _loaded = Trainer::<TestBackend>::load_checkpoint(
            &trainer.config,
            &path,
            &test_device(),
        );
        std::fs::remove_file(&path).ok();
    }
}
