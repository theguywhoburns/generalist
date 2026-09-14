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
    harness::{Record as MetricRecord, collate},
    model::{LoopedConfig, LoopedTransformer, StopMode, lm_loss},
    optim::{NewtonMuon, NewtonMuonConfig, PrecondInput, precondition_grads, split_grads},
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
    /// Whole-process stall watchdog: kill the run if no step completes
    /// within this many seconds (catches hangs no panic hook can see).
    #[config(default = 900)]
    pub stuck_timeout_secs: u64,
    #[config(default = "\"checkpoints\".to_string()")]
    pub ckpt_dir: String,
}

pub struct StepInfo {
    pub loss: f32,
    pub ponder: f32,
    pub steps_used: usize,
    pub mean_halt: f32,
}

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
        }
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

    pub fn train_step(&mut self, batch: &[&Instance]) -> StepInfo {
        let owned: Vec<Instance> = batch.iter().map(|i| (*i).clone()).collect();
        let col = collate(&owned, &self.device);
        let t = col.lengths.iter().max().copied().unwrap_or(0);
        assert!(
            t <= self.config.max_seq_len,
            "batch length {t} exceeds max_seq_len {}: raise max_seq_len or shorten instances",
            self.config.max_seq_len,
        );
        let (out, stats) = self
            .model
            .forward_act_with_stats(col.tokens, &self.config, &col.lengths);
        let loss = lm_loss(
            out.logits,
            col.targets,
            col.loss_mask,
            out.ponder.clone(),
            self.config.ponder_weight,
        );
        let loss_val = scalar_of(&loss);
        let ponder_val = scalar_of(&out.ponder);
        let steps_used = out.steps_used;
        let mean_halt = out.mean_halt;

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        let (muon_grads, adamw_grads) = split_grads::<B>(&self.muon_ids, grads);        self.precond.observe_stats(&stats);
        self.precond.maybe_refresh();
        let (muon_grads, leftover) =
            precondition_grads::<B>(&self.precond, &self.roles, muon_grads);
        assert!(leftover.is_empty(), "muon grad without precond role");

        let model = self.model.clone();
        self.model = self.muon.step(self.lr_muon, model, muon_grads);
        let model = self.model.clone();
        self.model = self.adamw.step(self.lr_adamw, model, adamw_grads);

        StepInfo { loss: loss_val, ponder: ponder_val, steps_used, mean_halt }
    }

    /// Greedy decode to EOS: exact-match, copy flag, loop stats per instance.
    /// Decodes on the inner (inference) backend: eval forwards must not
    /// register nodes on the global autodiff tape, which is reclaimed only
    /// by `backward()` — un-backwarded eval graphs pile up (~80MB/instance
    /// at 1M scale) and OOM both RAM and VRAM.
    pub fn evaluate(&self, instances: &[Instance], max_new: usize) -> Vec<MetricRecord> {
        let model = self.model.valid();
        instances
            .iter()
            .map(|inst| {
                let mut ids = inst.prompt_ids();
                let mut steps_sum = 0usize;
                let mut halt_sum = 0.0f32;
                let mut n_decode = 0usize;
                let mut out_bytes: Vec<u8> = vec![];
                loop {
                    let t = ids.len();
                    if t >= self.config.max_seq_len {
                        break;
                    }
                    // Bucket the physical length: decode grows t by 1 per
                    // step, which would recompile fused kernels every step.
                    let tb = crate::harness::bucket_len(t);
                    let mut padded = ids.clone();
                    padded.resize(tb, crate::harness::EOS as i64);
                    let tokens = Tensor::<B::InnerBackend, 2, Int>::from_data(
                        TensorData::new(padded, [1, tb]),
                        &self.device,
                    );
                    let out = model.forward(tokens, &self.config, StopMode::Act, &[t]);
                    steps_sum += out.steps_used;
                    halt_sum += out.mean_halt;
                    n_decode += 1;
                    let v = self.config.vocab_size;
                    let next = out.logits.slice([0..1, t - 1..t, 0..v]).reshape([v]).argmax(0);
                    let next_id = int_scalar(&next.into_data());
                    if next_id == crate::harness::EOS as i64 || out_bytes.len() >= max_new {
                        break;
                    }
                    ids.push(next_id);
                    out_bytes.push(next_id as u8);
                }
                let text = String::from_utf8_lossy(&out_bytes);
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
        let batch =
            Trainer::<B>::sample_banded_batch(&mut rng, &train_pool, run.train.batch_size);
        let info = trainer.train_step(&batch);
        watchdog.ping_step(step);
        if step % run.train.log_every == 0 {
            println!(
                "step {step:>5} loss {:.4} ponder {:.2} loops {} halt {:.2}",
                info.loss, info.ponder, info.steps_used, info.mean_halt
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
    match data.dtype {
        burn::tensor::DType::I64 => data.as_slice::<i64>().unwrap()[0],
        burn::tensor::DType::I32 => data.as_slice::<i32>().unwrap()[0] as i64,
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
        let named: Vec<(&str, burn::module::ParamId, usize)> = vec![
            ("embed", trainer.model.embed.weight.id, 2),
            ("q", trainer.model.block.attn.q.weight.id, 2),
            ("k", trainer.model.block.attn.k.weight.id, 2),
            ("v", trainer.model.block.attn.v.weight.id, 2),
            ("o", trainer.model.block.attn.o.weight.id, 2),
            ("gate", trainer.model.block.mlp.gate.weight.id, 2),
            ("up", trainer.model.block.mlp.up.weight.id, 2),
            ("down", trainer.model.block.mlp.down.weight.id, 2),
            ("norm1", trainer.model.block.norm1.gamma.id, 1),
            ("norm2", trainer.model.block.norm2.gamma.id, 1),
            ("norm_f", trainer.model.norm_f.gamma.id, 1),
            ("halt_w", trainer.model.halt.head.weight.id, 2),
            ("halt_b", trainer.model.halt.head.bias.as_ref().unwrap().id, 1),
            ("head", trainer.model.head.weight.id, 2),
        ];
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
