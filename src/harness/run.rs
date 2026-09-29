//! Run manifest types: what a resolved manifest means.
//!
//! Loading — `extends` resolution, merging, parse errors, validation — lives
//! in [`super::config`]. This file is the schema and the cross-field rules.
//!
//! ```bash
//! cargo run --example run -- configs/stage0-fixed.json
//! ```

use crate::{
    model::{LoopedConfig, StopConfig},
    optim::OptimConfig,
    train::TrainConfig,
};

use super::experiment::Experiment;

/// Everything needed to launch a run, in one file.
///
/// Serialization rules here are load-bearing:
///
/// - `stop` is `#[serde(default)]`, so a manifest that omits it gets ACT.
///   That is the headline configuration and the historical behavior, so
///   older manifests keep working and do not all have to carry a key that
///   says "the default".
/// - The nested `model` / `optim` / `train` / `experiment` blocks are **not**
///   defaulted: burn's `Config` derive emits no `#[serde(default)]`, so a
///   partial block is a parse error by design. Use `extends` to vary one
///   block — the loader merges before parsing, so partial *layers* work even
///   though partial *documents* do not.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunConfig {
    pub model: LoopedConfig,
    /// Optimizer stack. Internally tagged: pick the variant with `"kind"`.
    pub optim: OptimConfig,
    /// How the loop decides depth. Omit for ACT.
    #[serde(default)]
    pub stop: StopConfig,
    pub train: TrainConfig,
    pub experiment: Experiment,
}

impl RunConfig {
    /// Minimal smoke defaults: tiny pools, few steps.
    pub fn smoke() -> Self {
        Self {
            model: LoopedConfig::base_1m(),
            optim: OptimConfig::Muon(crate::optim::MuonTuning::new()),
            stop: StopConfig::default(),
            train: TrainConfig::new(),
            experiment: super::experiment::Experiment {
                per_cell: 8,
                seeds: vec![0],
                ..super::experiment::Experiment::stage0_trio()
            },
        }
    }

    /// Short label for log lines: task count, parameter count, optimizer,
    /// stop mode.
    pub fn summary(&self) -> String {
        format!(
            "{} | {} params | {} | {}",
            self.stop.kind_name(),
            self.model.param_count(),
            self.optim.name(),
            self.stop_summary()
        )
    }

    fn stop_summary(&self) -> String {
        match self.stop {
            StopConfig::Act { .. } => "act".to_string(),
            StopConfig::Fixed { loops } => format!("fixed x{loops}"),
            StopConfig::Converge => "converge".to_string(),
        }
    }

    /// Every cross-field problem, not just the first.
    ///
    /// Reporting in one pass matters here: a manifest that is wrong in three
    /// ways should say so once, before the pool is generated and the model is
    /// built, instead of surfacing a third of the problems after a long run.
    /// Cheap structural asserts (shape agreement) stay as asserts in the
    /// constructors, where they are programmer errors, not user errors.
    pub fn validate(&self) -> Vec<String> {
        let mut problems: Vec<String> = Vec::new();

        // --- model ---
        // `assert_valid` panics; a manifest is user input, so re-check the
        // subset that a user can actually get wrong, without panicking.
        let m = &self.model;
        if m.d_model != m.n_heads * m.head_dim {
            problems.push(format!(
                "model: d_model {} != n_heads {} * head_dim {}",
                m.d_model, m.n_heads, m.head_dim
            ));
        }
        if !m.head_dim.is_multiple_of(2) {
            problems.push(format!(
                "model: head_dim {} must be even for RoPE",
                m.head_dim
            ));
        }
        if m.max_loops == 0 {
            problems.push("model: max_loops must be >= 1".to_string());
        }
        if m.n_stages == 0 {
            problems.push("model: n_stages must be >= 1".to_string());
        }
        if m.blocks_per_stage == 0 {
            problems.push("model: blocks_per_stage must be >= 1".to_string());
        }
        if m.vocab_size < 256 {
            problems.push(format!(
                "model: vocab_size {} < 256; this is a byte-level model",
                m.vocab_size
            ));
        }
        // Sequences are padded UP to a bucket edge, and RoPE is sized to
        // `max_seq_len`. A `max_seq_len` below the top bucket edge therefore
        // breaks as soon as a real sequence lands in the top band — and
        // eval breaks later than training, because decode adds `max_new` on
        // top. Catching it here names the knob instead of surfacing a
        // broadcast panic from inside attention.
        if let Some(top) = crate::harness::BUCKET_EDGES.last()
            && m.max_seq_len < *top
        {
            problems.push(format!(
                "model: max_seq_len {} is below the top padding bucket {top}; \
                 padded batches would exceed it (RoPE is sized to max_seq_len)",
                m.max_seq_len
            ));
        }

        // --- stop vs model ---
        problems.extend(self.stop.validate(m));

        // --- train vs model ---
        let t = &self.train;
        problems.extend(t.lr_muon.validate("lr_muon"));
        problems.extend(t.lr_adamw.validate("lr_adamw"));
        if t.batch_size == 0 {
            problems.push("train: batch_size must be >= 1".to_string());
        }
        if t.accum_steps == 0 {
            problems.push("train: accum_steps must be >= 1".to_string());
        }
        if t.steps == 0 {
            problems.push("train: steps must be >= 1".to_string());
        }
        if t.eval_every == 0 {
            problems.push("train: eval_every must be >= 1 (0 would divide by zero)".to_string());
        }
        if t.log_every == 0 {
            problems.push("train: log_every must be >= 1 (0 would divide by zero)".to_string());
        }
        if t.eval_max_new == 0 {
            problems.push("train: eval_max_new must be >= 1".to_string());
        }
        if t.ckpt_dir.is_empty() {
            problems.push("train: ckpt_dir must not be empty".to_string());
        }
        // Holdout fraction, cross-checked against the pool the run will build.
        if !(0.0..1.0).contains(&t.eval_holdout) {
            problems.push(format!(
                "train: eval_holdout must be in [0, 1), got {}",
                t.eval_holdout
            ));
        } else if t.eval_holdout == 0.0 {
            problems.push(
                "train: eval_holdout is 0, so the eval split is drawn from the \
                 training pool and every reported accuracy is in-distribution; \
                 set a positive value to measure generalization"
                    .to_string(),
            );
        }
        // The holdout must leave enough instances per cell for the final eval's
        // per-block means and correlations to mean anything. Only enforced once
        // `per_cell` is large enough for the question to matter: a smoke
        // manifest exists to prove the pipeline runs, not to measure.
        let cells = (self.experiment.per_cell as f64) * self.experiment.seeds.len() as f64;
        let held_per_cell = t.eval_holdout * cells;
        if t.eval_holdout > 0.0 && self.experiment.per_cell >= 32 && held_per_cell < 8.0 {
            problems.push(format!(
                "train: eval_holdout {} leaves ~{held_per_cell:.1} eval instances per \
                 (task, track, seed) cell; the final-eval correlations and p90s need a \
                 handful to be non-degenerate, so raise per_cell or eval_holdout",
                t.eval_holdout
            ));
        }
        // A warmup longer than the run never reaches full weight: the
        // effective ponder pressure is silently wrong for the whole run.
        if t.ponder_warmup_steps > t.steps {
            problems.push(format!(
                "train: ponder_warmup_steps {} exceeds steps {}; the ponder weight never reaches its configured value",
                t.ponder_warmup_steps, t.steps
            ));
        }

        // --- train vs stop ---
        // Ponder only means something under ACT. A fixed-depth or converge run
        // has a constant step count, so the penalty is a constant offset in
        // the loss and a `ponder_weight` above ~1 just inflates the loss
        // without shaping anything.
        if !matches!(self.stop, StopConfig::Act { .. }) && m.ponder_weight != 0.0 {
            problems.push(format!(
                "train: ponder_weight is set but stop is {}; ponder is constant without ACT, so this only offsets the loss",
                self.stop_summary()
            ));
        }

        // --- experiment ---
        problems.extend(self.experiment.validate());

        problems
    }
}

impl StopConfig {
    /// Label used in log lines and error messages.
    pub fn kind_name(&self) -> &'static str {
        match self {
            StopConfig::Act { .. } => "act",
            StopConfig::Fixed { .. } => "fixed",
            StopConfig::Converge => "converge",
        }
    }
}

/// One experiment in a chain: a manifest plus where its weights come from.
/// `init_from` accepts an explicit checkpoint path or `"$prev"` for the
/// previous experiment's last checkpoint. First experiment uses `null`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlannedExperiment {
    pub name: String,
    /// Path to a run manifest, relative to the plan file. Resolved and
    /// validated at chain-load time, before anything trains.
    pub manifest: String,
    pub init_from: Option<String>,
}

/// Ordered experiments with checkpoint dependencies. After each experiment,
/// the runner re-evaluates all earlier experiments' splits (forgetting
/// checks). Explicit paths allow DAGs, not just chains.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExperimentPlan {
    pub experiments: Vec<PlannedExperiment>,
}

impl ExperimentPlan {
    /// Sentinel `init_from` meaning "the previous experiment's final
    /// checkpoint".
    pub const PREV: &'static str = "$prev";
}

#[cfg(test)]
mod manifest_tests {
    use super::*;
    use crate::harness::config::{divergent_keys, load_run, manifest_names, resolve};
    use std::path::{Path, PathBuf};

    fn dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("configs")
    }

    fn read(name: &str) -> String {
        std::fs::read_to_string(dir().join(name))
            .unwrap_or_else(|e| panic!("read configs/{name}: {e}"))
    }

    /// Every run manifest in `configs/` must load, resolve and validate.
    /// Discovers files by walking the directory: adding a manifest adds it to
    /// this test automatically, with no name list to keep in sync.
    #[test]
    fn every_run_manifest_loads_and_validates() {
        let names = manifest_names(&dir()).expect("read configs/");
        let mut checked = 0;
        for name in &names {
            let text = read(name);
            if super::super::config::is_chain_plan(&text) {
                continue;
            }
            load_run(&dir().join(name))
                .unwrap_or_else(|e| panic!("configs/{name} failed to load:\n{e}"));
            checked += 1;
        }
        assert!(checked >= 4, "only {checked} run manifests found");
    }

    /// Extending a base must be indistinguishable from writing the base out.
    /// This is the property that makes `extends` safe: a child that overrides
    /// nothing must equal its parent, and a child that overrides a leaf must
    /// change only that leaf.
    #[test]
    fn extends_children_equal_their_merged_base() {
        for name in manifest_names(&dir()).expect("read configs/") {
            let text = read(&name);
            if super::super::config::is_chain_plan(&text) {
                continue;
            }
            let path = dir().join(&name);
            let (merged, _) = resolve(&path, &mut Vec::new())
                .unwrap_or_else(|e| panic!("configs/{name} did not resolve:\n{e}"));
            let from_file =
                load_run(&path).unwrap_or_else(|e| panic!("configs/{name} did not load:\n{e}"));

            // Resolution must be deterministic, or `extends` is doing
            // something invisible.
            let expect = resolve(&path, &mut Vec::new()).unwrap().0;
            assert_eq!(
                merged, expect,
                "configs/{name}: resolve is not deterministic"
            );

            // The right invariant for `save_run` is a SEMANTIC fixed point,
            // not document equality. A checked-in manifest may omit a
            // serde-defaulted key (`stop.shuffle_train`, `stop` itself), while
            // serialization always writes it — so comparing the saved
            // document to the input would conflate "a default was omitted"
            // with "resolution is wrong". Compare typed values instead:
            // save -> load -> save must be stable, and the value must equal
            // the one loaded from the original file.
            let saved = serde_json::to_value(&from_file).expect("serialize");
            let reloaded: RunConfig = serde_json::from_value(saved.clone()).expect("saved parses");
            assert_eq!(
                serde_json::to_value(&reloaded).expect("re-serialize"),
                saved,
                "configs/{name}: save -> load -> save is not a fixed point"
            );
            assert_eq!(
                saved["model"],
                serde_json::to_value(&from_file.model).expect("model"),
                "configs/{name}: model block changed across the round trip"
            );
            // And the saved document must be loadable on its own, with no base
            // file present: that is the "self-contained" promise.
            assert!(
                !saved.as_object().expect("object").contains_key("extends"),
                "configs/{name}: saved output kept an extends key"
            );
        }
    }

    /// A child must override only what it names. Concretely: the set of
    /// top-level keys a child touches is small and intentional, and its
    /// distinct-from-base values are exactly the ones it wrote.
    #[test]
    fn stage0_children_vary_only_their_intended_keys() {
        let base = serde_json::from_str::<serde_json::Value>(&read("stage0-base.json"))
            .expect("base is JSON");
        for child in [
            "stage0-4block.json",
            "stage0-fixed.json",
            "stage0-oracle.json",
            "stage0-maponly.json",
            "stage0-fixed4.json",
            "stage0-converge.json",
        ] {
            let v = serde_json::from_str::<serde_json::Value>(&read(child))
                .unwrap_or_else(|e| panic!("configs/{child}: {e}"));
            // Every child declares its parent, and is otherwise small: the
            // point of the refactor is that variations are cheap to write.
            assert!(
                v.get("extends").is_some(),
                "{child} does not extend the base"
            );
            let own: Vec<&String> = v
                .as_object()
                .unwrap()
                .keys()
                .filter(|k| *k != "extends" && *k != "_comment")
                .collect();
            assert!(
                own.len() <= 4,
                "configs/{child} overrides {} top-level keys ({own:?}); it should be a thin variation",
                own.len()
            );
            // And the values it does set must actually differ from the base,
            // or the override is dead weight.
            let diff = divergent_keys(&[base.clone(), v]);
            assert!(!diff.is_empty(), "configs/{child} changes nothing");
        }
    }

    /// The fixed-depth control must differ from the ACT base in the two ways
    /// that make it a control, and the base must remain ACT.
    #[test]
    fn stop_mode_variations_are_wired_through() {
        let base = load_run(&dir().join("stage0-base.json")).expect("base loads");
        assert_eq!(base.stop, StopConfig::default());

        let fixed = load_run(&dir().join("stage0-fixed4.json")).expect("fixed4 loads");
        assert_eq!(fixed.stop, StopConfig::Fixed { loops: 4 });
        // Fixed depth has no learned step count, so a ponder penalty would be
        // a constant loss offset. The manifest must zero it.
        assert_eq!(
            fixed.model.ponder_weight, 0.0,
            "fixed4 kept a ponder weight"
        );

        let conv = load_run(&dir().join("stage0-converge.json")).expect("converge loads");
        assert_eq!(conv.stop, StopConfig::Converge);
    }

    /// The comparison grid's headline requirement: fixed-depth and converge
    /// runs must be reachable from a manifest at all. They were not before
    /// `StopConfig` existed.
    #[test]
    fn all_three_stop_modes_are_reachable_from_manifests() {
        let mut seen = vec![];
        for name in manifest_names(&dir()).expect("read configs/") {
            let text = read(&name);
            if super::super::config::is_chain_plan(&text) {
                continue;
            }
            seen.push(load_run(&dir().join(&name)).expect("loads").stop);
        }
        assert!(
            seen.contains(&StopConfig::default()),
            "no manifest uses act"
        );
        assert!(
            seen.iter().any(|s| matches!(s, StopConfig::Fixed { .. })),
            "no manifest uses fixed depth; that grid cell is unreachable"
        );
        assert!(
            seen.contains(&StopConfig::Converge),
            "no manifest uses converge; that grid cell is unreachable"
        );
    }

    /// The checked-in chain must load, and every manifest it names must
    /// resolve and validate. Chain plans do not extend, but their manifests
    /// do — so a broken grandchild is caught here.
    #[test]
    fn checked_in_chain_loads_and_every_manifest_resolves() {
        use crate::harness::config::load_chain;
        let path = dir().join("chain.json");
        let (plan, runs) =
            load_chain(&path).unwrap_or_else(|e| panic!("configs/chain.json failed to load:\n{e}"));
        assert!(!plan.experiments.is_empty());
        assert_eq!(plan.experiments.len(), runs.len());
        for ((name, cfg), exp) in runs.iter().zip(&plan.experiments) {
            assert_eq!(name, &exp.name);
            assert!(cfg.validate().is_empty(), "{name} invalid");
        }
        // The first stage must start from scratch.
        assert!(plan.experiments[0].init_from.is_none());
        // Later stages either continue or are explicit about their origin.
        for exp in &plan.experiments[1..] {
            match exp.init_from.as_deref() {
                Some(ExperimentPlan::PREV) | None => {}
                Some(p) => panic!("{}: unchecked explicit init_from {p}", exp.name),
            }
        }
    }

    /// The chain runs bottom-of-staircase first: the weights-only fixed-map
    /// rung must precede the oracle and ICL rungs, since each chains off the
    /// last. An ordering that put the ICL headline first would train the
    /// hardest task from random weights and make the later rungs meaningless.
    #[test]
    fn chain_orders_stages_substrate_first() {
        use crate::harness::config::load_chain;
        let (_, runs) = load_chain(&dir().join("chain.json")).expect("chain loads");
        let first = runs[0].1.experiment.tasks.clone();
        assert!(
            first.iter().any(|t| t == "subst-fst-fixed"),
            "chain does not start at the fixed-map rung: {first:?}"
        );
    }

    /// A broken manifest anywhere in a chain must fail before any stage
    /// trains — stage 3's typo must not surface after stage 2 has run.
    #[test]
    fn chain_validates_upfront_not_stage_by_stage() {
        let bad = std::env::temp_dir().join("generalist-chain-bad");
        std::fs::create_dir_all(&bad).unwrap();
        // A valid minimal manifest, then one with a guaranteed validation error.
        let good = serde_json::to_value(RunConfig::smoke()).unwrap();
        std::fs::write(bad.join("good.json"), good.to_string()).unwrap();
        let mut broken = good;
        broken["model"]["n_stages"] = serde_json::json!(0);
        std::fs::write(bad.join("broken.json"), broken.to_string()).unwrap();
        std::fs::write(
            bad.join("plan.json"),
            r#"{"experiments":[
                {"name":"a","manifest":"good.json","init_from":null},
                {"name":"b","manifest":"broken.json","init_from":"$prev"}
            ]}"#,
        )
        .unwrap();
        // Fails on the second stage's manifest, up front, naming it.
        match crate::harness::config::load_chain(&bad.join("plan.json")) {
            Err(crate::harness::ConfigError::Invalid { path, problems }) => {
                assert!(path.ends_with("broken.json"), "blamed {path:?}");
                assert!(
                    problems.iter().any(|p| p.contains("n_stages")),
                    "{problems:?}"
                );
            }
            other => panic!("expected Invalid for the broken stage, got {other:?}"),
        }
    }

    /// Every manifest must carry a warmup or constant LR deliberately, and
    /// the base's ramp must be consistent with its own step count.
    #[test]
    fn learning_rate_schedules_are_consistent_with_step_counts() {
        for name in manifest_names(&dir()).expect("read configs/") {
            let text = read(&name);
            if super::super::config::is_chain_plan(&text) {
                continue;
            }
            let cfg = load_run(&dir().join(&name)).expect("loads");
            if let crate::optim::LrConfig::Linear { warmup_steps, .. } = cfg.train.lr_muon {
                assert!(
                    warmup_steps < cfg.train.steps,
                    "configs/{name}: warmup_steps {warmup_steps} is not shorter than steps {}",
                    cfg.train.steps
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::Track;

    fn valid() -> RunConfig {
        let mut c = RunConfig::smoke();
        c.model.ponder_weight = 0.0;
        c
    }

    #[test]
    fn a_smoke_config_is_valid() {
        assert_eq!(valid().validate(), Vec::<String>::new());
    }

    #[test]
    fn omitted_stop_defaults_to_act() {
        // The backwards-compatibility guarantee: a manifest with no `stop`
        // key deserializes and means ACT.
        let cfg: RunConfig =
            serde_json::from_str(&serde_json::to_string(&valid()).unwrap()).unwrap();
        assert_eq!(cfg.stop, StopConfig::default());
        // And a document with the key absent parses to Act, not an error.
        let mut v = serde_json::to_value(valid()).unwrap();
        v.as_object_mut().unwrap().remove("stop");
        let back: RunConfig = serde_json::from_value(v).expect("omit stop");
        assert_eq!(back.stop, StopConfig::default());
    }

    #[test]
    fn every_variant_roundtrips_through_json() {
        for stop in [
            StopConfig::default(),
            StopConfig::Fixed { loops: 4 },
            StopConfig::Converge,
        ] {
            let mut c = valid();
            c.stop = stop;
            let text = serde_json::to_string(&c).unwrap();
            let back: RunConfig = serde_json::from_str(&text).unwrap();
            assert_eq!(back.stop, stop, "stop did not roundtrip: {stop:?}");
            assert_eq!(back.validate(), Vec::<String>::new());
        }
    }

    #[test]
    fn stop_tag_is_readable_in_json() {
        let mut c = valid();
        c.stop = StopConfig::Fixed { loops: 8 };
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["stop"]["kind"], json_str("fixed"));
        assert_eq!(v["stop"]["loops"], json_num(8));
    }

    #[test]
    fn all_problems_are_reported_at_once() {
        // The point of validate(): a manifest wrong in several ways says so
        // in one pass, not one error per run.
        let mut c = valid();
        c.model.d_model = 100; // != 2 * 16
        c.model.n_stages = 0;
        c.train.batch_size = 0;
        c.train.eval_every = 0;
        c.train.accum_steps = 0;
        let problems = c.validate();
        assert!(
            problems.len() >= 5,
            "expected many problems, got {problems:?}"
        );
        let joined = problems.join("\n");
        for needle in [
            "d_model",
            "n_stages",
            "batch_size",
            "eval_every",
            "accum_steps",
        ] {
            assert!(joined.contains(needle), "missing {needle} in {joined}");
        }
    }

    #[test]
    fn fixed_loops_above_max_loops_is_flagged() {
        let mut c = valid();
        c.model.max_loops = 4;
        c.stop = StopConfig::Fixed { loops: 8 };
        let joined = c.validate().join("\n");
        assert!(joined.contains("exceeds model.max_loops"), "{joined}");
    }

    #[test]
    fn fixed_loops_zero_is_flagged() {
        let mut c = valid();
        c.stop = StopConfig::Fixed { loops: 0 };
        assert!(c.validate().join("\n").contains("fixed loops must be >= 1"));
    }

    #[test]
    fn ponder_weight_under_non_act_is_flagged() {
        // Ponder is a constant offset without ACT, so a weight is almost
        // certainly a stale leftover from a previous manifest.
        for stop in [StopConfig::Fixed { loops: 2 }, StopConfig::Converge] {
            let mut c = valid();
            c.stop = stop;
            c.model.ponder_weight = 1e-3;
            let joined = c.validate().join("\n");
            assert!(
                joined.contains("ponder_weight is set but stop is"),
                "not flagged for {stop:?}: {joined}"
            );
        }
    }

    #[test]
    fn warmup_longer_than_the_run_is_flagged() {
        let mut c = valid();
        c.train.steps = 10;
        c.train.ponder_warmup_steps = 100;
        assert!(
            c.validate()
                .join("\n")
                .contains("never reaches its configured value")
        );
    }

    #[test]
    fn summary_names_the_stop_mode() {
        let mut c = valid();
        c.stop = StopConfig::Fixed { loops: 4 };
        let s = c.summary();
        assert!(s.contains("fixed x4"), "{s}");
        assert!(s.contains("muon"), "{s}");
        assert!(s.contains("params"), "{s}");
    }

    #[test]
    fn experiment_validation_catches_empty_task_lists() {
        let mut c = valid();
        c.experiment.tasks.clear();
        c.experiment.tracks = vec![Track::A, Track::B];
        assert!(c.validate().join("\n").contains("tasks"));
    }

    fn json_str(s: &str) -> Value {
        Value::String(s.to_string())
    }
    fn json_num(n: usize) -> Value {
        Value::from(n)
    }
    use serde_json::Value;
}
