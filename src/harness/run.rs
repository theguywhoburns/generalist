//! Run manifest: one JSON file dispatches a whole experiment.
//! Tweak values here; no recompile needed.
//!
//! ```bash
//! cargo run --example run -- configs/smoke.json
//! ```

use std::path::Path;

use crate::{
    model::LoopedConfig,
    optim::NewtonMuonConfig,
    train::TrainConfig,
};

use super::experiment::Experiment;

/// Everything needed to launch a run, in one file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunConfig {
    pub model: LoopedConfig,
    pub optim: NewtonMuonConfig,
    pub train: TrainConfig,
    pub experiment: Experiment,
}

impl RunConfig {
    /// Minimal smoke defaults: tiny pools, few steps.
    pub fn smoke() -> Self {
        Self {
            model: LoopedConfig::base_1m(),
            optim: NewtonMuonConfig::new(),
            train: TrainConfig::new(),
            experiment: super::experiment::Experiment {
                per_cell: 8,
                seeds: vec![0],
                ..super::experiment::Experiment::stage0_trio()
            },
        }
    }

    pub fn load_json(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&text).map_err(|e| e.to_string())
    }

    pub fn save_json(&self, path: &Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| e.to_string())
    }
}

/// One experiment in a chain: a manifest plus where its weights come from.
/// `init_from` accepts an explicit checkpoint path or `"$prev"` for the
/// previous experiment's last checkpoint. First experiment uses `null`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlannedExperiment {
    pub name: String,
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
    pub fn load_json(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&text).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_roundtrip() {
        let cfg = RunConfig::smoke();
        let text = serde_json::to_string(&cfg).unwrap();
        let back: RunConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back.experiment.per_cell, 8);
        assert_eq!(back.train.batch_size, cfg.train.batch_size);
        assert_eq!(back.model.d_model, 256);
    }

    #[test]
    fn all_checked_in_manifests_load() {
        // Every RunConfig manifest in configs/ must parse. No names are
        // hardcoded: deleting or adding a manifest updates this test
        // automatically. Chain plans (*chain.json) parse as ExperimentPlan.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("configs");
        let mut run_count = 0;
        let mut chain_count = 0;
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".json"))
            .collect();
        names.sort();
        for name in &names {
            let path = dir.join(name);
            let text = std::fs::read_to_string(&path).unwrap();
            if name.contains("chain") {
                let plan: super::ExperimentPlan = serde_json::from_str(&text).unwrap();
                assert!(!plan.experiments.is_empty(), "{name} has no experiments");
                chain_count += 1;
            } else {
                assert!(
                    text.contains("ponder_warmup_steps"),
                    "{name} missing ponder_warmup_steps key"
                );
                let cfg = RunConfig::load_json(&path).unwrap();
                assert!(cfg.model.param_count() > 0, "{name} has no params");
                assert!(!cfg.experiment.tasks.is_empty(), "{name} has no tasks");
                run_count += 1;
            }
        }
        assert!(run_count > 0 && chain_count > 0, "configs/ walk found nothing");
    }

    #[test]
    fn checked_in_chain_loads() {
        use super::ExperimentPlan;
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("configs/chain.json");
        let chain = ExperimentPlan::load_json(&path).unwrap();
        assert_eq!(chain.experiments.len(), 1);
        assert!(chain.experiments[0].init_from.is_none());
    }

    #[test]
    fn burn_configs_are_json_serde() {
        // Burn Config derive already implements Serialize/Deserialize.
        let m = LoopedConfig::base_1m();
        let text = serde_json::to_string(&m).unwrap();
        let back: LoopedConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back.param_count(), m.param_count());
    }
}
