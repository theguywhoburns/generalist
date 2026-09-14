//! Experiment dispatch: declarative config in, instances out.
//! Model training/eval consumes the instances; results come back as Records.

use crate::tasks::{DemoProtocol, HarnessRng, Instance, TaskRegistry, Track};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Experiment {
    /// Task names (registry keys). Empty = all builtin.
    pub tasks: Vec<String>,
    pub tracks: Vec<Track>,
    /// Instances per (task, track, seed) cell.
    pub per_cell: usize,
    pub seeds: Vec<u64>,
    pub protocol: DemoProtocol,
}

impl Default for Experiment {
    fn default() -> Self {
        Self {
            tasks: vec![],
            tracks: vec![Track::A, Track::B],
            per_cell: 256,
            seeds: vec![0, 1, 2],
            protocol: DemoProtocol::default(),
        }
    }
}

impl Experiment {
    pub fn stage0_trio() -> Self {
        Self {
            tasks: vec![
                "parity".to_string(),
                "dyck1".to_string(),
                "subst-fst".to_string(),
                "periodic".to_string(),
                "copy-rev-rep".to_string(),
                "scan-tiny".to_string(),
            ],
            ..Default::default()
        }
    }
}

/// Generate every instance for the experiment. Deterministic in
/// (task, track, seed, index): reseeding per cell keeps cells reproducible
/// regardless of execution order.
pub fn generate(cfg: &Experiment, registry: &TaskRegistry) -> Vec<Instance> {
    let names: Vec<String> = if cfg.tasks.is_empty() {
        registry.names().iter().map(|s| s.to_string()).collect()
    } else {
        cfg.tasks.clone()
    };
    let mut out = Vec::new();
    for name in &names {
        let task = registry
            .get(name)
            .unwrap_or_else(|| panic!("unknown task: {name}"));
        for track in &cfg.tracks {
            for seed in &cfg.seeds {
                let mut rng = HarnessRng::new(mix(name, *track, *seed));
                for _ in 0..cfg.per_cell {
                    let rule = task.sample_rule(&mut rng, *track);
                    out.push(cfg.protocol.build(
                        task.name(),
                        task.stage(),
                        *track,
                        rule.as_ref(),
                        &mut rng,
                    ));
                }
            }
        }
    }
    out
}

fn mix(name: &str, track: Track, seed: u64) -> u64 {
    let mut h: u64 = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for b in name.bytes() {
        h = h.wrapping_mul(31).wrapping_add(b as u64);
    }
    h.wrapping_add(match track {
        Track::A => 0xA,
        Track::B => 0xB,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_covers_cells_evenly() {
        let registry = TaskRegistry::builtin();
        let cfg = Experiment {
            per_cell: 16,
            seeds: vec![0, 1],
            ..Experiment::stage0_trio()
        };
        let data = generate(&cfg, &registry);
        // 6 tasks x 2 tracks x 2 seeds x 16
        assert_eq!(data.len(), 384);
        for inst in &data {
            assert!(!inst.prompt.is_empty() && !inst.target.is_empty());
            assert!(inst.info.k <= 8);
        }
    }

    #[test]
    fn generation_is_deterministic() {
        let registry = TaskRegistry::builtin();
        let cfg = Experiment::stage0_trio();
        let a = generate(&cfg, &registry);
        let b = generate(&cfg, &registry);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.prompt, y.prompt);
            assert_eq!(x.target, y.target);
        }
    }

    #[test]
    #[should_panic(expected = "unknown task")]
    fn unknown_task_panics() {
        let registry = TaskRegistry::builtin();
        let cfg = Experiment {
            tasks: vec!["nope".to_string()],
            ..Default::default()
        };
        let _ = generate(&cfg, &registry);
    }
}
