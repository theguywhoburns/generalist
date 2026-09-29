use burn::config::Config;

/// How the shared looped block decides depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopMode {
    /// Exactly `loops` applications of the block. No halting head, no ponder.
    Fixed { loops: usize },
    /// Graves-style ACT: per-token halt distribution + ponder penalty.
    /// Only `max_loops` and `ponder_weight` are forced, never the exact depth.
    /// `shuffle_train` is a hint the trainer acts on (it resamples the
    /// execution order each step); the model itself takes the order as an
    /// argument, so the mode itself carries no behaviour.
    Act {
        /// Resample stage order every training step. See [`StopConfig`].
        shuffle_train: bool,
    },
    /// Latent convergence (RD-VLA style): stop when the relative state
    /// change stays below `conv_tol` for `conv_patience` steps.
    Converge,
}

/// Manifest-facing, internally tagged form of [`StopMode`].
///
/// [`StopMode`] is an enum the model dispatches on but which no run
/// manifest could previously reach: training hardcoded `Act` in two places,
/// so the comparison grid's `looped fixed {1,4,8,16}` and converge cells were
/// unreachable without editing Rust. This is that choice, expressed in JSON:
///
/// ```json
/// "stop": { "kind": "act" }
/// "stop": { "kind": "fixed", "loops": 4 }
/// "stop": { "kind": "converge" }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum StopConfig {
    /// Graves-style ACT: learned per-token halting + ponder penalty.
    ///
    /// `shuffle_train` resamples the stage execution order on every training
    /// forward. It is the **causal control** for the role diagnostic: a model
    /// trained under random order either still collapses under reversal
    /// (order-dependence is structural) or does not (order-dependence was
    /// learned, and therefore avoidable). Note the training loss of a
    /// shuffled run is NOT comparable to an ordered one — the model is
    /// solving a different, permutation-robust function.
    ///
    /// Eval always uses the trained order; shuffling is a training-time
    /// intervention only.
    Act {
        /// Resample stage order every training step. See above.
        #[serde(default)]
        shuffle_train: bool,
    },
    /// Exactly `loops` applications of every stage. No halting head, no
    /// ponder, and a constant `block_halts` — the compute-matched control.
    Fixed {
        /// Loop iterations. Must be >= 1; a large value is a depth sweep.
        loops: usize,
    },
    /// Latent convergence: iterate until the relative state change stays
    /// below `LoopedConfig::conv_tol` for `LoopedConfig::conv_patience`
    /// steps. Step count is data-dependent, so cost is not constant.
    Converge,
}

impl StopConfig {
    /// The dispatch value the model takes. `Fixed`'s `loops` and
    /// `Converge`'s tolerances are read back off `LoopedConfig`.
    pub fn to_mode(self) -> StopMode {
        match self {
            StopConfig::Act { shuffle_train } => StopMode::Act { shuffle_train },
            StopConfig::Fixed { loops } => StopMode::Fixed { loops },
            StopConfig::Converge => StopMode::Converge,
        }
    }

    /// Whether training should resample stage order each step. Only ACT
    /// can: `Fixed`/`Converge` are single-stage-execution modes where order
    /// is the whole computation.
    pub fn shuffle_training(&self) -> bool {
        matches!(
            self,
            StopConfig::Act {
                shuffle_train: true
            }
        )
    }

    /// Cross-field checks that `LoopedConfig::assert_valid` cannot see.
    /// Returns every problem, not just the first.
    pub fn validate(&self, cfg: &LoopedConfig) -> Vec<String> {
        let mut errs = Vec::new();
        match *self {
            StopConfig::Fixed { loops } => {
                if loops == 0 {
                    errs.push("stop: fixed loops must be >= 1".to_string());
                } else if loops > cfg.max_loops {
                    // Not fatal, but it silently exceeds the depth the rest of
                    // the config (residual scale, RoPE, budget comments)
                    // was sized for, so say so rather than let it pass.
                    errs.push(format!(
                        "stop: fixed loops {loops} exceeds model.max_loops {}; \
                         residual_scale and the memory budget assume max_loops",
                        cfg.max_loops
                    ));
                }
            }
            // ACT and Converge both read max_loops; the assert in
            // `assert_valid` already covers `max_loops >= 1`.
            StopConfig::Act { .. } | StopConfig::Converge => {}
        }
        if self.shuffle_training() && cfg.n_stages < 2 {
            // With one stage the permutation is the identity, so the flag
            // would silently do nothing while appearing to be a control.
            errs.push(
                "stop: shuffle_train is set but model.n_stages is 1, so the \
                 order permutation is the identity and the control is a no-op"
                    .to_string(),
            );
        }
        errs
    }
}

impl Default for StopConfig {
    /// ACT is the headline configuration, so it is the default a manifest
    /// gets by omitting the key entirely. `shuffle_train` defaults off.
    fn default() -> Self {
        StopConfig::Act {
            shuffle_train: false,
        }
    }
}

/// Option-A base dimensions: `d_model = 256`, byte-level vocab.
///
/// Param budget (untied LM head):
/// `embed 65_536 + attn 262_144 + mlp 589_824 + norms 768 + halt 257 + head 65_536 = 984_065`.
#[derive(Config, Debug)]
pub struct LoopedConfig {
    #[config(default = 256)]
    pub vocab_size: usize,
    #[config(default = 256)]
    pub d_model: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    #[config(default = 64)]
    pub head_dim: usize,
    #[config(default = 768)]
    pub ffn_hidden: usize,
    #[config(default = 8)]
    pub max_loops: usize,
    #[config(default = 1e-3)]
    pub ponder_weight: f64,
    /// Deep-start halt bias (Sapunov 2604.21999: `-3` avoids the shallow-halt trap).
    #[config(default = -3.0)]
    pub halt_bias_init: f64,
    #[config(default = 1e-3)]
    pub conv_tol: f64,
    #[config(default = 2)]
    pub conv_patience: usize,
    #[config(default = 512)]
    pub max_seq_len: usize,
    /// Sequential looped stages (4 stages × 1 block = current headline).
    /// More stages = more independently-halted compute units.
    #[config(default = 1)]
    pub n_stages: usize,
    /// Encoder layers per stage, sharing one gate. 1 = per-block ACT;
    /// all blocks in one stage = global ACT over the stack (ablation axis).
    #[config(default = 1)]
    pub blocks_per_stage: usize,
}

impl LoopedConfig {
    /// Agreed Option-A 1M layout.
    pub fn base_1m() -> Self {
        Self::new()
    }

    pub fn assert_valid(&self) {
        assert_eq!(
            self.d_model,
            self.n_heads * self.head_dim,
            "d_model must equal n_heads * head_dim"
        );
        assert_eq!(self.head_dim % 2, 0, "head_dim must be even for RoPE");
        assert!(self.max_loops >= 1, "max_loops must be >= 1");
    }

    /// Exact parameter count for the untied Option-A layout.
    pub fn param_count(&self) -> usize {
        let d = self.d_model;
        let v = self.vocab_size;
        let h = self.ffn_hidden;
        let n = self.n_stages * self.blocks_per_stage; // total encoder layers
        let embed = v * d;
        let block = 4 * d * d + 3 * d * h + 2 * d; // attn + mlp + 2 norms
        let norms = d; // norm_f
        let halt = self.n_stages * (d + 1); // one halting gate per stage
        let head = v * d; // untied LM head
        embed + n * block + norms + halt + head
    }

    /// Residual branch scale `1 / sqrt(2 * max_loops)`.
    pub fn residual_scale(&self) -> f64 {
        1.0 / (2.0 * self.max_loops as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn option_a_is_1m_class() {
        let cfg = LoopedConfig::base_1m();
        assert_eq!(cfg.param_count(), 984_065);
    }

    #[test]
    fn two_blocks_is_18m_class() {
        let cfg = LoopedConfig::base_1m().with_n_stages(2);
        // Second stage (block + own halt gate): 852_480 + 257.
        assert_eq!(cfg.param_count(), 984_065 + 852_737);
        assert_eq!(cfg.param_count(), 1_836_802);
    }

    #[test]
    fn four_blocks_param_count() {
        let cfg = LoopedConfig::base_1m().with_n_stages(4);
        assert_eq!(cfg.param_count(), 3_542_276);
    }

    #[test]
    fn two_blocks_one_stage_shares_a_gate() {
        // 1 stage × 2 blocks: second block without its own gate.
        let cfg = LoopedConfig::base_1m().with_blocks_per_stage(2);
        assert_eq!(cfg.param_count(), 984_065 + 852_480);
        assert_eq!(cfg.param_count(), 1_836_545);
    }
}
