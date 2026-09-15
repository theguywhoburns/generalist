use burn::config::Config;

/// How the shared looped block decides depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopMode {
    /// Exactly `loops` applications of the block. No halting head, no ponder.
    Fixed { loops: usize },
    /// Graves-style ACT: per-token halt distribution + ponder penalty.
    /// Only `max_loops` and `ponder_weight` are forced, never the exact depth.
    Act,
    /// Latent convergence (RD-VLA style): stop when the relative state
    /// change stays below `conv_tol` for `conv_patience` steps.
    Converge,
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
    /// Stacked distinct blocks looped as one unit (1 = headline config).
    /// 2 blocks ≈ 1.84M params; the comparison axis for width-vs-depth.
    #[config(default = 1)]
    pub n_blocks: usize,
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
        let embed = v * d;
        let block = 4 * d * d + 3 * d * h + 2 * d; // attn + mlp + 2 norms
        let norms = d; // norm_f
        let halt = self.n_blocks * (d + 1); // one halting gate per block
        let head = v * d; // untied LM head
        embed + self.n_blocks * block + norms + halt + head
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
        let cfg = LoopedConfig::base_1m().with_n_blocks(2);
        // Second block (attn+mlp+norms+own halt gate): 852_480 + 257.
        assert_eq!(cfg.param_count(), 984_065 + 852_737);
        assert_eq!(cfg.param_count(), 1_836_802);
    }

    #[test]
    fn four_blocks_param_count() {
        let cfg = LoopedConfig::base_1m().with_n_blocks(4);
        assert_eq!(cfg.param_count(), 3_542_276);
    }
}
