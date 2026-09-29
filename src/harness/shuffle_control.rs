//! The stage-order shuffle diagnostic, read against its own control.
//!
//! # Why a control is required
//!
//! The shuffle diagnostic reverses stage order and checks whether accuracy
//! collapses. Collapse is **guaranteed by construction** and therefore means
//! nothing on its own: stages run sequentially and each stage's readout is the
//! next stage's input, so reversal reverses the data flow. The same collapse
//! would appear for random weights.
//!
//! So the trained-vs-shuffled gap is only interpretable against a floor
//! measured the same way. Both models are evaluated on the same split, in both
//! orders:
//!
//! ```text
//!   trained_order_acc  - trained_shuffled_acc    (observed collapse)
//!   control_order_acc  - control_shuffled_acc    (collapse from reversal alone)
//! ```
//!
//! The quantity carrying meaning is the **excess**: the trained gap minus the
//! control gap. If the two are close, the stages are order-interchangeable and
//! the "learned roles" reading is unsupported. Only a trained gap materially
//! larger than the floor is evidence of specialization — the trained model
//! loses *more* from reversal than an untrained one does.
//!
//! The threshold is deliberately conservative. This is a claim about
//! mechanism, and the control gap is itself estimated from a finite eval
//! split, so a small positive excess is not evidence.

/// Minimum excess collapse, in accuracy points, before the role claim is
/// supported. Below this the control gap and the trained gap are
/// indistinguishable at the resolution of a finite eval split.
pub const SHUFFLE_EXCESS_THRESHOLD: f64 = 0.10;

/// The four accuracies behind the diagnostic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShuffleControl {
    /// Trained model, trained stage order.
    pub trained_order: f64,
    /// Trained model, reversed stage order.
    pub trained_shuffled: f64,
    /// Random-weight control, trained stage order.
    pub control_order: f64,
    /// Random-weight control, reversed stage order.
    pub control_shuffled: f64,
}

impl ShuffleControl {
    pub fn new(t: f64, ts: f64, c: f64, cs: f64) -> Self {
        Self {
            trained_order: t,
            trained_shuffled: ts,
            control_order: c,
            control_shuffled: cs,
        }
    }

    /// Accuracy lost by reversing stage order.
    pub fn trained_gap(&self) -> f64 {
        self.trained_order - self.trained_shuffled
    }

    /// Accuracy lost by reversal with no training at all: the floor.
    pub fn control_gap(&self) -> f64 {
        self.control_order - self.control_shuffled
    }

    /// Collapse attributable to learned roles, i.e. trained gap minus floor.
    /// Near zero means the observed collapse was structural.
    pub fn excess_collapse(&self) -> f64 {
        self.trained_gap() - self.control_gap()
    }

    /// Whether the data supports "the stages learned distinct roles".
    pub fn supports_role_specialization(&self) -> bool {
        self.excess_collapse() > SHUFFLE_EXCESS_THRESHOLD
    }

    /// Machine-readable JSONL record, so a sweep can aggregate the verdict
    /// without scraping the human-readable line.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"eval\":\"shuffle-control\",\"trained_order\":{:.6},\"trained_shuffled\":{:.6},\
             \"control_order\":{:.6},\"control_shuffled\":{:.6},\"trained_gap\":{:.6},\
             \"control_gap\":{:.6},\"excess_collapse\":{:.6},\"roles_supported\":{}}}",
            self.trained_order,
            self.trained_shuffled,
            self.control_order,
            self.control_shuffled,
            self.trained_gap(),
            self.control_gap(),
            self.excess_collapse(),
            self.supports_role_specialization(),
        )
    }
}

impl std::fmt::Display for ShuffleControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "trained {:.3}->{:.3} (gap {:+.3}) | control {:.3}->{:.3} (gap {:+.3}) \
             | excess {:+.3} | roles {}",
            self.trained_order,
            self.trained_shuffled,
            self.trained_gap(),
            self.control_order,
            self.control_shuffled,
            self.control_gap(),
            self.excess_collapse(),
            if self.supports_role_specialization() {
                "SUPPORTED"
            } else {
                "NOT supported (collapse is structural or within noise)"
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The case the control exists to catch: a big trained gap that is
    /// entirely explained by reversal mechanics.
    #[test]
    fn structural_collapse_is_not_reported_as_specialization() {
        // Both models start near chance and fall to ~0 when reversed. The
        // trained gap looks dramatic (0.6) but the floor is 0.59 too.
        let c = ShuffleControl::new(0.60, 0.00, 0.60, 0.01);
        assert!((c.trained_gap() - 0.60).abs() < 1e-9);
        assert!((c.control_gap() - 0.59).abs() < 1e-9);
        assert!((c.excess_collapse() - 0.01).abs() < 1e-9);
        assert!(
            !c.supports_role_specialization(),
            "structural collapse was reported as learned roles"
        );
    }

    /// The case that should be supported: the trained model loses much more
    /// from reversal than an untrained one does.
    #[test]
    fn excess_collapse_above_threshold_supports_roles() {
        let c = ShuffleControl::new(0.80, 0.10, 0.20, 0.15);
        assert!((c.trained_gap() - 0.70).abs() < 1e-9);
        assert!((c.control_gap() - 0.05).abs() < 1e-9);
        assert!((c.excess_collapse() - 0.65).abs() < 1e-9);
        assert!(c.supports_role_specialization());
    }

    #[test]
    fn threshold_is_strict() {
        // Note: the exact boundary is not testable in f64 — 0.60 - 0.20 is
        // 0.39999999999999997, so an "exactly at threshold" case lands an ULP
        // either side of it. Assert unambiguously below and above instead;
        // the comparison itself is strict (`>`), so equality is not support.
        // trained_gap 0.50, control_gap 0.45 -> excess 0.05, below 0.10.
        let below = ShuffleControl::new(0.50, 0.00, 0.65, 0.20);
        assert!((below.excess_collapse() - 0.05).abs() < 1e-9);
        assert!(!below.supports_role_specialization());

        // control_gap 0.38 -> excess 0.12, above 0.10.
        let above = ShuffleControl::new(0.50, 0.00, 0.60, 0.22);
        assert!((above.excess_collapse() - 0.12).abs() < 1e-9);
        assert!(above.supports_role_specialization());
    }

    /// An order-agnostic model collapses by the same amount trained and
    /// untrained: the canonical "no roles learned" result.
    #[test]
    fn order_agnostic_model_reads_as_no_roles() {
        // Both gaps 0.50, so the excess is exactly zero.
        let c = ShuffleControl::new(0.70, 0.20, 0.55, 0.05);
        assert!((c.trained_gap() - 0.50).abs() < 1e-9);
        assert!((c.control_gap() - 0.50).abs() < 1e-9);
        assert!((c.excess_collapse() - 0.0).abs() < 1e-9);
        assert!(!c.supports_role_specialization());
    }

    #[test]
    fn json_carries_every_field_a_sweep_needs() {
        let c = ShuffleControl::new(0.8, 0.1, 0.2, 0.15);
        let text = c.to_json();
        for key in [
            "\"trained_order\"",
            "\"trained_shuffled\"",
            "\"control_order\"",
            "\"control_shuffled\"",
            "\"trained_gap\"",
            "\"control_gap\"",
            "\"excess_collapse\"",
            "\"roles_supported\"",
        ] {
            assert!(text.contains(key), "missing {key} in {text}");
        }
        // Valid JSON, one line.
        assert!(!text.contains('\n'), "record is not one line");
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(parsed["roles_supported"], serde_json::json!(true));
        assert!((parsed["excess_collapse"].as_f64().unwrap() - 0.65).abs() < 1e-6);
    }

    #[test]
    fn display_names_the_verdict() {
        let structural = ShuffleControl::new(0.6, 0.0, 0.6, 0.01);
        assert!(structural.to_string().contains("NOT supported"));
        let real = ShuffleControl::new(0.8, 0.1, 0.2, 0.15);
        assert!(real.to_string().contains("SUPPORTED"));
        // And never the bare word "supported" in the negative case.
        assert!(!structural.to_string().contains("| roles SUPPORTED"));
    }
}
