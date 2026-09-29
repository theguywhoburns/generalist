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

/// How far above chance the control's trained-order accuracy must sit before
/// its gap is treated as a real floor. Generous, because a barely-above-chance
/// control has a gap that is mostly noise.
pub const CONTROL_HEADROOM: f64 = 3.0;

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

    /// Whether the control is capable of supplying a floor at all.
    ///
    /// # Why this gate exists
    ///
    /// A control measures "how much does reversal cost a model that cannot
    /// do the task?" If the control scores ~0 in the trained order, it has no
    /// accuracy to lose, so its gap is 0 by arithmetic rather than by
    /// measurement — and `excess` silently collapses to the raw `trained_gap`,
    /// which is the very confound the control was added to remove.
    ///
    /// This is not hypothetical: on the weights-only rungs the untrained model
    /// scores 0.000 in both orders, so every run reported
    /// `control_gap = 0.000` and `roles_supported = true` — a verdict that
    /// looked like evidence and was not. The control is uninformative there,
    /// so the diagnostic must say so rather than emit a confident reading.
    pub fn control_is_informative(&self, chance: f64) -> bool {
        self.control_order > chance * CONTROL_HEADROOM
    }

    /// Verdict on learned role specialization, or `None` when the control
    /// cannot support one.
    ///
    /// `chance` is the task's chance-level accuracy (`1/alphabet` for
    /// byte-string tasks). Returning `None` rather than `false` matters: "the
    /// control could not measure the floor" and "the stages are
    /// interchangeable" are different findings, and collapsing them into one
    /// boolean is how the degenerate case looked like a positive result.
    pub fn role_verdict(&self, chance: f64) -> Option<bool> {
        if !self.control_is_informative(chance) {
            return None;
        }
        Some(self.excess_collapse() > SHUFFLE_EXCESS_THRESHOLD)
    }

    /// Machine-readable JSONL record, so a sweep can aggregate the verdict
    /// without scraping the human-readable line.
    ///
    /// `roles_supported` is `null` when the control could not supply a floor,
    /// so a downstream aggregation cannot silently read "unknown" as "yes".
    pub fn to_json(&self, chance: f64) -> String {
        let verdict = match self.role_verdict(chance) {
            Some(v) => v.to_string(),
            None => "null".to_string(),
        };
        format!(
            "{{\"eval\":\"shuffle-control\",\"trained_order\":{:.6},\"trained_shuffled\":{:.6},\
             \"control_order\":{:.6},\"control_shuffled\":{:.6},\"trained_gap\":{:.6},\
             \"control_gap\":{:.6},\"excess_collapse\":{:.6},\"chance\":{:.6},\
             \"control_informative\":{},\"roles_supported\":{}}}",
            self.trained_order,
            self.trained_shuffled,
            self.control_order,
            self.control_shuffled,
            self.trained_gap(),
            self.control_gap(),
            self.excess_collapse(),
            chance,
            self.control_is_informative(chance),
            verdict,
        )
    }
}

impl std::fmt::Display for ShuffleControl {
    /// The four accuracies and both gaps. Deliberately carries NO verdict:
    /// whether the control can support one depends on the task's chance
    /// level, which `Display` has no way to know. Use [`Self::summary`].
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "trained {:.3}->{:.3} (gap {:+.3}) | control {:.3}->{:.3} (gap {:+.3}) \
             | excess {:+.3}",
            self.trained_order,
            self.trained_shuffled,
            self.trained_gap(),
            self.control_order,
            self.control_shuffled,
            self.control_gap(),
            self.excess_collapse(),
        )
    }
}

impl ShuffleControl {
    /// One-line human reading, including whether the verdict is available.
    pub fn summary(&self, chance: f64) -> String {
        let verdict = match self.role_verdict(chance) {
            Some(true) => "roles SUPPORTED (excess clears the threshold)".to_string(),
            Some(false) => {
                "roles NOT supported (collapse is structural or within noise)".to_string()
            }
            None => format!(
                "roles UNKNOWN: control scores {chance:.3} (chance), so it has no \
                 accuracy to lose and cannot supply a floor. Run this on a rung \
                 where the untrained model beats chance."
            ),
        };
        format!("{self} | {verdict}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The case the control exists to catch: a big trained gap that is
    /// entirely explained by reversal mechanics. The control must be above
    /// chance for this reading to be available at all.
    #[test]
    fn structural_collapse_is_not_reported_as_specialization() {
        let c = ShuffleControl::new(0.60, 0.00, 0.60, 0.01);
        assert!((c.trained_gap() - 0.60).abs() < 1e-9);
        assert!((c.control_gap() - 0.59).abs() < 1e-9);
        assert!((c.excess_collapse() - 0.01).abs() < 1e-9);
        assert!(
            !c.role_verdict(0.05).expect("control is above chance"),
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
        assert!(c.role_verdict(0.05).expect("control is above chance"));
    }

    /// An order-agnostic model collapses by the same amount trained and
    /// untrained: the canonical "no roles learned" result.
    #[test]
    fn order_agnostic_model_reads_as_no_roles() {
        let c = ShuffleControl::new(0.70, 0.20, 0.55, 0.05);
        assert!((c.trained_gap() - 0.50).abs() < 1e-9);
        assert!((c.control_gap() - 0.50).abs() < 1e-9);
        assert!((c.excess_collapse() - 0.0).abs() < 1e-9);
        assert!(!c.role_verdict(0.05).expect("control is above chance"));
    }

    // ---- the degenerate case, pinned as a regression ----

    /// REGRESSION, from a real run: on the weights-only rungs the untrained
    /// model scores 0.000 in BOTH orders, so `control_gap` is 0, `excess`
    /// equals the raw `trained_gap`, and the old boolean reported
    /// `roles_supported = true` — a confident-looking verdict built on a
    /// control that measured nothing.
    ///
    /// This is the exact shape of the `subst-fst-fixed` runs.
    #[test]
    fn control_at_chance_yields_no_verdict_rather_than_a_false_one() {
        let observed = ShuffleControl::new(0.687, 0.000, 0.000, 0.000);
        let chance = 0.05;
        assert!(
            !observed.control_is_informative(chance),
            "a control at chance was treated as a real floor"
        );
        assert_eq!(
            observed.role_verdict(chance),
            None,
            "degenerate control produced a verdict"
        );
        // The excess is still reported, but it is the raw gap and must not be
        // read as evidence: the control contributed exactly nothing.
        assert!((observed.excess_collapse() - 0.687).abs() < 1e-9);
        // And the human line says UNKNOWN rather than SUPPORTED.
        let line = observed.summary(chance);
        assert!(line.contains("UNKNOWN"), "{line}");
        assert!(!line.contains("SUPPORTED"), "{line}");
    }

    /// The gate is a real threshold, not a special case for exactly zero.
    #[test]
    fn control_gate_scales_with_chance() {
        // Same control, different chance levels: informative against a hard
        // task's chance, uninformative against an easy one's.
        let c = ShuffleControl::new(0.70, 0.10, 0.20, 0.15);
        assert!(c.control_is_informative(0.05), "0.20 >> 3x0.05");
        // chance 0.10 requires control_order > 0.30; 0.20 fails.
        assert!(!c.control_is_informative(0.10));
        // chance 0.05 requires > 0.15; 0.20 passes.
        assert!(c.control_is_informative(0.05));
    }

    #[test]
    fn threshold_is_strict() {
        // Note: the exact boundary is not testable in f64 — 0.60 - 0.20 is
        // 0.39999999999999997, so an "exactly at threshold" case lands an ULP
        // either side. Assert unambiguously below and above instead; the
        // comparison is strict (`>`), so equality is not support.
        // trained_gap 0.50, control_gap 0.45 -> excess 0.05, below 0.10.
        let below = ShuffleControl::new(0.50, 0.00, 0.65, 0.20);
        assert!((below.excess_collapse() - 0.05).abs() < 1e-9);
        assert!(!below.role_verdict(0.05).unwrap());

        // control_gap 0.38 -> excess 0.12, above 0.10.
        let above = ShuffleControl::new(0.50, 0.00, 0.60, 0.22);
        assert!((above.excess_collapse() - 0.12).abs() < 1e-9);
        assert!(above.role_verdict(0.05).unwrap());
    }

    /// `roles_supported` must serialize as JSON `null` when unknown, so a
    /// downstream aggregation cannot read "no measurement" as "yes".
    #[test]
    fn json_marks_an_unknown_verdict_as_null() {
        let degenerate = ShuffleControl::new(0.687, 0.0, 0.0, 0.0);
        let text = degenerate.to_json(0.05);
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert!(parsed["roles_supported"].is_null(), "not null: {text}");
        assert_eq!(parsed["control_informative"], serde_json::json!(false));
        assert_eq!(parsed["chance"], serde_json::json!(0.05));

        // And a measurable one still serializes a real boolean.
        let good = ShuffleControl::new(0.8, 0.1, 0.2, 0.15);
        let parsed: serde_json::Value = serde_json::from_str(&good.to_json(0.05)).unwrap();
        assert_eq!(parsed["roles_supported"], serde_json::json!(true));
        assert_eq!(parsed["control_informative"], serde_json::json!(true));
    }

    #[test]
    fn json_carries_every_field_a_sweep_needs() {
        let c = ShuffleControl::new(0.8, 0.1, 0.2, 0.15);
        let text = c.to_json(0.05);
        for key in [
            "\"trained_order\"",
            "\"trained_shuffled\"",
            "\"control_order\"",
            "\"control_shuffled\"",
            "\"trained_gap\"",
            "\"control_gap\"",
            "\"excess_collapse\"",
            "\"chance\"",
            "\"control_informative\"",
            "\"roles_supported\"",
        ] {
            assert!(text.contains(key), "missing {key} in {text}");
        }
        assert!(!text.contains('\n'), "record is not one line");
    }

    /// `Display` carries no verdict: it cannot know the chance level.
    #[test]
    fn display_is_verdict_free_and_summary_is_not() {
        let c = ShuffleControl::new(0.8, 0.1, 0.2, 0.15);
        let d = c.to_string();
        assert!(d.contains("excess"), "{d}");
        assert!(!d.contains("SUPPORTED"), "Display leaked a verdict: {d}");
        assert!(c.summary(0.05).contains("SUPPORTED"));
    }
}
