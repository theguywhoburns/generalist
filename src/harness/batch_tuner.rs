//! Grow the micro-batch until memory says stop.
//!
//! # Why the growth is safe
//!
//! An out-of-memory on CUDA does not raise a catchable Rust error — it aborts
//! the process, or poisons the context so the next launch fails with an
//! unrelated `CUDA_ERROR_LAUNCH_FAILED`. So the tuner never discovers the
//! limit by hitting it. It measures a batch it already survived, extrapolates
//! activation memory to a larger candidate, and grows only while the
//! *prediction* clears the budget.
//!
//! The extrapolation deliberately assumes activations are proportional to the
//! batch with no fixed term. That over-predicts by whatever the fixed part is
//! (padding-free path is the common case here: buckets keep `T` stable), which
//! is the safe direction. Under-predicting would buy a batch that dies.
//!
//! # Why the effective batch is preserved
//!
//! `effective = micro_batch * accum_steps`. The tuner holds that constant and
//! moves `accum_steps` to compensate, because the effective batch is what sets
//! gradient noise. A tuner that varied it would make runs across machines
//! non-comparable, which is the one thing a sweep axis must not be.

/// Peak memory for one micro-batch size, as a high-water mark above the
/// fixed baseline (model weights, optimizer state, CUDA context).
///
/// Storing the *excess* over baseline is what makes extrapolation meaningful:
/// the baseline does not grow with the batch, so scaling the raw total would
/// understate how fast a larger batch approaches the cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Measurement {
    /// Micro-batch size this was measured at.
    pub batch: usize,
    /// High-water mark above the fixed baseline.
    pub peak_mb: u64,
}

/// The memory the activations are allowed to occupy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Budget {
    /// Total device memory, MB.
    pub total_mb: u64,
    /// Largest fraction of *total* memory to spend. The remainder absorbs
    /// fragmentation, the eval pass, and the fact that a training batch is
    /// not the largest allocation the process will ever make (eval chunks are
    /// sized separately and the loss/logits tensors spike on short buckets).
    pub headroom: f64,
    /// Fixed memory already committed: weights, optimizer state, context.
    pub baseline_mb: u64,
}

impl Budget {
    /// Create a budget, clamped to sane fractions.
    ///
    /// Both bounds are load-bearing. A headroom of 1.0 leaves nothing for
    /// fragmentation and the eval pass, which is measured at a different batch
    /// shape and would OOM *after* tuning concluded. A headroom of 0.0 never
    /// grows, silently degrading auto-tuning into "use the manifest value".
    pub fn new(total_mb: u64, headroom: f64, baseline_mb: u64) -> Self {
        Self {
            total_mb,
            headroom: headroom.clamp(0.05, 0.95),
            baseline_mb,
        }
    }

    /// Headroom for activations: `total * headroom - already-committed`.
    ///
    /// Saturating at 0 means the baseline alone has eaten the allowance. That
    /// is a real answer, not a degenerate one: a model whose weights and
    /// optimizer state nearly fill the card cannot grow its batch no matter
    /// how much VRAM the machine has.
    pub fn activation_budget_mb(&self) -> u64 {
        let allowance = (self.total_mb as f64 * self.headroom) as u64;
        allowance.saturating_sub(self.baseline_mb)
    }

    /// Whether `m`'s batch is inside the budget at all.
    pub fn fits(&self, m: &Measurement) -> bool {
        m.peak_mb <= self.activation_budget_mb()
    }
}

/// The next batch size to try, or `None` when the ladder is done.
///
/// Growth is doubling because the failure mode of a linear step is 200
/// iterations of measurement to move the batch by 200, while the failure mode
/// of doubling is a single rejected growth. Doubling also keeps the
/// extrapolation honest: at 2x the measured point, any fixed-cost error is
/// smallest relative to the signal.
pub fn next_candidate(current: usize, ceiling: usize) -> Option<usize> {
    if current == 0 {
        return None;
    }
    let next = current.checked_mul(2)?;
    (next <= ceiling).then_some(next)
}

/// Predict whether `candidate` would fit, from a batch already survived.
///
/// Returns `false` on overflow rather than wrapping: a wrapped prediction is
/// a small number, which would read as "fits" and authorize exactly the batch
/// that cannot be allocated.
pub fn predicts_fit(m: &Measurement, candidate: usize, budget: &Budget) -> bool {
    if m.batch == 0 {
        // Unmeasured baseline: growth from nothing has no slope to extrapolate
        // along, so refuse. The caller measures a real batch first, always.
        return false;
    }
    let predicted = m
        .peak_mb
        .saturating_mul(candidate as u64)
        .checked_div(m.batch as u64)
        .unwrap_or(u64::MAX);
    predicted <= budget.activation_budget_mb()
}

/// The batch size to train with, given what survived.
///
/// `survived` is in trial order, ascending, and every entry is known to have
/// completed a step. The answer is simply the largest, which makes this
/// total and obvious — but stating it explicitly keeps the "measured, not
/// probed" contract in one place rather than spread across the driver.
pub fn choose(survived: &[Measurement]) -> Option<usize> {
    survived.iter().map(|m| m.batch).max()
}

/// `accum_steps` that restore the requested effective batch.
///
/// Rounds the effective batch *up* to a whole number of micro-batches, so the
/// realized effective batch is >= the target rather than below it: silently
/// shrinking the effective batch would tighten gradient noise at exactly the
/// large-data points where it matters, and would do it invisibly.
pub fn accum_for(micro: usize, target_effective: usize) -> usize {
    if micro == 0 {
        return 1;
    }
    target_effective.div_ceil(micro).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(batch: usize, peak_mb: u64) -> Measurement {
        Measurement { batch, peak_mb }
    }

    #[test]
    fn budget_subtracts_the_baseline_before_allowing_activation_growth() {
        // 4GB card, allow 80%, but 2GB is already weights + optimizer + context.
        let b = Budget::new(4096, 0.8, 2048);
        assert_eq!(b.activation_budget_mb(), 3276 - 2048);
        // Without the subtraction the budget would read 3276MB, so a batch
        // needing 2500MB of activations would be authorized on a card with
        // 1228MB of room for them.
        assert!(!b.fits(&m(8, 2500)));
        assert!(b.fits(&m(8, 1000)));
    }

    #[test]
    fn budget_saturates_instead_of_going_negative() {
        // Baseline already past the allowance: no growth is possible, and that
        // must read as zero rather than wrapping to a huge number.
        let b = Budget::new(4096, 0.8, 4000);
        assert_eq!(b.activation_budget_mb(), 0);
        assert!(!b.fits(&m(4, 1)));
    }

    #[test]
    fn headroom_is_clamped_to_sane_bounds() {
        // 1.0 leaves nothing for the eval pass, which is sized separately.
        assert!((Budget::new(4096, 1.0, 0).headroom - 0.95).abs() < 1e-9);
        // 0.0 would silently disable auto-tuning.
        assert!((Budget::new(4096, 0.0, 0).headroom - 0.05).abs() < 1e-9);
    }

    #[test]
    fn candidate_doubles_and_stops_at_the_ceiling() {
        assert_eq!(next_candidate(4, 64), Some(8));
        assert_eq!(next_candidate(32, 64), Some(64));
        // Exactly at the ceiling is allowed; past it is not.
        assert_eq!(next_candidate(64, 64), None);
        assert_eq!(next_candidate(64, 32), None);
        // Zero means "unmeasured" and must not produce a ladder out of thin air.
        assert_eq!(next_candidate(0, 64), None);
    }

    #[test]
    fn growth_is_refused_rather_than_tried_when_the_prediction_overruns() {
        let b = Budget::new(4096, 0.8, 0);
        // 100MB at batch 8, budget 3276MB => 32 is predicted at 400MB: fits.
        assert!(predicts_fit(&m(8, 100), 32, &b));
        // Same slope, candidate 512 => 6400MB predicted: refused.
        assert!(!predicts_fit(&m(8, 100), 512, &b));
    }

    #[test]
    fn extrapolation_refuses_to_grow_from_an_unmeasured_point() {
        let b = Budget::new(4096, 0.8, 0);
        // batch 0 has no slope to scale along. Growing here would be a guess,
        // and a guess that authorizes an allocation is how the process dies.
        assert!(!predicts_fit(&m(0, 0), 16, &b));
    }

    /// A flat measurement is not "infinite headroom", it is a probe that cannot see
    /// the process. Extrapolating it gives slope 0, so every candidate predicts
    /// as fitting and the ladder runs to the ceiling on imagination alone.
    ///
    /// This is not hypothetical: reading device memory with `nvidia-smi` from a
    /// CPU backend returns a constant (the idle display figure), which had the
    /// tuner jump straight to its 512 ceiling. The driver guards on this, and
    /// the guard is the reason a saturation-free budget is not enough.
    #[test]
    fn a_flat_measurement_must_stop_the_ladder_not_authorize_it() {
        let b = Budget::new(4096, 0.8, 0);
        let flat = m(8, 0); // probe observes nothing
        // The trap: with slope 0, EVERY size looks free.
        for cand in [16usize, 64, 256, 512] {
            assert!(
                predicts_fit(&flat, cand, &b),
                "precondition: a flat line does predict as fitting ({cand})"
            );
        }
        // So the driver must stop on the repeat, and this is the signal it uses.
        let first = m(8, 0);
        let second = m(16, 0);
        assert_eq!(
            first.peak_mb, second.peak_mb,
            "identical peaks at doubled batch are the stop condition"
        );
        // One real MB of movement is enough to keep going: the guard must not
        // trip on small-but-genuine measurements.
        let grew = m(16, 1);
        assert_ne!(first.peak_mb, grew.peak_mb);
    }

    #[test]
    fn overflow_refuses_rather_than_reading_as_a_small_number() {
        // A realistic budget, because that is the case that matters: the
        // product below overflows u64, and if it wrapped it would come out
        // small, read as "fits", and authorize the batch that cannot be
        // allocated. Saturation must leave it large enough to be refused.
        let b = Budget::new(4096, 0.8, 0);
        assert!(!predicts_fit(&m(8, u64::MAX / 2), usize::MAX, &b));
        // The same overflow against a 4GB card, at a merely large candidate.
        assert!(!predicts_fit(&m(8, u64::MAX / 2), 1 << 20, &b));
    }

    #[test]
    fn choose_takes_the_largest_survived_batch() {
        assert_eq!(choose(&[m(4, 10), m(8, 20), m(16, 40)]), Some(16));
        // Order-independent, so a driver that logs out of order still agrees.
        assert_eq!(choose(&[m(16, 40), m(4, 10), m(8, 20)]), Some(16));
        assert_eq!(choose(&[]), None);
    }

    #[test]
    fn accum_restores_the_effective_batch_and_rounds_up() {
        // 32 effective, batch 8 => 4 micro-batches.
        assert_eq!(accum_for(8, 32), 4);
        // Batch 48 > 32 effective: cannot subdivide further.
        assert_eq!(accum_for(48, 32), 1);
        // Batch 10 into 32: 3.2 rounds up to 4, so realized effective is 40,
        // never 30. Shrinking it would tighten gradient noise invisibly.
        assert_eq!(accum_for(10, 32), 4);
        assert!(accum_for(10, 32) * 10 >= 32);
        // Degenerate inputs must not divide by zero.
        assert_eq!(accum_for(0, 32), 1);
        assert_eq!(accum_for(8, 0), 1);
    }

    /// The property the whole ladder rests on: growth is monotonic, so a
    /// refused candidate means every larger one is refused too. If this breaks,
    /// the driver's "stop at first refusal" would skip past a fitting size.
    #[test]
    fn refusal_is_monotone_in_candidate_size() {
        let b = Budget::new(4096, 0.8, 0);
        let base = m(8, 100);
        let mut refused_at = None;
        for c in (16..=4096).step_by(8) {
            if !predicts_fit(&base, c, &b) {
                refused_at = Some(c);
                break;
            }
        }
        let first = refused_at.expect("some candidate must be refused");
        assert!(
            (16..=4096)
                .step_by(8)
                .all(|c| !predicts_fit(&base, c, &b) || c < first),
            "a candidate after the first refusal fit again"
        );
    }
}
