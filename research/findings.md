# findings — the current consolidated read

Start here. What we believe, how firmly, and what would change it.

Confidence key: **measured** = reproduced or n≥3 with per-seed recorded;
**single-pass** = one run, n=3, unresolved against seed spread;
**speculative** = hypothesized, not tested.

---

## The ladder, and where we are on it

| rung | status | evidence |
|---|---|---|
| memorization | **measured, strong** | in-distribution exact-match 1.000, byte 0.89–1.00, length ratio ~1.04 |
| generalization (new inputs, same rule) | **measured, strong** on the constant-rule rung only | held-out byte 0.329 → 0.850 as data grows |
| generalization (new *rule*) | **measured, absent** | held-out byte 0.18–0.27, below the 0.333 chance rate, flat over 18× data |
| in-context learning | **measured, absent** | the varying-rule rung *is* the ICL test; it fails |
| higher-order (infer/evaluate/apply) | **not measured** | no rung of the suite tests it |

All of this is at 492,418 params. It says nothing about other sizes, which is
exactly what the unmeasured size axis exists to determine.

---

## Settled

**Order-dependence is learned, not structural.** `measured`, n=3, auto-batch on,
919,172 params, 576 held-out instances. Order augmentation cut the
reversal collapse 0.179 → 0.049 (3.6×). Per-seed 0.210/0.207/0.120 vs
0.073/0.023/0.052.

This is the load-bearing result for the whole stage-role story: the role
diagnostic's premise — that the collapse is learned rather than a property of
sequential wiring — holds. Two independent signals agree: the profile reading
moves from uniform/one-global-head to the first *DIFFERENTIATED* reading in the
repo (2 of 3 seeds), and shuffled-order accuracy stops being exactly zero.

Not settled alongside it: the random-weight control read 0.000 in both orders on
both arms, so `roles_supported` is still `null`. The collapse is real and
measured; that it exceeds a chance floor is unconfirmed. Different questions,
neither substitutes for the other.

**The constant-rule transfer curve was not generalization.** `measured`.
Held-out byte accuracy on `subst-fst-fixed` rises 0.329 → 0.850 over 16 → 288
training instances. Require the map to be redrawn per instance
(`data-axis-varying-rule.json`) and that rise vanishes. What the curve measured
was applying one memorized constant map to new inputs.

**A 492K model does not do ICL on this rung, at any data size tested.**
`measured`. `subst-fst`, k ∈ {1,2,3,5} so the map is demonstrated every time,
held-out byte accuracy 0.181 / 0.241 / 0.218 / 0.265 at N = 16 / 48 / 128 / 288
— flat, and at or below the 0.333 chance rate. In-distribution exact-match is
0.64–0.79 at the same points. 18× more data buys nothing above chance.

---

## Single-pass, unresolved against seed spread

**Depth helps and appears to saturate by 4.** `single-pass`. Byte accuracy at
fixed depth 1/2/4/8: 0.201 / 0.406 / 0.444 / 0.405 held-out. The gap narrows
with depth (−0.067 → −0.018), so extra compute buys transfer rather than
memorization. Exact-match was 0.000 on both halves at *every* depth — this axis
was invisible until byte accuracy existed.

Unresolved: within-seed spread at depth 8 is 0.265–0.495, which exceeds every
between-depth difference. "Saturates by 4" is not established at n=3. No ACT
point, so there is still no compute-matched ACT-vs-fixed comparison, which is
the comparison the goal actually asks for.

**Depth does not differentiate the stages.** The profile read "uniform" or "one
global head" at every depth on the fixed-rule rung. Combined with the order-
augmentation result, this says depth and stage-role specialization are not
obviously the same axis — but the two were measured on different cells, so that
is a hypothesis, not a finding.

---

## Refuted, with the commit that holds the old numbers

| claim | why refuted | old numbers in |
|---|---|---|
| "memorization onset lies below 16 instances" | the run was undertrained; at fixed steps more data is fewer epochs, and its own artifact was in-distribution accuracy *falling* as N rose (0.708 → 0.500) | `8bcffc6` |
| "at chance on unseen rules" from a `k_set: [0]` sweep | zero in-context demonstrations: the rule was in no prompt and no weight, so the configuration could not support an induction claim | not committed; corrected in `16079a8` |
| "the model echoes the query" as the below-chance mechanism | `copy_rate` and `query_echo_rate` both 0.000 | `16079a8` |
| "gap narrowing means transfer" on the varying-rule rung | the gap drifted because in-distribution was pinned at ceiling; held-out never moved. The sweep tool now checks the held-out curve first | `dd34041` |
| "the model applies a wrong but consistent permutation" | 0 of 12 sampled outputs admit a consistent char→char map at all | not committed |
| "the model never proposes a symbol other than the query's or the target's" (0 of 141 sampled positions) | my analysis script searched the track's alphabet for a third symbol, but when the map has a **fixed point** the query and target symbols are equal and no third symbol exists — those positions were silently skipped. The real `off_pair_rate` on the full 44-instance held-out set is **0.337**, i.e. chance. The constraint is absent, and it is an artifact of the fixed-point positions, not a property of the model | not committed |

---

## Mechanism of the varying-rule failure: no mechanism, it is at chance

**Known** (`measured`, n=44 held-out / 32 in-distribution, 600 steps, k≥1 so the
rule is demonstrated in context every time):

| | held-out | in-distribution | chance |
|---|---|---|---|
| byte accuracy vs target | 0.297 | 0.901 | 0.333 |
| byte accuracy vs query | 0.267 | — | 0.333 |
| off-pair rate | 0.337 | 0.054 | 0.333 |
| exact-match | 0.000 | 0.844 | — |
| copy rate / query-echo rate | 0.000 / 0.000 | — | — |

All three held-out numbers sit at chance, from three different directions: the
output is no better than chance against the target, no better than chance
against the input, and proposes an "off-pair" symbol exactly as often as chance
predicts. `off_pair` — any position whose emitted symbol is neither the query's
nor the target's — is the strongest of the three, because no bias the other two
allow can game it.

**So on an unseen rule the output is statistically indistinguishable from a
uniform draw over the track's alphabet.** Length is approximately right (ratio
0.90) and the symbols stay in-alphabet. That is the whole of it.

**What it does have is memorization**: in-distribution byte accuracy 0.90 with
`off_pair` at 0.054 against a 0.333 chance rate. Reproducing a stored target
structurally cannot propose an off-pair symbol, so that low number is what
memorization looks like — not a competence that fails to transfer.

**Next is no longer a mechanism question**, since the answer is "none detectable
at this scale". The productive question is what would change that: a bigger
model, longer training, more demos per instance, or a rung whose output space
is larger than a 3-symbol alphabet so chance is lower and a partial competence
could show above it.---

## Open, in order of value

1. **Rule-CLASS holdout.** Train on some procedures, evaluate on one never seen
   in any form. Not implemented at all. This is what Goal questions 1 and 3
   require, and no amount of work on the two rungs above substitutes for it.
2. **Size axis.** Completely unmeasured. The 492K model is the *only* point on
   it, and it sits at the bottom. "Where on the ladder" is currently a statement
   about one cell.
3. **Compute-matched ACT vs fixed depth.** The depth axis measures only depth.
4. **Bracketing memorization onset.** The varying-rule held-out curve never
   rises, so it cannot bracket onset either. Onset needs the held-out curve to
   rise then fall, and no rung tested so far does that.
5. **Seeds.** n=3 cannot resolve anything below ~0.1. Every axis here has
   within-seed spread at or above its between-condition differences.
6. **Positive control for rule induction.** Nothing shows the model *can* induce
   a new map at this size and budget, so "at chance" names a failure without
   isolating which sub-skill is missing.