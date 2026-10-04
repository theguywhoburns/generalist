# findings — the current consolidated read

Start here. What we believe, how firmly, and what would change it.

Confidence key: **measured** = reproduced or n≥3 with per-seed recorded;
**single-pass** = one run, n=3, unresolved against seed spread;
**speculative** = hypothesized, not tested.

---

## Where we are: memorization yes, generalization only in its weak sense

| | status | evidence |
|---|---|---|
| memorization | **measured, strong** | in-distribution byte 0.90–0.93, `off_pair` 0.04–0.05 vs a 0.333 chance rate |
| generalization — same rule, new inputs | **measured, present** | held-out byte 0.329 → 0.850 as data grows 16 → 288 |
| generalization — new rule | **measured, absent** | held-out byte 0.18–0.34, at or below the 0.333 chance rate on three independent instruments, flat over 18× data and across every size run so far |
| reuse vs width (the architecture question) | **measured, against the premise** | at matched params **and** compute, width wins 2:1 on held-out; at matched compute with params free, 7.07× params buys 2.2× held-out |

Rows 2 and 3 are at 492,418 params. Row 4 is not a single cell and does not
reduce to one: see the two sections below, which reach opposite-looking
conclusions because they hold different things fixed.

The gap between rows 2 and 3 is the headline for *what the model learns*: transfer
to new inputs under a trained rule is real and scales with data; transfer to an
unseen rule is nothing, and "nothing" here means three instruments agreeing with
chance rather than one accuracy number.

**Row 4 is the headline for the research goal, and it is a negative.** Reuse does
not buy generalization without proportional parameters on this rung. But the
reason is disqualifying rather than decisive: this rung rewards
parameters-as-**storage** and is indifferent to parameters-as-**compute**, so it
cannot test depth's advantage even in principle. Treat "depth loses" as
untested and "depth was never tested by this task" as the finding. See
`research/AGENTS.md` contract 8 for why the three available comparisons are not
interchangeable — one of them was initially reported as the goal's answer and
that was wrong.

**Scope note.** In-context learning and higher-order inference were dropped as
*goals*. The varying-rule rung is exactly the ICL test, it is at chance at every
size and data scale measured, and chasing it is chasing a rung this model does
not reach. Its measurements stay below because they are results about
generalization — the strong sense — and deleting a negative result because it
stopped being the target would be exactly the error this file exists to prevent.

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

**Capacity does not move the new-rule number.** `measured`, n=3 except 256.
d_model 128 / 192 / 256 / 384 on the new-rule rung, 600 steps, auto-batch:

| d_model | params | held-out byte | held-out `off_pair` | in-dist byte |
|---|---|---|---|---|
| 128 | 492,418 | 0.269±0.03 | 0.318±0.02 | 0.926±0.01 |
| 192 | ~1.1M | 0.263±0.02 | 0.331±0.02 | 0.763±0.12 |
| 256 | ~1.8M | 0.340 (n=1) | 0.330 | 0.904 |
| 384 | ~3.2M | 0.280±0.01 | 0.339±0.01 | 0.840±0.12 |

Chance is 0.333. Every point sits at or below it, and `off_pair` is flat at
0.32–0.34 across a **6.5× parameter range**. So this is not a threshold that
capacity moves: a 6.5× increase in parameters buys nothing on the strong
generalization rung.

That is a real result and it narrows the question. It does **not** establish that
capacity is irrelevant to generalization — only that it is irrelevant *to this
task at this budget*, and the same-rule rung (where transfer exists) has not
been swept across size at all, which is where a scaling law is actually
measurable.

In-distribution byte accuracy is non-monotone (0.926 → 0.763 → 0.904 → 0.840)
with ±0.12 spread at two of the four sizes, so it is noise at n=3 rather than a
size effect.

**Demonstrations are not used as evidence, and on the memorized rung they actively harm the model.** `measured`, n=3 per point, effect ~8× the seed spread, consistent in every seed.

k = demos per instance, swept as a single value per point. Both rungs, 492,418
params, 600 steps, auto-batch.

*Same-rule rung* (rule in weights):

| k | held-out byte | in-dist byte | held-out exact |
|---|---|---|---|
| 0 | **0.909** | 0.993 | **0.59** |
| 1 | **0.519** | 0.965 | **0.01** |
| 2 | **0.448** | 0.812 | 0.00 |

*New-rule rung* (rule absent from weights, demonstrated per instance):

| k | held-out byte | in-dist byte |
|---|---|---|
| 0 | 0.322 | 0.881 |
| 1 | 0.279 | 0.876 |
| 2 | 0.287 | 0.850 |

One demonstration **halves** held-out accuracy on the same-rule rung and
collapses exact-match from 0.59 to 0.01, while in-distribution barely moves
(0.993 → 0.965). So the loss is specific to unseen inputs, not to the task.

**The dissociation is the result.** Where the rule is already in the weights,
demos are interference: the model attends to them and does worse. Where the
rule is absent, they supply nothing, because they are not being read. Those are
opposite-looking symptoms of one fact — demonstrations are not treated as
evidence about the rule.

This also retires the "the model cannot induce a rule, maybe it needs more
demonstrations" hypothesis: going from 0 to 8 demos is the largest manipulation
of evidence quantity available and it moves the new-rule number by less than the
seed spread.

**Caveats.** k changes prompt length, so it also changes the length bucket and
the auto-batch micro-batch. A length effect of that size is implausible as the
sole cause, but it is not excluded. k=4 and k=8 were lost when the watchdog
killed the run at ~20 minutes, so the curve is 0→2 rather than 0→8; the direction
is established, the shape beyond k=2 is not. k=0 on the same-rule rung is close to
unconditioned generation — there is no rule in the prompt at all — which is why
its 0.909 is an upper bound on "apply the memorized map with nothing to
distract" rather than a point of interest.


---

## Compute-matched: width beats reuse decisively, and the goal's premise is refuted

`measured`, n=3 per arm, same block-steps per token and same parameters to
within 0.05%. This is the experiment the goal's central question turns on, and
the only design in the repo that could have supported the premise.

| arm | params | in-dist byte | held-out byte | in-dist exact | held-out exact |
|---|---|---|---|---|---|
| looping: 4 stages × 8 loops | 919,172 | 0.390 | **0.323** | 0.000, 0.000, 0.000 | 0.000, 0.000, 0.000 |
| wide: 32 stages × 1 loop | 919,648 | **0.849** | **0.625** | 0.083, 0.792, 0.625 | 0.000, 0.219, 0.067 |

Wide's *worst* seed (held-out byte 0.567) beats looping's *best* (0.355). Complete
separation, far outside the seed spread.

**Correction, and it was mine.** Commit `eeb9257` and the first version of this
section reported the wide arm's held-out exact-match as `0.083 / 0.792 / 0.625`.
Those are its **in-distribution** exact-match. Read out of
`checkpoints-compute-matched-wide/*/run.jsonl`, its held-out exact is
`0.000 / 0.219 / 0.067`, mean **0.095** — not 0.500. The exact-match margin is
therefore 0.095 against 0.000, not 0.500 against 0.000, and the earlier text
overstated it by roughly 5×. The byte-accuracy numbers (0.625 vs 0.323) are
unaffected, and byte accuracy is what a sweep ranks on (`src/AGENTS.md`
contract 7), so the conclusion stands — but the claim was wrong and is corrected
here rather than quietly edited, per `research/AGENTS.md` contract 3.

**So reuse does not buy generalization without proportional parameters. It buys
generalization worse than those parameters spent on width would.** At equal
arithmetic and equal storage, distinct parameters win by roughly 2× on held-out
accuracy and by a factor of several on exact-match.

**Why, and it is not mysterious.** Memorization is storage in weights. At equal
parameters the storage is equal, so looping ought to be neutral, and is instead
*worse at memorization* (0.390 vs 0.849). The cause is the reuse itself: 4 weight
sets are asked to serve 32 transformations' worth of function. Reuse is
storage-inefficient — it compresses 32 learned transformations through 4
parameter sets.

**This also relabels the depth axis, and the earlier reading of it was wrong.**
That axis varied `max_loops` at fixed `n_stages = 2`, so depth 1 → 8 was 2 → 16
block-steps: compute and reuse rose together, and the conclusion "looping
improves generalization and memorization about equally" was really "more compute
helps." Holding compute fixed and spending the parameters on width instead is 2×
better. Two axes, opposite conclusions, because only one of them held compute
constant.

**A nuance that cuts the other way, stated because it would be easy to omit.**
In *relative* terms the looping arm retains more of what it learned:
0.323 / 0.390 = 83% of its in-distribution accuracy transfers, against
0.625 / 0.849 = 74% for wide. Per unit of learning, reuse is slightly better at
transfer. It simply learned far less, and absolute held-out accuracy is what a
scaling law is about.

**Uncontrolled difference.** Head count is 8 vs 4 and cannot be matched — an 8×
stage ratio forces an ~8× body-parameter ratio. `head_dim` (16) is matched. The
effect is large enough that head count is unlikely to explain it, but it is not
zero, and the params-free pair below removes it entirely.

---

## Params-free: 7.07× the parameters buys 2.2× the held-out accuracy

`measured`, n=3 per arm, **identical** `d_model` (128), `n_heads` (2), `head_dim`
(64), `ffn_hidden` (384), and identical compute (16 block-steps/token). Only
`n_stages` and `max_loops` differ, so head geometry and activation memory are
identical and the head-count confound above is gone. This is the design the goal's
question actually asks for: match compute, let parameters differ.

| arm | params | in-dist byte | held-out byte | in-dist exact | held-out exact |
|---|---|---|---|---|---|
| looping: 2 stages × 8 loops | 492,418 | 0.309, 0.278, 0.418 | **0.335** | 0.000 | **0.000** |
| wide: 16 stages × 1 loop | 3,479,696 | 1.000, 0.985, 0.923 | **0.739** | 1.000, 0.833, 0.667 | **0.270** |

Per-seed held-out byte: looping 0.305 / 0.298 / 0.401; wide 0.856 / 0.670 / 0.692.

**The gaps are the real finding, and they have opposite signatures.**

| arm | gap (in − out) | what that signature means |
|---|---|---|
| looping | −0.004, **+0.020**, −0.017 → mean **−0.000** | memorized nothing at all |
| wide | −0.144, −0.314, −0.231 → mean −0.230 | memorized, then transferred 76% of it |

The looping arm's gap is zero to within seed noise — and one seed is *positive*.
It is not memorizing badly and transferring badly. It is not memorizing **at
all**: in-distribution accuracy equals held-out accuracy because neither is above
chance. The 0.100% relative transfer this arm shows (0.335 / 0.335 = 100%) is
transfer of nothing and must not be read as reuse being good at transfer; the
compute-matched section's 83%-vs-74% nuance was the same trap on a smaller scale.

**So the diagnosis is storage, not compute.** At 492,418 parameters the 2-stage
model can neither store the rule nor apply it to new inputs, and the reason is
that four weight sets are being asked to serve sixteen transformations' worth of
function. The 16-stage model at 3,479,696 parameters stores it nearly perfectly
(in-dist exact 1.000 / 0.833 / 0.667) and transfers 76% of that to unseen inputs.

**This is also the strongest evidence yet that the task family cannot test depth's
advantage even in principle.** Three independent measurements now agree:

1. On the same-rule rung the held-out answer is "apply a map already in your
   weights" — that needs **storage of the rule**, not arithmetic to apply it.
2. The k-axis result says the model **does not use in-context demonstrations at
   all** (0.909 → 0.519 given one demo), so it is not performing sequential
   inference and there is no computation for extra loops to be spent on.
3. Depth 1 sits at in-distribution byte 0.201, **below the 0.333 chance rate**.
   A memorizing model cannot fail below chance on data it trained on; it is
   confidently applying a *wrong* map. A learning failure, not a compute failure.

**Therefore: "depth loses" is not established, and "depth was never tested by this
task" is.** Width wins here for a boring reason — the rung rewards
parameters-as-storage and is indifferent to parameters-as-compute. Any claim that
reuse cannot buy generalization has to survive a task whose answer requires
several dependent steps and cannot be looked up.

**Correction: I then wrote that such a task "does not exist yet." That was
wrong.** Four of the eight registered rungs already require dependent multi-step
computation — `dyck1`, `scan-tiny`, `parity`, `periodic` — and had never been
used for an architecture comparison. The gap was never a missing task; it was a
missing sweep. See the rung-monoculture section below.

---

## The architecture thread ran on exactly one rung, and it is the wrong one

`measured` by inspection of all six manifests. Every depth-vs-width experiment in
this repo — compute-matched pair, params-free pair, and all five frontier points —
resolves to `pool instances: 320`, which is `per_cell 160 × 2 tracks`, which is
`subst-fst-fixed` and nothing else.

| arm | extends | pool | rung |
|---|---|---|---|
| `compute-matched` | `data-axis-constant-rule` | 320 | subst-fst-fixed |
| `compute-matched-wide` | `compute-matched` | 320 | subst-fst-fixed |
| `paramsfree-looping` | `base-492k` | 320 | subst-fst-fixed |
| `paramsfree-wide` | `base-492k` | 320 | subst-fst-fixed |
| `frontier-s02l08` … `frontier-s16l01` | `base-492k` | 320 | subst-fst-fixed |

**Verified, because `jq '.experiment.tasks'` returns `null` for a manifest that
inherits its `experiment` block rather than erroring.** The null reads like a
missing setting and is not one; the loader resolves `extends` first. All six were
confirmed through `cargo run --example run`, which prints the resolved pool. Note
also that an absent or empty `tasks` key means **all eight builtin rungs**, so a
manifest relying on inheritance from a base with no `tasks` is not a
single-rung run at all. That is the default in `harness/experiment.rs` and it is
a live trap.

**Why this matters more than it looks.** `subst-fst-fixed` is the one rung in the
registry whose answer is a lookup. Held-out accuracy there is "apply a map
already in your weights", which needs **storage of the rule** and no arithmetic
to apply it. Three separate measurements now agree that the rung is
storage-limited:

1. the k axis shows in-context demos are unused (0.909 → 0.519 given one demo),
   so there is no sequential inference for extra loops to be spent on;
2. depth 1 sits at in-distribution byte 0.201, **below** the 0.333 chance rate on
   data it trained on — confidently applying a wrong map, a learning failure;
3. the params-free looped arm shows a gap of −0.000, i.e. it memorized nothing
   at all, in-distribution or out.

**So the entire depth-vs-width thread has been measured on the one rung
structurally incapable of distinguishing parameters-as-storage from
parameters-as-compute.** Every conclusion above about reuse is real *as stated*
and says nothing about depth.

**The compute-requiring rungs already exist and were never swept:**
`dyck1`, `scan-tiny`, `parity`, `periodic` all require dependent multi-step
computation. The missing measurement was a missing sweep, not a missing task.

`dyck1` is the best depth probe of the four, for three reasons:

- **The answer cannot be looked up.** It is `")".repeat(open_depth(input))`, a
  function of the input's stack state. No memorized rule map shortcuts it.
- **The length gradient is steep.** Held-out prefixes are 8..28 characters
  against demo prefixes of 2..10, so the sequential computation required grows
  roughly 3–14× at test time. That gradient *is* the depth probe: a reuse arm
  that adds passes should degrade more gracefully with length.
- **Track B adds unseen depth.** Nesting depth 4..6 never appears in training, so
  the held-out set contains genuinely unseen depth as well as unseen lengths.

**Instrument warning on `dyck1`, recorded before running rather than after.** Its
output alphabet is **unary** — `)` only. A model emitting only `)` scores byte
accuracy 1.0 whenever the true depth is 1, so byte accuracy alone cannot separate
a correct answer from a correct length by luck. The decisive readouts are
`length_exact_rate` and `mean_len_ratio` with byte accuracy beside them, and the
floor here is **not** 0.333: it is whatever a length-only guesser emitting the
training-set modal depth scores. That floor has to be computed from the eval set
before any arm is called above it. `scan-tiny` avoids the unary-output problem
and is the better second choice, because Track B holds out `thrice` and novel
`and` pairings — a compositional holdout already built in.

---

## The stages↔width frontier is U-shaped, and it refutes the storage-inefficiency account

`measured`, n=3 per point, all points at **492,418 params within 0.10%** and
**16 block-steps/token**, same-rule rung, 600 steps. Only `n_stages`,
`max_loops`, and the width that split forces differ.

| point | stages × loops | d_model | ffn | in-dist byte | held-out byte | held-out exact | gap |
|---|---|---|---|---|---|---|---|
| `s01l16` | 1 × 16 | 176 | 526 | **0.942** | **0.821** | **0.475** | −0.121 |
| `s02l08` | 2 × 8 | 128 | 384 | 0.394 | 0.393 | 0.000 | −0.001 |
| `s04l04` | 4 × 4 | 64 | 512 | 0.331 | 0.268 | 0.000 | −0.063 |
| `s08l02` | 8 × 2 | 64 | 213 | 0.481 | 0.477 | 0.000 | −0.004 |
| `s16l01` | 16 × 1 | 80 | 10 | **0.988** | **0.746** | **0.263** | −0.242 |

**Both endpoints beat every interior point by a wide margin** — held-out 0.821
and 0.746 against 0.268–0.477. The curve is not monotone in reuse.

**This refutes the storage-inefficiency account stated two sections above, and
that account was mine.** It predicts maximal reuse should be the *worst*
configuration, since one weight set serving 16 transformations is the most
extreme compression of function into storage. The measurement puts `1×16` first
and `2×8` near the bottom. Storage-inefficiency is monotone in reuse; this is
not. The earlier claim survives only for the interior of the range, and the
sentence "reuse is storage-inefficient" should be read as "reuse is
storage-inefficient **in the 2-to-8-stage band**", not as a general law.

**The discriminator is memorization, not transfer.** The two points that
memorize (in-dist 0.942, 0.988) transfer. The three that do not (in-dist 0.331,
0.394, 0.481 — at or near the 0.333 chance rate) sit at held-out ≈ in-dist ≈
chance with **exact-match 0.000 at every one of their fifteen runs**. Same
signature as the params-free looped arm: nothing memorized, so nothing to
transfer.

**A useful control falls out of this.** `frontier-s02l08` (n_heads 8, head_dim
16) and `paramsfree-looping` (n_heads 2, head_dim 64) are the *same* parameters
and the *same* compute with only head geometry differing. They read in-dist
0.394 vs 0.335 and held-out 0.393 vs 0.335. Head geometry does not move this
result, which retires the head-count worry for this configuration — the
uncontrolled difference in the compute-matched pair is not what drives it.

### The confound, stated before the result is used

**This sweep is confounded with effective batch size and cannot isolate the
stages↔width split.** The five manifests inherited `auto_batch: true` from
`base-492k.json`, and the five points have different `d_model` and therefore
different activation footprints. The tuner selected a different budget for
each: realized effective batches were **64 / 128 / 128 / 128–256 / 256** — and
within `s08l02` it gave different seeds different batches, so it is
nondeterministic there. The manifests asked for batch 6 and the tuner could not
subdivide a micro-batch, so every point trained at 64 or above, 10× the
requested value.

`src/AGENTS.md` contract 5 and root contract 5 both say a sweep arm must differ
from its neighbour only in the intervention. This one varied batch too. The
compute-matched pair got it right (`auto_batch: false`, documented in its
`_comment`); the params-free pair got lucky — both arms independently landed on
effective 128, verified post hoc from `.runs/paramsfree.log`, so that comparison
is sound.

The batch does **not** obviously explain the pattern, and that is worth stating
precisely rather than as reassurance: the two points that memorize received the
*smallest* (64) and the *largest* (256) batches, while the three that memorize
nothing all received 128. "Does not obviously explain it" is not "does not
explain it". The pattern is also confounded with the attention-versus-FFN
allocation each split forces — `ffn_hidden` runs 526 → 384 → 512 → 213 → 10
while `d_model` runs 176 → 128 → 64 → 64 → 80, so neither axis falls
monotonically and `n_stages` is traded against both at once.

**Re-running with `auto_batch: false, batch_size: 6` on all five points, so the
only remaining differences are the split and the width it forces.** That run is
the one to cite. Until it lands, this section's shape is a hypothesis that the
first pass is consistent with and that the batch confound is sufficient to
explain.

---

## Depth buys compute, not reuse (superseded by the compute-matched result above)

**This section is kept because its error is instructive.** Read the section above
first: at matched compute and matched parameters, width beats reuse 2:1.
`single-pass`, n=3 (n=2 at depth 8), 492,418 params, same-rule rung. This is the
closest thing to a direct answer to the goal's central question — does looping
buy generalization without proportional parameters — and it is a partial one.

| depth | in-dist byte | held-out byte | gap |
|---|---|---|---|
| 1 | 0.201 | 0.134 | −0.067 |
| 2 | 0.406 | 0.324 | −0.082 |
| 4 | 0.444 | 0.403 | −0.041 |
| 8 | 0.474 | 0.433 | −0.041 |

Looping improves held-out accuracy a lot — 0.134 → 0.433, **+0.30 absolute,
3.2× relative**, the largest single effect measured anywhere in this repo. It
improves in-distribution accuracy by essentially the same amount (+0.27), and the
generalization gap is flat (−0.067 → −0.041).

So at this scale looping is **not selectively buying generalization over
memorization**; it makes the model better at both, roughly proportionally.

Two refinements:

- The 1→4 rise is real; 4→8 is **not resolved**. Per-seed held-out at depth 4
  spans 0.340–0.440 and at depth 8 spans 0.380–0.485, so the +0.030 difference
  sits inside the noise. Depth 1 is genuinely undertrained (in-dist 0.20), so
  most of the headline effect is really "depth 1 does not learn this task".
- In the regime where the model is actually learning (depth 2→8), held-out rises
  faster than in-distribution: +0.109 vs +0.068, so the gap halves. That hints
  looping favours transfer past the undertrained regime — a hypothesis at n=3
  with 0.10–0.22 seed spread, not a result.

**The confound that stops this answering the question.** Depth *is* compute:
depth 8 does 8× the forward passes of depth 1. This therefore measures "more
compute per token helps", not "reusing parameters helps". The control — a wider
model at matched FLOPs, or any non-looped model at matched compute — **does not
exist in this repo**. That is now the top open item, because until it exists
this table cannot distinguish parameter reuse from arithmetic.
**Depth does not differentiate the stages.** The profile read "uniform" or "one
global head" at every depth on the fixed-rule rung. Combined with the order-
augmentation result, this says depth and stage-role specialization are not
obviously the same axis — but the two were measured on different cells, so that
is a hypothesis, not a finding.

**Being told the rule explicitly does not help, and is worse.** `single-pass`,
n=1, confounded — see the oracle section below. Held-out byte accuracy 0.297 on
the induction rung vs 0.152 on the oracle rung, with mean length ratio
collapsing 0.90 → 0.56 and some outputs empty. Recorded because it is a real
measurement, but the comparison is confounded by prompt length and the drop
cannot be attributed to anything yet.

---

## Mechanism of the varying-rule failure: no mechanism, it is at chance

| claim | why refuted | old numbers in |
|---|---|---|
| "memorization onset lies below 16 instances" | the run was undertrained; at fixed steps more data is fewer epochs, and its own artifact was in-distribution accuracy *falling* as N rose (0.708 → 0.500) | `8bcffc6` |
| "at chance on unseen rules" from a `k_set: [0]` sweep | zero in-context demonstrations: the rule was in no prompt and no weight, so the configuration could not support an induction claim | not committed; corrected in `16079a8` |
| "the model echoes the query" as the below-chance mechanism | `copy_rate` and `query_echo_rate` both 0.000 | `16079a8` |
| "gap narrowing means transfer" on the varying-rule rung | the gap drifted because in-distribution was pinned at ceiling; held-out never moved. The sweep tool now checks the held-out curve first | `dd34041` |
| "the model applies a wrong but consistent permutation" | 0 of 12 sampled outputs admit a consistent char→char map at all | not committed |
| "the model never proposes a symbol other than the query's or the target's" (0 of 141 sampled positions) | my analysis script searched the track's alphabet for a third symbol, but when the map has a **fixed point** the query and target symbols are equal and no third symbol exists — those positions were silently skipped. The real `off_pair_rate` on the full 44-instance held-out set is **0.337**, i.e. chance. The constraint is absent, and it is an artifact of the fixed-point positions, not a property of the model | not committed |

---

## The oracle rung does not help, and the comparison is confounded

`single-pass`, n=1, `varying-rule-oracle.json`. The intent was the sharpest
available test: `subst-fst-oracle` *states* the map in the prompt
(`MAP a->c b->a c->b`), so oracle-minus-normal isolates rule **execution** from
rule **induction**.

It made things worse, and the comparison is confounded, so neither the size of
the effect nor its direction can be read as a fact about induction:

| | induction (`subst-fst`) | oracle (`subst-fst-oracle`) |
|---|---|---|
| held-out byte accuracy | 0.297 | 0.152 |
| held-out mean length ratio | 0.90 | 0.56 |
| length exact | 0.11 | 0.02 |

**The confound, measured** (`examples/prompt_lens.rs`): the 20-char header is
prepended to every prompt, which moves the length distribution hard.

```
              mean len   len<64    64..128
subst-fst        64.6     54.5%     45.5%
oracle           84.4     13.0%     86.8%
```

41% of instances cross out of the short bucket, doubling their padded T, which
changes the B×T row trim and so the micro-batch the model sees. The oracle arm
therefore differs from the induction arm in **two** ways — rule stated vs
inferred, *and* longer context at a higher padded T — and the 0.145 drop in byte
accuracy is consistent with either.

**What is not confounded**: some oracle outputs are *empty* (one sampled
instance emitted nothing for a 16-character target), and mean length ratio 0.56
means the model is systematically truncating. Degenerate output of that kind is
a worse failure than the induction arm's at-chance-but-well-formed output, and it
did not need the length story to appear.

**What would de-confound it**: a length-matched control — the same
`subst-fst` with ~20 chars of inert filler prepended, so both arms share a
length distribution and the header's *content* is isolated from its *length*.
That needs a task variant, not a manifest, so it is queued below rather than
claimed.

---

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

0. **Run the architecture comparison on `dyck1`.** Designed and preflighted, not
   yet run: `research/experiments/arch-dyck1-looping.json` (4 stages × 8 loops,
   919,172 params) against `arch-dyck1-wide.json` (32 stages × 1 loop, 919,648),
   both 32 block-steps/token, pool 400 → ~40 held-out, `k_set [0]`. Same
   compute-matched pair as on `subst-fst-fixed`, on the rung whose answer is a
   function of the input's stack state and whose held-out prefixes are 3–14×
   longer than its demos. **Every reuse conclusion so far is scoped to the one
   rung that cannot test it, and this is the experiment that leaves that scope.**
   Three outcomes count: the looped arm wins (reuse does work width cannot buy),
   they tie (depth was arithmetic in disguise), or width wins again (the premise
   is wrong on a rung that genuinely rewards computation). Read
   `length_exact_rate` and `mean_len_ratio` first — the output alphabet is unary.
1. **The representational-capacity decisive point, next on the same rung.**
   `4 stages × 4 loops` at `d_model = 256`, `ffn_hidden = 748` → 3,480,836
   params, against `16 stages × 1 loop` at `d_model = 128`, `ffn_hidden = 384`
   → 3,479,696. Matched parameters to 0.03%, matched block-steps to the step,
   `head_dim` matched, and `ffn:d` preserved at 2.92 vs 3.00 so it is not the
   degenerate-FFN corner the frontier's `16×1` endpoint falls into. The looped
   arm gets 4× the width per stage at identical storage and arithmetic. If
   **representational capacity** is its deficit it should climb well above the
   looped arm's number toward the wide arm's; if **storage** is the deficit it
   will not move. Cheapest experiment that discriminates the two.
2. **Rule-CLASS holdout.** Train on some procedures, evaluate on one never seen
   in any form. Not implemented at all. This is what a generalization claim needs
   and no amount of work on the two `subst-fst` rungs substitutes for it.
3. **The same-rule curve across model sizes.** Sweeping `d_model` on
   `subst-fst-fixed`, where held-out byte rises 0.329 → 0.850 with data, is the
   one measurement here that could produce an actual scaling law rather than a
   null. Needs the data axis at 3+ sizes, not a single size.
4. **A rung whose output space beats chance.** Every measurement on the new-rule
   rung sits on a 3-symbol alphabet where chance is 0.333, so a weak partial
   competence cannot be distinguished from guessing at all — and a 6.5× capacity
   increase not moving the number is equally consistent with "there is no weak
   competence" and "the instrument cannot see one". A larger alphabet settles it.
5. **Seeds.** n=3 cannot resolve anything below ~0.1, and every axis here has
   within-seed spread at or above its between-condition differences. The 256 point
   of the size axis is n=1 only because a concurrent run starved the card and the
   watchdog killed the sweep. Fill in `d_model` 256 seeds s1/s2 serially.
6. **De-confound the oracle rung.** Add ~20 chars of inert filler to `subst-fst`
   so both arms share a length distribution, isolating the header's content from
   its length. Needs a task variant, not a manifest. Until then "execution vs
   induction" is untested rather than answered.
7. **`InstanceInfo` should carry the oracle header.** The sample dump shows demos,
   query and target but not the header — the only part of the prompt that differs
   between the oracle and induction arms.
8. **Bracketing memorization onset.** The new-rule held-out curve never rises, so
   it cannot bracket onset; the same-rule curve rises 0.329 → 0.850 across a 18×
   data range and shows no sign of turning over. Onset needs a curve that rises
   then falls, and no rung tested so far does that.
9. **Do not run two sweeps concurrently.** A 4GB card plus two auto-batch tuners
   means each measures its baseline before the other has allocated, and the
   watchdog kills the run at ~20 minutes with `CUDA_ERROR_DEINITIALIZED`. This is
   what cost the 256 seeds on the size axis.
