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
| reuse vs width, on a rung that rewards computation | **measured, against the premise** | `scan-tiny`, matched params and compute: wide 0.940–0.970 vs looping 0.302–0.505 held-out byte, against a measured 0.663 length-only floor |
| rule-class transfer (compositional holdout) | **measured, partial** | `scan-tiny` Track B holds out `thrice` and novel `and` pairings; the wide arm stays above both floors on every seed |
| gate structure | **measured, for reuse — once** | one gate over a 4-block stack 0.891 vs two gates over 2-block stacks 0.569, differing by a single halting head |

Rows 2 and 3 are at 492,418 params. Rows 4–6 are not a single cell each and do not
reduce to one: see the sections below, which reach different conclusions because
they hold different things fixed and run on different rungs.

The gap between rows 2 and 3 is the headline for *what the model learns*: transfer
to new inputs under a trained rule is real and scales with data; transfer to an
unseen rule is nothing, and "nothing" here means three instruments agreeing with
chance rather than one accuracy number.

**Row 4 is the headline for the research goal, and it is a negative.** On a rung
whose answer is `execute(command)` — which no memorized map can satisfy — reuse
does not buy generalization without proportional parameters. Wide's *worst* seed
beats looping's *best* by 0.435, every wide seed clears the length-only floor, and
every looping seed sits below it. The looped arm's `length_exact` of 0.61–0.65
against wide's 0.91–0.94 at similar `mean_len_ratio` says its failure is
compositional (wrong symbols) rather than formatting (wrong length).

**This took three attempts to measure honestly, and the earlier two were wrong in
opposite directions.** `subst-fst-fixed` rewards parameters-as-**storage** and is
indifferent to parameters-as-**compute**, so it could not test depth's advantage at
all; `dyck1` does reward computation but saturated, both arms reaching byte 1.000.
`scan-tiny` is the first rung that both rewards computation and is hard enough to
separate the arms. See `research/AGENTS.md` contract 8 for why the three
compute-matched comparisons are not interchangeable — one was initially reported
as the goal's answer and that was wrong.

**Row 6 is the one result that favours reuse**, and it was measured on the storage
rung, so it does not yet contradict row 4. Whether the gate effect and the width
effect are the same finding is open, and `scan-tiny` is where to settle it.

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
task" IS.** Width wins here for a boring reason — the rung rewards
parameters-as-storage and is indifferent to parameters-as-compute. Any claim that
reuse cannot buy generalization has to survive a task whose answer requires
several dependent steps and cannot be looked up.

### The storage-limited diagnosis is confirmed, and `dyck1` then turned out too easy

`dyck1` (answer = `")".repeat(open_depth(input))`, held-out prefixes 3–14× longer
than its demos) at matched params and compute, n=3, 2000 steps:

| arm | byte | exact | length_exact | mean_len_ratio |
|---|---|---|---|---|
| looping 4×8 | 1.000 / 0.798 / 1.000 | 1.000 / 0.524 / **0.000** | 1.00 / 0.52 / 0.00 | 1.00 / 0.83 / **10.51** |
| wide 32×1 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 0.728 | 1.00 / 1.00 / 0.73 | 1.00 / 1.00 / 1.17 |

Both clear the measured length-only floor (0.779 byte / 0.629 exact, from 89 real
eval instances) decisively, so `dyck1` *is* learnable at 919K and it does reward
computation. **That confirms the storage-limited diagnosis as a property of
`subst-fst-fixed` rather than of the model.**

But both arms reach byte 1.000 on 2 of 3 seeds, so the rung **saturates and cannot
discriminate them** — the opposite failure from the frontier, where everything sat
at the chance floor. The comparison is inconclusive because the task is too easy.

**The instrument warning earned its place immediately, and again.** Looping seed 2
scores byte **1.000** and exact **0.000** because every target byte is correct
(2/2, 1/1 hits) but EOS is never emitted — the decode runs to the 16-char cap on
**81 of 81** instances, `mean_len_ratio` 10.51. Byte accuracy alone calls that a
perfect run; exact-match alone calls it a total failure. Only the length metrics
identify it as a non-terminating decode. Without them this seed would have been
reported as a catastrophic failure. Wide's worst seed (byte 1.000, ratio 1.17)
fails gracefully by comparison — mildly wrong length.

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

## `scan-tiny`: the rung that finally discriminates the arms, and width wins 2:1

`measured`, n=3 per arm, matched parameters (919,172 vs 919,648) and matched
block-steps (32/token), 2000 steps, `per_cell` 400, batch 64,
`micro_bt_budget` 3072, `auto_batch` off, `k_set: [0]`. **Eval sets verified
identical between arms within every seed** (35/37, 31/21, 24/41 per track), so the
comparison is properly paired.

`scan-tiny`'s answer is `execute(command)` — parse clause structure, expand
`twice`/`thrice`, concatenate. No per-instance map exists to memorize, so the rung
rewards computation rather than storage, which is the property
`subst-fst-fixed` lacks. Output alphabet is 4 symbols (J/W/L/R) and output length
varies 1–9, so both a symbol floor and a length floor exist and are measurable.

**Measured floors on this rung: 0.663 byte for a length-only guesser** (always
emit 4 symbols — the modal target length) **and 0.250 for a constant-symbol
guesser.**

| arm | byte | exact | length_exact | mean_len_ratio |
|---|---|---|---|---|
| **looping 4×8** | 0.302 / 0.464 / 0.505 | 0.014 / 0.077 / 0.062 | 0.61 / 0.65 / 0.40 | 0.87 / 0.95 / **1.76** |
| **wide 32×1** | **0.970 / 0.957 / 0.940** | **0.611 / 0.885 / 0.769** | 0.65 / **0.94** / 0.91 | 1.08 / 1.00 / 1.01 |

**Wide's worst seed (0.940) beats looping's best (0.505) by 0.435.** Every wide
seed clears both floors; **every looping seed sits below the 0.663 length-only
floor.** This is the largest, cleanest separation between the two architectures
anywhere in this repo, and it is on a rung that cannot be satisfied by memorization.

**The looped arm's failure is compositional, not formatting.** Its
`length_exact` is 0.61–0.65 against wide's 0.91–0.94, and its `mean_len_ratio` is
0.87–0.95 on two seeds — so it frequently gets the output *length* approximately
right while getting the *symbols* wrong. It has learned roughly how many actions to
emit and not which. That is the signature of a model that memorized length
statistics rather than the composition.

### The Track A/B split is a real compositional holdout, and it partially transfers

Track B holds out `thrice` and novel `and` pairings — a rule-class holdout built
into the task, which is the one `research/findings.md` has listed as not
implemented at all. Held-out byte by track, against each track's own floor:

| arm | seed | Track A | Track B |
|---|---|---|---|
| looping | 0 | 0.289 (floor 0.572) | 0.308 (floor 0.766) |
| looping | 1 | 0.539 (floor 0.776) | 0.393 (floor 0.715) |
| looping | 2 | 0.583 (floor 0.872) | 0.466 (floor 0.741) |
| wide | 0 | 0.976 (floor 0.572) | 0.968 (floor 0.766) |
| wide | 1 | **1.000** (floor 0.776) | 0.916 (floor 0.715) |
| wide | 2 | 0.969 (floor 0.872) | 0.926 (floor 0.741) |

**The wide arm does partly transfer across the compositional holdout.** Track B
costs it 0.024–0.063 byte against Track A (0.976→0.968, 1.000→0.916,
0.969→0.926) and 0.143–0.479 exact-match, while staying above both floors on every
seed. So **held-out `thrice` and novel `and` pairings are learnable in part** —
this is the first rule-class-style transfer measured anywhere in this repo, and it
is real.

The exact-match drop is larger than the byte drop, which is the length metric's
doing again: on Track B the wide arm gets the length right more often than the
exact action sequence.

**The looped arm is below its own floor on every track of every seed**, so nothing
about its compositional behaviour can be read as transfer rather than noise.

**What this does not settle.** Head count is 8 vs 4 and remains unmatchable at an
8× stage ratio — the uncontrolled difference carried from the `subst-fst` pair.
`head_dim` (16) is matched. The effect is large enough that head count is unlikely
to explain it, but it is not zero.

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

## Gate structure decides whether the memorized solution is executable

`measured`, n=3 per arm, same-rule rung, `auto_batch: false`, batch 6 pinned.
This is the first result in the repo where **reuse wins**, and it is also the
cleanest comparison available, because the two arms differ by *one halting head*.

`param_count` scales the encoder stack by `n_stages × blocks_per_stage` but scales
the halting gates by `n_stages` **alone**. So both arms below hold four distinct
blocks at `d_model` 96 / `ffn_hidden` 256, and differ only in how those four
blocks are grouped and watched:

| arm | config | blocks | gates | block-apps/token | params |
|---|---|---|---|---|---|
| A | 1 stage × 4 blocks × 4 loops | 4 | **1** | 16 | 492,481 |
| B | 2 stages × 2 blocks × 4 loops | 4 | **2** | 16 | 492,578 |

The 97-parameter difference is exactly one gate head (`d_model + 1 = 97`). Block
applications are matched, verified from the per-stage halt vector `bh`: A reads
`[4.000]` → 4 × 4 = 16, B reads `[4.000, 4.000]` → (4+4) × 2 = 16.

| budget | arm | in-dist byte | held-out byte | exact-in | exact-out | gap |
|---|---|---|---|---|---|---|
| 600 steps | A: 1 gate | 0.854 | 0.748 | 0.541 | 0.231 | −0.107 |
| 600 steps | B: 2 gates | 0.603 | 0.467 | 0.028 | 0.000 | −0.136 |
| **2000 steps** | **A: 1 gate** | **0.908** | **0.891** | **0.805** | **0.517** | −0.017 |
| **2000 steps** | **B: 2 gates** | 0.710 | 0.569 | 0.042 | 0.022 | −0.141 |

Per-seed held-out at 2000 steps: A **0.952 / 0.862 / 0.860**, B **0.589 / 0.567 /
0.552**. Complete separation — A's *worst* seed beats B's *best* by 0.27.

**The 600-step pass was not underfitting, and that was worth checking.** Arm B sat
at in-dist 0.603, which reads as "hasn't finished fitting", and this repo has
already retracted one conclusion built on an undertrained pass. At 3.3× the
budget on *both* arms the gap **widened** (0.281 → 0.322), so it is not a budget
artifact.

**Both arms memorize under teacher forcing, and this is the crux.** Training
loss at step 1800 is **0.0058** for A and **0.0142** for B — both essentially
zero. Both models learned the map. Yet A reaches in-dist byte 1.000 / exact 0.958
and B reaches 0.624 / 0.000.

**So the gate structure does not decide whether the solution is learned. It
decides whether it can be executed free-running.** Training CE is teacher-forced;
eval is greedy autoregressive decode. A model can memorize a transducer under
teacher forcing and still fail to run it, and B does exactly that.

**This contradicts the reuse story this repo has been telling, in the opposite
direction from every earlier measurement.** The earlier looped-arm failures
(`paramsfree-looping`, and the interior frontier points) were *storage*-limited:
they memorized nothing, in-distribution or out. Arm B is a different failure. It
memorizes under teacher forcing and cannot execute. Those are two distinct
failure modes that both read as "low held-out accuracy" on the instruments used
so far, and this repo has been treating them as one.

**Corroborating context, weaker.** At 600 steps arm A (`1 gate, 4 blocks`, held-out
0.748) tied `s01l16` (`16 gates, 1 block`, 0.766) inside a ±0.011 spread, and beat
`s16l01` (`16 gates, 16 blocks`, 0.625) by 0.12. So at matched parameters and
matched block applications, **fewer gates over more reuse beats more gates over
less reuse** — the first reuse win in the repo. That comparison is *not* clean
the way A-vs-B is: `s16l01` is 91% attention / 9% FFN (`ffn_hidden = 10`) against
A's 40/60, so width allocation moves with it.

**Unresolved, and the probe that would have answered it was invalid — my
diagnosis of *why* was wrong.** The obvious mechanism question is *why* B cannot
execute. I tried to read the emitted strings by reloading the 2000-step
checkpoint, and got `0x00` PAD out of both arms, with two architecturally
different models producing byte-identical output. I wrote that up as a suspected
checkpoint-restore bug. **It is not one, and the claim is retracted.**

`checkpoint_round_trip_restores_weights` in `src/train.rs` saves a trained
checkpoint, reloads it into a fresh `Trainer` through the same `init_from`
argument the chain uses, and asserts the loss on a fixed batch matches to 1e-6.
**It passes.** `load_checkpoint` restores weights correctly.

The probe failed because `init_from` **is not a manifest key.** It is a parameter
to `run_stage`, and `examples/train.rs` passes `None` unconditionally. So
`train.init_from` in my probe manifest was accepted and ignored, the model was
freshly initialized, and it emitted PAD. Chaining works — through
`examples/chain.rs`, which resolves `$prev` and passes the path programmatically.

**The real hazard is the one that misled me: the config schema silently ignores
unknown keys.** Verified by putting `train.totally_made_up_knob = 123` in a
manifest — it loads clean and runs. A manifest can therefore say something that
has no effect and produce a perfectly plausible run, which is the same failure
shape as a metric that cannot see the error it is not named for. `deny_unknown_fields`
would close it, and `_comment` is safe because it is stripped per-file during
resolution, before parsing. Not done here: it would reject any checked-in
manifest carrying a stale key, and that needs a sweep rather than a drive-by.

**The mechanism is therefore still unidentified, and no claim is made about it.**

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

0. **Rule out the head-count confound on `scan-tiny`,** now that it carries the
   decisive result. Head count is 8 vs 4 and cannot be matched at an 8× stage
   ratio, which is the one uncontrolled variable left in the strongest finding in
   this repo. `frontier-s02l08` vs `paramsfree-looping` showed head geometry does
   not move *that* configuration (0.394 vs 0.335 in-dist at identical params and
   compute), which is suggestive but was measured on the storage rung. A
   compute-matched pair with `head_dim` 16 on both arms and head counts forced
   equal — by giving up some of the stage ratio — would close it.
1. **The representational-capacity decisive point.** `4 stages × 4 loops` at
   `d_model = 256`, `ffn_hidden = 748` → 3,480,836 params, against `16 stages × 1
   loop` at `d_model = 128`, `ffn_hidden = 384` → 3,479,696. Matched parameters to
   0.03%, matched block-steps, `head_dim` matched, `ffn:d` preserved at 2.92 vs
   3.00. Now much more worth running on `scan-tiny` than on `subst-fst-fixed`,
   because that rung discriminates: if the looped arm's deficit is
   representational, 4× the width per stage at identical storage should close much
   of the 0.435 gap. If it does not move, the deficit is not width.
2. **The gate-structure result on a discriminating rung.** One gate over a 4-block
   stack beat two gates over 2-block stacks 0.891 vs 0.569 on `subst-fst-fixed`,
   differing by a single halting head. That is the only reuse win in the repo, and
   it was measured on the storage rung where `scan-tiny` now shows looping at
   0.302–0.505. Running the two gate arms on `scan-tiny` says whether the gate
   effect and the width effect are the same finding or two different ones.
3. **The same-rule curve across model sizes.** Sweeping `d_model` on
   `subst-fst-fixed`, where held-out byte rises 0.329 → 0.850 with data, is the one
   measurement here that could produce an actual scaling law rather than a null.
   `scan-tiny` is now a better candidate for it, since it has a measured floor and
   discriminates between architectures.
4. **A rung whose output space beats chance on the new-rule axis.** Every
   measurement on the new-rule rung sits on a 3-symbol alphabet where chance is
   0.333, so a weak partial competence cannot be distinguished from guessing.
5. **Seeds.** n=3 cannot resolve anything below ~0.1, and every axis here has
   within-seed spread at or above its between-condition differences. The 256 point
   of the size axis is n=1 only because a concurrent run starved the card. Fill in
   `d_model` 256 seeds s1/s2 serially.
6. **De-confound the oracle rung.** Add ~20 chars of inert filler to `subst-fst`
   so both arms share a length distribution. Needs a task variant, not a manifest.
7. **`InstanceInfo` should carry the oracle header.** The sample dump shows demos,
   query and target but not the header.
8. **Bracketing memorization onset.** No rung tested so far has a held-out curve
   that rises then falls.
9. **Do not run two sweeps concurrently.** A 4GB card plus two auto-batch tuners
   starves the watchdog at ~20 minutes with `CUDA_ERROR_DEINITIALIZED`.
10. **Remember `batch_size` is the accumulation target, not the micro size.**
    `train.micro_bt_budget` is the knob; the shipped 2048 caps a `dyck1`-shaped
    micro at 32 rows whatever `batch_size` says. See `src/train.rs`.
