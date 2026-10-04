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

All at 492,418 params. The gap between rows 2 and 3 is the headline: transfer
to new inputs under a trained rule is real and scales with data; transfer to an
unseen rule is nothing, and "nothing" here means three instruments agreeing with
chance rather than one accuracy number.

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

| arm | params | in-dist byte | held-out byte | held-out exact |
|---|---|---|---|---|
| looping: 4 stages × 8 loops | 919,172 | 0.390 | **0.323** | 0.000, 0.000, 0.000 |
| wide: 32 stages × 1 loop | 919,648 | **0.849** | **0.625** | 0.083, 0.792, 0.625 |

Wide's *worst* seed (held-out 0.567) beats looping's *best* (0.355). Complete
separation, far outside the seed spread.

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
zero, and the params-free pair removes it entirely by holding `d_model` fixed.

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

0. **The same-rule curve across model sizes.** Now the highest-value item.
   The new-rule size axis came back flat over 6.5×, which rules out "a bigger
   model would have induced the rule" but says nothing about the rung where
   transfer *exists*. Sweeping `d_model` on `subst-fst-fixed`, where held-out byte
   rises 0.329 → 0.850 with data, is the one measurement here that could produce
   an actual scaling law rather than a null. Needs the data axis at 3+ sizes, not
   a single size.
1. **Rule-CLASS holdout.** Train on some procedures, evaluate on one never seen
   in any form. Not implemented at all. This is what a generalization claim needs
   and no amount of work on the two `subst-fst` rungs substitutes for it.
2. **A rung whose output space beats chance.** Every measurement on the new-rule
   rung sits on a 3-symbol alphabet where chance is 0.333, so a weak partial
   competence cannot be distinguished from guessing at all — and a 6.5× capacity
   increase not moving the number is equally consistent with "there is no weak
   competence" and "the instrument cannot see one". A larger alphabet settles it.
3. **Compute-matched ACT vs fixed depth.** The depth axis measures only depth,
   and depth is the axis the goal cares about (does reuse buy generalization).
4. **Bracketing memorization onset.** The new-rule held-out curve never rises, so
   it cannot bracket onset. Onset needs the curve to rise then fall, and no rung
   tested so far does that.
5. **Seeds.** n=3 cannot resolve anything below ~0.1, and every axis here has
   within-seed spread at or above its between-condition differences. The 256 point
   of the size axis is n=1 only because a concurrent run starved the card and the
   watchdog killed the sweep.
6. **De-confound the oracle rung.** Add ~20 chars of inert filler to `subst-fst`
   so both arms share a length distribution, isolating the header's content from
   its length. Needs a task variant, not a manifest. Until then "execution vs
   induction" is untested rather than answered.
7. **`InstanceInfo` should carry the oracle header.** The sample dump shows demos,
   query and target but not the header — the only part of the prompt that differs
   between the oracle and induction arms.
8. **Do not run two sweeps concurrently.** A 4GB card plus two auto-batch tuners
   means each measures its baseline before the other has allocated, and the
   watchdog kills the run at ~20 minutes with `CUDA_ERROR_DEINITIALIZED`. This is
   what cost the 256 seeds on the size axis.
