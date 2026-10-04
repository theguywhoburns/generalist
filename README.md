# Generalist

A byte-level looped transformer (`src/model`) trained with a stock Muon/AdamW
hybrid optimizer (`src/optim`), used to measure how **generalization** varies
with model capacity and training-data scale, and whether looping a fixed
parameter set buys generalization without proportionally more parameters.

## Research goal

> To characterize how generalization changes as a function of model capacity and
> training-data scale, and to determine whether iterative reuse of a fixed
> parameter set — a looped architecture — buys generalization without a
> proportional increase in trainable parameters.
>
> Concretely: does held-out accuracy scale with parameter count and with data,
> in what regime, and does looping shift that regime?

Two axes, one question. **Capacity** (`d_model`, `n_stages`) and **data**
(`max_train_instances`) are the independent variables; **held-out accuracy** is
the dependent one; **looping** is the intervention whose effect on the
capacity axis is the thing worth knowing. The quantity of interest is the
**generalization gap** — in-distribution accuracy minus held-out accuracy, both
from the same run — because held-out accuracy alone cannot say whether a model
is failing or merely not being helped by more data.

### What "generalization" means here, precisely

Two distinguishable senses, and conflating them produced a wrong result in this
repo before:

- **Same rule, new inputs.** The latent rule is fixed and trained on; the eval
  asks for it on strings the model has not seen. This is the rung where transfer
  is measurable and non-zero here.
- **New rule, new inputs.** The rule differs per instance, so the model must
  acquire it rather than recall it. Measured, and **absent** at every size
  tested.

The second is the stronger sense and it is where the interesting negative lives.
The first is where a scaling curve exists to be measured. Both are in scope.

### Status

| rung | status | evidence |
|---|---|---|
| memorization | **measured, strong** | in-distribution byte accuracy 0.90–0.93, `off_pair` 0.04–0.05 against a 0.333 chance rate |
| generalization, same rule | **measured, present** | held-out byte accuracy 0.329 → 0.850 as data grows 16 → 288, at fixed model |
| generalization, new rule | **measured, absent** | held-out byte 0.18–0.34, at or below the 0.333 chance rate on all three instruments, flat over an 18× data range and across a 6.5× parameter range |
| reuse vs width | **measured against the premise, on one rung** | at matched parameters *and* compute, width wins 2:1 on held-out byte; at matched compute with parameters free, 7.07× parameters buys 2.2× |

The headline is the gap between rows 2 and 3. A 492K model transfers to new
inputs under a trained rule, and transfers **nothing** to a rule it has not seen,
where "nothing" is established on three independent measures rather than one
accuracy number.

**Row 4 is the headline for the research goal, and it is negative — with a scope
limit attached.** Reuse does not buy generalization without proportional
parameters *on the rung every architecture experiment has run on*, and that rung
rewards parameters-as-**storage** while being indifferent to
parameters-as-**compute**. So the width side of the comparison is measured and
real, while "depth loses" is **untested** and "depth was never tested by this
task" is the finding. See "Depth vs width".

**Cell coverage.** Rows 1–3 are at 492,418 parameters. Row 4 deliberately spans
492,418 → 3,479,696, because a depth-vs-width comparison holding capacity fixed
cannot answer the question — that constraint is the point of the params-free
design. The size axis exists too (492K → ~3.2M) and is negative on the
new-rule rung, so capacity is no longer a single-cell extrapolation there; it is
still a single rung.

**Independent variables** (all manifest-reachable today):

| axis | knob | reachable values |
|---|---|---|
| model size | `model.d_model` / `ffn_hidden` / `n_stages` | 256 (984,065 params) … 4 stages (3,542,276) … |
| compute depth `L` | `stop`, `model.max_loops` | `act` (learned), `fixed {1,2,4,8,16}`, `converge` |
| data | `train.max_train_instances`, `task` | any cap ≥ 0, 8 tasks, both tracks |
| context `k` | `experiment.protocol.k_set`, `k0_rate` | `{0,1,2,3,5,8}`, k=0 at 15% |
| optimization | `train.lr_muon` / `lr_adamw` schedules | `constant` / `linear` / `cosine` / `step` |

The data axis is `train.max_train_instances`, **not** `experiment.per_cell`.
Varying `per_cell` moves the training pool and the eval set together, so each
point's held-out accuracy is measured on a different split and the curve is not
comparable across points. The cap is applied after the holdout split, so the
eval set is identical at every point and the only thing that changes along the
axis is how much the model trained on. See "Measuring the gap".

**Dependent variable:** held-out accuracy — exact-match **and** byte accuracy —
plus the train/held-out gap, copy-rate, and the per-stage halt profile. Every
held-out accuracy is measured on instances the model never trained on (see
"Eval"), and every accuracy is reported next to the in-distribution number it
should be compared against (see "Measuring the gap").

**The four questions:**

1. **Scaling with size.** Does held-out accuracy scale with log-params, and is
   the slope the same on Track A and Track B?
2. **Scaling with compute depth.** Fix parameters, sweep `L`. Does extra
   depth buy generalization, and at what rate? Crucially: does the accuracy
   come from *adaptive* depth (halt depth tracks difficulty) or from
   *degenerate early halting* (all tokens halt immediately, and accuracy is
   unchanged)? These look identical in an accuracy table and are opposite
   claims; only the per-stage halt profile separates them.
3. **Where memorization begins.** Operationally, the point in
   (params × depth × data) space where held-out accuracy stops tracking
   in-distribution accuracy — the two curves diverge, or held-out accuracy
   saturates while in-distribution accuracy keeps climbing. The crossover
   location and the *form* of the law on either side of it is the headline
   result. This is measured as the gap between the two halves of a single run,
   not inferred from held-out accuracy alone; see "Measuring the gap".
4. **Weight scale vs. compute depth.** This model has two escape routes to
   more effective capacity: a bigger `d_model` at fixed depth, or more loop
   iterations at fixed size. If the scaling law has a different shape along
   each, that is the interesting finding; if not, depth is redundant with
   width and the cheaper axis wins.

`subst-fst-fixed` and `subst-fst` are the two senses of generalization
above, and they are separated by one flag. `subst-fst-fixed` holds one
substitution map constant, so held-out accuracy measures applying a trained rule
to new inputs. `subst-fst` redraws the map per instance, so it measures
acquiring a rule that is nowhere in the weights. Accuracy-vs-k is one slice of
the dependent variable, not a goal in itself.

## Setup

Byte-level `LoopedTransformer` (`src/model/`), untied LM head, `RoPE`,
RMSNorm, causal shift with the target tail as the only scored positions.

`LoopedTransformer = Vec<LoopedStage>`. Each `LoopedStage` holds
`blocks_per_stage` encoder blocks **and its own halting gate**. Stages run
**sequentially**: each iterates its whole block stack to its gate's fixed
point, then hands its readout to the next stage. This "learned asynchronous
depth" is the compute-depth axis; maximum total depth is
`n_stages × max_loops`. With one single-block stage the model is exactly
classic single-gate ACT.

- `LoopedConfig::base_1m()` — 256-dim, 4 heads × 64 head_dim, 768 FFN,
  `n_stages = 1`, **984,065 parameters**, untied head. The exact count is
  computed analytically by `param_count()` and pinned by tests at
  **984,065** and **3,542,276** (`n_stages = 4`).
- The shipped `configs/stage0-base.json` uses `n_stages = 4`, so **the
  default run is 3,542,276 parameters, not 1M**. "base_1m" names the
  single-stage layout only.
- The scaling runs in "Current results" use a deliberately smaller model —
  `d_model` 128, `n_stages` 2, `max_loops` 4, **492,418 parameters** — so that
  600 steps is enough to fit anything at all on a 4 GB laptop. It is the
  smallest point on the size axis, which has since been swept upward on the
  varying-rule rung (flat at chance from 492K to ~3.2M; see "Size axis"). The
  order-augmented arm uses `d_model` 128 at the full `n_stages` 4, **919,172
  parameters**, on the full 6-task suite; at 576 held-out instances per eval
  pass that eval set is an order of magnitude larger than the ~29 used by the
  two scaling axes.
- Halt gates deep-start at `halt_bias_init = -3.0` (~4.7% halt probability),
  so training begins near max depth and ponder pressure shortens it — the
  shallow-halt trap is the failure mode this avoids.

**Three stop modes**, selected by the manifest `stop` key:

| mode | behaviour |
|---|---|
| `act` | Graves-style per-token halting + ponder penalty (default when the key is absent) |
| `fixed {loops}` | exactly `loops` applications of every stage; no gate, no ponder. The compute-matched control |
| `converge` | iterate until relative latent change stays under `conv_tol` for `conv_patience` steps |

`fixed` and `converge` have no learned step count, so `ponder_weight` is a
constant loss offset for them; the loader flags a manifest that sets both. A
`fixed` `loops` above `model.max_loops` is also flagged rather than silently
honoured, because the residual scale and the memory budget are both sized for
`max_loops` — a depth sweep past 8 has to raise it too.

## Eval

Byte-level, shifted-causal LM over a 256-symbol byte vocabulary (`PAD = 0x00`,
`EOS = 0x01`; task alphabets are printable ASCII, so 0x00/0x01 are free).
Training scores target bytes + the EOS terminator only — prompt bytes and pads
are masked out. Eval decodes greedily until EOS (capped by
`train.eval_max_new`) and scores **exact string equality with the target**,
plus the fraction of target bytes correct; the EOS byte terminates the row and
is not itself part of the compared string. Both accuracies are logged, for
reasons given under "Measuring the gap".

- `train.eval_holdout` (default `0.1`) splits each pool **inside `run_stage`,
  before anything samples from it**. No reported accuracy is measured on
  training data.
- The partition hashes each instance's **own identity** (task, track, k,
  prompt bytes), not its pool index, so it is stable under reordering and
  changes with the seed. It is per-(task, track) cell, so a cell that runs
  dry is visible rather than silently reshuffling the pool.
- `eval_holdout: 0` is **rejected by the loader** as in-distribution. The
  only escape hatch for reproducing a pre-holdout number.
- **Every accuracy number produced before this split existed is
  in-distribution and is not comparable.** The checked-in
  `checkpoints-fixed/run.jsonl` predates it (records are labelled `eval:
  final`, carry no `seed`, and were written from the training pool).
- Re-run all earlier stages after each new stage; the chain runner
  re-evaluates every earlier split, so forgetting shows up as a drop in a
  `previous-N` eval row.
- `train.eval_in_distribution` (default on) additionally scores
  `train.indist_eval_per_cell` instances drawn from the **training** pool,
  through the same decoder and the same `eval_max_new` limit as the held-out
  split, and logs both halves as a `memorization` record. It is a diagnostic,
  not a second headline metric, so it is kept small: eval decode is the slowest
  part of a run.
- The in-distribution sample is drawn **after** `max_train_instances` is
  applied. Sampled before the cap, a capped run would score instances it never
  trained on and report them as in-distribution, collapsing the gap at exactly
  the points where memorization matters most.

Logged per run (`run.jsonl`, one JSON object per evaluated instance): task,
track, k, `correct`, `byte_hits`, `byte_total`, `copied`, `steps`, `halt`,
per-stage `bh`, and the run `seed` — so a multi-seed comparison is a filter,
not a string join. Final-eval summaries additionally carry per-stage
p50/p90/std, utilization shares, between-stage halt correlations, and — see
"Reading the stages" below — the profile evidence and the shuffle control. The
`memorization` record carries `indist_accuracy`, `heldout_accuracy`, `gap`, and
the byte-accuracy variant of each, so no run needs to be re-scraped from
stdout to answer a scaling question.

## Reading the stages

**The stage-order shuffle is a weak test, and is not the one to rely on.**
Reversing stage order collapses accuracy *by construction*: stages run
sequentially, so reversal reverses the data flow, and the same collapse
appears for random weights. A collapse on its own is *necessary* for role
specialization but nowhere near sufficient.

It also has a hidden premise: that the collapse is **learned**, not a fixed
property of sequential stage execution. A model wired to run stages in order
would collapse under reversal no matter what it learned, and the collapse would
then carry nothing about roles. That premise is not an assumption any more —
the order-augmented arm below has tested it, and the test came back positive.

Three measurements carry the claim instead, weakest to strongest:

1. **Profile evidence** (`src/harness/role_evidence.rs`) — needs no control
   and no intervention, so it works on every rung. If stages do different
   jobs, then (a) the compute split is non-uniform — normalized entropy of the
   per-stage halt shares below 1.0 — and (b) the gates are not one head in
   four costumes — mean off-diagonal correlation of the per-stage halt
   vectors near 0. Four profiles are distinguished, and the reading is
   refused entirely below 2.0 mean halt steps, which is the
   degenerate-early-halting case the `1.5e-2` arm exhibited (profile
   `5.2 / 1.6 / 1.0 / 1.0` — a collapse to shallow, not differentiated depth).
2. **Random-weight control** — measures how much reversal costs a model that
   cannot do the task. `run_stage` runs it whenever `shuffle_eval` is set:

   ```text
     trained_gap = trained_order_acc - trained_shuffled_acc      (observed)
     control_gap = control_order_acc - control_shuffled_acc      (floor)
     excess      = trained_gap - control_gap
   ```

   **This control is degenerate on weights-only rungs.** The untrained model
   scores 0.000 in *both* orders, so `control_gap` is 0, `excess` collapses to
   the raw gap, and the verdict reduces to the original confound. So
   `roles_supported` is emitted as JSON `null` with `control_informative:
   false` whenever the control sits within 3× of chance: "the control could
   not measure the floor" and "the stages are interchangeable" are different
   findings, and collapsing them is how the degenerate case first looked like
   a positive result.
3. **Order-augmented training** (`configs/stage0-orderaug.json`) — the causal
   test. `stop.act.shuffle_train` resamples the stage order every training
   step; read against `stage0-4block.json` it isolates that one variable. If
   an order-augmented model *still* collapses under eval-time reversal,
   order-dependence is structural. If it does not, order-dependence was
   learned and is therefore avoidable — which is the claim the diagnostic
   exists to make. The order-augmented run's training loss is **not**
   comparable to the ordered run's: it solves a different,
   permutation-robust function.

   **This has now been run, and the answer is the second branch.** Order
   augmentation cut the reversal collapse by 72% (0.179 → 0.049), so
   order-dependence is learned, the premise holds, and the collapse is
   readable as evidence about roles. `examples/arm_compare.rs` runs the two
   arms; the table and its caveats are under "Order augmentation" in "Current
   results". Note what the positive result does *not* fix: instrument 2 above
   is still degenerate on this rung, so the collapse is now known to be learned
   but is still not certified as exceeding a chance floor. Those are two
   separate questions and only one of them has an answer.

All three land in `run.jsonl` — the `*-roles` and `shuffle-control` records,
and the pool-level `final-trained-pool` / `final-shuffled-pool` summaries the
reversal collapse is computed from — so a sweep can aggregate them without
scraping stdout.

**Scope.** These three answer one question — *do the stages do different jobs* —
and every one of them is about role specialization. They say nothing about
whether an accuracy number is generalization, which is what the next section is
for. A clean role reading on a model that memorizes is still a model that
memorizes, so the two instruments are never substitutes for each other.

Where the three stand: instrument 1 produced the first differentiated reading in
this repo on the order-augmented arm (2 of 3 seeds), instrument 3 has returned
its positive branch, and instrument 2 is still degenerate on every rung
measured so far. The role claim therefore rests on instruments 1 and 3, and it
rests on one cell.

## Measuring the gap

Two instruments, both in every run's log, and both there because the obvious
proxy for each was measured first and found unable to carry the claim.

**1. Byte accuracy.** Exact-match is a conjunction over every target byte. On
the ~20-byte targets here it reports 0.000 identically for "19 of 20 right" and
for "nothing right", so it cannot rank a model that is partially correct, and
it cannot show a trend in the number of errors. `Record.byte_hits /
byte_total` and `Summary::byte_accuracy` report **micro-averaged** byte
accuracy: hits and total summed over all instances, then divided. The
micro-average is the one that matters — a per-instance mean weights a 20-byte
target the same as a 200-byte target and overstates the short-target score by
up to 4.6× on this pool, which is enough to invert any gap computed from it.

This is not a hypothetical fix. The depth axis below scored **0.000 exact-match
on both halves at every depth** while training CE sat at 0.45. On exact-match
alone the axis reads "depth does nothing"; it is the metric that could not see
the effect, not an absence of one. Exact-match stays in the log as the headline
number because it is the operative definition of a correct answer — but byte
accuracy is what a sweep ranks on.

**2. The train/held-out gap.** Held-out accuracy alone **cannot locate
memorization onset**, because it saturates the same way whether the model solved
the task or stopped benefiting from more data. The signal is the divergence, and
the divergence needs both halves from one run. So each run logs a
`memorization` record with `indist_accuracy`, `heldout_accuracy`, and their
`gap`, plus the byte-accuracy variant of each.

The reason to prefer the gap over a falling held-out number is that they mean
different things. Held-out accuracy can fall because the task got harder along
the axis, because the training budget bought less, or because the model
overfit; the gap only moves for the last of those. It is also a difference of
two accuracies, so **its variance exceeds either half's** — `scaling_sweep.rs`
prints per-seed detail whenever more than one seed is requested, and a gap
reported without its spread is not a measurement.

**A gap is also the wrong instrument when one of its two terms is pinned.** On
the varying-rule data rung, in-distribution byte accuracy sat at 0.97–1.00, so
the in-distribution term had almost no room to move; the gap drifted
-0.722 → -0.666 while held-out byte accuracy stayed at chance throughout. The
drift was the pinned term, not transfer. So `scaling_sweep.rs` now checks
whether the held-out curve moved before it reads the gap, suppresses the gap
verdict when it did not, and prints the chance rate explicitly when held-out
lands at or below it. The general rule: read the held-out column on its own
first, and treat a gap trend as evidence only when that column moved with it.

```text
  indist_accuracy, indist_byte_accuracy   — from the training pool
  heldout_accuracy, heldout_byte_accuracy — from the holdout split
  gap = heldout - indist                  (negative: the model is behind out of sample)
```

**3. A floor is a property of the rung, not of the metric, and the rungs queued
next move it a long way.** The 0.333 chance rate quoted throughout this README
is a fact about a **3-symbol output alphabet**, not about byte accuracy as an
instrument. `dyck1`, the first rung queued to test depth properly, has a
**unary** output alphabet — `)` only, so a model emitting nothing but `)`
scores byte accuracy **1.0** whenever the true depth is 1 and byte accuracy
alone cannot separate a right answer from a right length by luck. The decisive
readouts there are `length_exact_rate` and `mean_len_ratio` with byte accuracy
beside them, and **the floor is not 0.333** — it is whatever a length-only
guesser emitting the training-set modal depth scores, computed from the eval
set before any arm is called above it. `scan-tiny` avoids the unary problem.

A related limit on the tables below: **the depth-axis logs predate
`off_pair_rate`**, `length_exact_rate` and `mean_len_ratio`, so those three
instruments do not exist on that axis. Everything after it was logged with all
of them.

## In-context demonstration protocol

- Every instance ships k demonstrations of **that instance's** latent rule
  (input→output pairs). k is sampled per instance from `{0,1,2,3,5,8}`;
  k=0 at ~15%.
- Demonstrations are resampled per instance. No fixed demo set per task
  family.
- Layout randomized per instance: demo order, separator (`->`, `:`, `=`,
  `|`), query-first vs. query-last.
- ~5% of instances contain one corrupted demonstration (one character
  flipped) for majority-rule robustness.
- Demo inputs are drawn from a **different length/symbol regime** than the
  query (for `subst-fst`: demos 2–8 chars, queries 8–20). Copy-rate — the
  fraction of model outputs appearing verbatim among demo outputs — is
  logged per eval.
- **Anti-copy policies.** `DemoPolicy::ExcludeAnswer` (every task but parity)
  resamples any demo whose output exactly equals the query target, so
  verbatim copying scores ~0. `DemoPolicy::Balanced` (parity only, binary
  outputs) forces equal class coverage per instance, because for binary
  outputs pure exclusion makes *every* demo the opposite class — a trivial
  flip-the-demos shortcut. Balanced holds every non-inductive strategy
  (copy-random, copy-nearest, majority vote, flip-majority) at exactly chance
  while true induction scores 100%, so absolute accuracy separates cleanly.
  Policy enforcement runs after corruption, so a corrupted demo that happens
  to reproduce the target is resampled; this microscopically lowers the
  effective corruption rate in exchange for a hard guarantee.
- Oracle tasks prepend an explicit rule statement (`MAP a->c b->a c->b`) to
  the prompt. The header is context, never scored, so oracle-vs-normal deltas
  isolate rule **execution** from rule **induction**.
- Invariance is claimed only over the trained k-distribution, not arbitrary
  counts. `MAX_K = 16` is the context budget; the loader flags a `k_set`
  above it.

## Tracks and rungs

Every stage runs twin tracks:

- **Track A** — rule family seen in training, held-out instances.
- **Track B** — rule never seen, defined only by k prompt demos (for
  `subst-fst`, disjoint alphabets: A maps within `{a,b,c}`, B within
  `{x,y,z}`).

The `subst-fst` family is a staircase, and the order matters: nothing above
the bottom rung is meaningful until the bottom rung works.

| rung | manifest | task | what it isolates |
|---|---|---|---|
| substrate | `stage0-fixed.json` | `subst-fst-fixed`, `k_set: [0]` | one constant substitution `a→b b→c c→a`, **zero demonstrations**. Weights-only transduction |
| mapping | `stage0-maponly.json` | `subst-fst-oracle`, `k_set: [0]` | rule execution with the map handed over, zero induction pressure |
| oracle | `stage0-oracle.json` | `subst-fst-oracle` | execution with demos present too |
| induction | `stage0-4block.json` | full Stage-0 suite, k ∈ {0,1,2,3,5,8} | rule induction: the k=0 → k>0 delta |

`configs/chain.json` runs them bottom-of-staircase first, each chaining off
the previous weights via `init_from: $prev`.

**Caveat on all four rungs:** the holdout is **instance-level, not
rule-level**. `subst-fst-fixed` is the extreme case — one constant rule for the
whole pool, so the model saw the rule itself during training and the eval
merely asks it to apply a memorized map. On `subst-fst` the eval instances do
carry rules that never appear in the training pool, but they are new *instances
of the same rule family* (Track A) or the same procedure over new symbols
(Track B). **This is no longer only a caveat: it has been measured.** The data
axis on `subst-fst` returns chance-level held-out accuracy at every data size
while in-distribution exact-match reaches 1.000, so the constant-rule transfer
was map application and not generalization. Holding out rule *classes* — a
procedure never trained on in any form — is still not implemented. See "Known
limitations", item 1.

## Curriculum

**Stage 0 — pattern prediction. Implemented.** 8 registered tasks: `parity`,
`dyck1`, `subst-fst`, `subst-fst-oracle`, `subst-fst-fixed`, `periodic`,
`copy-rev-rep`, `scan-tiny` (all `stage() == 0`). Byte-encoded, procedural,
both tracks. The default suite in `stage0-base.json` trains on 6 of them — the
two `subst-fst` variants are reached through their own manifests, as rungs
rather than as suite members.

**Stages 1–3 are a plan, not code.** Nothing below is implemented, and no
number in this README depends on it.

- *Stage 1 — state tracking.* Multi-counters, variable binding (overwrite,
  nesting), delayed match / NADs, Lights Out 3×3 → 5×5 (solvable +
  certified unsolvable; unsolvable = bare refusal), grid pathfinding.
- *Stage 2 — constraints, structured generation, retrieval.* Logic puzzles,
  JSON schema adherence, symbolic lookup with missing-key refusal,
  templated retrieval QA with abstention, note-taking QA, tool-call
  simulation.
- *Stage 3 — math.* Arithmetic → multi-step → symbolic manipulation with
  step-by-step traces, style variation with disjoint train/test styles,
  assumption flips, domain conditioning (`REAL|COMPLEX|IEEE`).

## Comparison grid

Intended axes: looped+ACT (headline) × looped `fixed {1,2,4,8,16}` ×
non-looped at matched params × non-looped at matched compute × blocks `{1,2,4}`,
≥3 seeds per cell.

**Implemented and manifest-reachable today:**

- the stop-mode axis: `act` / `fixed {loops}` / `converge` via the `stop` key.
  `stage0-4block.json` (ACT) vs `stage0-fixed4.json` (fixed ×4) differ only
  in `stop` plus the `ponder_weight` zeroing that fixed depth requires and
  the `ckpt_dir`; `stage0-converge.json` covers the convergence control.
- the order axis: `stage0-orderaug.json` sets `stop.act.shuffle_train`, and
  differs from `stage0-4block.json` in that key plus its `ckpt_dir`. Run, and
  it is the one axis in the grid with a settled positive result; see "Order
  augmentation" in "Current results".
- the data axis, by `train.max_train_instances` (cap the pool after the
  holdout, eval set held fixed). `experiment.per_cell` also varies the pool but
  moves the eval set with it, so it is not a sweep axis.
- the size axis, by editing `model.d_model` / `ffn_hidden` / `n_stages`. Run.
- the depth-vs-width columns, which are the `model` block again
  (`n_stages`, `max_loops`, `d_model`, `ffn_hidden`) rather than a new knob:
  `research/experiments/compute-matched{,-wide}.json` match parameters *and*
  compute, `paramsfree-{looping,wide}.json` match compute with parameters
  free, and `frontier-s*s*l*.json` traces the stages↔width split at matched
  parameters and compute. All the parameter counts were solved with
  `examples/param_match.rs`, never by hand. Results: "Depth vs width".
- the context axis, by editing `protocol.k_set`.
- the seed axis, by editing `train.seed` or `--seeds=` on the sweep tool.

`examples/scaling_sweep.rs` drives the named axes (`data` | `depth` |
`params` | `k`) over a base manifest, reads both accuracies and the gap out of
each run's `run.jsonl`, and reports mean halt and the profile reading at every
point so a fixed-depth win stays distinguishable from a degenerate-halting
artifact. The depth-vs-width arms run as two separate sweeps: `arm_compare`
refuses to pair them, correctly, because the intervention is several model
fields at once rather than one.

`examples/arm_compare.rs` drives the order axis as a paired two-arm comparison
(`<A.json> <B.json>`, N seeds), one seed per run, each run's checkpoint
directory carrying both its arm and its seed so the arms cannot overwrite each
other. It is log-driven rather than return-value-driven: each arm is run, and
both halves of each run are then read back out of that arm's `run.jsonl`, so a
derived number always has a source in a log and a rerun is what verifies it. It
**refuses** to report a comparison whose arms differ by more than the
intervention — model, experiment, optim, steps, LRs, batch, and every eval knob
are compared field by field, the two `ckpt_dir`s excepted, and both arms are
required to be ACT. That refusal is the point: without it, a config edit that
moves two knobs produces a difference that gets attributed entirely to the one
under test, which is worse than no result because it looks clean. Its report
also prints the random-weight control's informativeness next to the collapse,
so a collapse is never printed as role evidence where the control could not
have measured a floor.

## Current results

Nine measurements exist, plus a throughput note; a tenth is in flight. They do
not all support the same reading, so read them in this order:

| # | measurement | rung | n | reading |
|---|---|---|---|---|
| 1 | data axis, constant rule | `subst-fst-fixed` | 3 / point | transfer rises with data — and is map application, not generalization |
| 2 | data axis, varying rule | `subst-fst` | 3 / point | flat at chance over 18× data; refutes item 1 |
| 3 | size axis | `subst-fst` | 3 / point (256 is n=1) | flat at chance over 6.5× parameters |
| 4 | compute-depth axis | `subst-fst-fixed` | 3 / point (8 is n=2) | **superseded by 5–6** — it moved depth and compute together |
| 5 | depth vs width, matched params *and* compute | `subst-fst-fixed` | 3 / arm | width wins 2:1 |
| 6 | depth vs width, matched compute, params free | `subst-fst-fixed` | 3 / arm | 7.07× params buys 2.2× held-out |
| 7 | context axis (`k`) | both `subst-fst` rungs | 3 / point | demos are not evidence; on the memorized rung one demo halves held-out. Cited under "Depth vs width", not given its own section |
| 8 | Muon LR sweep | `subst-fst-fixed` | 2 × 3 | the two learning rates are not distinguishable |
| 9 | order augmentation | 6-task suite | 3 / arm | the one settled positive result |
| — | stages↔width frontier | `subst-fst-fixed` | in flight | **not for citation** |

Everything below reports the cell, the instrument and its floor, and states the
limits. The LR sweep came first and is the weakest. The size axis is a genuine
negative that narrows the question without answering Goal question 1. The depth
axis is a first pass whose earlier conclusion has since been **refuted** by 5
and 6, and the correction is stated in place rather than by deletion.

### Data axis, constant rule: the memorization-onset curve

The instrument is the train/held-out gap ("Measuring the gap"), and the axis is
`train.max_train_instances`, so the eval set is the same ~29 held-out instances
at every point. 492,418-parameter model (`d_model` 128, `n_stages` 2,
`max_loops` 4), 600 steps, auto-batch on, batch 6 × indistinguishable 12, 3
seeds per point, RTX 3050 4 GB. Task `subst-fst-fixed`: one constant
substitution map for the whole pool, `k_set: [0]`.

| train N | byte-in | byte-out | gap | exact-in | exact-out |
|---|---|---|---|---|---|
| 16 | 0.944 | 0.329 | -0.614 | 0.688 | 0.000 |
| 48 | 0.988 | 0.433 | -0.555 | 0.885 | 0.000 |
| 128 | 0.985 | 0.704 | -0.281 | 0.906 | 0.312 |
| 288 | 0.977 | 0.850 | -0.127 | 0.938 | 0.478 |

In-distribution accuracy is 0.95–0.99 at every data size, and **the gap narrows
monotonically as data grows**. More data buys transfer here, not memorization —
the opposite sign to what a pure-memorization account predicts.

**What "transfer" means on this rung, stated before the next table.** The rule
is one constant map for the entire pool, so the held-out curve measures how
well the model applies a map it was trained on to inputs it was not. It is a
real measurement and the table stands; it is not evidence about the Goal's
questions 1 and 3. The next subsection runs the identical axis with the rule
redrawn per instance, and the transfer does not survive: 18× more data buys
nothing above chance once the map has to be induced.

**Onset was not observed.** The held-out curve is still rising at 288
instances and the gap is still -0.127, so the crossover lies beyond the right
edge of this sweep and the axis has *not bracketed* the onset. A gap of 0.127
with held-out byte accuracy at 0.850 is not a saturated model. What the table
does fix is the direction of travel; extending the axis is the open item, not
this table. The rise is a rise in map application on a fixed rule, so it does
not locate an onset in rule transfer either — see limitation 2.

**This refutes the earlier reading of the same axis.** A fixed-batch run at 300
steps and batch 6 put held-out exact-match at 0.000 at the two smallest data
sizes and 0.079 / 0.111 at the two largest, which was read as "onset lies
below 16 instances" and "the model is in the pure-memorization regime
throughout". That run was badly undertrained. The confound was the ordinary one
— at a fixed step count, more data is fewer epochs — and the artifact is
visible in that same table, where in-distribution accuracy *fell* as N rose
(0.708 → 0.500), which is a budget effect, not a law. The rerun with the budget
and the card fixed does not reproduce it. Anyone re-deriving the old numbers
from commit `8bcffc6` should read this table instead.

```bash
cargo run --example scaling_sweep -- \
    research/experiments/data-axis-constant-rule.json data gpu 16 48 128 288 --seeds=0,1,2
```

### Data axis, varying rule: 18× more data and no transfer

Same instrument, same model, same budget as the table above — 492,418
parameters (`d_model` 128, `n_stages` 2, `max_loops` 4), 600 steps, auto-batch
on, ~46 held-out instances, 3 seeds per point, RTX 3050 4 GB. The **only**
change is the task: `subst-fst` instead of `subst-fst-fixed`. `subst-fst`
redraws the substitution map per instance — now pinned by a test, so the two
tasks cannot silently converge — which means every held-out instance carries a
map the model never saw. **k ∈ {1, 2, 3, 5} with `k0_rate` 0, so the rule is
always demonstrated in the prompt** — the model is shown the map before being
asked to apply it.

> **A confound that was in the first pass of this table, and what it cost.** The
> first run of this sweep inherited `k_set: [0]` from the constant-rule base, so
> every instance had **zero in-context demonstrations**: the rule was never
> shown, only implied by a constant the model had to hold in weights. It
> measured 0.249 / 0.315 / 0.297 / 0.311 and was read as "at chance on unseen
> rules" — a conclusion that configuration could not support, since with no
> demonstration there is nothing in the prompt to induce a map *from*. The
> numbers below are the rerun at k ≥ 1. The headline conclusion happens to
> survive, but it survived a test it could not have passed, and the earlier
> table should not be read as evidence about rule induction. The old numbers
> are at commit `16079a8`.

| train N | byte-in | byte-out | gap | exact-in | exact-out |
|---|---|---|---|---|---|
| 16 | 0.963 | 0.181 | -0.783 | 0.792 | 0.000 |
| 48 | 0.970 | 0.241 | -0.729 | 0.812 | 0.000 |
| 128 | 0.950 | 0.218 | -0.731 | 0.802 | 0.000 |
| 288 | 0.888 | 0.265 | -0.623 | 0.635 | 0.000 |

Held-out byte accuracy is 0.18–0.27 and **flat across an 18× range of training
data**. Chance on a 3-symbol alphabet is 1/3 = 0.333, so the model is **at or
below chance on unseen rules at every point on the axis**, and in-distribution
exact-match is 0.64–0.79.

| | constant rule (`subst-fst-fixed`) | varying rule (`subst-fst`) |
|---|---|---|
| held-out byte accuracy, N = 16 → 288 | 0.329 → 0.850 | 0.181 → 0.265 |
| gap, N = 16 → 288 | -0.614 → -0.127 | -0.783 → -0.623 |
| against the 0.333 chance rate | rises from below to well above | **below it at every point** |
| rule shown in context? | n/a (it is in the weights) | yes, k ∈ {1, 2, 3, 5} |

**Being below chance is a stronger statement than being at chance, and the two
obvious explanations for it are both refuted by measurement.** At chance would
mean uniform guessing; below chance means systematically wrong. Two policies
could produce that — replaying a demo output, or echoing the query input — and
both were checked against the run log at N = 288, seed 0:

```
eval [final-trained-pool]: acc 0.00 byte 0.273 copy 0.00 echo 0.00
```

`copy` (output appeared among demo outputs) and `echo` (output equalled the
query input verbatim) are **both 0.000**. The model is not replaying anything it
was shown and is not echoing the question, yet it lands below the 0.333 chance
rate. **The mechanism is therefore unidentified**, and the honest statement is
"systematically wrong, by an unidentified mechanism", not a story about
copying. `query_echo_rate` was added to `Record` and `Summary` specifically to
test this and earned its place by failing to explain the result. The remaining
diagnostic — the per-track breakdown, since a mean over Track A and Track B can
hide one at chance and the other near zero, plus the emitted strings themselves
— does not exist yet.

**These are two rungs of one task family, not contradictory measurements.** The
two differ only in whether the map is constant or induced, and that single
difference is what separates "the model generalizes" from "the model applies a
memorized map". The constant-rule table is unchanged by this one.

**This is not undertraining, and it is not a gap artifact.** In-distribution
exact-match is 0.64–0.79 while held-out is at or below chance, and
in-distribution accuracy *falls* as N grows (0.963 → 0.888), so the
undertraining confound that produced the refuted "onset below 16" reading cannot
be the cause here. Nor can the gap drift mean transfer: in-distribution was
pinned near ceiling with no room to fall, so -0.722 → -0.666 is the
near-immutable side moving while the held-out side never left chance.
`scaling_sweep.rs` now checks whether the held-out curve actually moved
**first** and suppresses the gap reading when it did not.

**What this rung does not separate.** There is no positive control here:
nothing in this table shows the model can induce a *new* map from in-context
demonstrations at this size and budget, so "at chance on unseen rules" is a
statement about the rung, not a diagnosis of which sub-skill fails.

```bash
cargo run --example scaling_sweep -- \
    research/experiments/data-axis-varying-rule.json data gpu 16 48 128 288 --seeds=0,1,2
```

### Size axis: 6.5× the parameters, flat at chance

`d_model` swept by `scaling_sweep`'s `params` axis, which derives
`n_heads = d_model / 64` and holds `head_dim` at 64, so the change is width and
not head layout. Same rung as the varying-rule table above — `subst-fst`,
`k ∈ {1, 2, 3, 5}`, `k0_rate` 0, 600 steps, auto-batch on (the tuner holds the
*effective* batch fixed, so samples per optimizer step, and therefore epochs
over the pool, are constant across sizes), ~44 held-out instances, 3 seeds per
point. Instruments: held-out **byte accuracy** with `off_pair_rate` beside it —
`off_pair` is the count of positions emitting a symbol that is neither the
query's nor the target's, and it is the readout a positional bias cannot game.

| `d_model` | params | held-out byte | held-out `off_pair` | in-dist byte |
|---|---|---|---|---|
| 128 | 492,418 | 0.269±0.03 | 0.318±0.02 | 0.926±0.01 |
| 192 | ~1.1M | 0.263±0.02 | 0.331±0.02 | 0.763±0.12 |
| 256 | ~1.8M | 0.340 (**n=1**) | 0.330 | 0.904 |
| 384 | ~3.2M | 0.280±0.01 | 0.339±0.01 | 0.840±0.12 |

Chance is 0.333. Every point sits at or below it, and `off_pair` is flat at
0.32–0.34 across a **6.5× parameter range**. So "a bigger model would have
induced the rule" is refuted: 6.5× the parameters buys nothing above chance on
the strong generalization rung. `d_model` 256 is **n=1** — seeds 1 and 2 were
lost to the concurrency mistake described under "Next experiments".

**Two limits, and they matter more than the null.** First, this sweep ran on
the **varying-rule** rung, so it is evidence about rule *induction*; it is
**not** an answer to Goal question 1, which is about the same-rule rung where
held-out byte rises 0.329 → 0.850 with data and where a scaling law is
actually measurable. That rung has not been swept across size at all. Second, a
flat result is equally consistent with "capacity does not help here" and "this
instrument cannot see a weak competence", because every point sits on a
3-symbol alphabet where chance is 0.333 — see "Measuring the gap", instrument
3. In-distribution byte accuracy is non-monotone (0.926 → 0.763 → 0.904 →
0.840) with ±0.12 spread at two of the four sizes, so it is noise at n=3 rather
than a size effect.

```bash
cargo run --example scaling_sweep -- \
    research/experiments/size-axis-varying-rule.json params gpu 128 192 256 384 --seeds=0,1,2
```

### Compute-depth axis: a first pass, and a conclusion that was wrong

Fixed batches (auto-batch **off**), batch 6, 600 steps, 492,418 parameters
(`d_model` 128, `n_stages` **2**), `subst-fst-fixed`, `k_set: [0]`, ~29 held-out
instances, 3 seeds per point (2 at depth 8). Instrument: byte accuracy. **These
logs predate `off_pair_rate`, `length_exact_rate` and `mean_len_ratio`** — none of
the three exists on this axis.

The manifest sets `stop.act`, not `stop.fixed` — but ACT **saturated at its cap
on every run**: halt takes exactly one distinct value per point and it equals
`max_loops` (1.000 / 2.000 / 4.000 / 8.000) in every seed, verified from the
per-instance `halt` field. So the axis is equivalent to fixed depth at the cap,
and the block-steps column below is the realized arithmetic rather than a
configured intention.

| depth | block-steps/token | byte-in | byte-out | gap |
|---|---|---|---|---|
| 1 | 2 | 0.201 | 0.134 | -0.067 |
| 2 | 4 | 0.406 | 0.324 | -0.082 |
| 4 | 8 | 0.444 | 0.403 | -0.041 |
| 8 | 16 | 0.474 (n=2) | 0.433 (n=2) | -0.041 (n=2) |

Depth 8 is n=2: the third seed's log is not on disk, and the depth-8 row this
README used to carry (0.405 / 0.387 / -0.018 at n=3) is not supported by the
surviving `run.jsonl` records. The numbers above are read from those records, and
seed 2 should be filled in before anything is claimed about the 4-vs-8
difference.

**The conclusion this section used to carry was wrong, and the reason was a
confound.** It read: *"depth helps, the gap narrows with it, extra compute buys
transfer rather than memorization."* That is a statement about **compute**, not
about reuse. `n_stages` was pinned at 2 while `max_loops` varied, so depth 1 → 8
was 2 → 16 block-steps per token: arithmetic and weight-reuse rose **together**,
and this table cannot separate them. Holding compute fixed and spending the same
parameters on width instead is ~2× better on held-out byte accuracy — 0.625
against 0.323 at matched parameters *and* matched compute (comparison (i)
below). So the honest reading of this axis is "**more compute helps**", and the
reuse question is not touched by it. The old numbers are in commits `d8b7aa9`
and `0e0d15a`; the confound and its correction are `eeb9257` and `ddddc9e`.

Three things the table still supports, none of them about reuse:

- depth 1 → 8 moves held-out byte 0.134 → 0.433, **+0.30 absolute, 3.2×
  relative**, the largest single effect measured anywhere in this repo — but most
  of it is "depth 1 does not learn this task" (in-dist 0.20). The 4 → 8
  difference is **not** resolved: within-seed spread at depth 8 (0.380–0.485
  held-out) exceeds it, at n=2 on top of that.
- the gap does not widen with depth, so nothing here is the signature of
  growing memorization. But a near-flat gap only carries information when the
  in-distribution term is off the floor, and at depth 1 it is not.
- **Exact-match was 0.000 on both halves at every depth.** On exact-match alone
  this table reads "depth does nothing" — the metric failure described under
  "Measuring the gap", not a null result. Nothing else about the axis would
  have survived without byte accuracy.

The profile read "uniform" at every depth, and the axis is `fixed` only, so no
point here is compute-matched to any other and nothing here speaks to ACT.

```bash
cargo run --example scaling_sweep -- \
    research/experiments/depth-axis.json depth gpu 1 2 4 8 --seeds=0,1,2
```

### Depth vs width: three comparisons, and what each can answer

All cells below: `subst-fst-fixed`, `k_set: [0]` with `k0_rate` 1.0 (zero
in-context demonstrations), 600 steps, ~29 held-out instances, **n = 3 seeds per
arm**, RTX 3050 4 GB. Instruments: held-out **byte accuracy**, exact-match beside
it, and the in-minus-out gap from the same run, against a **0.333 chance floor**
on this 3-symbol alphabet.

Two cell details differ between the comparisons and are stated per comparison
rather than averaged over. **Stop mode is `act` on all of them**, inherited from
`base-492k.json` — but ACT never halted early here: mean halt equals `max_loops`
exactly at every point (16, 8, 4, 2, 1), so the realized arithmetic per token was
exactly `n_stages × max_loops`. **The compute matching is therefore verified by
the halt profile, not merely by configuration**, which is the stronger form.
Batch differs too: (i) pins `auto_batch: false, batch_size: 6`; (ii) inherits
`auto_batch: true` and both arms independently settled on an effective batch of
128 — verified post hoc from the sweep log, so the arms match, but by luck
rather than by construction.

Three comparisons exist in this repo and **they are not interchangeable** —
each holds different quantities fixed, and one of them was initially reported as
the research goal's answer, which was wrong. All parameter counts were solved
with `examples/param_match.rs`, not by hand; hand-matching got the first attempt
50% wrong twice.

#### (i) Matched parameters *and* matched compute — reuse pattern at equal storage

Both arms 32 block-steps/token and ~919K parameters, matched to 0.05%.

| arm | params | byte-in | byte-out | exact-out, per seed |
|---|---|---|---|---|
| looping: 4 stages × 8 loops, `d_model` 128, 8 heads | 919,172 | 0.390 | **0.323** | 0.000 / 0.000 / 0.000 |
| wide: 32 stages × 1 loop, `d_model` 64, 4 heads | 919,648 | **0.849** | **0.625** | 0.000 / 0.219 / 0.067 |

Complete separation: wide's **worst** seed (0.567) beats looping's **best**
(0.355), far outside the seed spread. At equal arithmetic and equal storage,
distinct parameter sets beat 4 reused ones ~2× on held-out byte accuracy and by
a factor of several on exact-match. Reuse does not buy generalization without
proportional parameters; it buys it worse than the same parameters spent on
width would.

*One number to correct before someone re-derives it: the per-seed held-out
exact-match here is 0.000 / 0.219 / 0.067 (mean 0.095). The triple 0.083 / 0.792 /
0.625 in commit `eeb9257` is this arm's **in-distribution** exact-match,
mislabelled as held-out; it is corrected in place in `research/findings.md`.*

**The mechanism is not mysterious.** Memorization is storage in weights, so at
equal parameters the storage is equal and looping ought to be *neutral*. It is
instead **worse at memorization** (0.390 against 0.849). The cause is the reuse
itself: 4 weight sets are asked to serve 32 transformations' worth of function.

**A nuance that cuts the other way, stated because it would be easy to omit.** In
*relative* terms the looping arm retains more of what it learned — 0.323 / 0.390
= 83% of its in-distribution accuracy transfers, against 0.625 / 0.849 = 74% for
wide. Per unit of learning, reuse is slightly better at transfer. It simply
learned far less, and absolute held-out accuracy is what a scaling law is about.

**What this comparison cannot answer.** It hands both arms the same storage, so
depth cannot win on the parameter axis by construction. It answers *"does the
reuse pattern matter at equal capacity?"*, not the goal's question.

**Uncontrolled difference.** Head **count** is 8 against 4 and cannot be
matched — an 8× stage ratio forces an ~8× body-parameter ratio. `head_dim` (16)
*is* matched. The effect is far too large for head count to explain it, and
(ii) removes the confound entirely.

#### (ii) Matched compute, parameters free — the goal's question

Identical `d_model` (128), `n_heads` (2), `head_dim` (64), `ffn_hidden` (384)
and identical compute (16 block-steps/token). Only `n_stages` / `max_loops`
differ, so head geometry and activation memory are identical and the head-count
confound above is gone.

| arm | params | byte-in | byte-out | exact-in | exact-out |
|---|---|---|---|---|---|
| looping: 2 stages × 8 loops | 492,418 | 0.309 / 0.278 / 0.418 | **0.335** | 0.000 | **0.000** |
| wide: 16 stages × 1 loop | 3,479,696 | 1.000 / 0.985 / 0.923 | **0.739** | 1.000 / 0.833 / 0.667 | **0.270** |

**7.07× the parameters buys 2.2× the held-out byte accuracy** (0.739 against
0.335) and takes exact-match from 0.000 to 0.270.

**A trap in this table, stated because it is easy to report backwards.** The
looping arm's gap is **zero** — per seed −0.004 / **+0.020** / −0.017, mean
−0.000, one seed positive — while its in-distribution byte accuracy sits at the
**0.333 chance rate**. Read together, that is not perfect transfer: the model
memorized **nothing at all**, in-distribution or out. Its 100% relative transfer
(0.335 ÷ 0.335) is transfer of nothing and is not a transfer advantage. The wide
arm's gap is −0.144 / −0.314 / −0.231, mean −0.230: it memorized, then
transferred 76% of it. The relative-retention nuance in (i) is the same trap at
smaller scale, and this table is where it becomes dangerous: a ratio of two
chance-level numbers carries no information at all.

#### Why width wins here, and why that is a scope limit rather than a verdict

Three independent measurements say this rung is **storage-limited, not
compute-limited**:

1. On `subst-fst-fixed` the held-out answer is *"apply a map already in your
   weights"*. That needs **storage of the rule**, not arithmetic to apply it.
2. The context axis says in-context demonstrations are **unused**. On the
   same-rule rung, held-out byte falls **0.909** (k=0) → **0.519** (k=1) →
   **0.448** (k=2) and held-out exact-match 0.59 → 0.01 → 0.00, n=3 per point,
   same cell, while in-distribution byte barely moves (0.993 → 0.965). One
   demonstration *halves* held-out accuracy: the model attends to demos and
   does worse. Demos are not evidence, so there is no sequential inference for
   extra loops to be spent on.
3. Depth 1 sits at in-distribution byte **0.201 — below the 0.333 chance rate**
   on data it trained on. A memorizing model cannot fail below chance on its
   own training set; it is confidently applying a *wrong* map. A learning
   failure, not a compute failure.

**So "depth loses" is NOT established, and "depth was never tested by this task"
IS.** Any claim that reuse cannot buy generalization has to survive a rung whose
answer requires several dependent steps and cannot be looked up.

#### The rung monoculture — a methodological result in its own right

Every architecture experiment in this repo ran on `subst-fst-fixed` and nothing
else. **All nine** architecture manifests — the compute-matched pair, the
params-free pair and all five frontier points — resolve to `pool instances: 320`
(= `per_cell` 160 × 2 tracks). Each was confirmed by preflighting it, because
`jq '.experiment.tasks'` returns `null` for a manifest that inherits its
`experiment` block, and that `null` reads like a missing setting when it is not
one.

**Config trap, recorded because it is silent: an absent or empty
`experiment.tasks` key means ALL EIGHT builtin rungs, not none.** A manifest
inheriting from a base with no `tasks` is a silent pooled mixture of the whole
registry, and it produces a perfectly plausible number. See
`research/experiments/AGENTS.md`, contract 7.

**The missing measurement was a missing sweep, not a missing task.** Four of the
eight registered rungs already require dependent multi-step computation —
`dyck1`, `scan-tiny`, `parity`, `periodic` — and none has ever been swept for
architecture. `dyck1` is the best depth probe of the four: its answer is
`")".repeat(open_depth(input))`, a function of the input's stack state, so no
memorized map shortcuts it; its held-out prefixes run 8–28 characters against
demo prefixes of 2–10, so the computation required grows ~3–14× at test time;
and Track B holds out nesting depths 4–6, which never appear in training. Read
"Measuring the gap" instrument 3 before running it — the output alphabet is
unary, and byte accuracy alone will mislead.

#### Stages↔width frontier — in flight, not for citation

Five points at matched parameters (within 0.10%) and matched 16
block-steps/token, sweeping only the stages↔width split: 1×16, 2×8, 4×4, 8×2,
16×1, all on `subst-fst-fixed`, n=3 planned per point.

> Partial, in flight, not for citation: at identical parameters and identical
> compute, the 1×16 point reads held-out byte ≈0.82 against 2×8's 0.32. If that
> holds, reuse is not monotonically bad — the worst configuration is 2 stages,
> not maximal reuse — which undercuts a simple "reuse is storage-inefficient"
> account. It is confounded with width (d_model 128 → 176, ffn 384 → 526) and
> untested on a compute-requiring rung.

Two confounds are already recorded in those manifests, before running: the
width sequence is **not** monotone (`d_model` 176 → 128 → 64 → 64 → 80) because
the solver takes the smallest width that reaches the target and `ffn_hidden`
collapses instead at high stage counts (512 → 213 → 10, which makes the 16×1
endpoint a degenerate-FFN corner); and head count varies unavoidably, because
trading stages for width at fixed parameters *is* trading heads for stages.

### Throughput, and why the batch tuner exists

| batch | steps/s | instances/s | memory used |
|---|---|---|---|
| fixed 8 | 16.8 | 134 | 190 MB |
| auto (B×T 8192, window 128) | 7.8 | 998 | 1470 MB |

7.4× the data per second for 7.7× the memory. Steps per second *drops* — from
16.8 to 7.8 — because each step is now 16× the work, so **throughput is not
comparable at fixed step counts** and two runs are only comparable at equal
instances seen. That is the same confound as the undertrained data sweep under
"Data axis, constant rule" above, and it is why the data-axis results are stated
in instances rather than steps.

### Optimization: a Muon learning-rate sweep

Task `subst-fst-fixed` (weights-only rung: constant substitution, `k_set: [0]`
→ zero demonstrations), `configs/stage0-fixed.json`: 500 steps, 4 stages × 8
max loops, batch 6 × accum 2, held-out n = 96 per run (48 per track × 2
tracks, both sampling the *same* constant rule), **2 LRs × 3 seeds**, RTX 3050
4 GB.

| `lr_muon` | s0 | s1 | s2 | mean | spread | mean halt |
|---|---|---|---|---|---|---|
| `2e-3` | 0.583 | 0.760 | 0.719 | 0.687 | 0.177 | ~22.6 |
| `1.5e-2` | 0.688 | 0.792 | 0.802 | 0.761 | 0.114 | ~9.6 |

**The two learning rates are not distinguishable.** The between-arm
difference in means (0.074) is smaller than the seed-to-seed spread inside a
single arm (0.177). Three seeds is not enough to separate an effect this
size. `lr_muon = 2e-3` in `configs/stage0-base.json` (the `max_lr` of a linear
warmup from `2e-4`) is a **reasonable pick, not a validated one**.

An earlier single-seed sweep returned 0.833 for *both* `2e-3` and `1.5e-2`.
**That number did not reproduce** and was noise. It is recorded here only so
nobody re-derives it from commit `c30b9ec` and believes it.

**The compute result is the interesting one, and it cuts against the depth
story.** Per-stage mean halt (seed 0, over 96 instances):

| `lr_muon` | stage 0 | stage 1 | stage 2 | stage 3 |
|---|---|---|---|---|
| `2e-3` | 6.1 | 6.0 | 6.1 | 6.4 |
| `1.5e-2` | 5.2 | 1.6 | 1.0 | 1.0 |

(Seed 0 only, which is why these need not sum to the mean-halt column above
— that column averages all three seeds.)

The higher LR reaches comparable accuracy with ~2.3× less compute — but it
does so by **not using stages 1–3**, which halt after about one step. That is
degenerate early halting, not smarter depth allocation. A depth axis built
on mean-halt `L` would read this as "the model discovered it does not need
depth", and would be wrong about the mechanism. This distinction is the
central confound in the depth-scaling question and is called out again under
"Next experiments".

Reproduce with:

```bash
cargo run --example lr_sweep -- \
    configs/stage0-fixed.json gpu 2e-3 1.5e-2 --seeds=0,1,2
```

Per-run logs land in gitignored `checkpoints*/run.jsonl`, so the table is not
a checked-in artifact — re-running is how to verify it. `lr_sweep` reports
**held-out** accuracy (never training loss: a lower LR trades training loss
for map application here — not rule generalization, see limitation 1) and
prints an explicit single-seed caveat, quoting
the binomial SE at the observed n, when `--seeds` is omitted.

### Order augmentation: order-dependence is learned, not structural

The causal test from "Reading the stages", run on the premise the role
diagnostic rests on. `configs/stage0-orderaug.json` against
`configs/stage0-4block.json`, differing only in `stop.act.shuffle_train`:
919,172-param model (`d_model` 128, `n_stages` 4), 500 steps, ACT, the full
6-task suite at k ∈ {0,1,2,3,5,8}, 576 held-out instances per eval pass,
3 seeds per arm, RTX 3050 4 GB. The collapse is exact-match accuracy in trained
order minus the same in reversed order, over the same instances.

| | ordered (A) | order-augmented (B) |
|---|---|---|
| reversal collapse (mean of 3 seeds) | 0.179 | 0.049 |
| per-seed collapse | 0.210, 0.207, 0.120 | 0.073, 0.023, 0.052 |
| held-out byte accuracy | 0.382, 0.380, 0.384 | 0.342, 0.379, 0.371 |
| profile reading | "uniform" / "one global head (flat + correlated)" | "DIFFERENTIATED (skewed + independent)" in 2 of 3 seeds |
| shuffled-order accuracy | 0.000 on all 3 seeds | 0.069, 0.096, 0.118 |

Order augmentation cut the collapse by 72% (3.6× smaller), so **order-dependence
is learned, not structural.** A model that collapses under reversal only
because stages are wired in sequence would have collapsed just as hard with the
order resampled during training; this one did not. So the trained-order
collapse does carry information about learned roles, and the role diagnostic
stands as an instrument.

Two independent signals agree, and neither is the collapse. The profile reading
needs no control at all, and it moves from uniform / one-global-head in the
ordered arm to the first "DIFFERENTIATED" reading anywhere in this repo. And
shuffled-order accuracy stops being exactly zero, so the reversed order is no
longer fatal.

**Held-out byte accuracy barely moves (0.382 → 0.364 mean), and it is not what
the arms are compared on.** That is the expected price, not a failure: the
order-augmented arm is solving a different, permutation-robust function, so its
training loss was never comparable to the ordered arm's and its accuracy need
not be either. The arms are compared on the collapse and the profile — **not**
on accuracy or loss.

**The random-weight control is still degenerate, on both arms.** It read 0.000
in *both* orders on both arms, so `control_informative: false` and
`roles_supported: null` on every run. The collapse sizes above are real and
measured; what is unconfirmed is that they exceed a chance floor. The causal
test therefore says "learned, not structural" while the shuffle control still
declines to certify the collapse. Both statements are true, they are about
different questions, and neither substitutes for the other.

What this does not establish, stated plainly:

- **n = 3, and the per-seed ranges overlap.** Arm A spans 0.120–0.210 and arm B
  spans 0.023–0.073, so the 3.6× is a ratio of 3-seed means. The direction is
  consistent across every seed; the effect size is not resolved better than
  n=3, which is the same threshold described in limitation 5.
- **One operating point.** One model size, one depth, one suite, one step
  budget. "Order-dependence is learned" is a statement about this cell; it does
  not generalize to the depth and size axes the queue is about to sweep.
- **Not comparable to the tables above.** Both arms are measured on the same
  576 held-out instances, so the within-table comparison holds, but the ~29-
  instance splits used by the two scaling axes cannot be put in this table.
- **Provenance, and what is left to check.** Arm A's numbers were recovered from
  pre-existing logs through `arm_compare`'s per-instance fallback rather than
  re-run, because pool accuracy is defined as the mean of the per-instance
  `correct` flags and the tally reproduces it exactly. Only arm A's `run.jsonl`
  files remain on disk; arm B's were not retained. As with every other table
  here, nothing is a checked-in artifact — re-running is how to verify any of
  it, and the command below is the whole check.

```bash
cargo run --example arm_compare -- \
    research/experiments/arm-ordered.json research/experiments/arm-orderaug.json gpu 0 1 2
```

## Known limitations

Listed with what would change the conclusion, strongest first.

1. **The holdout is instance-level, not rule-class-level.** The measurement
   that sharpened this now exists. The same data axis was run on `subst-fst`,
   which redraws the substitution map per instance (pinned by a test), so every
   held-out instance carries a map the model never saw, demonstrated in the
   prompt at k ∈ {1, 2, 3, 5}: held-out byte accuracy is 0.18–0.27 and flat
   from 16 to 288 training instances, **below** the 0.333 chance rate for a
   3-symbol alphabet at every point, while in-distribution exact-match is
   0.64–0.79. So the 0.329 → 0.850 rise on `subst-fst-fixed` was never
   generalization. It was learning to apply one memorized constant map to new
   inputs, and that transfer does not survive requiring a fresh map per
   instance.
   **The `subst-fst-fixed` numbers are unchanged by this and remain what they
   always were** — a real measurement of map application on a constant rule,
   including the 0.687–0.761 LR-sweep figures, which measure transduction of a
   memorized rule. Measuring the gap does not soften any of it: a gap between
   two accuracies on the *same* constant rule is a gap measured on one rule.

   **What is still open, and it is the half that matters.** Neither rung holds
   out a rule *class*. Both measure transfer **within one task family**:
   `subst-fst` varies the instance's map, but the family, the procedure, the
   alphabet and the output format are the ones that were trained on. Holding
   out an entire task family or procedure — trained on some procedures,
   evaluated on one never seen in any form — **is not implemented at all**, and
   it is what Goal questions 1 and 3 actually require: "scales with parameters"
   and "where memorization begins" are both claims about an axis the model has
   never been on. Until that exists, the honest answer to "does this model
   generalize?" is **no measurable transfer on the one rung where the rule
   varies**, and the sweep says nothing about questions 1 and 3.

   Two smaller open points ride on this item. The varying-rule rung has **no
   positive control**: nothing in it shows the model *can* induce a new map from
   in-context demonstrations at this size and budget, so "below chance" names
   the failure without isolating which sub-skill is missing. The mechanism is
   specifically **unidentified** — the two obvious candidates (replaying a demo
   output, echoing the query input) are both measured at 0.000 — and the rung
   reports one number per point, so whether the reading is uniform across Track
   A and Track B is not established. A per-track breakdown plus the emitted
   strings is the concrete next step.
2. **Memorization onset has not been bracketed.** The constant-rule data axis
   stops at 288 training instances with the held-out curve still rising and the
   gap still open at -0.127, so the crossover is beyond the right edge of the
   measured range. What that table establishes is the *sign* — more data closes
   the gap on this rung — not where the sign would flip. **The varying-rule rung
   cannot bracket onset either, and for a different reason: its held-out curve
   never rises at all**, sitting flat at chance from 16 to 288 instances.
   Onset requires the held-out curve to first rise and then fall, and that rung
   has no rise to fall from. What it shows instead is that the low-data end is
   already saturated on the fitting side — in-distribution exact-match is
   1.000 at three of four sizes — so the whole measured range sits on one side
   of any crossover. Onset also needs
   the other two axes first, because onset in (params × depth × data) space is a
   joint location and only one coordinate has been swept, and it needs the
   rule-class holdout in item 1 before "memorization" can mean anything other
   than fitting a map the model was given.
3. **The depth axis has one pass, at one operating point, and its earlier
   conclusion is refuted.** 492,418 params, fixed batches, `fixed {1,2,4,8}`
   only, 3 seeds (2 at depth 8). No ACT point and therefore no
   compute-matched ACT-vs-fixed comparison, no `loops: 16`, one model size, one
   rung. Within-seed spread exceeds every between-depth difference, so the
   apparent saturation is not resolved. The axis also moved compute and reuse
   together — see "Compute-depth axis" — so it measures arithmetic, not the
   reuse pattern, and the profile read "uniform" at every depth, so it says
   nothing about whether stages take on different roles with depth.
4. **The stage-order shuffle diagnostic is weak, and the control behind it is
   degenerate at the current operating point.**
   Reversing stage order collapses accuracy *by construction* — stages are
   sequential, so reversal reverses the data flow. The random-weight control
   added to give a floor (`src/harness/shuffle_control.rs`) returns 0.000 in
   *both* orders on weights-only rungs, so `control_gap` is 0.000 and
   `excess == trained_gap`: the control contributes nothing and the verdict
   reduces to the original unconfounded claim. The checked-in
   `checkpoints-fixed/run.jsonl` (pre-holdout, written before the control
   existed) makes this concrete: 0.771 trained-order vs 0.000 shuffled against
   a floor of 0.000 — an "excess" of exactly the whole collapse.

   The code now *refuses* to answer rather than answering wrongly: on that
   data `roles_supported` serializes as `null` with `control_informative:
   false`, and the human line says `roles UNKNOWN`. But an honest `UNKNOWN` is
   not a measurement.

   **What the order-augmented arm resolved, and what it did not.** It has now
   been run (`configs/stage0-orderaug.json`, 3 seeds, 919,172 params), and it
   is positive: the reversal collapse fell from 0.179 to 0.049, so the
   collapse is learned rather than structural and the "learned roles" framing
   stands — the earlier possibility that it needed replacing is closed. The
   other instrument here has not changed. The control read 0.000 on **both**
   arms, so `control_informative: false` and `roles_supported: null` hold on
   every run of this measurement too. The collapse is known to be real and
   learned; it is still **not certified to exceed a chance floor**, and the two
   facts do not substitute for each other. Closing that needs a rung where the
   untrained model scores above chance — see item 7 under "Next experiments".
5. **Single task, three seeds, and a smaller eval split than the LR sweep.**
   The scaling runs use ~29 held-out instances, so the binomial SE there is
   ~0.09 at p = 0.7 before seed variance, and the gap is a difference of two
   such numbers. No effect smaller than ~0.1 in byte accuracy is resolvable,
   which is the same threshold at which the depth axis stops being legible. The
   order-augmented comparison is outside this limitation on two of the three
   counts — full 6-task suite, 576 held-out instances per eval pass — but
   **not** on the third: it is also 3 seeds, with overlapping per-seed ranges,
   so its 3.6× is directional rather than resolved.
6. **The non-looped baselines exist — and they are one rung deep.** The
   compute-matched and params-free pairs under "Depth vs width" were built with
   `examples/param_match.rs` and run at n=3 per arm. They are the control the
   depth axis was missing, and they refute the reuse premise on `subst-fst-fixed`.
   What they cannot do is test depth, and that is not a defect of the design —
   it is a defect of the rung. **All nine architecture manifests resolve to
   `pool instances: 320`**, i.e. `subst-fst-fixed` and nothing else, whose answer
   is a lookup and therefore rewards parameters-as-storage while being
   indifferent to parameters-as-compute. What would change this conclusion: one
   architecture sweep on `dyck1`, `scan-tiny`, `parity` or `periodic`, where the
   answer requires dependent multi-step computation. Full diagnosis, the
   uncontrolled differences, and the config trap are under "Depth vs width".
7. **The size axis is measured, on the new-rule rung only, and it is flat at
   chance.** A **6.5× parameter range moves nothing**, so "a bigger model would
   have induced the rule" is refuted; the table, the cell and the instruments are
   under "Size axis", including the n=1 point at `d_model` 256. Two limits ride
   on it. It ran on the **varying-rule** rung, so it is evidence about rule
   induction and **not** an answer to Goal question 1: the same-rule curve,
   where held-out byte rises 0.329 → 0.850 with data, has not been swept across
   size at all, and that is where a scaling law is actually measurable. And a
   flat result is equally consistent with "capacity does not help here" and
   "the instrument cannot see a weak competence", because every point sits on a
   3-symbol alphabet where chance is 0.333. Goal question 4 (weight scale
   against compute depth) does have numbers now; see "Depth vs width".

## Next experiments

Ordered by how much each would reduce uncertainty. Items marked *(partly done)*
were started and the remainder is stated explicitly; item 0 is closed and is
kept here for the record, not as work.

0. **Run the order-augmented arm** *(done — the premise held, nothing about
   roles needs replacing)*. `stage0-orderaug.json` vs `stage0-4block.json`,
   3 seeds each, 919,172 params. Both instruments for reading the stages are
   now used and the arm returned the informative branch: order augmentation cut
   the reversal collapse by 72% (0.179 → 0.049), so order-dependence is
   **learned**, not structural. The "learned roles" framing stands, and the
   possible outcome that would have retired it — an order-augmented model that
   still collapses — did not occur.

   **What it buys the queue.** It was run before any further depth work, so the
   depth sweep is now known to be measuring learned order-dependence rather
   than a structural artifact of sequential stage execution: a collapse recorded
   at any depth is a fact about what the model learned at that depth, not a
   property of the wiring. That was the reason this was sequenced first, and it
   is discharged.

   **What is still not discharged.** The random-weight control read 0.000 on
   both arms, so `control_informative: false` and `roles_supported: null` stand
   on every run, and the collapse is still uncertified against a chance floor.
   That is what the oracle rung below is for. The closed test itself does not
   need rerunning, but anything read off it inherits its n=3.
1. **Run the architecture comparison on `dyck1`.** Every reuse conclusion so far
   is scoped to the one rung that cannot test it (limitation 6), and this leaves
   that scope. Designed and preflighted: `arch-dyck1-looping.json` (4 stages ×
   8 loops, `d_model` 128, `ffn_hidden` 384, **919,172 params**) against
   `arch-dyck1-wide.json` (32 stages × 1 loop, `d_model` 64, `ffn_hidden` 58,
   **919,648**), both 32 block-steps/token, pool 400 → ~40 held-out,
   `k_set [0]`, `k0_rate` 1.0, 600 steps, batch 6, auto-batch off, n=3 per arm.
   The same compute-matched pair as on `subst-fst-fixed`, on the rung whose
   answer is a function of the input's stack state and whose held-out prefixes
   are ~3–14× longer than its demos.

   Three outcomes count: the looped arm wins (reuse does work width cannot buy),
   they tie (depth was arithmetic in disguise), or width wins again (the premise
   is wrong on a rung that genuinely rewards computation). Read
   `length_exact_rate` and `mean_len_ratio` **first** — the output alphabet is
   unary, so byte accuracy alone cannot separate a right answer from a right
   length by luck, and the floor is the training-set modal depth, not 0.333
   ("Measuring the gap", instrument 3). The two arms differ in several model
   fields at once, so `arm_compare` will correctly refuse to pair them; run them
   as two sweeps.

   ```bash
   cargo run --example scaling_sweep -- \
       research/experiments/arch-dyck1-looping.json depth gpu 8 --seeds=0,1,2
   cargo run --example scaling_sweep -- \
       research/experiments/arch-dyck1-wide.json depth gpu 1 --seeds=0,1,2
   ```
2. **The representational-capacity decisive point, next on the same rung.**
   `4 stages × 4 loops` at `d_model = 256`, `ffn_hidden = 748` → **3,480,836**
   params, against `16 stages × 1 loop` at `d_model = 128`, `ffn_hidden = 384`
   → **3,479,696**. Matched parameters to 0.03%, matched block-steps to the
   step, `head_dim` matched, and `ffn:d` preserved at 2.92 against 3.00 so it is
   not the degenerate-FFN corner the frontier's `16×1` endpoint falls into. The
   looped arm gets 4× the width per stage at identical storage and arithmetic.
   If **representational capacity** is its deficit it should climb well above
   the looped arm's number toward the wide arm's; if **storage** is the deficit
   it will not move. Cheapest experiment that discriminates the two. Solve the
   widths with `cargo run --example param_match -- 4 4 3479696`.
3. **Rule-class holdout.** Train on some procedures, evaluate on one never seen
   in any form. **Not implemented at all**, and it is what a generalization
   claim needs; no amount of work on the two `subst-fst` rungs substitutes for
   it (limitation 1).
4. **The same-rule curve across model sizes.** Sweeping `d_model` on
   `subst-fst-fixed`, where held-out byte rises 0.329 → 0.850 with data, is the
   one measurement here that could produce an actual scaling law rather than a
   null. Needs the data axis at 3+ sizes, not a single size, and
   `same-rule-size.json` makes it reachable. This is also the only remaining
   route to Goal question 1 (limitation 7).
5. **A rung whose output space beats chance.** Every measurement on the
   new-rule rung sits on a 3-symbol alphabet where chance is 0.333, so a weak
   partial competence cannot be distinguished from guessing at all — and the
   6.5× capacity increase not moving the number is equally consistent with
   "there is no weak competence" and "the instrument cannot see one". A larger
   alphabet settles it. `dyck1` has the opposite problem — a unary alphabet
   where 0.333 is the wrong floor — so it is not the answer to this item.
6. **Seeds.** n=3 cannot resolve anything below ~0.1, and every axis here has
   within-seed spread at or above its between-condition differences. Fill in
   `d_model` 256 seeds s1/s2 and depth-8 seed 2 — item 9 below is why they are
   missing.
7. **De-confound the oracle rung.** Add ~20 chars of inert filler to
   `subst-fst` so both arms share a length distribution, isolating the header's
   content from its length. Needs a task variant, not a manifest. Until then
   "execution vs induction" is untested rather than answered. The same rung is
   also the one place an untrained model should score above chance — the map is
   in the prompt — which is what limitation 4 needs to make the shuffle control
   non-degenerate, and what limitation 1 needs as a positive control for
   induction.
8. **Bracket memorization onset.** The new-rule held-out curve never rises, so
   it cannot bracket onset; the same-rule curve rises 0.329 → 0.850 across an
   18× data range and shows no sign of turning over. Onset needs a curve that
   rises then falls, and no rung tested so far does that (limitation 2).
9. **Never run two sweeps at once on the 4 GB card.** Two auto-batch tuners
   means each measures its baseline before the other has allocated, and the
   watchdog kills the run at ~20 minutes with `CUDA_ERROR_DEINITIALIZED`. This
   is what cost the `d_model` 256 seeds and, earlier, the k=4 and k=8 points
   and depth-8 seed 2.

## Decisive read

- Looped+ACT holds where fixed variants collapse, halt rises with difficulty
  → adaptive computation, not memorization.
- Converge-holds / ACT-fails → the halting head is the failure, not recurrence.
- All configs track each other → tasks memorized; harden splits (rule-level
  holdout, not instance-level).
- All fail → learnability check: train one cell on oracle traces. Oracle
  fails too → data/capacity problem, not architecture.

Pre-registered bars (**targets, not measurements**): Track B (k≥2)
exact-match ≥ 80%; 2× length ≥ 60%; schema validity ≥ 95%; missing-key
false answers ≤ 5%; disjoint-style math within 10 pts of in-style.

## Non-goals

Natural language. Scale.

## Layout

- `src/model/` — looped transformer (config, attention, MLP, block, halting,
  masking, model, `ParamKind` inventory)
- `src/optim/` — stock Muon (burn), LR schedules, hybrid Muon/AdamW
  partitioning and gradient accumulation
- `src/tasks/` — harness core (registry, demo protocol, seeded RNG) + the 8
  Stage-0 tasks
- `src/harness/` — experiment dispatch, batch collator, JSONL metrics,
  manifest loader, role evidence + shuffle control, batch tuner (decision
  logic behind `train.auto_batch`)
- `src/train.rs` — manifest-driven training loop (ACT + Muon/AdamW, holdout
  split, in-distribution eval, auto-batch, checkpoints)
- `src/test_backend.rs` — single swap point for the test-suite backend
- `src/main.rs` — CPU smoke binary (param budget, forward shapes, optimizer
  build)
- `examples/run.rs` — manifest load + validate + dispatch dry-run
- `examples/train.rs` — single-manifest training runner
- `examples/chain.rs` — chained experiments with checkpoint dependencies +
  forgetting evals
- `examples/lr_sweep.rs` — two-axis (LR × seed) sweep over held-out accuracy
- `examples/scaling_sweep.rs` — named-axis (`data` | `depth` | `params` | `k`)
  sweep reporting both accuracies and the gap, with `--auto-batch`
- `examples/arm_compare.rs` — paired two-arm comparison (ordered vs
  order-augmented) over seeds, log-driven, refusing any pair whose arms differ
  by more than the intervention
- `examples/param_match.rs` — solve both arms of a depth-vs-width pair at
  matched parameters (searches `head_dim`, binary-searches `ffn_hidden`; an
  explicit third argument pins the target, which is what lets the
  stages↔width frontier hold parameters fixed)
- `examples/verify_port.rs` — asserts a research manifest resolves to the same
  config as a reference manifest, field by field
- `configs/` — run manifests + chains (JSON, no recompile to tweak)
- `research/` — the running experiment record: `findings.md` is the
  consolidated read of what we currently believe, `experiments/` holds one
  versioned manifest per reported result with its run command in `_comment`,
  and `runs/` holds the gitignored `run.jsonl` logs every reported number is
  read back out of. `AGENTS.md` files throughout the tree hold the local
  contracts; `AGENTS.md` at the root is the index.

## Optimizer

**Stock burn Muon** on the 2D hidden matrices (attention Q/K/V/O, MLP
gate/up/down), **AdamW** on everything else (embedding, LM head, RMSNorm
scales, halt gates) — the Muon paper's recipe. Every parameter carries a
semantic `ParamKind`; routing is on `(rank, kind)`, never on name strings, so
a new module classifies itself once and routing follows. 7 Muon matrices per
block.

**Newton-Muon was removed.** It pre-conditioned gradients by an inverse
activation second moment, but burn 0.21's portable `Tensor` API exposes no
Cholesky or dense-inverse op, so the implementation was a **diagonal (Jacobi)
approximation** — a per-input-feature rescale, not a Newton step. It also
scaled gradients by a factor that depended on the realized activation
covariance, which made `lr_muon` a moving target (a deleted test measured the
preconditioner at ~16.4 on a unit covariance, and its value on a trained net is
not knowable from a manifest). Stock Muon is the honest baseline.

Two LR adjustments are available (`Original` is the default — the published
Muon default, flat 1.0 on the attention matrices; `MatchRmsAdamW` is Moonshot's
variant, for reusing an AdamW-tuned rate unchanged). This project keeps
**separate** `lr_muon` and `lr_adamw` knobs, so the `0.2` factor in
`MatchRmsAdamW` is only a scale constant.

**LR schedules** (`src/optim/lr.rs`) are manifest-selectable: `constant`,
`linear` (warmup), `cosine`, `step`. Schedulers advance **once per optimizer
step, never per micro-batch** — advancing per micro-batch would silently
stretch a `warmup_steps: 200` ramp across `accum_steps` times as many
updates. Note that burn's cosine annealer **restarts** at `lr` each cycle
rather than holding at `min_lr`; set `total_steps ≥ steps` for a true
decay-to-floor.

## Config loader

A run manifest is a complete spec in one JSON file, loaded in four stages
(`src/harness/config.rs`): resolve the `extends` chain (with cycle detection),
deep-merge, parse, validate. Each stage has its own error type — Io, Parse,
Schema, Cycle, Invalid — so a missing file, malformed JSON, and a schema
mismatch are distinguishable, and validation reports **every** problem at
once, before any pool is generated or any weights are touched.

`extends` makes a variation cost only its difference. Merge is recursive for
objects; **arrays replace wholesale**, so `k_set: [0]` genuinely narrows the
set rather than unioning with an inherited `[0,1,2,3,5,8]`.

```json
{ "extends": "stage0-base.json",
  "experiment": { "tasks": ["subst-fst-oracle"] },
  "train": { "ckpt_dir": "checkpoints-oracle" } }
```

A `_comment` key is stripped per-file during resolution, so a child's comment
cannot leak into a sibling extending the same base. `save_run` writes a
standalone manifest with no `extends` — the way to turn a resolved config into
an editable starting point.

**Path variables.** Any string in any manifest may use `$name` or `${name}`:

| variable | resolves to |
|---|---|
| `$curdir` | directory of the manifest that mentions it |
| `$parent_dir` | its parent |
| `$repo_root` | the crate root |
| `$experiment_dir` | `research/experiments` |
| `$configs_dir` | `configs` |

Expansion is **per file**, before the parent is read, so `$curdir` means the
directory of the file that writes it rather than whichever file resolved first.
An unknown variable is a **load error** listing what is available, not a silent
passthrough — a typo that survived as a literal path would either fail much
later with a confusing message or, worse, name a real directory and load the
wrong file. A `$` that does not start a valid reference is left alone, so `costs
$5` survives. A reference must start with a letter or underscore, so `$5` is
currency and `$x_1` is a (rejected, unknown) reference.

**An absent or empty `experiment.tasks` key means all eight builtin rungs, not
none.** `harness/experiment.rs` resolves an empty list to the whole registry, so
a manifest inheriting from a base with no `tasks` is a silent pooled mixture and
produces a plausible number. Corollary for reading manifests back:
`jq '.experiment.tasks'` returns `null` for a manifest that legitimately
inherits its `experiment` block, because `jq` does not resolve `extends`. Never
conclude a manifest's rung from `jq` — run
`cargo run --example run -- <manifest>` and read the resolved pool count.

Variations are internally tagged enums, so each is a manifest edit:

| key | variants |
|---|---|
| `optim` | `{"kind": "muon", ...}` |
| `stop` | `{"kind": "act"}` · `{"kind": "act", "shuffle_train": true}` · `{"kind": "fixed", "loops": 4}` · `{"kind": "converge"}` |
| `train.lr_muon` / `lr_adamw` | `{"kind": "constant", "lr": …}` · `linear` (warmup) · `cosine` · `step` |
| `train.eval_holdout` | float in `[0, 1)`; `0` is rejected as in-distribution |
| `train.eval_in_distribution` | bool, default on — logs the `memorization` record |
| `train.indist_eval_per_cell` | instances drawn from the training pool for that record |
| `train.max_train_instances` | cap on the training pool, applied **after** the holdout; `0` = no cap |
| `train.auto_batch` | bool, default **off** — see "Auto-batch" below |
| `train.dump_samples` | decoded eval samples written to `<ckpt_dir>/samples.txt`; `0` = off |
| `experiment.protocol.k_set` | array, **replaces** rather than unions |

The `model`, `optim`, and `train` blocks are **not** serde-defaulted: a
partial block is a parse error by design, so a checked-in manifest fully
determines the run. `stop` is the exception (`#[serde(default)]` → ACT), so
older manifests keep working. Use `extends` to vary one block.

## Auto-batch

`train.auto_batch` grows the micro-batch to fill the GPU instead of leaving it
at whatever a manifest hardcoded — the shipped `stage0-base.json` held 190 MB of
4 GB. Three keys: `auto_batch` (off by default), `auto_batch_max` (the ceiling
on the tuner's B×T budget) and `auto_batch_headroom` (the fraction of total
device memory activations may occupy; the remainder absorbs fragmentation and
the eval pass, which is sized separately and would otherwise OOM after tuning
had already concluded).

Two design points, both load-bearing:

- **The effective batch is preserved.** The tuner holds
  `batch_size × accum_steps` constant and moves `accum_steps` to compensate,
  because the effective batch is what sets gradient noise. A tuner that varied
  it would make runs on different cards non-comparable, which is the one thing
  a sweep axis must not do. When rounding prevents exact restoration the run
  says so rather than reporting a match that is not one, and the realized
  `batch_size` and `accum_steps` are logged at the top of the run — without
  that line an auto-tuned run is not reproducible, since the batch is a
  function of the card it ran on.
- **It measures by extrapolating, never by hitting the limit.** A CUDA OOM
  aborts the process rather than raising a catchable error, so the limit cannot
  be found by reaching it. The tuner measures a batch it has already survived,
  extrapolates activation memory to a candidate, and grows only while the
  prediction clears the budget. It also refuses to run on a non-CUDA backend
  (device memory reads flat there, which looks like unlimited headroom), and
  stops on a non-monotonic or flat reading, keeping the last trustworthy
  budget rather than treating a held high-water mark as free space.

The ladder's axis is the **B×T budget**, not `batch_size` alone: the banded
sampler's row trim already caps rows per length band, so a larger `batch_size`
at the default budget delivers no more rows and measures as a flat memory line.
A knob that has stopped being connected to the thing it names is worse than no
knob, because the tuner believes it is still growing something.

**Why it is off by default.** The tuner compensates the effective batch by
adjusting `accum_steps`, which cannot subdivide below one micro-batch: when the
selected budget does not divide the target, the effective batch ends up
*slightly* different from the manifest's. On a controlled sweep where the whole
claim is that one variable moved, a batch that silently drifted is a second
variable. Opt in per run — `scaling_sweep --auto-batch`, or `"auto_batch":
true` in a manifest whose throughput matters more than an exactly pinned
batch — and log the realized values with the result. Measured effect on the
3050 is under "Throughput, and why the batch tuner exists".
