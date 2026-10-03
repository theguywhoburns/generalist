# Generalist

A byte-level looped transformer (`src/model`) trained with a stock Muon/AdamW
hybrid optimizer (`src/optim`), used to **construct a generalization scaling
law**: how held-out generalization varies with model scale and with compute
depth, and where memorization takes over.

## Goal

The question is not "can a transformer do in-context learning". It is:

> **How does generalization scale with parameters and with compute depth, and
> where does memorization begin?**

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

In-context learning is the **measure**, not the goal: accuracy-vs-k is one
slice of the dependent variable, and the `subst-fst` family separates rule
*execution* from rule *induction*.

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
  smallest point on the size axis, which is not yet swept.
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

All three land in `run.jsonl` (`*-roles` and `shuffle-control` records) so a
sweep can aggregate them without scraping stdout.

**Scope.** These three answer one question — *do the stages do different jobs* —
and every one of them is about role specialization. They say nothing about
whether an accuracy number is generalization, which is what the next section is
for. A clean role reading on a model that memorizes is still a model that
memorizes, so the two instruments are never substitutes for each other.

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

```text
  indist_accuracy, indist_byte_accuracy   — from the training pool
  heldout_accuracy, heldout_byte_accuracy — from the holdout split
  gap = heldout - indist                  (negative: the model is behind out of sample)
```

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
(Track B). Holding out rule *classes* is not implemented. See "Known
limitations".

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
param-matched non-looped × compute-matched non-looped (≈L × params, L =
measured mean halt) × blocks `{1,2,4}`, ≥3 seeds per cell.

**Implemented and manifest-reachable today:**

- the stop-mode axis: `act` / `fixed {loops}` / `converge` via the `stop` key.
  `stage0-4block.json` (ACT) vs `stage0-fixed4.json` (fixed ×4) differ only
  in `stop` plus the `ponder_weight` zeroing that fixed depth requires and
  the `ckpt_dir`; `stage0-converge.json` covers the convergence control.
- the order axis: `stage0-orderaug.json` sets `stop.act.shuffle_train`, and
  differs from `stage0-4block.json` in that key plus its `ckpt_dir`.
- the data axis, by `train.max_train_instances` (cap the pool after the
  holdout, eval set held fixed). `experiment.per_cell` also varies the pool but
  moves the eval set with it, so it is not a sweep axis.
- the size axis, by editing `model.d_model` / `ffn_hidden` / `n_stages`.
- the context axis, by editing `protocol.k_set`.
- the seed axis, by editing `train.seed` or `--seeds=` on the sweep tool.

`examples/scaling_sweep.rs` drives the named axes (`data` | `depth` |
`params`) over a base manifest, reads both accuracies and the gap out of each
run's `run.jsonl`, and reports mean halt and the profile reading at every point
so a fixed-depth win stays distinguishable from a degenerate-halting artifact.

**Not implemented:** the non-looped columns (param-matched and
compute-matched), and any manifest knob for them.

## Current results

Three measurements exist, plus a throughput note. All of them are on
`subst-fst-fixed`, the weights-only rung, and all are reported here with the
caveats that keep them honest rather than as findings. The LR sweep came first
and is the weakest of the three; the two scaling axes are first passes, not
curves.

### Data axis: the memorization-onset curve

The instrument is the train/held-out gap ("Measuring the gap"), and the axis is
`train.max_train_instances`, so the eval set is the same ~29 held-out instances
at every point. 492,418-parameter model (`d_model` 128, `n_stages` 2,
`max_loops` 4), 600 steps, auto-batch on, 3 seeds per point, RTX 3050 4 GB.

| train N | byte-in | byte-out | gap | exact-in | exact-out |
|---|---|---|---|---|---|
| 16 | 0.944 | 0.329 | -0.614 | 0.688 | 0.000 |
| 48 | 0.988 | 0.433 | -0.555 | 0.885 | 0.000 |
| 128 | 0.985 | 0.704 | -0.281 | 0.906 | 0.312 |
| 288 | 0.977 | 0.850 | -0.127 | 0.938 | 0.478 |

In-distribution accuracy is 0.95–0.99 at every data size, and **the gap narrows
monotonically as data grows**. More data buys transfer here, not memorization —
the opposite sign to what a pure-memorization account predicts.

**Onset was not observed.** The held-out curve is still rising at 288
instances and the gap is still -0.127, so the crossover lies beyond the right
edge of this sweep and the axis has *not bracketed* the onset. A gap of 0.127
with held-out byte accuracy at 0.850 is not a saturated model. What the table
does fix is the direction of travel; extending the axis is the open item, not
this table.

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
cargo run --release --example scaling_sweep -- configs/<base>.json data gpu 16 48 128 288 --seeds=0,1,2 --auto-batch
```

### Compute-depth axis: a first pass, not a curve

Fixed batches (no auto-batch), 600 steps, same model and task, `stop.fixed`
with `loops` as the axis, 3 seeds per point.

| depth | byte-in | byte-out | gap |
|---|---|---|---|
| 1 | 0.201 | 0.134 | -0.067 |
| 2 | 0.406 | 0.324 | -0.082 |
| 4 | 0.444 | 0.403 | -0.041 |
| 8 | 0.405 | 0.387 | -0.018 |

Depth helps, and **the gap narrows with it** (-0.067 → -0.018): extra compute
buys transfer rather than memorization. That is the direction Goal question 2
asks for, and it is the first non-zero evidence on the axis.

**Exact-match was 0.000 on both halves at every depth.** On exact-match alone
this table reads "depth does nothing" — which is the metric failure described
under "Measuring the gap", not a null result. Nothing else about the axis would
have survived without byte accuracy.

**"Saturates by 4" is not resolved.** Within-seed spread at every depth exceeds
the between-depth differences (depth=8 seeds span 0.265–0.495 byte-out), so
three seeds cannot separate depth 4 from depth 8 or speak to where the knee is.
The profile reading was "uniform" at every depth, so this says nothing about
stage roles — see "Reading the stages". It also says nothing about ACT: the axis
is `fixed` only, so no point here is compute-matched to any other.

```bash
cargo run --release --example scaling_sweep -- configs/<base>.json depth gpu 1 2 4 8 --seeds=0,1,2
```

### Throughput, and why the batch tuner exists

| batch | steps/s | instances/s | memory used |
|---|---|---|---|
| fixed 8 | 16.8 | 134 | 190 MB |
| auto (B×T 8192, window 128) | 7.8 | 998 | 1470 MB |

7.4× the data per second for 7.7× the memory. Steps per second *drops* — from
16.8 to 7.8 — because each step is now 16× the work, so **throughput is not
comparable at fixed step counts** and two runs are only comparable at equal
instances seen. That is the same confound as the undertrained data sweep above,
and it is why the data-axis result is stated in instances rather than steps.

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
cargo run --release --example lr_sweep -- configs/stage0-fixed.json gpu 2e-3 1.5e-2 --seeds=0,1,2
```

Per-run logs land in gitignored `checkpoints*/run.jsonl`, so the table is not
a checked-in artifact — re-running is how to verify it. `lr_sweep` reports
**held-out** accuracy (never training loss: a lower LR trades training loss
for generalization here) and prints an explicit single-seed caveat, quoting
the binomial SE at the observed n, when `--seeds` is omitted.

## Known limitations

Listed with what would change the conclusion, strongest first.

1. **Held-out eval is instance-level, not rule-level.** For
   `subst-fst-fixed` the rule is **constant by construction**, so holding out
   instances does not make the rule unseen — the model was trained on the very
   map the eval asks it to apply, and the 0.687–0.761 numbers measure
   transduction of a memorized rule, not generalization. On the `subst-fst`
   rungs the eval rules are new instances of a trained rule family (Track A) or
   the same procedure over unseen symbols (Track B), which is stronger but
   still not rule-level. **Holding out rule classes is not implemented.** This
   is the single biggest caveat on every number above, and it means the sweep
   says nothing about the Goal's questions 1 and 3. Measuring the gap does not
   soften it: a gap between two accuracies on the *same* constant rule is still
   a gap measured on one rule, and the data-axis result above is a claim about
   transfer on the easiest possible axis.
2. **Memorization onset has not been bracketed.** The data axis stops at 288
   training instances with the held-out curve still rising and the gap still
   open at -0.127, so the crossover is beyond the right edge of the measured
   range. What the table does establish is the *sign* — more data closes the
   gap on this rung — not where the sign would flip. Onset also needs the other
   two axes first, because onset in (params × depth × data) space is a joint
   location and only one coordinate has been swept.
3. **The depth axis has one pass, at one operating point.** 492,418 params,
   fixed batches, `fixed {1,2,4,8}` only, 3 seeds. No ACT point and therefore
   no compute-matched ACT-vs-fixed comparison, no `loops: 16`, one model size,
   one task, one rung. Within-seed spread exceeds every between-depth
   difference at n=3, so the apparent saturation is not resolved. And because
   the profile read "uniform" at every depth, the axis as measured says nothing
   about whether stages take on different roles with depth — which is the part
   of Goal question 2 the LR sweep above flagged as the central confound.
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
   not a measurement. See "Reading the stages" for the two instruments that
   do work here — profile shape (no control needed) and order-augmented
   training (causal) — and note that the order-augmented arm has been
   **specified but not yet run**.
5. **Single task, three seeds, and a smaller eval split than the LR sweep.**
   The scaling runs use ~29 held-out instances, so the binomial SE there is
   ~0.09 at p = 0.7 before seed variance, and the gap is a difference of two
   such numbers. No effect smaller than ~0.1 in byte accuracy is resolvable,
   which is the same threshold at which the depth axis stops being legible.
6. **The non-looped columns of the comparison grid are not implemented.** Only
   the stop-mode axis (and, by editing the model block, the size axis) is
   reachable from manifests. Compute-matched non-looped baselines — the thing
   that decides whether depth is doing anything at all — do not exist.
7. **The size axis is unmeasured.** `scaling_sweep.rs` takes a `params` axis,
   but no `d_model` sweep has been run. Goal questions 1 and 4 have no numbers
   behind them at all.

Also worth stating plainly: what "the depth axis has not been run" used to mean
is now narrower. There is a first pass over `fixed {1,2,4,8}` with real
numbers and a real gap, and it is legible only because of byte accuracy. There
is still **no ACT-vs-fixed comparison at matched compute**, **no size sweep**,
and no point in the depth sweep where the held-out curve turns over.

## Next experiments

Ordered by how much each would reduce uncertainty. Items marked *(partly done)*
were started and the remainder is stated explicitly.

0. **Run the order-augmented arm** (`stage0-orderaug.json` vs
   `stage0-4block.json`, 3 seeds each). Both instruments for reading the
   stages now exist and neither has been used at scale: this is the cheapest
   test of whether the 4-stage structure means anything, it costs two runs per
   seed, and it is the one cell where a *negative* result is genuinely
   informative — if an order-augmented model still collapses under reversal,
   the sequential dependency is structural and the whole "learned roles"
   framing needs replacing. Do this before any further depth work, because it
   decides what a depth sweep would even be measuring.
1. **Extend the sweep to the k>0 and oracle rungs** (`stage0-oracle`,
   `stage0-4block`, `subst-fst`). This does two things at once: it moves the
   headline number onto a task where the rule is genuinely held out
   (limitation 1), and it puts the model on a rung where the random-weight
   control scores above chance, which is what makes the shuffle control
   non-degenerate (limitation 4). Highest value per GPU-hour after item 0.
2. **Finish the depth axis** *(partly done: a first pass over `fixed
   {1,2,4,8}` exists)*. Still open, in order of value: add **ACT** as a
   compute-matched point, which is the comparison Goal question 2 actually
   asks; add `loops: 16` (raising `max_loops` alongside it); and enough seeds
   to resolve the 4-vs-8 difference the current n=3 pass cannot. Keep reporting
   byte accuracy, the gap, and the per-stage profile at every point — the
   existing pass reads "uniform" at every depth, so nothing there distinguishes
   a real depth effect from a wider-capacity effect, and that ambiguity is the
   confound the LR sweep above already demonstrated.
3. **Run the size axis** *(unmeasured; the tool supports it)*: param-matched
   models at several `d_model` (128 / 256 / 384 / 512, with `n_heads × head_dim`
   kept consistent and `vocab_size ≥ 256` for byte level), fixed depth, fixed
   data. Fit held-out accuracy against log-params and check whether the Track A
   and Track B slopes differ. This is what "scaling with size" means as a
   measurement rather than a plan. Note that the model already measured on the
   other two axes (d_model 128, 492,418 params) is the *smallest* point on this
   curve, which is worth knowing when the two results are compared.
4. **Enough seeds to resolve ~0.1 effects** *(partly quantified)*. Three seeds
   cannot: the within-arm spread is 0.177 on the LR sweep and 0.265–0.495 on
   the depth sweep's depth=8 point. Either 8–10 seeds per cell or a larger eval
   split (`indist_eval_per_cell` / `eval_split` take the first N per cell —
   raising them trades eval time for resolution directly, and the gap needs
   both halves raised, not one).
5. **Bracket memorization onset** *(instrument done, location not)*. The
   in-distribution path exists as its own deliberately named metric
   (`train.eval_in_distribution` → `memorization` record) rather than a config
   flip, exactly as intended; `eval_holdout: 0` remains loader-rejected. What
   is missing is the range: the axis has to be pushed past 288 instances, and
   to be swept at two or more model sizes and depths so the crossover can be
   located in more than one coordinate. Hold the instances-seen budget constant
   along the axis — the earlier undertrained pass at a fixed step count is what
   produced the refuted "onset below 16" reading.
6. **Add the compute-matched non-looped baseline** so "depth" has something
   to be better than. Without it the depth axis measures only depth.

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
- `examples/scaling_sweep.rs` — named-axis (`data` | `depth` | `params`) sweep
  reporting both accuracies and the gap, with `--auto-batch`
- `configs/` — run manifests + chains (JSON, no recompile to tweak)

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
