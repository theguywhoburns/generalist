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
| data | `experiment.per_cell`, `task` | 8 tasks, both tracks |
| context `k` | `experiment.protocol.k_set`, `k0_rate` | `{0,1,2,3,5,8}`, k=0 at 15% |
| optimization | `train.lr_muon` / `lr_adamw` schedules | `constant` / `linear` / `cosine` / `step` |

**Dependent variable:** held-out exact-match accuracy (plus copy-rate and the
per-stage halt profile). Every accuracy is measured on instances the model
never trained on — see "Eval" below.

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
   result.
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
`train.eval_max_new`) and scores **exact string equality with the target**;
the EOS byte terminates the row and is not itself part of the compared
string.

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

Logged per run (`run.jsonl`, one JSON object per evaluated instance): task,
track, k, `correct`, `copied`, `steps`, `halt`, per-stage `bh`, and the run
`seed` — so a multi-seed comparison is a filter, not a string join. Final-eval
summaries additionally carry per-stage p50/p90/std, utilization shares, and
between-stage halt correlations (≈1.0 everywhere means the per-stage gates are
one global head in disguise).

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
- the size axis, by editing `model.d_model` / `ffn_hidden` / `n_stages`.
- the context axis, by editing `protocol.k_set`.
- the seed axis, by editing `train.seed` or `--seeds=` on the sweep tool.

**Not implemented:** the non-looped columns (param-matched and
compute-matched), and any manifest knob for them.

## Current results

One measurement exists. It is a Muon learning-rate sweep, and it is reported
here with its caveats rather than as a finding.

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
   says nothing about the Goal's questions 1 and 3.
2. **The stage-order shuffle diagnostic is currently uninformative.**
   Reversing stage order collapses accuracy *by construction* — stages are
   sequential, so reversal reverses the data flow. A random-weight control was
   added (`src/harness/shuffle_control.rs`) to give a floor:

   ```text
     trained_order_acc  - trained_shuffled_acc      (observed collapse)
     control_order_acc  - control_shuffled_acc      (floor: reversal alone)
     excess             = trained_gap - control_gap
   ```

   But an untrained model scores **0.000 in both orders**, so
   `control_gap` is always 0.000, `excess == trained_gap`, and the
   `roles_supported: true` verdict degenerates to the original unconfounded
   claim. The checked-in `checkpoints-fixed/run.jsonl` (pre-holdout, and
   written before the control existed) makes this concrete: 0.771
   trained-order vs 0.000 shuffled against a floor of 0.000 — an "excess" of
   exactly the whole collapse. **Do not present the shuffle result as evidence
   of role specialization.** The control is only informative on a rung where
   the control model scores above chance. The code is right; the *instrument*
   is blind at the current operating point.
3. **Single task, three seeds, n = 96.** Binomial SE at n = 96, p = 0.7 is
   ~0.047, and seed-to-seed variance is larger still. A 0.01 difference is
   about one instance. No effect smaller than ~0.1 is resolvable at this
   sample size.
4. **The non-looped columns of the comparison grid are not implemented.** Only
   the stop-mode axis (and, by editing the model block, the size axis) is
   reachable from manifests. Compute-matched non-looped baselines — the thing
   that decides whether depth is doing anything at all — do not exist.
5. **`examples/lr_sweep.rs` is the only sweep tool.** It is two-axis (LR ×
   seed), reports held-out accuracy, and prints the single-seed caveat. It
   cannot sweep a model or stop-mode axis; those are hand-written manifests.

Also worth stating plainly: the depth axis has **not** been run. There is no
`fixed {1,2,4,8,16}` sweep, no ACT-vs-fixed comparison at matched compute,
and no size sweep. The Goal's questions 2 and 4 have **zero** measurements
behind them.

## Next experiments

Ordered by how much each would reduce uncertainty.

1. **Extend the sweep to the k>0 and oracle rungs** (`stage0-oracle`,
   `stage0-4block`, `subst-fst`). This does two things at once: it moves the
   headline number onto a task where the rule is genuinely held out
   (limitation 1), and it puts the model on a rung where the random-weight
   control scores above chance, which is what makes the shuffle diagnostic
   non-degenerate (limitation 2). Highest value per GPU-hour.
2. **Run the depth axis properly**: `fixed {1,2,4,8,16}` and ACT, at fixed
   parameters and fixed data, 3+ seeds per point (raising `max_loops` alongside
   `loops` for the 16 point). This is the actual compute-depth scaling curve
   and it is simply missing. Report the per-stage halt profile alongside
   accuracy for every point, so a fixed-depth win and a degenerate-halting
   artifact are distinguishable — the current sweep shows exactly how easy that
   is to get wrong.
3. **Run the size axis**: param-matched models at several `d_model`
   (128 / 256 / 384 / 512, with `n_heads × head_dim` kept consistent and
   `vocab_size ≥ 256` for byte level), fixed depth, fixed data. Fit
   held-out accuracy against log-params and check whether the Track A and
   Track B slopes differ. This is what "scaling with size" means as a
   measurement rather than a plan.
4. **Enough seeds to resolve ~0.1 effects.** Three seeds cannot; the
   within-arm spread in the table above is 0.177. Either 8–10 seeds per cell
   or a larger eval split (`eval_split` takes the first 48 per cell —
   raising it trades eval time for resolution directly).
5. **Measure memorization onset.** Requires (2) and (3) first. Operational
   definition to be fixed in advance: the point where train and held-out
   curves diverge, or where held-out accuracy saturates while in-distribution
   accuracy keeps climbing. Needs an in-distribution eval path that is
   *explicitly* separate from the generalization number — `eval_holdout: 0`
   exists for that and is loader-rejected everywhere else, which is the
   right default and the wrong tool for this specific job. Worth a
   deliberately named second metric rather than a config flip.
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
  manifest loader, shuffle control
- `src/train.rs` — manifest-driven training loop (ACT + Muon/AdamW, holdout
  split, eval, checkpoints)
- `src/test_backend.rs` — single swap point for the test-suite backend
- `src/main.rs` — CPU smoke binary (param budget, forward shapes, optimizer
  build)
- `examples/run.rs` — manifest load + validate + dispatch dry-run
- `examples/train.rs` — single-manifest training runner
- `examples/chain.rs` — chained experiments with checkpoint dependencies +
  forgetting evals
- `examples/lr_sweep.rs` — two-axis (LR × seed) sweep over held-out accuracy
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
| `stop` | `{"kind": "act"}` · `{"kind": "fixed", "loops": 4}` · `{"kind": "converge"}` |
| `train.lr_muon` / `lr_adamw` | `{"kind": "constant", "lr": …}` · `linear` (warmup) · `cosine` · `step` |
| `train.eval_holdout` | float in `[0, 1)`; `0` is rejected as in-distribution |
| `experiment.protocol.k_set` | array, **replaces** rather than unions |

The `model`, `optim`, and `train` blocks are **not** serde-defaulted: a
partial block is a parse error by design, so a checked-in manifest fully
determines the run. `stop` is the exception (`#[serde(default)]` → ACT), so
older manifests keep working. Use `extends` to vary one block.
