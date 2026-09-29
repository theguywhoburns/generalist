# Generalist

1M-parameter looped byte-level transformer (`src/model`) + stock Muon/AdamW
hybrid optimizer (`src/optim`). Research vehicle for one question:

**Given a few in-context examples, can transformers (and their looped
counterparts) actually generalize?**

## Goal

Provided a few in-context examples, generalize and solve symbolic tasks
without issue. Context supplies the rule; weights supply the execution
machinery.

## Setup

Byte-level looped transformer (headline: 1 tied block, ~1M params, 256-dim;
diagnostic controls: 2/4 blocks). Learned (ACT) or convergence-based halting.
Procedural data, byte-encoded. Held-out splits only: unseen rules, longer
sequences, novel symbols, noisy context.

Every stage runs twin tracks:

- **Track A** — rule seen in training, held-out instances.
- **Track B** — rule never seen, defined only by k prompt examples.
  The k=0→k>0 accuracy delta is the ICL signal.

Instances are split into train and held-out eval *before* anything touches the
pool (`train.eval_holdout`, default 10%), so no reported accuracy is measured
on data the model trained on. The holdout is deterministic per instance, so
forgetting checks across chained stages compare against the same instances.

## In-context demonstration protocol

- Every task instance ships with k demonstrations of THAT INSTANCE's latent
  rule (input→output pairs), k sampled per instance from {0,1,2,3,5,8};
  k=0 at ~15% frequency.
- Demonstrations are resampled per instance. No fixed demo set per task family.
- Demo inputs drawn from a different length/symbol regime than the query
  input, to block copy/nearest-neighbor shortcuts. Log copy-rate (fraction
  of outputs that are exact substrings of demo outputs).
- Layout randomized per instance: demo order, separators, query position,
  template style.
- ~5% of instances contain one corrupted demonstration (wrong output) to
  train majority-rule robustness.
- Exception: known-operator procedural core (math operator semantics,
  state-tracking primitives) also receives bare-instruction instances, so
  k=0 works for operators already in weights. Principle: examples define
  novel rules; weights define known operators and the execution machinery.
- Eval reports: accuracy as a function of k (first-class result — saturation
  at k=2 holding flat to k=8 is the target; needing k=8 is memorization with
  demos); instruction-only vs demo-only vs both ablation; k=0 on unseen
  rules as leakage check (expect ~chance; higher means train leakage).
- Invariance holds over the trained k-distribution only, not arbitrary
  counts. Context budget caps PoC k at ~8–10.

## Curriculum

**Stage 0 — pattern prediction (first).** Periodic/structured continuation,
FST transduction (unseen maps held out), regular recognition (parity, modular
counting, majority), Dyck-1 → Dyck-2, copy/reverse/repeat with length splits,
tiny SCAN compositions.

**Stage 1 — state tracking.** Multi-counters, variable binding chains
(overwrite, nesting), delayed match / NADs, Lights Out 3×3 → 5×5 (solvable +
certified unsolvable; unsolvable = bare refusal), grid pathfinding.

**Stage 2 — constraints, structured generation, retrieval.** Small logic
puzzles, JSON schema adherence + escape translation (unique schema per
example), symbolic lookup with missing-key refusal, templated retrieval QA
with abstention (no free text), note-taking QA, tool-call simulation.

**Stage 3 — math (last).** Arithmetic → multi-step problems → symbolic
manipulation, increasing difficulty, each with step-by-step breakdown traces
(graded per intermediate step). Style variation with disjoint train/test
styles (renamed operators, remapped symbols, reworded templates), plus:
assumption flips (`ASSUME x>0` vs `x<0` vs none), domain conditioning
(`REAL|COMPLEX|IEEE`: `sqrt(-1)`/`1/0` → undefined/`i`/`inf`),
finite/nonfinite classification, exact vs approximate output modes.

## Comparison grid

Looped+ACT (headline) × looped fixed {1,4,8,16} × param-matched non-looped ×
compute-matched non-looped (≈L× params, L = measured mean halt, trained
post-hoc) × blocks {1,2,4}. ≥3 seeds per cell.

The stop-mode half of this grid is manifest-reachable via the `stop` key —
`configs/stage0-4block.json` (act) and `configs/stage0-fixed4.json` (fixed ×4)
differ only in that one key, and `configs/stage0-converge.json` covers the
convergence control. The non-looped columns are not implemented.

## Eval

Held-out only: unseen rules, 2× length, novel symbols, noisy context. The
instance-level split is enforced in `run_stage`, not left to the caller —
`eval_holdout: 0` is rejected by the loader unless you are deliberately
reproducing a pre-holdout number.

Re-run all earlier stages after each new stage (forgetting check). Log halt
depth per token position (syntax vs reasoning tokens), not just per task.

**The stage-order shuffle needs a control.** Reversing stage order collapses
accuracy *by construction*: stages run sequentially, so reversal reverses the
data flow, and the same collapse appears for random weights. The collapse on
its own is therefore not evidence of learned role specialization. So every run
with `shuffle_eval` also evaluates an **untrained** model of the same
architecture on the same split in both orders, and reports the *excess*
collapse:

```text
  trained_order_acc - trained_shuffled_acc      (observed collapse)
  control_order_acc - control_shuffled_acc      (floor: reversal alone)
  excess = trained_gap - control_gap
```

Only an excess above 0.10 supports the "stages learned distinct roles"
reading; near zero means the stages are order-interchangeable. The verdict
goes to `run.jsonl` as a `shuffle-control` record carrying `roles_supported`,
so a sweep can aggregate it without scraping stdout.

Pre-registered bars: Track B (k≥2) exact-match ≥80%; 2× length ≥60%; schema
validity ≥95%; missing-key false answers ≤5%; disjoint-style math within
10pts of in-style.

## Decisive read

- Looped+ACT holds where fixed variants collapse, halt rises with
  difficulty → adaptive computation, not memorization.
- Converge-holds / ACT-fails → halting head is the failure, not recurrence.
- All configs track each other → tasks memorized, harden splits.
- All fail → learnability check: train one cell on oracle traces. Oracle
  fails too → data/capacity problem, not architecture.

## Non-goals

Natural language, scale.

## Layout

- `src/model/` — looped transformer (config, attention, MLP, block, halting, masking, model)
- `src/optim/` — stock Muon (burn), LR schedules, hybrid Muon/AdamW partitioning
- `src/tasks/` — harness core (registry, demo protocol, seeded RNG) + Stage-0 tasks
- `src/harness/` — experiment dispatch, batch collator, JSONL metrics, manifest loader
- `src/train.rs` — manifest-driven training loop (ACT + Muon/AdamW, eval, checkpoints)
- `src/test_backend.rs` — single swap point for the test-suite backend
- `examples/run.rs` — manifest load + validate + dispatch dry-run
- `examples/train.rs` — single-manifest training runner
- `examples/chain.rs` — chained experiments with checkpoint dependencies + forgetting evals
- `configs/` — run manifests + chains (JSON, no recompile to tweak)

## Config loader

A run manifest is a complete spec in one JSON file, loaded in four stages
(`src/harness/config.rs`): resolve the `extends` chain, deep-merge, parse,
validate. Each stage has its own error type, so a missing file, malformed
JSON, and a schema mismatch are distinguishable — and validation reports
**every** problem at once, before any pool is generated or weights touched.

`extends` makes a variation cost only its difference. Merge is recursive for
objects; **arrays replace wholesale**, so `k_set: [0]` genuinely narrows the
set rather than unioning with an inherited `[0,1,2,3,5,8]`.

```json
{ "extends": "stage0-base.json",
  "experiment": { "tasks": ["subst-fst-oracle"] },
  "train": { "ckpt_dir": "checkpoints-oracle" } }
```

Variations are internally tagged enums, so each is a manifest edit:

| key | variants |
|---|---|
| `optim` | `{"kind": "muon", ...}` |
| `stop` | `{"kind": "act"}` · `{"kind": "fixed", "loops": 4}` · `{"kind": "converge"}` |
| `train.lr_muon` / `lr_adamw` | `{"kind": "constant", "lr": …}` · `linear` (warmup) · `cosine` · `step` |

`stop` is the grid axis: `act` is the headline, `fixed` the compute-matched
control, `converge` the convergence control. Omit the key for ACT.
`save_run` writes a standalone manifest (no `extends`), which is the way to
turn a resolved config into an editable starting point.
