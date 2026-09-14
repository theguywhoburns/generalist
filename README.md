# Generalist

1M-parameter looped byte-level transformer (`src/model`) + Newton–Muonn optimizer
reimplementation (`src/optim`). Research vehicle for one question:

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

## Eval

Held-out only: unseen rules, 2× length, novel symbols, noisy context.
Re-run all earlier stages after each new stage (forgetting check). Log halt
depth per token position (syntax vs reasoning tokens), not just per task.

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

- `src/model/` — looped transformer (config, attention, MLP, block, halting, model)
- `src/optim/` — Newton–Muon preconditioner + hybrid Muon/AdamW training glue
- `src/tasks/` — harness core (registry, demo protocol, seeded RNG) + Stage-0 tasks
- `src/harness/` — experiment dispatch, batch collator, JSONL metrics, run manifests
- `src/train.rs` — manifest-driven training loop (ACT + Newton–Muon, eval, checkpoints)
- `src/test_backend.rs` — single swap point for the test-suite backend
- `examples/profile.rs` — CPU micro-profile
- `examples/run.rs` — manifest dispatch dry-run
- `examples/train.rs` — single-manifest training runner
- `examples/chain.rs` — chained experiments with checkpoint dependencies + forgetting evals
- `configs/` — run manifests + chains (JSON, no recompile to tweak)
