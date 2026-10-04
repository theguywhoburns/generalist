# research/ — running experiment record

## Purpose

The lab notebook. Every experiment lands here first: what was run, on what, and
what came out — including results that refuted an earlier one of ours.

This directory is deliberately separate from `README.md`. The README is the
stable doc: what the code is, what each instrument measures, what the current
headline numbers are. It changes rarely and deliberately. Everything in here
changes constantly and is allowed to be messy while a result is settling.

**A result graduates from here to the README when it has either survived a
second measurement or answered a question the README actually asks.** Not
before. Several results in this repo looked decisive on one run and did not
reproduce; those live here permanently.

## Ownership

- `experiments/` — one file per experiment thread, newest last.
- `findings.md` — the current consolidated read of what we believe, with the
  confidence and the refutations inline. This is the one to read first.

## Local Contracts

1. **Every result records its cell, not just its number.** Model, task rung,
   `k_set`, holdout fraction, steps, seeds, eval-set size, and whether
   auto-batch was on. A bare accuracy is uninterpretable and several of ours
   were.
2. **State the instrument and its floor alongside the number.** Byte accuracy
   with exact-match at 0.000 is a different claim from exact-match alone; a gap
   whose held-out side sits at chance is not a gap finding.
3. **Record refutations in the same file as the original result,** with the
   commit that holds the old number. Do not quietly edit a number.
4. **A negative result is a result.** "Order augmentation did not change X" and
   "held-out stayed at chance across an 18× data range" are findings, not
   failures to report around.
5. **Never present a single-seed number as a finding.** n=3 is the floor, and
   within-seed spread has exceeded between-condition spread at every axis
   measured so far. Say n and the spread.
6. **When a confound is found, state which earlier conclusions it invalidates**
   before stating the new one.
7. **Cross-check `k_set` and the identity of the rule before writing anything
   about induction or generalization.** See root AGENTS.md contract 6.

## Work Guidance

Running an experiment:

1. Write the manifest into `research/experiments/`, `extends`-ing `base-492k.json`
   (or an arm file). Use `$` path variables — `$curdir`, `$parent_dir`,
   `$repo_root`, `$experiment_dir`, `$configs_dir` — never absolute paths. A
   manifest that produces a reported number gets a `_comment` carrying the exact
   command; that is its provenance.
2. Scratch manifests that will never be cited belong in a temp dir, not here.
   The split is: `research/experiments/` is versioned and reviewable, `configs/`
   is the supported default surface.
2. Preflight with `cargo run --example run -- <manifest>`, which resolves
   `extends`, expands `$` variables, validates, and prints pool cell counts
   without touching the GPU. Validation failures here are the cheapest possible
   feedback.
3. GPU sweeps: `cargo run --no-default-features --features cuda --example
   scaling_sweep -- <manifest> <axis> gpu <values> --seeds=a,b,c`. Pass `gpu`
   explicitly — omitting it silently runs on CPU and every number is then
   meaningless (device memory reads a flat 12MB there).
4. Two-arm interventions: `examples/arm_compare.rs`, which refuses to compare
   arms differing by more than the intervention.
5. For any below-chance or otherwise inexplicable number, do not theorize —
   set `train.dump_samples` and read `samples.txt`. Three conclusions in this
   repo were wrong before someone looked at the actual emitted strings.
6. Read results out of `<ckpt_dir>/run.jsonl`, not stdout, so the numbers
   survive a crash.

Recording it: new thread → new file in `experiments/`, then update
`findings.md`. Update the child index below.

## Verification

Before calling an experiment done:

- [ ] manifest passes `cargo run --example run` validation
- [ ] manifest lives in `research/experiments/`, uses `$` variables, and carries its run command
- [ ] `k_set` and rule identity verified against the claim being made
- [ ] n ≥ 3 seeds, and per-seed numbers recorded, not just the mean
- [ ] eval-set size recorded, and identical across points on the axis
- [ ] the companion metric checked (length vs byte; copy vs echo vs accuracy)
- [ ] result compared against its chance floor where one exists
- [ ] any refutation of an earlier result names the commit holding the old number

## Child DOX Index

| Path | Owns |
|---|---|
| `research/findings.md` | current consolidated read, with confidence levels |
| `research/experiments/AGENTS.md` | per-experiment records; index of threads |