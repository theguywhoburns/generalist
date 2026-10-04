# DOX framework

- DOX is installed here as the agent instruction hierarchy
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable
  docs must stay understandable from the nearest applicable AGENTS.md plus
  every parent AGENTS.md above it

## Read Before Editing

1. Read this file.
2. Identify every file or folder you expect to touch.
3. Walk from the repository root to each target path.
4. Read every AGENTS.md found along each route.
5. If a parent lists a child whose scope contains the path, read that child and
   continue from there.
6. The nearest AGENTS.md is the local contract; parents carry repo-wide rules.
7. If docs conflict, the closer doc controls local detail, but no child doc may
   weaken DOX.

Do not rely on memory. Re-read the applicable DOX chain in the current session
before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects purpose, scope,
durable structure, contracts, workflow, operating rules, required inputs or
outputs, or the child index. Update parents when parent-level structure or the
child index changes. Remove stale or contradictory text immediately.

## Where things live

- `README.md` — the stable project doc: what the code is, what each instrument
  measures, how to run it, and the current headline results. It is edited
  deliberately and changes rarely.
- `research/` — the running record: every experiment, its provenance, what it
  refuted, and what is open. This is where findings land **first**; a result
  graduates to the README only once it has survived a second measurement or
  answered a question the README actually asks. Keeping them apart is what
  stops the README from becoming a lab notebook.
- `src/`, `configs/`, `examples/` — see their child docs.

## Build and verify

**Use only the default feature set.** `cargo build`, `cargo test`,
`cargo clippy`, `cargo fmt` — nothing else.

- No `--release`. `[profile.dev] opt-level = 1` is the working profile and is
  what every measurement in this repo was taken on. Switching profiles
  silently changes the numbers, not just the speed.
- No `--no-default-features`, no `--features`, no `--all-features`.
- **Why this is a hard rule and not a preference:** every distinct feature set
  is a *separate artifact cache*. Naming `--no-default-features --features
  ndarray` and `--features cuda` alongside the default maintains three parallel
  caches and pays a full recompile of burn and its dependencies for each. The
  default `cargo build` is instant; the "extra" verification builds were costing
  minutes each and bought a check that belongs in a test, not in the loop.

### Commands

Build only what you touched, while iterating:

```bash
cargo build --lib                      # editing src/
cargo build --example scaling_sweep     # editing one example
cargo build --bin generalist            # editing src/main.rs
```

Full gate before committing or calling a task done — not on every edit:

```bash
cargo test                    # full suite; also the broadest build
cargo clippy --all-targets    # must be 0 warnings
cargo fmt --check             # must be clean
```

`cargo test` already builds every test target, so it is the widest build
available; `--all-targets` on `build` is rarely worth its time on a small edit.

### Known blind spot this leaves

The default build has both features on, so it **cannot** catch an ungated
single-feature compile error — naming `burn::backend::NdArray` without a `#[cfg]`
gate compiles fine here and fails only under `--features cuda` alone. That class
of bug was real once (see Local Contracts 1). The mitigation is the `#[cfg]`
gates themselves plus contract 1, not a separate build in the loop. If a
single-feature build ever needs checking again, do it **once**, deliberately,
rather than folding it into the working loop.

## Local Contracts

These are not stylistic preferences. Each one exists because ignoring it has
already caused a wrong result, a broken build, or a wrong conclusion.

1. **Never name a backend outside a `#[cfg]` gate.** `burn::backend::NdArray`
   does not exist in a cuda-only build, and `src/main.rs`,
   `src/test_backend.rs`, and every `examples/*.rs` CPU branch were broken by
   this until gated. `default = ["ndarray", "cuda"]` hides it, so the GPU
   feature set gets no coverage unless it is built explicitly.
2. **A metric must not be able to hide a failure it is not named for.**
   `byte_accuracy` zips against the target and truncates to the shorter string,
   so a wrong-length output scores as partially correct. `length_exact_rate`
   and `mean_len_ratio` exist because of that. When adding a metric, ask what it
   cannot see, and add the companion metric in the same change.
3. **Holdout identity hashes instance content, never pool position.** Any new
   partitioning key must be derived from the instance itself (task, track, k,
   prompt bytes) so it is stable under reordering and changes with seed.
4. **On the data axis, vary `train.max_train_instances`, never
   `experiment.per_cell`.** `per_cell` moves the training pool and the eval set
   together, so each point is scored on a different split. The cap is applied
   after the holdout so the eval set is identical at every point.
5. **A sweep arm must differ from its control only in the intervention.**
   `examples/arm_compare.rs` refuses to run otherwise. Do not relax that.
6. **Verify `k_set` actually demonstrates the rule before reporting a result
   about rule induction.** A sweep that inherits `k_set: [0]` measures nothing —
   the rule is in no prompt and no weight. This happened once and produced a
   confidently wrong conclusion.
7. **Compare a run's loss only against itself.** The order-augmented arm
   solves a permutation-robust function; its loss is not the ordered arm's
   loss at equal steps.

## User Preferences

- Keep going without checking in. Implement, run, and write it down; do not
  stop to ask whether to continue.
- Report negative and refuted results as prominently as positive ones, and say
  plainly when an earlier conclusion of ours was wrong.
- When a result is refuted, correct it in place and say which commit holds the
  old numbers, so nobody re-derives them.

## Child DOX Index

| Path | Owns |
|---|---|
| `research/AGENTS.md` | running experiment record, provenance, refutations, open work |
| `research/experiments/AGENTS.md` | versioned experiment manifests and their provenance |
| `src/AGENTS.md` | library code conventions and module contracts |
| `configs/AGENTS.md` | manifest schema conventions and the extends/merge rules |
| `examples/AGENTS.md` | sweep and tool conventions, what each tool is for |
