# examples/ — tools

## Purpose

Executables that *run experiments*, rather than the library. Everything here
reads its results out of `run.jsonl` rather than stdout, so a run that dies
halfway still has usable numbers.

## Ownership

| tool | purpose |
|---|---|
| `run` | dry run: resolve, validate, print cell counts. No GPU. Preflight everything with this. |
| `train` | one manifest, one run, on GPU or CPU. |
| `chain` | ordered experiments whose checkpoints feed later ones, with forgetting evals. |
| `scaling_sweep` | one axis (`data` / `depth` / `params` / `k`) × seeds; reports both accuracies, the gap, length, halt and profile. |
| `lr_sweep` | LR × seeds. Two-axis, reports spread, refuses to rank on one seed. |
| `arm_compare` | two manifests that must differ **only** in the intervention, and which reports whether order-dependence is learned or structural. |
| `param_match` | solve the parameter counts for a depth-vs-width pair. Searches `head_dim`, binary-searches `ffn_hidden`; optional third arg pins the target. |
| `verify_port` | assert a research manifest resolves to the same config as a reference manifest. Field-by-field. |

## Local Contracts

1. **Never name a backend outside a `#[cfg]` gate.** Both backends' branches are
   feature-gated; see root AGENTS.md contract 1. These are the files that broke
   the cuda-only build.
2. **`arm_compare` must keep refusing arms that differ by more than the
   intervention.** Do not add an override. A comparison that isolates nothing
   looks exactly as clean as one that does, which is what makes it dangerous.
3. **`gpu` is an explicit argument, and omitting it silently runs on CPU.**
   `scaling_sweep … data 128` without `gpu` produces a complete, plausible,
   meaningless table. Say so in the usage string, and never infer the device.
4. **Results are read from `run.jsonl`,** parsed with `jget`/`jbool`-style
   helpers, never scraped from the human-facing print lines. The print format
   changes; the log keys are the contract.
5. **A tool must state its instrument's floor next to its number.** Exact-match
   at 0.000 is unreadable without byte accuracy beside it; a gap is unreadable
   when one side is at chance. Say which, in the output.
   **And a floor is not the same thing as a chance rate.** On `dyck1` a run scored
   byte **1.000** with exact **0.000** because every target byte was correct but EOS
   was never emitted — it ran to the decode cap on 81 of 81 instances
   (`mean_len_ratio` 10.51). Byte accuracy alone called that perfect, exact-match
   alone called it a failure, and only `length_exact_rate` / `mean_len_ratio` named
   it. A tool that reports byte and exact without the length pair cannot tell a
   model that learned nothing from one that learned everything and cannot stop.
6. **`--seeds` defaults to one seed and the summary must say so** rather than
   presenting a single-seed point as a ranking. This bit us on the LR sweep.
7. **Prefers byte accuracy to exact-match, and reports both.** Never one alone.
8. **A parameter-matching tool takes an explicit target and never derives it from
   the arm it is matching.** The derived target moves with `n_stages`, so it
   cannot hold parameters fixed while the stages↔width split varies — which is
   the only way to trace that frontier. `param_match`'s optional third argument
   exists for this; do not remove it and do not make it the default.
9. **`param_match` must account for `blocks_per_stage`,** and the fourth argument
   is what supplies it. `param_count` scales the encoder stack by
   `n_stages × blocks_per_stage` but scales the halting gates by `n_stages`
   **alone**, so one stage holding four blocks has **one** gate, not four. A tool
   that omits the field silently counts a multi-block configuration as if it were
   single-block, and reports a parameter total that is wrong in the direction that
   makes the config look affordable. `param_match` predates this and its `count()`
   defaulted the field to 1 until the fourth argument was added.
10. **Hand-matching `d_model` to hit a parameter target does not work.** Body
   parameters scale roughly with `d²`, so stepping `d_model` overshoots the
   target by a large margin, and the config validator's `d_model == n_heads *
   head_dim` constraint can make a target *unreachable* rather than merely hard.
   Hand-matching got this 50% wrong twice before the solver existed. Search
   `head_dim`, then binary-search `ffn_hidden` as the gentle second knob.

## Work Guidance

- A new tool earns its place by answering a question no existing metric can,
  or by making a comparison that is easy to get wrong impossible to get wrong
  (that is the whole argument for `arm_compare` and `verify_port`).
- When adding a tool that reuses another's parsing, factor the helper rather
  than copy it — the copies are how the `-pool` label matching drifted.
- Print what the tool decided *and* why, including the cases where it refused
  or stopped early. Silent early exits read as successful runs.

## Verification

Default features only (see root AGENTS.md).

```bash
cargo build --example run     # smoke: the shared manifest-resolution path
cargo run   --example run -- configs/stage0-base.json
```

There is no separate harness for the tools; `cargo test` builds them, so they
are covered by compiling. Add a tool-level check to `research/` if the tool
makes a claim worth pinning.