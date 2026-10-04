# experiments/ — versioned experiment manifests

## Purpose

Runnable, reviewable provenance for every number reported anywhere in this
repo. Each file here produced a specific result; the `_comment` says which, and
carries the exact command.

These are **not** part of the supported manifest surface — `configs/` is. This
directory is a lab notebook that happens to be executable, so it may carry
cells that are too small, too short, or otherwise unsuitable as defaults.

## Ownership

- `base-492k.json` — shared base for the 492,418-param scaling experiments.
  Every other 492K manifest extends it.
- `arm-ordered.json` / `arm-orderaug.json` — the order-augmentation pair. Read
  them together; neither means anything alone.

## Local Contracts

1. **A manifest here must reproduce the run it documents.** When porting a
   manifest from scratch space, verify it rather than eyeballing it:
   ```bash
   cargo run --example verify_port -- research/experiments/X.json /tmp/opencode/X.json
   ```
   `verify_port` compares the fully resolved configs field by field and exits
   non-zero on any divergence. It has already caught four real ones, including a
   depth axis that had silently switched to auto-batch.
2. **Record the settings that actually ran, not the tidy ones.** The depth axis
   runs at fixed batch because that is what was measured; the data axes run
   with auto-batch because that is what produced the reported table. A manifest
   "cleaned up" to a uniform batch no longer reproduces its result.
3. **Put the run command in the `_comment`,** multi-line, with the axis and
   seeds. Someone reproducing this should not have to reconstruct the
   invocation from the manifest.
4. **Note what is deliberately off** and why — `auto_batch: false` on the depth
   axis, `shuffle_eval: false` on the weights-only rung. A knob that looks
   misconfigured invites a well-meaning fix that breaks provenance.
5. **Checkpoints go under `$curdir/checkpoints-*`,** never a bare relative
   path, so a run from the repo root and a run from this directory land in the
   same place.
6. **Use `$` path variables, never absolute paths.** `$curdir`,
   `$parent_dir`, `$repo_root`, `$experiment_dir`, `$configs_dir`. An absolute
   path in a committed manifest breaks for everyone else; an unknown variable
   is rejected by the loader, so a typo fails loudly instead of silently
   pointing at the wrong file.

## Work Guidance

Adding a new experiment:

1. Extend `base-492k.json` (or an arm file) rather than copying a config whole.
   Arrays replace rather than union on merge, so `k_set: [0]` is expressible —
   keep that property.
2. Override only what the axis varies; leave the rest inherited.
3. Preflight with `cargo run --example run -- <manifest>`. It resolves
   `extends`, expands `$` variables, validates, and prints cell counts.
4. Run the axis. For two-arm interventions use `examples/arm_compare.rs`, which
   refuses arms that differ by more than the intervention.
5. Read results from `run.jsonl`, not stdout.
6. Record the result in `research/findings.md`, and add a line to the index
   below if the manifest is worth keeping long-term.

## Verification

- [ ] `cargo run --example run -- <manifest>` loads and validates
- [ ] `$` variables used, no absolute paths
- [ ] run command present in `_comment`
- [ ] deliberate deviations from the base documented
- [ ] if ported from scratch: `verify_port` reports OK

## Child DOX Index

This directory has no child directories. One file per experiment thread is the
convention; add an `AGENTS.md` only if a subdirectory becomes necessary.