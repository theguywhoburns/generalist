# configs/ — the supported manifest surface

## Purpose

Runnable, documented default configurations. These are what a user is expected
to start from and vary, so unlike `research/experiments/` they are held to a
higher bar: they must be sensible on their own and must not encode a
one-off experimental cell.

## Ownership

One file per cell of the comparison grid. `stage0-base.json` is the root that
every other `stage0-*.json` extends.

## Local Contracts

0. **Unknown keys fail loudly.** `load_run` rejects any key not in `model`,
   `train`, or `experiment` before parsing, naming the offending path — so a typo
   or a key that names a `run_stage` *parameter* rather than a config field stops
   the run instead of silently doing nothing. This is deliberate: a manifest can
   otherwise say something with no effect and produce a perfectly plausible run,
   which is the same failure shape as a metric that cannot see the error it is not
   named for. It is checked in the loader rather than via serde's
   `deny_unknown_fields` because `train`, `model` and `optim` come from burn's
   `Config` derive, which declares only `#[config(...)]` as a helper attribute —
   a `#[serde(...)]` container attribute on those structs does not compile. The
   known-key set is derived from each block's own `Default`, so adding a field
   cannot desynchronise it.
   Two limits worth knowing. Nested objects (`train.lr_muon`,
   `experiment.protocol`, the `optim`/`stop` variants) are still serde's job,
   because their key sets depend on an internally tagged `kind` and a static list
   would have to be duplicated per variant. And `$` path expansion happens
   *earlier* than the key check, so a manifest with both an unknown key and a bad
   path variable reports the path variable first.
   `unknown_keys_are_rejected_rather_than_ignored` and
   `checked_in_manifests_use_only_known_keys` in `src/harness/config.rs` pin
   both halves: the first that a bad key is refused, the second that every
   manifest in the repo still loads.

1. **Every key must be present.** burn's `Config` derive emits no
   `#[serde(default)]`, so a nested block that omits a field fails to parse with
   a bare "missing field". Adding a `TrainConfig`/`LoopedConfig` field means
   adding it to *both* `stage0-base.json` and `smoke-tiny.json` — the test suite
   loads every manifest here and will say so.
2. **`extends` is relative and shallow.** `"extends": "stage0-base.json"`.
   `$` path variables (`$curdir`, `$configs_dir`, `$repo_root`, …) are available
   and preferred over absolute paths.
3. **A child states only its difference from the parent.** Do not restate
   shared settings to change them; edit `stage0-base.json`.
4. **Arrays replace, never union, on merge.** This is load-bearing: `k_set` is a
   set specification, and a union would make `k_set: [0]` inexpressible — the
   operation the fixed-map and map-only manifests depend on.
5. **`_comment` is documentation, never parsed.** It is stripped per file
   during resolution, so a child's comment cannot leak into a sibling.
6. **`eval_holdout: 0` is loader-rejected.** An in-distribution number does not
   measure generalization. If you genuinely want to reproduce a pre-holdout
   number, that is a research manifest, not one of these.
7. **A manifest that changes the *meaning* of a metric needs a comment saying
   so.** `subst-fst-fixed` (one constant rule) and `subst-fst` (map redrawn per
   instance) differ only in task name and produce incomparable accuracies.

## Work Guidance

- Adding a manifest: extend the root, override the minimum, validate with
  `cargo run --example run -- configs/<name>.json`.
- Two manifests whose only difference is the thing under test are a good pair;
  say in the comment what that thing is, because the name rarely makes it
  obvious.
- Anything too small or too short to be a sensible default belongs in
  `research/experiments/`, not here.

## Verification

`cargo test` loads every manifest in this directory and checks they resolve,
merge, and validate — including a round-trip check that `save_run` output
reloads identically. Adding a manifest without full keys fails the suite.

```bash
cargo run --example run -- configs/<name>.json
```