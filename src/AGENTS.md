# src/ — library code

## Purpose

The research library: model, optimizer split, data generation, training loop,
metrics, and the harness that makes a run reproducible from a manifest.

## Ownership

- `model/` — the looped transformer and its config (`LoopedConfig`,
  `StopConfig`, `StopMode`).
- `optim/` — Muon + AdamW hybrid, LR schedules, `OptimConfig`.
- `tasks/` — the task registry and the latent rules behind each task.
- `harness/` — manifest loading, run orchestration, metrics, the role
  instruments, batch tuning.
- `train.rs` — `Trainer`, `run_stage`, holdout partition, sample dump.
- `test_backend.rs` — the single swap point for the suite's backend.

## Local Contracts

1. **Backend names go inside `#[cfg]` gates.** See root AGENTS.md contract 1.
2. **Every metric gets its companion in the same change.** Ask what the new
   metric cannot see, and add that readout alongside it. `byte_accuracy` cannot
   see length errors; `copy_rate` cannot see query echo; exact-match cannot see
   partial correctness. Each of those blind spots has already produced a wrong
   conclusion here.
3. **Measurements are recorded per-instance.** `Record` carries the fields the
   aggregate hides (`byte_hits`, `len_out`, `echoed_query`, `seed`) so a later
   question can be answered from an existing log. Adding a field to `Summary`
   without it on `Record` forecloses that.
4. **The pool summary is logged, not only printed.** Anything a later tool
   needs across runs has to be in `run.jsonl`; stdout is lost on a crash, and
   two CUDA `DEINITIALIZED` aborts have already destroyed runs this way.
5. **Never allocate on the host in the training loop** at a size proportional to
   batch × sequence. It was the original cause of the OOM deaths on this card.
6. **Tests use `TestBackend` + `test_device()`**, never a backend named
   directly, so the suite can be pointed at CUDA by editing one file.
7. **Checkpoint restore is pinned by a round-trip test and must stay pinned.**
   `checkpoint_round_trip_restores_weights` saves a trained checkpoint, reloads it
   into a fresh `Trainer` through the same `init_from` argument
   `examples/chain.rs` uses, and asserts the loss on a fixed batch matches to
   1e-6. It asserts on a **continuous** quantity rather than greedy decode,
   because a freshly-initialized model can coincidentally emit the same decoded
   string but cannot match a float.
   `init_from` is a **parameter to `run_stage`, not a manifest key** — writing
   `train.init_from` into a manifest is accepted and silently ignored, which is
   exactly how a probe here came to load no weights at all and emit PAD while
   presenting as a checkpoint bug. If chaining ever needs to be reachable from a
   manifest, that is a loader change with its own test, not a manifest edit.

## Work Guidance

- New metric: add to `Record`, aggregate in `Summary`, emit in `to_json`, print
  in the eval line, and pin the blind spot with a test that would fail if the
  metric started hiding the failure it is not named for.
- New config knob: `#[config(default = ...)]`, document the default *and why*,
  add to every manifest in `configs/` (burn's derive emits no
  `#[serde(default)]`, so nested blocks must be complete), and add a validation
  rule if the value can be wrong.
- Prefer a pure function plus a thin driver for anything with arithmetic in it.
  The batch tuner's decision logic is pure and has 11 tests; the part that
  touches the GPU is a thin shell.
- Comments should explain *why*, and should record what was tried and rejected
  when a future reader would otherwise retry it.

## Verification

Default features only (see root AGENTS.md). While iterating build just the lib:
`cargo build --lib`. Before committing:

```bash
cargo test                    # must pass
cargo clippy --all-targets    # 0 warnings
cargo fmt --check
```