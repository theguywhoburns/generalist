//! Experiment harness: dataset dispatch + JSONL metric records.
//!
//! New experiment = build an [`Experiment`] (task set, tracks, seeds,
//! instance counts), call [`generate`], train/eval downstream, then log
//! [`Record`]s via [`Jsonl`]. New metric = one function over `Record`s.

pub mod batch;
pub mod experiment;
pub mod metrics;
pub mod run;

pub use batch::{BUCKET_EDGES, Collated, EOS, bucket_len, collate};
pub use experiment::{Experiment, generate};
pub use run::{ExperimentPlan, PlannedExperiment, RunConfig};
pub use metrics::{Jsonl, Record, summarize};
