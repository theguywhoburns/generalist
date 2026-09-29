//! Experiment harness: dataset dispatch + JSONL metric records.
//!
//! New experiment = build an [`Experiment`] (task set, tracks, seeds,
//! instance counts), call [`generate`], train/eval downstream, then log
//! [`Record`]s via [`Jsonl`]. New metric = one function over `Record`s.

pub mod batch;
pub mod config;
pub mod experiment;
pub mod metrics;
pub mod role_evidence;
pub mod run;
pub mod shuffle_control;

pub use batch::{
    BUCKET_EDGES, Collated, EOS, PAD, bucket_len, bucket_len_within, collate, collate_seqs,
};
pub use config::{ConfigError, load_chain, load_plan, load_run, save_run};
pub use experiment::{Experiment, generate};
pub use metrics::{Jsonl, Record, summarize};
pub use role_evidence::{ProfileReading, RoleEvidence, evidence_from};
pub use run::{ExperimentPlan, PlannedExperiment, RunConfig};
pub use shuffle_control::ShuffleControl;
