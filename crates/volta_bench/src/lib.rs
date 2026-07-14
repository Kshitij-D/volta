//! Volta Benchmark Suite
//!
//! Reproduces the evaluation of "Equivalence Checking of ML GPU Kernels":
//! Harris reductions (Table 1), Boehm matmuls (Table 2), attention variants
//! (Table 3), causal attention (Table 4), the LLM-generated convolution
//! (Section 6.2), Claude-Code GEMMs (Table 5), TileLang pairs (Table 6),
//! and the FaialAA data-race benchmarks (Table 7), all over the PTX
//! collected in `kernels/`.

pub mod benchmarks;
pub mod config;
pub mod reporter;
pub mod runner;
pub mod z3_compare;

pub use benchmarks::all_benchmarks;
pub use config::{BenchmarkCategory, BenchmarkDef, BenchmarkSuite, ExpectedOutcome, KernelRun};
pub use reporter::{export_json, print_all_results, print_op_counts, print_results_table, print_summary};
pub use runner::{ActualOutcome, BenchmarkResult, BenchmarkRunner, BenchmarkStats, RunnerConfig};
pub use z3_compare::{Z3CompareRow, compare_one};

/// Default kernels directory: the paper benchmark collection.
pub const KERNELS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels");
