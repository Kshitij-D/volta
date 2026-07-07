//! Benchmark execution: parse, lower, symbolically execute, and (for
//! equivalence benchmarks) check the verification conditions.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use volta_analysis::driver::{
    AnalysisError, EquivCheckOptions, EquivOutcome, FootprintPolicy, analyze_kernel,
    check_output_equivalence_with,
};
use volta_analysis::eval::{AnalysisOutput, EvalError, Stats};
use volta_frontend::ascii::AsAscii;
use volta_frontend::ast::Module;
use volta_frontend::parse::Parser;

use crate::config::{BenchmarkCategory, BenchmarkDef, ExpectedOutcome, KernelRun};

/// Statistics collected from a benchmark run
#[derive(Debug, Clone, Default)]
pub struct BenchmarkStats {
    /// Symbolic-execution wall time (both kernels), seconds
    pub exec_secs: f64,
    /// VC-checking wall time, seconds
    pub vc_secs: f64,
    /// bar.sync executions across all threads (optimized kernel if present)
    pub block_syncs: u64,
    /// Warp-level sync executions across all threads
    pub warp_syncs: u64,
    /// Instructions executed (both kernels)
    pub instructions: u64,
    /// Output elements compared
    pub elements_checked: u64,
    /// Output elements in the footprint (>= elements_checked when sampling)
    pub elements_total: u64,
}

/// Actual outcome of running a benchmark
#[derive(Debug, Clone)]
pub enum ActualOutcome {
    Equivalent,
    NotEquivalent {
        mismatches: usize,
        first: String,
    },
    /// The analysis rejected the kernel (data race, deadlock, or another
    /// soundness error); `is_race` distinguishes true races.
    Rejected {
        description: String,
        is_race: bool,
    },
    RaceFree,
    Error {
        message: String,
    },
}

impl ActualOutcome {
    pub fn matches(&self, expected: ExpectedOutcome) -> bool {
        match (self, expected) {
            (Self::Equivalent, ExpectedOutcome::Equivalent) => true,
            (Self::RaceFree, ExpectedOutcome::RaceFree) => true,
            (Self::Rejected { is_race, .. }, ExpectedOutcome::DataRace) => *is_race,
            _ => false,
        }
    }

    pub fn status(&self) -> &'static str {
        match self {
            Self::Equivalent => "EQUIV",
            Self::NotEquivalent { .. } => "DIFF",
            Self::Rejected { is_race: true, .. } => "RACE",
            Self::Rejected { is_race: false, .. } => "REJECT",
            Self::RaceFree => "OK",
            Self::Error { .. } => "ERR",
        }
    }
}

/// Result of running a single benchmark
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub category: BenchmarkCategory,
    pub elapsed_secs: f64,
    pub outcome: ActualOutcome,
    pub stats: BenchmarkStats,
    pub passed: bool,
}

/// Benchmark runner configuration
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Base directory for kernel files
    pub kernels_dir: PathBuf,
    pub verbose: bool,
    /// Check at most this many output elements per array (0 = all).
    pub sample: u64,
    /// Confirm every verdict with the f64 numeric oracle.
    pub verify_numeric: bool,
    /// Recycle the VC intern tables past this many terms (0 = never).
    pub recycle_terms: usize,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            kernels_dir: PathBuf::from(crate::KERNELS_DIR),
            verbose: false,
            sample: 0,
            verify_numeric: false,
            recycle_terms: volta_analysis::equiv::DEFAULT_RECYCLE_TERMS,
        }
    }
}

pub struct BenchmarkRunner {
    config: RunnerConfig,
}

impl BenchmarkRunner {
    pub fn new(config: RunnerConfig) -> Self {
        Self { config }
    }

    pub fn run(&self, def: &BenchmarkDef) -> BenchmarkResult {
        let start = Instant::now();
        let (outcome, stats) = match self.run_inner(def) {
            Ok((outcome, stats)) => (outcome, stats),
            Err(e) => (
                ActualOutcome::Error {
                    message: format!("{:#}", e),
                },
                BenchmarkStats::default(),
            ),
        };
        let elapsed_secs = start.elapsed().as_secs_f64();
        let passed = outcome.matches(def.expected);
        BenchmarkResult {
            name: def.name.clone(),
            category: def.category,
            elapsed_secs,
            outcome,
            stats,
            passed,
        }
    }

    pub fn run_all(&self, defs: &[BenchmarkDef]) -> Vec<BenchmarkResult> {
        defs.iter()
            .map(|def| {
                if self.config.verbose {
                    eprintln!("running {} ...", def.name);
                }
                let result = self.run(def);
                if self.config.verbose {
                    eprintln!(
                        "  -> {} in {:.1}s",
                        result.outcome.status(),
                        result.elapsed_secs
                    );
                }
                result
            })
            .collect()
    }

    fn run_inner(&self, def: &BenchmarkDef) -> Result<(ActualOutcome, BenchmarkStats)> {
        let mut stats = BenchmarkStats::default();

        // Analyze the reference kernel.
        let exec0 = Instant::now();
        let reference = match self.analyze(&def.reference)? {
            Ok(output) => output,
            Err(e) => {
                stats.exec_secs = exec0.elapsed().as_secs_f64();
                return Ok((rejected_outcome(e), stats));
            }
        };
        record_exec_stats(&mut stats, reference.stats);

        let Some(optimized_run) = &def.optimized else {
            // Race-check benchmark: reaching the end means no race.
            stats.exec_secs = exec0.elapsed().as_secs_f64();
            return Ok((ActualOutcome::RaceFree, stats));
        };

        // Analyze the optimized kernel.
        let optimized = match self.analyze(optimized_run)? {
            Ok(output) => output,
            Err(e) => {
                stats.exec_secs = exec0.elapsed().as_secs_f64();
                return Ok((rejected_outcome(e), stats));
            }
        };
        // Report the optimized kernel's sync counts (the paper's tables).
        stats.block_syncs = optimized.stats.block_syncs;
        stats.warp_syncs = optimized.stats.warp_syncs;
        stats.instructions += optimized.stats.instructions;
        stats.exec_secs = exec0.elapsed().as_secs_f64();

        // Check the verification conditions.
        let vc0 = Instant::now();
        let outcome = self.check_equivalence(&reference, &optimized, &mut stats)?;
        stats.vc_secs = vc0.elapsed().as_secs_f64();
        Ok((outcome, stats))
    }

    /// Run one kernel. The outer error is an infrastructure failure (I/O,
    /// parse, lowering); the inner error is an analysis rejection (race,
    /// deadlock, structured-CTA violation, ...).
    fn analyze(&self, run: &KernelRun) -> Result<Result<AnalysisOutput, EvalError>> {
        let path = self.config.kernels_dir.join(&run.path);
        let module = load_module(&path)?;
        match analyze_kernel(&module, Some(&run.kernel), run.config.clone()) {
            Ok(output) => Ok(Ok(output)),
            Err(AnalysisError::Eval(e)) => Ok(Err(e)),
            Err(e) => Err(anyhow!("{}: {}", run.path, e)),
        }
    }

    /// Compare the two output footprints on their intersection (a
    /// grid-stride reference and a tiled kernel cover different slices of
    /// the output; arrays present only in the optimized run are ignored).
    /// The actual element loop lives in `volta_analysis::driver`.
    fn check_equivalence(
        &self,
        reference: &AnalysisOutput,
        optimized: &AnalysisOutput,
        stats: &mut BenchmarkStats,
    ) -> Result<ActualOutcome> {
        let options = EquivCheckOptions {
            footprints: FootprintPolicy::Intersect,
            sample: self.config.sample,
            verify_numeric: self.config.verify_numeric,
            recycle_terms: self.config.recycle_terms,
        };
        let report = check_output_equivalence_with(reference, optimized, &options)
            .context("checking output equivalence")?;
        stats.elements_checked = report.elements_checked;
        stats.elements_total = report.elements_total;
        Ok(match report.outcome {
            EquivOutcome::Equivalent => ActualOutcome::Equivalent,
            EquivOutcome::NotEquivalent { mismatches } => {
                let first = mismatches
                    .first()
                    .map(|m| format!("{}[{}]", m.array, m.index))
                    .unwrap_or_default();
                ActualOutcome::NotEquivalent {
                    mismatches: mismatches.len(),
                    first,
                }
            }
        })
    }
}

fn record_exec_stats(stats: &mut BenchmarkStats, s: Stats) {
    stats.instructions += s.instructions;
    stats.block_syncs = s.block_syncs;
    stats.warp_syncs = s.warp_syncs;
}

fn rejected_outcome(e: EvalError) -> ActualOutcome {
    let is_race = matches!(e, EvalError::DataRace { .. });
    ActualOutcome::Rejected {
        description: e.to_string(),
        is_race,
    }
}

/// Load and parse a PTX module.
pub fn load_module(path: &Path) -> Result<Module> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let ascii_src = contents
        .as_bytes()
        .as_ascii_slice()
        .context("file contains non-ASCII characters")?;
    let mut parser = Parser::new(ascii_src);
    parser
        .parse_module()
        .map_err(|e| anyhow!("parse error: {}", e.error.title()))
}
