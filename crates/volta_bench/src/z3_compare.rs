//! Side-by-side timing comparison: `volta_analysis::canon`'s decision
//! procedure vs. Z3, on the same equivalence benchmarks - reproducing
//! ("decides vs.
//! cannot decide", not "fast vs. slow") against Volta's from-scratch
//! reimplementation and its own kernel corpus.

use std::path::Path;
use std::time::{Duration, Instant};

use volta_analysis::driver::{
    EquivCheckOptions, EquivOutcome, FootprintPolicy, analyze_kernel, check_output_equivalence_with,
};

use crate::config::{BenchmarkCategory, BenchmarkDef, KernelRun};
use crate::runner::load_module;

/// One benchmark's result under both backends.
#[derive(Debug, Clone)]
pub struct Z3CompareRow {
    pub name: String,
    pub category: BenchmarkCategory,
    /// Symbolic-execution time (both kernels) - identical setup cost for
    /// both backends, included for context, not part of the comparison.
    pub exec_secs: f64,
    pub decision_secs: f64,
    pub decision_status: String,
    /// Z3 solve time, summed across all checked elements.
    pub z3_secs: f64,
    pub z3_equiv: usize,
    pub z3_not_equiv: usize,
    pub z3_unknown: usize,
    pub z3_unsupported: usize,
    pub z3_error: usize,
    /// Set when the row couldn't be produced at all (bad kernel, not an
    /// equivalence benchmark, decision-procedure error, ...); the other
    /// fields are left at their defaults in that case.
    pub error: Option<String>,
}

fn empty_row(def: &BenchmarkDef, error: String) -> Z3CompareRow {
    Z3CompareRow {
        name: def.name.clone(),
        category: def.category,
        exec_secs: 0.0,
        decision_secs: 0.0,
        decision_status: "N/A".to_string(),
        z3_secs: 0.0,
        z3_equiv: 0,
        z3_not_equiv: 0,
        z3_unknown: 0,
        z3_unsupported: 0,
        z3_error: 0,
        error: Some(error),
    }
}

fn run_kernel(kernels_dir: &Path, run: &KernelRun) -> anyhow::Result<volta_analysis::eval::AnalysisOutput> {
    let module = load_module(&kernels_dir.join(&run.path))?;
    Ok(analyze_kernel(&module, Some(&run.kernel), run.config.clone())?)
}

/// Run one equivalence benchmark through both backends. Never panics or
/// aborts a batch: failures (missing optimized kernel, analysis error,
/// decision-procedure error) become `Z3CompareRow::error`, so a caller
/// looping over many benchmarks can keep going.
pub fn compare_one(
    kernels_dir: &Path,
    def: &BenchmarkDef,
    sample: u64,
    recycle_terms: usize,
    z3_timeout: Option<Duration>,
) -> Z3CompareRow {
    let Some(optimized_run) = &def.optimized else {
        return empty_row(
            def,
            "no optimized kernel (not an equivalence benchmark)".to_string(),
        );
    };

    let exec0 = Instant::now();
    let reference = match run_kernel(kernels_dir, &def.reference) {
        Ok(o) => o,
        Err(e) => return empty_row(def, format!("reference kernel: {:#}", e)),
    };
    let optimized = match run_kernel(kernels_dir, optimized_run) {
        Ok(o) => o,
        Err(e) => return empty_row(def, format!("optimized kernel: {:#}", e)),
    };
    let exec_secs = exec0.elapsed().as_secs_f64();

    let options = EquivCheckOptions {
        footprints: FootprintPolicy::Intersect,
        sample,
        verify_numeric: false,
        recycle_terms,
    };
    let d0 = Instant::now();
    let decision_status = match check_output_equivalence_with(&reference, &optimized, &options) {
        Ok(report) => match report.outcome {
            EquivOutcome::Equivalent => "EQUIV".to_string(),
            EquivOutcome::NotEquivalent { mismatches } => format!("DIFF({})", mismatches.len()),
        },
        Err(e) => return empty_row(def, format!("decision procedure: {}", e)),
    };
    let decision_secs = d0.elapsed().as_secs_f64();

    let (z3_secs, z3_equiv, z3_not_equiv, z3_unknown, z3_unsupported, z3_error, error) =
        match volta_z3::check_output_equivalence(
            &reference,
            &optimized,
            FootprintPolicy::Intersect,
            sample,
            z3_timeout,
        ) {
            Ok(report) => {
                let (e, n, u, us, er) = report.counts();
                (report.total_solve_secs(), e, n, u, us, er, None)
            }
            Err(e) => (0.0, 0, 0, 0, 0, 0, Some(format!("z3: {}", e))),
        };

    Z3CompareRow {
        name: def.name.clone(),
        category: def.category,
        exec_secs,
        decision_secs,
        decision_status,
        z3_secs,
        z3_equiv,
        z3_not_equiv,
        z3_unknown,
        z3_unsupported,
        z3_error,
        error,
    }
}

pub fn export_json(rows: &[Z3CompareRow], path: &Path) -> anyhow::Result<()> {
    let entries: Vec<_> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "name": r.name,
                "category": r.category.name(),
                "exec_secs": r.exec_secs,
                "decision_secs": r.decision_secs,
                "decision_status": r.decision_status,
                "z3_secs": r.z3_secs,
                "z3_equivalent": r.z3_equiv,
                "z3_not_equivalent": r.z3_not_equiv,
                "z3_unknown": r.z3_unknown,
                "z3_unsupported": r.z3_unsupported,
                "z3_error": r.z3_error,
                "error": r.error,
            })
        })
        .collect();
    std::fs::write(path, serde_json::to_string_pretty(&entries)?)?;
    Ok(())
}
