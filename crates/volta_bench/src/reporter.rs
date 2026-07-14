//! Result reporting: tables, summaries, and JSON export.

use std::io::Write;
use std::path::Path;

use anyhow::Result;
use serde_json::json;

use crate::config::BenchmarkCategory;
use crate::runner::{ActualOutcome, BenchmarkResult};

/// Print a per-instruction-kind execution profile, most-executed first.
/// Same format as `volta_cli`'s own profile table, so output looks
/// consistent whichever tool produced it.
pub fn print_op_counts(
    out: &mut impl Write,
    label: &str,
    counts: &std::collections::BTreeMap<&'static str, u64>,
) -> Result<()> {
    if counts.is_empty() {
        return Ok(());
    }
    let total: u64 = counts.values().sum();
    let mut entries: Vec<_> = counts.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    writeln!(out, "{} profile:", label)?;
    for (kind, count) in entries {
        let pct = 100.0 * *count as f64 / total.max(1) as f64;
        writeln!(out, "  {:<16} {:>10}  ({:>5.1}%)", kind, count, pct)?;
    }
    Ok(())
}

/// Print one category's results as a table (paper-style columns).
pub fn print_results_table(
    out: &mut impl Write,
    results: &[BenchmarkResult],
    category: BenchmarkCategory,
) -> Result<()> {
    writeln!(out, "\n{} ({})", category.name(), category.table_ref())?;
    writeln!(
        out,
        "{:<28} {:>7} {:>9} {:>9} {:>11} {:>11} {:>9}",
        "Benchmark", "Status", "Exec (s)", "VC (s)", "#BlockSync", "#WarpSync", "Elems"
    )?;
    writeln!(out, "{}", "-".repeat(90))?;
    for r in results.iter().filter(|r| r.category == category) {
        let elems = if r.stats.elements_checked == r.stats.elements_total {
            format!("{}", r.stats.elements_total)
        } else {
            format!("{}/{}", r.stats.elements_checked, r.stats.elements_total)
        };
        writeln!(
            out,
            "{:<28} {:>7} {:>9.2} {:>9.2} {:>11} {:>11} {:>9}",
            r.name,
            r.outcome.status(),
            r.stats.exec_secs,
            r.stats.vc_secs,
            r.stats.block_syncs,
            r.stats.warp_syncs,
            elems,
        )?;
        if !r.passed {
            writeln!(out, "    UNEXPECTED: {}", describe(&r.outcome))?;
        }
    }
    Ok(())
}

/// Print all results grouped by category.
pub fn print_all_results(out: &mut impl Write, results: &[BenchmarkResult]) -> Result<()> {
    for category in BenchmarkCategory::all() {
        if results.iter().any(|r| r.category == category) {
            print_results_table(out, results, category)?;
        }
    }
    print_summary(out, results)
}

pub fn print_summary(out: &mut impl Write, results: &[BenchmarkResult]) -> Result<()> {
    let passed = results.iter().filter(|r| r.passed).count();
    writeln!(out, "\n{}/{} benchmarks passed", passed, results.len())?;
    for r in results.iter().filter(|r| !r.passed) {
        writeln!(out, "  FAILED {}: {}", r.name, describe(&r.outcome))?;
    }
    Ok(())
}

pub fn describe(outcome: &ActualOutcome) -> String {
    match outcome {
        ActualOutcome::Equivalent => "equivalent".to_string(),
        ActualOutcome::NotEquivalent { mismatches, first } => {
            format!("{} mismatched elements (first: {})", mismatches, first)
        }
        ActualOutcome::Rejected { description, .. } => description.clone(),
        ActualOutcome::RaceFree => "race-free".to_string(),
        ActualOutcome::Error { message } => message.clone(),
    }
}

/// Export results as JSON.
pub fn export_json(results: &[BenchmarkResult], path: &Path) -> Result<()> {
    let entries: Vec<_> = results
        .iter()
        .map(|r| {
            json!({
                "name": r.name,
                "category": r.category.name(),
                "status": r.outcome.status(),
                "detail": describe(&r.outcome),
                "passed": r.passed,
                "elapsed_secs": r.elapsed_secs,
                "exec_secs": r.stats.exec_secs,
                "vc_secs": r.stats.vc_secs,
                "block_syncs": r.stats.block_syncs,
                "warp_syncs": r.stats.warp_syncs,
                "instructions": r.stats.instructions,
                "elements_checked": r.stats.elements_checked,
                "elements_total": r.stats.elements_total,
            })
        })
        .collect();
    std::fs::write(path, serde_json::to_string_pretty(&entries)?)?;
    Ok(())
}
