//! Volta benchmark runner CLI.
//!
//! Note: run in release mode; symbolic execution of the larger kernels is
//! ~20x slower unoptimized.
//!
//! ```bash
//! cargo run --release -p volta_bench -- all --sample 16
//! cargo run --release -p volta_bench -- category reduction
//! cargo run --release -p volta_bench -- single "(Red-1, Red-2)"
//! cargo run --release -p volta_bench -- list
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use volta_bench::{
    BenchmarkCategory, BenchmarkRunner, KERNELS_DIR, RunnerConfig, all_benchmarks, export_json,
    print_all_results, print_results_table, print_summary,
};

#[derive(Parser)]
#[command(name = "volta-bench")]
#[command(about = "Volta benchmark runner - reproduces the paper evaluation")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbose output (prints progress per benchmark)
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Custom kernels directory
    #[arg(long, global = true)]
    kernels_dir: Option<PathBuf>,

    /// Check at most this many output elements per array (0 = all).
    #[arg(long, global = true, default_value_t = 0)]
    sample: u64,

    /// Confirm every equivalence verdict with the f64 numeric oracle
    #[arg(long, global = true)]
    verify_numeric: bool,

    /// Recycle the VC intern tables past this many interned terms. Lower
    /// values bound VC memory at the cost of re-canonicalizing shared
    /// structure (0 = never recycle).
    #[arg(long, global = true, default_value_t = volta_analysis::equiv::DEFAULT_RECYCLE_TERMS)]
    recycle_terms: usize,
}

#[derive(Subcommand)]
enum Commands {
    /// Run all benchmarks
    All {
        /// Export results to JSON file
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Run benchmarks for one category
    Category {
        /// reduction | matmul | attention | causal | conv | agent | tilelang | race
        category: String,
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Run a single benchmark by name
    Single { name: String },
    /// List all benchmarks
    List,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let runner_config = RunnerConfig {
        kernels_dir: cli
            .kernels_dir
            .unwrap_or_else(|| PathBuf::from(KERNELS_DIR)),
        verbose: cli.verbose,
        sample: cli.sample,
        verify_numeric: cli.verify_numeric,
        recycle_terms: cli.recycle_terms,
    };

    match cli.command {
        Commands::All { json } => {
            let suite = all_benchmarks();
            println!("Running {} benchmarks...", suite.benchmarks.len());
            let runner = BenchmarkRunner::new(runner_config);
            let results = runner.run_all(&suite.benchmarks);
            let mut stdout = std::io::stdout();
            print_all_results(&mut stdout, &results).unwrap();
            if let Some(path) = json {
                export_json(&results, &path).unwrap();
                println!("Results exported to {}", path.display());
            }
            exit_by_pass(results.iter().all(|r| r.passed))
        }
        Commands::Category { category, json } => {
            let Some(category) = parse_category(&category) else {
                eprintln!("Unknown category: {}", category);
                eprintln!(
                    "Available: reduction, matmul, attention, causal, conv, agent, tilelang, race"
                );
                return ExitCode::FAILURE;
            };
            let suite = all_benchmarks();
            let filtered: Vec<_> = suite
                .filter_category(category)
                .into_iter()
                .cloned()
                .collect();
            println!(
                "Running {} benchmarks for {}...",
                filtered.len(),
                category.name()
            );
            let runner = BenchmarkRunner::new(runner_config);
            let results = runner.run_all(&filtered);
            let mut stdout = std::io::stdout();
            print_results_table(&mut stdout, &results, category).unwrap();
            print_summary(&mut stdout, &results).unwrap();
            if let Some(path) = json {
                export_json(&results, &path).unwrap();
                println!("Results exported to {}", path.display());
            }
            exit_by_pass(results.iter().all(|r| r.passed))
        }
        Commands::Single { name } => {
            let suite = all_benchmarks();
            let Some(def) = suite.benchmarks.iter().find(|b| b.name == name) else {
                eprintln!("Benchmark not found: {}", name);
                eprintln!("Use 'volta-bench list' to see available benchmarks.");
                return ExitCode::FAILURE;
            };
            println!("Running {} ...", name);
            let runner = BenchmarkRunner::new(runner_config);
            let result = runner.run(def);
            println!("Status:  {}", result.outcome.status());
            println!(
                "Detail:  {}",
                volta_bench::reporter::describe(&result.outcome)
            );
            println!("Passed:  {}", if result.passed { "yes" } else { "no" });
            println!("Exec:    {:.2}s", result.stats.exec_secs);
            println!("VC:      {:.2}s", result.stats.vc_secs);
            println!("Instrs:  {}", result.stats.instructions);
            println!(
                "Syncs:   {} block, {} warp",
                result.stats.block_syncs, result.stats.warp_syncs
            );
            println!(
                "Elems:   {} checked of {}",
                result.stats.elements_checked, result.stats.elements_total
            );
            if !result.passed {
                let mut out = Vec::new();
                print_summary(&mut out, std::slice::from_ref(&result)).unwrap();
                print!("{}", String::from_utf8_lossy(&out));
            }
            exit_by_pass(result.passed)
        }
        Commands::List => {
            let suite = all_benchmarks();
            for category in suite.categories() {
                println!("{}:", category.name());
                for b in suite.filter_category(category) {
                    println!("  - {}", b.name);
                }
            }
            println!("Total: {} benchmarks", suite.benchmarks.len());
            ExitCode::SUCCESS
        }
    }
}

fn exit_by_pass(passed: bool) -> ExitCode {
    if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn parse_category(name: &str) -> Option<BenchmarkCategory> {
    match name.to_lowercase().as_str() {
        "reduction" | "red" => Some(BenchmarkCategory::Reduction),
        "matmul" | "mm" => Some(BenchmarkCategory::MatMul),
        "attention" | "attn" => Some(BenchmarkCategory::Attention),
        "causal" | "causal-attention" | "causal-attn" => Some(BenchmarkCategory::CausalAttention),
        "convolution" | "conv" => Some(BenchmarkCategory::Convolution),
        "agent" | "agent-generated" => Some(BenchmarkCategory::AgentGenerated),
        "compiler" | "compiler-generated" | "tilelang" | "tl" => {
            Some(BenchmarkCategory::CompilerGenerated)
        }
        "datarace" | "race" | "races" => Some(BenchmarkCategory::DataRace),
        _ => None,
    }
}
