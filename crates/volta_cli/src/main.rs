//! Volta CLI - PTX analysis tool
//!
//! Commands:
//! - `volta parse <file>` - Parse a PTX file and report any errors
//! - `volta analyze <file>` - Symbolically execute one kernel
//! - `volta compare <file1> <file2>` - Check two kernels for equivalence

mod run_log;

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use volta_analysis::driver::{
    EquivCheckOptions, EquivOutcome, FootprintPolicy, VcDump, VcSnapshot, analyze_kernel,
    check_output_equivalence_with,
};
use volta_analysis::equiv::DEFAULT_RECYCLE_TERMS;
use volta_analysis::eval::{AnalysisConfig, AnalysisOutput, ArrayDef, ArrayKind, ParamValue};
use volta_frontend::ascii::{AsAscii, AsciiChar};
use volta_frontend::ast::{Module, TopLevelItem};
use volta_frontend::file_cache::FileCache;
use volta_frontend::parse;
use volta_frontend::report::{Report, locate_path, report_error};

/// Log level for controlling output verbosity
#[cfg(feature = "logging")]
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum LogLevel {
    /// Only show errors
    Error,
    /// Show warnings and errors
    #[default]
    Warn,
    /// Show info, warnings, and errors
    Info,
    /// Show debug output and above
    Debug,
    /// Show all log output including trace
    Trace,
}

#[cfg(feature = "logging")]
impl From<LogLevel> for log::LevelFilter {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Error => log::LevelFilter::Error,
            LogLevel::Warn => log::LevelFilter::Warn,
            LogLevel::Info => log::LevelFilter::Info,
            LogLevel::Debug => log::LevelFilter::Debug,
            LogLevel::Trace => log::LevelFilter::Trace,
        }
    }
}

/// How to pair up two kernels' written footprints (mirrors
/// `driver::FootprintPolicy`, just with `clap::ValueEnum` attached).
#[derive(Debug, Clone, Copy, ValueEnum)]
enum FootprintArg {
    /// Same output arrays, same written indices, element for element.
    Exact,
    /// Compare only the common written indices per array (e.g. a
    /// grid-stride reference vs. a tiled kernel).
    Intersect,
}

impl From<FootprintArg> for FootprintPolicy {
    fn from(f: FootprintArg) -> Self {
        match f {
            FootprintArg::Exact => FootprintPolicy::Exact,
            FootprintArg::Intersect => FootprintPolicy::Intersect,
        }
    }
}

/// Which decision procedure to check equivalence with.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendArg {
    /// `volta_analysis::canon` - Volta's own decision procedure (default)
    Decision,
    /// Z3, via SMT-LIB2 shelled out to the `z3` CLI binary. Covers a
    /// narrower fragment (see `volta_z3::translate`'s docs) and reports
    /// per-element unsat/sat/unknown rather than a single verdict.
    Z3,
}

#[derive(Parser)]
#[command(name = "volta")]
#[command(about = "The Volta PTX analysis engine.")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Log level for output verbosity
    #[cfg(feature = "logging")]
    #[arg(long, value_enum, default_value = "warn", global = true)]
    log_level: LogLevel,

    /// Directory for per-run log files
    #[arg(long, global = true, default_value = "volta-logs")]
    log_dir: PathBuf,

    /// Don't write a per-run log file
    #[arg(long, global = true)]
    no_log_file: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse a PTX file and check for syntax errors
    Parse {
        /// PTX file to parse
        file: PathBuf,
    },

    /// Symbolically execute a kernel: detect races/deadlocks and print the
    /// symbolic output tensors
    Analyze {
        /// PTX file to analyze
        file: PathBuf,

        /// Kernel entry name (defaults to the first kernel in the module)
        #[arg(short, long)]
        kernel: Option<String>,

        /// Block dimensions, e.g. "128" or "32,4,1"
        #[arg(short, long, default_value = "1")]
        block: String,

        /// Grid dimensions, e.g. "64,64"
        #[arg(short, long, default_value = "1")]
        grid: String,

        /// Global array: "name:base:elem_width:len:kind" where kind is
        /// in|out|inout|index (e.g. "in:0x10000:4:128:in"). Repeatable.
        #[arg(long = "array")]
        arrays: Vec<String>,

        /// Kernel parameter (in declaration order): "int:N", "float:X",
        /// "sym:name", or "ptr:array_name". Repeatable.
        #[arg(long = "param")]
        params: Vec<String>,

        /// Module-scope .global variable value: "NAME=value". Repeatable.
        #[arg(long = "global")]
        globals: Vec<String>,

        /// Dynamic (extern) shared memory size in bytes
        #[arg(long, default_value_t = 0)]
        dyn_shared: u64,

        /// Print up to N elements of each output array
        #[arg(long, default_value_t = 8)]
        print_outputs: u64,

        /// Skip the per-instruction-kind execution profile (shown by default)
        #[arg(long = "no-profile", action = clap::ArgAction::SetFalse, default_value_t = true)]
        profile: bool,
    },

    /// Check two kernels for semantic equivalence (each is also checked for
    /// data races/deadlocks). Arrays/params/globals are shared by both
    /// kernels unless a `--block2`/`--grid2` override is given.
    Compare {
        /// Reference PTX file (omit when using --from-dump)
        file1: Option<PathBuf>,

        /// Optimized PTX file (omit when using --from-dump)
        file2: Option<PathBuf>,

        /// Reference kernel entry name (defaults to the first in the module)
        #[arg(long)]
        kernel1: Option<String>,

        /// Optimized kernel entry name (defaults to the first in the module)
        #[arg(long)]
        kernel2: Option<String>,

        /// Block dimensions for both kernels, e.g. "128" or "32,4,1"
        #[arg(short, long, default_value = "1")]
        block: String,

        /// Block dimensions for the optimized kernel only, if it differs
        #[arg(long)]
        block2: Option<String>,

        /// Grid dimensions for both kernels, e.g. "64,64"
        #[arg(short, long, default_value = "1")]
        grid: String,

        /// Grid dimensions for the optimized kernel only, if it differs
        #[arg(long)]
        grid2: Option<String>,

        /// Global array, shared by both kernels: "name:base:elem_width:len:kind"
        /// (e.g. "in:0x10000:4:128:in"). Repeatable.
        #[arg(long = "array")]
        arrays: Vec<String>,

        /// Kernel parameter, shared by both kernels (in declaration order):
        /// "int:N", "float:X", "sym:name", or "ptr:array_name". Repeatable.
        #[arg(long = "param")]
        params: Vec<String>,

        /// Module-scope .global variable, shared by both kernels:
        /// "NAME=value". Repeatable.
        #[arg(long = "global")]
        globals: Vec<String>,

        /// Dynamic (extern) shared memory size in bytes, shared by both kernels
        #[arg(long, default_value_t = 0)]
        dyn_shared: u64,

        /// How to pair up the two kernels' written footprints
        #[arg(long, value_enum, default_value = "intersect")]
        footprint: FootprintArg,

        /// Check at most this many common elements per array (0 = all)
        #[arg(long, default_value_t = 0)]
        sample: u64,

        /// Confirm every verdict with the f64 numeric oracle
        #[arg(long)]
        verify_numeric: bool,

        /// Recycle the VC intern tables past this many interned terms (0 = never)
        #[arg(long, default_value_t = DEFAULT_RECYCLE_TERMS)]
        recycle_terms: usize,

        /// Skip the per-instruction-kind execution profile (shown by default)
        #[arg(long = "no-profile", action = clap::ArgAction::SetFalse, default_value_t = true)]
        profile: bool,

        /// Which decision procedure to use
        #[arg(long, value_enum, default_value = "decision")]
        backend: BackendArg,

        /// Per-query Z3 timeout in seconds, only used with --backend z3 (0 = no limit)
        #[arg(long, default_value_t = 30)]
        z3_timeout: u64,

        /// After symbolic execution, dump both kernels' verification
        /// conditions (the expression arena + output footprint) to this
        /// file. Reload them later with --from-dump to rerun the
        /// equivalence check without parsing/symbolic execution.
        #[arg(long)]
        dump_vcs: Option<PathBuf>,

        /// Skip parsing and symbolic execution entirely and check
        /// equivalence directly from a --dump-vcs file. FILE1/FILE2 and the
        /// launch-config flags are ignored when this is set.
        #[arg(long)]
        from_dump: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let command_name = match &cli.command {
        Commands::Parse { .. } => "parse",
        Commands::Analyze { .. } => "analyze",
        Commands::Compare { .. } => "compare",
    };
    let mut log = run_log::RunLog::open(&cli.log_dir, command_name, cli.no_log_file);

    #[cfg(feature = "logging")]
    env_logger::Builder::new()
        .filter_level(cli.log_level.into())
        .format_timestamp(None)
        .format_target(false)
        .target(env_logger::Target::Pipe(log.tee(io::stderr())))
        .init();

    let code = match cli.command {
        Commands::Parse { file } => cmd_parse(&file),
        Commands::Analyze {
            file,
            kernel,
            block,
            grid,
            arrays,
            params,
            globals,
            dyn_shared,
            print_outputs,
            profile,
        } => cmd_analyze(
            &file,
            kernel.as_deref(),
            &block,
            &grid,
            &arrays,
            &params,
            &globals,
            dyn_shared,
            print_outputs,
            profile,
            &mut log,
        ),
        Commands::Compare {
            file1,
            file2,
            kernel1,
            kernel2,
            block,
            block2,
            grid,
            grid2,
            arrays,
            params,
            globals,
            dyn_shared,
            footprint,
            sample,
            verify_numeric,
            recycle_terms,
            profile,
            backend,
            z3_timeout,
            dump_vcs,
            from_dump,
        } => cmd_compare(CompareArgs {
            file1,
            file2,
            kernel1,
            kernel2,
            block,
            block2,
            grid,
            grid2,
            arrays,
            params,
            globals,
            dyn_shared,
            footprint,
            sample,
            verify_numeric,
            recycle_terms,
            profile,
            backend,
            z3_timeout,
            dump_vcs,
            from_dump,
        }, &mut log),
    };

    if let Some(path) = log.path() {
        eprintln!("log: {}", path.display());
    }
    code
}

/// Parse "x[,y[,z]]" dimensions.
fn parse_dims(s: &str) -> Result<(u32, u32, u32), String> {
    let mut parts = s.split(',').map(|p| p.trim().parse::<u32>());
    let x = parts
        .next()
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(1);
    let y = parts
        .next()
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(1);
    let z = parts
        .next()
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(1);
    Ok((x, y, z))
}

fn parse_u64_value(s: &str) -> Result<u64, String> {
    if let Some(hex) = s.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).map_err(|e| e.to_string())
    } else {
        s.parse()
            .map_err(|e: std::num::ParseIntError| e.to_string())
    }
}

/// Parse "name:base:elem_width:len:kind".
fn parse_array(s: &str) -> Result<ArrayDef, String> {
    let parts: Vec<&str> = s.split(':').collect();
    let [name, base, elem_width, len, kind] = parts.as_slice() else {
        return Err(format!(
            "expected name:base:elem_width:len:kind, got '{}'",
            s
        ));
    };
    let kind = match *kind {
        "in" => ArrayKind::Input,
        "out" => ArrayKind::Output,
        "inout" => ArrayKind::InputOutput,
        "index" => ArrayKind::IndexInput,
        other => return Err(format!("unknown array kind '{}'", other)),
    };
    Ok(ArrayDef {
        name: name.to_string(),
        base: parse_u64_value(base)?,
        elem_width: parse_u64_value(elem_width)?,
        len: parse_u64_value(len)?,
        kind,
    })
}

/// Parse "int:N" | "float:X" | "sym:name" | "ptr:array".
fn parse_param(s: &str) -> Result<ParamValue, String> {
    let Some((kind, value)) = s.split_once(':') else {
        return Err(format!("expected kind:value, got '{}'", s));
    };
    match kind {
        "int" => Ok(ParamValue::Int(
            value.parse().map_err(|e| format!("{}", e))?,
        )),
        "float" => Ok(ParamValue::Float(
            value.parse().map_err(|e| format!("{}", e))?,
        )),
        "sym" => Ok(ParamValue::SymFloat(value.to_string())),
        "ptr" => Ok(ParamValue::ArrayPtr(value.to_string())),
        other => Err(format!("unknown param kind '{}'", other)),
    }
}

/// Shared launch-config inputs, parsed into an `AnalysisConfig` by
/// `build_config`. Used by both `analyze` and `compare` so the two
/// commands' array/param/global flags behave identically.
struct ConfigInput<'a> {
    block: &'a str,
    grid: &'a str,
    arrays: &'a [String],
    params: &'a [String],
    globals: &'a [String],
    dyn_shared: u64,
}

fn build_config(input: ConfigInput) -> Result<AnalysisConfig, String> {
    let block_dim = parse_dims(input.block).map_err(|e| format!("invalid --block: {}", e))?;
    let mut config = AnalysisConfig::new(block_dim);
    config.grid_dim = parse_dims(input.grid).map_err(|e| format!("invalid --grid: {}", e))?;
    config.dynamic_shared_bytes = input.dyn_shared;
    for a in input.arrays {
        config
            .arrays
            .push(parse_array(a).map_err(|e| format!("invalid --array: {}", e))?);
    }
    for p in input.params {
        config
            .params
            .push(parse_param(p).map_err(|e| format!("invalid --param: {}", e))?);
    }
    for g in input.globals {
        let (name, value) = g
            .split_once('=')
            .ok_or_else(|| format!("invalid --global (expected NAME=value): {}", g))?;
        let v: i64 = value
            .parse()
            .map_err(|e: std::num::ParseIntError| format!("invalid --global value: {}", e))?;
        config.global_values.push((name.to_string(), v));
    }
    Ok(config)
}

/// Print a per-instruction-kind execution profile, most-executed first.
fn print_op_counts(label: &str, counts: &BTreeMap<&'static str, u64>) {
    if counts.is_empty() {
        return;
    }
    let total: u64 = counts.values().sum();
    let mut entries: Vec<_> = counts.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    println!("  {} profile:", label);
    for (kind, count) in entries {
        let pct = 100.0 * *count as f64 / total.max(1) as f64;
        println!("    {:<16} {:>10}  ({:>5.1}%)", kind, count, pct);
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_analyze(
    file: &Path,
    kernel: Option<&str>,
    block: &str,
    grid: &str,
    arrays: &[String],
    params: &[String],
    globals: &[String],
    dyn_shared: u64,
    print_outputs: u64,
    profile: bool,
    log: &mut run_log::RunLog,
) -> ExitCode {
    let module = match load_module(file) {
        Ok(m) => m,
        Err(code) => return code,
    };

    let config = match build_config(ConfigInput {
        block,
        grid,
        arrays,
        params,
        globals,
        dyn_shared,
    }) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let start = Instant::now();
    let result = analyze_kernel(&module, kernel, config);
    let elapsed = start.elapsed().as_secs_f64();

    match result {
        Ok(output) => {
            println!("Analysis complete: no data races or deadlocks detected.");
            println!(
                "  time: {:.3}s  instructions: {}  block syncs: {}  warp syncs: {}",
                elapsed,
                output.stats.instructions,
                output.stats.block_syncs,
                output.stats.warp_syncs
            );
            for (name, elems) in &output.outputs {
                println!("  output '{}': {} element(s) written", name, elems.len());
                for (index, expr) in elems.iter().take(print_outputs as usize) {
                    println!(
                        "    {}[{}] = {}",
                        name,
                        index,
                        output.arena.display_expr(*expr)
                    );
                }
                if elems.len() as u64 > print_outputs {
                    println!("    ... ({} more)", elems.len() as u64 - print_outputs);
                }
            }
            if profile {
                print_op_counts("instruction", &output.op_counts);
            }
            log.record(&format!(
                "analyze {}: OK in {:.3}s, {} instructions",
                file.display(),
                elapsed,
                output.stats.instructions
            ));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("analysis failed: {}", e);
            log.record(&format!("analyze {}: FAILED: {}", file.display(), e));
            ExitCode::FAILURE
        }
    }
}

/// Arguments for `cmd_compare`, bundled to keep the call site readable.
struct CompareArgs {
    file1: Option<PathBuf>,
    file2: Option<PathBuf>,
    kernel1: Option<String>,
    kernel2: Option<String>,
    block: String,
    block2: Option<String>,
    grid: String,
    grid2: Option<String>,
    arrays: Vec<String>,
    params: Vec<String>,
    globals: Vec<String>,
    dyn_shared: u64,
    footprint: FootprintArg,
    sample: u64,
    verify_numeric: bool,
    recycle_terms: usize,
    profile: bool,
    backend: BackendArg,
    z3_timeout: u64,
    dump_vcs: Option<PathBuf>,
    from_dump: Option<PathBuf>,
}

fn cmd_compare(args: CompareArgs, log: &mut run_log::RunLog) -> ExitCode {
    if args.from_dump.is_some() && args.dump_vcs.is_some() {
        eprintln!("note: --dump-vcs is a no-op with --from-dump (nothing new to dump)");
    }

    let (reference, optimized, exec_secs): (AnalysisOutput, AnalysisOutput, Option<f64>) =
        if let Some(dump_path) = &args.from_dump {
            if args.file1.is_some() || args.file2.is_some() {
                eprintln!("note: --from-dump ignores FILE1/FILE2 and the launch-config flags");
            }
            let dump = match load_dump(dump_path) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("error: failed to read dump {}: {}", dump_path.display(), e);
                    return ExitCode::FAILURE;
                }
            };
            (
                dump.reference.into_analysis_output(),
                dump.optimized.into_analysis_output(),
                None,
            )
        } else {
            let (Some(file1), Some(file2)) = (args.file1.as_ref(), args.file2.as_ref()) else {
                eprintln!("error: compare needs FILE1 and FILE2 (or --from-dump)");
                return ExitCode::FAILURE;
            };

            let module1 = match load_module(file1) {
                Ok(m) => m,
                Err(code) => return code,
            };
            let module2 = match load_module(file2) {
                Ok(m) => m,
                Err(code) => return code,
            };

            let config1 = match build_config(ConfigInput {
                block: &args.block,
                grid: &args.grid,
                arrays: &args.arrays,
                params: &args.params,
                globals: &args.globals,
                dyn_shared: args.dyn_shared,
            }) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("error: {}", e);
                    return ExitCode::FAILURE;
                }
            };
            let config2 = match build_config(ConfigInput {
                block: args.block2.as_deref().unwrap_or(&args.block),
                grid: args.grid2.as_deref().unwrap_or(&args.grid),
                arrays: &args.arrays,
                params: &args.params,
                globals: &args.globals,
                dyn_shared: args.dyn_shared,
            }) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("error: {}", e);
                    return ExitCode::FAILURE;
                }
            };

            let start = Instant::now();
            let reference = match analyze_kernel(&module1, args.kernel1.as_deref(), config1) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("error: reference kernel: {}", e);
                    log.record(&format!("compare: reference kernel FAILED: {}", e));
                    return ExitCode::FAILURE;
                }
            };
            let optimized = match analyze_kernel(&module2, args.kernel2.as_deref(), config2) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("error: optimized kernel: {}", e);
                    log.record(&format!("compare: optimized kernel FAILED: {}", e));
                    return ExitCode::FAILURE;
                }
            };
            let exec_secs = start.elapsed().as_secs_f64();

            if args.profile {
                print_op_counts("reference instruction", &reference.op_counts);
                print_op_counts("optimized instruction", &optimized.op_counts);
            }
            println!(
                "Exec: {:.3}s  instructions: {}  block syncs: {}  warp syncs: {}",
                exec_secs,
                reference.stats.instructions + optimized.stats.instructions,
                optimized.stats.block_syncs,
                optimized.stats.warp_syncs
            );

            if let Some(dump_path) = &args.dump_vcs {
                let dump = VcDump {
                    reference: VcSnapshot::from_output(reference),
                    optimized: VcSnapshot::from_output(optimized),
                };
                if let Err(e) = write_dump(&dump, dump_path) {
                    eprintln!("error: failed to write dump {}: {}", dump_path.display(), e);
                    return ExitCode::FAILURE;
                }
                println!("Dumped verification conditions to {}", dump_path.display());
                (
                    dump.reference.into_analysis_output(),
                    dump.optimized.into_analysis_output(),
                    Some(exec_secs),
                )
            } else {
                (reference, optimized, Some(exec_secs))
            }
        };

    if exec_secs.is_none() {
        println!("Loaded verification conditions from dump (no fresh symbolic execution).");
    }

    match args.backend {
        BackendArg::Decision => {
            let options = EquivCheckOptions {
                footprints: args.footprint.into(),
                sample: args.sample,
                verify_numeric: args.verify_numeric,
                recycle_terms: args.recycle_terms,
            };
            let vc_start = Instant::now();
            let report = check_output_equivalence_with(&reference, &optimized, &options);
            let vc_secs = vc_start.elapsed().as_secs_f64();

            match report {
                Ok(report) => {
                    let elems = if report.elements_checked == report.elements_total {
                        format!("{}", report.elements_total)
                    } else {
                        format!("{}/{}", report.elements_checked, report.elements_total)
                    };
                    println!("VC check: {:.3}s  elements: {}", vc_secs, elems);
                    match report.outcome {
                        EquivOutcome::Equivalent => {
                            println!("EQUIVALENT");
                            log.record(&format!(
                                "compare: EQUIVALENT ({} elements, exec {:?}, vc {:.3}s)",
                                elems, exec_secs, vc_secs
                            ));
                            ExitCode::SUCCESS
                        }
                        EquivOutcome::NotEquivalent { mismatches } => {
                            println!("NOT EQUIVALENT: {} mismatched element(s)", mismatches.len());
                            for m in mismatches.iter().take(10) {
                                println!("  {}[{}]", m.array, m.index);
                            }
                            if mismatches.len() > 10 {
                                println!("  ... ({} more)", mismatches.len() - 10);
                            }
                            log.record(&format!(
                                "compare: NOT EQUIVALENT, {} mismatches",
                                mismatches.len()
                            ));
                            ExitCode::FAILURE
                        }
                    }
                }
                Err(e) => {
                    eprintln!("error: {}", e);
                    log.record(&format!("compare: FAILED: {}", e));
                    ExitCode::FAILURE
                }
            }
        }
        BackendArg::Z3 => {
            if !volta_z3::z3_available() {
                eprintln!("error: z3 is not installed / not on PATH (try: apt-get install z3)");
                return ExitCode::FAILURE;
            }
            let timeout = if args.z3_timeout == 0 {
                None
            } else {
                Some(std::time::Duration::from_secs(args.z3_timeout))
            };
            let vc_start = Instant::now();
            let report = volta_z3::check_output_equivalence(
                &reference,
                &optimized,
                args.footprint.into(),
                args.sample,
                timeout,
            );
            let vc_secs = vc_start.elapsed().as_secs_f64();

            match report {
                Ok(report) => {
                    let (equiv, not_equiv, unknown, unsupported, error) = report.counts();
                    println!(
                        "VC check: {:.3}s (z3 solve time {:.3}s)  elements: {}",
                        vc_secs,
                        report.total_solve_secs(),
                        report.elements.len()
                    );
                    println!(
                        "  equivalent: {}  not-equivalent: {}  unknown: {}  unsupported: {}  error: {}",
                        equiv, not_equiv, unknown, unsupported, error
                    );
                    for e in report.elements.iter().filter(|e| {
                        !matches!(e.outcome, volta_z3::ElementOutcome::Equivalent)
                    }).take(10) {
                        println!("  {}[{}]: {:?}", e.array, e.index, e.outcome);
                    }
                    log.record(&format!(
                        "compare (z3): equiv={} not_equiv={} unknown={} unsupported={} error={} (vc {:.3}s)",
                        equiv, not_equiv, unknown, unsupported, error, vc_secs
                    ));
                    if not_equiv > 0 {
                        ExitCode::FAILURE
                    } else {
                        ExitCode::SUCCESS
                    }
                }
                Err(e) => {
                    eprintln!("error: {}", e);
                    log.record(&format!("compare (z3): FAILED: {}", e));
                    ExitCode::FAILURE
                }
            }
        }
    }
}

fn write_dump(dump: &VcDump, path: &Path) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = io::BufWriter::new(file);
    bincode::serialize_into(writer, dump).map_err(io::Error::other)
}

fn load_dump(path: &Path) -> io::Result<VcDump> {
    let file = std::fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    bincode::deserialize_from(reader).map_err(io::Error::other)
}

/// Load and parse a module, reporting errors nicely.
fn load_module(file: &Path) -> Result<Module, ExitCode> {
    let mut files = FileCache::new();
    let contents = match files.read(file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to read {}: {}", file.display(), e);
            return Err(ExitCode::FAILURE);
        }
    };
    let ascii_src: &[AsciiChar] = match contents.as_bytes().as_ascii_slice() {
        Some(src) => src,
        None => {
            eprintln!("error: file contains non-ASCII character");
            return Err(ExitCode::FAILURE);
        }
    };
    let mut parser = parse::Parser::new(ascii_src);
    match parser.parse_module().map_err(locate_path(file)) {
        Ok(module) => Ok(module),
        Err(e) => {
            let _ = report_error(
                &mut std::io::stderr(),
                &files,
                Report {
                    path: e.path.as_deref(),
                    span: e.span,
                    title: e.error.title(),
                    message: e.error.message().as_deref(),
                },
            );
            Err(ExitCode::FAILURE)
        }
    }
}

/// Parse a PTX file and report any errors
fn cmd_parse(file: &Path) -> ExitCode {
    let mut files = FileCache::new();

    let contents = match files.read(file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to read {}: {}", file.display(), e);
            return ExitCode::FAILURE;
        }
    };

    let ascii_src: &[AsciiChar] = match contents.as_bytes().as_ascii_slice() {
        Some(src) => src,
        None => {
            eprintln!("error: file contains non-ASCII character");
            return ExitCode::FAILURE;
        }
    };

    let mut parser = parse::Parser::new(ascii_src);
    match parser.parse_module().map_err(locate_path(file)) {
        Ok(module) => {
            println!("Parsed successfully: {}", file.display());
            print_module_summary(&module);
            ExitCode::SUCCESS
        }
        Err(e) => {
            let _ = report_error(
                &mut std::io::stderr(),
                &files,
                Report {
                    path: e.path.as_deref(),
                    span: e.span,
                    title: e.error.title(),
                    message: e.error.message().as_deref(),
                },
            );
            ExitCode::FAILURE
        }
    }
}

/// Print a summary of the parsed module
fn print_module_summary(module: &Module) {
    let mut entries = 0;
    let mut functions = 0;

    for item in &module.items {
        match item {
            TopLevelItem::Entry(_) => entries += 1,
            TopLevelItem::Function(_) => functions += 1,
            _ => {}
        }
    }

    println!("  Entries (kernels): {}", entries);
    if functions > 0 {
        println!("  Functions: {}", functions);
    }
}
