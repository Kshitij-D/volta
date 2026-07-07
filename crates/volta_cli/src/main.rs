//! Volta CLI - PTX analysis tool
//!
//! Commands:
//! - `volta parse <file>` - Parse a PTX file and report any errors

use std::path::PathBuf;
use std::process::ExitCode;

#[cfg(feature = "logging")]
use clap::ValueEnum;
use clap::{Parser, Subcommand};
use volta_analysis::driver::analyze_kernel;
use volta_analysis::eval::{AnalysisConfig, ArrayDef, ArrayKind, ParamValue};
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
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    #[cfg(feature = "logging")]
    env_logger::Builder::new()
        .filter_level(cli.log_level.into())
        .format_timestamp(None)
        .format_target(false)
        .init();

    match cli.command {
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
        ),
    }
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

#[allow(clippy::too_many_arguments)]
fn cmd_analyze(
    file: &PathBuf,
    kernel: Option<&str>,
    block: &str,
    grid: &str,
    arrays: &[String],
    params: &[String],
    globals: &[String],
    dyn_shared: u64,
    print_outputs: u64,
) -> ExitCode {
    let module = match load_module(file) {
        Ok(m) => m,
        Err(code) => return code,
    };

    let block_dim = match parse_dims(block) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: invalid --block: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let mut config = AnalysisConfig::new(block_dim);
    config.grid_dim = match parse_dims(grid) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: invalid --grid: {}", e);
            return ExitCode::FAILURE;
        }
    };
    config.dynamic_shared_bytes = dyn_shared;
    for a in arrays {
        match parse_array(a) {
            Ok(def) => config.arrays.push(def),
            Err(e) => {
                eprintln!("error: invalid --array: {}", e);
                return ExitCode::FAILURE;
            }
        }
    }
    for p in params {
        match parse_param(p) {
            Ok(v) => config.params.push(v),
            Err(e) => {
                eprintln!("error: invalid --param: {}", e);
                return ExitCode::FAILURE;
            }
        }
    }
    for g in globals {
        let Some((name, value)) = g.split_once('=') else {
            eprintln!("error: invalid --global (expected NAME=value): {}", g);
            return ExitCode::FAILURE;
        };
        match value.parse::<i64>() {
            Ok(v) => config.global_values.push((name.to_string(), v)),
            Err(e) => {
                eprintln!("error: invalid --global value: {}", e);
                return ExitCode::FAILURE;
            }
        }
    }

    match analyze_kernel(&module, kernel, config) {
        Ok(output) => {
            println!("Analysis complete: no data races or deadlocks detected.");
            println!(
                "  instructions: {}  block syncs: {}  warp syncs: {}",
                output.stats.instructions, output.stats.block_syncs, output.stats.warp_syncs
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
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("analysis failed: {}", e);
            ExitCode::FAILURE
        }
    }
}

/// Load and parse a module, reporting errors nicely.
fn load_module(file: &PathBuf) -> Result<Module, ExitCode> {
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
fn cmd_parse(file: &PathBuf) -> ExitCode {
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
