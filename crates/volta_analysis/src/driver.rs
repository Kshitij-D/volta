//! High-level analysis driver: parse tree in, per-output-element symbolic
//! expressions (or a race/deadlock/structured-CTA error) out.

use std::fmt;

use volta_frontend::ast::{Function, Module, TopLevelItem, VarDecl};

use crate::equiv::{DEFAULT_RECYCLE_TERMS, EquivError, EquivSession};
use crate::eval::{AnalysisConfig, AnalysisOutput, EvalError, Interpreter};
use crate::logging::info;
use crate::lower_error::LowerError;
use crate::lowering::lower_function;
use crate::numeric;
use crate::symbolic::ExprId;

/// Errors from the end-to-end analysis of one kernel.
#[derive(Debug)]
pub enum AnalysisError {
    KernelNotFound { name: Option<String> },
    Lower(LowerError),
    Eval(EvalError),
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelNotFound { name: Some(name) } => {
                write!(f, "no kernel named '{}' in module", name)
            }
            Self::KernelNotFound { name: None } => write!(f, "no kernel entry in module"),
            Self::Lower(e) => write!(f, "lowering failed: {}", e),
            Self::Eval(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for AnalysisError {}

impl From<LowerError> for AnalysisError {
    fn from(e: LowerError) -> Self {
        Self::Lower(e)
    }
}

impl From<EvalError> for AnalysisError {
    fn from(e: EvalError) -> Self {
        Self::Eval(e)
    }
}

/// Find a kernel entry point by name, or the unique entry if `name` is None.
pub fn find_kernel<'m>(
    module: &'m Module,
    name: Option<&str>,
) -> Result<&'m Function, AnalysisError> {
    let mut entries = module.items.iter().filter_map(|item| match item {
        TopLevelItem::Entry(f) => Some(f),
        _ => None,
    });
    match name {
        Some(name) => {
            entries
                .find(|f| f.name.to_string() == name)
                .ok_or(AnalysisError::KernelNotFound {
                    name: Some(name.to_string()),
                })
        }
        None => entries
            .next()
            .ok_or(AnalysisError::KernelNotFound { name: None }),
    }
}

/// Module-level variable declarations (extern shared memory, module globals).
pub fn module_vars(module: &Module) -> Vec<VarDecl> {
    module
        .items
        .iter()
        .filter_map(|item| match item {
            TopLevelItem::Variable(v) => Some(v.clone()),
            _ => None,
        })
        .collect()
}

/// Analyze one kernel: lower it and symbolically execute all threads of
/// CTA (0,0,0) under the given configuration.
pub fn analyze_kernel(
    module: &Module,
    kernel: Option<&str>,
    config: AnalysisConfig,
) -> Result<AnalysisOutput, AnalysisError> {
    let func = find_kernel(module, kernel)?;
    let vars = module_vars(module);
    let program = lower_function(func, &vars)?;
    info!(
        "analyzing kernel {:?}: block={:?} grid={:?}",
        kernel, config.block_dim, config.grid_dim
    );
    let mut interp = Interpreter::new(&program, config)?;
    interp.run()?;
    Ok(interp.into_output()?)
}

/// A single output element where the two kernels disagree.
#[derive(Debug, Clone)]
pub struct Mismatch {
    pub array: String,
    pub index: u64,
}

/// Result of comparing two analysis outputs.
#[derive(Debug)]
pub enum EquivOutcome {
    Equivalent,
    NotEquivalent { mismatches: Vec<Mismatch> },
}

/// Errors from output comparison.
#[derive(Debug)]
pub enum EquivCheckError {
    /// The two outputs have different arrays or element counts.
    ShapeMismatch { message: String },
    /// The underlying symbolic check failed.
    Equiv(EquivError),
    /// The f64 oracle contradicted (or could not confirm) a verdict.
    Numeric { message: String },
}

impl fmt::Display for EquivCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { message } => write!(f, "output shape mismatch: {}", message),
            Self::Equiv(e) => write!(f, "equivalence check failed: {}", e),
            Self::Numeric { message } => write!(f, "numeric oracle: {}", message),
        }
    }
}

impl std::error::Error for EquivCheckError {}

impl From<EquivError> for EquivCheckError {
    fn from(e: EquivError) -> Self {
        Self::Equiv(e)
    }
}

/// How to pair up the two kernels' written footprints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FootprintPolicy {
    /// The footprints must be identical: same output arrays, same written
    /// indices, element for element.
    Exact,
    /// Compare only the intersection of the written indices per array
    /// (e.g. a grid-stride reference and a tiled kernel cover different
    /// slices of the output). Arrays present only in the optimized output
    /// (auxiliary buffers) are ignored; an empty intersection over a
    /// nonempty array is an error.
    Intersect,
}

/// Options for [`check_output_equivalence_with`].
#[derive(Debug, Clone)]
pub struct EquivCheckOptions {
    pub footprints: FootprintPolicy,
    /// Check at most this many common elements per array (0 = all).
    pub sample: u64,
    /// Confirm every verdict with the f64 numeric oracle.
    pub verify_numeric: bool,
    /// Recycle the VC intern tables past this many interned terms
    /// (0 = never); see `EquivSession::with_recycle_terms`.
    pub recycle_terms: usize,
}

impl Default for EquivCheckOptions {
    fn default() -> Self {
        Self {
            footprints: FootprintPolicy::Exact,
            sample: 0,
            verify_numeric: false,
            recycle_terms: DEFAULT_RECYCLE_TERMS,
        }
    }
}

/// The outcome of a comparison plus how much of the footprint it covered.
#[derive(Debug)]
pub struct EquivCheckReport {
    pub outcome: EquivOutcome,
    /// Elements actually compared (less than total when sampling).
    pub elements_checked: u64,
    /// Comparable elements in the (possibly intersected) footprints.
    pub elements_total: u64,
}

/// Check two analysis outputs element by element under `options`. One
/// `EquivSession` is shared across all elements: structure shared between
/// elements (and between the two kernels) canonicalizes once.
pub fn check_output_equivalence_with(
    reference: &AnalysisOutput,
    optimized: &AnalysisOutput,
    options: &EquivCheckOptions,
) -> Result<EquivCheckReport, EquivCheckError> {
    if options.footprints == FootprintPolicy::Exact
        && reference.outputs.len() != optimized.outputs.len()
    {
        return Err(EquivCheckError::ShapeMismatch {
            message: format!(
                "{} output arrays vs {}",
                reference.outputs.len(),
                optimized.outputs.len()
            ),
        });
    }

    let mut session = EquivSession::with_recycle_terms(options.recycle_terms);
    let mut mismatches = Vec::new();
    let mut elements_checked = 0u64;
    let mut elements_total = 0u64;

    for (name, ref_elems) in &reference.outputs {
        let Some((_, opt_elems)) = optimized.outputs.iter().find(|(n, _)| n == name) else {
            return Err(EquivCheckError::ShapeMismatch {
                message: format!("optimized run has no output array '{}'", name),
            });
        };

        // Pair up comparable elements per the footprint policy.
        let common: Vec<(u64, ExprId, ExprId)> = match options.footprints {
            FootprintPolicy::Exact => {
                if ref_elems.len() != opt_elems.len() {
                    return Err(EquivCheckError::ShapeMismatch {
                        message: format!(
                            "array '{}': {} elements written vs {}",
                            name,
                            ref_elems.len(),
                            opt_elems.len()
                        ),
                    });
                }
                let mut common = Vec::with_capacity(ref_elems.len());
                for (&(ri, r), &(oi, o)) in ref_elems.iter().zip(opt_elems.iter()) {
                    if ri != oi {
                        return Err(EquivCheckError::ShapeMismatch {
                            message: format!(
                                "array '{}': written footprints differ (element {} vs {})",
                                name, ri, oi
                            ),
                        });
                    }
                    common.push((ri, r, o));
                }
                common
            }
            FootprintPolicy::Intersect => {
                let mut common = Vec::new();
                let (mut i, mut j) = (0usize, 0usize);
                while i < ref_elems.len() && j < opt_elems.len() {
                    let (ri, r) = ref_elems[i];
                    let (oi, o) = opt_elems[j];
                    match ri.cmp(&oi) {
                        std::cmp::Ordering::Less => i += 1,
                        std::cmp::Ordering::Greater => j += 1,
                        std::cmp::Ordering::Equal => {
                            common.push((ri, r, o));
                            i += 1;
                            j += 1;
                        }
                    }
                }
                if common.is_empty() && !(ref_elems.is_empty() && opt_elems.is_empty()) {
                    return Err(EquivCheckError::ShapeMismatch {
                        message: format!(
                            "array '{}': the two kernels' CTA-0 footprints do not overlap",
                            name
                        ),
                    });
                }
                common
            }
        };

        elements_total += common.len() as u64;
        let limit = match options.sample {
            0 => common.len(),
            n => common.len().min(n as usize),
        };
        for &(index, r, o) in common.iter().take(limit) {
            let equivalent = session.check(&reference.arena, r, &optimized.arena, o)?;
            if options.verify_numeric {
                numeric::verify_verdict(&reference.arena, r, &optimized.arena, o, equivalent)
                    .map_err(|message| EquivCheckError::Numeric {
                        message: format!("array '{}' element {}: {}", name, index, message),
                    })?;
            }
            if !equivalent {
                mismatches.push(Mismatch {
                    array: name.clone(),
                    index,
                });
            }
            elements_checked += 1;
        }
    }

    let outcome = if mismatches.is_empty() {
        EquivOutcome::Equivalent
    } else {
        EquivOutcome::NotEquivalent { mismatches }
    };
    Ok(EquivCheckReport {
        outcome,
        elements_checked,
        elements_total,
    })
}

/// Check that two analysis outputs agree on every element of every output
/// array under the default options: identical footprints required, all
/// elements checked, no numeric oracle.
pub fn check_output_equivalence(
    reference: &AnalysisOutput,
    optimized: &AnalysisOutput,
) -> Result<EquivOutcome, EquivCheckError> {
    check_output_equivalence_with(reference, optimized, &EquivCheckOptions::default())
        .map(|report| report.outcome)
}
