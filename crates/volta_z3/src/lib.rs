//! Z3 backend: translate a verification-condition pair to SMT-LIB2, shell
//! out to the `z3` CLI binary, and interpret unsat/sat/unknown. This is a
//! timing/capability comparison point against `volta_analysis::canon`'s own
//! decision procedure - not a replacement for it. See the `translate`
//! module for exactly which fragment of `ExprNode` this backend covers,
//! why, and how DAG sharing is preserved via SMT-LIB2 `let`.
//!
//! This  methodology (generate
//! SMT-LIB2, shell out to `z3`, read `sat`/`unsat`/`unknown` off stdout)
//! rather than linking Z3 via FFI: it needs only the `z3` binary on PATH,
//! no `libz3-dev`/`libclang-dev`/bindgen.

mod translate;

pub use translate::Unsupported;

use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use volta_analysis::symbolic::{ExprArena, ExprId};

use translate::{Builder, translate_root};

/// Outcome of one Z3 query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Z3Verdict {
    /// `unsat` on `(not (= a b))`: Z3 proved the two expressions equal.
    Equivalent,
    /// `sat`: Z3 found a model where they differ.
    NotEquivalent,
    /// `unknown` or the solver timed out - Z3 has no procedure for this
    /// query. The expected result for exponential-heavy (softmax/
    /// attention) VCs at realistic size; 
    Unknown,
}

#[derive(Debug, thiserror::Error)]
pub enum Z3Error {
    #[error(transparent)]
    Unsupported(#[from] Unsupported),
    #[error("failed to run `z3` ({0}) - is it installed and on PATH?")]
    Spawn(std::io::Error),
    #[error("z3 produced unexpected output: {0:?}")]
    UnexpectedOutput(String),
}

/// One element's check: the verdict and how long the `z3` subprocess took.
/// Translation time isn't included - that front-end cost is the same for
/// both backends and isn't the thing being compared.
#[derive(Debug, Clone)]
pub struct Z3CheckResult {
    pub verdict: Z3Verdict,
    pub solve_secs: f64,
}

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_query_path() -> std::path::PathBuf {
    let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("volta_z3_{}_{}.smt2", std::process::id(), n))
}

/// Check whether `a` (in `arena_a`) and `b` (in `arena_b`) are equal over
/// the reals, using Z3 instead of `volta_analysis::canon` as the decision
/// procedure. `timeout` bounds the solver call (`None` = no limit,
pub fn check_equivalent(
    arena_a: &ExprArena,
    a: ExprId,
    arena_b: &ExprArena,
    b: ExprId,
    timeout: Option<Duration>,
) -> Result<Z3CheckResult, Z3Error> {
    let mut builder = Builder::new();
    let ta = translate_root(&mut builder, arena_a, a)?;
    let tb = translate_root(&mut builder, arena_b, b)?;

    let body = builder.wrap_in_lets(&format!("(not (= {} {}))", ta, tb));
    let mut query = builder.preamble();
    query.push_str(&format!("(assert {})\n", body));
    query.push_str("(check-sat)\n");

    let path = temp_query_path();
    std::fs::write(&path, &query).map_err(Z3Error::Spawn)?;

    let mut cmd = Command::new("z3");
    if let Some(t) = timeout {
        cmd.arg(format!("-T:{}", t.as_secs().max(1)));
    }
    cmd.arg(&path);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let start = Instant::now();
    let output = cmd.output().map_err(Z3Error::Spawn)?;
    let solve_secs = start.elapsed().as_secs_f64();
    let _ = std::fs::remove_file(&path);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let verdict = match stdout.lines().map(str::trim).find(|l| !l.is_empty()) {
        Some("unsat") => Z3Verdict::Equivalent,
        Some("sat") => Z3Verdict::NotEquivalent,
        Some("unknown") | Some("timeout") => Z3Verdict::Unknown,
        _ => return Err(Z3Error::UnexpectedOutput(stdout.into_owned())),
    };

    Ok(Z3CheckResult { verdict, solve_secs })
}

/// Is the `z3` binary available on PATH? Check this before a whole
/// benchmark run for a clean error instead of N failures.
pub fn z3_available() -> bool {
    Command::new("z3")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Per-element outcome from the Z3 backend - a finer-grained result than
/// the decision procedure's binary equivalent/not, since Z3 can also fail
/// to decide (`Unknown`) or refuse a VC outright (`Unsupported`).
#[derive(Debug, Clone)]
pub enum ElementOutcome {
    Equivalent,
    NotEquivalent,
    Unknown,
    Unsupported(String),
    /// The `z3` call itself failed for a reason other than an unsupported
    /// fragment (e.g. malformed output) - recorded per element rather than
    /// aborting the whole comparison.
    Error(String),
}

#[derive(Debug, Clone)]
pub struct ElementResult {
    pub array: String,
    pub index: u64,
    pub outcome: ElementOutcome,
    pub solve_secs: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Z3EquivReport {
    pub elements: Vec<ElementResult>,
}

impl Z3EquivReport {
    /// (equivalent, not_equivalent, unknown, unsupported, error) counts.
    pub fn counts(&self) -> (usize, usize, usize, usize, usize) {
        let mut c = (0, 0, 0, 0, 0);
        for e in &self.elements {
            match &e.outcome {
                ElementOutcome::Equivalent => c.0 += 1,
                ElementOutcome::NotEquivalent => c.1 += 1,
                ElementOutcome::Unknown => c.2 += 1,
                ElementOutcome::Unsupported(_) => c.3 += 1,
                ElementOutcome::Error(_) => c.4 += 1,
            }
        }
        c
    }

    pub fn total_solve_secs(&self) -> f64 {
        self.elements.iter().map(|e| e.solve_secs).sum()
    }
}

/// z3 wasn't found on PATH - checked once up front so a missing install
/// fails fast with one clear message instead of N per-element failures.
#[derive(Debug, thiserror::Error)]
#[error("z3 is not installed / not on PATH (try: apt-get install z3)")]
pub struct Z3NotFound;

/// Check every paired output element (per `footprints`, exactly like
/// `volta_analysis::driver::check_output_equivalence_with`) with Z3 instead
/// of the decision procedure. Unlike the decision procedure, this never
/// aborts partway through a run over a single element's failure - each
/// element's outcome (including "unsupported" or a solver error) is
/// recorded independently, since the whole point is comparing coverage as
/// well as speed.
pub fn check_output_equivalence(
    reference: &volta_analysis::eval::AnalysisOutput,
    optimized: &volta_analysis::eval::AnalysisOutput,
    footprints: volta_analysis::driver::FootprintPolicy,
    sample: u64,
    timeout: Option<Duration>,
) -> Result<Z3EquivReport, volta_analysis::driver::EquivCheckError> {
    if !z3_available() {
        // Surface as a shape-mismatch-shaped error so callers that already
        // handle `EquivCheckError` don't need a second error type just for
        // this one fatal, checked-up-front case.
        return Err(volta_analysis::driver::EquivCheckError::ShapeMismatch {
            message: Z3NotFound.to_string(),
        });
    }

    let paired = volta_analysis::driver::paired_elements(reference, optimized, footprints)?;
    let mut elements = Vec::new();
    for (name, common) in paired {
        let limit = match sample {
            0 => common.len(),
            n => common.len().min(n as usize),
        };
        for (index, r, o) in common.into_iter().take(limit) {
            let (outcome, solve_secs) =
                match check_equivalent(&reference.arena, r, &optimized.arena, o, timeout) {
                    Ok(res) => (
                        match res.verdict {
                            Z3Verdict::Equivalent => ElementOutcome::Equivalent,
                            Z3Verdict::NotEquivalent => ElementOutcome::NotEquivalent,
                            Z3Verdict::Unknown => ElementOutcome::Unknown,
                        },
                        res.solve_secs,
                    ),
                    Err(Z3Error::Unsupported(u)) => (ElementOutcome::Unsupported(u.0), 0.0),
                    Err(e) => (ElementOutcome::Error(e.to_string()), 0.0),
                };
            elements.push(ElementResult {
                array: name.clone(),
                index,
                outcome,
                solve_secs,
            });
        }
    }
    Ok(Z3EquivReport { elements })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(arena_a: &ExprArena, a: ExprId, arena_b: &ExprArena, b: ExprId) -> Z3Verdict {
        if !z3_available() {
            panic!("z3 is not on PATH - install it to run these tests");
        }
        check_equivalent(arena_a, a, arena_b, b, Some(Duration::from_secs(10)))
            .unwrap()
            .verdict
    }

    #[test]
    fn commutative_add_is_equivalent() {
        // x + 1  vs  1 + x: same arena, different tree shape. Z3's own
        // arithmetic (not an opaque atom) proves this trivially.
        let mut ar = ExprArena::new();
        let x = ar.named("x");
        let one = ar.int(1);
        let lhs = ar.add(x, one);
        let rhs = ar.add(one, x);
        assert_eq!(check(&ar, lhs, &ar, rhs), Z3Verdict::Equivalent);
    }

    #[test]
    fn cross_arena_named_symbol_correlates_by_string() {
        // Two independent arenas, each with its own "x" - correlated by
        // name, exactly like `canon`'s own convention.
        let mut ar_a = ExprArena::new();
        let x_a = ar_a.named("x");
        let two_a = ar_a.int(2);
        let lhs = ar_a.mul(x_a, two_a);

        let mut ar_b = ExprArena::new();
        let x_b = ar_b.named("x");
        let rhs = ar_b.add(x_b, x_b);

        assert_eq!(check(&ar_a, lhs, &ar_b, rhs), Z3Verdict::Equivalent);
    }

    #[test]
    fn distinct_symbols_are_not_equivalent() {
        let mut ar = ExprArena::new();
        let x = ar.named("x");
        let y = ar.named("y");
        assert_eq!(check(&ar, x, &ar, y), Z3Verdict::NotEquivalent);
    }

    #[test]
    fn exp_addition_law_is_unknown_without_an_axiom() {
        // exp(a)*exp(b) vs exp(a+b): true over the reals, but Z3 alone
        // (native `^`, no axiom) has no decision procedure for symbolic
        // real exponents - reproduces documented finding
        let mut ar = ExprArena::new();
        let a = ar.named("a");
        let b = ar.named("b");
        let ea = ar.exp(a);
        let eb = ar.exp(b);
        let lhs = ar.mul(ea, eb);
        let sum = ar.add(a, b);
        let rhs = ar.exp(sum);
        assert_eq!(check(&ar, lhs, &ar, rhs), Z3Verdict::Unknown);
    }

    #[test]
    fn select_is_unsupported() {
        // A concrete condition constant-folds away (see `ExprArena`'s
        // "constructors constant-fold eagerly"), so use a symbolic
        // predicate to force an actual `Select` node.
        let mut ar = ExprArena::new();
        let c = ar.named("cond");
        let t = ar.int(1);
        let f = ar.int(0);
        let id = ar.select(c, t, f);
        let result = check_equivalent(&ar, id, &ar, id, None);
        assert!(matches!(result, Err(Z3Error::Unsupported(_))));
    }
}
