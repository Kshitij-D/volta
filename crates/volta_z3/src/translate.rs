//! Translates the arithmetic + Exp + Max/Min fragment of `ExprArena` into
//! SMT-LIB2 text for Z3, mirroring `volta_analysis::canon`'s own atom
//! boundary and symbol-naming convention so the two backends are compared
//! on equal footing (same fragment, same cross-arena symbol correlation:
//! `NamedSymbol` by its string, `Symbol` by `s{id}` - see
//! `canon::canonicalize::canon_value`).
//!
//! Ops outside this fragment (`Select`, comparisons, bitwise, `Rem`,
//! `Log`, `Sqrt`, `Abs`, `SymbolicRead`) are refused outright
//! (`Unsupported`) rather than modeled as opaque atoms: an opaque atom is
//! only sound if syntactically-distinct-but-equal occurrences share a key,
//! and getting that right in general needs `canon`'s own `Rat`-based
//! normalization of the atom's arguments. `Max`/`Min` are the one
//! exception - needed for every softmax/attention benchmark - modeled as
//! opaque atoms using the same "flatten, sort, dedup" rule `canon` itself
//! applies for the same op (see `canonicalize_maxmin`), so Z3 sees exactly
//! what the decision procedure would (and wouldn't) distinguish. Note this
//! only normalizes the direct Max/Min argument list, not commutativity
//! *inside* those arguments (`Max(a+b, c)` vs `Max(b+a, c)` are not
//! recognized as the same atom) - a documented incompleteness, not a
//! silent one.
//!
//! `ExprArena` is a DAG, not a tree - shared subexpressions (e.g. the
//! softmax row-max, reused by every term) are the norm, not the exception.
//! Every compound node is therefore translated once and bound to a fresh
//! SMT-LIB2 `let` variable, memoized per arena by `ExprId`; naively
//! re-expanding the DAG into a flat tree would blow up exponentially on
//! exactly the attention/softmax benchmarks this backend exists to test.

use std::collections::HashMap;
use std::fmt::Write as _;

use volta_analysis::symbolic::{ExprArena, ExprId, ExprNode};

#[derive(Debug, Clone, thiserror::Error)]
#[error("unsupported by the Z3 backend: {0}")]
pub struct Unsupported(pub String);

/// Accumulates declared symbols, opaque Max/Min atoms, and `let` bindings
/// across a query. Reference and optimized sides of one VC element share a
/// `Builder`, so a shared input symbol (e.g. both kernels reading `in[5]`)
/// collapses to the same Z3 constant - the same correlation `canon` relies
/// on. `bindings` is filled in dependency order (a binding's definition can
/// only reference earlier ones), so wrapping the final query in nested
/// `let`s in reverse order is always well-scoped.
#[derive(Default)]
pub struct Builder {
    reals: std::collections::BTreeMap<String, ()>,
    maxmin: std::collections::BTreeMap<(bool, Vec<String>), String>,
    next_atom: u32,
    bindings: Vec<(String, String)>,
    next_let: u32,
}

impl Builder {
    pub fn new() -> Self {
        Self::default()
    }

    fn declare_real(&mut self, quoted_name: String) -> String {
        self.reals.entry(quoted_name.clone()).or_insert(());
        quoted_name
    }

    /// Bind a compound expression's definition to a fresh variable and
    /// return the variable name - callers must have already resolved any
    /// child references (to earlier variable names or literals) inside
    /// `def` before calling this.
    fn bind(&mut self, def: String) -> String {
        let name = format!("|t{}|", self.next_let);
        self.next_let += 1;
        self.bindings.push((name.clone(), def));
        name
    }

    fn maxmin_atom(&mut self, is_max: bool, mut args: Vec<String>) -> String {
        args.sort();
        args.dedup();
        let key = (is_max, args);
        if let Some(name) = self.maxmin.get(&key) {
            return name.clone();
        }
        let name = format!("|{}_{}|", if is_max { "max" } else { "min" }, self.next_atom);
        self.next_atom += 1;
        self.maxmin.insert(key, name.clone());
        self.reals.entry(name.clone()).or_insert(());
        name
    }

    /// SMT-LIB2 preamble: the `e` constant plus every declared symbol and
    /// opaque atom, as `declare-const`s.
    pub fn preamble(&self) -> String {
        let mut out = String::from("(define-fun e () Real 2.718281828459045)\n");
        for name in self.reals.keys() {
            let _ = writeln!(out, "(declare-const {} Real)", name);
        }
        out
    }

    /// Wrap `body` in every accumulated `let` binding, innermost binding
    /// (last emitted) closest to `body` and outermost (first emitted)
    /// on the outside - so each definition's free variables are already
    /// in scope.
    pub fn wrap_in_lets(&self, body: &str) -> String {
        let mut out = body.to_string();
        for (name, def) in self.bindings.iter().rev() {
            out = format!("(let (({} {})) {})", name, def, out);
        }
        out
    }
}

fn quote(name: &str) -> String {
    format!("|{}|", name.replace('\\', "\\\\").replace('|', "\\|"))
}

fn real_literal(v: f64) -> Result<String, Unsupported> {
    if !v.is_finite() {
        return Err(Unsupported(format!("non-finite float constant {}", v)));
    }
    let mag = v.abs();
    let s = format!("{}", mag);
    let body = if s.contains('e') || s.contains('E') {
        // Rust's f64 Display never emits this, but stay defensive.
        format!("{:.17}", mag)
    } else if s.contains('.') {
        s
    } else {
        format!("{}.0", s)
    };
    Ok(if v.is_sign_negative() && mag != 0.0 {
        format!("(- {})", body)
    } else {
        body
    })
}

fn int_literal(v: i64) -> String {
    let body = format!("{}.0", (v as i128).unsigned_abs());
    if v < 0 { format!("(- {})", body) } else { body }
}

fn is_add(node: &ExprNode) -> Option<(ExprId, ExprId)> {
    match node {
        ExprNode::Add(a, b) => Some((*a, *b)),
        _ => None,
    }
}

fn is_mul(node: &ExprNode) -> Option<(ExprId, ExprId)> {
    match node {
        ExprNode::Mul(a, b) => Some((*a, *b)),
        _ => None,
    }
}

fn is_max(node: &ExprNode) -> Option<(ExprId, ExprId)> {
    match node {
        ExprNode::Max(a, b) => Some((*a, *b)),
        _ => None,
    }
}

fn is_min(node: &ExprNode) -> Option<(ExprId, ExprId)> {
    match node {
        ExprNode::Min(a, b) => Some((*a, *b)),
        _ => None,
    }
}

/// Flatten a chain of the same associative op (`a op (b op c)` etc.) into
/// its leaves, so the SMT-LIB2 encoding is one n-ary `(+ ...)`/`(* ...)`
/// instead of a deeply nested binary tree - avoiding the O(n^2)
/// binary-fold encoding. Stops (and doesn't recurse further) as soon as a
/// leaf is already memoized, so a shared sub-chain reachable from two
/// different parents is only ever flattened once.
fn flatten(
    arena: &ExprArena,
    memo: &HashMap<ExprId, String>,
    id: ExprId,
    split: fn(&ExprNode) -> Option<(ExprId, ExprId)>,
    out: &mut Vec<ExprId>,
) {
    if memo.contains_key(&id) {
        out.push(id);
        return;
    }
    match split(arena.node(id)) {
        Some((a, b)) => {
            flatten(arena, memo, a, split, out);
            flatten(arena, memo, b, split, out);
        }
        None => out.push(id),
    }
}

fn paren_join(op: &str, parts: &[String]) -> String {
    if parts.len() == 1 {
        parts[0].clone()
    } else {
        format!("({} {})", op, parts.join(" "))
    }
}

fn translate_nary(
    bld: &mut Builder,
    memo: &mut HashMap<ExprId, String>,
    arena: &ExprArena,
    id: ExprId,
    op: &str,
    split: fn(&ExprNode) -> Option<(ExprId, ExprId)>,
) -> Result<String, Unsupported> {
    let mut leaves = Vec::new();
    flatten(arena, memo, id, split, &mut leaves);
    let parts = leaves
        .into_iter()
        .map(|leaf| translate(bld, memo, arena, leaf))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(bld.bind(paren_join(op, &parts)))
}

fn translate_maxmin(
    bld: &mut Builder,
    memo: &mut HashMap<ExprId, String>,
    arena: &ExprArena,
    id: ExprId,
    is_max_op: bool,
) -> Result<String, Unsupported> {
    let split = if is_max_op { is_max } else { is_min };
    let mut leaves = Vec::new();
    flatten(arena, memo, id, split, &mut leaves);
    let parts = leaves
        .into_iter()
        .map(|leaf| translate(bld, memo, arena, leaf))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(bld.maxmin_atom(is_max_op, parts))
}

/// Short, best-effort name for an unsupported node - the exact variant
/// name via `Debug`, not the full (potentially huge) subtree.
fn describe(node: &ExprNode) -> String {
    let s = format!("{:?}", node);
    s.split(|c: char| c == ' ' || c == '{' || c == '(')
        .next()
        .unwrap_or(&s)
        .to_string()
}

/// Translate one expression into an SMT-LIB2 term (a literal, a declared
/// symbol, or a `let`-bound variable for a compound expression), recording
/// any symbols/atoms/bindings it needs into `bld`. `memo` is a per-arena
/// cache (reference and optimized sides must use separate `memo`s - the
/// same `ExprId` value means different things in different arenas) - every
/// node is computed at most once no matter how many parents reference it.
fn translate(
    bld: &mut Builder,
    memo: &mut HashMap<ExprId, String>,
    arena: &ExprArena,
    id: ExprId,
) -> Result<String, Unsupported> {
    if let Some(cached) = memo.get(&id) {
        return Ok(cached.clone());
    }
    let result = translate_uncached(bld, memo, arena, id)?;
    memo.insert(id, result.clone());
    Ok(result)
}

fn translate_uncached(
    bld: &mut Builder,
    memo: &mut HashMap<ExprId, String>,
    arena: &ExprArena,
    id: ExprId,
) -> Result<String, Unsupported> {
    match arena.node(id) {
        ExprNode::IntConst(v) => Ok(int_literal(*v)),
        ExprNode::FloatConst(v) => real_literal(*v),
        ExprNode::BoolConst(v) => Ok(if *v { "1.0".to_string() } else { "0.0".to_string() }),

        ExprNode::NamedSymbol(sid) => Ok(bld.declare_real(quote(arena.string(*sid)))),
        ExprNode::Symbol(sym) => Ok(bld.declare_real(quote(&format!("s{}", sym.0)))),

        ExprNode::Add(..) => translate_nary(bld, memo, arena, id, "+", is_add),
        ExprNode::Mul(..) => translate_nary(bld, memo, arena, id, "*", is_mul),
        ExprNode::Sub(a, b) => {
            let ta = translate(bld, memo, arena, *a)?;
            let tb = translate(bld, memo, arena, *b)?;
            Ok(bld.bind(format!("(- {} {})", ta, tb)))
        }
        ExprNode::Div(a, b) => {
            let ta = translate(bld, memo, arena, *a)?;
            let tb = translate(bld, memo, arena, *b)?;
            Ok(bld.bind(format!("(/ {} {})", ta, tb)))
        }
        ExprNode::Neg(a) => {
            let ta = translate(bld, memo, arena, *a)?;
            Ok(bld.bind(format!("(- {})", ta)))
        }
        ExprNode::Rcp(a) => {
            let ta = translate(bld, memo, arena, *a)?;
            Ok(bld.bind(format!("(/ 1.0 {})", ta)))
        }
        ExprNode::Fma(a, b, c) => {
            let ta = translate(bld, memo, arena, *a)?;
            let tb = translate(bld, memo, arena, *b)?;
            let tc = translate(bld, memo, arena, *c)?;
            Ok(bld.bind(format!("(+ (* {} {}) {})", ta, tb, tc)))
        }
        ExprNode::Exp(a) => {
            let ta = translate(bld, memo, arena, *a)?;
            Ok(bld.bind(format!("(^ e {})", ta)))
        }

        ExprNode::Max(..) => translate_maxmin(bld, memo, arena, id, true),
        ExprNode::Min(..) => translate_maxmin(bld, memo, arena, id, false),

        // Conversions: identity over the reals (matches
        // `canon::canonicalize`'s treatment of the same nodes exactly).
        // Not let-bound themselves - they forward straight to the already
        // memoized/bound child.
        ExprNode::ToFloat(a)
        | ExprNode::SignExtend { value: a, .. }
        | ExprNode::ZeroExtend { value: a, .. }
        | ExprNode::Truncate { value: a, .. } => translate(bld, memo, arena, *a),

        other => Err(Unsupported(describe(other))),
    }
}

/// Translate a whole root expression, starting a fresh per-arena memo.
/// Use one call per side (reference/optimized) sharing the same `Builder`.
pub fn translate_root(bld: &mut Builder, arena: &ExprArena, id: ExprId) -> Result<String, Unsupported> {
    let mut memo = HashMap::new();
    translate(bld, &mut memo, arena, id)
}
