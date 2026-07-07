//! Equivalence checking of symbolic expressions.
//!
//! A thin wrapper over the `canon` decision procedure (see
//! `docs/canonicalizer-plan.md`). Use `EquivSession` when checking many
//! elements of the same kernel pair: the session's intern tables and memos
//! make shared structure (score polynomials, softmax denominators, the
//! other 63 columns of the row) free after the first element.

use crate::canon::{CanonError, Session};
use crate::logging::info;
use crate::symbolic::{ExprArena, ExprId};

/// Errors from equivalence checking.
#[derive(Debug)]
pub enum EquivError {
    /// The decision procedure failed (undefined value, coefficient
    /// overflow, or an oversized VC).
    Canon(CanonError),
}

impl From<CanonError> for EquivError {
    fn from(e: CanonError) -> Self {
        Self::Canon(e)
    }
}

impl std::fmt::Display for EquivError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Canon(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for EquivError {}

pub type EquivResult<T> = Result<T, EquivError>;

/// A reusable equivalence-checking session for one kernel pair.
///
/// Sessions self-limit their memory: when the intern tables grow past a
/// bound (chain-heavy VCs intern millions of intermediate terms per
/// element), the tables are dropped and rebuilt lazily from the still-live
/// `ExprArena`s. This trades some re-canonicalization of shared structure
/// for a hard cap on resident memory.
pub struct EquivSession {
    session: Session,
    recycle_terms: usize,
}

/// Default recycle bound. Bytes per term are workload-dependent: polynomial
/// terms (matmul) run a few hundred bytes, exp-heavy attention terms average
/// 2-4 KB, so 4M terms retains roughly 1-16 GiB.
pub const DEFAULT_RECYCLE_TERMS: usize = 4_000_000;

impl EquivSession {
    pub fn new() -> Self {
        Self::with_recycle_terms(DEFAULT_RECYCLE_TERMS)
    }

    /// A session that recycles its intern tables once they exceed
    /// `recycle_terms` interned terms (`0` = never recycle). Lower values
    /// bound resident memory; each recycle re-canonicalizes structure that
    /// later elements would otherwise share.
    pub fn with_recycle_terms(recycle_terms: usize) -> Self {
        Self {
            session: Session::new(),
            recycle_terms,
        }
    }

    /// Check whether two expressions are equivalent over the reals.
    pub fn check(
        &mut self,
        arena1: &ExprArena,
        e1: ExprId,
        arena2: &ExprArena,
        e2: ExprId,
    ) -> EquivResult<bool> {
        if self.recycle_terms != 0 && self.session.interned_terms() > self.recycle_terms {
            info!(
                "recycling VC session at {} interned terms",
                self.session.interned_terms()
            );
            self.session = Session::new();
        }
        Ok(self.session.check_equivalent(arena1, e1, arena2, e2)?)
    }
}

impl Default for EquivSession {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot equivalence check (a fresh session per call; prefer
/// `EquivSession` in loops).
pub fn check_equivalent(
    arena1: &ExprArena,
    e1: ExprId,
    arena2: &ExprArena,
    e2: ExprId,
) -> EquivResult<bool> {
    EquivSession::new().check(arena1, e1, arena2, e2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_equivalence() {
        // (a + b) == (b + a)
        let mut arena = ExprArena::new();
        let a = arena.named("a");
        let b = arena.named("b");
        let e1 = arena.add(a, b);
        let e2 = arena.add(b, a);
        assert!(check_equivalent(&arena, e1, &arena, e2).unwrap());
    }

    #[test]
    fn test_distributivity() {
        // a * (b + c) == a*b + a*c
        let mut arena = ExprArena::new();
        let a = arena.named("a");
        let b = arena.named("b");
        let c = arena.named("c");
        let bc = arena.add(b, c);
        let e1 = arena.mul(a, bc);
        let ab = arena.mul(a, b);
        let ac = arena.mul(a, c);
        let e2 = arena.add(ab, ac);
        assert!(check_equivalent(&arena, e1, &arena, e2).unwrap());
    }

    #[test]
    fn test_not_equivalent() {
        let mut arena = ExprArena::new();
        let a = arena.named("a");
        let b = arena.named("b");
        assert!(!check_equivalent(&arena, a, &arena, b).unwrap());
    }

    #[test]
    fn test_reduction_pattern() {
        let mut arena = ExprArena::new();
        let i0 = arena.named("input_0");
        let i1 = arena.named("input_1");
        let i2 = arena.named("input_2");
        let i3 = arena.named("input_3");

        let t1 = arena.add(i3, i2);
        let t2 = arena.add(i1, i0);
        let e1 = arena.add(t1, t2);

        let t3 = arena.add(i3, i1);
        let t4 = arena.add(i2, i0);
        let e2 = arena.add(t3, t4);
        assert!(check_equivalent(&arena, e1, &arena, e2).unwrap());
    }

    #[test]
    fn test_exp_identity() {
        // exp(a) * exp(b) == exp(a + b)
        let mut arena = ExprArena::new();
        let a = arena.named("a");
        let b = arena.named("b");
        let ea = arena.exp(a);
        let eb = arena.exp(b);
        let e1 = arena.mul(ea, eb);
        let ab = arena.add(a, b);
        let e2 = arena.exp(ab);
        assert!(check_equivalent(&arena, e1, &arena, e2).unwrap());
    }

    #[test]
    fn test_fma_expansion() {
        // fma(a, b, c) == a*b + c
        let mut arena = ExprArena::new();
        let a = arena.named("a");
        let b = arena.named("b");
        let c = arena.named("c");
        let e1 = arena.fma(a, b, c);
        let ab = arena.mul(a, b);
        let e2 = arena.add(ab, c);
        assert!(check_equivalent(&arena, e1, &arena, e2).unwrap());
    }

    #[test]
    fn test_softmax_normalization_equivalence() {
        // exp(a)/(exp(a)+exp(b)) == exp(a-M)/(exp(a-M)+exp(b-M)), M = max(a,b)
        let mut arena = ExprArena::new();
        let a = arena.named("a");
        let b = arena.named("b");
        let m = arena.max(a, b);

        let ea = arena.exp(a);
        let eb = arena.exp(b);
        let d1 = arena.add(ea, eb);
        let e1 = arena.div(ea, d1);

        let am = arena.sub(a, m);
        let bm = arena.sub(b, m);
        let eam = arena.exp(am);
        let ebm = arena.exp(bm);
        let d2 = arena.add(eam, ebm);
        let e2 = arena.div(eam, d2);

        assert!(check_equivalent(&arena, e1, &arena, e2).unwrap());
    }

    #[test]
    fn test_session_reuse_across_elements() {
        // The same session checks several related identities.
        let mut arena = ExprArena::new();
        let mut session = EquivSession::new();
        for i in 0..4 {
            let a = arena.named(format!("a[{}]", i));
            let b = arena.named(format!("b[{}]", i));
            let ab = arena.add(a, b);
            let ba = arena.add(b, a);
            assert!(session.check(&arena, ab, &arena, ba).unwrap());
        }
    }

    #[test]
    fn test_undefined_error() {
        let mut arena = ExprArena::new();
        let a = arena.named("a");
        let u = arena.undefined();
        assert!(check_equivalent(&arena, a, &arena, u).is_err());
    }
}
