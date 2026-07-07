//! Runtime values and per-thread register files.

use crate::symbolic::ExprId;
use crate::symbols::RegId;
use crate::types::{RegClass, RegCounts};

/// A value held in a register or a memory granule.
///
/// `Pair` models a packed pair of 16-bit halves living in one 32-bit
/// register/word — the representation nvcc uses for f16 data (loaded from
/// global as `u32`, distributed by `ldmatrix`, consumed by `mma`). We track
/// the two halves as separate real-valued expressions and never bit-encode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    /// A single scalar expression
    Scalar(ExprId),
    /// Two packed 16-bit halves: (lo, hi)
    Pair(ExprId, ExprId),
}

impl Value {
    /// The scalar expression, if this is a scalar.
    pub fn as_scalar(self) -> Option<ExprId> {
        match self {
            Self::Scalar(e) => Some(e),
            Self::Pair(_, _) => None,
        }
    }
}

/// A per-thread register file, indexed by (class, index).
///
/// Registers start uninitialized; reading one before writing it is an
/// analysis error (surfaced by the interpreter, which knows the pc).
#[derive(Debug, Clone)]
pub struct RegFile {
    classes: [Vec<Option<Value>>; RegClass::COUNT],
}

impl RegFile {
    pub fn new(counts: &RegCounts) -> Self {
        let class_vec = |class: RegClass| vec![None; counts.get(class) as usize];
        Self {
            classes: [
                class_vec(RegClass::Pred),
                class_vec(RegClass::Bits8),
                class_vec(RegClass::Bits16),
                class_vec(RegClass::Bits32),
                class_vec(RegClass::Bits64),
                class_vec(RegClass::Bits128),
            ],
        }
    }

    pub fn read(&self, reg: RegId) -> Option<Value> {
        self.classes[reg.class as usize][reg.index.0 as usize]
    }

    pub fn write(&mut self, reg: RegId, value: Value) {
        self.classes[reg.class as usize][reg.index.0 as usize] = Some(value);
    }
}
