//! Analysis configuration: launch dimensions, parameter values, and the
//! input/output arrays that define the kernel's memory interface.

/// Value bound to a kernel parameter, positional by `ParamId` order.
#[derive(Debug, Clone)]
pub enum ParamValue {
    /// Concrete integer (also used for raw pointer values)
    Int(i64),
    /// Concrete float
    Float(f64),
    /// Symbolic float input with the given name (e.g. "alpha")
    SymFloat(String),
    /// Pointer to a named array from `AnalysisConfig::arrays`
    ArrayPtr(String),
}

/// Whether an array is a kernel input (pre-initialized with fresh symbols),
/// an output (extracted after execution), or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayKind {
    Input,
    Output,
    InputOutput,
    /// An input of concrete integers where element i holds the value i
    /// (identity mapping). For index arrays that feed addressing, e.g.
    /// OpenMM's `posq[particles[i]]`: symbolic elements would make the
    /// derived addresses symbolic and violate structured-CTA.
    IndexInput,
}

impl ArrayKind {
    pub fn is_input(self) -> bool {
        matches!(self, Self::Input | Self::InputOutput | Self::IndexInput)
    }

    pub fn is_output(self) -> bool {
        matches!(self, Self::Output | Self::InputOutput)
    }
}

/// A global-memory array visible to the kernel.
///
/// Input arrays are pre-populated with named symbols `name[0]`, `name[1]`,
/// ... at granule width `elem_width`. Bases must not fall in the reserved
/// module-global region (see `symbols::MODULE_GLOBAL_BASE`).
#[derive(Debug, Clone)]
pub struct ArrayDef {
    pub name: String,
    pub base: u64,
    /// Element width in bytes (2 for f16, 4 for f32/int, 8 for f64)
    pub elem_width: u64,
    /// Number of elements
    pub len: u64,
    pub kind: ArrayKind,
}

impl ArrayDef {
    pub fn size_bytes(&self) -> u64 {
        self.elem_width * self.len
    }
}

/// Full configuration for analyzing one kernel.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Threads per block; the CTA under analysis is block (0,0,0)
    pub block_dim: (u32, u32, u32),
    /// Grid dimensions (only used for `%nctaid`)
    pub grid_dim: (u32, u32, u32),
    /// Parameter values in `ParamId` (declaration) order
    pub params: Vec<ParamValue>,
    /// Global-memory arrays
    pub arrays: Vec<ArrayDef>,
    /// Concrete values for module-scope `.global` variables, by PTX name
    pub global_values: Vec<(String, i64)>,
    /// Size of dynamic (extern) shared memory in bytes
    pub dynamic_shared_bytes: u64,
    /// Abort analysis after this many executed instructions
    pub max_instructions: u64,
}

impl AnalysisConfig {
    pub fn new(block_dim: (u32, u32, u32)) -> Self {
        Self {
            block_dim,
            grid_dim: (1, 1, 1),
            params: Vec::new(),
            arrays: Vec::new(),
            global_values: Vec::new(),
            dynamic_shared_bytes: 0,
            max_instructions: 2_000_000_000,
        }
    }

    pub fn num_threads(&self) -> u32 {
        self.block_dim.0 * self.block_dim.1 * self.block_dim.2
    }

    pub fn array(&self, name: &str) -> Option<&ArrayDef> {
        self.arrays.iter().find(|a| a.name == name)
    }
}
