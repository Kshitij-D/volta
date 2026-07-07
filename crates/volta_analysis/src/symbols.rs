//! Symbol table for PTX lowering
//!
//! This module provides the symbol table that tracks register declarations,
//! labels, parameters, and shared memory allocations during lowering.

use std::collections::HashMap;
use std::fmt;

use id_collections::{IdVec, id_type};
use volta_frontend::ast::ScalarType;

use crate::lower_error::{LowerError, LowerResult};
use crate::lowered::InstrId;
use crate::types::{RegClass, RegCounts};

/// The kind of symbol in the PTX namespace
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Register,
    Parameter,
    Label,
    SharedVariable,
    LocalVariable,
    GlobalVariable,
}

impl fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Register => write!(f, "register"),
            Self::Parameter => write!(f, "parameter"),
            Self::Label => write!(f, "label"),
            Self::SharedVariable => write!(f, "shared variable"),
            Self::LocalVariable => write!(f, "local variable"),
            Self::GlobalVariable => write!(f, "global variable"),
        }
    }
}

/// Index of a register within its class
#[id_type]
pub struct RegIndex(pub u32);

impl fmt::Display for RegIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A resolved register reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegId {
    pub class: RegClass,
    pub index: RegIndex,
}

impl RegId {
    /// Create a new RegId
    pub fn new(class: RegClass, index: u32) -> Self {
        Self {
            class,
            index: RegIndex(index),
        }
    }
}

impl fmt::Display for RegId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}[{}]", self.class, self.index)
    }
}

/// Parameter index
#[id_type]
pub struct ParamId(pub u32);

impl fmt::Display for ParamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "param:{}", self.0)
    }
}

/// Information about a declared register
#[derive(Debug, Clone)]
pub struct RegInfo {
    pub id: RegId,
    pub name: String,
    pub declared_type: ScalarType,
}

/// Special register kinds (thread-dependent values known at runtime)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialRegKind {
    // Thread identification
    TidX,
    TidY,
    TidZ,
    NtidX,
    NtidY,
    NtidZ,
    CtaidX,
    CtaidY,
    CtaidZ,
    NctaidX,
    NctaidY,
    NctaidZ,
    // Warp-level
    LaneId,
    WarpId,
    NWarpId,
    LanemaskEq,
    LanemaskLt,
    LanemaskLe,
    LanemaskGt,
    LanemaskGe,
    // SM identification
    SmId,
    NSmId,
    // Grid
    GridId,
    // Clock
    Clock,
    Clock64,
    // Dynamic shared memory
    DynamicSmemSize,
}

impl SpecialRegKind {
    /// Get the type of this special register
    pub fn ty(&self) -> ScalarType {
        match self {
            // 32-bit unsigned
            Self::TidX
            | Self::TidY
            | Self::TidZ
            | Self::NtidX
            | Self::NtidY
            | Self::NtidZ
            | Self::CtaidX
            | Self::CtaidY
            | Self::CtaidZ
            | Self::NctaidX
            | Self::NctaidY
            | Self::NctaidZ
            | Self::LaneId
            | Self::WarpId
            | Self::NWarpId
            | Self::LanemaskEq
            | Self::LanemaskLt
            | Self::LanemaskLe
            | Self::LanemaskGt
            | Self::LanemaskGe
            | Self::Clock
            | Self::SmId
            | Self::NSmId
            | Self::GridId
            | Self::DynamicSmemSize => ScalarType::U32,
            // 64-bit unsigned
            Self::Clock64 => ScalarType::U64,
        }
    }

    /// Parse from PTX name (with or without % prefix)
    pub fn from_name(name: &str) -> Option<Self> {
        let name = name.strip_prefix('%').unwrap_or(name);
        Some(match name {
            "tid.x" => Self::TidX,
            "tid.y" => Self::TidY,
            "tid.z" => Self::TidZ,
            "ntid.x" => Self::NtidX,
            "ntid.y" => Self::NtidY,
            "ntid.z" => Self::NtidZ,
            "ctaid.x" => Self::CtaidX,
            "ctaid.y" => Self::CtaidY,
            "ctaid.z" => Self::CtaidZ,
            "nctaid.x" => Self::NctaidX,
            "nctaid.y" => Self::NctaidY,
            "nctaid.z" => Self::NctaidZ,
            "laneid" => Self::LaneId,
            "warpid" => Self::WarpId,
            "nwarpid" => Self::NWarpId,
            "lanemask_eq" => Self::LanemaskEq,
            "lanemask_lt" => Self::LanemaskLt,
            "lanemask_le" => Self::LanemaskLe,
            "lanemask_gt" => Self::LanemaskGt,
            "lanemask_ge" => Self::LanemaskGe,
            "clock" => Self::Clock,
            "clock64" => Self::Clock64,
            "smid" => Self::SmId,
            "nsmid" => Self::NSmId,
            "gridid" => Self::GridId,
            "dynamic_smem_size" => Self::DynamicSmemSize,
            _ => return None,
        })
    }

    /// Get the PTX name for this special register
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TidX => "%tid.x",
            Self::TidY => "%tid.y",
            Self::TidZ => "%tid.z",
            Self::NtidX => "%ntid.x",
            Self::NtidY => "%ntid.y",
            Self::NtidZ => "%ntid.z",
            Self::CtaidX => "%ctaid.x",
            Self::CtaidY => "%ctaid.y",
            Self::CtaidZ => "%ctaid.z",
            Self::NctaidX => "%nctaid.x",
            Self::NctaidY => "%nctaid.y",
            Self::NctaidZ => "%nctaid.z",
            Self::LaneId => "%laneid",
            Self::WarpId => "%warpid",
            Self::NWarpId => "%nwarpid",
            Self::LanemaskEq => "%lanemask_eq",
            Self::LanemaskLt => "%lanemask_lt",
            Self::LanemaskLe => "%lanemask_le",
            Self::LanemaskGt => "%lanemask_gt",
            Self::LanemaskGe => "%lanemask_ge",
            Self::Clock => "%clock",
            Self::Clock64 => "%clock64",
            Self::SmId => "%smid",
            Self::NSmId => "%nsmid",
            Self::GridId => "%gridid",
            Self::DynamicSmemSize => "%dynamic_smem_size",
        }
    }
}

/// Information about a kernel parameter
#[derive(Debug, Clone)]
pub struct ParamInfo {
    pub id: ParamId,
    pub name: String,
    pub ty: ScalarType,
    pub size_bytes: u64,
}

/// Information about a shared memory allocation
#[derive(Debug, Clone)]
pub struct SharedMemInfo {
    pub name: String,
    pub offset: u64,
    pub size_bytes: u64,
    pub element_ty: ScalarType,
    pub is_extern: bool,
}

/// Information about a function-scope local memory allocation (e.g. the
/// `__local_depot` stack array nvcc emits for spilled locals).
#[derive(Debug, Clone)]
pub struct LocalMemInfo {
    pub name: String,
    pub offset: u64,
    pub size_bytes: u64,
    pub element_ty: ScalarType,
}

/// Information about a module-scope `.global` variable (e.g. values set by the
/// host via `cudaMemcpyToSymbol`). Assigned addresses in a reserved region so
/// they cannot collide with caller-chosen input array addresses.
#[derive(Debug, Clone)]
pub struct GlobalVarInfo {
    pub name: String,
    pub addr: u64,
    pub size_bytes: u64,
    pub element_ty: ScalarType,
}

/// Base address for module-scope `.global` variables. Analysis configs must
/// not place input/output arrays at or above this address.
pub const MODULE_GLOBAL_BASE: u64 = 0x7000_0000_0000_0000;

/// Symbol table built during lowering
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// All declared names mapped to their kind (for duplicate checking).
    /// PTX uses a single namespace for all identifiers within a function.
    all_names: HashMap<String, SymbolKind>,

    /// Register name → info
    registers: HashMap<String, RegInfo>,
    /// Count of registers per class
    reg_counts: [u32; RegClass::COUNT],

    /// Parameter name → info
    params: HashMap<String, ParamInfo>,
    /// Parameters in declaration order
    params_ordered: IdVec<ParamId, ParamInfo>,

    /// Label name → instruction PC
    labels: HashMap<String, InstrId>,

    /// Shared memory allocations
    shared_vars: HashMap<String, SharedMemInfo>,
    /// Total static shared memory size
    shared_mem_size: u64,
    /// Has extern shared memory
    has_extern_shared: bool,

    /// Function-scope local memory allocations (per-thread space)
    local_vars: HashMap<String, LocalMemInfo>,
    /// Total local memory size per thread
    local_mem_size: u64,

    /// Module-scope `.global` variables
    global_vars: HashMap<String, GlobalVarInfo>,
    /// Bytes of the reserved module-global region allocated so far
    module_global_size: u64,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            all_names: HashMap::new(),
            registers: HashMap::new(),
            reg_counts: [0; RegClass::COUNT],
            params: HashMap::new(),
            params_ordered: IdVec::new(),
            labels: HashMap::new(),
            shared_vars: HashMap::new(),
            shared_mem_size: 0,
            has_extern_shared: false,
            local_vars: HashMap::new(),
            local_mem_size: 0,
            global_vars: HashMap::new(),
            module_global_size: 0,
        }
    }

    /// Reserve a name in the global namespace.
    /// Returns an error if the name is already in use.
    fn reserve_name(&mut self, name: &str, kind: SymbolKind) -> LowerResult<()> {
        if let Some(&existing) = self.all_names.get(name) {
            return Err(LowerError::DuplicateName {
                name: name.to_string(),
                existing,
                attempted: kind,
            });
        }
        self.all_names.insert(name.to_string(), kind);
        Ok(())
    }

    /// Declare a register or range of registers
    ///
    /// For parameterized registers like `%r<100>`, this declares %r0 through %r99.
    /// Returns the RegId of the first register declared.
    pub fn declare_register(
        &mut self,
        base_name: &str,
        ty: ScalarType,
        count: u32,
    ) -> Result<RegId, LowerError> {
        let class = RegClass::from_scalar_type(ty);

        // PTX scopes declarations to blocks, but we flatten them. nvcc
        // re-declares helper registers like `temp_param_reg` in every callseq
        // block, so an identical single-register re-declaration is idempotent.
        if count == 1
            && let Some(existing) = self.registers.get(base_name)
            && existing.declared_type == ty
        {
            return Ok(existing.id);
        }

        let base_index = self.reg_counts[class as usize];

        // Parse the base name (strip angle brackets if present)
        let base = base_name.trim_end_matches(|c: char| c == '<' || c == '>' || c.is_ascii_digit());

        for i in 0..count {
            let name = if count == 1 {
                base_name.to_string()
            } else {
                format!("{}{}", base, i)
            };

            self.reserve_name(&name, SymbolKind::Register)?;

            let id = RegId::new(class, base_index + i);
            self.registers.insert(
                name.clone(),
                RegInfo {
                    id,
                    name,
                    declared_type: ty,
                },
            );
        }

        self.reg_counts[class as usize] += count;
        Ok(RegId::new(class, base_index))
    }

    /// Declare a kernel parameter
    /// Returns the ParamId of the declared parameter.
    pub fn declare_param(
        &mut self,
        name: &str,
        ty: ScalarType,
        size_bytes: u64,
    ) -> LowerResult<ParamId> {
        self.reserve_name(name, SymbolKind::Parameter)?;

        let id = ParamId(self.params_ordered.len() as u32);
        let info = ParamInfo {
            id,
            name: name.to_string(),
            ty,
            size_bytes,
        };
        self.params.insert(name.to_string(), info.clone());
        let _ = self.params_ordered.push(info);
        Ok(id)
    }

    /// Declare a label at the given PC
    pub fn declare_label(&mut self, name: &str, pc: InstrId) -> LowerResult<()> {
        self.reserve_name(name, SymbolKind::Label)?;
        self.labels.insert(name.to_string(), pc);
        Ok(())
    }

    /// Declare shared memory variable
    pub fn declare_shared(
        &mut self,
        name: &str,
        element_ty: ScalarType,
        num_elements: u64,
        is_extern: bool,
        alignment: u64,
    ) -> LowerResult<()> {
        self.reserve_name(name, SymbolKind::SharedVariable)?;

        // Align offset
        let elem_size = (element_ty.bits() as u64).div_ceil(8);
        let alignment = alignment.max(elem_size);
        let offset = (self.shared_mem_size + alignment - 1) & !(alignment - 1);

        let size_bytes = if is_extern {
            self.has_extern_shared = true;
            0 // Size determined at runtime
        } else {
            num_elements * elem_size
        };

        self.shared_vars.insert(
            name.to_string(),
            SharedMemInfo {
                name: name.to_string(),
                offset,
                size_bytes,
                element_ty,
                is_extern,
            },
        );

        self.shared_mem_size = offset + size_bytes;
        Ok(())
    }

    /// Declare a function-scope local memory variable (per-thread space)
    pub fn declare_local(
        &mut self,
        name: &str,
        element_ty: ScalarType,
        num_elements: u64,
        alignment: u64,
    ) -> LowerResult<()> {
        self.reserve_name(name, SymbolKind::LocalVariable)?;

        let elem_size = (element_ty.bits() as u64).div_ceil(8);
        let alignment = alignment.max(elem_size).max(1);
        let offset = (self.local_mem_size + alignment - 1) & !(alignment - 1);
        let size_bytes = num_elements * elem_size;

        self.local_vars.insert(
            name.to_string(),
            LocalMemInfo {
                name: name.to_string(),
                offset,
                size_bytes,
                element_ty,
            },
        );

        self.local_mem_size = offset + size_bytes;
        Ok(())
    }

    /// Declare a module-scope `.global` variable, assigning it an address in
    /// the reserved module-global region.
    pub fn declare_global_var(
        &mut self,
        name: &str,
        element_ty: ScalarType,
        num_elements: u64,
        alignment: u64,
    ) -> LowerResult<()> {
        self.reserve_name(name, SymbolKind::GlobalVariable)?;

        let elem_size = (element_ty.bits() as u64).div_ceil(8);
        let alignment = alignment.max(elem_size).max(1);
        let offset = (self.module_global_size + alignment - 1) & !(alignment - 1);
        let size_bytes = num_elements * elem_size;

        self.global_vars.insert(
            name.to_string(),
            GlobalVarInfo {
                name: name.to_string(),
                addr: MODULE_GLOBAL_BASE + offset,
                size_bytes,
                element_ty,
            },
        );

        self.module_global_size = offset + size_bytes;
        Ok(())
    }

    /// Look up a register, returning full info for type checking
    pub fn get_register(&self, name: &str) -> Option<&RegInfo> {
        self.registers.get(name)
    }

    /// Look up just the RegId
    pub fn resolve_register(&self, name: &str) -> Option<RegId> {
        self.registers.get(name).map(|info| info.id)
    }

    /// Check if name is a special register
    pub fn resolve_special_reg(&self, name: &str) -> Option<SpecialRegKind> {
        SpecialRegKind::from_name(name)
    }

    /// Look up a parameter
    pub fn get_param(&self, name: &str) -> Option<&ParamInfo> {
        self.params.get(name)
    }

    /// Look up a label
    pub fn resolve_label(&self, name: &str) -> Option<InstrId> {
        self.labels.get(name).copied()
    }

    /// Look up shared memory variable
    pub fn get_shared_var(&self, name: &str) -> Option<&SharedMemInfo> {
        self.shared_vars.get(name)
    }

    /// Iterate over all shared memory variables
    pub fn shared_vars(&self) -> impl Iterator<Item = &SharedMemInfo> {
        self.shared_vars.values()
    }

    /// Iterate over all function-scope local memory variables
    pub fn local_vars(&self) -> impl Iterator<Item = &LocalMemInfo> {
        self.local_vars.values()
    }

    /// Look up a function-scope local memory variable
    pub fn get_local_var(&self, name: &str) -> Option<&LocalMemInfo> {
        self.local_vars.get(name)
    }

    /// Total local memory size per thread
    pub fn local_mem_size(&self) -> u64 {
        self.local_mem_size
    }

    /// Look up a module-scope `.global` variable
    pub fn get_global_var(&self, name: &str) -> Option<&GlobalVarInfo> {
        self.global_vars.get(name)
    }

    /// Iterate over all module-scope `.global` variables
    pub fn global_vars(&self) -> impl Iterator<Item = &GlobalVarInfo> {
        self.global_vars.values()
    }

    /// Get register name from ID (for error messages)
    pub fn register_name(&self, id: RegId) -> Option<&str> {
        self.registers
            .values()
            .find(|info| info.id == id)
            .map(|info| info.name.as_str())
    }

    /// Get register count for a class
    pub fn register_count(&self, class: RegClass) -> u32 {
        self.reg_counts[class as usize]
    }

    /// Get all register counts
    pub fn register_counts(&self) -> RegCounts {
        RegCounts::new(self.reg_counts)
    }

    /// Get parameters in order
    pub fn params(&self) -> &IdVec<ParamId, ParamInfo> {
        &self.params_ordered
    }

    /// Get parameter by ID
    pub fn get_param_by_id(&self, id: ParamId) -> Option<&ParamInfo> {
        self.params_ordered.get(id)
    }

    /// Get total static shared memory size
    pub fn shared_mem_size(&self) -> u64 {
        self.shared_mem_size
    }

    /// Check if kernel uses dynamic (extern) shared memory
    pub fn has_extern_shared(&self) -> bool {
        self.has_extern_shared
    }

    /// Find similar register names for error suggestions
    pub fn find_similar_registers(&self, name: &str) -> Vec<String> {
        self.registers
            .keys()
            .filter(|k| levenshtein_distance(k, name) <= 2)
            .take(3)
            .cloned()
            .collect()
    }

    /// Get all register names (for debugging)
    pub fn all_register_names(&self) -> impl Iterator<Item = &str> {
        self.registers.keys().map(|s| s.as_str())
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple Levenshtein distance for typo suggestions
fn levenshtein_distance(a: &str, b: &str) -> usize {
    if a == b {
        return 0;
    }
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }

    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let mut prev: Vec<usize> = (0..=b_chars.len()).collect();
    let mut curr = vec![0; b_chars.len() + 1];

    for (i, ca) in a_chars.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b_chars.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[b_chars.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_declaration() {
        let mut symbols = SymbolTable::new();
        symbols.declare_register("%r", ScalarType::U32, 10).unwrap();

        // Should have %r0 through %r9
        assert!(symbols.get_register("%r0").is_some());
        assert!(symbols.get_register("%r9").is_some());
        assert!(symbols.get_register("%r10").is_none());

        // All should be U32
        assert_eq!(
            symbols.get_register("%r5").unwrap().declared_type,
            ScalarType::U32
        );
    }

    #[test]
    fn test_single_register() {
        let mut symbols = SymbolTable::new();
        symbols
            .declare_register("%myReg", ScalarType::F32, 1)
            .unwrap();

        assert!(symbols.get_register("%myReg").is_some());
        assert_eq!(
            symbols.get_register("%myReg").unwrap().declared_type,
            ScalarType::F32
        );
    }

    #[test]
    fn test_duplicate_register_error() {
        let mut symbols = SymbolTable::new();
        symbols.declare_register("%r", ScalarType::U32, 5).unwrap();

        // Try to declare overlapping range
        let result = symbols.declare_register("%r", ScalarType::U32, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_special_registers() {
        assert_eq!(
            SpecialRegKind::from_name("%tid.x"),
            Some(SpecialRegKind::TidX)
        );
        assert_eq!(
            SpecialRegKind::from_name("tid.x"),
            Some(SpecialRegKind::TidX)
        );
        assert_eq!(
            SpecialRegKind::from_name("%laneid"),
            Some(SpecialRegKind::LaneId)
        );
        assert_eq!(SpecialRegKind::from_name("%unknown"), None);
    }

    #[test]
    fn test_labels() {
        use id_collections::Id;
        let mut symbols = SymbolTable::new();
        symbols
            .declare_label("LOOP", InstrId::from_index(10))
            .unwrap();
        symbols
            .declare_label("END", InstrId::from_index(50))
            .unwrap();

        assert_eq!(symbols.resolve_label("LOOP"), Some(InstrId::from_index(10)));
        assert_eq!(symbols.resolve_label("END"), Some(InstrId::from_index(50)));
        assert_eq!(symbols.resolve_label("MISSING"), None);
    }

    #[test]
    fn test_duplicate_label_error() {
        use id_collections::Id;
        let mut symbols = SymbolTable::new();
        symbols
            .declare_label("LOOP", InstrId::from_index(10))
            .unwrap();

        // Try to declare same label again
        let result = symbols.declare_label("LOOP", InstrId::from_index(20));
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_param_error() {
        let mut symbols = SymbolTable::new();
        symbols.declare_param("ptr", ScalarType::U64, 8).unwrap();

        // Try to declare same param again
        let result = symbols.declare_param("ptr", ScalarType::U64, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_shared_error() {
        let mut symbols = SymbolTable::new();
        symbols
            .declare_shared("smem", ScalarType::F32, 256, false, 4)
            .unwrap();

        // Try to declare same shared var again
        let result = symbols.declare_shared("smem", ScalarType::F32, 128, false, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_levenshtein() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        assert_eq!(levenshtein_distance("hello", "hallo"), 1);
        assert_eq!(levenshtein_distance("hello", ""), 5);
        assert_eq!(levenshtein_distance("", "hello"), 5);
    }

    // =========================================================================
    // Cross-namespace conflict tests (PTX uses a single namespace)
    // =========================================================================

    #[test]
    fn test_label_conflicts_with_param() {
        use id_collections::Id;
        let mut symbols = SymbolTable::new();
        symbols.declare_param("foo", ScalarType::U64, 8).unwrap();

        // Label should conflict with param
        let result = symbols.declare_label("foo", InstrId::from_index(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_label_conflicts_with_register() {
        use id_collections::Id;
        let mut symbols = SymbolTable::new();
        symbols.declare_register("foo", ScalarType::U32, 1).unwrap();

        // Label should conflict with register
        let result = symbols.declare_label("foo", InstrId::from_index(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_label_conflicts_with_shared() {
        use id_collections::Id;
        let mut symbols = SymbolTable::new();
        symbols
            .declare_shared("foo", ScalarType::U32, 256, false, 4)
            .unwrap();

        // Label should conflict with shared
        let result = symbols.declare_label("foo", InstrId::from_index(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_param_conflicts_with_register() {
        let mut symbols = SymbolTable::new();
        symbols.declare_register("foo", ScalarType::U32, 1).unwrap();

        // Param should conflict with register
        let result = symbols.declare_param("foo", ScalarType::U64, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_shared_conflicts_with_param() {
        let mut symbols = SymbolTable::new();
        symbols.declare_param("foo", ScalarType::U64, 8).unwrap();

        // Shared should conflict with param
        let result = symbols.declare_shared("foo", ScalarType::U32, 256, false, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_register_conflicts_with_label() {
        use id_collections::Id;
        let mut symbols = SymbolTable::new();
        symbols
            .declare_label("foo", InstrId::from_index(0))
            .unwrap();

        // Register should conflict with label
        let result = symbols.declare_register("foo", ScalarType::U32, 1);
        assert!(result.is_err());
    }
}
