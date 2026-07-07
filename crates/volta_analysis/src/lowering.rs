//! Lowering pass: AST → LoweredProgram
//!
//! This module implements the lowering pass that transforms the parsed AST into
//! a form suitable for abstract interpretation. The lowering:
//!
//! 1. Collects register declarations and builds a symbol table
//! 2. Resolves register references to indices
//! 3. Resolves labels to instruction PCs
//! 4. Performs type checking on operands
//! 5. Converts complex instruction variants to a uniform representation

use std::collections::HashMap;

use volta_common::Span;
use volta_frontend::ascii::AsciiSliceExt;
use volta_frontend::ast::{
    self, AbsInstr, AddInstr, Address, AddressBase, BarInstr, BarMode, BraInstr, CallInstr,
    CmpOp as AstCmpOp, CvtInstr, DivInstr, FmaInstr, FromAscii, Function, FunctionBody,
    Instruction, InstructionOp, LdInstr, MadInstr, MaxInstr, MinInstr, MulInstr, MulMode, NegInstr,
    Operand as AstOperand, ParsedInstruction, ScalarType, SetpInstr, ShflMode as AstShflMode,
    ShflSyncInstr, StInstr, StateSpace, Statement, SubInstr, VarDecl,
};
use volta_frontend::instr::InstrKind;
use volta_frontend::instr_parse::parse_instruction;
use volta_frontend::lex::DottedIdent;

use id_collections::{Id, IdVec};

use crate::lower_error::{LowerError, LowerResult};
use crate::lowered::{
    BinOp, CmpOp, InstrId, LoweredInstr, LoweredProgram, MemSpace, MembarScope,
    MulMode as LoweredMulMode, Operand, Predicate, ShflMode, UnaryOp,
};
use crate::source_map::SourceMapBuilder;
use crate::symbols::{RegId, SpecialRegKind, SymbolTable};
use crate::tensor_core::{MmaLayout, MmaOperand, MmaShape};
use crate::types::{ScalarTypeExt, TypeCompatibility, check_type_compatibility};

// =============================================================================
// Type-Aware Operand Resolution
// =============================================================================

/// An operand resolved with its type information (for type checking)
#[derive(Debug, Clone)]
pub struct ResolvedOperand {
    /// The lowered operand
    pub operand: Operand,
    /// The declared type of the operand (None for immediates, which are polymorphic)
    pub ty: Option<ScalarType>,
    /// The name of the operand (for error messages)
    pub name: Option<String>,
}

impl ResolvedOperand {
    /// Create a resolved operand for a register
    fn register(operand: Operand, ty: ScalarType, name: String) -> Self {
        Self {
            operand,
            ty: Some(ty),
            name: Some(name),
        }
    }

    /// Create a resolved operand for an immediate (type is polymorphic)
    fn immediate(operand: Operand) -> Self {
        Self {
            operand,
            ty: None,
            name: None,
        }
    }

    /// Create a resolved operand for a special register
    fn special_reg(operand: Operand, ty: ScalarType, name: String) -> Self {
        Self {
            operand,
            ty: Some(ty),
            name: Some(name),
        }
    }
}

/// Resolved destination register with type information
#[derive(Debug, Clone)]
pub struct ResolvedDst {
    /// The register ID
    pub reg: RegId,
    /// The declared type
    pub ty: ScalarType,
    /// The register name (for error messages)
    pub name: String,
}

/// State of a block-scope `.param` variable used by the nvcc callseq idiom:
///
/// ```text
/// .param .b32 param0;
/// st.param.f32 [param0+0], %f1;      // -> Stored(%f1)
/// .param .b32 retval0;
/// call.uni (retval0), __symexpf, (param0);   // -> retval0 = PendingExp(%f1)
/// ld.param.f32 %f2, [retval0+0];     // -> emit %f2 = exp(%f1)
/// ```
///
/// No instructions are emitted for the `.param` traffic itself; the call
/// collapses to a single `UnaryOp::Exp` emitted at the consuming `ld.param`.
#[derive(Debug, Clone, Copy)]
enum LocalParamSlot {
    /// Declared but not yet written
    Empty,
    /// Holds the operand last stored via `st.param`
    Stored(Operand),
    /// Holds the pending result of `__symexpf` applied to the operand
    PendingExp(Operand),
}

/// Context for the lowering pass
pub struct LoweringContext {
    /// Symbol table being built
    symbols: SymbolTable,
    /// Source map builder for tracking spans
    source_map_builder: SourceMapBuilder,
    /// Lowered instructions
    instructions: Vec<LoweredInstr>,
    /// Predicates for each instruction
    predicates: Vec<Option<Predicate>>,
    /// Pending labels (label name → will point to next instruction)
    pending_labels: Vec<String>,
    /// Pending label spans (to be associated with the next instruction)
    pending_label_spans: Vec<Span>,
    /// Forward references to resolve (PC, label name)
    forward_refs: Vec<(InstrId, String)>,
    /// Current instruction span (set before lowering each instruction)
    current_span: Option<Span>,
    /// Block-scope `.param` variables (callseq idiom); flat because the
    /// blocks re-declare them before each use
    local_params: HashMap<String, LocalParamSlot>,
}

impl LoweringContext {
    pub fn new() -> Self {
        Self {
            symbols: SymbolTable::new(),
            source_map_builder: SourceMapBuilder::new(),
            instructions: Vec::new(),
            predicates: Vec::new(),
            pending_labels: Vec::new(),
            pending_label_spans: Vec::new(),
            forward_refs: Vec::new(),
            current_span: None,
            local_params: HashMap::new(),
        }
    }

    /// Get current PC (next instruction index)
    fn current_pc(&self) -> InstrId {
        InstrId::from_index(self.instructions.len() as u32)
    }

    /// Emit an instruction
    fn emit(&mut self, instr: LoweredInstr, predicate: Option<Predicate>) -> LowerResult<()> {
        // Resolve pending labels to this PC
        let pc = self.current_pc();
        for label in self.pending_labels.drain(..) {
            self.symbols.declare_label(&label, pc)?;
        }

        // Record instruction span
        self.source_map_builder
            .record_instruction(pc, self.current_span);

        // Record any pending label spans
        for label_span in self.pending_label_spans.drain(..) {
            self.source_map_builder.record_pending_label(label_span);
        }

        self.instructions.push(instr);
        self.predicates.push(predicate);
        Ok(())
    }

    /// Record a label to be resolved to the next instruction
    fn record_label(&mut self, name: &str, span: Option<Span>) {
        self.pending_labels.push(name.to_string());
        if let Some(s) = span {
            self.pending_label_spans.push(s);
        }
    }

    /// Record a forward reference to a label
    fn record_forward_ref(&mut self, label: &str) {
        self.forward_refs
            .push((self.current_pc(), label.to_string()));
    }

    /// Resolve all forward references, patching branch targets.
    fn resolve_forward_refs(&mut self) -> LowerResult<()> {
        for (pc, label) in &self.forward_refs {
            let Some(target) = self.symbols.resolve_label(label) else {
                return Err(LowerError::UndefinedLabel {
                    name: label.clone(),
                });
            };
            if let Some(LoweredInstr::Bra { target: t }) =
                self.instructions.get_mut(pc.to_index() as usize)
            {
                *t = target;
            }
        }
        Ok(())
    }

    /// Resolve a register operand to RegId
    fn resolve_register(&self, name: &str) -> LowerResult<RegId> {
        // Look up register by exact name (including any % prefix)
        self.symbols.resolve_register(name).ok_or_else(|| {
            let suggestions = self.symbols.find_similar_registers(name);
            LowerError::UndefinedRegister {
                name: name.to_string(),
                suggestions,
            }
        })
    }

    /// Resolve a memory-variable symbol (shared, local, or module-global) to
    /// its address operand. Shared and local variables live in their own
    /// per-space address spaces, so their "address" is the space-relative
    /// offset assigned by the symbol table; module globals get absolute
    /// addresses in a reserved region.
    fn resolve_mem_symbol(&self, name: &str) -> Option<Operand> {
        if let Some(info) = self.symbols.get_shared_var(name) {
            return Some(Operand::ImmU64(info.offset));
        }
        if let Some(info) = self.symbols.get_local_var(name) {
            return Some(Operand::ImmU64(info.offset));
        }
        if let Some(info) = self.symbols.get_global_var(name) {
            return Some(Operand::ImmU64(info.addr));
        }
        None
    }

    /// Resolve an operand (register, immediate, or special register)
    fn resolve_operand(&self, op: &AstOperand) -> LowerResult<Operand> {
        match op {
            AstOperand::Ident(name) => {
                let name_str = name.to_string();
                // PTX predefined constant: warp size
                if name_str == "WARP_SZ" {
                    return Ok(Operand::ImmI64(32));
                }
                // Check for special register first (e.g., %tid.x, %ntid.x)
                if let Some(kind) = SpecialRegKind::from_name(&name_str) {
                    return Ok(Operand::SpecialReg(kind));
                }
                // Check if it's a declared register
                if let Some(reg_id) = self.symbols.resolve_register(&name_str) {
                    return Ok(Operand::Reg(reg_id));
                }
                // Check if it's a shared/local/global memory symbol
                if let Some(op) = self.resolve_mem_symbol(&name_str) {
                    return Ok(op);
                }
                // Not found - return error with suggestions
                let suggestions = self.symbols.find_similar_registers(&name_str);
                Err(LowerError::UndefinedRegister {
                    name: name_str,
                    suggestions,
                })
            }
            AstOperand::ImmInt(val) => Ok(Operand::ImmI64(*val)),
            AstOperand::ImmUInt(val) => Ok(Operand::ImmU64(*val)),
            AstOperand::ImmFloat(val) => Ok(Operand::ImmF64(*val)),
            AstOperand::Symbol(name) => {
                let name_str = name.to_string();
                // Check if it's a shared/local/global memory symbol
                if let Some(op) = self.resolve_mem_symbol(&name_str) {
                    return Ok(op);
                }
                // Unknown symbol
                Err(LowerError::UnsupportedInstruction {
                    instruction: format!("symbol reference: {}", name),
                    reason: Some(format!(
                        "Unknown symbol '{}' - not a shared, local, or global variable",
                        name
                    )),
                })
            }
            AstOperand::Underscore => {
                // Underscore means "don't care" - we create a dummy register
                // This is used for predicates we want to discard
                Err(LowerError::UnsupportedInstruction {
                    instruction: "underscore operand".to_string(),
                    reason: Some("Underscore operands not yet implemented".to_string()),
                })
            }
            AstOperand::Address(addr) => {
                // Address operand - resolve the base
                self.resolve_address(addr)
            }
            AstOperand::Vector(_ops) => {
                // Vector operand - not handled inline, should be handled by instruction
                Err(LowerError::UnsupportedInstruction {
                    instruction: "vector operand".to_string(),
                    reason: Some(
                        "Vector operands should be handled by the instruction".to_string(),
                    ),
                })
            }
            AstOperand::PredicateOperand {
                negated: _,
                name: _,
            } => Err(LowerError::UnsupportedInstruction {
                instruction: "predicate operand".to_string(),
                reason: Some("Use resolve_predicate_operand for predicates".to_string()),
            }),
            AstOperand::PredicatePair(_, _) => Err(LowerError::UnsupportedInstruction {
                instruction: "predicate pair".to_string(),
                reason: Some("Predicate pairs not yet implemented".to_string()),
            }),
            AstOperand::VectorElement(name, component) => {
                // Check if this is actually a special register like %tid.x
                // The parser might mis-parse %tid.x as a VectorElement
                let full_name = format!(
                    "{}.{}",
                    name,
                    match component.canonicalize() {
                        ast::CanonVectorComponent::X => "x",
                        ast::CanonVectorComponent::Y => "y",
                        ast::CanonVectorComponent::Z => "z",
                        ast::CanonVectorComponent::W => "w",
                    }
                );
                if let Some(kind) = SpecialRegKind::from_name(&full_name) {
                    return Ok(Operand::SpecialReg(kind));
                }
                // Otherwise it's a real vector element which we don't support yet
                Err(LowerError::UnsupportedInstruction {
                    instruction: "vector element".to_string(),
                    reason: Some("Vector elements not yet implemented".to_string()),
                })
            }
            AstOperand::Expr(_) => Err(LowerError::UnsupportedInstruction {
                instruction: "expression operand".to_string(),
                reason: Some("Expression operands not yet implemented".to_string()),
            }),
        }
    }

    /// Resolve an operand with type information (for type checking)
    fn resolve_operand_typed(&self, op: &AstOperand) -> LowerResult<ResolvedOperand> {
        match op {
            AstOperand::Ident(name) => {
                let name_str = name.to_string();
                // PTX predefined constant: warp size (polymorphic immediate)
                if name_str == "WARP_SZ" {
                    return Ok(ResolvedOperand::immediate(Operand::ImmI64(32)));
                }
                // Check for special register first
                if let Some(kind) = SpecialRegKind::from_name(&name_str) {
                    let ty = kind.ty();
                    return Ok(ResolvedOperand::special_reg(
                        Operand::SpecialReg(kind),
                        ty,
                        name_str,
                    ));
                }
                // Check if it's a declared register
                if let Some(info) = self.symbols.get_register(&name_str) {
                    return Ok(ResolvedOperand::register(
                        Operand::Reg(info.id),
                        info.declared_type,
                        name_str,
                    ));
                }
                // Check if it's a shared/local/global memory symbol. The
                // resulting address is a constant, so treat it like an
                // immediate (polymorphic - `mov.u32` takes shared addresses,
                // `mov.u64` takes local depot addresses).
                if let Some(op) = self.resolve_mem_symbol(&name_str) {
                    return Ok(ResolvedOperand::immediate(op));
                }
                // Not found - return error
                let suggestions = self.symbols.find_similar_registers(&name_str);
                Err(LowerError::UndefinedRegister {
                    name: name_str,
                    suggestions,
                })
            }
            AstOperand::ImmInt(_) | AstOperand::ImmUInt(_) | AstOperand::ImmFloat(_) => {
                // Immediates are polymorphic - their type is determined by context
                Ok(ResolvedOperand::immediate(self.resolve_operand(op)?))
            }
            AstOperand::VectorElement(name, component) => {
                // Check if this is actually a special register like %tid.x
                let full_name = format!(
                    "{}.{}",
                    name,
                    match component.canonicalize() {
                        ast::CanonVectorComponent::X => "x",
                        ast::CanonVectorComponent::Y => "y",
                        ast::CanonVectorComponent::Z => "z",
                        ast::CanonVectorComponent::W => "w",
                    }
                );
                if let Some(kind) = SpecialRegKind::from_name(&full_name) {
                    let ty = kind.ty();
                    return Ok(ResolvedOperand::special_reg(
                        Operand::SpecialReg(kind),
                        ty,
                        full_name,
                    ));
                }
                // Otherwise delegate to resolve_operand (which will error)
                Ok(ResolvedOperand::immediate(self.resolve_operand(op)?))
            }
            _ => {
                // For other operand types, delegate to resolve_operand
                // These are typically address operands or other special cases
                Ok(ResolvedOperand::immediate(self.resolve_operand(op)?))
            }
        }
    }

    /// Resolve a destination register with type information (for type checking)
    fn resolve_dst_typed(&self, op: &AstOperand) -> LowerResult<ResolvedDst> {
        match op {
            AstOperand::Ident(name) => {
                let name_str = name.to_string();
                // Check if it's a special register (not allowed as destination)
                if SpecialRegKind::from_name(&name_str).is_some() {
                    return Err(LowerError::SpecialRegAsDestination {
                        instruction: "destination".to_string(),
                        register: name_str,
                    });
                }
                // Look up the register
                if let Some(info) = self.symbols.get_register(&name_str) {
                    return Ok(ResolvedDst {
                        reg: info.id,
                        ty: info.declared_type,
                        name: name_str,
                    });
                }
                // Not found
                let suggestions = self.symbols.find_similar_registers(&name_str);
                Err(LowerError::UndefinedRegister {
                    name: name_str,
                    suggestions,
                })
            }
            _ => Err(LowerError::InvalidOperand {
                instruction: "instruction".to_string(),
                operand: format!("{:?}", op),
                reason: "destination must be a register",
            }),
        }
    }

    /// Resolve an address operand to base operand
    fn resolve_address(&self, addr: &Address) -> LowerResult<Operand> {
        match &addr.base {
            AddressBase::Register(name) => {
                let name_str = name.to_string();
                let reg_id = self.resolve_register(&name_str)?;
                Ok(Operand::Reg(reg_id))
            }
            AddressBase::Symbol(name) => {
                // Shared, local, or module-global variables used as a base
                let name_str = name.to_string();
                if let Some(op) = self.resolve_mem_symbol(&name_str) {
                    Ok(op)
                } else {
                    Err(LowerError::UndefinedSymbol { name: name_str })
                }
            }
            AddressBase::Immediate(val) => Ok(Operand::ImmI64(*val)),
        }
    }

    /// Get offset from address
    fn get_address_offset(&self, addr: &Address) -> i64 {
        match &addr.offset {
            Some(expr) => {
                // Try to evaluate constant expression
                Self::eval_const_expr(expr).unwrap_or(0)
            }
            None => 0,
        }
    }

    /// Try to evaluate a constant expression
    fn eval_const_expr(expr: &ast::Expr) -> Option<i64> {
        match expr {
            ast::Expr::IntLitS(v) => Some(*v),
            ast::Expr::IntLitU(v) => Some(*v as i64),
            ast::Expr::Binary(lhs, op, rhs) => {
                let l = Self::eval_const_expr(lhs)?;
                let r = Self::eval_const_expr(rhs)?;
                Some(match op {
                    ast::BinaryOp::Add => l.wrapping_add(r),
                    ast::BinaryOp::Sub => l.wrapping_sub(r),
                    ast::BinaryOp::Mul => l.wrapping_mul(r),
                    ast::BinaryOp::Div => l.checked_div(r)?,
                    ast::BinaryOp::Shl => l.wrapping_shl(r as u32),
                    ast::BinaryOp::Shr => l.wrapping_shr(r as u32),
                    _ => return None,
                })
            }
            ast::Expr::Unary(op, inner) => {
                let v = Self::eval_const_expr(inner)?;
                Some(match op {
                    ast::UnaryOp::Neg => v.wrapping_neg(),
                    ast::UnaryOp::Pos => v,
                    _ => return None,
                })
            }
            _ => None,
        }
    }

    /// Resolve destination register (must not be a special register)
    fn resolve_dst(&self, op: &AstOperand) -> LowerResult<RegId> {
        match op {
            AstOperand::Ident(name) => {
                let name_str = name.to_string();
                // Check if it's a special register
                if SpecialRegKind::from_name(&name_str).is_some() {
                    return Err(LowerError::SpecialRegAsDestination {
                        instruction: "destination".to_string(),
                        register: name_str,
                    });
                }
                self.resolve_register(&name_str)
            }
            _ => Err(LowerError::InvalidOperand {
                instruction: "instruction".to_string(),
                operand: format!("{:?}", op),
                reason: "destination must be a register",
            }),
        }
    }

    /// Resolve a vector of destination registers
    fn resolve_dst_vector(&self, op: &AstOperand) -> LowerResult<Vec<RegId>> {
        match op {
            AstOperand::Vector(regs) => {
                let mut result = Vec::with_capacity(regs.len());
                for reg_op in regs {
                    match reg_op {
                        AstOperand::Ident(name) => {
                            let name_str = name.to_string();
                            result.push(self.resolve_register(&name_str)?);
                        }
                        _ => {
                            return Err(LowerError::InvalidOperand {
                                instruction: "vector load".to_string(),
                                operand: format!("{:?}", reg_op),
                                reason: "vector element must be a register",
                            });
                        }
                    }
                }
                Ok(result)
            }
            AstOperand::Ident(name) => {
                // Single register - return as single-element vector
                let name_str = name.to_string();
                Ok(vec![self.resolve_register(&name_str)?])
            }
            _ => Err(LowerError::InvalidOperand {
                instruction: "vector load".to_string(),
                operand: format!("{:?}", op),
                reason: "destination must be a register or vector of registers",
            }),
        }
    }

    /// Resolve a predicate guard
    fn resolve_predicate(&self, pred: &ast::Predicate) -> LowerResult<Predicate> {
        let name_str = pred.reg.to_string();
        let reg = self.resolve_register(&name_str)?;
        Ok(Predicate {
            reg,
            negated: pred.negated,
        })
    }

    /// Convert AST memory space to lowered memory space
    fn convert_space(&self, space: Option<StateSpace>) -> MemSpace {
        match space {
            Some(StateSpace::Global) => MemSpace::Global,
            Some(StateSpace::Shared) => MemSpace::Shared,
            Some(StateSpace::Local) => MemSpace::Local,
            Some(StateSpace::Param) => MemSpace::Param,
            Some(StateSpace::Const) => MemSpace::Const,
            _ => MemSpace::Global, // Default to global
        }
    }

    /// Convert AST comparison operator to lowered comparison operator
    fn convert_cmp_op(&self, op: AstCmpOp) -> CmpOp {
        match op {
            AstCmpOp::Eq => CmpOp::Eq,
            AstCmpOp::Ne => CmpOp::Ne,
            AstCmpOp::Lt => CmpOp::Lt,
            AstCmpOp::Le => CmpOp::Le,
            AstCmpOp::Gt => CmpOp::Gt,
            AstCmpOp::Ge => CmpOp::Ge,
            AstCmpOp::Lo => CmpOp::Lo,
            AstCmpOp::Ls => CmpOp::Ls,
            AstCmpOp::Hi => CmpOp::Hi,
            AstCmpOp::Hs => CmpOp::Hs,
            AstCmpOp::Equ => CmpOp::Equ,
            AstCmpOp::Neu => CmpOp::Neu,
            AstCmpOp::Ltu => CmpOp::Ltu,
            AstCmpOp::Leu => CmpOp::Leu,
            AstCmpOp::Gtu => CmpOp::Gtu,
            AstCmpOp::Geu => CmpOp::Geu,
            AstCmpOp::Num => CmpOp::Num,
            AstCmpOp::Nan => CmpOp::Nan,
        }
    }

    /// Convert AST shuffle mode to lowered shuffle mode
    fn convert_shfl_mode(&self, mode: AstShflMode) -> ShflMode {
        match mode {
            AstShflMode::Up => ShflMode::Up,
            AstShflMode::Down => ShflMode::Down,
            AstShflMode::Bfly => ShflMode::Bfly,
            AstShflMode::Idx => ShflMode::Idx,
        }
    }

    /// Convert AST mul mode to lowered mul mode
    fn convert_mul_mode(&self, mode: MulMode) -> LoweredMulMode {
        match mode {
            MulMode::Hi => LoweredMulMode::Hi,
            MulMode::Lo => LoweredMulMode::Lo,
            MulMode::Wide => LoweredMulMode::Wide,
        }
    }

    // =========================================================================
    // Type Checking Helpers
    // =========================================================================

    /// Check that an operand's type is compatible with the instruction's expected type.
    /// Per PTX 9.4:
    /// - Bit-types (.bX) are compatible with any type of same size
    /// - Signed/unsigned integers of same size are compatible
    /// - Float types must match exactly (no float<->int mixing)
    fn check_operand_type(
        &self,
        resolved: &ResolvedOperand,
        expected: ScalarType,
        instruction: &str,
    ) -> LowerResult<()> {
        // Immediates are polymorphic - they take on the instruction's type
        let Some(actual) = resolved.ty else {
            return Ok(());
        };

        match check_type_compatibility(actual, expected) {
            TypeCompatibility::Exact | TypeCompatibility::Compatible => Ok(()),
            TypeCompatibility::Incompatible { reason: _ } => {
                let name = resolved
                    .name
                    .clone()
                    .unwrap_or_else(|| "operand".to_string());
                Err(LowerError::TypeMismatch {
                    register: name,
                    declared_type: actual,
                    used_as: expected,
                    instruction: instruction.to_string(),
                    hint: self.type_mismatch_hint(actual, expected),
                })
            }
        }
    }

    /// Check that a destination register's type is compatible with the instruction's type.
    fn check_dst_type(
        &self,
        dst: &ResolvedDst,
        expected: ScalarType,
        instruction: &str,
    ) -> LowerResult<()> {
        match check_type_compatibility(dst.ty, expected) {
            TypeCompatibility::Exact | TypeCompatibility::Compatible => Ok(()),
            TypeCompatibility::Incompatible { reason: _ } => Err(LowerError::TypeMismatch {
                register: dst.name.clone(),
                declared_type: dst.ty,
                used_as: expected,
                instruction: instruction.to_string(),
                hint: self.type_mismatch_hint(dst.ty, expected),
            }),
        }
    }

    /// Relaxed type checking for ld/st/cvt instructions (PTX 9.4.1).
    /// Allows operands to be wider than the instruction type.
    /// The value will be truncated (store) or extended (load) as needed.
    fn check_operand_type_relaxed(
        &self,
        resolved: &ResolvedOperand,
        instr_ty: ScalarType,
        instruction: &str,
    ) -> LowerResult<()> {
        // Immediates are polymorphic
        let Some(actual) = resolved.ty else {
            return Ok(());
        };

        let actual_bits = actual.bits();
        let instr_bits = instr_ty.bits();

        // Operand can be wider than instruction type (will be truncated/extended)
        if actual_bits >= instr_bits {
            // Still check type category compatibility (no float<->int mixing)
            // unless one is a bit-type
            if actual.is_bits_type() || instr_ty.is_bits_type() {
                return Ok(());
            }
            // Float<->int mixing is still invalid
            if actual.is_float() != instr_ty.is_float() {
                let name = resolved
                    .name
                    .clone()
                    .unwrap_or_else(|| "operand".to_string());
                return Err(LowerError::TypeMismatch {
                    register: name,
                    declared_type: actual,
                    used_as: instr_ty,
                    instruction: instruction.to_string(),
                    hint: "Cannot mix float and integer types; use cvt for conversion".to_string(),
                });
            }
            return Ok(());
        }

        // Operand is narrower than instruction type - not allowed even with relaxed rules
        let name = resolved
            .name
            .clone()
            .unwrap_or_else(|| "operand".to_string());
        Err(LowerError::TypeMismatch {
            register: name,
            declared_type: actual,
            used_as: instr_ty,
            instruction: instruction.to_string(),
            hint: format!(
                "Operand is {} bits but instruction requires at least {} bits",
                actual_bits, instr_bits
            ),
        })
    }

    /// Relaxed type checking for destination registers (ld instructions).
    /// Destination can be wider than instruction type (value will be extended).
    fn check_dst_type_relaxed(
        &self,
        dst: &ResolvedDst,
        instr_ty: ScalarType,
        instruction: &str,
    ) -> LowerResult<()> {
        let dst_bits = dst.ty.bits();
        let instr_bits = instr_ty.bits();

        // Destination can be wider (value will be extended)
        if dst_bits >= instr_bits {
            // Check type category compatibility
            if dst.ty.is_bits_type() || instr_ty.is_bits_type() {
                return Ok(());
            }
            if dst.ty.is_float() != instr_ty.is_float() {
                return Err(LowerError::TypeMismatch {
                    register: dst.name.clone(),
                    declared_type: dst.ty,
                    used_as: instr_ty,
                    instruction: instruction.to_string(),
                    hint: "Cannot mix float and integer types; use cvt for conversion".to_string(),
                });
            }
            return Ok(());
        }

        // Destination is narrower - not allowed
        Err(LowerError::TypeMismatch {
            register: dst.name.clone(),
            declared_type: dst.ty,
            used_as: instr_ty,
            instruction: instruction.to_string(),
            hint: format!(
                "Destination register is {} bits but instruction produces {} bits",
                dst_bits, instr_bits
            ),
        })
    }

    /// Generate a helpful hint for type mismatch errors
    fn type_mismatch_hint(&self, actual: ScalarType, expected: ScalarType) -> String {
        // Float<->int mismatch
        if actual.is_float() && expected.is_integer() {
            return format!(
                "Use cvt.{}.{} to convert float to integer",
                crate::types::format_scalar_type(expected),
                crate::types::format_scalar_type(actual)
            );
        }
        if actual.is_integer() && expected.is_float() {
            return format!(
                "Use cvt.{}.{} to convert integer to float",
                crate::types::format_scalar_type(expected),
                crate::types::format_scalar_type(actual)
            );
        }

        // Size mismatch
        if actual.bits() != expected.bits() {
            return format!(
                "Type size mismatch: {} is {} bits, expected {} bits",
                crate::types::format_scalar_type(actual),
                actual.bits(),
                expected.bits()
            );
        }

        "Declare register with compatible type".to_string()
    }
}

impl Default for LoweringContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Lower a PTX function to a LoweredProgram
///
/// `module_vars` should contain module-level variable declarations (e.g., extern shared memory).
pub fn lower_function(func: &Function, module_vars: &[VarDecl]) -> LowerResult<LoweredProgram> {
    let body = func.body.as_ref().ok_or(LowerError::NoFunctionBody {
        name: func.name.to_string(),
    })?;

    let mut ctx = LoweringContext::new();

    // First: collect module-level declarations (e.g., extern shared memory)
    for var in module_vars {
        collect_var_decl(&mut ctx, var)?;
    }

    // Second: collect function-level declarations
    collect_declarations(&mut ctx, &func.params, body)?;

    // Third: lower instructions
    lower_body(&mut ctx, body)?;

    // Resolve forward references
    ctx.resolve_forward_refs()?;

    Ok(LoweredProgram {
        instructions: IdVec::from_vec(ctx.instructions),
        predicates: IdVec::from_vec(ctx.predicates),
        symbols: ctx.symbols,
        source_map: ctx.source_map_builder.build(),
        entry_pc: InstrId::from_index(0),
    })
}

/// Collect all declarations (registers, labels, shared memory)
fn collect_declarations(
    ctx: &mut LoweringContext,
    params: &[ast::Parameter],
    body: &FunctionBody,
) -> LowerResult<()> {
    // Collect kernel parameters
    for param in params {
        let ty = param.ty.scalar;
        let size_bytes = ty.size_bytes() as u64;
        ctx.symbols
            .declare_param(&param.name.to_string(), ty, size_bytes)?;
    }

    // Collect declarations from body
    for stmt in &body.statements {
        collect_stmt_declarations(ctx, stmt)?;
    }

    Ok(())
}

/// Collect declarations from a statement
fn collect_stmt_declarations(ctx: &mut LoweringContext, stmt: &Statement) -> LowerResult<()> {
    match stmt {
        Statement::Variable(var_decl) => {
            collect_var_decl(ctx, var_decl)?;
        }
        Statement::Block(stmts) => {
            for s in stmts {
                collect_stmt_declarations(ctx, s)?;
            }
        }
        Statement::Label(_) | Statement::Instruction(_) | Statement::Directive(_) => {
            // These don't introduce declarations
        }
    }
    Ok(())
}

/// Collect a variable declaration
fn collect_var_decl(ctx: &mut LoweringContext, var: &VarDecl) -> LowerResult<()> {
    let name = var.name.to_string();
    let ty = var.ty.scalar;
    let count = var.param_count.unwrap_or(1);

    let num_elements: u64 = var
        .array_dims
        .iter()
        .filter_map(|d| *d)
        .map(|d| d as u64)
        .product();
    let num_elements = if num_elements == 0 { 1 } else { num_elements };
    let alignment = var.align.unwrap_or(ty.size_bytes() as u32) as u64;

    match var.space {
        StateSpace::Reg => {
            ctx.symbols.declare_register(&name, ty, count)?;
        }
        StateSpace::Shared => {
            let is_extern = matches!(var.linkage, ast::Linkage::Extern);
            ctx.symbols
                .declare_shared(&name, ty, num_elements, is_extern, alignment)?;
        }
        StateSpace::Param => {
            // Block-scope `.param` slots used by the callseq idiom. These are
            // virtual: no memory is allocated and no instructions touch them.
            ctx.local_params.insert(name, LocalParamSlot::Empty);
        }
        StateSpace::Local => {
            // Function-scope local memory (e.g. the __local_depot stack array)
            ctx.symbols
                .declare_local(&name, ty, num_elements, alignment)?;
        }
        StateSpace::Global => {
            // Module-scope variable (e.g. set by the host via
            // cudaMemcpyToSymbol); the driver binds its value by name.
            ctx.symbols
                .declare_global_var(&name, ty, num_elements, alignment)?;
        }
        _ => {
            // Other state spaces (const, tex) - handle as needed
        }
    }

    Ok(())
}

/// Lower the function body
fn lower_body(ctx: &mut LoweringContext, body: &FunctionBody) -> LowerResult<()> {
    for stmt in &body.statements {
        lower_statement(ctx, stmt)?;
    }
    Ok(())
}

/// Lower a statement
fn lower_statement(ctx: &mut LoweringContext, stmt: &Statement) -> LowerResult<()> {
    match stmt {
        Statement::Label(label) => {
            ctx.record_label(&label.name.to_string(), Some(label.span));
        }
        Statement::Instruction(instr) => {
            // Set current span for error reporting
            ctx.current_span = Some(instr.span);
            lower_instruction(ctx, instr)?;
        }
        Statement::Block(stmts) => {
            for s in stmts {
                lower_statement(ctx, s)?;
            }
        }
        Statement::Variable(_) => {
            // Already handled in first pass
        }
        Statement::Directive(_) => {
            // Directives are ignored for now
        }
    }
    Ok(())
}

/// Lower an instruction
fn lower_instruction(ctx: &mut LoweringContext, instr: &Instruction) -> LowerResult<()> {
    // Resolve predicate if present
    let predicate = match &instr.predicate {
        Some(p) => Some(ctx.resolve_predicate(p)?),
        None => None,
    };

    match &instr.op {
        InstructionOp::Parsed(parsed) => {
            lower_parsed_instruction(ctx, parsed, predicate)?;
        }
        InstructionOp::Unparsed {
            kind,
            modifiers,
            operands,
        } => {
            // Try to parse the unparsed instruction into a strongly-typed form
            match parse_instruction(*kind, modifiers.clone(), operands.clone()) {
                Ok(parsed) => {
                    lower_parsed_instruction(ctx, &parsed, predicate)?;
                }
                Err(e) => {
                    // If parsing fails, report the error
                    Err(LowerError::UnsupportedInstruction {
                        instruction: format!("{:?}", kind),
                        reason: Some(format!("Instruction parsing failed: {:?}", e)),
                    })?;
                }
            }
        }
    }

    Ok(())
}

/// Lower a parsed instruction
fn lower_parsed_instruction(
    ctx: &mut LoweringContext,
    instr: &ParsedInstruction,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    match instr {
        // =========================================================================
        // Arithmetic - Add
        // =========================================================================
        ParsedInstruction::Add(add_instr) => {
            lower_add(ctx, add_instr, predicate)?;
        }

        // =========================================================================
        // Arithmetic - Sub
        // =========================================================================
        ParsedInstruction::Sub(sub) => {
            lower_sub(ctx, sub, predicate)?;
        }

        // =========================================================================
        // Arithmetic - Mul
        // =========================================================================
        ParsedInstruction::Mul(mul) => {
            lower_mul(ctx, mul, predicate)?;
        }

        // =========================================================================
        // Arithmetic - Mad (multiply-add)
        // =========================================================================
        ParsedInstruction::Mad(mad) => {
            lower_mad(ctx, mad, predicate)?;
        }

        // =========================================================================
        // Arithmetic - FMA (fused multiply-add)
        // =========================================================================
        ParsedInstruction::Fma(fma) => {
            lower_fma(ctx, fma, predicate)?;
        }

        // =========================================================================
        // Arithmetic - Div
        // =========================================================================
        ParsedInstruction::Div(div) => {
            lower_div(ctx, div, predicate)?;
        }

        // =========================================================================
        // Arithmetic - Rem
        // =========================================================================
        ParsedInstruction::Rem(rem) => {
            let dst_typed = ctx.resolve_dst_typed(&rem.dst)?;
            let src_a_typed = ctx.resolve_operand_typed(&rem.src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(&rem.src_b)?;

            ctx.check_dst_type(&dst_typed, rem.ty, "rem")?;
            ctx.check_operand_type(&src_a_typed, rem.ty, "rem")?;
            ctx.check_operand_type(&src_b_typed, rem.ty, "rem")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Rem,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: rem.ty,
                },
                predicate,
            )?;
        }

        // =========================================================================
        // Arithmetic - Neg
        // =========================================================================
        ParsedInstruction::Neg(neg) => {
            lower_neg(ctx, neg, predicate)?;
        }

        // =========================================================================
        // Float unary - Rcp / Sqrt / Rsqrt
        // =========================================================================
        ParsedInstruction::Rcp(rcp) => {
            let (ty, dst, src) = match rcp {
                ast::RcpInstr::Approx { ty, dst, src, .. } => (*ty, dst, src),
                ast::RcpInstr::Ieee { ty, dst, src, .. } => (*ty, dst, src),
            };
            lower_float_unary(ctx, UnaryOp::Rcp, "rcp", ty, dst, src, predicate)?;
        }
        ParsedInstruction::Sqrt(sqrt) => {
            let (ty, dst, src) = match sqrt {
                ast::SqrtInstr::Approx { dst, src, .. } => (ScalarType::F32, dst, src),
                ast::SqrtInstr::Ieee { ty, dst, src, .. } => (*ty, dst, src),
            };
            lower_float_unary(ctx, UnaryOp::Sqrt, "sqrt", ty, dst, src, predicate)?;
        }
        ParsedInstruction::Rsqrt(rsqrt) => {
            lower_float_unary(
                ctx,
                UnaryOp::Rsqrt,
                "rsqrt",
                rsqrt.ty,
                &rsqrt.dst,
                &rsqrt.src,
                predicate,
            )?;
        }

        // =========================================================================
        // Arithmetic - Abs
        // =========================================================================
        ParsedInstruction::Abs(abs) => {
            lower_abs(ctx, abs, predicate)?;
        }

        // =========================================================================
        // Arithmetic - Min/Max
        // =========================================================================
        ParsedInstruction::Min(min) => {
            lower_min(ctx, min, predicate)?;
        }
        ParsedInstruction::Max(max) => {
            lower_max(ctx, max, predicate)?;
        }

        // =========================================================================
        // Logic - And/Or/Xor
        // =========================================================================
        ParsedInstruction::And(logic)
        | ParsedInstruction::Or(logic)
        | ParsedInstruction::Xor(logic) => {
            let (op, instr_name) = match instr {
                ParsedInstruction::And(_) => (BinOp::And, "and"),
                ParsedInstruction::Or(_) => (BinOp::Or, "or"),
                ParsedInstruction::Xor(_) => (BinOp::Xor, "xor"),
                _ => unreachable!(),
            };
            let dst_typed = ctx.resolve_dst_typed(&logic.dst)?;
            let src_a_typed = ctx.resolve_operand_typed(&logic.src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(&logic.src_b)?;

            ctx.check_dst_type(&dst_typed, logic.ty, instr_name)?;
            ctx.check_operand_type(&src_a_typed, logic.ty, instr_name)?;
            ctx.check_operand_type(&src_b_typed, logic.ty, instr_name)?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: logic.ty,
                },
                predicate,
            )?;
        }

        // =========================================================================
        // Logic - Not
        // =========================================================================
        ParsedInstruction::Not(not) => {
            let dst_typed = ctx.resolve_dst_typed(&not.dst)?;
            let src_typed = ctx.resolve_operand_typed(&not.src)?;

            ctx.check_dst_type(&dst_typed, not.ty, "not")?;
            ctx.check_operand_type(&src_typed, not.ty, "not")?;

            ctx.emit(
                LoweredInstr::UnaryOp {
                    op: UnaryOp::Not,
                    dst: dst_typed.reg,
                    src: src_typed.operand,
                    ty: not.ty,
                },
                predicate,
            )?;
        }

        // =========================================================================
        // Shift - Shl/Shr
        // =========================================================================
        ParsedInstruction::Shl(shift) | ParsedInstruction::Shr(shift) => {
            let (op, instr_name) = match instr {
                ParsedInstruction::Shl(_) => (BinOp::Shl, "shl"),
                ParsedInstruction::Shr(_) => (BinOp::Shr, "shr"),
                _ => unreachable!(),
            };
            let dst_typed = ctx.resolve_dst_typed(&shift.dst)?;
            let src_a_typed = ctx.resolve_operand_typed(&shift.src_a)?;
            // For shift, src_b is the shift amount - typically u32, but we check against instruction type
            let src_b_typed = ctx.resolve_operand_typed(&shift.src_b)?;

            ctx.check_dst_type(&dst_typed, shift.ty, instr_name)?;
            ctx.check_operand_type(&src_a_typed, shift.ty, instr_name)?;
            // Shift amount is typically smaller, but we allow same type or compatible
            ctx.check_operand_type(&src_b_typed, shift.ty, instr_name)?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: shift.ty,
                },
                predicate,
            )?;
        }

        // =========================================================================
        // Bit Field Insert - Bfi
        // =========================================================================
        ParsedInstruction::Bfi(bfi) => {
            let dst_typed = ctx.resolve_dst_typed(&bfi.dst)?;
            let src_a_typed = ctx.resolve_operand_typed(&bfi.src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(&bfi.src_b)?;
            // start and len are typically u32 immediates or registers
            let start = ctx.resolve_operand(&bfi.start)?;
            let len = ctx.resolve_operand(&bfi.len)?;

            ctx.check_dst_type(&dst_typed, bfi.ty, "bfi")?;
            ctx.check_operand_type(&src_a_typed, bfi.ty, "bfi")?;
            ctx.check_operand_type(&src_b_typed, bfi.ty, "bfi")?;

            ctx.emit(
                LoweredInstr::Bfi {
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    start,
                    len,
                    ty: bfi.ty,
                },
                predicate,
            )?;
        }

        // =========================================================================
        // Comparison - Setp
        // =========================================================================
        ParsedInstruction::Setp(setp) => {
            lower_setp(ctx, setp, predicate)?;
        }

        // =========================================================================
        // Selection - Selp
        // =========================================================================
        ParsedInstruction::Selp(selp) => {
            let dst_typed = ctx.resolve_dst_typed(&selp.dst)?;
            let src_a_typed = ctx.resolve_operand_typed(&selp.src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(&selp.src_b)?;
            let pred_typed = ctx.resolve_operand_typed(&selp.src_c)?;

            ctx.check_dst_type(&dst_typed, selp.ty, "selp")?;
            ctx.check_operand_type(&src_a_typed, selp.ty, "selp")?;
            ctx.check_operand_type(&src_b_typed, selp.ty, "selp")?;
            // Predicate operand must be pred type
            ctx.check_operand_type(&pred_typed, ScalarType::Pred, "selp")?;

            ctx.emit(
                LoweredInstr::Selp {
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    pred: pred_typed.operand,
                    ty: selp.ty,
                },
                predicate,
            )?;
        }

        // =========================================================================
        // Data Movement - Mov
        // =========================================================================
        ParsedInstruction::Mov(mov) => {
            match &mov.src {
                AstOperand::Vector(elements) if elements.len() == 2 => {
                    // Vector-to-scalar pack: mov.bN dst, {lo, hi}
                    // PTX ISA 9.7.9.4 semantics: d = lo | (hi << w)
                    // where w = type_bits / 2
                    let dst_typed = ctx.resolve_dst_typed(&mov.dst)?;
                    let dst = dst_typed.reg;
                    let elem_width = mov.ty.bits() / 2;

                    let lo = ctx.resolve_operand(&elements[0])?;
                    let hi = ctx.resolve_operand(&elements[1])?;

                    // Step 1: dst = hi << w
                    ctx.emit(
                        LoweredInstr::BinOp {
                            op: BinOp::Shl,
                            dst,
                            src_a: hi,
                            src_b: Operand::ImmI64(elem_width as i64),
                            ty: mov.ty,
                        },
                        predicate,
                    )?;

                    // Step 2: dst = dst | lo
                    ctx.emit(
                        LoweredInstr::BinOp {
                            op: BinOp::Or,
                            dst,
                            src_a: Operand::Reg(dst),
                            src_b: lo,
                            ty: mov.ty,
                        },
                        predicate,
                    )?;
                }
                AstOperand::Vector(elements) => {
                    return Err(LowerError::UnsupportedInstruction {
                        instruction: format!(
                            "mov.{:?} (vector pack with {} elements)",
                            mov.ty,
                            elements.len()
                        ),
                        reason: Some("only 2-element vector pack is supported".to_string()),
                    });
                }
                _ => {
                    // Normal scalar mov
                    let dst_typed = ctx.resolve_dst_typed(&mov.dst)?;
                    let src_typed = ctx.resolve_operand_typed(&mov.src)?;

                    ctx.check_dst_type(&dst_typed, mov.ty, "mov")?;
                    ctx.check_operand_type(&src_typed, mov.ty, "mov")?;

                    ctx.emit(
                        LoweredInstr::Mov {
                            dst: dst_typed.reg,
                            src: src_typed.operand,
                            ty: mov.ty,
                        },
                        predicate,
                    )?;
                }
            }
        }

        // =========================================================================
        // Data Movement - Load
        // =========================================================================
        ParsedInstruction::Ld(ld) => {
            lower_load(ctx, ld, predicate)?;
        }

        // =========================================================================
        // Data Movement - Store
        // =========================================================================
        ParsedInstruction::St(st) => {
            lower_store(ctx, st, predicate)?;
        }

        // =========================================================================
        // Data Movement - Cvt (type conversion)
        // =========================================================================
        ParsedInstruction::Cvt(cvt) => {
            lower_cvt(ctx, cvt, predicate)?;
        }

        // =========================================================================
        // Data Movement - Cvta (address conversion)
        // =========================================================================
        ParsedInstruction::Cvta(cvta) => {
            let dst = ctx.resolve_dst(&cvta.dst)?;
            let src = ctx.resolve_operand(&cvta.src)?;
            let space = ctx.convert_space(Some(cvta.space));
            ctx.emit(LoweredInstr::Cvta { dst, src, space }, predicate)?;
        }

        // =========================================================================
        // Warp Shuffle - ShflSync
        // =========================================================================
        ParsedInstruction::ShflSync(shfl) => {
            lower_shfl_sync(ctx, shfl, predicate)?;
        }

        // =========================================================================
        // Control Flow - Branch
        // =========================================================================
        ParsedInstruction::Bra(bra) => {
            lower_branch(ctx, bra, predicate)?;
        }

        // =========================================================================
        // Control Flow - Return/Exit
        // =========================================================================
        ParsedInstruction::Ret(_) => {
            ctx.emit(LoweredInstr::Ret, predicate)?;
        }
        ParsedInstruction::Exit => {
            ctx.emit(LoweredInstr::Exit, predicate)?;
        }

        // =========================================================================
        // Synchronization - Bar
        // =========================================================================
        ParsedInstruction::Bar(bar) => {
            lower_bar(ctx, bar, predicate)?;
        }

        // =========================================================================
        // Synchronization - Membar
        // =========================================================================
        ParsedInstruction::Membar(membar) => {
            let scope = match membar.level {
                ast::MemScope::Cta => MembarScope::Cta,
                ast::MemScope::Cluster => MembarScope::Cta, // Treat cluster as CTA
                ast::MemScope::Gpu => MembarScope::Gpu,
                ast::MemScope::Sys => MembarScope::Sys,
            };
            ctx.emit(LoweredInstr::Membar { scope }, predicate)?;
        }

        // =========================================================================
        // Function call (callseq idiom for __symexpf)
        // =========================================================================
        ParsedInstruction::Call(call) => {
            lower_call(ctx, call, predicate)?;
        }

        // =========================================================================
        // Warp query - activemask
        // =========================================================================
        ParsedInstruction::Activemask(am) => {
            let dst = ctx.resolve_dst(&am.dst)?;
            ctx.emit(LoweredInstr::Activemask { dst }, predicate)?;
        }

        // =========================================================================
        // Other - NOP placeholder
        // =========================================================================
        ParsedInstruction::Brkpt => {
            ctx.emit(LoweredInstr::Nop, predicate)?;
        }
        ParsedInstruction::Trap => {
            // Reaching a trap during evaluation is an analysis error
            ctx.emit(LoweredInstr::Trap, predicate)?;
        }

        // =========================================================================
        // Other - ld.global.nc (non-coherent global load)
        // Treat as regular global load - we don't model cache coherence
        // =========================================================================
        ParsedInstruction::Other {
            kind: InstrKind::LdGlobalNc,
            modifiers,
            operands,
        } => {
            lower_ld_global_nc(ctx, modifiers, operands, predicate)?;
        }

        // =========================================================================
        // Tensor Core - ldmatrix.sync
        // =========================================================================
        ParsedInstruction::Other {
            kind: InstrKind::Ldmatrix,
            modifiers,
            operands,
        } => {
            lower_ldmatrix(ctx, modifiers, operands, predicate)?;
        }

        // =========================================================================
        // Tensor Core - mma.sync
        // =========================================================================
        ParsedInstruction::Other {
            kind: InstrKind::Mma,
            modifiers,
            operands,
        } => {
            lower_mma(ctx, modifiers, operands, predicate)?;
        }

        // =========================================================================
        // Tensor Core - wmma.load
        // =========================================================================
        ParsedInstruction::Other {
            kind: InstrKind::WmmaLoad,
            modifiers,
            operands,
        } => {
            lower_wmma_load(ctx, modifiers, operands, predicate)?;
        }

        // =========================================================================
        // Tensor Core - wmma.store
        // =========================================================================
        ParsedInstruction::Other {
            kind: InstrKind::WmmaStore,
            modifiers,
            operands,
        } => {
            lower_wmma_store(ctx, modifiers, operands, predicate)?;
        }

        // =========================================================================
        // Tensor Core - wmma.mma
        // =========================================================================
        ParsedInstruction::Other {
            kind: InstrKind::WmmaMma,
            modifiers,
            operands,
        } => {
            lower_wmma_mma(ctx, modifiers, operands, predicate)?;
        }

        // =========================================================================
        // Unsupported instructions
        // =========================================================================
        _ => {
            return Err(LowerError::UnsupportedInstruction {
                instruction: format!("{:?}", instr),
                reason: Some("Instruction not yet implemented".to_string()),
            });
        }
    }

    Ok(())
}

// =========================================================================
// Instruction Lowering Helpers
// =========================================================================

fn lower_add(
    ctx: &mut LoweringContext,
    add: &AddInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    match add {
        AddInstr::Integer {
            ty,
            dst,
            src_a,
            src_b,
        } => {
            // Resolve with type information
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            // Type check against instruction type
            ctx.check_dst_type(&dst_typed, *ty, "add")?;
            ctx.check_operand_type(&src_a_typed, *ty, "add")?;
            ctx.check_operand_type(&src_b_typed, *ty, "add")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Add,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: *ty,
                },
                predicate,
            )?;
        }
        AddInstr::IntegerSat {
            sat: _,
            dst,
            src_a,
            src_b,
        } => {
            let ty = ScalarType::S32;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, ty, "add.sat")?;
            ctx.check_operand_type(&src_a_typed, ty, "add.sat")?;
            ctx.check_operand_type(&src_b_typed, ty, "add.sat")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Add,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty,
                },
                predicate,
            )?;
        }
        AddInstr::Float32 {
            dst, src_a, src_b, ..
        } => {
            let ty = ScalarType::F32;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, ty, "add.f32")?;
            ctx.check_operand_type(&src_a_typed, ty, "add.f32")?;
            ctx.check_operand_type(&src_b_typed, ty, "add.f32")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Add,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty,
                },
                predicate,
            )?;
        }
        AddInstr::Float32x2 {
            dst, src_a, src_b, ..
        } => {
            let ty = ScalarType::F32x2;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, ty, "add.f32x2")?;
            ctx.check_operand_type(&src_a_typed, ty, "add.f32x2")?;
            ctx.check_operand_type(&src_b_typed, ty, "add.f32x2")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Add,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty,
                },
                predicate,
            )?;
        }
        AddInstr::Float64 {
            dst, src_a, src_b, ..
        } => {
            let ty = ScalarType::F64;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, ty, "add.f64")?;
            ctx.check_operand_type(&src_a_typed, ty, "add.f64")?;
            ctx.check_operand_type(&src_b_typed, ty, "add.f64")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Add,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty,
                },
                predicate,
            )?;
        }
        AddInstr::HalfF16 {
            ty,
            dst,
            src_a,
            src_b,
            ..
        }
        | AddInstr::HalfBf16 {
            ty,
            dst,
            src_a,
            src_b,
            ..
        } => {
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, *ty, "add.f16/bf16")?;
            ctx.check_operand_type(&src_a_typed, *ty, "add.f16/bf16")?;
            ctx.check_operand_type(&src_b_typed, *ty, "add.f16/bf16")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Add,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: *ty,
                },
                predicate,
            )?;
        }
        AddInstr::MixedPrecision {
            dst, src_a, src_b, ..
        } => {
            // Mixed precision add: f32 result from half inputs
            // Source types are f16, destination is f32
            let dst_ty = ScalarType::F32;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            // Destination must be f32-compatible
            ctx.check_dst_type(&dst_typed, dst_ty, "add.f32 (mixed)")?;
            // Sources are f16 - check they're float-compatible (relaxed for mixed precision)
            // Note: We allow f16 sources for this instruction
            ctx.check_operand_type(&src_a_typed, ScalarType::F16, "add.f32 (mixed)")?;
            ctx.check_operand_type(&src_b_typed, ScalarType::F16, "add.f32 (mixed)")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Add,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: dst_ty,
                },
                predicate,
            )?;
        }
    }
    Ok(())
}

fn lower_sub(
    ctx: &mut LoweringContext,
    sub: &SubInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    match sub {
        SubInstr::Integer {
            ty,
            dst,
            src_a,
            src_b,
        } => {
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, *ty, "sub")?;
            ctx.check_operand_type(&src_a_typed, *ty, "sub")?;
            ctx.check_operand_type(&src_b_typed, *ty, "sub")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Sub,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: *ty,
                },
                predicate,
            )?;
        }
        SubInstr::IntegerSat {
            sat: _,
            dst,
            src_a,
            src_b,
        } => {
            let ty = ScalarType::S32;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, ty, "sub.sat")?;
            ctx.check_operand_type(&src_a_typed, ty, "sub.sat")?;
            ctx.check_operand_type(&src_b_typed, ty, "sub.sat")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Sub,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty,
                },
                predicate,
            )?;
        }
        SubInstr::Float32 {
            dst, src_a, src_b, ..
        } => {
            let ty = ScalarType::F32;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, ty, "sub.f32")?;
            ctx.check_operand_type(&src_a_typed, ty, "sub.f32")?;
            ctx.check_operand_type(&src_b_typed, ty, "sub.f32")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Sub,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty,
                },
                predicate,
            )?;
        }
        SubInstr::Float32x2 {
            dst, src_a, src_b, ..
        } => {
            let ty = ScalarType::F32x2;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, ty, "sub.f32x2")?;
            ctx.check_operand_type(&src_a_typed, ty, "sub.f32x2")?;
            ctx.check_operand_type(&src_b_typed, ty, "sub.f32x2")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Sub,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty,
                },
                predicate,
            )?;
        }
        SubInstr::Float64 {
            dst, src_a, src_b, ..
        } => {
            let ty = ScalarType::F64;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, ty, "sub.f64")?;
            ctx.check_operand_type(&src_a_typed, ty, "sub.f64")?;
            ctx.check_operand_type(&src_b_typed, ty, "sub.f64")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Sub,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty,
                },
                predicate,
            )?;
        }
        SubInstr::HalfF16 {
            ty,
            dst,
            src_a,
            src_b,
            ..
        }
        | SubInstr::HalfBf16 {
            ty,
            dst,
            src_a,
            src_b,
            ..
        } => {
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, *ty, "sub.f16/bf16")?;
            ctx.check_operand_type(&src_a_typed, *ty, "sub.f16/bf16")?;
            ctx.check_operand_type(&src_b_typed, *ty, "sub.f16/bf16")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Sub,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: *ty,
                },
                predicate,
            )?;
        }
        SubInstr::MixedPrecision {
            dst, src_a, src_b, ..
        } => {
            // Mixed precision sub: f32 result from half inputs
            let dst_ty = ScalarType::F32;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, dst_ty, "sub.f32 (mixed)")?;
            ctx.check_operand_type(&src_a_typed, ScalarType::F16, "sub.f32 (mixed)")?;
            ctx.check_operand_type(&src_b_typed, ScalarType::F16, "sub.f32 (mixed)")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Sub,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: dst_ty,
                },
                predicate,
            )?;
        }
    }
    Ok(())
}

fn lower_mul(
    ctx: &mut LoweringContext,
    mul: &MulInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    match mul {
        MulInstr::Integer {
            mode,
            ty,
            dst,
            src_a,
            src_b,
        } => {
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            match mode {
                MulMode::Wide => {
                    // For mul.wide, sources are type `ty`, but destination is 2x wider
                    // e.g., mul.wide.s32 produces s64 result
                    let dst_ty = ty.widen().unwrap_or(*ty);
                    ctx.check_dst_type(&dst_typed, dst_ty, "mul.wide")?;
                    ctx.check_operand_type(&src_a_typed, *ty, "mul.wide")?;
                    ctx.check_operand_type(&src_b_typed, *ty, "mul.wide")?;

                    ctx.emit(
                        LoweredInstr::MulWide {
                            dst: dst_typed.reg,
                            src_a: src_a_typed.operand,
                            src_b: src_b_typed.operand,
                            src_ty: *ty,
                        },
                        predicate,
                    )?;
                }
                MulMode::Hi => {
                    ctx.check_dst_type(&dst_typed, *ty, "mul.hi")?;
                    ctx.check_operand_type(&src_a_typed, *ty, "mul.hi")?;
                    ctx.check_operand_type(&src_b_typed, *ty, "mul.hi")?;

                    ctx.emit(
                        LoweredInstr::MulHi {
                            dst: dst_typed.reg,
                            src_a: src_a_typed.operand,
                            src_b: src_b_typed.operand,
                            ty: *ty,
                        },
                        predicate,
                    )?;
                }
                MulMode::Lo => {
                    ctx.check_dst_type(&dst_typed, *ty, "mul.lo")?;
                    ctx.check_operand_type(&src_a_typed, *ty, "mul.lo")?;
                    ctx.check_operand_type(&src_b_typed, *ty, "mul.lo")?;

                    ctx.emit(
                        LoweredInstr::BinOp {
                            op: BinOp::Mul,
                            dst: dst_typed.reg,
                            src_a: src_a_typed.operand,
                            src_b: src_b_typed.operand,
                            ty: *ty,
                        },
                        predicate,
                    )?;
                }
            }
        }
        MulInstr::Float {
            ty,
            dst,
            src_a,
            src_b,
            ..
        } => {
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, *ty, "mul.f")?;
            ctx.check_operand_type(&src_a_typed, *ty, "mul.f")?;
            ctx.check_operand_type(&src_b_typed, *ty, "mul.f")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Mul,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: *ty,
                },
                predicate,
            )?;
        }
    }
    Ok(())
}

fn lower_mad(
    ctx: &mut LoweringContext,
    mad: &MadInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    match mad {
        MadInstr::Integer {
            mode,
            sat: _,
            ty,
            dst,
            src_a,
            src_b,
            src_c,
        } => {
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;
            let src_c_typed = ctx.resolve_operand_typed(src_c)?;

            let instr_name = match mode {
                MulMode::Hi => "mad.hi",
                MulMode::Lo => "mad.lo",
                MulMode::Wide => "mad.wide",
            };

            // For mad.wide, destination is 2x wider
            let dst_ty = if matches!(mode, MulMode::Wide) {
                ty.widen().unwrap_or(*ty)
            } else {
                *ty
            };

            ctx.check_dst_type(&dst_typed, dst_ty, instr_name)?;
            ctx.check_operand_type(&src_a_typed, *ty, instr_name)?;
            ctx.check_operand_type(&src_b_typed, *ty, instr_name)?;
            // src_c is the accumulator, same type as destination
            ctx.check_operand_type(&src_c_typed, dst_ty, instr_name)?;

            ctx.emit(
                LoweredInstr::Mad {
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    src_c: src_c_typed.operand,
                    ty: *ty,
                    mode: ctx.convert_mul_mode(*mode),
                },
                predicate,
            )?;
        }
        MadInstr::Float {
            ty,
            dst,
            src_a,
            src_b,
            src_c,
            ..
        } => {
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;
            let src_c_typed = ctx.resolve_operand_typed(src_c)?;

            ctx.check_dst_type(&dst_typed, *ty, "mad.f")?;
            ctx.check_operand_type(&src_a_typed, *ty, "mad.f")?;
            ctx.check_operand_type(&src_b_typed, *ty, "mad.f")?;
            ctx.check_operand_type(&src_c_typed, *ty, "mad.f")?;

            ctx.emit(
                LoweredInstr::Fma {
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    src_c: src_c_typed.operand,
                    ty: *ty,
                },
                predicate,
            )?;
        }
    }
    Ok(())
}

fn lower_fma(
    ctx: &mut LoweringContext,
    fma: &FmaInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let (ty, instr_name, dst, src_a, src_b, src_c) = match fma {
        FmaInstr::Float32 {
            dst,
            src_a,
            src_b,
            src_c,
            ..
        } => (ScalarType::F32, "fma.rn.f32", dst, src_a, src_b, src_c),
        FmaInstr::Float32x2 {
            dst,
            src_a,
            src_b,
            src_c,
            ..
        } => (ScalarType::F32x2, "fma.rn.f32x2", dst, src_a, src_b, src_c),
        FmaInstr::Float64 {
            dst,
            src_a,
            src_b,
            src_c,
            ..
        } => (ScalarType::F64, "fma.rn.f64", dst, src_a, src_b, src_c),
        FmaInstr::HalfF16Sat {
            ty,
            dst,
            src_a,
            src_b,
            src_c,
            ..
        } => (*ty, "fma.rn.sat.f16", dst, src_a, src_b, src_c),
        FmaInstr::HalfF16Relu {
            ty,
            dst,
            src_a,
            src_b,
            src_c,
            ..
        } => (*ty, "fma.rn.relu.f16", dst, src_a, src_b, src_c),
        FmaInstr::HalfBf16 {
            ty,
            dst,
            src_a,
            src_b,
            src_c,
            ..
        } => (*ty, "fma.rn.bf16", dst, src_a, src_b, src_c),
        FmaInstr::Oob {
            ty,
            dst,
            src_a,
            src_b,
            src_c,
            ..
        } => (*ty, "fma.rn.oob", dst, src_a, src_b, src_c),
        FmaInstr::MixedPrecision {
            dst,
            src_a,
            src_b,
            src_c,
            ..
        } => (
            ScalarType::F32,
            "fma.rn.f32 (mixed)",
            dst,
            src_a,
            src_b,
            src_c,
        ),
    };

    let dst_typed = ctx.resolve_dst_typed(dst)?;
    let src_a_typed = ctx.resolve_operand_typed(src_a)?;
    let src_b_typed = ctx.resolve_operand_typed(src_b)?;
    let src_c_typed = ctx.resolve_operand_typed(src_c)?;

    // For mixed precision FMA, sources are f16, destination is f32
    if matches!(fma, FmaInstr::MixedPrecision { .. }) {
        ctx.check_dst_type(&dst_typed, ScalarType::F32, instr_name)?;
        ctx.check_operand_type(&src_a_typed, ScalarType::F16, instr_name)?;
        ctx.check_operand_type(&src_b_typed, ScalarType::F16, instr_name)?;
        ctx.check_operand_type(&src_c_typed, ScalarType::F32, instr_name)?;
    } else {
        ctx.check_dst_type(&dst_typed, ty, instr_name)?;
        ctx.check_operand_type(&src_a_typed, ty, instr_name)?;
        ctx.check_operand_type(&src_b_typed, ty, instr_name)?;
        ctx.check_operand_type(&src_c_typed, ty, instr_name)?;
    }

    ctx.emit(
        LoweredInstr::Fma {
            dst: dst_typed.reg,
            src_a: src_a_typed.operand,
            src_b: src_b_typed.operand,
            src_c: src_c_typed.operand,
            ty,
        },
        predicate,
    )?;
    Ok(())
}

fn lower_div(
    ctx: &mut LoweringContext,
    div: &DivInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let (ty, instr_name, dst, src_a, src_b) = match div {
        DivInstr::Integer {
            ty,
            dst,
            src_a,
            src_b,
        } => (*ty, "div", dst, src_a, src_b),
        DivInstr::Approx {
            dst, src_a, src_b, ..
        } => (ScalarType::F32, "div.approx.f32", dst, src_a, src_b),
        DivInstr::Full {
            dst, src_a, src_b, ..
        } => (ScalarType::F32, "div.full.f32", dst, src_a, src_b),
        DivInstr::Ieee {
            ty,
            dst,
            src_a,
            src_b,
            ..
        } => (*ty, "div.rn", dst, src_a, src_b),
    };

    let dst_typed = ctx.resolve_dst_typed(dst)?;
    let src_a_typed = ctx.resolve_operand_typed(src_a)?;
    let src_b_typed = ctx.resolve_operand_typed(src_b)?;

    ctx.check_dst_type(&dst_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_a_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_b_typed, ty, instr_name)?;

    ctx.emit(
        LoweredInstr::BinOp {
            op: BinOp::Div,
            dst: dst_typed.reg,
            src_a: src_a_typed.operand,
            src_b: src_b_typed.operand,
            ty,
        },
        predicate,
    )?;
    Ok(())
}

fn lower_neg(
    ctx: &mut LoweringContext,
    neg: &NegInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let (ty, instr_name, dst, src) = match neg {
        NegInstr::Integer { ty, dst, src } => (*ty, "neg", dst, src),
        NegInstr::Float32 { dst, src, .. } => (ScalarType::F32, "neg.f32", dst, src),
        NegInstr::Float64 { dst, src } => (ScalarType::F64, "neg.f64", dst, src),
        NegInstr::HalfF16 { ty, dst, src, .. } => (*ty, "neg.f16", dst, src),
        NegInstr::HalfBf16 { ty, dst, src } => (*ty, "neg.bf16", dst, src),
    };

    let dst_typed = ctx.resolve_dst_typed(dst)?;
    let src_typed = ctx.resolve_operand_typed(src)?;

    ctx.check_dst_type(&dst_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_typed, ty, instr_name)?;

    ctx.emit(
        LoweredInstr::UnaryOp {
            op: UnaryOp::Neg,
            dst: dst_typed.reg,
            src: src_typed.operand,
            ty,
        },
        predicate,
    )?;
    Ok(())
}

fn lower_abs(
    ctx: &mut LoweringContext,
    abs: &AbsInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let (ty, instr_name, dst, src) = match abs {
        AbsInstr::Integer { ty, dst, src } => (*ty, "abs", dst, src),
        AbsInstr::Float32 { dst, src, .. } => (ScalarType::F32, "abs.f32", dst, src),
        AbsInstr::Float64 { dst, src } => (ScalarType::F64, "abs.f64", dst, src),
        AbsInstr::HalfF16 { ty, dst, src, .. } => (*ty, "abs.f16", dst, src),
        AbsInstr::HalfBf16 { ty, dst, src } => (*ty, "abs.bf16", dst, src),
    };

    let dst_typed = ctx.resolve_dst_typed(dst)?;
    let src_typed = ctx.resolve_operand_typed(src)?;

    ctx.check_dst_type(&dst_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_typed, ty, instr_name)?;

    ctx.emit(
        LoweredInstr::UnaryOp {
            op: UnaryOp::Abs,
            dst: dst_typed.reg,
            src: src_typed.operand,
            ty,
        },
        predicate,
    )?;
    Ok(())
}

/// Shared lowering for float unary ops (rcp, sqrt, rsqrt, ...)
fn lower_float_unary(
    ctx: &mut LoweringContext,
    op: UnaryOp,
    instr_name: &str,
    ty: ScalarType,
    dst: &AstOperand,
    src: &AstOperand,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let dst_typed = ctx.resolve_dst_typed(dst)?;
    let src_typed = ctx.resolve_operand_typed(src)?;

    ctx.check_dst_type(&dst_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_typed, ty, instr_name)?;

    ctx.emit(
        LoweredInstr::UnaryOp {
            op,
            dst: dst_typed.reg,
            src: src_typed.operand,
            ty,
        },
        predicate,
    )?;
    Ok(())
}

fn lower_min(
    ctx: &mut LoweringContext,
    min: &MinInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let (ty, instr_name, dst, src_a, src_b) = match min {
        MinInstr::Integer {
            ty,
            dst,
            src_a,
            src_b,
        } => (*ty, "min", dst, src_a, src_b),
        MinInstr::IntegerRelu {
            ty,
            dst,
            src_a,
            src_b,
            ..
        } => (*ty, "min.relu", dst, src_a, src_b),
        MinInstr::Float32 {
            dst, src_a, src_b, ..
        } => (ScalarType::F32, "min.f32", dst, src_a, src_b),
        MinInstr::Float32Acc {
            dst, src_a, src_b, ..
        } => (ScalarType::F32, "min.f32", dst, src_a, src_b),
        MinInstr::Float64 { dst, src_a, src_b } => (ScalarType::F64, "min.f64", dst, src_a, src_b),
        MinInstr::HalfF16 {
            ty,
            dst,
            src_a,
            src_b,
            ..
        } => (*ty, "min.f16", dst, src_a, src_b),
        MinInstr::HalfBf16 {
            ty,
            dst,
            src_a,
            src_b,
            ..
        } => (*ty, "min.bf16", dst, src_a, src_b),
    };

    let dst_typed = ctx.resolve_dst_typed(dst)?;
    let src_a_typed = ctx.resolve_operand_typed(src_a)?;
    let src_b_typed = ctx.resolve_operand_typed(src_b)?;

    ctx.check_dst_type(&dst_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_a_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_b_typed, ty, instr_name)?;

    ctx.emit(
        LoweredInstr::BinOp {
            op: BinOp::Min,
            dst: dst_typed.reg,
            src_a: src_a_typed.operand,
            src_b: src_b_typed.operand,
            ty,
        },
        predicate,
    )?;
    Ok(())
}

fn lower_max(
    ctx: &mut LoweringContext,
    max: &MaxInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let (ty, instr_name, dst, src_a, src_b) = match max {
        MaxInstr::Integer {
            ty,
            dst,
            src_a,
            src_b,
        } => (*ty, "max", dst, src_a, src_b),
        MaxInstr::IntegerRelu {
            ty,
            dst,
            src_a,
            src_b,
            ..
        } => (*ty, "max.relu", dst, src_a, src_b),
        MaxInstr::Float32 {
            dst, src_a, src_b, ..
        } => (ScalarType::F32, "max.f32", dst, src_a, src_b),
        MaxInstr::Float32Acc {
            dst, src_a, src_b, ..
        } => (ScalarType::F32, "max.f32", dst, src_a, src_b),
        MaxInstr::Float64 { dst, src_a, src_b } => (ScalarType::F64, "max.f64", dst, src_a, src_b),
        MaxInstr::HalfF16 {
            ty,
            dst,
            src_a,
            src_b,
            ..
        } => (*ty, "max.f16", dst, src_a, src_b),
        MaxInstr::HalfBf16 {
            ty,
            dst,
            src_a,
            src_b,
            ..
        } => (*ty, "max.bf16", dst, src_a, src_b),
    };

    let dst_typed = ctx.resolve_dst_typed(dst)?;
    let src_a_typed = ctx.resolve_operand_typed(src_a)?;
    let src_b_typed = ctx.resolve_operand_typed(src_b)?;

    ctx.check_dst_type(&dst_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_a_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_b_typed, ty, instr_name)?;

    ctx.emit(
        LoweredInstr::BinOp {
            op: BinOp::Max,
            dst: dst_typed.reg,
            src_a: src_a_typed.operand,
            src_b: src_b_typed.operand,
            ty,
        },
        predicate,
    )?;
    Ok(())
}

fn lower_setp(
    ctx: &mut LoweringContext,
    setp: &SetpInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    match setp {
        SetpInstr::Simple {
            cmp_op,
            ty,
            dst_p,
            dst_q: _, // Second predicate destination ignored for now
            src_a,
            src_b,
            ..
        } => {
            // Destination is a predicate register
            let dst_typed = ctx.resolve_dst_typed(dst_p)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            // Check destination is pred type
            ctx.check_dst_type(&dst_typed, ScalarType::Pred, "setp")?;
            // Check sources match instruction type
            ctx.check_operand_type(&src_a_typed, *ty, "setp")?;
            ctx.check_operand_type(&src_b_typed, *ty, "setp")?;

            ctx.emit(
                LoweredInstr::Setp {
                    cmp: ctx.convert_cmp_op(*cmp_op),
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: *ty,
                },
                predicate,
            )?;
        }
        SetpInstr::WithBoolOp {
            cmp_op,
            ty,
            dst_p,
            dst_q: _, // Second predicate destination ignored for now
            src_a,
            src_b,
            ..
        } => {
            // Boolean combination not yet implemented
            let dst_typed = ctx.resolve_dst_typed(dst_p)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, ScalarType::Pred, "setp")?;
            ctx.check_operand_type(&src_a_typed, *ty, "setp")?;
            ctx.check_operand_type(&src_b_typed, *ty, "setp")?;

            ctx.emit(
                LoweredInstr::Setp {
                    cmp: ctx.convert_cmp_op(*cmp_op),
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: *ty,
                },
                predicate,
            )?;
        }
    }
    Ok(())
}

fn lower_load(
    ctx: &mut LoweringContext,
    ld: &LdInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    // Param-space loads read either a kernel parameter or a block-scope
    // `.param` slot (the callseq idiom); both are handled symbolically
    // rather than as memory accesses.
    if ld.space == Some(StateSpace::Param) {
        return lower_param_load(ctx, ld, predicate);
    }

    // Resolve the address operand
    let (base, offset) = match &ld.addr {
        AstOperand::Address(addr) => {
            let base = ctx.resolve_address(addr)?;
            let offset = ctx.get_address_offset(addr);
            (base, offset)
        }
        _ => {
            let base = ctx.resolve_operand(&ld.addr)?;
            (base, 0)
        }
    };

    let space = ctx.convert_space(ld.space);
    let instr_name = format!("ld.{:?}", ld.ty);

    // Check if destination is a vector (e.g., {%f1, %f2, %f3, %f4})
    match &ld.dst {
        AstOperand::Vector(_) => {
            // Vector load - emit LoadVec
            // For vectors, we'd need to check each element, but for now just resolve
            let dst_regs = ctx.resolve_dst_vector(&ld.dst)?;
            ctx.emit(
                LoweredInstr::LoadVec {
                    dst: dst_regs,
                    space,
                    base,
                    offset,
                    ty: ld.ty,
                },
                predicate,
            )?;
        }
        _ => {
            // Single register load - use relaxed type checking (PTX 9.4.1)
            // Destination can be wider than instruction type (value is extended)
            let dst_typed = ctx.resolve_dst_typed(&ld.dst)?;
            ctx.check_dst_type_relaxed(&dst_typed, ld.ty, &instr_name)?;

            ctx.emit(
                LoweredInstr::Load {
                    dst: dst_typed.reg,
                    space,
                    base,
                    offset,
                    ty: ld.ty,
                },
                predicate,
            )?;
        }
    }
    Ok(())
}

fn lower_store(
    ctx: &mut LoweringContext,
    st: &StInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    // Param-space stores write a block-scope `.param` slot (callseq idiom)
    if st.space == Some(StateSpace::Param) {
        return lower_param_store(ctx, st, predicate);
    }

    // Resolve the address operand
    let (base, offset) = match &st.addr {
        AstOperand::Address(addr) => {
            let base = ctx.resolve_address(addr)?;
            let offset = ctx.get_address_offset(addr);
            (base, offset)
        }
        _ => {
            let base = ctx.resolve_operand(&st.addr)?;
            (base, 0)
        }
    };

    let space = ctx.convert_space(st.space);
    let instr_name = format!("st.{:?}", st.ty);

    // Check if source is a vector (e.g., {%f1, %f2, %f3, %f4})
    if let AstOperand::Vector(_) = &st.src {
        // Vector store - emit StoreVec
        // For vectors, we'd need to check each element, but for now just resolve
        let src_regs = ctx.resolve_dst_vector(&st.src)?;
        ctx.emit(
            LoweredInstr::StoreVec {
                space,
                base,
                offset,
                src: src_regs,
                ty: st.ty,
            },
            predicate,
        )?;
        return Ok(());
    }

    // Single register store - use relaxed type checking (PTX 9.4.1)
    // Source can be wider than instruction type (value is truncated)
    let src_typed = ctx.resolve_operand_typed(&st.src)?;
    ctx.check_operand_type_relaxed(&src_typed, st.ty, &instr_name)?;

    ctx.emit(
        LoweredInstr::Store {
            space,
            base,
            offset,
            src: src_typed.operand,
            ty: st.ty,
        },
        predicate,
    )?;
    Ok(())
}

/// Extract the `(symbol, offset)` of a param-space address like `[param0+0]`.
fn param_addr_symbol(ctx: &LoweringContext, addr: &AstOperand) -> LowerResult<(String, i64)> {
    if let AstOperand::Address(addr) = addr
        && let AddressBase::Symbol(name) = &addr.base
    {
        return Ok((name.to_string(), ctx.get_address_offset(addr)));
    }
    Err(LowerError::UnsupportedInstruction {
        instruction: "param-space access".to_string(),
        reason: Some("param accesses must use a symbol base address".to_string()),
    })
}

/// Lower `ld.param`: kernel parameters become `LoadParam`; block-scope
/// `.param` slots (callseq idiom) collapse to the deferred call result.
fn lower_param_load(
    ctx: &mut LoweringContext,
    ld: &LdInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let (name, offset) = param_addr_symbol(ctx, &ld.addr)?;
    let dst = ctx.resolve_dst(&ld.dst)?;

    // Block-scope `.param` slot (callseq idiom)?
    if let Some(slot) = ctx.local_params.get(&name).copied() {
        if offset != 0 {
            return Err(LowerError::UnsupportedInstruction {
                instruction: format!("ld.param [{}+{}]", name, offset),
                reason: Some(".param slots are scalar; nonzero offsets unsupported".to_string()),
            });
        }
        if predicate.is_some() {
            return Err(LowerError::UnsupportedInstruction {
                instruction: format!("ld.param [{}]", name),
                reason: Some("predicated callseq accesses are not supported".to_string()),
            });
        }
        return match slot {
            LocalParamSlot::PendingExp(src) => ctx.emit(
                LoweredInstr::UnaryOp {
                    op: UnaryOp::Exp,
                    dst,
                    src,
                    ty: ld.ty,
                },
                None,
            ),
            LocalParamSlot::Stored(src) => ctx.emit(
                LoweredInstr::Mov {
                    dst,
                    src,
                    ty: ld.ty,
                },
                None,
            ),
            LocalParamSlot::Empty => Err(LowerError::UnsupportedInstruction {
                instruction: format!("ld.param [{}]", name),
                reason: Some("read of a .param slot that was never written".to_string()),
            }),
        };
    }

    // Kernel parameter
    let Some(info) = ctx.symbols.get_param(&name) else {
        return Err(LowerError::UndefinedSymbol { name });
    };
    let param_id = info.id;
    if offset != 0 {
        return Err(LowerError::UnsupportedInstruction {
            instruction: format!("ld.param [{}+{}]", name, offset),
            reason: Some("param loads with nonzero offsets are not supported".to_string()),
        });
    }
    ctx.emit(LoweredInstr::LoadParam { dst, param_id }, predicate)
}

/// Lower `st.param`: records the stored operand in a block-scope `.param`
/// slot for the following `call`. Emits no instruction.
fn lower_param_store(
    ctx: &mut LoweringContext,
    st: &StInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    if predicate.is_some() {
        return Err(LowerError::UnsupportedInstruction {
            instruction: "st.param".to_string(),
            reason: Some("predicated callseq accesses are not supported".to_string()),
        });
    }
    let (name, offset) = param_addr_symbol(ctx, &st.addr)?;
    if offset != 0 {
        return Err(LowerError::UnsupportedInstruction {
            instruction: format!("st.param [{}+{}]", name, offset),
            reason: Some(".param slots are scalar; nonzero offsets unsupported".to_string()),
        });
    }
    if !ctx.local_params.contains_key(&name) {
        return Err(LowerError::UndefinedSymbol { name });
    }
    let src = ctx.resolve_operand(&st.src)?;
    ctx.local_params.insert(name, LocalParamSlot::Stored(src));
    Ok(())
}

/// Lower a direct call. Only `__symexpf` (the paper's hook for symbolic exp)
/// is supported. The call emits nothing; it marks the retval slot as the
/// pending exp of the argument, which the consuming `ld.param` materializes.
fn lower_call(
    ctx: &mut LoweringContext,
    call: &CallInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let target_name = match &call.target {
        AstOperand::Ident(name) => name.to_string(),
        other => {
            return Err(LowerError::UnsupportedInstruction {
                instruction: format!("call via {:?}", other),
                reason: Some("only direct calls are supported".to_string()),
            });
        }
    };
    if target_name != "__symexpf" {
        return Err(LowerError::UnsupportedInstruction {
            instruction: format!("call {}", target_name),
            reason: Some("only calls to the __symexpf symbolic-exp hook are supported".to_string()),
        });
    }
    if predicate.is_some() {
        return Err(LowerError::UnsupportedInstruction {
            instruction: format!("call {}", target_name),
            reason: Some("predicated calls are not supported".to_string()),
        });
    }

    let [arg] = call.arguments.as_slice() else {
        return Err(LowerError::UnsupportedInstruction {
            instruction: format!("call {}", target_name),
            reason: Some("__symexpf takes exactly one argument".to_string()),
        });
    };
    let [ret] = call.return_operands.as_slice() else {
        return Err(LowerError::UnsupportedInstruction {
            instruction: format!("call {}", target_name),
            reason: Some("__symexpf returns exactly one value".to_string()),
        });
    };
    let (AstOperand::Ident(arg_name), AstOperand::Ident(ret_name)) = (arg, ret) else {
        return Err(LowerError::UnsupportedInstruction {
            instruction: format!("call {}", target_name),
            reason: Some("call operands must be .param slot names".to_string()),
        });
    };

    let arg_name = arg_name.to_string();
    let src = match ctx.local_params.get(&arg_name) {
        Some(LocalParamSlot::Stored(op)) => *op,
        Some(_) => {
            return Err(LowerError::UnsupportedInstruction {
                instruction: format!("call {}", target_name),
                reason: Some(format!(
                    ".param slot '{}' was not written before the call",
                    arg_name
                )),
            });
        }
        None => return Err(LowerError::UndefinedSymbol { name: arg_name }),
    };

    let ret_name = ret_name.to_string();
    if !ctx.local_params.contains_key(&ret_name) {
        return Err(LowerError::UndefinedSymbol { name: ret_name });
    }
    ctx.local_params
        .insert(ret_name, LocalParamSlot::PendingExp(src));
    Ok(())
}

fn lower_cvt(
    ctx: &mut LoweringContext,
    cvt: &CvtInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    match cvt {
        CvtInstr::Standard {
            dst_type,
            src_type,
            dst,
            src,
            ..
        } => {
            let dst = ctx.resolve_dst(dst)?;
            let src = ctx.resolve_operand(src)?;
            ctx.emit(
                LoweredInstr::Cvt {
                    dst,
                    src,
                    dst_ty: *dst_type,
                    src_ty: *src_type,
                },
                predicate,
            )?;
        }
        CvtInstr::Pack {
            dst_type,
            src_type,
            dst,
            src_a,
            src_b: _, // Second source operand ignored for simplified version
            ..
        } => {
            // Pack conversion - emit simplified version
            let dst = ctx.resolve_dst(dst)?;
            let src = ctx.resolve_operand(src_a)?;
            ctx.emit(
                LoweredInstr::Cvt {
                    dst,
                    src,
                    dst_ty: *dst_type,
                    src_ty: *src_type,
                },
                predicate,
            )?;
        }
    }
    Ok(())
}

fn lower_shfl_sync(
    ctx: &mut LoweringContext,
    shfl: &ShflSyncInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    // The destination may be written `d|p` where `p` receives the validity
    // predicate; the parser surfaces that as a PredicatePair operand.
    let (dst, pair_pred) = match &shfl.dst {
        AstOperand::PredicatePair(d, p) => (
            ctx.resolve_register(&d.to_string())?,
            Some(ctx.resolve_register(&p.to_string())?),
        ),
        other => (ctx.resolve_dst(other)?, None),
    };
    let dst_pred = match (&shfl.dst_pred, pair_pred) {
        (Some(op), None) => Some(ctx.resolve_dst(op)?),
        (None, p) => p,
        (Some(_), Some(_)) => {
            return Err(LowerError::UnsupportedInstruction {
                instruction: "shfl.sync".to_string(),
                reason: Some("conflicting predicate destinations".to_string()),
            });
        }
    };
    let src = ctx.resolve_operand(&shfl.src)?;
    let offset_or_lane = ctx.resolve_operand(&shfl.src_b)?;
    let clamp = ctx.resolve_operand(&shfl.src_c)?;
    let membermask = ctx.resolve_operand(&shfl.membermask)?;

    ctx.emit(
        LoweredInstr::ShflSync {
            mode: ctx.convert_shfl_mode(shfl.mode),
            dst,
            dst_pred,
            src,
            offset_or_lane,
            clamp,
            membermask,
        },
        predicate,
    )?;
    Ok(())
}

fn lower_branch(
    ctx: &mut LoweringContext,
    bra: &BraInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    // Get target label
    let target_name = match &bra.target {
        AstOperand::Symbol(name) => name.to_string(),
        AstOperand::Ident(name) => name.to_string(), // Labels are parsed as identifiers
        _ => {
            return Err(LowerError::InvalidBranchTarget {
                target: format!("{:?}", bra.target),
            });
        }
    };

    // Try to resolve the label
    let target = match ctx.symbols.resolve_label(&target_name) {
        Some(pc) => pc,
        None => {
            // Forward reference - record it and emit placeholder
            ctx.record_forward_ref(&target_name);
            InstrId::from_index(0) // Will be patched later
        }
    };

    ctx.emit(LoweredInstr::Bra { target }, predicate)?;
    Ok(())
}

fn lower_bar(
    ctx: &mut LoweringContext,
    bar: &BarInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    // Get barrier ID from first operand
    let barrier_id = match bar.operands.first() {
        Some(AstOperand::ImmInt(v)) => *v as u32,
        Some(AstOperand::ImmUInt(v)) => *v as u32,
        _ => 0, // Default barrier
    };

    match bar.mode {
        BarMode::Sync => {
            if bar.operands.len() > 1 {
                // bar.sync with thread count
                let thread_count = ctx.resolve_operand(&bar.operands[1])?;
                ctx.emit(
                    LoweredInstr::BarSyncCount {
                        barrier_id,
                        thread_count,
                    },
                    predicate,
                )?;
            } else {
                ctx.emit(LoweredInstr::BarSync { barrier_id }, predicate)?;
            }
        }
        BarMode::Arrive => {
            // bar.arrive - just emit sync for now
            ctx.emit(LoweredInstr::BarSync { barrier_id }, predicate)?;
        }
        BarMode::Red => {
            // bar.red - reduction barrier, emit sync
            ctx.emit(LoweredInstr::BarSync { barrier_id }, predicate)?;
        }
    }
    Ok(())
}

/// Lower ld.global.nc (non-coherent global load)
/// Treat as regular global load - we don't model cache coherence
fn lower_ld_global_nc(
    ctx: &mut LoweringContext,
    modifiers: &[DottedIdent],
    operands: &[AstOperand],
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    // Parse modifiers to get type
    // Modifiers are like ["v4", "u32"] or just ["u32"]
    // Vector width is inferred from the destination operand
    let mut elem_type = ScalarType::U32; // default

    for modifier in modifiers {
        let mod_ascii = modifier.to_ascii_string();
        // Skip vector modifier (v2, v4) - we determine vector from operand
        if !mod_ascii.as_bytes().starts_with(b"v")
            && let Some(ty) = ScalarType::from_ascii(&mod_ascii)
        {
            elem_type = ty;
        }
    }

    // Operands: [destination, address]
    if operands.len() < 2 {
        return Err(LowerError::InvalidOperand {
            instruction: "ld.global.nc".to_string(),
            operand: format!("{:?}", operands),
            reason: "expected destination and address operands",
        });
    }

    let dst_operand = &operands[0];
    let addr_operand = &operands[1];

    // Resolve address
    let (base, offset) = match addr_operand {
        AstOperand::Address(addr) => {
            let base = ctx.resolve_address(addr)?;
            let offset = ctx.get_address_offset(addr);
            (base, offset)
        }
        _ => {
            let base = ctx.resolve_operand(addr_operand)?;
            (base, 0)
        }
    };

    // Check if destination is a vector
    match dst_operand {
        AstOperand::Vector(_) => {
            // Vector load
            let dst_regs = ctx.resolve_dst_vector(dst_operand)?;
            ctx.emit(
                LoweredInstr::LoadVec {
                    dst: dst_regs,
                    space: MemSpace::Global,
                    base,
                    offset,
                    ty: elem_type,
                },
                predicate,
            )?;
        }
        _ => {
            // Single register load
            let dst = ctx.resolve_dst(dst_operand)?;
            ctx.emit(
                LoweredInstr::Load {
                    dst,
                    space: MemSpace::Global,
                    base,
                    offset,
                    ty: elem_type,
                },
                predicate,
            )?;
        }
    }

    Ok(())
}

// =========================================================================
// Tensor Core Lowering Helpers
// =========================================================================

/// Parse a `ScalarType` from a modifier string (e.g., "f16", "f32", "b16").
fn parse_scalar_type_modifier(modifier: &DottedIdent) -> Option<ScalarType> {
    let ascii = modifier.to_ascii_string();
    ScalarType::from_ascii(&ascii)
}

/// Lower `ldmatrix.sync.aligned[.trans].x{1,2,4}.m8n8.shared.b16`
fn lower_ldmatrix(
    ctx: &mut LoweringContext,
    modifiers: &[DottedIdent],
    operands: &[AstOperand],
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let mut trans = false;
    let mut num: Option<u32> = None;

    for modifier in modifiers {
        let s = modifier.to_string();
        match s.as_str() {
            "trans" => trans = true,
            "x1" => num = Some(1),
            "x2" => num = Some(2),
            "x4" => num = Some(4),
            _ => {}
        }
    }

    let num = num.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "ldmatrix".to_string(),
        reason: Some("missing x1/x2/x4 modifier".to_string()),
    })?;

    if operands.len() < 2 {
        return Err(LowerError::InvalidOperand {
            instruction: "ldmatrix".to_string(),
            operand: format!("{:?}", operands),
            reason: "expected destination vector and address operands",
        });
    }

    let dst = ctx.resolve_dst_vector(&operands[0])?;
    let addr = ctx.resolve_operand(&operands[1])?;

    ctx.emit(
        LoweredInstr::Ldmatrix {
            dst,
            addr,
            num,
            trans,
        },
        predicate,
    )?;
    Ok(())
}

/// Lower `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
fn lower_mma(
    ctx: &mut LoweringContext,
    modifiers: &[DottedIdent],
    operands: &[AstOperand],
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let mut shape: Option<MmaShape> = None;
    let mut layouts: Vec<MmaLayout> = Vec::new();
    let mut types: Vec<ScalarType> = Vec::new();

    for modifier in modifiers {
        let s = modifier.to_string();
        if let Some(sh) = MmaShape::parse(&s) {
            shape = Some(sh);
        } else if let Some(layout) = MmaLayout::parse(&s) {
            layouts.push(layout);
        } else if let Some(ty) = parse_scalar_type_modifier(modifier) {
            types.push(ty);
        }
    }

    let shape = shape.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "mma".to_string(),
        reason: Some("missing shape modifier (e.g., m16n8k16)".to_string()),
    })?;

    if layouts.len() < 2 {
        return Err(LowerError::UnsupportedInstruction {
            instruction: "mma".to_string(),
            reason: Some(format!(
                "expected 2 layout modifiers (row/col), found {}",
                layouts.len()
            )),
        });
    }

    if types.len() < 4 {
        return Err(LowerError::UnsupportedInstruction {
            instruction: "mma".to_string(),
            reason: Some(format!(
                "expected 4 type modifiers (d_type, a_type, b_type, c_type), found {}",
                types.len()
            )),
        });
    }

    if operands.len() < 4 {
        return Err(LowerError::InvalidOperand {
            instruction: "mma".to_string(),
            operand: format!("{:?}", operands),
            reason: "expected 4 vector operands (dst, src_a, src_b, src_c)",
        });
    }

    let dst = ctx.resolve_dst_vector(&operands[0])?;
    let src_a = ctx.resolve_dst_vector(&operands[1])?;
    let src_b = ctx.resolve_dst_vector(&operands[2])?;
    let src_c = ctx.resolve_dst_vector(&operands[3])?;

    let a_layout = layouts[0];
    let b_layout = layouts[1];
    // PTX syntax order: d_type, a_type, b_type, c_type
    let d_type = types[0];
    let a_type = types[1];
    let b_type = types[2];
    let c_type = types[3];

    ctx.emit(
        LoweredInstr::Mma {
            shape,
            dst,
            src_a,
            src_b,
            src_c,
            a_layout,
            b_layout,
            a_type,
            b_type,
            d_type,
            c_type,
        },
        predicate,
    )?;
    Ok(())
}

/// Lower `wmma.load.{a,b,c}.sync.aligned.{row,col}.m16n16k16[.shared].f16`
fn lower_wmma_load(
    ctx: &mut LoweringContext,
    modifiers: &[DottedIdent],
    operands: &[AstOperand],
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let mut operand_kind: Option<MmaOperand> = None;
    let mut layout: Option<MmaLayout> = None;
    let mut shape: Option<MmaShape> = None;
    let mut elem_type: Option<ScalarType> = None;
    let mut space = MemSpace::Global; // PTX default: generic (we treat as global)

    for modifier in modifiers {
        let s = modifier.to_string();
        if s == "shared" {
            space = MemSpace::Shared;
        } else if let Some(op) = MmaOperand::parse(&s) {
            operand_kind = Some(op);
        } else if let Some(l) = MmaLayout::parse(&s) {
            layout = Some(l);
        } else if let Some(sh) = MmaShape::parse(&s) {
            shape = Some(sh);
        } else if let Some(ty) = parse_scalar_type_modifier(modifier) {
            elem_type = Some(ty);
        }
    }

    let operand_kind = operand_kind.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "wmma.load".to_string(),
        reason: Some("missing operand modifier (a, b, or c)".to_string()),
    })?;

    let layout = layout.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "wmma.load".to_string(),
        reason: Some("missing layout modifier (row or col)".to_string()),
    })?;

    let shape = shape.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "wmma.load".to_string(),
        reason: Some("missing shape modifier (e.g., m16n16k16)".to_string()),
    })?;

    let elem_type = elem_type.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "wmma.load".to_string(),
        reason: Some("missing element type modifier (e.g., f16)".to_string()),
    })?;

    if operands.len() < 3 {
        return Err(LowerError::InvalidOperand {
            instruction: "wmma.load".to_string(),
            operand: format!("{:?}", operands),
            reason: "expected destination vector, address, and stride operands",
        });
    }

    let dst = ctx.resolve_dst_vector(&operands[0])?;
    let addr = ctx.resolve_operand(&operands[1])?;
    let stride = ctx.resolve_operand(&operands[2])?;

    ctx.emit(
        LoweredInstr::WmmaLoad {
            operand: operand_kind,
            shape,
            layout,
            dst,
            addr,
            stride,
            elem_type,
            space,
        },
        predicate,
    )?;
    Ok(())
}

/// Lower `wmma.store.d.sync.aligned.{row,col}.m16n16k16[.shared].f32`
fn lower_wmma_store(
    ctx: &mut LoweringContext,
    modifiers: &[DottedIdent],
    operands: &[AstOperand],
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let mut layout: Option<MmaLayout> = None;
    let mut shape: Option<MmaShape> = None;
    let mut elem_type: Option<ScalarType> = None;
    let mut space = MemSpace::Global; // PTX default: generic (we treat as global)

    for modifier in modifiers {
        let s = modifier.to_string();
        if s == "shared" {
            space = MemSpace::Shared;
        } else if let Some(l) = MmaLayout::parse(&s) {
            layout = Some(l);
        } else if let Some(sh) = MmaShape::parse(&s) {
            shape = Some(sh);
        } else if let Some(ty) = parse_scalar_type_modifier(modifier) {
            elem_type = Some(ty);
        }
    }

    let layout = layout.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "wmma.store".to_string(),
        reason: Some("missing layout modifier (row or col)".to_string()),
    })?;

    let shape = shape.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "wmma.store".to_string(),
        reason: Some("missing shape modifier (e.g., m16n16k16)".to_string()),
    })?;

    let elem_type = elem_type.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "wmma.store".to_string(),
        reason: Some("missing element type modifier (e.g., f32)".to_string()),
    })?;

    if operands.len() < 3 {
        return Err(LowerError::InvalidOperand {
            instruction: "wmma.store".to_string(),
            operand: format!("{:?}", operands),
            reason: "expected address, source vector, and stride operands",
        });
    }

    let addr = ctx.resolve_operand(&operands[0])?;
    let src = ctx.resolve_dst_vector(&operands[1])?;
    let stride = ctx.resolve_operand(&operands[2])?;

    ctx.emit(
        LoweredInstr::WmmaStore {
            shape,
            layout,
            src,
            addr,
            stride,
            elem_type,
            space,
        },
        predicate,
    )?;
    Ok(())
}

/// Lower `wmma.mma.sync.aligned.{row}.{row}.m16n16k16.f32.f32`
fn lower_wmma_mma(
    ctx: &mut LoweringContext,
    modifiers: &[DottedIdent],
    operands: &[AstOperand],
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let mut layouts: Vec<MmaLayout> = Vec::new();
    let mut shape: Option<MmaShape> = None;
    let mut types: Vec<ScalarType> = Vec::new();

    for modifier in modifiers {
        let s = modifier.to_string();
        if let Some(layout) = MmaLayout::parse(&s) {
            layouts.push(layout);
        } else if let Some(sh) = MmaShape::parse(&s) {
            shape = Some(sh);
        } else if let Some(ty) = parse_scalar_type_modifier(modifier) {
            types.push(ty);
        }
    }

    let shape = shape.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "wmma.mma".to_string(),
        reason: Some("missing shape modifier (e.g., m16n16k16)".to_string()),
    })?;

    if layouts.len() < 2 {
        return Err(LowerError::UnsupportedInstruction {
            instruction: "wmma.mma".to_string(),
            reason: Some(format!(
                "expected 2 layout modifiers (row/col), found {}",
                layouts.len()
            )),
        });
    }

    if types.len() < 2 {
        return Err(LowerError::UnsupportedInstruction {
            instruction: "wmma.mma".to_string(),
            reason: Some(format!(
                "expected 2 type modifiers (d_type, c_type), found {}",
                types.len()
            )),
        });
    }

    if operands.len() < 4 {
        return Err(LowerError::InvalidOperand {
            instruction: "wmma.mma".to_string(),
            operand: format!("{:?}", operands),
            reason: "expected 4 vector operands (dst, src_a, src_b, src_c)",
        });
    }

    let dst = ctx.resolve_dst_vector(&operands[0])?;
    let src_a = ctx.resolve_dst_vector(&operands[1])?;
    let src_b = ctx.resolve_dst_vector(&operands[2])?;
    let src_c = ctx.resolve_dst_vector(&operands[3])?;

    let a_layout = layouts[0];
    let b_layout = layouts[1];
    let d_type = types[0];
    let c_type = types[1];

    ctx.emit(
        LoweredInstr::WmmaMma {
            shape,
            dst,
            src_a,
            src_b,
            src_c,
            a_layout,
            b_layout,
            d_type,
            c_type,
        },
        predicate,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use volta_frontend::ascii::AsciiString;

    /// Helper to convert &str to AsciiString for tests
    fn ascii(s: &str) -> AsciiString {
        AsciiString::try_from(s.to_string()).unwrap()
    }

    #[test]
    fn test_lowering_context_creation() {
        let ctx = LoweringContext::new();
        assert_eq!(ctx.current_pc(), InstrId::from_index(0));
        assert!(ctx.instructions.is_empty());
    }

    #[test]
    fn test_label_recording() {
        let mut ctx = LoweringContext::new();
        ctx.record_label("LOOP", Some(Span(0, 4)));
        ctx.emit(LoweredInstr::Nop, None).unwrap();

        assert_eq!(
            ctx.symbols.resolve_label("LOOP"),
            Some(InstrId::from_index(0))
        );
    }

    #[test]
    fn test_special_register_from_name() {
        // Test that special registers can be resolved from their names
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
        assert!(SpecialRegKind::from_name("unknown").is_none());
    }

    // =========================================================================
    // Type Checking Tests
    // =========================================================================

    /// Helper to create a context with declared registers
    fn ctx_with_registers(regs: &[(&str, ScalarType)]) -> LoweringContext {
        let mut ctx = LoweringContext::new();
        for (name, ty) in regs {
            ctx.symbols.declare_register(name, *ty, 1).unwrap();
        }
        ctx
    }

    #[test]
    fn test_type_check_exact_match() {
        // Register declared as u32, used with add.u32 - should succeed
        let ctx = ctx_with_registers(&[("%r0", ScalarType::U32)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_dst_type(&dst, ScalarType::U32, "add.u32");
        assert!(result.is_ok());
    }

    #[test]
    fn test_type_check_signed_unsigned_compatible() {
        // Register declared as s32, used with add.u32 - should succeed (compatible)
        let ctx = ctx_with_registers(&[("%r0", ScalarType::S32)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_dst_type(&dst, ScalarType::U32, "add.u32");
        assert!(
            result.is_ok(),
            "s32 should be compatible with u32 instruction"
        );
    }

    #[test]
    fn test_type_check_bits_compatible_with_any() {
        // Register declared as b32, used with add.f32 - should succeed (bits compatible)
        let ctx = ctx_with_registers(&[("%r0", ScalarType::B32)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_dst_type(&dst, ScalarType::F32, "add.f32");
        assert!(
            result.is_ok(),
            "b32 should be compatible with f32 instruction"
        );
    }

    #[test]
    fn test_type_check_float_int_incompatible() {
        // Register declared as f32, used with add.u32 - should FAIL
        let ctx = ctx_with_registers(&[("%f0", ScalarType::F32)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%f0")))
            .unwrap();
        let result = ctx.check_dst_type(&dst, ScalarType::U32, "add.u32");
        assert!(
            result.is_err(),
            "f32 should NOT be compatible with u32 instruction"
        );

        // Verify error message contains useful information
        let err = result.unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("Type error"),
            "Error should mention type error"
        );
        assert!(msg.contains("%f0"), "Error should mention register name");
    }

    #[test]
    fn test_type_check_size_mismatch() {
        // Register declared as u64, used with add.u32 - should FAIL (size mismatch)
        let ctx = ctx_with_registers(&[("%r0", ScalarType::U64)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_dst_type(&dst, ScalarType::U32, "add.u32");
        assert!(
            result.is_err(),
            "u64 should NOT be compatible with u32 instruction (size mismatch)"
        );
    }

    #[test]
    fn test_type_check_immediates_are_polymorphic() {
        // Immediates should be compatible with any instruction type
        let ctx = ctx_with_registers(&[]);

        let imm = ctx.resolve_operand_typed(&AstOperand::ImmInt(42)).unwrap();

        // Should work with u32
        assert!(
            ctx.check_operand_type(&imm, ScalarType::U32, "add.u32")
                .is_ok()
        );
        // Should work with f32
        assert!(
            ctx.check_operand_type(&imm, ScalarType::F32, "add.f32")
                .is_ok()
        );
        // Should work with s64
        assert!(
            ctx.check_operand_type(&imm, ScalarType::S64, "add.s64")
                .is_ok()
        );
    }

    // =========================================================================
    // Relaxed Type Checking Tests (PTX 9.4.1)
    // =========================================================================

    #[test]
    fn test_relaxed_load_wider_dest_ok() {
        // ld.u8 into u32 register - should succeed (value is zero-extended)
        let ctx = ctx_with_registers(&[("%r0", ScalarType::U32)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_dst_type_relaxed(&dst, ScalarType::U8, "ld.u8");
        assert!(
            result.is_ok(),
            "Loading u8 into u32 register should be allowed"
        );
    }

    #[test]
    fn test_relaxed_load_same_size_ok() {
        // ld.u32 into u32 register - should succeed
        let ctx = ctx_with_registers(&[("%r0", ScalarType::U32)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_dst_type_relaxed(&dst, ScalarType::U32, "ld.u32");
        assert!(result.is_ok());
    }

    #[test]
    fn test_relaxed_load_narrower_dest_fails() {
        // ld.u32 into u8 register - should FAIL (dest too narrow)
        let ctx = ctx_with_registers(&[("%r0", ScalarType::U8)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_dst_type_relaxed(&dst, ScalarType::U32, "ld.u32");
        assert!(result.is_err(), "Loading u32 into u8 register should fail");
    }

    #[test]
    fn test_relaxed_load_float_int_still_incompatible() {
        // ld.f32 into u64 register - should FAIL (float/int mismatch)
        let ctx = ctx_with_registers(&[("%r0", ScalarType::U64)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_dst_type_relaxed(&dst, ScalarType::F32, "ld.f32");
        assert!(
            result.is_err(),
            "Loading f32 into integer register should fail even with relaxed checking"
        );
    }

    #[test]
    fn test_relaxed_store_wider_source_ok() {
        // st.u8 from u32 register - should succeed (value is truncated)
        let ctx = ctx_with_registers(&[("%r0", ScalarType::U32)]);

        let src = ctx
            .resolve_operand_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_operand_type_relaxed(&src, ScalarType::U8, "st.u8");
        assert!(
            result.is_ok(),
            "Storing from u32 register to u8 should be allowed (truncation)"
        );
    }

    #[test]
    fn test_relaxed_store_narrower_source_fails() {
        // st.u32 from u8 register - should FAIL (source too narrow)
        let ctx = ctx_with_registers(&[("%r0", ScalarType::U8)]);

        let src = ctx
            .resolve_operand_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_operand_type_relaxed(&src, ScalarType::U32, "st.u32");
        assert!(
            result.is_err(),
            "Storing from u8 register to u32 should fail"
        );
    }

    // =========================================================================
    // Special Register Type Tests
    // =========================================================================

    #[test]
    fn test_special_reg_type_resolution() {
        let ctx = ctx_with_registers(&[]);

        // %tid.x is u32
        let tid = ctx
            .resolve_operand_typed(&AstOperand::Ident(ascii("%tid.x")))
            .unwrap();
        assert_eq!(tid.ty, Some(ScalarType::U32));

        // Should be compatible with u32 instruction
        assert!(
            ctx.check_operand_type(&tid, ScalarType::U32, "add.u32")
                .is_ok()
        );

        // Should NOT be compatible with f32 instruction
        assert!(
            ctx.check_operand_type(&tid, ScalarType::F32, "add.f32")
                .is_err()
        );
    }

    // =========================================================================
    // Hint Generation Tests
    // =========================================================================

    #[test]
    fn test_hint_for_float_to_int() {
        let ctx = ctx_with_registers(&[]);
        let hint = ctx.type_mismatch_hint(ScalarType::F32, ScalarType::U32);
        assert!(hint.contains("cvt"), "Hint should suggest cvt instruction");
    }

    #[test]
    fn test_hint_for_int_to_float() {
        let ctx = ctx_with_registers(&[]);
        let hint = ctx.type_mismatch_hint(ScalarType::S32, ScalarType::F32);
        assert!(hint.contains("cvt"), "Hint should suggest cvt instruction");
    }

    #[test]
    fn test_hint_for_size_mismatch() {
        let ctx = ctx_with_registers(&[]);
        let hint = ctx.type_mismatch_hint(ScalarType::U64, ScalarType::U32);
        assert!(
            hint.contains("bits") || hint.contains("size"),
            "Hint should mention size mismatch"
        );
    }
}
