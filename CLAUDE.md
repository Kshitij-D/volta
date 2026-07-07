# Volta Architecture Documentation

**Purpose**: Codebase context for AI-assisted development

## Overview

Volta is an abstract interpreter for NVIDIA PTX kernels, implementing the approach from "Equivalence Checking of ML GPU Kernels" (arXiv:2511.12638). It detects data races and verifies kernel equivalence.

## Coding practices

- Take advantage of the type system to ensure correctness. E.g., rather than using a `u32`, to avoid mixing up different kinds of indices, consider using a new type that wraps a `u32`. Likewise, consider where it is better to use a custom, two variant enum in place of a `bool`.
- Shared state makes reasoning about code complex. Err on the side of slightly less efficient but pure implementations.

## Crate Structure

```
crates/
├── volta_common/     # Base utilities (spans, file caching, error reporting)
├── volta_frontend/   # PTX lexer and parser
├── volta_analysis/   # Abstract interpreter
├── volta_bench/      # Paper-evaluation benchmark harness
└── volta_cli/        # Command-line interface
```

### Dependency Graph

```
volta_cli ──┐         volta_bench ──┐
    ├── volta_analysis ◄────────────┤
    │       ├── volta_frontend ◄────┘
    │       │       └── volta_common
    │       └── volta_common
    └── volta_frontend
            └── volta_common
```

## Crate: volta_common

**Path**: `crates/volta_common/`

- `Span` - Source location (low + high byte offset)
- `FileCache` - Caches file content to make sure we always use a consistent version of each file
- `Locate<E>` - Error wrapper with optional location info (span + file path)
- `report_error` - Produces an error message from a title, message, span, and file content. Extracts out and includes the code snippet at the given span in the given file content

The pattern is to create an error kind type, and then an alias for `Locate` of that error kind. `locate_span` can be used to tag a `Locate` with a span if it does not already have one. `locate_path` can be used to tag a `Locate` with a path if it does not already have one.

## Crate: volta_frontend

**Path**: `crates/volta_frontend/`

### Lexer (`lex.rs`)

Tokenizes PTX source. Key methods: `next()`, `peek()`, `expect(kind)`.

### Parser (`parse.rs`)

Pratt parser producing AST. Entry point: `parse_module()`.

### AST (`ast.rs`)

- `Module` - Top-level: version, target, address_size, directives
- `Function` - Kernel/device function with params and body
- `Instruction` - Generic instruction with mnemonic string and operands
- `ScalarType` - Pred, Signed/Unsigned/Float/Bits with width

### Instruction Parsing (`instr.rs`, `instr_parse.rs`)

- `InstrTrie` - O(n) lookup for PTX mnemonics → `InstrKind`
- `ParsedInstruction` - Strongly-typed enum (~80 instruction variants)
- Converts generic `Instruction` to typed variants with validated modifiers

## Crate: volta_analysis

**Path**: `crates/volta_analysis/`

### ID Types

Strongly-typed IDs (`#[id_type]` from `id_collections`), each declared next
to its subsystem:

- `InstrId` (`lowered.rs`), `ThreadId` (`eval/mod.rs`), `ParamId` and
  `RegId` = `RegClass` + `RegIndex` (`symbols.rs`; class: Pred,
  Bits8/16/32/64/128), `ExprId`/`StringId`/`SymbolId` (`symbolic.rs`;
  `SymbolId::fresh()` draws from a process-global counter)
- `types.rs` holds `ScalarTypeExt` (width/kind helpers over the AST's
  `ScalarType`), not IDs

### Symbolic Expressions (`symbolic.rs`)

Arena-allocated: nodes live in an `ExprArena`, referenced by copyable `ExprId`
handles. Constructors constant-fold eagerly.

- **Atoms**: `IntConst`, `FloatConst`, `BoolConst`, `Symbol(SymbolId)`, `NamedSymbol(StringId)`, `Undefined`
- **Arithmetic**: `Add`, `Sub`, `Mul`, `Div`, `Rem`, `Neg`, `Fma`
- **Transcendental**: `Exp`, `Log`, `Sqrt`, `Rcp`
- **Bitwise**: `BitAnd`, `BitOr`, `BitXor`, `BitNot`, `Shl`, `Shr`, `LShr`
- **Comparison**: `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge` (return boolean)
- **Boolean**: `And`, `Or`, `Not`
- **Other**: `Select` (ternary), `Min`, `Max`, `Abs`, type conversions

### Lowering (`lowering.rs`, `lowered.rs`)

Converts AST to linear instruction format:

- `LoweredProgram` - `IdVec<InstrId, LoweredInstr>` + `SymbolTable` + `SourceMap`
- `LoweredInstr` variants: `LoadParam`, `Load`, `Store`, `Mov`, `BinOp`, `UnaryOp`, `Fma`, `Mad`, `MulWide`, `MulHi`, `Setp`, `Selp`, `Cvt`, `Bra`, `Ret`, `Exit`, `BarSync`, `BarWarpSync`, `ShflSync`, `Ldmatrix`, `Mma`, `WmmaLoad/Store/Mma`, `Activemask`, `Trap`, etc.
- `SymbolTable` - Register/param/label name → ID resolution; assigns addresses to shared/local/module-global variables
- `SourceMap` - Maps lowered elements back to source spans
- The nvcc callseq idiom for `call __symexpf` (the paper's symbolic-exp hook) collapses to `UnaryOp::Exp` at lowering time

### Special Registers (`symbols.rs`)

`SpecialRegKind`: `TidX/Y/Z`, `NtidX/Y/Z`, `CtaidX/Y/Z`, `LaneId`, `WarpId`, etc.

### Evaluator (`eval/`)

The interpreter from the paper (per-thread round-robin symbolic execution):

- `eval/interp.rs` - `Interpreter`: scheduler (run a thread until it blocks or exits), instruction evaluation into the arena, barrier firing per the paper's Sync rule (exited threads count as arrived), deadlock detection, structured-CTA concreteness checks
- `eval/value.rs` - `Value::{Scalar, Pair}` (`Pair` = packed f16 halves in a 32-bit register) and per-thread `RegFile`
- `eval/memory.rs` - byte-addressed granule memory; 4-byte reads combine two 2-byte granules into a `Pair`, 2-byte accesses split `Pair` granules; program writes are `dirty` (the output footprint)
- `eval/race.rs` - χ-context race detection per byte (paper Section 3.2); full-CTA barrier sync is a wholesale clear
- `eval/warp.rs` - warp-cooperative ops (`shfl.sync`, `ldmatrix`, `mma.sync`, `wmma.*`): block until all mask lanes converge at the pc, sync χ, execute via the `tensor_core.rs` fragment tables with exact per-lane access attribution
- `eval/config.rs` - `AnalysisConfig`: launch dims, positional `ParamValue`s (int/float/symbolic-float/array-pointer), `ArrayDef`s (`Input`/`Output`/`InputOutput`/`IndexInput`), module-global values, dynamic shared size
- `eval/error.rs` - `EvalError`: `DataRace`, `Deadlock`, `NotConcrete`, `OutOfBounds`, `UndefinedOutput`, `TrapReached`, etc.

Key semantics: input-array symbols materialize lazily on first read; reads of
never-written registers/shared bytes yield `Undefined` (an error only if it
reaches an output or a concreteness point) - the paper's race example and
nvcc's `selp` accumulator-init idiom both rely on this.

### Driver (`driver.rs`)

- `analyze_kernel(module, kernel_name, config) -> Result<AnalysisOutput, AnalysisError>`
- `AnalysisOutput`: per-output-array written elements as `(index, ExprId)` + `Stats` (instructions, block syncs; warp syncs counted per fired group, not per thread)
- `check_output_equivalence_with(ref, opt, options)` - the per-element
  check via one shared `EquivSession`. `EquivCheckOptions`:
  `FootprintPolicy::{Exact, Intersect}` (Intersect compares the common
  written indices - a grid-stride reference vs a tiled kernel), `sample`,
  `verify_numeric` (f64 oracle per element), `recycle_terms`. Returns a
  report with the outcome plus checked/total element counts.
- `check_output_equivalence(ref, opt)` - the strict Default-options wrapper
  (identical footprints, all elements)

### Decision procedure (`canon/`, `equiv.rs`, `numeric.rs`)

The paper's canonicalizer, in Rust:

- `canon/` - expressions canonicalize to interned `Σ c·monomial·e^{poly}`
  rationals in one memoized bottom-up pass per `Session` (both kernels, all
  VC elements share intern tables). Exact i128 rational coefficients;
  `e^a·e^b` fuses at term multiplication; max/min flatten into sorted atoms;
  ops outside the fragment (sqrt/log/bitwise/comparisons/select/symbolic
  array reads) become opaque `Atom::Uninterp` atoms over an `UninterpOp`
  enum - sound, incomplete. Fraction equality goes id-compare →
  monomial-quotient (softmax rescaling) → cross-multiplication under a term
  budget. Two load-bearing invariants: single-use chain intermediates stay
  *transient* owned vectors (interning everything retains O(K²) per
  accumulator), and polys sort by *descending* TermId so chain unwinding
  appends in O(1).
- `equiv.rs` - thin wrapper: `EquivSession` (reuse across elements;
  recycles its intern tables past a configurable term bound -
  `with_recycle_terms`, default `DEFAULT_RECYCLE_TERMS` = 4M) and one-shot
  `check_equivalent`. Memory scale: exp-heavy attention terms run 2-4 KB
  each, so one warm FlashAttention output row retains several GiB; small
  bounds trade re-canonicalization time for bounded memory.
- `numeric.rs` - the f64 oracle: seeded random inputs, memoized DAG eval;
  `verify_verdict` confirms EQUIV/DIFF claims (volta-bench
  `--verify-numeric`). Agreement at random points ⇒ equality almost surely
  for this fragment (the paper's own Schwartz-Zippel argument).

### Logging (`logging.rs`)

Gated by the `logging` feature (`volta_analysis`, passed through by
`volta_cli`); without it the macros are no-op stubs. Wired at the decision
points: barrier/warp-group fires (trace), deadlock (warn), launch config,
completion stats, and VC session recycles (info), fraction-equality
escalation (debug). `cargo run -p volta_cli --features logging --
--log-level info analyze ...` narrates a run.

## Crate: volta_bench

**Path**: `crates/volta_bench/`

Reproduces the paper's evaluation over `kernels/` (the `.cu` + `.ptx` for
every benchmark in the paper, organized by table/section).
Benchmark definitions with full launch/param configs live in
`src/benchmarks/*.rs`. Run with `cargo run --release -p volta_bench --
category <reduction|matmul|attention|causal|conv|agent|tilelang|race>
[--sample N] [--verify-numeric] [--recycle-terms N]` (also `all`, `single
<name>`, `list`; release mode matters: ~20x). The element loop is
`driver::check_output_equivalence_with` with `FootprintPolicy::Intersect`.
Memory: full-element attention wants tens of GiB warm - on small machines
run one category at a time under `ulimit -v` with `--recycle-terms 250000`
(bounded at ~5 GiB, slower VCs).

## Crate: volta_cli

**Path**: `crates/volta_cli/`

Commands:

- `volta parse <file>` - Check syntax
- `volta analyze <file> -k <kernel> -b 32,4 -g 1,2 --array name:base:width:len:kind --param ptr:name ...` - Run symbolic execution, report races/deadlocks, print output expressions

## Data Flow

```
PTX Source
    │
    ▼
Lexer (lex.rs) ──► Tokenizes source
    │
    ▼
Parser (parse.rs) ──► Builds AST
    │
    ▼
Instruction Parser (instr_parse.rs) ──► Strongly-typed ParsedInstruction
    │
    ▼
Lowering (lowering.rs) ──► LoweredProgram (resolved RegIds, InstrIds)
    │
    ▼
Evaluator (eval/interp.rs) ──► Symbolic execution with N threads
    │                           (χ race detection, warp/tensor-core ops)
    ▼
AnalysisOutput ──► per-element output expressions + statistics
    │
    ▼
Decision procedure (canon/ via equiv.rs) ──► per-element VC checking
                                             (+ numeric.rs oracle)
```

## Key Design Decisions

### 1. Symbolic Execution over Concrete Values

All values are `Expr` (symbolic expressions). This allows analyzing behavior for arbitrary thread indices and detecting races without enumerating all inputs.

### 2. Concrete Addresses for Race Detection

Memory addresses must be concrete (`u64`) for race detection. Thread indices are concrete (specific block configuration). Symbolic address accesses produce `SymbolicAddress` error.

### 3. χ-Context for Race Detection

From the paper (Section 4.2): Track which threads haven't synchronized since each memory access. After barrier, threads in sync set remove each other from "needs sync" sets. Race detected when accessing thread is in the "needs sync" set.

### 4. Round-Robin Scheduling

Simple, deterministic interleaving. Sufficient for race detection (any interleaving that produces a race proves the race exists).

### 5. Strongly-Typed IDs

Newtype pattern for IDs (`RegId`, `InstrId`, `ThreadId`) prevents mixing at compile time. Uses `IdVec<K, V>` for type-safe indexed collections.

### 6. Two-Phase Instruction Parsing

1. Lexer/Parser → generic `Instruction { op: String, operands }`
2. Instruction Parser → strongly-typed `ParsedInstruction` variants

Enables robust modifier validation and better error messages.

### 7. Separate Memory Spaces

Global, shared, local, param memories are separate `Memory` instances, matching PTX memory model.

## Adding a New Instruction

1. **Frontend**: Add `InstrKind` variant in `instr.rs`, add to trie
2. **Instruction Parsing**: Add `ParsedInstruction` variant, parser function
   in `instr_parse.rs` (use `expect_operands` for exact-arity operand lists)
3. **Lowering**: Add `LoweredInstr` variant, lowering case
4. **Evaluation**: Add evaluation case in `eval/interp.rs`
