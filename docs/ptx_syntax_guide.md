# PTX Instruction Syntax Block Guide

This document explains how to read and interpret the syntax blocks extracted from the NVIDIA PTX ISA documentation.

## Overview

Each PTX instruction is documented with a **Syntax** block that describes all valid forms of the instruction. The syntax uses a notation system with placeholders, optional elements, and type constraints.

## Basic Instruction Format

PTX instructions follow this general pattern:

```
opcode.modifier1.modifier2.type  destination, source1, source2, ...;
```

- **opcode**: The instruction mnemonic (e.g., `add`, `mul`, `ld`)
- **modifiers**: Optional suffixes separated by dots (e.g., `.sat`, `.wide`, `.hi`)
- **type**: Data type specifier (e.g., `.u32`, `.f32`, `.s16`)
- **operands**: Comma-separated destination and source operands

## Notation Conventions

### Curly Braces `{...}` - Optional Elements

Curly braces indicate **optional** modifiers or components that may be omitted.

```
add{.sat}.s32 d, a, b;
```

This means both of these are valid:
- `add.s32 d, a, b;` (without saturation)
- `add.sat.s32 d, a, b;` (with saturation)

### Type Definitions `.type = { ... }`

Lines starting with `.type = ` define the **valid values** that can substitute for `.type` in the instruction template:

```
add.type d, a, b;

.type = { .u16, .u32, .u64,
          .s16, .s32, .s64 };
```

This means you can write:
- `add.u16 d, a, b;`
- `add.u32 d, a, b;`
- `add.s64 d, a, b;`
- (and so on for any type in the set)

### Mode/Modifier Definitions `.mode = { ... }`

Similar to type definitions, these specify valid values for modifier placeholders:

```
mul.mode.type d, a, b;

.mode = { .hi, .lo, .wide };
.type = { .u16, .u32, .u64 };
```

Valid combinations include:
- `mul.hi.u32 d, a, b;`
- `mul.lo.s16 d, a, b;`
- `mul.wide.u16 d, a, b;`

### Multiple Syntax Variants

When an instruction has multiple forms, each valid pattern is listed on its own line:

```
mad.mode.type  d, a, b, c;
mad.hi.sat.s32 d, a, b, c;

.mode = { .hi, .lo, .wide };
.type = { .u16, .u32, .u64, .s16, .s32, .s64 };
```

The second line shows a **specific variant** where `.sat` is only valid with `.hi.s32`.

The name of the instruction is the longest common prefix in the block. Here that's just "mad".

### Operand Placeholders

| Placeholder | Meaning |
|-------------|---------|
| `d` | Destination register |
| `a`, `b`, `c` | Source operands (registers, immediates, or addresses) |
| `p`, `q` | Predicate registers |
| `[a]` | Memory address (square brackets indicate indirection) |

### Comments `//`

Inline comments provide additional constraints or clarifications:

```
add{.sat}.s32 d, a, b;  // .sat applies only to .s32
```

## Common Type Specifiers

### Integer Types
| Type | Description |
|------|-------------|
| `.u8`, `.u16`, `.u32`, `.u64` | Unsigned integers (8/16/32/64-bit) |
| `.s8`, `.s16`, `.s32`, `.s64` | Signed integers (8/16/32/64-bit) |
| `.b8`, `.b16`, `.b32`, `.b64` | Untyped bits (8/16/32/64-bit) |

### Floating-Point Types
| Type | Description |
|------|-------------|
| `.f16`, `.f16x2` | Half precision (16-bit), packed pair |
| `.bf16`, `.bf16x2` | Brain float (16-bit), packed pair |
| `.f32` | Single precision (32-bit) |
| `.f64` | Double precision (64-bit) |
| `.tf32` | TensorFloat-32 (19-bit mantissa) |

### Packed Types
| Type | Description |
|------|-------------|
| `.u16x2`, `.s16x2` | Two 16-bit integers packed in 32 bits |
| `.f16x2`, `.bf16x2` | Two 16-bit floats packed in 32 bits |

## Common Modifiers

### Arithmetic Modifiers
| Modifier | Meaning |
|----------|---------|
| `.sat` | Saturate result to type's min/max (no overflow) |
| `.rn`, `.rz`, `.rm`, `.rp` | Rounding modes (nearest, zero, minus, plus infinity) |
| `.ftz` | Flush denormals to zero |
| `.hi`, `.lo` | Return high/low half of result |
| `.wide` | Result is twice the width of inputs |

### Memory Modifiers
| Modifier | Meaning |
|----------|---------|
| `.global`, `.shared`, `.local`, `.const` | Memory space |
| `.volatile` | Prevents optimization of memory access |
| `.relaxed`, `.acquire`, `.release` | Memory ordering |

## Example: Reading a Complex Syntax Block

```
fma.rnd{.ftz}{.sat}.f32 d, a, b, c;
fma.rnd{.ftz}.relu.f16 d, a, b, c;
fma.rnd{.relu}.bf16 d, a, b, c;

.rnd = { .rn };
```

This tells us:
1. `fma.rnd` (fused multiply-add) has multiple forms depending on the data type
2. For `.f32`: rounding is required, `.ftz` and `.sat` are optional
3. For `.f16`: rounding is required, `.ftz` is optional, `.relu` is available
4. For `.bf16`: rounding is required, `.relu` is optional
5. The only valid rounding mode is `.rn` (round to nearest)

Valid instructions include:
- `fma.rn.f32 d, a, b, c;`
- `fma.rn.ftz.sat.f32 d, a, b, c;`
- `fma.rn.relu.f16 d, a, b, c;`
- `fma.rn.bf16 d, a, b, c;`

Note that the name of the instruction is "fma.rnd" not "fma" because "fma.rnd" is the longest prefix.

This matters because, e.g., "st.async.weak" is the "st.async" instruction with
the "weak" modifier.  But "st.weak" is the "st" instruction with the "weak"
modifier.  The way you know this is that, when you read the documentation, you
will see two different blocks.  One block where everything is prefixed
"st.async" and a different block where everything is prefixed "st", rather than
a single block with a mix of both.

## Guard Predicates

All PTX instructions support optional guard predicates for conditional execution:

```
@p    add.s32 d, a, b;    // execute only if predicate p is true
@!p   add.s32 d, a, b;    // execute only if predicate p is false
```

Guard predicates are not shown in the syntax blocks but are always valid.

## References

- [NVIDIA PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- Section 9.1: Format and Semantics of Instruction Descriptions
- Section 9.4: Type Information for Instructions and Operands
