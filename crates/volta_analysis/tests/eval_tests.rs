//! End-to-end evaluator tests.
//!
//! Small synthetic kernels (1-2 threads) validate each mechanism cheaply;
//! the Harris reduction kernels from the paper benchmarks (64-128 threads,
//! ~100 instructions each) validate real equivalence checking.

use std::path::PathBuf;

use volta_analysis::AnalysisError;
use volta_analysis::driver::{EquivOutcome, analyze_kernel, check_output_equivalence};
use volta_analysis::eval::{
    AnalysisConfig, AnalysisOutput, ArrayDef, ArrayKind, EvalError, ParamValue,
};
use volta_frontend::ascii::AsAscii;
use volta_frontend::ast::Module;
use volta_frontend::parse::Parser;

fn parse(src: &str) -> Module {
    let ascii = src.as_bytes().as_ascii_slice().expect("ascii source");
    Parser::new(ascii)
        .parse_module()
        .unwrap_or_else(|e| panic!("parse error: {:?}", e.error))
}

/// The volta_bench paper-benchmark kernel tree (PTX + CUDA sources).
const KERNELS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../volta_bench/kernels");

fn parse_file(rel: &str) -> Module {
    let path = PathBuf::from(KERNELS_DIR).join(rel);
    let src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    parse(&src)
}

const HEADER: &str = ".version 8.0\n.target sm_80\n.address_size 64\n\n";

fn wrap(body: &str) -> String {
    format!("{}{}", HEADER, body)
}

/// in/out f32 arrays at fixed bases.
fn in_out_config(threads: u32, len: u64) -> AnalysisConfig {
    let mut config = AnalysisConfig::new((threads, 1, 1));
    config.arrays = vec![
        ArrayDef {
            name: "in".to_string(),
            base: 0x10000,
            elem_width: 4,
            len,
            kind: ArrayKind::Input,
        },
        ArrayDef {
            name: "out".to_string(),
            base: 0x20000,
            elem_width: 4,
            len,
            kind: ArrayKind::Output,
        },
    ];
    config.params = vec![
        ParamValue::ArrayPtr("in".to_string()),
        ParamValue::ArrayPtr("out".to_string()),
    ];
    config
}

fn display_output(output: &AnalysisOutput, array: &str, index: u64) -> String {
    let (_, elems) = output
        .outputs
        .iter()
        .find(|(n, _)| n == array)
        .expect("output array");
    let (_, e) = elems
        .iter()
        .find(|(i, _)| *i == index)
        .expect("element written");
    output.arena.display_expr(*e)
}

// =========================================================================
// Synthetic kernels: one mechanism each
// =========================================================================

/// Two threads write the same shared address without synchronization.
#[test]
fn test_shared_write_write_race() {
    let src = wrap(
        ".visible .entry k()
{
    .reg .b32 %r<3>;
    .shared .align 4 .b8 sdata[8];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, sdata;
    st.shared.u32 [%r2], %r1;
    ret;
}
",
    );
    let module = parse(&src);
    let config = AnalysisConfig::new((2, 1, 1));
    let err = analyze_kernel(&module, None, config).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Eval(EvalError::DataRace { .. })),
        "expected data race, got: {}",
        err
    );
}

/// Neighbor exchange through shared memory, correctly synchronized:
/// out[tid] = in[tid ^ 1].
const SWAP_BODY: &str = ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<3>;
    .reg .b32 %r<7>;
    .reg .b64 %rd<7>;
    .shared .align 4 .b8 sdata[8];

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    mov.u32 %r1, %tid.x;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd3, %r2;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    mov.u32 %r3, sdata;
    add.s32 %r4, %r3, %r2;
    st.shared.f32 [%r4], %f1;
    BARRIER
    xor.b32 %r5, %r1, 1;
    shl.b32 %r5, %r5, 2;
    add.s32 %r6, %r3, %r5;
    ld.shared.f32 %f2, [%r6];
    add.s64 %rd5, %rd2, %rd3;
    st.global.f32 [%rd5], %f2;
    ret;
}
";

#[test]
fn test_shared_exchange_with_barrier() {
    let src = wrap(&SWAP_BODY.replace("BARRIER", "bar.sync 0;"));
    let module = parse(&src);
    let output = analyze_kernel(&module, None, in_out_config(2, 2)).unwrap();
    assert_eq!(display_output(&output, "out", 0), "in[1]");
    assert_eq!(display_output(&output, "out", 1), "in[0]");
    assert_eq!(output.stats.block_syncs, 2);
}

#[test]
fn test_shared_exchange_without_barrier_races() {
    let src = wrap(&SWAP_BODY.replace("BARRIER", ""));
    let module = parse(&src);
    let err = analyze_kernel(&module, None, in_out_config(2, 2)).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Eval(EvalError::DataRace { .. })),
        "expected data race, got: {}",
        err
    );
}

/// Threads waiting on different barrier ids never fire: deadlock.
#[test]
fn test_mismatched_barriers_deadlock() {
    let src = wrap(
        ".visible .entry k()
{
    .reg .pred %p<2>;
    .reg .b32 %r<2>;

    mov.u32 %r1, %tid.x;
    setp.eq.s32 %p1, %r1, 0;
    @%p1 bra $L1;
    bar.sync 0;
    bra $L2;
$L1:
    bar.sync 1;
$L2:
    ret;
}
",
    );
    let module = parse(&src);
    let err = analyze_kernel(&module, None, AnalysisConfig::new((2, 1, 1))).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Eval(EvalError::Deadlock { .. })),
        "expected deadlock, got: {}",
        err
    );
}

/// A thread that exits counts as having arrived at the barrier (paper's
/// Sync rule allows `return`), so this does NOT deadlock.
#[test]
fn test_exited_thread_releases_barrier() {
    let src = wrap(
        ".visible .entry k()
{
    .reg .pred %p<2>;
    .reg .b32 %r<2>;

    mov.u32 %r1, %tid.x;
    setp.eq.s32 %p1, %r1, 0;
    @%p1 ret;
    bar.sync 0;
    ret;
}
",
    );
    let module = parse(&src);
    analyze_kernel(&module, None, AnalysisConfig::new((2, 1, 1))).unwrap();
}

/// An uninitialized shared read is tolerated during execution (the paper's
/// race example depends on it), but an output computed from one is an error.
#[test]
fn test_uninitialized_shared_read_flows_to_output() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    .shared .align 4 .b8 sdata[8];

    ld.param.u64 %rd2, [k_param_1];
    mov.u32 %r1, sdata;
    ld.shared.u32 %r2, [%r1];
    st.global.u32 [%rd2], %r2;
    ret;
}
",
    );
    let module = parse(&src);
    let err = analyze_kernel(&module, None, in_out_config(1, 1)).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Eval(EvalError::UndefinedOutput { .. })),
        "expected undefined output, got: {}",
        err
    );
}

#[test]
fn test_out_of_bounds_shared_access() {
    let src = wrap(
        ".visible .entry k()
{
    .reg .b32 %r<3>;
    .shared .align 4 .b8 sdata[8];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, sdata;
    st.shared.u32 [%r2+64], %r1;
    ret;
}
",
    );
    let module = parse(&src);
    let err = analyze_kernel(&module, None, AnalysisConfig::new((1, 1, 1))).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Eval(EvalError::OutOfBounds { .. })),
        "expected out-of-bounds, got: {}",
        err
    );
}

/// nvcc's u16 magic-number division (Conv2D-opt's index cache): the
/// immediate -17873 is the u16 constant 47663 = ceil(2^19/11); operands of
/// `mul.wide.u16` must be reinterpreted as unsigned, not consumed at their
/// producer's signedness.
#[test]
fn test_mul_wide_u16_magic_division() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .b16 %rs<5>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_1];
    mov.u16 %rs1, 14;
    mul.wide.u16 %r1, %rs1, -17873;
    shr.u32 %r2, %r1, 19;
    cvt.u16.u32 %rs2, %r2;
    mul.lo.s16 %rs3, %rs2, 11;
    sub.s16 %rs4, %rs1, %rs3;
    cvt.u32.u16 %r3, %rs4;
    st.global.u32 [%rd1], %r2;
    st.global.u32 [%rd1+4], %r3;
    ret;
}
",
    );
    let module = parse(&src);
    let output = analyze_kernel(&module, None, in_out_config(1, 2)).expect("analysis");
    // 14 / 11 = 1, 14 % 11 = 3.
    assert_eq!(display_output(&output, "out", 0), "1");
    assert_eq!(display_output(&output, "out", 1), "3");
}

/// `mul.hi.u32` must zero-extend a canonically-signed operand: the high
/// half of 0xFFFFFFFF * 2 is 1, not -1's sign fill.
#[test]
fn test_mul_hi_u32_negative_operand() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_1];
    mov.u32 %r1, -1;
    mul.hi.u32 %r2, %r1, 2;
    st.global.u32 [%rd1], %r2;
    ret;
}
",
    );
    let module = parse(&src);
    let output = analyze_kernel(&module, None, in_out_config(1, 1)).expect("analysis");
    assert_eq!(display_output(&output, "out", 0), "1");
}

/// Branching on input data violates structured-CTA.
#[test]
fn test_symbolic_branch_rejected() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<2>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.global.f32 %f1, [%rd1];
    setp.gt.f32 %p1, %f1, 0f00000000;
    @%p1 bra $L1;
$L1:
    ret;
}
",
    );
    let module = parse(&src);
    let err = analyze_kernel(&module, None, in_out_config(1, 1)).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Eval(EvalError::NotConcrete { .. })),
        "expected structured-CTA violation, got: {}",
        err
    );
}

/// The `__symexpf` callseq idiom becomes symbolic exp.
#[test]
fn test_symexpf_callseq() {
    let src = wrap(
        ".extern .func  (.param .b32 func_retval0) __symexpf
(
    .param .b32 __symexpf_param_0
)
;

.visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<3>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    ld.global.f32 %f1, [%rd1];
    { // callseq 0, 0
    .reg .b32 temp_param_reg;
    .param .b32 param0;
    st.param.f32 [param0+0], %f1;
    .param .b32 retval0;
    call.uni (retval0),
    __symexpf,
    (
    param0
    );
    ld.param.f32 %f2, [retval0+0];
    } // callseq 0
    st.global.f32 [%rd2], %f2;
    ret;
}
",
    );
    let module = parse(&src);
    let output = analyze_kernel(&module, None, in_out_config(1, 1)).unwrap();
    assert_eq!(display_output(&output, "out", 0), "exp(in[0])");
}

/// shfl.sync.idx exchanges lane values (2 lanes, mask 0x3):
/// out[tid] = in[tid ^ 1].
#[test]
fn test_shfl_sync_idx_exchange() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<3>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<7>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    mov.u32 %r1, %tid.x;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd3, %r2;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    xor.b32 %r3, %r1, 1;
    mov.u32 %r4, 31;
    mov.u32 %r5, 3;
    shfl.sync.idx.b32 %f2, %f1, %r3, %r4, %r5;
    add.s64 %rd5, %rd2, %rd3;
    st.global.f32 [%rd5], %f2;
    ret;
}
",
    );
    let module = parse(&src);
    let output = analyze_kernel(&module, None, in_out_config(2, 2)).unwrap();
    assert_eq!(display_output(&output, "out", 0), "in[1]");
    assert_eq!(display_output(&output, "out", 1), "in[0]");
    // Warp syncs count once per fired group.
    assert_eq!(output.stats.warp_syncs, 1);
}

// =========================================================================
// Paper kernels: Harris reductions
// =========================================================================

/// Config for the reduction kernels: int arrays, `n` input elements.
fn reduction_config(threads: u32, n: u64) -> AnalysisConfig {
    let mut config = AnalysisConfig::new((threads, 1, 1));
    config.arrays = vec![
        ArrayDef {
            name: "in".to_string(),
            base: 0x10000,
            elem_width: 4,
            len: n,
            kind: ArrayKind::Input,
        },
        ArrayDef {
            name: "out".to_string(),
            base: 0x20000,
            elem_width: 4,
            len: 1,
            kind: ArrayKind::Output,
        },
    ];
    config.params = vec![
        ParamValue::ArrayPtr("in".to_string()),
        ParamValue::ArrayPtr("out".to_string()),
    ];
    config
}

fn run_reduction(
    file: &str,
    kernel: &str,
    threads: u32,
    n: u64,
    dynamic_shared: u64,
) -> AnalysisOutput {
    let module = parse_file(file);
    let mut config = reduction_config(threads, n);
    config.dynamic_shared_bytes = dynamic_shared;
    analyze_kernel(&module, Some(kernel), config)
        .unwrap_or_else(|e| panic!("{} failed: {}", file, e))
}

fn red1() -> AnalysisOutput {
    run_reduction(
        "01_reduction/Red-1.ptx",
        "_Z17reduce1024_1blockPKiPi",
        128,
        128,
        0,
    )
}

#[test]
fn test_red1_race_free() {
    let output = red1();
    assert_eq!(output.outputs.len(), 1);
    assert!(output.stats.block_syncs > 0);
}

#[test]
fn test_red1_self_equivalence() {
    let a = red1();
    let b = red1();
    assert!(matches!(
        check_output_equivalence(&a, &b).unwrap(),
        EquivOutcome::Equivalent
    ));
}

#[test]
fn test_red1_red2_equivalent() {
    let a = red1();
    let b = run_reduction("01_reduction/Red-2.ptx", "_Z7reduce2PiS_", 128, 128, 0);
    assert!(matches!(
        check_output_equivalence(&a, &b).unwrap(),
        EquivOutcome::Equivalent
    ));
}

#[test]
fn test_red1_red3_equivalent() {
    let a = red1();
    // Red-3 uses extern (dynamic) shared memory: 128 ints.
    let b = run_reduction("01_reduction/Red-3.ptx", "_Z7reduce3PiS_", 128, 128, 512);
    assert!(matches!(
        check_output_equivalence(&a, &b).unwrap(),
        EquivOutcome::Equivalent
    ));
}

#[test]
fn test_red1_red4_equivalent() {
    let a = red1();
    // Red-4 adds `in[i] + in[i + 64]` on load and reduces the low half of
    // sdata. Threads 64..128 read `in[128..192)`, but those values only land
    // in the unused half of sdata, so the sum still covers exactly
    // `in[0..128)`. The input array must span 192 elements to keep the loads
    // in bounds.
    let b = run_reduction("01_reduction/Red-4.ptx", "_Z7reduce0PiS_", 128, 192, 0);
    assert!(matches!(
        check_output_equivalence(&a, &b).unwrap(),
        EquivOutcome::Equivalent
    ));
}

/// A deliberately wrong reduction (dropping one element) must be caught.
#[test]
fn test_red1_wrong_input_not_equivalent() {
    let a = red1();
    // Same kernel, but with only 127 of the 128 inputs shared: rename the
    // input array so its symbols differ.
    let module = parse_file("01_reduction/Red-1.ptx");
    let mut config = reduction_config(128, 128);
    config.arrays[0].name = "other".to_string();
    config.params[0] = ParamValue::ArrayPtr("other".to_string());
    let b = analyze_kernel(&module, Some("_Z17reduce1024_1blockPKiPi"), config).unwrap();
    assert!(matches!(
        check_output_equivalence(&a, &b).unwrap(),
        EquivOutcome::NotEquivalent { .. }
    ));
}
