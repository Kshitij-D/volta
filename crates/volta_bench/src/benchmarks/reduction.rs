//! Harris reduction tutorial (Table 1). Red-1..4 are checked for
//! equivalence against Red-1; Red-5/6/7 use deprecated warp-synchronous
//! programming and must be rejected.

use volta_analysis::eval::{AnalysisConfig, ParamValue};

use crate::config::{
    BenchmarkCategory, BenchmarkDef, ExpectedOutcome, KernelRun, f32_input, f32_output,
};

const IN_BASE: u64 = 0x10000;
const OUT_BASE: u64 = 0x20000;

/// `reduceN(const int* g_idata, int* g_odata)` over `n` int elements
/// (element width 4, same layout as f32).
fn config(threads: u32, n: u64, dynamic_shared: u64) -> AnalysisConfig {
    let mut config = AnalysisConfig::new((threads, 1, 1));
    config.arrays = vec![f32_input("in", IN_BASE, n), f32_output("out", OUT_BASE, 1)];
    config.params = vec![
        ParamValue::ArrayPtr("in".to_string()),
        ParamValue::ArrayPtr("out".to_string()),
    ];
    config.dynamic_shared_bytes = dynamic_shared;
    config
}

fn kernel(file: &str, kernel: &str, threads: u32, n: u64, dynamic_shared: u64) -> KernelRun {
    KernelRun::new(
        &format!("01_reduction/{}", file),
        kernel,
        config(threads, n, dynamic_shared),
    )
}

fn red1() -> KernelRun {
    kernel("Red-1.ptx", "_Z17reduce1024_1blockPKiPi", 128, 128, 0)
}

pub fn benchmarks() -> Vec<BenchmarkDef> {
    let red2 = kernel("Red-2.ptx", "_Z7reduce2PiS_", 128, 128, 0);
    // Red-3 uses extern (dynamic) shared memory: 128 ints.
    let red3 = kernel("Red-3.ptx", "_Z7reduce3PiS_", 128, 128, 512);
    // Red-4 adds `in[i] + in[i+64]` on load; threads 64..128 read
    // `in[128..192)` into the unused half of sdata, so the input must span
    // 192 elements while the sum still covers exactly in[0..128).
    let red4 = kernel("Red-4.ptx", "_Z7reduce0PiS_", 128, 192, 0);
    // Red-5/6/7 were compiled with BLOCKSIZE=8 (8 threads); their
    // warp-synchronous tails read past the initialized region, so the
    // dynamic shared allocation must cover the over-reads (40 ints).
    let red5 = kernel("Red-5_racy.ptx", "_Z7reduce0PiS_", 8, 128, 256);
    let red6 = kernel("Red-6_racy.ptx", "_Z7reduce6ILj8EEvPiS0_", 8, 128, 256);
    let red7 = kernel(
        "Red-7_racy.ptx",
        "_Z12reduce1blockILj8EEvPKiPi",
        8,
        128,
        256,
    );

    vec![
        BenchmarkDef::equivalence(
            "(Red-1, Red-1)",
            BenchmarkCategory::Reduction,
            red1(),
            red1(),
        ),
        BenchmarkDef::equivalence("(Red-1, Red-2)", BenchmarkCategory::Reduction, red1(), red2),
        BenchmarkDef::equivalence("(Red-1, Red-3)", BenchmarkCategory::Reduction, red1(), red3),
        BenchmarkDef::equivalence("(Red-1, Red-4)", BenchmarkCategory::Reduction, red1(), red4),
        BenchmarkDef {
            name: "Red-5 (racy)".to_string(),
            category: BenchmarkCategory::Reduction,
            expected: ExpectedOutcome::DataRace,
            reference: red5,
            optimized: None,
        },
        BenchmarkDef {
            name: "Red-6 (racy)".to_string(),
            category: BenchmarkCategory::Reduction,
            expected: ExpectedOutcome::DataRace,
            reference: red6,
            optimized: None,
        },
        BenchmarkDef {
            name: "Red-7 (racy)".to_string(),
            category: BenchmarkCategory::Reduction,
            expected: ExpectedOutcome::DataRace,
            reference: red7,
            optimized: None,
        },
    ]
}
