//! FaialAA data-race benchmarks (Table 7): pre-fix (racy) and post-fix
//! versions of OpenMM and Megatron-LM kernels.
//!
//! Scalar dimensions are baked into these PTX files (they were compiled
//! from specialized sources); array extents here are sized generously and
//! only serve bounds checking.

use volta_analysis::eval::{AnalysisConfig, ParamValue};

use crate::config::{
    BenchmarkCategory, BenchmarkDef, KernelRun, f32_inout, f32_input, f32_output, u32_index,
    u32_inout, u32_input, u32_output,
};

fn pair(name: &str, file_base: &str, kernel: &str, config: AnalysisConfig) -> [BenchmarkDef; 2] {
    pair_with(name, file_base, kernel, config.clone(), config)
}

fn pair_with(
    name: &str,
    file_base: &str,
    kernel: &str,
    pre_config: AnalysisConfig,
    post_config: AnalysisConfig,
) -> [BenchmarkDef; 2] {
    let pre = KernelRun::new(
        &format!("08_races/{}-pre_racy.ptx", file_base),
        kernel,
        pre_config,
    );
    let post = KernelRun::new(
        &format!("08_races/{}-post_fixed.ptx", file_base),
        kernel,
        post_config,
    );
    [
        BenchmarkDef::race_check(
            format!("{} (pre-fix)", name),
            BenchmarkCategory::DataRace,
            pre,
            true,
        ),
        BenchmarkDef::race_check(
            format!("{} (post-fix)", name),
            BenchmarkCategory::DataRace,
            post,
            false,
        ),
    ]
}

/// `computeBucketPositions(unsigned int* bucketOffset)`: an exclusive-scan
/// over bucket counts; the pre-fix version misses a barrier in the scan.
fn bucket_positions() -> AnalysisConfig {
    let mut config = AnalysisConfig::new((64, 1, 1));
    config.arrays = vec![u32_inout("bucketOffset", 0x1_0000_0000, 1 << 20)];
    config.params = vec![ParamValue::ArrayPtr("bucketOffset".to_string())];
    config.dynamic_shared_bytes = 64 * 1024;
    config
}

/// `computeRange(const int* data, int* range)` with the `MAX_KEY` module
/// global bounding the key space.
fn compute_range() -> AnalysisConfig {
    let mut config = AnalysisConfig::new((64, 1, 1));
    config.arrays = vec![
        u32_input("data", 0x1_0000_0000, 1 << 20),
        u32_output("range", 0x2_0000_0000, 1 << 20),
    ];
    config.params = vec![
        ParamValue::ArrayPtr("data".to_string()),
        ParamValue::ArrayPtr("range".to_string()),
    ];
    config.global_values = vec![("MAX_KEY".to_string(), 32)];
    config.dynamic_shared_bytes = 64 * 1024;
    config
}

/// `computeRMSDPart1(posq, referencePos, particles, buffer)` with the
/// particle count baked into the PTX. `particles` is an index array
/// (`posq[particles[i]]`), so it holds concrete identity indices.
fn reduce_value() -> AnalysisConfig {
    let mut config = AnalysisConfig::new((64, 1, 1));
    config.arrays = vec![
        f32_input("posq", 0x1_0000_0000, 1 << 20),
        f32_input("refpos", 0x2_0000_0000, 1 << 20),
        u32_index("particles", 0x3_0000_0000, 1 << 20),
        f32_inout("buffer", 0x4_0000_0000, 1 << 20),
    ];
    config.params = vec![
        ParamValue::ArrayPtr("posq".to_string()),
        ParamValue::ArrayPtr("refpos".to_string()),
        ParamValue::ArrayPtr("particles".to_string()),
        ParamValue::ArrayPtr("buffer".to_string()),
    ];
    config.dynamic_shared_bytes = 64 * 1024;
    config
}

/// `cuApplyLayerNorm(out, mean, invvar, vals, n1, n2, eps, gamma, beta,
/// has_gamma, has_beta)`: one warp-row per n1 index.
///
/// One warp per block (y = 1), for both versions. Both compilations of this
/// Megatron fork dereference an unassigned `__shared__ U* buf` on the
/// `blockDim.y > 1` inter-warp reduction path, which nvcc compiles to `trap`;
/// with y > 1 the first thread of warp 2 traps before any race is
/// reachable. One warp is also all the pre-fix race needs: thread (0,0)
/// writes the `smu`/`sinv` shared broadcast values that every thread then
/// reads with no barrier in between (the `__syncthreads()` the fix commit
/// adds).
fn layer_norm() -> AnalysisConfig {
    const N1: i64 = 2;
    const N2: i64 = 1024;
    let mut config = AnalysisConfig::new((32, 1, 1));
    config.grid_dim = (1, 2, 1);
    config.arrays = vec![
        f32_output("out", 0x1_0000_0000, (N1 * N2) as u64),
        f32_output("mean", 0x2_0000_0000, N1 as u64),
        f32_output("invvar", 0x3_0000_0000, N1 as u64),
        f32_input("vals", 0x4_0000_0000, (N1 * N2) as u64),
        f32_input("gamma", 0x5_0000_0000, N2 as u64),
        f32_input("beta", 0x6_0000_0000, N2 as u64),
    ];
    config.params = vec![
        ParamValue::ArrayPtr("out".to_string()),
        ParamValue::ArrayPtr("mean".to_string()),
        ParamValue::ArrayPtr("invvar".to_string()),
        ParamValue::ArrayPtr("vals".to_string()),
        ParamValue::Int(N1),
        ParamValue::Int(N2),
        ParamValue::Float(1e-5),
        ParamValue::ArrayPtr("gamma".to_string()),
        ParamValue::ArrayPtr("beta".to_string()),
        ParamValue::Int(1),
        ParamValue::Int(1),
    ];
    config.dynamic_shared_bytes = 64 * 1024;
    config
}

/// `cuComputeGradInput(dout, input, mean, invvar, eps, gamma, grad_input)`
/// with n1/n2 baked; a huge grid.y makes CTA (0,0) process only row 0 of
/// the grid-strided loop. See `layer_norm` for the per-version `block_y`.
fn grad_input(block_y: u32) -> AnalysisConfig {
    let mut config = AnalysisConfig::new((32, block_y, 1));
    config.grid_dim = (1, 1 << 20, 1);
    config.arrays = vec![
        f32_input("dout", 0x1_0000_0000, 1 << 22),
        f32_input("input", 0x2_0000_0000, 1 << 22),
        f32_input("mean", 0x3_0000_0000, 1 << 20),
        f32_input("invvar", 0x4_0000_0000, 1 << 20),
        f32_input("gamma", 0x5_0000_0000, 1 << 12),
        f32_output("grad_input", 0x6_0000_0000, 1 << 22),
    ];
    config.params = vec![
        ParamValue::ArrayPtr("dout".to_string()),
        ParamValue::ArrayPtr("input".to_string()),
        ParamValue::ArrayPtr("mean".to_string()),
        ParamValue::ArrayPtr("invvar".to_string()),
        ParamValue::Float(1e-5),
        ParamValue::ArrayPtr("gamma".to_string()),
        ParamValue::ArrayPtr("grad_input".to_string()),
    ];
    config.dynamic_shared_bytes = 64 * 1024;
    config
}

pub fn benchmarks() -> Vec<BenchmarkDef> {
    let mut benches = Vec::new();
    benches.extend(pair(
        "BucketPositions",
        "BucketPositions",
        "_Z22computeBucketPositionsPj",
        bucket_positions(),
    ));
    benches.extend(pair(
        "ComputeRange",
        "ComputeRange",
        "_Z12computeRangePKiPi",
        compute_range(),
    ));
    benches.extend(pair(
        "ReduceValue",
        "ReduceValue",
        "computeRMSDPart1",
        reduce_value(),
    ));
    benches.extend(pair(
        "LayerNorm",
        "LayerNorm",
        "_Z16cuApplyLayerNormIfffEvPT1_PT0_S3_PKT_iiS2_PKS0_S8_ii",
        layer_norm(),
    ));
    benches.extend(pair_with(
        "GradInput",
        "GradInput",
        "_Z18cuComputeGradInputIfffEvPKT1_PKT_PKT0_S8_S6_S2_PS3_",
        grad_input(4),
        grad_input(1),
    ));
    benches
}
