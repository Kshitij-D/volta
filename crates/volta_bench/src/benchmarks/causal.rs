//! Causal attention (Table 4): each attention level from Table 3 with a
//! lower-triangular mask, as a fused/naive pair. "Fused" skips work above
//! the diagonal via conditionals; "naive" always computes the weight and
//! multiplies by a constant 0/1 triangular matrix. The fused version is
//! checked against the naive one at each level. The masking difference
//! evaporates during symbolic evaluation, so the two sides produce nearly
//! identical expressions and the VCs are much cheaper than Table 3's.

use crate::benchmarks::attention::config;
use crate::config::{BenchmarkCategory, BenchmarkDef, KernelRun};

pub fn benchmarks() -> Vec<BenchmarkDef> {
    // (paper name, mangled entry, threads, aux l/m output buffers)
    let levels: [(&str, &str, u32, bool); 4] = [
        ("Causal-Attention", "_Z10sdpa_basicPKfS0_S0_Pff", 1, false),
        (
            "Causal-FA1",
            "_Z12sdpa_fa_likePKfS0_S0_PffS1_S1_",
            128,
            true,
        ),
        (
            "Causal-FA1-TC",
            "_Z12sdpa_fa_likePKfS0_S0_PffS1_S1_",
            128,
            true,
        ),
        ("Causal-FA2-TC", "_Z11sdpa_fa2_tcPKfS0_S0_Pff", 128, false),
    ];

    levels
        .iter()
        .map(|&(name, entry, threads, aux)| {
            let variant = |suffix: &str| {
                KernelRun::new(
                    &format!("04_causal_attention/{}-{}.ptx", name, suffix),
                    entry,
                    config(threads, aux),
                )
            };
            BenchmarkDef::equivalence(
                format!("({}-naive, {}-fused)", name, name),
                BenchmarkCategory::CausalAttention,
                variant("naive"),
                variant("fused"),
            )
        })
        .collect()
}
