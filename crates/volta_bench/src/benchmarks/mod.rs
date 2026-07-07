//! Benchmark definitions, one submodule per paper category.

pub mod agent;
pub mod attention;
pub mod causal;
pub mod conv2d;
pub mod matmul;
pub mod races;
pub mod reduction;
pub mod tilelang;

use crate::config::BenchmarkSuite;

/// All benchmarks in paper order.
pub fn all_benchmarks() -> BenchmarkSuite {
    let mut suite = BenchmarkSuite::new();
    suite.extend(reduction::benchmarks());
    suite.extend(matmul::benchmarks());
    suite.extend(attention::benchmarks());
    suite.extend(causal::benchmarks());
    suite.extend(conv2d::benchmarks());
    suite.extend(agent::benchmarks());
    suite.extend(tilelang::benchmarks());
    suite.extend(races::benchmarks());
    suite
}
