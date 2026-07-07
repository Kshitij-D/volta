//! Logging support for volta_analysis.
//!
//! When the `logging` feature is enabled, this module re-exports the log crate macros.
//! When disabled, it provides no-op stub macros for zero overhead.

// When logging is enabled, re-export log crate macros
#[cfg(feature = "logging")]
pub use log::{debug, info, trace, warn};

// When logging is disabled, provide no-op stub macros
#[cfg(not(feature = "logging"))]
macro_rules! trace {
    ($($arg:tt)*) => {};
}

#[cfg(not(feature = "logging"))]
macro_rules! debug {
    ($($arg:tt)*) => {};
}

#[cfg(not(feature = "logging"))]
macro_rules! info {
    ($($arg:tt)*) => {};
}

#[cfg(not(feature = "logging"))]
macro_rules! warn_ {
    ($($arg:tt)*) => {};
}

#[cfg(not(feature = "logging"))]
pub(crate) use {debug, info, trace, warn_ as warn};
