pub mod file_cache;
pub mod report;

/// A span in the source code (start and end byte offsets)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Span(pub usize, pub usize);
