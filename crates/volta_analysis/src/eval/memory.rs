//! Byte-addressed granule memory.
//!
//! Memory is a map from byte address to *granule*: a value tagged with the
//! width (in bytes) it was written at. Reads normally match a granule
//! exactly, with two sanctioned exceptions that arise from how nvcc handles
//! f16 data:
//!
//! - a 4-byte read over two adjacent 2-byte granules yields a packed
//!   `Value::Pair`, and
//! - a 2-byte read of either half of a 4-byte `Pair` granule yields that
//!   half (writes split such granules on demand).
//!
//! Any other reinterpretation (e.g. reading half of an f32) is an error, as
//! is reading bytes that were never written. Bounds are *not* checked here;
//! the interpreter validates accesses against declared regions first.

use std::collections::HashMap;

use crate::eval::value::Value;

/// Widest granule we ever store (8 bytes); bounds the overlap scans.
const MAX_WIDTH: u64 = 8;

/// A single granule: `width` bytes holding `value`.
///
/// `dirty` distinguishes values the program stored from values placed by
/// analysis setup (initial inputs, lazily materialized input symbols); the
/// dirty cells of an output array are the kernel's output footprint.
#[derive(Debug, Clone, Copy)]
struct Cell {
    width: u64,
    value: Value,
    dirty: bool,
}

/// Why a memory access failed. The interpreter attaches thread/pc/space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemAccessError {
    /// Read of bytes never written (address of the first missing byte).
    Uninitialized { addr: u64 },
    /// Access at a width incompatible with the granule(s) present.
    Reinterpret { addr: u64, width: u64 },
}

/// One memory space (global, shared, or one thread's local).
#[derive(Debug, Clone, Default)]
pub struct Memory {
    cells: HashMap<u64, Cell>,
}

impl Memory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Read `width` bytes at `addr`.
    pub fn read(&self, addr: u64, width: u64) -> Result<Value, MemAccessError> {
        // Exact granule match.
        if let Some(cell) = self.cells.get(&addr)
            && cell.width == width
        {
            return Ok(cell.value);
        }

        // 4-byte read combining two adjacent 2-byte scalars into a pair.
        if width == 4
            && let (Some(lo), Some(hi)) = (self.cells.get(&addr), self.cells.get(&(addr + 2)))
            && let (
                Cell {
                    width: 2,
                    value: Value::Scalar(l),
                    ..
                },
                Cell {
                    width: 2,
                    value: Value::Scalar(h),
                    ..
                },
            ) = (lo, hi)
        {
            return Ok(Value::Pair(*l, *h));
        }

        // 2-byte read of one half of a 4-byte pair granule.
        if width == 2 {
            if let Some(Cell {
                width: 4,
                value: Value::Pair(lo, _),
                ..
            }) = self.cells.get(&addr)
            {
                return Ok(Value::Scalar(*lo));
            }
            if addr >= 2
                && let Some(Cell {
                    width: 4,
                    value: Value::Pair(_, hi),
                    ..
                }) = self.cells.get(&(addr - 2))
            {
                return Ok(Value::Scalar(*hi));
            }
        }

        // Failed: distinguish "bytes present at another width" from "missing".
        for byte in addr..addr + width {
            if self.covering_cell(byte).is_some() {
                return Err(MemAccessError::Reinterpret { addr, width });
            }
        }
        Err(MemAccessError::Uninitialized { addr })
    }

    /// Write `width` bytes at `addr` on behalf of the program (marks the
    /// granule dirty).
    pub fn write(&mut self, addr: u64, width: u64, value: Value) -> Result<(), MemAccessError> {
        self.put(addr, width, value, true)
    }

    /// Place an analysis-setup value (initial input or module global);
    /// the granule is not part of the program's output footprint.
    pub fn init(&mut self, addr: u64, width: u64, value: Value) -> Result<(), MemAccessError> {
        self.put(addr, width, value, false)
    }

    /// Whether any granule starts at `addr` (used to avoid re-materializing
    /// lazily-created input symbols).
    pub fn has_cell_at(&self, addr: u64) -> bool {
        self.cells.contains_key(&addr)
    }

    /// The dirty granules (program-written), as `(addr, width, value)`.
    pub fn dirty_cells(&self) -> impl Iterator<Item = (u64, u64, Value)> + '_ {
        self.cells
            .iter()
            .filter(|(_, c)| c.dirty)
            .map(|(&addr, c)| (addr, c.width, c.value))
    }

    /// Store a granule, replacing fully-covered granules and splitting
    /// partially-covered `Pair` granules. A partial overlap with any other
    /// granule is a reinterpretation error.
    fn put(
        &mut self,
        addr: u64,
        width: u64,
        value: Value,
        dirty: bool,
    ) -> Result<(), MemAccessError> {
        let end = addr + width;
        loop {
            let mut covered: Vec<u64> = Vec::new();
            let mut partial: Option<u64> = None;

            let scan_start = addr.saturating_sub(MAX_WIDTH - 1);
            for start in scan_start..end {
                let Some(cell) = self.cells.get(&start) else {
                    continue;
                };
                let cell_end = start + cell.width;
                if cell_end <= addr || start >= end {
                    continue; // no overlap
                }
                if start >= addr && cell_end <= end {
                    covered.push(start);
                } else {
                    partial = Some(start);
                    break;
                }
            }

            if let Some(start) = partial {
                // Only a 4-byte pair granule can be split to resolve a
                // partial overlap; anything else is a reinterpretation.
                self.split_pair(start, addr, width)?;
                continue; // re-scan with the split applied
            }

            for start in covered {
                self.cells.remove(&start);
            }
            self.cells.insert(
                addr,
                Cell {
                    width,
                    value,
                    dirty,
                },
            );
            return Ok(());
        }
    }

    /// Split the 4-byte `Pair` granule at `start` into two 2-byte scalars,
    /// preserving its dirtiness. `(addr, width)` identify the offending
    /// access for error reporting.
    fn split_pair(&mut self, start: u64, addr: u64, width: u64) -> Result<(), MemAccessError> {
        match self.cells.get(&start) {
            Some(Cell {
                width: 4,
                value: Value::Pair(lo, hi),
                dirty,
            }) => {
                let (lo, hi, dirty) = (*lo, *hi, *dirty);
                self.cells.remove(&start);
                self.cells.insert(
                    start,
                    Cell {
                        width: 2,
                        value: Value::Scalar(lo),
                        dirty,
                    },
                );
                self.cells.insert(
                    start + 2,
                    Cell {
                        width: 2,
                        value: Value::Scalar(hi),
                        dirty,
                    },
                );
                Ok(())
            }
            _ => Err(MemAccessError::Reinterpret { addr, width }),
        }
    }

    /// Find the granule covering `byte`, if any.
    fn covering_cell(&self, byte: u64) -> Option<u64> {
        let scan_start = byte.saturating_sub(MAX_WIDTH - 1);
        for start in scan_start..=byte {
            if let Some(cell) = self.cells.get(&start)
                && start + cell.width > byte
            {
                return Some(start);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::ExprArena;

    fn scalars(arena: &mut ExprArena, n: i64) -> Value {
        Value::Scalar(arena.int(n))
    }

    #[test]
    fn test_exact_roundtrip() {
        let mut arena = ExprArena::new();
        let mut mem = Memory::new();
        let v = scalars(&mut arena, 42);
        mem.write(0x100, 4, v).unwrap();
        assert_eq!(mem.read(0x100, 4).unwrap(), v);
    }

    #[test]
    fn test_uninitialized_read() {
        let mem = Memory::new();
        assert_eq!(
            mem.read(0x100, 4),
            Err(MemAccessError::Uninitialized { addr: 0x100 })
        );
    }

    #[test]
    fn test_overwrite() {
        let mut arena = ExprArena::new();
        let mut mem = Memory::new();
        mem.write(0x100, 4, scalars(&mut arena, 1)).unwrap();
        let v2 = scalars(&mut arena, 2);
        mem.write(0x100, 4, v2).unwrap();
        assert_eq!(mem.read(0x100, 4).unwrap(), v2);
    }

    #[test]
    fn test_combine_halves_into_pair() {
        let mut arena = ExprArena::new();
        let mut mem = Memory::new();
        let lo = arena.named("lo");
        let hi = arena.named("hi");
        mem.write(0x10, 2, Value::Scalar(lo)).unwrap();
        mem.write(0x12, 2, Value::Scalar(hi)).unwrap();
        assert_eq!(mem.read(0x10, 4).unwrap(), Value::Pair(lo, hi));
    }

    #[test]
    fn test_split_pair_on_half_read() {
        let mut arena = ExprArena::new();
        let mut mem = Memory::new();
        let lo = arena.named("lo");
        let hi = arena.named("hi");
        mem.write(0x10, 4, Value::Pair(lo, hi)).unwrap();
        assert_eq!(mem.read(0x10, 2).unwrap(), Value::Scalar(lo));
        assert_eq!(mem.read(0x12, 2).unwrap(), Value::Scalar(hi));
    }

    #[test]
    fn test_split_pair_on_half_write() {
        let mut arena = ExprArena::new();
        let mut mem = Memory::new();
        let lo = arena.named("lo");
        let hi = arena.named("hi");
        mem.write(0x10, 4, Value::Pair(lo, hi)).unwrap();
        // Overwrite just the low half; the high half must survive.
        let new_lo = arena.named("new_lo");
        mem.write(0x10, 2, Value::Scalar(new_lo)).unwrap();
        assert_eq!(mem.read(0x10, 2).unwrap(), Value::Scalar(new_lo));
        assert_eq!(mem.read(0x12, 2).unwrap(), Value::Scalar(hi));
    }

    #[test]
    fn test_scalar_half_read_is_reinterpretation() {
        let mut arena = ExprArena::new();
        let mut mem = Memory::new();
        mem.write(0x10, 4, scalars(&mut arena, 5)).unwrap();
        assert_eq!(
            mem.read(0x10, 2),
            Err(MemAccessError::Reinterpret {
                addr: 0x10,
                width: 2
            })
        );
    }

    #[test]
    fn test_wide_write_replaces_halves() {
        let mut arena = ExprArena::new();
        let mut mem = Memory::new();
        mem.write(0x10, 2, scalars(&mut arena, 1)).unwrap();
        mem.write(0x12, 2, scalars(&mut arena, 2)).unwrap();
        let v = scalars(&mut arena, 3);
        mem.write(0x10, 4, v).unwrap();
        assert_eq!(mem.read(0x10, 4).unwrap(), v);
        // Old halves are gone.
        assert_eq!(
            mem.read(0x10, 2),
            Err(MemAccessError::Reinterpret {
                addr: 0x10,
                width: 2
            })
        );
    }

    #[test]
    fn test_partial_scalar_overlap_is_error() {
        let mut arena = ExprArena::new();
        let mut mem = Memory::new();
        mem.write(0x10, 4, scalars(&mut arena, 1)).unwrap();
        // A 4-byte write overlapping half of the previous scalar granule.
        let v = scalars(&mut arena, 2);
        assert_eq!(
            mem.write(0x12, 4, v),
            Err(MemAccessError::Reinterpret {
                addr: 0x12,
                width: 4
            })
        );
    }
}
