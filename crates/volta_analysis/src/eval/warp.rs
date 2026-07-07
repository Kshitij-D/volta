//! Warp-cooperative operations: `bar.warp.sync`, `shfl.sync`, `ldmatrix`,
//! `mma.sync`, and the `wmma` family.
//!
//! Each of these is modeled as the paper's `sync I` for the participating
//! lane group, followed by the operation's cooperative data movement /
//! compute. The group fires only once every participating lane has arrived
//! at the same pc with the same mask (see `Interpreter::find_ready_warp_group`).
//!
//! Memory reads/writes performed by these ops are attributed to the exact
//! lane that owns each fragment element (per the PTX fragment tables in
//! `tensor_core`), so race checking stays byte- and thread-precise.

use volta_frontend::ast::ScalarType;

use crate::eval::error::{EvalError, EvalResult};
use crate::eval::interp::Interpreter;
use crate::eval::value::Value;
use crate::eval::{ThreadId, WARP_SIZE};
use crate::lowered::{InstrId, LoweredInstr, MemSpace, Operand, ShflMode};
use crate::symbolic::ExprId;
use crate::symbols::RegId;
use crate::tensor_core::{
    FragmentElement, MmaLayout, MmaOperand, MmaShape, m16n8k16_f16, m16n16k16_f16,
};

/// A dense matrix of expressions being assembled from lane fragments.
struct Grid {
    cols: usize,
    cells: Vec<Option<ExprId>>,
}

impl Grid {
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            cols,
            cells: vec![None; rows * cols],
        }
    }

    fn set(&mut self, row: u32, col: u32, e: ExprId) {
        self.cells[row as usize * self.cols + col as usize] = Some(e);
    }

    fn get(&self, row: u32, col: u32, pc: InstrId) -> EvalResult<ExprId> {
        self.cells[row as usize * self.cols + col as usize].ok_or_else(|| EvalError::Unsupported {
            pc,
            what: format!("incomplete fragment mapping at ({}, {})", row, col),
        })
    }
}

/// Element offset (in elements) within a matrix laid out in memory.
fn elem_offset(layout: MmaLayout, row: u32, col: u32, stride: u64) -> u64 {
    match layout {
        MmaLayout::Row => row as u64 * stride + col as u64,
        MmaLayout::Col => col as u64 * stride + row as u64,
    }
}

impl Interpreter<'_> {
    /// Execute a complete warp group blocked at `pc` with lane mask `mask`.
    pub(in crate::eval) fn execute_warp_op(
        &mut self,
        pc: InstrId,
        mask: u32,
        members: &[ThreadId],
    ) -> EvalResult<()> {
        let instr = self
            .program
            .instruction(pc)
            .expect("warp group blocked at a valid pc")
            .clone();

        // Every warp-cooperative op is a synchronization point for its group
        // (the paper's `sync I`). The group's accesses are bracketed by
        // syncs: the sync *before* keeps the op's reads from racing with the
        // lanes' own pre-op writes (ldmatrix after st.shared), and the sync
        // *after* (below) keeps the op's writes from racing with the lanes'
        // post-op reads (wmma.store followed by per-lane ld.shared) - a
        // converged warp is synchronized on both sides of the op.
        self.stats.warp_syncs += 1;
        self.sync_warp_group(members);

        match &instr {
            LoweredInstr::BarWarpSync { .. } => {}
            LoweredInstr::ShflSync {
                mode,
                dst,
                dst_pred,
                src,
                offset_or_lane,
                clamp,
                ..
            } => {
                self.exec_shfl_sync(
                    pc,
                    mask,
                    members,
                    *mode,
                    *dst,
                    *dst_pred,
                    src,
                    offset_or_lane,
                    clamp,
                )?;
            }
            LoweredInstr::Ldmatrix {
                dst,
                addr,
                num,
                trans,
            } => {
                self.exec_ldmatrix(pc, members, dst, addr, *num, *trans)?;
            }
            LoweredInstr::Mma { .. } => self.exec_mma(pc, members, &instr)?,
            LoweredInstr::WmmaLoad { .. } => self.exec_wmma_load(pc, members, &instr)?,
            LoweredInstr::WmmaStore { .. } => self.exec_wmma_store(pc, members, &instr)?,
            LoweredInstr::WmmaMma { .. } => self.exec_wmma_mma(pc, members, &instr)?,
            other => {
                return Err(EvalError::Unsupported {
                    pc,
                    what: format!("warp-op dispatch for {:?}", other),
                });
            }
        }

        self.sync_warp_group(members);
        self.advance_warp_group(members);
        Ok(())
    }

    /// `shfl.sync`: exchange register values within the mask group,
    /// following the PTX ISA source-lane computation.
    #[allow(clippy::too_many_arguments)]
    fn exec_shfl_sync(
        &mut self,
        pc: InstrId,
        mask: u32,
        members: &[ThreadId],
        mode: ShflMode,
        dst: RegId,
        dst_pred: Option<RegId>,
        src: &Operand,
        offset_or_lane: &Operand,
        clamp: &Operand,
    ) -> EvalResult<()> {
        // Gather every lane's source value first.
        let mut lane_src: [Option<Value>; WARP_SIZE as usize] = [None; WARP_SIZE as usize];
        for &m in members {
            let lane = (m.0 % WARP_SIZE) as usize;
            lane_src[lane] = Some(self.operand_value(m, pc, src)?);
        }

        // Compute each lane's source lane per the PTX ISA pseudocode.
        let mut results: Vec<(ThreadId, bool, Value)> = Vec::with_capacity(members.len());
        for &m in members {
            let lane = (m.0 % WARP_SIZE) as i64;
            let b = self.concrete_operand(m, pc, offset_or_lane, "shfl.sync lane operand")?;
            let c = self.concrete_operand(m, pc, clamp, "shfl.sync clamp operand")?;
            let bval = b & 0x1f;
            let cval = c & 0x1f;
            let segmask = (c >> 8) & 0x1f;
            let max_lane = (lane & segmask) | (cval & !segmask);
            let min_lane = lane & segmask;
            let (j0, pval) = match mode {
                ShflMode::Up => {
                    let j = lane - bval;
                    (j, j >= max_lane)
                }
                ShflMode::Down => {
                    let j = lane + bval;
                    (j, j <= max_lane)
                }
                ShflMode::Bfly => {
                    let j = lane ^ bval;
                    (j, j <= max_lane)
                }
                ShflMode::Idx => {
                    let j = min_lane | (bval & !segmask);
                    (j, j <= max_lane)
                }
            };
            let j = if pval { j0 } else { lane };
            debug_assert!((0..WARP_SIZE as i64).contains(&j));
            if mask & (1u32 << j) == 0 {
                return Err(EvalError::WarpMismatch {
                    pc,
                    reason: format!(
                        "lane {} reads lane {} which is outside mask {:#010x}",
                        lane, j, mask
                    ),
                });
            }
            let value = lane_src[j as usize].expect("mask lanes were gathered");
            results.push((m, pval, value));
        }

        for (m, pval, value) in results {
            self.threads[m].regs.write(dst, value);
            if let Some(p) = dst_pred {
                let b = self.arena.bool_val(pval);
                self.threads[m].regs.write(p, Value::Scalar(b));
            }
        }
        Ok(())
    }

    /// `ldmatrix.sync.aligned.xN.m8n8.shared.b16`: cooperative load of N
    /// 8x8 b16 matrices. Lane `i*8 + r` supplies the address of row `r` of
    /// matrix `i`; lane `l` receives elements (row `l/4`, cols `(l%4)*2`,
    /// `(l%4)*2+1`) of each matrix as a packed pair.
    fn exec_ldmatrix(
        &mut self,
        pc: InstrId,
        members: &[ThreadId],
        dst: &[RegId],
        addr: &Operand,
        num: u32,
        trans: bool,
    ) -> EvalResult<()> {
        if trans {
            return Err(EvalError::Unsupported {
                pc,
                what: "ldmatrix .trans".to_string(),
            });
        }
        if members.len() != WARP_SIZE as usize {
            return Err(EvalError::WarpMismatch {
                pc,
                reason: "ldmatrix requires a full warp".to_string(),
            });
        }
        if dst.len() != num as usize {
            return Err(EvalError::Unsupported {
                pc,
                what: format!("ldmatrix x{} with {} destination registers", num, dst.len()),
            });
        }

        // Row addresses come from the first num*8 lanes.
        let mut row_addr = vec![[0u64; 8]; num as usize];
        for i in 0..num as usize {
            for r in 0..8 {
                let m = members[i * 8 + r];
                row_addr[i][r] = self.concrete_operand(m, pc, addr, "ldmatrix row address")? as u64;
            }
        }

        for &m in members {
            let lane = m.0 % WARP_SIZE;
            for (i, reg) in dst.iter().enumerate() {
                let byte = row_addr[i][(lane / 4) as usize] + (lane % 4) as u64 * 4;
                let v = self.mem_read(m, pc, MemSpace::Shared, byte, 4)?;
                self.threads[m].regs.write(*reg, v);
            }
        }
        Ok(())
    }

    /// `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
    fn exec_mma(
        &mut self,
        pc: InstrId,
        members: &[ThreadId],
        instr: &LoweredInstr,
    ) -> EvalResult<()> {
        let LoweredInstr::Mma {
            shape,
            dst,
            src_a,
            src_b,
            src_c,
            a_layout,
            b_layout,
            ..
        } = instr
        else {
            unreachable!()
        };

        if *shape != MmaShape::new(16, 8, 16) {
            return Err(EvalError::Unsupported {
                pc,
                what: format!("mma shape {}", shape),
            });
        }
        if (*a_layout, *b_layout) != (MmaLayout::Row, MmaLayout::Col) {
            return Err(EvalError::Unsupported {
                pc,
                what: "mma with non-row.col layouts".to_string(),
            });
        }
        if members.len() != WARP_SIZE as usize {
            return Err(EvalError::WarpMismatch {
                pc,
                reason: "mma.sync requires a full warp".to_string(),
            });
        }

        let mut a = Grid::new(16, 16);
        let mut b = Grid::new(16, 8);
        let mut c = Grid::new(16, 8);
        for &m in members {
            let lane = m.0 % WARP_SIZE;
            self.gather_f16_fragment(pc, m, src_a, &m16n8k16_f16::matrix_a(lane), &mut a)?;
            self.gather_f16_fragment(pc, m, src_b, &m16n8k16_f16::matrix_b(lane), &mut b)?;
            self.gather_f32_fragment(pc, m, src_c, &m16n8k16_f16::matrix_cd(lane), &mut c)?;
        }

        let d = self.matmul_acc(pc, &a, &b, &c, 16, 8, 16)?;

        for &m in members {
            let lane = m.0 % WARP_SIZE;
            for elem in m16n8k16_f16::matrix_cd(lane) {
                let e = d.get(elem.row, elem.col, pc)?;
                self.threads[m]
                    .regs
                    .write(dst[elem.reg_idx], Value::Scalar(e));
            }
        }
        Ok(())
    }

    /// `wmma.load.{a,b,c}.sync.aligned.{row,col}.m16n16k16{.shared}.{f16,f32}`.
    fn exec_wmma_load(
        &mut self,
        pc: InstrId,
        members: &[ThreadId],
        instr: &LoweredInstr,
    ) -> EvalResult<()> {
        let LoweredInstr::WmmaLoad {
            operand,
            shape,
            layout,
            dst,
            addr,
            stride,
            elem_type,
            space,
        } = instr
        else {
            unreachable!()
        };
        self.check_wmma_shape(pc, members, *shape)?;

        let base = self.uniform_concrete(pc, members, addr, "wmma.load address")? as u64;
        let stride = self.uniform_concrete(pc, members, stride, "wmma.load stride")? as u64;

        match operand {
            MmaOperand::A | MmaOperand::B => {
                if !matches!(elem_type, ScalarType::F16 | ScalarType::B16) {
                    return Err(EvalError::Unsupported {
                        pc,
                        what: format!("wmma.load a/b with element type {:?}", elem_type),
                    });
                }
                for &m in members {
                    let lane = m.0 % WARP_SIZE;
                    let elems = match operand {
                        MmaOperand::A => m16n16k16_f16::matrix_a_row(lane),
                        _ => m16n16k16_f16::matrix_b_row(lane),
                    };
                    let mut lo: Vec<Option<ExprId>> = vec![None; dst.len()];
                    let mut hi: Vec<Option<ExprId>> = vec![None; dst.len()];
                    for elem in &elems {
                        let off = elem_offset(*layout, elem.row, elem.col, stride);
                        let byte = base + off * 2;
                        let v = self.mem_read(m, pc, *space, byte, 2)?;
                        let Value::Scalar(e) = v else {
                            return Err(EvalError::ValueKindMismatch {
                                thread: m,
                                pc,
                                what: "wmma.load f16 element is not a scalar",
                            });
                        };
                        if elem.high_half == Some(true) {
                            hi[elem.reg_idx] = Some(e);
                        } else {
                            lo[elem.reg_idx] = Some(e);
                        }
                    }
                    for (r, reg) in dst.iter().enumerate() {
                        let (Some(l), Some(h)) = (lo[r], hi[r]) else {
                            return Err(EvalError::Unsupported {
                                pc,
                                what: "incomplete wmma fragment".to_string(),
                            });
                        };
                        self.threads[m].regs.write(*reg, Value::Pair(l, h));
                    }
                }
            }
            MmaOperand::C => {
                for &m in members {
                    let lane = m.0 % WARP_SIZE;
                    for elem in m16n16k16_f16::matrix_cd_f32(lane) {
                        let off = elem_offset(*layout, elem.row, elem.col, stride);
                        let byte = base + off * 4;
                        let v = self.mem_read(m, pc, *space, byte, 4)?;
                        let Value::Scalar(e) = v else {
                            return Err(EvalError::ValueKindMismatch {
                                thread: m,
                                pc,
                                what: "wmma.load f32 element is not a scalar",
                            });
                        };
                        self.threads[m]
                            .regs
                            .write(dst[elem.reg_idx], Value::Scalar(e));
                    }
                }
            }
            MmaOperand::D => {
                return Err(EvalError::Unsupported {
                    pc,
                    what: "wmma.load.d".to_string(),
                });
            }
        }
        Ok(())
    }

    /// `wmma.store.d.sync.aligned.{row,col}.m16n16k16{.shared}.f32`.
    fn exec_wmma_store(
        &mut self,
        pc: InstrId,
        members: &[ThreadId],
        instr: &LoweredInstr,
    ) -> EvalResult<()> {
        let LoweredInstr::WmmaStore {
            shape,
            layout,
            src,
            addr,
            stride,
            space,
            ..
        } = instr
        else {
            unreachable!()
        };
        self.check_wmma_shape(pc, members, *shape)?;

        let base = self.uniform_concrete(pc, members, addr, "wmma.store address")? as u64;
        let stride = self.uniform_concrete(pc, members, stride, "wmma.store stride")? as u64;

        for &m in members {
            let lane = m.0 % WARP_SIZE;
            for elem in m16n16k16_f16::matrix_cd_f32(lane) {
                let v = self.read_reg(m, pc, src[elem.reg_idx])?;
                let off = elem_offset(*layout, elem.row, elem.col, stride);
                let byte = base + off * 4;
                self.mem_write(m, pc, *space, byte, 4, v)?;
            }
        }
        Ok(())
    }

    /// `wmma.mma.sync.aligned.{row,col}.{row,col}.m16n16k16.f32.f32`.
    ///
    /// The layouts describe how A/B were loaded; the fragments themselves
    /// are opaque, so the compute only needs the fragment position maps.
    fn exec_wmma_mma(
        &mut self,
        pc: InstrId,
        members: &[ThreadId],
        instr: &LoweredInstr,
    ) -> EvalResult<()> {
        let LoweredInstr::WmmaMma {
            shape,
            dst,
            src_a,
            src_b,
            src_c,
            ..
        } = instr
        else {
            unreachable!()
        };
        self.check_wmma_shape(pc, members, *shape)?;

        let mut a = Grid::new(16, 16);
        let mut b = Grid::new(16, 16);
        let mut c = Grid::new(16, 16);
        for &m in members {
            let lane = m.0 % WARP_SIZE;
            self.gather_f16_fragment(pc, m, src_a, &m16n16k16_f16::matrix_a_row(lane), &mut a)?;
            self.gather_f16_fragment(pc, m, src_b, &m16n16k16_f16::matrix_b_row(lane), &mut b)?;
            self.gather_f32_fragment(pc, m, src_c, &m16n16k16_f16::matrix_cd_f32(lane), &mut c)?;
        }

        let d = self.matmul_acc(pc, &a, &b, &c, 16, 16, 16)?;

        for &m in members {
            let lane = m.0 % WARP_SIZE;
            for elem in m16n16k16_f16::matrix_cd_f32(lane) {
                let e = d.get(elem.row, elem.col, pc)?;
                self.threads[m]
                    .regs
                    .write(dst[elem.reg_idx], Value::Scalar(e));
            }
        }
        Ok(())
    }

    // =====================================================================
    // Shared helpers
    // =====================================================================

    fn check_wmma_shape(
        &self,
        pc: InstrId,
        members: &[ThreadId],
        shape: MmaShape,
    ) -> EvalResult<()> {
        if shape != MmaShape::new(16, 16, 16) {
            return Err(EvalError::Unsupported {
                pc,
                what: format!("wmma shape {}", shape),
            });
        }
        if members.len() != WARP_SIZE as usize {
            return Err(EvalError::WarpMismatch {
                pc,
                reason: "wmma requires a full warp".to_string(),
            });
        }
        Ok(())
    }

    /// Resolve an operand that must be concrete and identical on every lane.
    fn uniform_concrete(
        &mut self,
        pc: InstrId,
        members: &[ThreadId],
        op: &Operand,
        what: &'static str,
    ) -> EvalResult<i64> {
        let mut result: Option<i64> = None;
        for &m in members {
            let v = self.concrete_operand(m, pc, op, what)?;
            match result {
                None => result = Some(v),
                Some(prev) if prev == v => {}
                Some(prev) => {
                    return Err(EvalError::WarpMismatch {
                        pc,
                        reason: format!("{} differs across lanes ({} vs {})", what, prev, v),
                    });
                }
            }
        }
        result.ok_or(EvalError::WarpMismatch {
            pc,
            reason: "empty warp group".to_string(),
        })
    }

    /// Place one lane's packed-f16 fragment registers into a matrix grid.
    fn gather_f16_fragment(
        &mut self,
        pc: InstrId,
        m: ThreadId,
        regs: &[RegId],
        elems: &[FragmentElement],
        grid: &mut Grid,
    ) -> EvalResult<()> {
        for elem in elems {
            let v = self.read_reg(m, pc, regs[elem.reg_idx])?;
            let Value::Pair(lo, hi) = v else {
                return Err(EvalError::ValueKindMismatch {
                    thread: m,
                    pc,
                    what: "matrix fragment register does not hold a packed f16 pair",
                });
            };
            let e = if elem.high_half == Some(true) { hi } else { lo };
            grid.set(elem.row, elem.col, e);
        }
        Ok(())
    }

    /// Place one lane's f32 accumulator fragment registers into a grid.
    fn gather_f32_fragment(
        &mut self,
        pc: InstrId,
        m: ThreadId,
        regs: &[RegId],
        elems: &[FragmentElement],
        grid: &mut Grid,
    ) -> EvalResult<()> {
        for elem in elems {
            let v = self.read_reg(m, pc, regs[elem.reg_idx])?;
            let Value::Scalar(e) = v else {
                return Err(EvalError::ValueKindMismatch {
                    thread: m,
                    pc,
                    what: "accumulator fragment register holds a packed pair",
                });
            };
            grid.set(elem.row, elem.col, e);
        }
        Ok(())
    }

    /// `D = A * B + C` over the arena, as an fma chain per element.
    #[allow(clippy::too_many_arguments)] // internal helper; the args are the mma shape
    fn matmul_acc(
        &mut self,
        pc: InstrId,
        a: &Grid,
        b: &Grid,
        c: &Grid,
        m: u32,
        n: u32,
        k: u32,
    ) -> EvalResult<Grid> {
        let mut d = Grid::new(m as usize, n as usize);
        for i in 0..m {
            for j in 0..n {
                let mut acc = c.get(i, j, pc)?;
                for kk in 0..k {
                    let av = a.get(i, kk, pc)?;
                    let bv = b.get(kk, j, pc)?;
                    acc = self.arena.fma(av, bv, acc);
                }
                d.set(i, j, acc);
            }
        }
        Ok(d)
    }
}
