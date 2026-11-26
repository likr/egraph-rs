use egraph_core::{CoreError, CoreResult};

use crate::csr_matrix::CsrMatrix;

use super::workspace::ChebyshevWorkspace;

pub(super) const CHEBYSHEV_TILE_COLS: usize = 16;

pub(super) fn validate_csr_indices_in_bounds(mat: &CsrMatrix) -> CoreResult<()> {
    let n = mat.num_nodes();
    for &col in mat.indices() {
        if (col as usize) >= n {
            return Err(CoreError::index_out_of_bounds(
                col as usize,
                n,
                "ChebyshevWorkspace::csr_indices",
            ));
        }
    }
    Ok(())
}

impl ChebyshevWorkspace {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fused_step_const<const W: usize>(
        mat: &CsrMatrix,
        unshifted_scale: Option<f64>,
        prev: &[f64],
        curr: &[f64],
        next: &mut [f64],
        ca: f64,
        out_a: Option<&mut [f64]>,
        cb: f64,
        out_b: Option<&mut [f64]>,
        is_k1: bool,
    ) -> CoreResult<()> {
        let n = mat.num_nodes();
        let len = n
            .checked_mul(W)
            .ok_or_else(|| CoreError::invalid_input("ChebyshevWorkspace tile size overflow"))?;
        if curr.len() < len || next.len() < len || (!is_k1 && prev.len() < len) {
            return Err(CoreError::dimension_mismatch(
                len,
                curr.len().min(next.len()),
                "ChebyshevWorkspace::fused_step_const",
            ));
        }

        let (curr_rows, curr_rem) = curr[..len].as_chunks::<W>();
        let (next_rows, next_rem) = next[..len].as_chunks_mut::<W>();
        if curr_rows.len() != n
            || !curr_rem.is_empty()
            || next_rows.len() != n
            || !next_rem.is_empty()
        {
            return Err(CoreError::dimension_mismatch(
                len,
                curr.len().min(next.len()),
                "ChebyshevWorkspace::fused_step_const::chunks",
            ));
        }

        let prev_rows: &[[f64; W]] = if is_k1 {
            &[]
        } else {
            let (rows, rem) = prev[..len].as_chunks::<W>();
            if rows.len() != n || !rem.is_empty() {
                return Err(CoreError::dimension_mismatch(
                    len,
                    prev.len(),
                    "ChebyshevWorkspace::fused_step_const::prev",
                ));
            }
            rows
        };

        let mut oa_rows = match out_a {
            Some(oa) => {
                if oa.len() < len {
                    return Err(CoreError::dimension_mismatch(
                        len,
                        oa.len(),
                        "ChebyshevWorkspace::fused_step_const::out_a",
                    ));
                }
                let (rows, rem) = oa[..len].as_chunks_mut::<W>();
                if rows.len() != n || !rem.is_empty() {
                    return Err(CoreError::dimension_mismatch(
                        len,
                        oa.len(),
                        "ChebyshevWorkspace::fused_step_const::out_a",
                    ));
                }
                Some(rows)
            }
            None => None,
        };

        let mut ob_rows = match out_b {
            Some(ob) => {
                if ob.len() < len {
                    return Err(CoreError::dimension_mismatch(
                        len,
                        ob.len(),
                        "ChebyshevWorkspace::fused_step_const::out_b",
                    ));
                }
                let (rows, rem) = ob[..len].as_chunks_mut::<W>();
                if rows.len() != n || !rem.is_empty() {
                    return Err(CoreError::dimension_mismatch(
                        len,
                        ob.len(),
                        "ChebyshevWorkspace::fused_step_const::out_b",
                    ));
                }
                Some(rows)
            }
            None => None,
        };

        let indptr = mat.indptr();
        let indices = mat.indices();
        let values = mat.values();

        for (row, next_row) in next_rows.iter_mut().enumerate() {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            let mut acc = [0.0_f64; W];
            for (&col_u32, &mval) in indices[start..end].iter().zip(&values[start..end]) {
                let x_row = &curr_rows[col_u32 as usize];
                for j in 0..W {
                    acc[j] += mval * x_row[j];
                }
            }

            let mut nxt = [0.0_f64; W];
            if let Some(alpha) = unshifted_scale {
                let curr_row = &curr_rows[row];
                if is_k1 {
                    for j in 0..W {
                        nxt[j] = alpha * acc[j] - curr_row[j];
                    }
                } else {
                    let prev_row = &prev_rows[row];
                    for j in 0..W {
                        nxt[j] = 2.0 * (alpha * acc[j] - curr_row[j]) - prev_row[j];
                    }
                }
            } else if is_k1 {
                nxt = acc;
            } else {
                let prev_row = &prev_rows[row];
                for j in 0..W {
                    nxt[j] = 2.0 * acc[j] - prev_row[j];
                }
            }

            *next_row = nxt;
            if let Some(oa) = oa_rows.as_deref_mut() {
                let row_oa = &mut oa[row];
                for j in 0..W {
                    row_oa[j] += ca * nxt[j];
                }
            }
            if let Some(ob) = ob_rows.as_deref_mut() {
                let row_ob = &mut ob[row];
                for j in 0..W {
                    row_ob[j] += cb * nxt[j];
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn fused_step_tail(
        mat: &CsrMatrix,
        unshifted_scale: Option<f64>,
        prev: &[f64],
        curr: &[f64],
        next: &mut [f64],
        ca: f64,
        mut out_a: Option<&mut [f64]>,
        cb: f64,
        mut out_b: Option<&mut [f64]>,
        w: usize,
        is_k1: bool,
    ) -> CoreResult<()> {
        if w == 0 || w > CHEBYSHEV_TILE_COLS {
            return Err(CoreError::invalid_input(format!(
                "ChebyshevWorkspace tile width must be in 1..={CHEBYSHEV_TILE_COLS}, got {w}"
            )));
        }
        let n = mat.num_nodes();
        let len = n
            .checked_mul(w)
            .ok_or_else(|| CoreError::invalid_input("ChebyshevWorkspace tile size overflow"))?;
        if curr.len() < len || next.len() < len || (!is_k1 && prev.len() < len) {
            return Err(CoreError::dimension_mismatch(
                len,
                curr.len().min(next.len()),
                "ChebyshevWorkspace::fused_step_tail",
            ));
        }
        if let Some(oa) = out_a.as_deref()
            && oa.len() < len
        {
            return Err(CoreError::dimension_mismatch(
                len,
                oa.len(),
                "ChebyshevWorkspace::fused_step_tail::out_a",
            ));
        }
        if let Some(ob) = out_b.as_deref()
            && ob.len() < len
        {
            return Err(CoreError::dimension_mismatch(
                len,
                ob.len(),
                "ChebyshevWorkspace::fused_step_tail::out_b",
            ));
        }

        let indptr = mat.indptr();
        let indices = mat.indices();
        let values = mat.values();
        let curr = &curr[..len];
        let next = &mut next[..len];

        for row in 0..n {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            let mut acc = [0.0_f64; CHEBYSHEV_TILE_COLS];
            for (&col_u32, &mval) in indices[start..end].iter().zip(&values[start..end]) {
                let x_off = (col_u32 as usize) * w;
                let x_row = &curr[x_off..x_off + w];
                for j in 0..w {
                    acc[j] += mval * x_row[j];
                }
            }

            let row_off = row * w;
            let mut nxt = [0.0_f64; CHEBYSHEV_TILE_COLS];
            if let Some(alpha) = unshifted_scale {
                let curr_row = &curr[row_off..row_off + w];
                if is_k1 {
                    for j in 0..w {
                        nxt[j] = alpha * acc[j] - curr_row[j];
                    }
                } else {
                    let prev_row = &prev[row_off..row_off + w];
                    for j in 0..w {
                        nxt[j] = 2.0 * (alpha * acc[j] - curr_row[j]) - prev_row[j];
                    }
                }
            } else if is_k1 {
                nxt[..w].copy_from_slice(&acc[..w]);
            } else {
                let prev_row = &prev[row_off..row_off + w];
                for j in 0..w {
                    nxt[j] = 2.0 * acc[j] - prev_row[j];
                }
            }

            next[row_off..row_off + w].copy_from_slice(&nxt[..w]);
            if let Some(oa) = out_a.as_deref_mut() {
                let row_oa = &mut oa[row_off..row_off + w];
                for j in 0..w {
                    row_oa[j] += ca * nxt[j];
                }
            }
            if let Some(ob) = out_b.as_deref_mut() {
                let row_ob = &mut ob[row_off..row_off + w];
                for j in 0..w {
                    row_ob[j] += cb * nxt[j];
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn dispatch_fused_step(
        mat: &CsrMatrix,
        unshifted_scale: Option<f64>,
        prev: &[f64],
        curr: &[f64],
        next: &mut [f64],
        ca: f64,
        out_a: Option<&mut [f64]>,
        cb: f64,
        out_b: Option<&mut [f64]>,
        w: usize,
        is_k1: bool,
    ) -> CoreResult<()> {
        match w {
            16 => Self::fused_step_const::<16>(
                mat,
                unshifted_scale,
                prev,
                curr,
                next,
                ca,
                out_a,
                cb,
                out_b,
                is_k1,
            ),
            8 => Self::fused_step_const::<8>(
                mat,
                unshifted_scale,
                prev,
                curr,
                next,
                ca,
                out_a,
                cb,
                out_b,
                is_k1,
            ),
            4 => Self::fused_step_const::<4>(
                mat,
                unshifted_scale,
                prev,
                curr,
                next,
                ca,
                out_a,
                cb,
                out_b,
                is_k1,
            ),
            1 => Self::fused_step_const::<1>(
                mat,
                unshifted_scale,
                prev,
                curr,
                next,
                ca,
                out_a,
                cb,
                out_b,
                is_k1,
            ),
            _ => Self::fused_step_tail(
                mat,
                unshifted_scale,
                prev,
                curr,
                next,
                ca,
                out_a,
                cb,
                out_b,
                w,
                is_k1,
            ),
        }
    }

    pub(super) fn run_tile_degrees(
        &mut self,
        mat: &CsrMatrix,
        unshifted_scale: Option<f64>,
        coeffs_a: &[f64],
        coeffs_b: Option<&[f64]>,
        w: usize,
    ) -> CoreResult<()> {
        let n = mat.num_nodes();
        let tile_len = n * w;
        let max_len = coeffs_a.len().max(coeffs_b.map_or(0, |b| b.len()));
        if max_len <= 1 || n == 0 || w == 0 {
            return Ok(());
        }

        let ca1 = coeffs_a.get(1).copied().unwrap_or(0.0);
        let cb1 = coeffs_b.and_then(|cb| cb.get(1).copied()).unwrap_or(0.0);
        Self::dispatch_fused_step(
            mat,
            unshifted_scale,
            &[],
            &self.t_prev[..tile_len],
            &mut self.t_curr[..tile_len],
            ca1,
            if ca1 != 0.0 {
                Some(&mut self.out_a_tile[..tile_len])
            } else {
                None
            },
            cb1,
            if coeffs_b.is_some() && cb1 != 0.0 {
                Some(&mut self.out_b_tile[..tile_len])
            } else {
                None
            },
            w,
            true,
        )?;

        for k in 2..max_len {
            let ca = coeffs_a.get(k).copied().unwrap_or(0.0);
            let cb = coeffs_b.and_then(|b| b.get(k).copied()).unwrap_or(0.0);
            Self::dispatch_fused_step(
                mat,
                unshifted_scale,
                &self.t_prev[..tile_len],
                &self.t_curr[..tile_len],
                &mut self.t_next[..tile_len],
                ca,
                if ca != 0.0 {
                    Some(&mut self.out_a_tile[..tile_len])
                } else {
                    None
                },
                cb,
                if coeffs_b.is_some() && cb != 0.0 {
                    Some(&mut self.out_b_tile[..tile_len])
                } else {
                    None
                },
                w,
                false,
            )?;
            std::mem::swap(&mut self.t_prev, &mut self.t_curr);
            std::mem::swap(&mut self.t_curr, &mut self.t_next);
        }
        Ok(())
    }
}
