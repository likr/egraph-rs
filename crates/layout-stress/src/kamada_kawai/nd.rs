use egraph_core::{CoreError, CoreResult, IterativeSolver, LayoutNd, Scalar, SolverStatus};

use super::{HESSIAN_DET_TOL, KamadaKawaiSolver, MAX_NEWTON_STEP, MIN_SEPARATION};

#[inline]
fn euclid_norm(vec: &[f64]) -> f64 {
    let mut sum_sq = 0.0_f64;
    for &v in vec {
        sum_sq += v * v;
    }
    if sum_sq.is_finite() {
        return sum_sq.sqrt();
    }
    let mut max_abs = 0.0_f64;
    for &v in vec {
        let av = v.abs();
        if av > max_abs {
            max_abs = av;
        }
    }
    if !max_abs.is_finite() || max_abs == 0.0 {
        return max_abs;
    }
    let mut scaled_sq = 0.0_f64;
    for &v in vec {
        let r = v / max_abs;
        scaled_sq += r * r;
    }
    max_abs * scaled_sq.sqrt()
}

#[inline]
fn euclid_dist_slice_vs_flat(a: &[f64], coords: &[f64], node_j: usize, d: usize) -> f64 {
    let base_j = node_j * d;
    let mut sum_sq = 0.0_f64;
    for k in 0..d {
        let diff = a[k] - coords[base_j + k];
        sum_sq += diff * diff;
    }
    if sum_sq.is_finite() {
        return sum_sq.sqrt();
    }
    let mut max_abs = 0.0_f64;
    for k in 0..d {
        let av = (a[k] - coords[base_j + k]).abs();
        if av > max_abs {
            max_abs = av;
        }
    }
    if !max_abs.is_finite() || max_abs == 0.0 {
        return max_abs;
    }
    let mut scaled_sq = 0.0_f64;
    for k in 0..d {
        let r = (a[k] - coords[base_j + k]) / max_abs;
        scaled_sq += r * r;
    }
    max_abs * scaled_sq.sqrt()
}

/// Solves `h * out = rhs` (`d x d` dense row-major) in-place via Gaussian elimination
/// with partial pivoting. Returns `false` if any pivot magnitude is `<= pivot_tol` or non-finite.
fn solve_linear_system_pp(
    h: &mut [f64],
    rhs: &[f64],
    out: &mut [f64],
    d: usize,
    pivot_tol: f64,
) -> bool {
    out[..d].copy_from_slice(&rhs[..d]);

    for col in 0..d {
        let mut pivot_row = col;
        let mut pivot_abs = h[col * d + col].abs();
        for r in (col + 1)..d {
            let cand = h[r * d + col].abs();
            if cand > pivot_abs || (!pivot_abs.is_finite() && cand.is_finite()) {
                pivot_abs = cand;
                pivot_row = r;
            }
        }
        if !pivot_abs.is_finite() || pivot_abs <= pivot_tol {
            return false;
        }
        if pivot_row != col {
            let row_col = col * d;
            let row_piv = pivot_row * d;
            for k in col..d {
                h.swap(row_col + k, row_piv + k);
            }
            out.swap(col, pivot_row);
        }

        let diag = h[col * d + col];
        for row in (col + 1)..d {
            let factor = h[row * d + col] / diag;
            if !factor.is_finite() {
                return false;
            }
            let row_off = row * d;
            let col_off = col * d;
            for k in (col + 1)..d {
                h[row_off + k] -= factor * h[col_off + k];
            }
            out[row] -= factor * out[col];
        }
    }

    for row in (0..d).rev() {
        let row_off = row * d;
        let mut sum = out[row];
        for k in (row + 1)..d {
            sum -= h[row_off + k] * out[k];
        }
        let diag = h[row_off + row];
        if !diag.is_finite() || diag.abs() <= pivot_tol {
            return false;
        }
        let val = sum / diag;
        if !val.is_finite() {
            return false;
        }
        out[row] = val;
    }

    true
}

impl<S: Scalar> KamadaKawaiSolver<S> {
    fn select_node_nd(&mut self, d: usize) -> Option<(usize, f64)> {
        let n = self.num_nodes;
        let eps_sq = self.config.eps * self.config.eps;
        let mut best_m = 0usize;
        let mut best_delta2 = 0.0_f64;

        for m in 0..n {
            self.nd_grad.fill(0.0);
            let xm_off = m * d;
            let row = m * n;
            for i in 0..n {
                if i == m {
                    continue;
                }
                let k_mi = self.k[row + i];
                if k_mi <= 0.0 {
                    continue;
                }
                let dist = euclid_dist_slice_vs_flat(
                    &self.nd_coords[xm_off..xm_off + d],
                    &self.nd_coords,
                    i,
                    d,
                );
                if dist > MIN_SEPARATION && dist.is_finite() {
                    let f = k_mi * (1.0 - self.l[row + i] / dist);
                    let xi_off = i * d;
                    for a in 0..d {
                        self.nd_grad[a] +=
                            f * (self.nd_coords[xm_off + a] - self.nd_coords[xi_off + a]);
                    }
                }
            }
            let delta2: f64 = self.nd_grad.iter().map(|&g| g * g).sum();
            if delta2.is_finite() && delta2 > best_delta2 {
                best_delta2 = delta2;
                best_m = m;
            }
        }

        if best_delta2 < eps_sq {
            None
        } else {
            Some((best_m, best_delta2))
        }
    }
}

impl<S: Scalar> IterativeSolver<S, LayoutNd<S>> for KamadaKawaiSolver<S> {
    fn step(&mut self, layout: &mut LayoutNd<S>) -> CoreResult<SolverStatus> {
        let d = layout.dims();
        if d == 0 {
            return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
        }
        if layout.num_nodes() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.num_nodes(),
                "KamadaKawaiSolver::step",
            ));
        }
        layout.validate_finite()?;
        if let Some(expected_dims) = self.nd_dims
            && d != expected_dims
        {
            return Err(CoreError::dimension_mismatch(
                expected_dims,
                d,
                "KamadaKawaiSolver::step::dims",
            ));
        }
        self.nd_dims = Some(d);

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let needed_coords = n
            .checked_mul(d)
            .ok_or_else(|| CoreError::invalid_input("LayoutNd coordinate buffer overflow"))?;
        let needed_hessian = d
            .checked_mul(d)
            .ok_or_else(|| CoreError::invalid_input("KamadaKawai Hessian size overflow"))?;
        if self.nd_coords.len() != needed_coords {
            self.nd_coords.resize(needed_coords, 0.0);
        }
        if self.nd_grad.len() != d {
            self.nd_grad.resize(d, 0.0);
            self.nd_delta.resize(d, 0.0);
            self.nd_xm.resize(d, 0.0);
        }
        if self.nd_hessian.len() != needed_hessian {
            self.nd_hessian.resize(needed_hessian, 0.0);
        }

        for (dst, &src) in self.nd_coords.iter_mut().zip(layout.coords().iter()) {
            *dst = src.to_f64();
        }

        let Some((m, _)) = self.select_node_nd(d) else {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        };

        let eps_sq = self.config.eps * self.config.eps;
        let row = m * n;
        self.nd_xm
            .copy_from_slice(&self.nd_coords[m * d..(m + 1) * d]);

        for _ in 0..self.config.max_inner_iters {
            self.nd_grad.fill(0.0);
            self.nd_hessian.fill(0.0);

            for i in 0..n {
                if i == m {
                    continue;
                }
                let k_mi = self.k[row + i];
                if k_mi <= 0.0 {
                    continue;
                }
                let l_mi = self.l[row + i];
                let dist = euclid_dist_slice_vs_flat(&self.nd_xm, &self.nd_coords, i, d);
                if dist <= MIN_SEPARATION || !dist.is_finite() {
                    continue;
                }
                let dist2 = dist * dist;
                let d3 = dist2 * dist;
                let xi_off = i * d;

                if d3 > 0.0 && d3.is_finite() {
                    let inv_d3 = l_mi / d3;
                    for a in 0..d {
                        let da = self.nd_xm[a] - self.nd_coords[xi_off + a];
                        let h_aa = k_mi * (1.0 - inv_d3 * (dist2 - da * da));
                        self.nd_hessian[a * d + a] += h_aa;
                        for b in (a + 1)..d {
                            let db = self.nd_xm[b] - self.nd_coords[xi_off + b];
                            let h_ab = k_mi * inv_d3 * da * db;
                            self.nd_hessian[a * d + b] += h_ab;
                            self.nd_hessian[b * d + a] += h_ab;
                        }
                    }
                }

                let f = k_mi * (1.0 - l_mi / dist);
                for a in 0..d {
                    self.nd_grad[a] += f * (self.nd_xm[a] - self.nd_coords[xi_off + a]);
                }
            }

            let delta2: f64 = self.nd_grad.iter().map(|&g| g * g).sum();
            if !delta2.is_finite() || delta2 < eps_sq {
                break;
            }

            if !solve_linear_system_pp(
                &mut self.nd_hessian,
                &self.nd_grad,
                &mut self.nd_delta,
                d,
                HESSIAN_DET_TOL,
            ) {
                // Singular or near-singular Hessian: take one bounded steepest-descent fallback step
                // and exit the inner Newton loop cleanly so degenerate configurations never loop.
                let g_norm = delta2.sqrt();
                if g_norm > 0.0 && g_norm.is_finite() {
                    let step_len = (0.1 / g_norm).min(0.25);
                    let mut all_finite = true;
                    for a in 0..d {
                        let cand = self.nd_xm[a] - step_len * self.nd_grad[a];
                        if !cand.is_finite() {
                            all_finite = false;
                            break;
                        }
                        self.nd_delta[a] = cand;
                    }
                    if all_finite {
                        self.nd_xm.copy_from_slice(&self.nd_delta);
                    }
                }
                break;
            }

            if !self.nd_delta.iter().all(|v| v.is_finite()) {
                break;
            }
            let step_norm = euclid_norm(&self.nd_delta);
            if !step_norm.is_finite() {
                break;
            }
            if step_norm > MAX_NEWTON_STEP {
                let scale = MAX_NEWTON_STEP / step_norm;
                for v in &mut self.nd_delta {
                    *v *= scale;
                }
            }

            let mut all_finite = true;
            for a in 0..d {
                if !(self.nd_xm[a] - self.nd_delta[a]).is_finite() {
                    all_finite = false;
                    break;
                }
            }
            if !all_finite {
                break;
            }
            for a in 0..d {
                self.nd_xm[a] -= self.nd_delta[a];
            }

            if step_norm <= self.config.eps * 1e-2 {
                break;
            }
        }

        let dst = layout.node_mut(m)?;
        for a in 0..d {
            let new_val = S::from_f64(self.nd_xm[a]);
            if new_val.is_finite() {
                dst[a] = new_val;
            }
        }

        self.iter += 1;
        self.status = if self.iter >= self.max_iterations {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iter
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.max_iterations
    }
}
