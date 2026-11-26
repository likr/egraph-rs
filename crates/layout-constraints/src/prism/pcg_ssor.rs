use egraph_core::CoreResult;

use super::pcg::{SparsePcgWorkspace, ssor_apply_1rhs, ssor_apply_2rhs};

impl SparsePcgWorkspace {
    /// Solves two sparse symmetric positive semi-definite systems `A x = bx` and `A y = by`
    /// simultaneously using split `L + D + U` CSR storage and Symmetric SSOR preconditioning.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn solve_ldu_2rhs_ssor(
        &mut self,
        diag: &[f64],
        diag_inv: &[f64],
        l_indptr: &[u32],
        l_indices: &[u32],
        l_values: &[f64],
        u_indptr: &[u32],
        u_indices: &[u32],
        u_values: &[f64],
        bx: &[f64],
        by: &[f64],
        x: &mut [f64],
        y: &mut [f64],
        tol: f64,
        max_iters: usize,
        omega: f64,
    ) -> CoreResult<usize> {
        let n = bx.len();
        if n == 0 || max_iters == 0 {
            return Ok(0);
        }
        self.ensure_capacity(n);

        // Compute initial residuals r_x = bx - A x and r_y = by - A y in a single pass over A.
        let mut rr0_x = 0.0_f64;
        let mut rr0_y = 0.0_f64;
        for i in 0..n {
            let di = diag[i];
            let mut ax = di * x[i];
            let mut ay = di * y[i];
            let ls = l_indptr[i] as usize;
            let le = l_indptr[i + 1] as usize;
            for pos in ls..le {
                let j = l_indices[pos] as usize;
                let v = l_values[pos];
                ax += v * x[j];
                ay += v * y[j];
            }
            let us = u_indptr[i] as usize;
            let ue = u_indptr[i + 1] as usize;
            for pos in us..ue {
                let j = u_indices[pos] as usize;
                let v = u_values[pos];
                ax += v * x[j];
                ay += v * y[j];
            }
            let rxi = bx[i] - ax;
            let ryi = by[i] - ay;
            self.r[i] = rxi;
            self.r_y[i] = ryi;
            rr0_x += rxi * rxi;
            rr0_y += ryi * ryi;
        }

        let tol_sq = tol * tol;
        let stop_rr_x = tol_sq * rr0_x;
        let stop_rr_y = tol_sq * rr0_y;
        let mut active_x = rr0_x.is_finite() && rr0_x > 1e-30 && rr0_x > stop_rr_x;
        let mut active_y = rr0_y.is_finite() && rr0_y > 1e-30 && rr0_y > stop_rr_y;
        if !active_x && !active_y {
            return Ok(0);
        }

        let mut rz_old_x = 0.0_f64;
        let mut rz_old_y = 0.0_f64;
        if active_x && active_y {
            ssor_apply_2rhs(
                n,
                diag_inv,
                l_indptr,
                l_indices,
                l_values,
                u_indptr,
                u_indices,
                u_values,
                omega,
                &self.r[..n],
                &self.r_y[..n],
                &mut self.z[..n],
                &mut self.z_y[..n],
            );
            for i in 0..n {
                let zx = self.z[i];
                let zy = self.z_y[i];
                self.p[i] = zx;
                self.p_y[i] = zy;
                rz_old_x += self.r[i] * zx;
                rz_old_y += self.r_y[i] * zy;
            }
        } else if active_x {
            ssor_apply_1rhs(
                n,
                diag_inv,
                l_indptr,
                l_indices,
                l_values,
                u_indptr,
                u_indices,
                u_values,
                omega,
                &self.r[..n],
                &mut self.z[..n],
            );
            for i in 0..n {
                let zx = self.z[i];
                self.p[i] = zx;
                rz_old_x += self.r[i] * zx;
            }
        } else {
            ssor_apply_1rhs(
                n,
                diag_inv,
                l_indptr,
                l_indices,
                l_values,
                u_indptr,
                u_indices,
                u_values,
                omega,
                &self.r_y[..n],
                &mut self.z_y[..n],
            );
            for i in 0..n {
                let zy = self.z_y[i];
                self.p_y[i] = zy;
                rz_old_y += self.r_y[i] * zy;
            }
        }

        if !rz_old_x.is_finite() || rz_old_x <= 1e-30 {
            active_x = false;
        }
        if !rz_old_y.is_finite() || rz_old_y <= 1e-30 {
            active_y = false;
        }
        if !active_x && !active_y {
            return Ok(0);
        }

        for iter in 0..max_iters {
            if active_x && active_y {
                let mut denom_x = 0.0_f64;
                let mut denom_y = 0.0_f64;
                for i in 0..n {
                    let di = diag[i];
                    let mut sx = di * self.p[i];
                    let mut sy = di * self.p_y[i];
                    let ls = l_indptr[i] as usize;
                    let le = l_indptr[i + 1] as usize;
                    for pos in ls..le {
                        let j = l_indices[pos] as usize;
                        let v = l_values[pos];
                        sx += v * self.p[j];
                        sy += v * self.p_y[j];
                    }
                    let us = u_indptr[i] as usize;
                    let ue = u_indptr[i + 1] as usize;
                    for pos in us..ue {
                        let j = u_indices[pos] as usize;
                        let v = u_values[pos];
                        sx += v * self.p[j];
                        sy += v * self.p_y[j];
                    }
                    self.ap[i] = sx;
                    self.ap_y[i] = sy;
                    denom_x += self.p[i] * sx;
                    denom_y += self.p_y[i] * sy;
                }

                let mut alpha_x = 0.0_f64;
                let mut alpha_y = 0.0_f64;
                if !denom_x.is_finite() || denom_x <= 1e-30 {
                    active_x = false;
                } else {
                    alpha_x = rz_old_x / denom_x;
                    if !alpha_x.is_finite() {
                        active_x = false;
                    }
                }
                if !denom_y.is_finite() || denom_y <= 1e-30 {
                    active_y = false;
                } else {
                    alpha_y = rz_old_y / denom_y;
                    if !alpha_y.is_finite() {
                        active_y = false;
                    }
                }
                if !active_x && !active_y {
                    return Ok(iter);
                }

                let mut rr_x = 0.0_f64;
                let mut rr_y = 0.0_f64;
                if active_x && active_y {
                    for i in 0..n {
                        let cand_x = x[i] + alpha_x * self.p[i];
                        let cand_y = y[i] + alpha_y * self.p_y[i];
                        if !cand_x.is_finite() {
                            active_x = false;
                        } else {
                            x[i] = cand_x;
                            let rxi = self.r[i] - alpha_x * self.ap[i];
                            self.r[i] = rxi;
                            rr_x += rxi * rxi;
                        }
                        if !cand_y.is_finite() {
                            active_y = false;
                        } else {
                            y[i] = cand_y;
                            let ryi = self.r_y[i] - alpha_y * self.ap_y[i];
                            self.r_y[i] = ryi;
                            rr_y += ryi * ryi;
                        }
                    }
                } else if active_x {
                    for i in 0..n {
                        let cand_x = x[i] + alpha_x * self.p[i];
                        if !cand_x.is_finite() {
                            active_x = false;
                            break;
                        }
                        x[i] = cand_x;
                        let rxi = self.r[i] - alpha_x * self.ap[i];
                        self.r[i] = rxi;
                        rr_x += rxi * rxi;
                    }
                } else {
                    for i in 0..n {
                        let cand_y = y[i] + alpha_y * self.p_y[i];
                        if !cand_y.is_finite() {
                            active_y = false;
                            break;
                        }
                        y[i] = cand_y;
                        let ryi = self.r_y[i] - alpha_y * self.ap_y[i];
                        self.r_y[i] = ryi;
                        rr_y += ryi * ryi;
                    }
                }

                if active_x && (!rr_x.is_finite() || rr_x == 0.0 || rr_x <= stop_rr_x) {
                    active_x = false;
                }
                if active_y && (!rr_y.is_finite() || rr_y == 0.0 || rr_y <= stop_rr_y) {
                    active_y = false;
                }
                if !active_x && !active_y {
                    return Ok(iter + 1);
                }

                if active_x && active_y {
                    ssor_apply_2rhs(
                        n,
                        diag_inv,
                        l_indptr,
                        l_indices,
                        l_values,
                        u_indptr,
                        u_indices,
                        u_values,
                        omega,
                        &self.r[..n],
                        &self.r_y[..n],
                        &mut self.z[..n],
                        &mut self.z_y[..n],
                    );
                    let mut rz_new_x = 0.0_f64;
                    let mut rz_new_y = 0.0_f64;
                    for i in 0..n {
                        rz_new_x += self.r[i] * self.z[i];
                        rz_new_y += self.r_y[i] * self.z_y[i];
                    }
                    let beta_x = if rz_new_x.is_finite() && rz_new_x > 1e-30 {
                        rz_new_x / rz_old_x
                    } else {
                        active_x = false;
                        0.0
                    };
                    let beta_y = if rz_new_y.is_finite() && rz_new_y > 1e-30 {
                        rz_new_y / rz_old_y
                    } else {
                        active_y = false;
                        0.0
                    };
                    if !beta_x.is_finite() {
                        active_x = false;
                    }
                    if !beta_y.is_finite() {
                        active_y = false;
                    }
                    if !active_x && !active_y {
                        return Ok(iter + 1);
                    }
                    rz_old_x = rz_new_x;
                    rz_old_y = rz_new_y;
                    if active_x && active_y {
                        for i in 0..n {
                            self.p[i] = self.z[i] + beta_x * self.p[i];
                            self.p_y[i] = self.z_y[i] + beta_y * self.p_y[i];
                        }
                    } else if active_x {
                        for i in 0..n {
                            self.p[i] = self.z[i] + beta_x * self.p[i];
                        }
                    } else {
                        for i in 0..n {
                            self.p_y[i] = self.z_y[i] + beta_y * self.p_y[i];
                        }
                    }
                } else if active_x {
                    ssor_apply_1rhs(
                        n,
                        diag_inv,
                        l_indptr,
                        l_indices,
                        l_values,
                        u_indptr,
                        u_indices,
                        u_values,
                        omega,
                        &self.r[..n],
                        &mut self.z[..n],
                    );
                    let mut rz_new_x = 0.0_f64;
                    for i in 0..n {
                        rz_new_x += self.r[i] * self.z[i];
                    }
                    if !rz_new_x.is_finite() || rz_new_x <= 1e-30 {
                        return Ok(iter + 1);
                    }
                    let beta_x = rz_new_x / rz_old_x;
                    if !beta_x.is_finite() {
                        return Ok(iter + 1);
                    }
                    rz_old_x = rz_new_x;
                    for i in 0..n {
                        self.p[i] = self.z[i] + beta_x * self.p[i];
                    }
                } else {
                    ssor_apply_1rhs(
                        n,
                        diag_inv,
                        l_indptr,
                        l_indices,
                        l_values,
                        u_indptr,
                        u_indices,
                        u_values,
                        omega,
                        &self.r_y[..n],
                        &mut self.z_y[..n],
                    );
                    let mut rz_new_y = 0.0_f64;
                    for i in 0..n {
                        rz_new_y += self.r_y[i] * self.z_y[i];
                    }
                    if !rz_new_y.is_finite() || rz_new_y <= 1e-30 {
                        return Ok(iter + 1);
                    }
                    let beta_y = rz_new_y / rz_old_y;
                    if !beta_y.is_finite() {
                        return Ok(iter + 1);
                    }
                    rz_old_y = rz_new_y;
                    for i in 0..n {
                        self.p_y[i] = self.z_y[i] + beta_y * self.p_y[i];
                    }
                }
            } else {
                return self.solve_ldu_1rhs_tail(
                    iter, max_iters, active_x, diag, diag_inv, l_indptr, l_indices, l_values,
                    u_indptr, u_indices, u_values, x, y, rz_old_x, rz_old_y, stop_rr_x, stop_rr_y,
                    omega,
                );
            }
        }

        Ok(max_iters)
    }
}
