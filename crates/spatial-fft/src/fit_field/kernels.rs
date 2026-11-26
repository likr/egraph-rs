use egraph_core::{CoreResult, Scalar};

use crate::fft::Complex64;
use crate::ibfft::{IbFftGridSpec, cell_grid_indices, stencil_axis};

use super::FitField2d;
use super::types::FitFieldOutput;

impl<S: Scalar> FitField2d<S> {
    pub(super) fn build_kl_kernel_fft(&mut self, spec: &IbFftGridSpec) -> CoreResult<()> {
        let base_denom = 1.0 + spec.softening_sq;
        let pad_nx = spec.pad_nx;
        let pad_ny = spec.pad_ny;

        // Pack even real kernels K_1 + i * K_2 into one complex FFT:
        // because both kernels are circularly even, Re(FFT) = FFT(K_1) and Im(FFT) = FFT(K_2).
        for v in 0..pad_ny {
            let dy = if v < spec.ny {
                v as f64 * spec.hy
            } else {
                (v as f64 - pad_ny as f64) * spec.hy
            };
            let dy_sq = dy * dy;
            let row = v * pad_nx;
            for u in 0..pad_nx {
                let dx = if u < spec.nx {
                    u as f64 * spec.hx
                } else {
                    (u as f64 - pad_nx as f64) * spec.hx
                };
                let k1 = 1.0 / (base_denom + dx * dx + dy_sq);
                self.ibfft.kernel_buf[row + u] = Complex64::new(k1, k1 * k1);
            }
        }
        self.ws
            .fft_2d(&mut self.ibfft.kernel_buf, pad_nx, pad_ny, false)
    }

    pub(super) fn convolve_c1(&mut self, spec: &IbFftGridSpec) -> CoreResult<()> {
        let pad_nx = spec.pad_nx;
        let pad_ny = spec.pad_ny;

        // 1) grid_1 * (FFT(K_1) + i FFT(K_2)) -> IFFT yields phi_1_z + i * phi_1_k2.
        for (dst, (&g1, &kf)) in self.ibfft.conv_buf.iter_mut().zip(
            self.ibfft
                .grid_1_fft
                .iter()
                .zip(self.ibfft.kernel_buf.iter()),
        ) {
            *dst = g1 * kf;
        }
        self.ws
            .fft_2d(&mut self.ibfft.conv_buf, pad_nx, pad_ny, true)?;

        let order = spec.effective_order;
        let nx = spec.nx;
        let ny = spec.ny;
        for cell in &self.ibfft.active_cells {
            let (u_idx, v_idx, len) = cell_grid_indices(cell.u0, cell.v0, order);
            for p in &self.ibfft.binned[cell.start..cell.end] {
                let (_, _, wu, _) = stencil_axis(p.u_real, nx, order);
                let (_, _, wv, _) = stencil_axis(p.v_real, ny, order);
                let mut acc_z = 0.0;
                let mut acc_k2 = 0.0;
                for k in 0..len {
                    let row_off = v_idx[k] * pad_nx;
                    let mut rz = 0.0;
                    let mut rk2 = 0.0;
                    for l in 0..len {
                        let val = self.ibfft.conv_buf[row_off + u_idx[l]];
                        rz += wu[l] * val.re;
                        rk2 += wu[l] * val.im;
                    }
                    acc_z += wv[k] * rz;
                    acc_k2 += wv[k] * rk2;
                }
                let idx = p.orig_idx as usize;
                self.phi_1_z[idx] = acc_z;
                self.phi_1_k2[idx] = acc_k2;
            }
        }

        // 2) (grid_x + i grid_y) * FFT(K_2) -> IFFT yields phi_x_k2 + i * phi_y_k2.
        for (dst, (&gxy, &kf)) in self.ibfft.conv_buf.iter_mut().zip(
            self.ibfft
                .grid_xy_fft
                .iter()
                .zip(self.ibfft.kernel_buf.iter()),
        ) {
            *dst = gxy.scale(kf.im);
        }
        self.ws
            .fft_2d(&mut self.ibfft.conv_buf, pad_nx, pad_ny, true)
    }

    pub(super) fn gather_c1_potentials(&mut self, spec: &IbFftGridSpec) {
        let order = spec.effective_order;
        let nx = spec.nx;
        let ny = spec.ny;
        let pad_nx = spec.pad_nx;
        for cell in &self.ibfft.active_cells {
            let (u_idx, v_idx, len) = cell_grid_indices(cell.u0, cell.v0, order);
            for p in &self.ibfft.binned[cell.start..cell.end] {
                let (_, _, wu, _) = stencil_axis(p.u_real, nx, order);
                let (_, _, wv, _) = stencil_axis(p.v_real, ny, order);
                let mut acc_x = 0.0;
                let mut acc_y = 0.0;
                for k in 0..len {
                    let row_off = v_idx[k] * pad_nx;
                    let mut rx = 0.0;
                    let mut ry = 0.0;
                    for l in 0..len {
                        let val = self.ibfft.conv_buf[row_off + u_idx[l]];
                        rx += wu[l] * val.re;
                        ry += wu[l] * val.im;
                    }
                    acc_x += wv[k] * rx;
                    acc_y += wv[k] * ry;
                }
                let idx = p.orig_idx as usize;
                self.phi_x_k2[idx] = acc_x;
                self.phi_y_k2[idx] = acc_y;
            }
        }
    }

    pub(super) fn assemble_kl_forces(
        &self,
        n: usize,
        spec: &IbFftGridSpec,
        out: &mut FitFieldOutput<S>,
    ) {
        let base_denom = 1.0 + spec.softening_sq;
        let mut k1_local = [[0.0_f64; 7]; 7];
        for (du_idx, row) in k1_local.iter_mut().enumerate() {
            let dx = (du_idx as isize - 3) as f64 * spec.hx;
            for (dv_idx, cell) in row.iter_mut().enumerate() {
                let dy = (dv_idx as isize - 3) as f64 * spec.hy;
                *cell = 1.0 / (base_denom + dx * dx + dy * dy);
            }
        }

        let order = spec.effective_order;
        let nx = spec.nx;
        let ny = spec.ny;
        let mut total_z = 0.0_f64;

        // Accumulate raw unnormalized forces directly in output buffer before normalizing by total_z.
        for cell in &self.ibfft.active_cells {
            let (u_idx, v_idx, len) = cell_grid_indices(cell.u0, cell.v0, order);
            for p in &self.ibfft.binned[cell.start..cell.end] {
                let (_, _, wu, _) = stencil_axis(p.u_real, nx, order);
                let (_, _, wv, _) = stencil_axis(p.v_real, ny, order);

                let mut self_z = 0.0;
                for (k1, &uk1) in u_idx[..len].iter().enumerate() {
                    for (l1, &vl1) in v_idx[..len].iter().enumerate() {
                        let w1 = wu[k1] * wv[l1];
                        for (k2, &uk2) in u_idx[..len].iter().enumerate() {
                            let du = (uk2 as isize - uk1 as isize + 3) as usize;
                            let w_u2 = wu[k2];
                            for (l2, &vl2) in v_idx[..len].iter().enumerate() {
                                let dv = (vl2 as isize - vl1 as isize + 3) as usize;
                                self_z += w1 * (w_u2 * wv[l2]) * k1_local[du][dv];
                            }
                        }
                    }
                }

                let idx = p.orig_idx as usize;
                let z_i = (self.phi_1_z[idx] - self_z).max(0.0);
                total_z += z_i;

                let fx = p.qx * self.phi_1_k2[idx] - self.phi_x_k2[idx];
                let fy = p.qy * self.phi_1_k2[idx] - self.phi_y_k2[idx];
                out.kl_repulsion_x[idx] = S::from_f64(fx);
                out.kl_repulsion_y[idx] = S::from_f64(fy);
            }
        }

        let denom = total_z.max(1e-12);
        out.total_z = S::from_f64(denom);

        // Normalize by global Z and enforce Cauchy upper bound ||F_rep / Z|| <= 0.5.
        for i in 0..n {
            let mut fx = out.kl_repulsion_x[i].to_f64() / denom;
            let mut fy = out.kl_repulsion_y[i].to_f64() / denom;
            let norm = fx.hypot(fy);
            if norm > 0.5 {
                let scale = 0.5 / norm;
                fx *= scale;
                fy *= scale;
            }
            out.kl_repulsion_x[i] = S::from_f64(fx);
            out.kl_repulsion_y[i] = S::from_f64(fy);
        }
    }

    pub(super) fn build_entropy_kernel_fft(&mut self, spec: &IbFftGridSpec) -> CoreResult<()> {
        let eps_eff = self.config.epsilon_r + spec.softening_sq;
        let pad_nx = spec.pad_nx;
        let pad_ny = spec.pad_ny;

        for v in 0..pad_ny {
            let dy = if v < spec.ny {
                v as f64 * spec.hy
            } else {
                (v as f64 - pad_ny as f64) * spec.hy
            };
            let dy_sq = dy * dy;
            let row = v * pad_nx;
            for u in 0..pad_nx {
                let dx = if u < spec.nx {
                    u as f64 * spec.hx
                } else {
                    (u as f64 - pad_nx as f64) * spec.hx
                };
                self.ibfft.kernel_buf[row + u] =
                    Complex64::new(1.0 / (eps_eff + dx * dx + dy_sq), 0.0);
            }
        }
        self.ws
            .fft_2d(&mut self.ibfft.kernel_buf, pad_nx, pad_ny, false)
    }

    pub(super) fn convolve_c2(&mut self, spec: &IbFftGridSpec) -> CoreResult<()> {
        let pad_nx = spec.pad_nx;
        let pad_ny = spec.pad_ny;
        let order = spec.effective_order;
        let nx = spec.nx;
        let ny = spec.ny;

        // 1) grid_1 * FFT(K_e) -> IFFT yields h_1.
        for (dst, (&g1, &kf)) in self.ibfft.conv_buf.iter_mut().zip(
            self.ibfft
                .grid_1_fft
                .iter()
                .zip(self.ibfft.kernel_buf.iter()),
        ) {
            *dst = g1.scale(kf.re);
        }
        self.ws
            .fft_2d(&mut self.ibfft.conv_buf, pad_nx, pad_ny, true)?;
        for cell in &self.ibfft.active_cells {
            let (u_idx, v_idx, len) = cell_grid_indices(cell.u0, cell.v0, order);
            for p in &self.ibfft.binned[cell.start..cell.end] {
                let (_, _, wu, _) = stencil_axis(p.u_real, nx, order);
                let (_, _, wv, _) = stencil_axis(p.v_real, ny, order);
                let mut acc = 0.0;
                for k in 0..len {
                    let row_off = v_idx[k] * pad_nx;
                    let mut r = 0.0;
                    for l in 0..len {
                        r += wu[l] * self.ibfft.conv_buf[row_off + u_idx[l]].re;
                    }
                    acc += wv[k] * r;
                }
                self.h1[p.orig_idx as usize] = acc;
            }
        }

        // 2) (grid_x + i grid_y) * FFT(K_e) -> IFFT yields h_2 + i * h_3.
        for (dst, (&gxy, &kf)) in self.ibfft.conv_buf.iter_mut().zip(
            self.ibfft
                .grid_xy_fft
                .iter()
                .zip(self.ibfft.kernel_buf.iter()),
        ) {
            *dst = gxy.scale(kf.re);
        }
        self.ws
            .fft_2d(&mut self.ibfft.conv_buf, pad_nx, pad_ny, true)
    }

    pub(super) fn gather_c2_potentials(&mut self, spec: &IbFftGridSpec) {
        let order = spec.effective_order;
        let nx = spec.nx;
        let ny = spec.ny;
        let pad_nx = spec.pad_nx;
        for cell in &self.ibfft.active_cells {
            let (u_idx, v_idx, len) = cell_grid_indices(cell.u0, cell.v0, order);
            for p in &self.ibfft.binned[cell.start..cell.end] {
                let (_, _, wu, _) = stencil_axis(p.u_real, nx, order);
                let (_, _, wv, _) = stencil_axis(p.v_real, ny, order);
                let mut acc_2 = 0.0;
                let mut acc_3 = 0.0;
                for k in 0..len {
                    let row_off = v_idx[k] * pad_nx;
                    let mut r2 = 0.0;
                    let mut r3 = 0.0;
                    for l in 0..len {
                        let val = self.ibfft.conv_buf[row_off + u_idx[l]];
                        r2 += wu[l] * val.re;
                        r3 += wu[l] * val.im;
                    }
                    acc_2 += wv[k] * r2;
                    acc_3 += wv[k] * r3;
                }
                let idx = p.orig_idx as usize;
                self.h2[idx] = acc_2;
                self.h3[idx] = acc_3;
            }
        }
    }

    pub(super) fn assemble_entropy_forces(&self, out: &mut FitFieldOutput<S>) {
        for cell in &self.ibfft.active_cells {
            for p in &self.ibfft.binned[cell.start..cell.end] {
                let idx = p.orig_idx as usize;
                let fx = p.qx * self.h1[idx] - self.h2[idx];
                let fy = p.qy * self.h1[idx] - self.h3[idx];
                out.entropy_repulsion_x[idx] = S::from_f64(fx);
                out.entropy_repulsion_y[idx] = S::from_f64(fy);
            }
        }
    }
}
