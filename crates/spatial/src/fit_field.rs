//! Fused 2D field evaluation for tsNET C1 (Student-$t$ KL repulsion) and C2 (entropy repulsion).
//!
//! By evaluating C1 and C2 on a single shared [`IbFft2d`] particle grid, [`FitField2d`] halves
//! particle-grid forward 2D FFTs from `4` down to `2` and reduces total `fft_2d` calls per
//! iteration from `10` (unfused) to `8` (or `5` when entropy is disabled).

use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};

use crate::fft::{Complex64, Fft2dWorkspace};
use crate::ibfft::{
    IbFft2d, IbFftConfig, IbFftGridSpec, InterpolationOrder, cell_grid_indices, stencil_axis,
};

/// Configuration for [`FitField2d`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitFieldConfig {
    /// Grid & interpolation configuration (`Bilinear` $p=1$ by default).
    pub ibfft: IbFftConfig,
    /// Entropy softening constant $\varepsilon_r > 0$ in $K_e(d) = 1 / (\varepsilon_r + d^2)$.
    pub epsilon_r: f64,
    /// Whether to evaluate C2 entropy repulsion alongside C1 KL repulsion.
    pub compute_entropy: bool,
}

impl Default for FitFieldConfig {
    fn default() -> Self {
        Self {
            ibfft: IbFftConfig::default(),
            epsilon_r: 0.05,
            compute_entropy: true,
        }
    }
}

impl FitFieldConfig {
    /// Validates field parameters.
    pub fn validate(&self) -> CoreResult<()> {
        self.ibfft.validate()?;
        if !self.epsilon_r.is_finite() || self.epsilon_r <= 0.0 {
            return Err(CoreError::invalid_input(
                "epsilon_r must be finite and strictly positive",
            ));
        }
        Ok(())
    }
}

/// Telemetry counters recorded for each [`FitField2d::evaluate`] invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FitFieldStats {
    /// Number of forward 2D FFTs executed on particle charge grids (`2` when `n >= 2`).
    pub particle_forward_ffts: u64,
    /// Number of forward 2D FFTs executed on kernel grids (`2` when fused, `1` for KL-only).
    pub kernel_forward_ffts: u64,
    /// Number of inverse 2D FFTs executed (`4` when fused, `2` for KL-only).
    pub inverse_ffts: u64,
    /// Total `fft_2d` calls (`particle_forward_ffts + kernel_forward_ffts + inverse_ffts <= 8`).
    pub total_fft_2d_calls: u64,
}

/// Reusable output buffers filled by [`FitField2d::evaluate`].
#[derive(Debug, Clone)]
pub struct FitFieldOutput<S: Scalar = f64> {
    /// Normalized C1 KL repulsive force $x$-components: $(1/Z) \sum_{j \neq i} (y_{i,x} - y_{j,x})/(1 + d_{ij}^2)^2$.
    pub kl_repulsion_x: Vec<S>,
    /// Normalized C1 KL repulsive force $y$-components.
    pub kl_repulsion_y: Vec<S>,
    /// Estimated global Student-$t$ partition sum $Z = \sum_{i \neq j} (1 + d_{ij}^2)^{-1}$.
    pub total_z: S,
    /// Unnormalized C2 entropy repulsive force $x$-components: $\sum_{j \neq i} (y_{i,x} - y_{j,x})/(\varepsilon_r + d_{ij}^2)$.
    pub entropy_repulsion_x: Vec<S>,
    /// Unnormalized C2 entropy repulsive force $y$-components.
    pub entropy_repulsion_y: Vec<S>,
    /// Call-count telemetry for the completed evaluation.
    pub stats: FitFieldStats,
}

impl<S: Scalar> Default for FitFieldOutput<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> FitFieldOutput<S> {
    /// Allocates an empty reusable output buffer.
    pub fn new() -> Self {
        Self {
            kl_repulsion_x: Vec::new(),
            kl_repulsion_y: Vec::new(),
            total_z: S::ONE,
            entropy_repulsion_x: Vec::new(),
            entropy_repulsion_y: Vec::new(),
            stats: FitFieldStats::default(),
        }
    }

    fn resize_and_zero(&mut self, n: usize, compute_entropy: bool) {
        self.kl_repulsion_x.resize(n, S::ZERO);
        self.kl_repulsion_x.fill(S::ZERO);
        self.kl_repulsion_y.resize(n, S::ZERO);
        self.kl_repulsion_y.fill(S::ZERO);
        if compute_entropy {
            self.entropy_repulsion_x.resize(n, S::ZERO);
            self.entropy_repulsion_x.fill(S::ZERO);
            self.entropy_repulsion_y.resize(n, S::ZERO);
            self.entropy_repulsion_y.fill(S::ZERO);
        } else {
            self.entropy_repulsion_x.clear();
            self.entropy_repulsion_y.clear();
        }
        self.total_z = S::ONE;
        self.stats = FitFieldStats::default();
    }
}

/// Fused C1 (Student-$t$ KL) + C2 (entropy) 2D FFT field evaluator.
#[derive(Debug, Clone)]
pub struct FitField2d<S: Scalar = f64> {
    config: FitFieldConfig,
    ws: Fft2dWorkspace,
    ibfft: IbFft2d<S>,
    // Per-particle intermediate potentials in `f64` to prevent cancellation before normalization.
    phi_1_z: Vec<f64>,
    phi_1_k2: Vec<f64>,
    phi_x_k2: Vec<f64>,
    phi_y_k2: Vec<f64>,
    h1: Vec<f64>,
    h2: Vec<f64>,
    h3: Vec<f64>,
    last_stats: FitFieldStats,
}

impl<S: Scalar> Default for FitField2d<S> {
    fn default() -> Self {
        Self::new(FitFieldConfig::default()).expect("default FitFieldConfig is valid")
    }
}

impl<S: Scalar> FitField2d<S> {
    /// Constructs a new [`FitField2d`] workspace.
    pub fn new(config: FitFieldConfig) -> CoreResult<Self> {
        config.validate()?;
        let ibfft = IbFft2d::new(config.ibfft)?;
        Ok(Self {
            config,
            ws: Fft2dWorkspace::new(),
            ibfft,
            phi_1_z: Vec::new(),
            phi_1_k2: Vec::new(),
            phi_x_k2: Vec::new(),
            phi_y_k2: Vec::new(),
            h1: Vec::new(),
            h2: Vec::new(),
            h3: Vec::new(),
            last_stats: FitFieldStats::default(),
        })
    }

    /// Returns the active [`FitFieldConfig`].
    #[inline]
    pub fn config(&self) -> FitFieldConfig {
        self.config
    }

    /// Returns the adaptive FFT grid geometry computed by the most recent evaluation.
    #[inline]
    pub fn grid_spec(&self) -> Option<IbFftGridSpec> {
        self.ibfft.grid_spec()
    }

    /// Updates `epsilon_r` without reallocating workspaces.
    pub fn set_epsilon_r(&mut self, epsilon_r: f64) -> CoreResult<()> {
        if !epsilon_r.is_finite() || epsilon_r <= 0.0 {
            return Err(CoreError::invalid_input(
                "epsilon_r must be finite and strictly positive",
            ));
        }
        self.config.epsilon_r = epsilon_r;
        Ok(())
    }

    /// Returns telemetry from the most recent [`FitField2d::evaluate`] call.
    #[inline]
    pub fn last_stats(&self) -> FitFieldStats {
        self.last_stats
    }

    /// Returns cumulative 2D FFT calls executed by the owned [`Fft2dWorkspace`].
    #[inline]
    pub fn fft_2d_call_count(&self) -> u64 {
        self.ws.fft_2d_call_count()
    }

    /// Resets the owned [`Fft2dWorkspace`] call counter.
    #[inline]
    pub fn reset_fft_2d_call_count(&mut self) {
        self.ws.reset_fft_2d_call_count();
    }

    /// Evaluates fused C1 KL repulsion and (when enabled) C2 entropy repulsion on `layout`.
    pub fn evaluate(
        &mut self,
        layout: &Layout2d<S>,
        out: &mut FitFieldOutput<S>,
    ) -> CoreResult<()> {
        self.evaluate_xy(layout.x(), layout.y(), out)
    }

    /// Slice-based variant of [`FitField2d::evaluate`].
    pub fn evaluate_xy(
        &mut self,
        xs: &[S],
        ys: &[S],
        out: &mut FitFieldOutput<S>,
    ) -> CoreResult<()> {
        if xs.len() != ys.len() {
            return Err(CoreError::dimension_mismatch(
                xs.len(),
                ys.len(),
                "FitField2d::evaluate_xy",
            ));
        }
        let n = xs.len();
        out.resize_and_zero(n, self.config.compute_entropy);
        self.last_stats = FitFieldStats::default();

        let count_before = self.ws.fft_2d_call_count();
        let particle_fwds = self.ibfft.prepare_xy(&mut self.ws, xs, ys)?;
        if n < 2 {
            // Validate finiteness already occurred in `prepare_xy`.
            return Ok(());
        }

        let spec = self.ibfft.grid_spec().expect("spec must exist when n >= 2");
        self.resize_potentials(n);

        // --- C1: Student-t KL repulsion (1 kernel FFT + 2 inverse FFTs) ---
        self.build_kl_kernel_fft(&spec)?;
        self.convolve_c1(&spec)?;
        self.gather_c1_potentials(&spec);
        self.assemble_kl_forces(n, &spec, out);

        let mut kernel_fwds = 1u64;
        let mut inv_ffts = 2u64;

        // --- C2: Entropy repulsion reusing the exact same particle forward FFTs ---
        if self.config.compute_entropy {
            self.build_entropy_kernel_fft(&spec)?;
            self.convolve_c2(&spec)?;
            self.gather_c2_potentials(&spec);
            self.assemble_entropy_forces(out);
            kernel_fwds += 1;
            inv_ffts += 2;
        }

        let total_calls = self.ws.fft_2d_call_count() - count_before;
        let stats = FitFieldStats {
            particle_forward_ffts: particle_fwds,
            kernel_forward_ffts: kernel_fwds,
            inverse_ffts: inv_ffts,
            total_fft_2d_calls: total_calls,
        };
        out.stats = stats;
        self.last_stats = stats;
        Ok(())
    }

    fn resize_potentials(&mut self, n: usize) {
        self.phi_1_z.resize(n, 0.0);
        self.phi_1_k2.resize(n, 0.0);
        self.phi_x_k2.resize(n, 0.0);
        self.phi_y_k2.resize(n, 0.0);
        if self.config.compute_entropy {
            self.h1.resize(n, 0.0);
            self.h2.resize(n, 0.0);
            self.h3.resize(n, 0.0);
        }
    }

    fn build_kl_kernel_fft(&mut self, spec: &IbFftGridSpec) -> CoreResult<()> {
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

    fn convolve_c1(&mut self, spec: &IbFftGridSpec) -> CoreResult<()> {
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

    fn gather_c1_potentials(&mut self, spec: &IbFftGridSpec) {
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

    fn assemble_kl_forces(&self, n: usize, spec: &IbFftGridSpec, out: &mut FitFieldOutput<S>) {
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

    fn build_entropy_kernel_fft(&mut self, spec: &IbFftGridSpec) -> CoreResult<()> {
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

    fn convolve_c2(&mut self, spec: &IbFftGridSpec) -> CoreResult<()> {
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

    fn gather_c2_potentials(&mut self, spec: &IbFftGridSpec) {
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

    fn assemble_entropy_forces(&self, out: &mut FitFieldOutput<S>) {
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

/// Convenience helper for setting [`IbFftConfig`] order inside [`FitFieldConfig`].
impl FitFieldConfig {
    /// Returns a copy with the requested interpolation order.
    pub fn with_order(mut self, order: InterpolationOrder) -> Self {
        self.ibfft.order = order;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force_kl_and_entropy(
        layout: &Layout2d<f64>,
        epsilon_r: f64,
    ) -> (Vec<f64>, Vec<f64>, f64, Vec<f64>, Vec<f64>) {
        let n = layout.len();
        let mut kl_x = vec![0.0; n];
        let mut kl_y = vec![0.0; n];
        let mut ent_x = vec![0.0; n];
        let mut ent_y = vec![0.0; n];
        let mut total_z = 0.0;

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = layout.x()[i] - layout.x()[j];
                let dy = layout.y()[i] - layout.y()[j];
                let d_sq = dx * dx + dy * dy;
                let w = 1.0 / (1.0 + d_sq);
                total_z += w;
                let w2 = w * w;
                kl_x[i] += w2 * dx;
                kl_y[i] += w2 * dy;

                let ke = 1.0 / (epsilon_r + d_sq);
                ent_x[i] += ke * dx;
                ent_y[i] += ke * dy;
            }
        }

        let denom = total_z.max(1e-12);
        for i in 0..n {
            kl_x[i] /= denom;
            kl_y[i] /= denom;
        }
        (kl_x, kl_y, denom, ent_x, ent_y)
    }

    #[test]
    fn test_fit_field_fused_fft_call_counts_and_accuracy_bilinear_and_cubic() {
        let layout = Layout2d::from_xy(
            vec![-3.1, -1.8, -0.4, 0.2, 1.3, 2.7, -2.5, 2.1, 0.7, -1.0],
            vec![-2.2, 1.5, -0.7, 2.4, -1.9, 0.8, 2.6, -2.8, 0.3, -0.1],
        )
        .unwrap();
        let eps_r = 0.05;
        let (ex_kl_x, ex_kl_y, ex_z, ex_ent_x, ex_ent_y) =
            brute_force_kl_and_entropy(&layout, eps_r);

        for (order, max_kl_err, max_ent_rel_err, max_z_rel_err) in [
            (InterpolationOrder::Bilinear, 5e-3, 1e-2, 0.02),
            (InterpolationOrder::Cubic, 1e-3, 1e-3, 0.01),
        ] {
            let mut field =
                FitField2d::<f64>::new(FitFieldConfig::default().with_order(order)).unwrap();
            let mut out = FitFieldOutput::new();

            field.reset_fft_2d_call_count();
            field.evaluate(&layout, &mut out).unwrap();

            assert_eq!(out.stats.particle_forward_ffts, 2);
            assert!(out.stats.total_fft_2d_calls <= 8);
            assert_eq!(out.stats.total_fft_2d_calls, 8);
            assert_eq!(field.fft_2d_call_count(), 8);

            let z_rel = (out.total_z - ex_z).abs() / ex_z;
            assert!(z_rel < max_z_rel_err, "{order:?}: z_rel={z_rel}");

            for i in 0..layout.len() {
                let kl_err =
                    (out.kl_repulsion_x[i] - ex_kl_x[i]).hypot(out.kl_repulsion_y[i] - ex_kl_y[i]);
                assert!(kl_err < max_kl_err, "{order:?}[{i}]: kl_err={kl_err}");

                let ent_norm = ex_ent_x[i].hypot(ex_ent_y[i]).max(1e-6);
                let ent_rel = (out.entropy_repulsion_x[i] - ex_ent_x[i])
                    .hypot(out.entropy_repulsion_y[i] - ex_ent_y[i])
                    / ent_norm;
                assert!(
                    ent_rel < max_ent_rel_err,
                    "{order:?}[{i}]: ent_rel={ent_rel}"
                );
            }
        }
    }

    #[test]
    fn test_fit_field_kl_only_uses_five_ffts_and_f32_supported() {
        let layout = Layout2d::from_xy(vec![0.0_f32, 1.0, -1.0], vec![0.5, -0.5, 0.2]).unwrap();
        let mut field = FitField2d::<f32>::new(FitFieldConfig {
            compute_entropy: false,
            ..FitFieldConfig::default()
        })
        .unwrap();
        let mut out = FitFieldOutput::new();

        field.evaluate(&layout, &mut out).unwrap();
        assert_eq!(out.stats.particle_forward_ffts, 2);
        assert_eq!(out.stats.total_fft_2d_calls, 5);
        assert!(out.entropy_repulsion_x.is_empty());
        assert!(out.total_z.is_finite() && out.total_z > 0.0);
    }

    #[test]
    fn test_fit_field_validation_errors() {
        assert!(matches!(
            FitField2d::<f64>::new(FitFieldConfig {
                epsilon_r: 0.0,
                ..FitFieldConfig::default()
            }),
            Err(CoreError::InvalidInput(_))
        ));

        let mut field = FitField2d::<f64>::default();
        let bad = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 1.0]).unwrap();
        let mut out = FitFieldOutput::new();
        assert!(matches!(
            field.evaluate(&bad, &mut out),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }
}
