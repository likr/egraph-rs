//! Screened Poisson unilateral upper-bound 2D density field (`f32` R2C FFT workspace).
//!
//! Solves the non-local Eulerian repulsion potential
//! $\hat{P}(k) = \frac{\hat{K}_{\sigma_0}(k)}{\|k\|^2 + \kappa_0^2}\widehat{\delta\rho}(k)$
//! for the unilateral excess density $\delta\rho(p) = \max(0, \rho_V(p) - \rho_{\max})$ with zero
//! entropy term ($\mu_{\mathrm{ent}} \equiv 0$, preserving whitespace). Supports exact variational
//! adjoint masking (`exact_adjoint: true`, default) via a compact `u64` excess bitmask as well as
//! the single-solve unmasked formulation (`exact_adjoint: false`).

use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};

use crate::fft_r2c::{Complex32, R2cFft2dWorkspaceF32};

/// Frequency-domain Green's operator used in [`ScreenedPoissonField2d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DensityFieldKernel {
    /// Long-range screened Poisson operator $\hat{G}(k) = (\|k\|^2 + \kappa_0^2)^{-1}$.
    #[default]
    ScreenedPoisson,
    /// Local single-Gaussian $L^2$ excess baseline $\hat{G}(k) = 1$.
    SingleGaussian,
}

/// Configuration for [`ScreenedPoissonField2d`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenedPoissonConfig {
    /// Density kernel operator (`default: DensityFieldKernel::ScreenedPoisson`).
    pub kernel: DensityFieldKernel,
    /// Active interior grid resolution $G \times G$ (`default: 2048`, power of two `>= 16`).
    pub grid_size: usize,
    /// Zero-padding multiplier (`1` or `2`, `default: 2`).
    pub padding_factor: usize,
    /// Relative border margin around the particle bounding box (`default: 0.1`).
    pub margin_ratio: f64,
    /// Default Gaussian smoothing width $\sigma_0$ as a fraction of domain span (`default: 0.015`).
    pub sigma_ratio: f64,
    /// Optional explicit physical Gaussian smoothing width $\sigma_0 > 0$.
    pub sigma: Option<f64>,
    /// Default screening length $\kappa_0^{-1}$ as a fraction of domain span (`default: 0.20`).
    pub screening_length_ratio: f64,
    /// Optional explicit screening wavenumber $\kappa_0 > 0$.
    pub kappa: Option<f64>,
    /// Default upper density cap multiplier relative to uniform domain density (`default: 4.0`).
    pub rho_max_factor: f64,
    /// Optional explicit density upper bound $\rho_{\max} \ge 0$.
    pub rho_max: Option<f64>,
    /// Whether to apply the exact variational support mask $\mathbf{1}_{\{\rho_V > \rho_{\max}\}}$
    /// before the final $K_{\sigma_0}$ adjoint convolution (`default: true`).
    pub exact_adjoint: bool,
}

impl Default for ScreenedPoissonConfig {
    fn default() -> Self {
        Self {
            kernel: DensityFieldKernel::ScreenedPoisson,
            grid_size: 2048,
            padding_factor: 2,
            margin_ratio: 0.1,
            sigma_ratio: 0.015,
            sigma: None,
            screening_length_ratio: 0.20,
            kappa: None,
            rho_max_factor: 4.0,
            rho_max: None,
            exact_adjoint: true,
        }
    }
}

impl ScreenedPoissonConfig {
    /// Validates all field hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.grid_size < 16 || !self.grid_size.is_power_of_two() {
            return Err(CoreError::invalid_input(format!(
                "grid_size must be a power of two >= 16, got {}",
                self.grid_size
            )));
        }
        if self.padding_factor != 1 && self.padding_factor != 2 {
            return Err(CoreError::invalid_input(format!(
                "padding_factor must be 1 or 2, got {}",
                self.padding_factor
            )));
        }
        if !self.margin_ratio.is_finite() || self.margin_ratio < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "margin_ratio must be finite and >= 0.0, got {}",
                self.margin_ratio
            )));
        }
        if !self.sigma_ratio.is_finite() || self.sigma_ratio <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "sigma_ratio must be finite and > 0.0, got {}",
                self.sigma_ratio
            )));
        }
        if let Some(s) = self.sigma
            && (!s.is_finite() || s <= 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "sigma must be finite and > 0.0, got {s}"
            )));
        }
        if !self.screening_length_ratio.is_finite() || self.screening_length_ratio <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "screening_length_ratio must be finite and > 0.0, got {}",
                self.screening_length_ratio
            )));
        }
        if let Some(k) = self.kappa
            && (!k.is_finite() || k <= 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "kappa must be finite and > 0.0, got {k}"
            )));
        }
        if !self.rho_max_factor.is_finite() || self.rho_max_factor < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "rho_max_factor must be finite and >= 0.0, got {}",
                self.rho_max_factor
            )));
        }
        if let Some(r) = self.rho_max
            && (!r.is_finite() || r < 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "rho_max must be finite and >= 0.0, got {r}"
            )));
        }
        Ok(())
    }
}

/// Diagnostic telemetry produced by [`ScreenedPoissonField2d::evaluate`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScreenedPoissonStats {
    /// Discrete Parseval variational energy $\mathcal{L}_{\mathrm{dens}}(X) \ge 0$.
    pub energy: f64,
    /// Peak smoothed density $\max_p \rho_V(p)$ on the grid.
    pub max_density: f64,
    /// Fraction of active interior grid cells where $\rho_V > \rho_{\max}$.
    pub excess_fraction: f64,
    /// Integrated excess density mass $\int \delta\rho(p)\,dp \in [0, 1]$.
    pub excess_mass: f64,
    /// Effective Gaussian smoothing width $\sigma_0$ used.
    pub sigma_used: f64,
    /// Effective screening wavenumber $\kappa_0$ used.
    pub kappa_used: f64,
    /// Effective upper density threshold $\rho_{\max}$ used.
    pub rho_max_used: f64,
    /// Active grid dimension $G$.
    pub grid_dim: usize,
    /// Padded FFT dimension $G_{\mathrm{pad}} = G \cdot \mathrm{padding\_factor}$.
    pub pad_dim: usize,
    /// Number of 2D R2C/C2R FFT calls executed in this evaluation (`0`, `2`, `4`, or `6`).
    pub fft_2d_calls: u64,
    /// `true` when zero cells exceeded $\rho_{\max}$ and the solve exited after smoothing.
    pub early_exit_no_excess: bool,
}

/// Reusable output buffers filled by [`ScreenedPoissonField2d::evaluate`].
#[derive(Debug, Clone)]
pub struct ScreenedPoissonOutput<S: Scalar = f64> {
    /// Descent force $x$-components $F_{i,x}^{\mathrm{dens}} = -\partial\mathcal{L}/\partial x_i$.
    pub force_x: Vec<S>,
    /// Descent force $y$-components $F_{i,y}^{\mathrm{dens}} = -\partial\mathcal{L}/\partial y_i$.
    pub force_y: Vec<S>,
    /// Variational energy $\mathcal{L}_{\mathrm{dens}}(X)$.
    pub energy: f64,
    /// Evaluation telemetry.
    pub stats: ScreenedPoissonStats,
}

impl<S: Scalar> Default for ScreenedPoissonOutput<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> ScreenedPoissonOutput<S> {
    /// Allocates an empty reusable output buffer.
    pub fn new() -> Self {
        Self {
            force_x: Vec::new(),
            force_y: Vec::new(),
            energy: 0.0,
            stats: ScreenedPoissonStats::default(),
        }
    }

    fn resize_and_zero(&mut self, n: usize) {
        self.force_x.resize(n, S::ZERO);
        self.force_x.fill(S::ZERO);
        self.force_y.resize(n, S::ZERO);
        self.force_y.fill(S::ZERO);
        self.energy = 0.0;
        self.stats = ScreenedPoissonStats::default();
    }
}

/// Reusable `f32` R2C-FFT Screened Poisson unilateral upper-bound density field evaluator.
#[derive(Debug, Clone)]
pub struct ScreenedPoissonField2d<S: Scalar = f64> {
    config: ScreenedPoissonConfig,
    ws: R2cFft2dWorkspaceF32,
    grid_buf: Vec<Complex32>,
    excess_mask: Vec<u64>,
    kx_sq: Vec<f32>,
    ky_sq: Vec<f32>,
    gauss_x: Vec<f32>,
    gauss_y: Vec<f32>,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Scalar> Default for ScreenedPoissonField2d<S> {
    fn default() -> Self {
        Self::new(ScreenedPoissonConfig::default()).expect("default ScreenedPoissonConfig is valid")
    }
}

impl<S: Scalar> ScreenedPoissonField2d<S> {
    /// Allocates a [`ScreenedPoissonField2d`] workspace for `config`.
    pub fn new(config: ScreenedPoissonConfig) -> CoreResult<Self> {
        config.validate()?;
        let pad_dim = config.grid_size * config.padding_factor;
        let stride = pad_dim / 2 + 1;
        let mask_words = (pad_dim * pad_dim).div_ceil(64);
        Ok(Self {
            config,
            ws: R2cFft2dWorkspaceF32::new(),
            grid_buf: vec![Complex32::ZERO; pad_dim * stride],
            excess_mask: vec![0u64; mask_words],
            kx_sq: vec![0.0; stride],
            ky_sq: vec![0.0; pad_dim],
            gauss_x: vec![0.0; stride],
            gauss_y: vec![0.0; pad_dim],
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the active [`ScreenedPoissonConfig`].
    #[inline]
    pub fn config(&self) -> ScreenedPoissonConfig {
        self.config
    }

    /// Updates the field configuration, resizing internal `f32` workspaces only when grid dimensions change.
    pub fn set_config(&mut self, config: ScreenedPoissonConfig) -> CoreResult<()> {
        config.validate()?;
        let new_pad = config.grid_size * config.padding_factor;
        let old_pad = self.config.grid_size * self.config.padding_factor;
        if new_pad != old_pad {
            let stride = new_pad / 2 + 1;
            let mask_words = (new_pad * new_pad).div_ceil(64);
            self.grid_buf.resize(new_pad * stride, Complex32::ZERO);
            self.excess_mask.resize(mask_words, 0u64);
            self.kx_sq.resize(stride, 0.0);
            self.ky_sq.resize(new_pad, 0.0);
            self.gauss_x.resize(stride, 0.0);
            self.gauss_y.resize(new_pad, 0.0);
        }
        self.config = config;
        Ok(())
    }

    /// Returns total heap bytes allocated by internal `f32` grids, bitmasks, and FFT tables.
    pub fn workspace_memory_bytes(&self) -> usize {
        use std::mem::size_of;
        self.grid_buf.capacity() * size_of::<Complex32>()
            + self.excess_mask.capacity() * size_of::<u64>()
            + (self.kx_sq.capacity()
                + self.ky_sq.capacity()
                + self.gauss_x.capacity()
                + self.gauss_y.capacity())
                * size_of::<f32>()
            + self.ws.workspace_memory_bytes()
    }

    /// Evaluates unilateral Screened Poisson energy and forces over the adaptive bounding box of `layout`.
    pub fn evaluate(
        &mut self,
        layout: &Layout2d<S>,
        out: &mut ScreenedPoissonOutput<S>,
    ) -> CoreResult<()> {
        self.evaluate_xy(layout.x(), layout.y(), out)
    }

    /// Slice-based variant of [`Self::evaluate`].
    pub fn evaluate_xy(
        &mut self,
        xs: &[S],
        ys: &[S],
        out: &mut ScreenedPoissonOutput<S>,
    ) -> CoreResult<()> {
        if xs.len() != ys.len() {
            return Err(CoreError::dimension_mismatch(
                xs.len(),
                ys.len(),
                "ScreenedPoissonField2d::evaluate_xy",
            ));
        }
        if xs.is_empty() {
            out.resize_and_zero(0);
            return Ok(());
        }
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for (&x_s, &y_s) in xs.iter().zip(ys.iter()) {
            let x = x_s.to_f64();
            let y = y_s.to_f64();
            if !x.is_finite() || !y.is_finite() {
                return Err(CoreError::non_finite("ScreenedPoissonField2d::evaluate_xy"));
            }
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        self.evaluate_xy_in_bounds(xs, ys, [min_x, max_x, min_y, max_y], out)
    }

    /// Evaluates unilateral Screened Poisson energy and forces inside fixed `bounds = [min_x, max_x, min_y, max_y]`.
    pub fn evaluate_in_bounds(
        &mut self,
        layout: &Layout2d<S>,
        bounds: [f64; 4],
        out: &mut ScreenedPoissonOutput<S>,
    ) -> CoreResult<()> {
        self.evaluate_xy_in_bounds(layout.x(), layout.y(), bounds, out)
    }

    /// Slice-based variant of [`Self::evaluate_in_bounds`].
    pub fn evaluate_xy_in_bounds(
        &mut self,
        xs: &[S],
        ys: &[S],
        bounds: [f64; 4],
        out: &mut ScreenedPoissonOutput<S>,
    ) -> CoreResult<()> {
        if xs.len() != ys.len() {
            return Err(CoreError::dimension_mismatch(
                xs.len(),
                ys.len(),
                "ScreenedPoissonField2d::evaluate_xy_in_bounds",
            ));
        }
        let [min_x, max_x, min_y, max_y] = bounds;
        if !min_x.is_finite()
            || !max_x.is_finite()
            || !min_y.is_finite()
            || !max_y.is_finite()
            || max_x < min_x
            || max_y < min_y
        {
            return Err(CoreError::invalid_input(format!(
                "bounds must be finite with min <= max, got {bounds:?}"
            )));
        }

        let n = xs.len();
        out.resize_and_zero(n);
        if n == 0 {
            return Ok(());
        }

        let g = self.config.grid_size;
        let pad_dim = g * self.config.padding_factor;
        let stride = pad_dim / 2 + 1;
        let dx_raw = max_x - min_x;
        let dy_raw = max_y - min_y;
        let w_span = dx_raw.max(dy_raw).max(1e-4);
        let cx = 0.5 * (min_x + max_x);
        let cy = 0.5 * (min_y + max_y);
        let interior_span = w_span * (1.0 + 2.0 * self.config.margin_ratio);
        let h = interior_span / (g as f64);
        let inv_h = 1.0 / h;
        let x0 = cx - 0.5 * interior_span;
        let y0 = cy - 0.5 * interior_span;

        let sigma_eff = self
            .config
            .sigma
            .unwrap_or_else(|| (self.config.sigma_ratio * w_span).max(1.5 * h));
        let kappa_eff = self
            .config
            .kappa
            .unwrap_or_else(|| 1.0 / (self.config.screening_length_ratio * w_span));
        let area_ref = dx_raw.max(2.0 * sigma_eff) * dy_raw.max(2.0 * sigma_eff);
        let rho_max_eff = self
            .config
            .rho_max
            .unwrap_or_else(|| self.config.rho_max_factor / area_ref);

        // 1. Bilinear splat of empirical density (1 / (N * h^2)) into packed Complex32 spatial buffer.
        self.grid_buf.fill(Complex32::ZERO);
        let m_cell = (1.0 / ((n as f64) * h * h)) as f32;
        let g_max_idx = (g - 2) as f64;
        for (&x_s, &y_s) in xs.iter().zip(ys.iter()) {
            let x = x_s.to_f64();
            let y = y_s.to_f64();
            if !x.is_finite() || !y.is_finite() {
                return Err(CoreError::non_finite(
                    "ScreenedPoissonField2d::evaluate_xy_in_bounds",
                ));
            }
            let u = (x - x0) * inv_h;
            let v = (y - y0) * inv_h;
            let u0_f = u.floor().clamp(0.0, g_max_idx);
            let v0_f = v.floor().clamp(0.0, g_max_idx);
            let sx = (u - u0_f).clamp(0.0, 1.0) as f32;
            let sy = (v - v0_f).clamp(0.0, 1.0) as f32;
            let u0 = u0_f as usize;
            let v0 = v0_f as usize;
            let omx = 1.0_f32 - sx;
            let omy = 1.0_f32 - sy;

            add_real_cell(&mut self.grid_buf, stride, v0, u0, m_cell * omx * omy);
            add_real_cell(&mut self.grid_buf, stride, v0, u0 + 1, m_cell * sx * omy);
            add_real_cell(&mut self.grid_buf, stride, v0 + 1, u0, m_cell * omx * sy);
            add_real_cell(&mut self.grid_buf, stride, v0 + 1, u0 + 1, m_cell * sx * sy);
        }

        // 2. Execute non-generic f32 R2C FFT pipeline.
        let stats = solve_density_pipeline_f32(
            &self.config,
            h,
            sigma_eff,
            kappa_eff,
            rho_max_eff,
            &mut self.ws,
            &mut self.grid_buf,
            &mut self.excess_mask,
            &mut self.kx_sq,
            &mut self.ky_sq,
            &mut self.gauss_x,
            &mut self.gauss_y,
        )?;

        out.energy = stats.energy;
        out.stats = stats;
        if stats.early_exit_no_excess {
            return Ok(());
        }

        // 3. Exact bilinear splat-transpose gradient gather: F_i = -(1 / N) * grad P(x_i).
        let inv_nh = 1.0 / ((n as f64) * h);
        for i in 0..n {
            let x = xs[i].to_f64();
            let y = ys[i].to_f64();
            let u = (x - x0) * inv_h;
            let v = (y - y0) * inv_h;
            let u0_f = u.floor().clamp(0.0, g_max_idx);
            let v0_f = v.floor().clamp(0.0, g_max_idx);
            let sx = (u - u0_f).clamp(0.0, 1.0);
            let sy = (v - v0_f).clamp(0.0, 1.0);
            let u0 = u0_f as usize;
            let v0 = v0_f as usize;

            let p00 = read_real_cell(&self.grid_buf, stride, v0, u0) as f64;
            let p10 = read_real_cell(&self.grid_buf, stride, v0, u0 + 1) as f64;
            let p01 = read_real_cell(&self.grid_buf, stride, v0 + 1, u0) as f64;
            let p11 = read_real_cell(&self.grid_buf, stride, v0 + 1, u0 + 1) as f64;

            let grad_x = ((1.0 - sy) * (p10 - p00) + sy * (p11 - p01)) * inv_nh;
            let grad_y = ((1.0 - sx) * (p01 - p00) + sx * (p11 - p10)) * inv_nh;
            out.force_x[i] = S::from_f64(-grad_x);
            out.force_y[i] = S::from_f64(-grad_y);
        }

        Ok(())
    }
}

#[inline]
fn add_real_cell(buf: &mut [Complex32], stride: usize, row: usize, col: usize, delta: f32) {
    let slot = &mut buf[row * stride + (col >> 1)];
    if (col & 1) == 0 {
        slot.re += delta;
    } else {
        slot.im += delta;
    }
}

#[inline]
fn read_real_cell(buf: &[Complex32], stride: usize, row: usize, col: usize) -> f32 {
    let slot = buf[row * stride + (col >> 1)];
    if (col & 1) == 0 { slot.re } else { slot.im }
}

#[allow(clippy::too_many_arguments)]
fn solve_density_pipeline_f32(
    config: &ScreenedPoissonConfig,
    h: f64,
    sigma_eff: f64,
    kappa_eff: f64,
    rho_max_eff: f64,
    ws: &mut R2cFft2dWorkspaceF32,
    grid_buf: &mut [Complex32],
    excess_mask: &mut [u64],
    kx_sq: &mut [f32],
    ky_sq: &mut [f32],
    gauss_x: &mut [f32],
    gauss_y: &mut [f32],
) -> CoreResult<ScreenedPoissonStats> {
    let g = config.grid_size;
    let pad_dim = g * config.padding_factor;
    let stride = pad_dim / 2 + 1;
    let count_before = ws.fft_2d_call_count();

    // Populate 1D Fourier wavenumber and Gaussian tables.
    let domain_len = (pad_dim as f64) * h;
    let d_omega = std::f64::consts::TAU / domain_len;
    let half_sigma_sq = 0.5 * sigma_eff * sigma_eff;
    for (kx, (k2_slot, gx_slot)) in kx_sq.iter_mut().zip(gauss_x.iter_mut()).enumerate() {
        let wx = (kx as f64) * d_omega;
        let wx2 = wx * wx;
        *k2_slot = wx2 as f32;
        *gx_slot = (-half_sigma_sq * wx2).exp() as f32;
    }
    let half_pad = pad_dim / 2;
    for (ky, (k2_slot, gy_slot)) in ky_sq.iter_mut().zip(gauss_y.iter_mut()).enumerate() {
        let signed = if ky <= half_pad {
            ky as isize
        } else {
            (ky as isize) - (pad_dim as isize)
        };
        let wy = (signed as f64) * d_omega;
        let wy2 = wy * wy;
        *k2_slot = wy2 as f32;
        *gy_slot = (-half_sigma_sq * wy2).exp() as f32;
    }

    // Smooth density: rho_V = K_sigma * rho_hat.
    ws.r2c_2d(grid_buf, pad_dim, pad_dim)?;
    for ky in 0..pad_dim {
        let gy = gauss_y[ky];
        let row = &mut grid_buf[ky * stride..ky * stride + stride];
        for kx in 0..stride {
            let k_gauss = gy * gauss_x[kx];
            row[kx] = row[kx].scale(k_gauss);
        }
    }
    ws.c2r_2d(grid_buf, pad_dim, pad_dim)?;

    // Unilateral excess source: delta_rho = max(0, rho_V - rho_max), mu_ent === 0.
    excess_mask.fill(0u64);
    let rho_max_f32 = rho_max_eff as f32;
    let half_width = pad_dim / 2;
    let mut max_density = 0.0_f64;
    let mut excess_sum = 0.0_f64;
    let mut total_excess_count = 0usize;
    let mut active_excess_count = 0usize;

    for v in 0..pad_dim {
        let row_off = v * stride;
        let cell_row_base = v * pad_dim;
        let in_active_y = v < g;
        for pair_idx in 0..half_width {
            let slot = &mut grid_buf[row_off + pair_idx];
            let u0 = pair_idx << 1;

            let r0 = slot.re;
            if (r0 as f64) > max_density {
                max_density = r0 as f64;
            }
            if r0 > rho_max_f32 {
                let d0 = r0 - rho_max_f32;
                slot.re = d0;
                excess_sum += d0 as f64;
                total_excess_count += 1;
                if in_active_y && u0 < g {
                    active_excess_count += 1;
                }
                let bit_idx = cell_row_base + u0;
                excess_mask[bit_idx >> 6] |= 1u64 << (bit_idx & 63);
            } else {
                slot.re = 0.0;
            }

            let r1 = slot.im;
            if (r1 as f64) > max_density {
                max_density = r1 as f64;
            }
            if r1 > rho_max_f32 {
                let d1 = r1 - rho_max_f32;
                slot.im = d1;
                excess_sum += d1 as f64;
                total_excess_count += 1;
                if in_active_y && (u0 + 1) < g {
                    active_excess_count += 1;
                }
                let bit_idx = cell_row_base + u0 + 1;
                excess_mask[bit_idx >> 6] |= 1u64 << (bit_idx & 63);
            } else {
                slot.im = 0.0;
            }
        }
    }

    let cell_area = h * h;
    let excess_mass = excess_sum * cell_area;
    let excess_fraction = (active_excess_count as f64) / ((g * g) as f64);

    if total_excess_count == 0 {
        return Ok(ScreenedPoissonStats {
            energy: 0.0,
            max_density,
            excess_fraction: 0.0,
            excess_mass: 0.0,
            sigma_used: sigma_eff,
            kappa_used: kappa_eff,
            rho_max_used: rho_max_eff,
            grid_dim: g,
            pad_dim,
            fft_2d_calls: ws.fft_2d_call_count() - count_before,
            early_exit_no_excess: true,
        });
    }

    // Forward FFT of delta_rho, apply Green operator, and accumulate discrete Parseval energy.
    ws.r2c_2d(grid_buf, pad_dim, pad_dim)?;
    let kappa_sq = (kappa_eff * kappa_eff) as f32;
    let is_screened = matches!(config.kernel, DensityFieldKernel::ScreenedPoisson);
    let exact_adjoint = config.exact_adjoint;
    let mut energy_accum = 0.0_f64;

    for ky in 0..pad_dim {
        let wy2 = ky_sq[ky];
        let gy = gauss_y[ky];
        let row = &mut grid_buf[ky * stride..ky * stride + stride];
        for kx in 0..stride {
            let g_hat = if is_screened {
                1.0_f32 / (kx_sq[kx] + wy2 + kappa_sq)
            } else {
                1.0_f32
            };
            let w_kx = if kx == 0 || kx == half_width {
                1.0_f64
            } else {
                2.0_f64
            };
            let c = row[kx];
            let norm_sq = (c.re as f64) * (c.re as f64) + (c.im as f64) * (c.im as f64);
            energy_accum += w_kx * (g_hat as f64) * norm_sq;

            let scale = if exact_adjoint {
                g_hat
            } else {
                g_hat * gy * gauss_x[kx]
            };
            row[kx] = c.scale(scale);
        }
    }

    let energy = 0.5 * cell_area * energy_accum / ((pad_dim * pad_dim) as f64);
    ws.c2r_2d(grid_buf, pad_dim, pad_dim)?;

    if exact_adjoint {
        // Multiply u = G * delta_rho by indicator 1_{rho_V > rho_max}, then convolve with K_sigma.
        for v in 0..pad_dim {
            let row_off = v * stride;
            let cell_row_base = v * pad_dim;
            for pair_idx in 0..half_width {
                let bit_base = cell_row_base + (pair_idx << 1);
                let word = excess_mask[bit_base >> 6];
                let shift0 = bit_base & 63;
                let slot = &mut grid_buf[row_off + pair_idx];
                if ((word >> shift0) & 1) == 0 {
                    slot.re = 0.0;
                }
                if ((word >> (shift0 + 1)) & 1) == 0 {
                    slot.im = 0.0;
                }
            }
        }

        ws.r2c_2d(grid_buf, pad_dim, pad_dim)?;
        for ky in 0..pad_dim {
            let gy = gauss_y[ky];
            let row = &mut grid_buf[ky * stride..ky * stride + stride];
            for kx in 0..stride {
                row[kx] = row[kx].scale(gy * gauss_x[kx]);
            }
        }
        ws.c2r_2d(grid_buf, pad_dim, pad_dim)?;
    }

    Ok(ScreenedPoissonStats {
        energy,
        max_density,
        excess_fraction,
        excess_mass,
        sigma_used: sigma_eff,
        kappa_used: kappa_eff,
        rho_max_used: rho_max_eff,
        grid_dim: g,
        pad_dim,
        fft_2d_calls: ws.fft_2d_call_count() - count_before,
        early_exit_no_excess: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_core::DeterministicRng;

    fn run_fd_gradient_check(exact_adjoint: bool, rho_max: Option<f64>, sigma: f64, kappa: f64) {
        let mut rng = DeterministicRng::from_seed_u64(77);
        let n = 16usize;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        // Tight overdense core (10 pts) + boundary (4 pts) + outer whitespace (2 pts),
        // offset from exact grid lines where bilinear splatting has C^0 derivative kinks.
        for i in 0..10 {
            let angle = 0.31 + (i as f64) * std::f64::consts::TAU / 10.0;
            xs.push(0.17 + 0.25 * angle.cos() + 0.05 * (rng.next_f64() - 0.5));
            ys.push(-0.13 + 0.25 * angle.sin() + 0.05 * (rng.next_f64() - 0.5));
        }
        for i in 0..4 {
            let angle = 0.43 + (i as f64) * std::f64::consts::TAU / 4.0;
            xs.push(0.17 + 0.70 * angle.cos());
            ys.push(-0.13 + 0.70 * angle.sin());
        }
        xs.push(3.23);
        ys.push(-2.91);
        xs.push(-3.13);
        ys.push(2.83);

        let cfg = ScreenedPoissonConfig {
            kernel: DensityFieldKernel::ScreenedPoisson,
            grid_size: 128,
            padding_factor: 2,
            margin_ratio: 0.1,
            sigma: Some(sigma),
            kappa: Some(kappa),
            rho_max,
            exact_adjoint,
            ..ScreenedPoissonConfig::default()
        };
        let mut field = ScreenedPoissonField2d::<f64>::new(cfg).unwrap();
        let bounds = [-5.0, 5.0, -5.0, 5.0];
        let mut base_out = ScreenedPoissonOutput::new();
        field
            .evaluate_xy_in_bounds(&xs, &ys, bounds, &mut base_out)
            .unwrap();
        assert!(!base_out.stats.early_exit_no_excess);
        assert_eq!(
            base_out.stats.fft_2d_calls,
            if exact_adjoint { 6 } else { 4 }
        );

        let eps = 2e-3_f64;
        let mut probe_out = ScreenedPoissonOutput::new();
        let mut max_err = 0.0_f64;
        let mut max_grad = 0.0_f64;
        for i in 0..n {
            let mut xp = xs.clone();
            let mut xm = xs.clone();
            xp[i] += eps;
            xm[i] -= eps;
            field
                .evaluate_xy_in_bounds(&xp, &ys, bounds, &mut probe_out)
                .unwrap();
            let lp = probe_out.energy;
            field
                .evaluate_xy_in_bounds(&xm, &ys, bounds, &mut probe_out)
                .unwrap();
            let lm = probe_out.energy;
            let fd_x = (lp - lm) / (2.0 * eps);
            let anal_grad_x = -base_out.force_x[i];
            max_grad = max_grad.max(anal_grad_x.abs());
            max_err = max_err.max((anal_grad_x - fd_x).abs());

            let mut yp = ys.clone();
            let mut ym = ys.clone();
            yp[i] += eps;
            ym[i] -= eps;
            field
                .evaluate_xy_in_bounds(&xs, &yp, bounds, &mut probe_out)
                .unwrap();
            let lpy = probe_out.energy;
            field
                .evaluate_xy_in_bounds(&xs, &ym, bounds, &mut probe_out)
                .unwrap();
            let lmy = probe_out.energy;
            let fd_y = (lpy - lmy) / (2.0 * eps);
            let anal_grad_y = -base_out.force_y[i];
            max_grad = max_grad.max(anal_grad_y.abs());
            max_err = max_err.max((anal_grad_y - fd_y).abs());
        }

        assert!(
            max_grad > 1e-4,
            "expected non-trivial gradient, got {max_grad}"
        );
        let rel_err = max_err / max_grad.max(1e-12);
        assert!(
            rel_err < 1e-2 && max_err < 2e-4,
            "FD mismatch (exact_adjoint={exact_adjoint}): max_err={max_err}, rel_err={rel_err}, max_grad={max_grad}"
        );

        // Pitfall #1 sign check: core particles must be pushed outward from core center, never imploding.
        let cx: f64 = xs[..10].iter().sum::<f64>() / 10.0;
        let cy: f64 = ys[..10].iter().sum::<f64>() / 10.0;
        let radial_dot: f64 = (0..10)
            .map(|i| (xs[i] - cx) * base_out.force_x[i] + (ys[i] - cy) * base_out.force_y[i])
            .sum();
        assert!(
            radial_dot > 0.0,
            "Pitfall #1 sign check failed (exact_adjoint={exact_adjoint}): radial_dot={radial_dot}"
        );
    }

    #[test]
    fn test_pitfall1_fd_exact_adjoint_and_unmasked_rho_zero() {
        run_fd_gradient_check(true, Some(0.05), 0.35, 0.5);
        run_fd_gradient_check(false, Some(0.0), 0.35, 0.5);
    }

    #[test]
    fn test_unilateral_whitespace_invariance_and_early_exit() {
        // Uniform 8x8 grid with rho_max above peak density -> early exit, zero force.
        let mut xs = Vec::with_capacity(64);
        let mut ys = Vec::with_capacity(64);
        for r in 0..8 {
            for c in 0..8 {
                xs.push(c as f64);
                ys.push(r as f64);
            }
        }
        let cfg = ScreenedPoissonConfig {
            grid_size: 128,
            padding_factor: 2,
            rho_max: Some(10.0),
            ..ScreenedPoissonConfig::default()
        };
        let mut field = ScreenedPoissonField2d::<f64>::new(cfg).unwrap();
        let mut out = ScreenedPoissonOutput::new();
        field.evaluate_xy(&xs, &ys, &mut out).unwrap();
        assert!(out.stats.early_exit_no_excess);
        assert_eq!(out.energy, 0.0);
        assert_eq!(out.stats.fft_2d_calls, 2);
        assert!(out.force_x.iter().all(|&f| f == 0.0));
        assert!(out.force_y.iter().all(|&f| f == 0.0));

        // Dense cluster near (-4, -4) + distant outlier in whitespace at (+4, +4).
        let mut cxs = Vec::new();
        let mut cys = Vec::new();
        for i in 0..32 {
            let ang = (i as f64) * std::f64::consts::TAU / 32.0;
            cxs.push(-4.0 + 0.1 * ang.cos());
            cys.push(-4.0 + 0.1 * ang.sin());
        }
        cxs.push(4.0);
        cys.push(4.0);

        let cfg2 = ScreenedPoissonConfig {
            grid_size: 128,
            padding_factor: 2,
            sigma: Some(0.25),
            rho_max: Some(0.1),
            exact_adjoint: true,
            ..ScreenedPoissonConfig::default()
        };
        let mut field2 = ScreenedPoissonField2d::<f64>::new(cfg2).unwrap();
        let mut out2 = ScreenedPoissonOutput::new();
        field2
            .evaluate_xy_in_bounds(&cxs, &cys, [-5.0, 5.0, -5.0, 5.0], &mut out2)
            .unwrap();
        assert!(!out2.stats.early_exit_no_excess);
        let outlier_fx = out2.force_x[32];
        let outlier_fy = out2.force_y[32];
        let core_force_max = out2.force_x[..32]
            .iter()
            .zip(out2.force_y[..32].iter())
            .map(|(&fx, &fy)| fx.hypot(fy))
            .fold(0.0_f64, f64::max);
        assert!(core_force_max > 1e-3);
        assert!(
            outlier_fx.abs() < 1e-8 && outlier_fy.abs() < 1e-8,
            "whitespace outlier force should be ~0 within f32 FFT noise: ({outlier_fx}, {outlier_fy})"
        );
    }

    #[test]
    fn test_pitfall2_plateau_expansion_screened_poisson_vs_single_gaussian() {
        // Dense uniform 40x40 plateau inside [-1, 1]^2 (with rho_max low enough that plateau is 3x overdense).
        let side = 40usize;
        let mut xs = Vec::with_capacity(side * side);
        let mut ys = Vec::with_capacity(side * side);
        for r in 0..side {
            for c in 0..side {
                let x = -1.0 + 2.0 * (c as f64) / ((side - 1) as f64);
                let y = -1.0 + 2.0 * (r as f64) / ((side - 1) as f64);
                if x * x + y * y <= 1.0 {
                    xs.push(x);
                    ys.push(y);
                }
            }
        }
        let n = xs.len();
        let bounds = [-3.0, 3.0, -3.0, 3.0];

        let sp_cfg = ScreenedPoissonConfig {
            kernel: DensityFieldKernel::ScreenedPoisson,
            grid_size: 256,
            padding_factor: 2,
            sigma: Some(0.08),
            kappa: Some(0.8),
            rho_max: Some(0.08),
            exact_adjoint: true,
            ..ScreenedPoissonConfig::default()
        };
        let sg_cfg = ScreenedPoissonConfig {
            kernel: DensityFieldKernel::SingleGaussian,
            ..sp_cfg
        };

        let mut sp_field = ScreenedPoissonField2d::<f64>::new(sp_cfg).unwrap();
        let mut sg_field = ScreenedPoissonField2d::<f64>::new(sg_cfg).unwrap();
        let mut sp_out = ScreenedPoissonOutput::new();
        let mut sg_out = ScreenedPoissonOutput::new();
        sp_field
            .evaluate_xy_in_bounds(&xs, &ys, bounds, &mut sp_out)
            .unwrap();
        sg_field
            .evaluate_xy_in_bounds(&xs, &ys, bounds, &mut sg_out)
            .unwrap();

        let mut sp_interior_radial = 0.0_f64;
        let mut sg_interior_radial = 0.0_f64;
        let mut interior_count = 0usize;
        for i in 0..n {
            let r = (xs[i] * xs[i] + ys[i] * ys[i]).sqrt();
            if r > 0.1 && r < 0.55 {
                let urx = xs[i] / r;
                let ury = ys[i] / r;
                sp_interior_radial += sp_out.force_x[i] * urx + sp_out.force_y[i] * ury;
                sg_interior_radial += sg_out.force_x[i] * urx + sg_out.force_y[i] * ury;
                interior_count += 1;
            }
        }
        sp_interior_radial /= interior_count as f64;
        sg_interior_radial /= interior_count as f64;

        assert!(
            sp_interior_radial > 1e-6,
            "ScreenedPoisson must exert positive outward radial force in plateau interior, got {sp_interior_radial}"
        );
        assert!(
            sg_interior_radial.abs() < 0.05 * sp_interior_radial,
            "SingleGaussian interior plateau force should be near zero relative to ScreenedPoisson: sg={sg_interior_radial}, sp={sp_interior_radial}"
        );
    }

    #[test]
    fn test_default_2048_memory_budget_and_finite_eval() {
        let mut field =
            ScreenedPoissonField2d::<f64>::new(ScreenedPoissonConfig::default()).unwrap();
        let mem = field.workspace_memory_bytes();
        assert!(
            mem <= 75_000_000,
            "2048x2048 padded f32 workspace exceeded 75 MB budget: {mem} bytes"
        );

        let mut rng = DeterministicRng::from_seed_u64(2048);
        let n = 1024usize;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            xs.push(rng.next_f64() * 2.0 - 1.0);
            ys.push(rng.next_f64() * 2.0 - 1.0);
        }
        let mut out = ScreenedPoissonOutput::new();
        field.evaluate_xy(&xs, &ys, &mut out).unwrap();
        assert_eq!(out.stats.grid_dim, 2048);
        assert_eq!(out.stats.pad_dim, 4096);
        assert!(out.energy.is_finite());
        assert!(out.force_x.iter().all(|v| v.is_finite()));
        assert!(out.force_y.iter().all(|v| v.is_finite()));
    }
}
