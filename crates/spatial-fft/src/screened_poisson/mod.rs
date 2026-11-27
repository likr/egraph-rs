//! Screened Poisson unilateral upper-bound 2D density field (`f32` R2C FFT workspace).
//!
//! Solves the non-local Eulerian repulsion potential
//! $\hat{P}(k) = \frac{\hat{K}_{\sigma_0}(k)}{\|k\|^2 + \kappa_0^2}\widehat{\delta\rho}(k)$
//! for the unilateral excess density $\delta\rho(p) = \max(0, \rho_V(p) - \rho_{\max})$ with zero
//! entropy term ($\mu_{\mathrm{ent}} \equiv 0$, preserving whitespace). Supports exact variational
//! adjoint masking (`exact_adjoint: true`, default) via a compact `u64` excess bitmask as well as
//! the single-solve unmasked formulation (`exact_adjoint: false`).

use egraph_core::{CoreError, CoreResult, EuclideanLayout, EuclideanStorageRef, Layout2d, Scalar};

use crate::fft_r2c::{Complex32, R2cFft2dWorkspaceF32};

mod config;
mod solver;
#[cfg(test)]
mod tests;

pub use config::{
    DensityFieldKernel, ScreenedPoissonConfig, ScreenedPoissonOutput, ScreenedPoissonStats,
};
use solver::{add_real_cell, read_real_cell, solve_density_pipeline_f32};

/// Reusable `f32` R2C-FFT Screened Poisson unilateral upper-bound density field evaluator.
#[derive(Debug, Clone)]
pub struct ScreenedPoissonField2d<S: Scalar = f64> {
    config: ScreenedPoissonConfig,
    ws: R2cFft2dWorkspaceF32,
    grid_buf: Vec<Complex32>,
    density_grid: Vec<S>,
    density_grid_dim: usize,
    excess_mask: Vec<u64>,
    kx_sq: Vec<f32>,
    ky_sq: Vec<f32>,
    gauss_x: Vec<f32>,
    gauss_y: Vec<f32>,
    scratch_x: Vec<S>,
    scratch_y: Vec<S>,
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
            density_grid: Vec::new(),
            density_grid_dim: 0,
            excess_mask: vec![0u64; mask_words],
            kx_sq: vec![0.0; stride],
            ky_sq: vec![0.0; pad_dim],
            gauss_x: vec![0.0; stride],
            gauss_y: vec![0.0; pad_dim],
            scratch_x: Vec::new(),
            scratch_y: Vec::new(),
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the active [`ScreenedPoissonConfig`].
    #[inline]
    pub fn config(&self) -> &ScreenedPoissonConfig {
        &self.config
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
        self.density_grid_dim = 0;
        self.config = config;
        Ok(())
    }

    /// Returns total heap bytes allocated by internal `f32` grids, bitmasks, density snapshot, and FFT tables.
    pub fn workspace_memory_bytes(&self) -> usize {
        use std::mem::size_of;
        self.grid_buf.capacity() * size_of::<Complex32>()
            + self.density_grid.capacity() * size_of::<S>()
            + self.excess_mask.capacity() * size_of::<u64>()
            + (self.kx_sq.capacity()
                + self.ky_sq.capacity()
                + self.gauss_x.capacity()
                + self.gauss_y.capacity())
                * size_of::<f32>()
            + (self.scratch_x.capacity() + self.scratch_y.capacity()) * size_of::<S>()
            + self.ws.workspace_memory_bytes()
    }

    /// Returns the row-major $G \times G$ smoothed empirical density $\rho_V$ from the most
    /// recent non-empty evaluation (`None` before evaluation or after an empty evaluation).
    #[inline]
    pub fn density_grid(&self) -> Option<&[S]> {
        if self.density_grid_dim > 0
            && self.density_grid.len() == self.density_grid_dim * self.density_grid_dim
        {
            Some(&self.density_grid)
        } else {
            None
        }
    }

    /// Returns the interior grid dimension $G$ of [`Self::density_grid`] (`0` when unpopulated).
    #[inline]
    pub fn density_grid_dim(&self) -> usize {
        if self.density_grid().is_some() {
            self.density_grid_dim
        } else {
            0
        }
    }

    /// Evaluates unilateral Screened Poisson energy and forces over the adaptive bounding box of `layout`.
    pub fn evaluate(
        &mut self,
        layout: &Layout2d<S>,
        out: &mut ScreenedPoissonOutput<S>,
    ) -> CoreResult<()> {
        self.evaluate_xy(layout.x(), layout.y(), out)
    }

    /// [`EuclideanLayout`] variant of [`Self::evaluate`].
    ///
    /// Uses zero-copy SoA slices for [`Layout2d`] and accepts 2D `LayoutNd`; fails fast with
    /// [`CoreError::DimensionMismatch`] when `layout.dims() != 2`.
    pub fn evaluate_euclidean<L: EuclideanLayout<Scalar = S> + ?Sized>(
        &mut self,
        layout: &L,
        out: &mut ScreenedPoissonOutput<S>,
    ) -> CoreResult<()> {
        match layout.as_euclidean_storage() {
            EuclideanStorageRef::Soa2d { x, y } => self.evaluate_xy(x, y, out),
            EuclideanStorageRef::FlatNd {
                coords,
                num_nodes,
                dims: 2,
            } => {
                let mut sx = std::mem::take(&mut self.scratch_x);
                let mut sy = std::mem::take(&mut self.scratch_y);
                let res = crate::deinterleave_flat_2d(
                    coords,
                    num_nodes,
                    &mut sx,
                    &mut sy,
                    "ScreenedPoissonField2d::evaluate_euclidean",
                )
                .and_then(|()| self.evaluate_xy(&sx, &sy, out));
                self.scratch_x = sx;
                self.scratch_y = sy;
                res
            }
            EuclideanStorageRef::Soa3d { .. } => Err(CoreError::dimension_mismatch(
                2,
                3,
                "ScreenedPoissonField2d::evaluate_euclidean",
            )),
            EuclideanStorageRef::FlatNd { dims, .. } => Err(CoreError::dimension_mismatch(
                2,
                dims,
                "ScreenedPoissonField2d::evaluate_euclidean",
            )),
        }
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
            self.density_grid_dim = 0;
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
        self.evaluate_xy_in_bounds(xs, ys, [min_x, min_y, max_x, max_y], out)
    }

    /// Evaluates unilateral Screened Poisson energy and forces inside fixed `bounds = [min_x, min_y, max_x, max_y]`.
    pub fn evaluate_in_bounds(
        &mut self,
        layout: &Layout2d<S>,
        bounds: [f64; 4],
        out: &mut ScreenedPoissonOutput<S>,
    ) -> CoreResult<()> {
        self.evaluate_xy_in_bounds(layout.x(), layout.y(), bounds, out)
    }

    /// [`EuclideanLayout`] variant of [`Self::evaluate_in_bounds`].
    ///
    /// Uses zero-copy SoA slices for [`Layout2d`] and accepts 2D `LayoutNd`; fails fast with
    /// [`CoreError::DimensionMismatch`] when `layout.dims() != 2`.
    pub fn evaluate_euclidean_in_bounds<L: EuclideanLayout<Scalar = S> + ?Sized>(
        &mut self,
        layout: &L,
        bounds: [f64; 4],
        out: &mut ScreenedPoissonOutput<S>,
    ) -> CoreResult<()> {
        match layout.as_euclidean_storage() {
            EuclideanStorageRef::Soa2d { x, y } => self.evaluate_xy_in_bounds(x, y, bounds, out),
            EuclideanStorageRef::FlatNd {
                coords,
                num_nodes,
                dims: 2,
            } => {
                let mut sx = std::mem::take(&mut self.scratch_x);
                let mut sy = std::mem::take(&mut self.scratch_y);
                let res = crate::deinterleave_flat_2d(
                    coords,
                    num_nodes,
                    &mut sx,
                    &mut sy,
                    "ScreenedPoissonField2d::evaluate_euclidean_in_bounds",
                )
                .and_then(|()| self.evaluate_xy_in_bounds(&sx, &sy, bounds, out));
                self.scratch_x = sx;
                self.scratch_y = sy;
                res
            }
            EuclideanStorageRef::Soa3d { .. } => Err(CoreError::dimension_mismatch(
                2,
                3,
                "ScreenedPoissonField2d::evaluate_euclidean_in_bounds",
            )),
            EuclideanStorageRef::FlatNd { dims, .. } => Err(CoreError::dimension_mismatch(
                2,
                dims,
                "ScreenedPoissonField2d::evaluate_euclidean_in_bounds",
            )),
        }
    }

    /// Slice-based variant of [`Self::evaluate_in_bounds`] expecting `bounds = [min_x, min_y, max_x, max_y]`.
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
        let [min_x, min_y, max_x, max_y] = bounds;
        if !min_x.is_finite()
            || !min_y.is_finite()
            || !max_x.is_finite()
            || !max_y.is_finite()
            || max_x < min_x
            || max_y < min_y
        {
            return Err(CoreError::invalid_input(format!(
                "bounds [min_x, min_y, max_x, max_y] must be finite with min <= max, got {bounds:?}"
            )));
        }

        let n = xs.len();
        out.resize_and_zero(n);
        self.density_grid_dim = 0;
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
            &mut self.density_grid,
            &mut self.excess_mask,
            &mut self.kx_sq,
            &mut self.ky_sq,
            &mut self.gauss_x,
            &mut self.gauss_y,
        )?;

        self.density_grid_dim = g;
        out.density_grid_dim = g;
        out.density_grid.resize(self.density_grid.len(), S::ZERO);
        out.density_grid.copy_from_slice(&self.density_grid);
        out.energy = stats.energy;
        let early_exit = stats.early_exit_no_excess;
        out.stats = stats;
        if early_exit {
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
