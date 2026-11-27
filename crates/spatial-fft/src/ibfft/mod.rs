//! Interpolation-Based 2D Fast Fourier Transform (`IbFft2d`) with `G_dual` particle binning.
//!
//! Supports $p=1$ bilinear interpolation (default, free of Runge oscillation) and $p=3$ cubic
//! Catmull-Rom / Lagrange interpolation over a bounded adaptive padded grid for non-circular
//! (open-boundary Toeplitz) 2D convolution on [`Fft2dWorkspace`].

use std::marker::PhantomData;

use egraph_core::{CoreError, CoreResult, EuclideanLayout, EuclideanStorageRef, Layout2d, Scalar};

use crate::fft::{Complex64, Fft2dWorkspace};

mod config;
#[cfg(test)]
mod tests;

pub(crate) use config::{BinnedParticle, DualCell, cell_grid_indices, stencil_axis};
pub use config::{IbFftConfig, IbFftGridSpec, InterpolationOrder};

/// Reusable `G_dual` particle binner and padded non-circular 2D FFT interpolator.
///
/// Groups particles into dual grid cells (`G_dual`) so S2G anterpolation and G2S interpolation
/// accumulate/read each cell's shared stencil in cache registers with zero per-particle stencil
/// allocations.
#[derive(Debug, Clone)]
pub struct IbFft2d<S: Scalar = f64> {
    config: IbFftConfig,
    spec: Option<IbFftGridSpec>,
    num_points: usize,
    /// Padded forward FFT of scalar charge `1`.
    pub(crate) grid_1_fft: Vec<Complex64>,
    /// Padded forward FFT of packed coordinate charges `qx + i * qy`.
    pub(crate) grid_xy_fft: Vec<Complex64>,
    /// Scratch buffer for kernel setup and frequency-domain products.
    pub(crate) kernel_buf: Vec<Complex64>,
    pub(crate) conv_buf: Vec<Complex64>,
    // G_dual CSR binning workspace (cleared in O(active_bins) per step).
    bin_counts: Vec<u32>,
    bin_dense: Vec<u32>,
    active_bin_ids: Vec<usize>,
    active_cursors: Vec<usize>,
    pub(crate) active_cells: Vec<DualCell>,
    pub(crate) binned: Vec<BinnedParticle>,
    scratch_x: Vec<S>,
    scratch_y: Vec<S>,
    _marker: PhantomData<S>,
}

impl<S: Scalar> Default for IbFft2d<S> {
    fn default() -> Self {
        Self::new(IbFftConfig::default()).expect("default IbFftConfig is valid")
    }
}

impl<S: Scalar> IbFft2d<S> {
    /// Creates a new [`IbFft2d`] workspace with validated configuration.
    pub fn new(config: IbFftConfig) -> CoreResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            spec: None,
            num_points: 0,
            grid_1_fft: Vec::new(),
            grid_xy_fft: Vec::new(),
            kernel_buf: Vec::new(),
            conv_buf: Vec::new(),
            bin_counts: Vec::new(),
            bin_dense: Vec::new(),
            active_bin_ids: Vec::new(),
            active_cursors: Vec::new(),
            active_cells: Vec::new(),
            binned: Vec::new(),
            scratch_x: Vec::new(),
            scratch_y: Vec::new(),
            _marker: PhantomData,
        })
    }

    /// Creates an [`IbFft2d`] workspace with default grid sizing and explicit [`InterpolationOrder`].
    pub fn with_order(order: InterpolationOrder) -> Self {
        Self::new(IbFftConfig::default().with_order(order))
            .expect("default IbFftConfig with valid order is valid")
    }

    /// Returns the configuration of this interpolator.
    #[inline]
    pub fn config(&self) -> &IbFftConfig {
        &self.config
    }

    /// Returns the adaptive grid geometry computed by the most recent [`IbFft2d::prepare`].
    #[inline]
    pub fn grid_spec(&self) -> Option<&IbFftGridSpec> {
        self.spec.as_ref()
    }

    /// Bins particles into `G_dual`, scatters `(1, qx, qy)` onto the zero-padded grid, and
    /// executes the two shared particle forward 2D FFTs (`grid_1_fft` and `grid_xy_fft`).
    ///
    /// Returns the number of forward 2D FFTs executed (`0` when `layout.len() < 2`, `2` otherwise).
    pub fn prepare(&mut self, ws: &mut Fft2dWorkspace, layout: &Layout2d<S>) -> CoreResult<u64> {
        self.prepare_xy(ws, layout.x(), layout.y())
    }

    /// [`EuclideanLayout`] variant of [`IbFft2d::prepare`].
    ///
    /// Uses zero-copy SoA slices for [`Layout2d`] and accepts 2D `LayoutNd`; fails fast with
    /// [`CoreError::DimensionMismatch`] when `layout.dims() != 2`.
    pub fn prepare_euclidean<L: EuclideanLayout<Scalar = S> + ?Sized>(
        &mut self,
        ws: &mut Fft2dWorkspace,
        layout: &L,
    ) -> CoreResult<u64> {
        match layout.as_euclidean_storage() {
            EuclideanStorageRef::Soa2d { x, y } => self.prepare_xy(ws, x, y),
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
                    "IbFft2d::prepare_euclidean",
                )
                .and_then(|()| self.prepare_xy(ws, &sx, &sy));
                self.scratch_x = sx;
                self.scratch_y = sy;
                res
            }
            EuclideanStorageRef::Soa3d { .. } => Err(CoreError::dimension_mismatch(
                2,
                3,
                "IbFft2d::prepare_euclidean",
            )),
            EuclideanStorageRef::FlatNd { dims, .. } => Err(CoreError::dimension_mismatch(
                2,
                dims,
                "IbFft2d::prepare_euclidean",
            )),
        }
    }

    /// Slice-based variant of [`IbFft2d::prepare`].
    pub fn prepare_xy(&mut self, ws: &mut Fft2dWorkspace, xs: &[S], ys: &[S]) -> CoreResult<u64> {
        if xs.len() != ys.len() {
            return Err(CoreError::dimension_mismatch(
                xs.len(),
                ys.len(),
                "IbFft2d::prepare_xy",
            ));
        }
        let n = xs.len();
        for i in 0..n {
            if !xs[i].is_finite() || !ys[i].is_finite() {
                return Err(CoreError::non_finite("IbFft2d::prepare_xy"));
            }
        }
        self.num_points = n;
        if n < 2 {
            self.spec = None;
            self.active_cells.clear();
            return Ok(0);
        }

        let mut min_x = xs[0].to_f64();
        let mut max_x = min_x;
        let mut min_y = ys[0].to_f64();
        let mut max_y = min_y;
        for i in 1..n {
            let x = xs[i].to_f64();
            let y = ys[i].to_f64();
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        let spec = IbFftGridSpec::from_bounds(min_x, max_x, min_y, max_y, &self.config);

        let pad_len = spec.pad_nx * spec.pad_ny;
        self.grid_1_fft.resize(pad_len, Complex64::ZERO);
        self.grid_1_fft.fill(Complex64::ZERO);
        self.grid_xy_fft.resize(pad_len, Complex64::ZERO);
        self.grid_xy_fft.fill(Complex64::ZERO);
        self.kernel_buf.resize(pad_len, Complex64::ZERO);
        self.conv_buf.resize(pad_len, Complex64::ZERO);

        self.bin_particles_g_dual(xs, ys, &spec);
        self.scatter_g_dual(&spec);
        self.spec = Some(spec.clone());

        ws.fft_2d(&mut self.grid_1_fft, spec.pad_nx, spec.pad_ny, false)?;
        ws.fft_2d(&mut self.grid_xy_fft, spec.pad_nx, spec.pad_ny, false)?;
        Ok(2)
    }

    /// Evaluates a single isotropic translation-invariant kernel `K(dist_sq)` over the most
    /// recently prepared particle grid using non-circular padded 2D FFT convolution.
    ///
    /// Computes for each particle `i`:
    /// - `phi_1[i] = sum_j K(||y_i - y_j||^2)` (including self-term at `j == i`; callers can
    ///   subtract `K(0)` if excluding self)
    /// - `force_x[i] = sum_j (y_{i,x} - y_{j,x}) * K(||y_i - y_j||^2)`
    /// - `force_y[i] = sum_j (y_{i,y} - y_{j,y}) * K(||y_i - y_j||^2)`
    pub fn evaluate_isotropic_kernel<F>(
        &mut self,
        ws: &mut Fft2dWorkspace,
        kernel_fn: F,
        out_phi_1: &mut [S],
        out_force_x: &mut [S],
        out_force_y: &mut [S],
    ) -> CoreResult<()>
    where
        F: Fn(f64) -> f64,
    {
        let n = self.num_points;
        if out_phi_1.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_phi_1.len(),
                "IbFft2d::evaluate_isotropic_kernel::out_phi_1",
            ));
        }
        if out_force_x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_force_x.len(),
                "IbFft2d::evaluate_isotropic_kernel::out_force_x",
            ));
        }
        if out_force_y.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_force_y.len(),
                "IbFft2d::evaluate_isotropic_kernel::out_force_y",
            ));
        }
        if n < 2 {
            out_phi_1.fill(S::ZERO);
            out_force_x.fill(S::ZERO);
            out_force_y.fill(S::ZERO);
            return Ok(());
        }

        let spec = self
            .spec
            .as_ref()
            .expect("spec must exist when num_points >= 2");
        let pad_nx = spec.pad_nx;
        let pad_ny = spec.pad_ny;

        // Build wrap-around symmetric real kernel on padded grid so circular FFT yields linear
        // (non-circular) convolution on the interior [0..nx) x [0..ny) domain.
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
                self.kernel_buf[row + u] = Complex64::new(kernel_fn(dx * dx + dy_sq), 0.0);
            }
        }
        ws.fft_2d(&mut self.kernel_buf, pad_nx, pad_ny, false)?;

        // Inverse FFT for scalar charge convolution -> phi_1.
        for (dst, (&g1, &kf)) in self
            .conv_buf
            .iter_mut()
            .zip(self.grid_1_fft.iter().zip(self.kernel_buf.iter()))
        {
            *dst = g1.scale(kf.re);
        }
        ws.fft_2d(&mut self.conv_buf, pad_nx, pad_ny, true)?;
        let order = spec.effective_order;
        let nx = spec.nx;
        let ny = spec.ny;
        for cell in &self.active_cells {
            let (u_idx, v_idx, len) = cell_grid_indices(cell.u0, cell.v0, order);
            for p in &self.binned[cell.start..cell.end] {
                let (_, _, wu, _) = stencil_axis(p.u_real, nx, order);
                let (_, _, wv, _) = stencil_axis(p.v_real, ny, order);
                let mut phi = 0.0;
                for k in 0..len {
                    let row_off = v_idx[k] * pad_nx;
                    let mut row_sum = 0.0;
                    for l in 0..len {
                        row_sum += wu[l] * self.conv_buf[row_off + u_idx[l]].re;
                    }
                    phi += wv[k] * row_sum;
                }
                out_phi_1[p.orig_idx as usize] = S::from_f64(phi);
            }
        }

        // Inverse FFT for packed coordinate convolution -> phi_x + i * phi_y.
        for (dst, (&gxy, &kf)) in self
            .conv_buf
            .iter_mut()
            .zip(self.grid_xy_fft.iter().zip(self.kernel_buf.iter()))
        {
            *dst = gxy.scale(kf.re);
        }
        ws.fft_2d(&mut self.conv_buf, pad_nx, pad_ny, true)?;
        for cell in &self.active_cells {
            let (u_idx, v_idx, len) = cell_grid_indices(cell.u0, cell.v0, order);
            for p in &self.binned[cell.start..cell.end] {
                let (_, _, wu, _) = stencil_axis(p.u_real, nx, order);
                let (_, _, wv, _) = stencil_axis(p.v_real, ny, order);
                let mut phi_x = 0.0;
                let mut phi_y = 0.0;
                for k in 0..len {
                    let row_off = v_idx[k] * pad_nx;
                    let mut rx = 0.0;
                    let mut ry = 0.0;
                    for l in 0..len {
                        let val = self.conv_buf[row_off + u_idx[l]];
                        rx += wu[l] * val.re;
                        ry += wu[l] * val.im;
                    }
                    phi_x += wv[k] * rx;
                    phi_y += wv[k] * ry;
                }
                let idx = p.orig_idx as usize;
                let phi_1 = out_phi_1[idx].to_f64();
                out_force_x[idx] = S::from_f64(p.qx * phi_1 - phi_x);
                out_force_y[idx] = S::from_f64(p.qy * phi_1 - phi_y);
            }
        }

        Ok(())
    }

    fn bin_particles_g_dual(&mut self, xs: &[S], ys: &[S], spec: &IbFftGridSpec) {
        let n = xs.len();
        let grid_cells = spec.nx * spec.ny;
        if self.bin_counts.len() < grid_cells {
            self.bin_counts.resize(grid_cells, 0);
            self.bin_dense.resize(grid_cells, 0);
        }

        self.active_bin_ids.clear();
        self.binned.resize(n, BinnedParticle::default());

        let inv_hx = 1.0 / spec.hx;
        let inv_hy = 1.0 / spec.hy;

        // First pass: assign cell IDs and record only touched bins for O(min(N, nx*ny)) reset.
        for i in 0..n {
            let qx = xs[i].to_f64() - spec.center_x;
            let qy = ys[i].to_f64() - spec.center_y;
            let u_real = (qx - spec.bbox_min_x) * inv_hx;
            let v_real = (qy - spec.bbox_min_y) * inv_hy;
            let (u0, _, _, _) = stencil_axis(u_real, spec.nx, spec.effective_order);
            let (v0, _, _, _) = stencil_axis(v_real, spec.ny, spec.effective_order);
            let bin = v0 * spec.nx + u0;
            if self.bin_counts[bin] == 0 {
                self.active_bin_ids.push(bin);
            }
            self.bin_counts[bin] += 1;
            self.binned[i] = BinnedParticle {
                orig_idx: i as u32,
                qx,
                qy,
                u_real,
                v_real,
            };
        }

        // Deterministic row-major order over active G_dual cells.
        self.active_bin_ids.sort_unstable();
        let num_active = self.active_bin_ids.len();
        self.active_cells.clear();
        self.active_cells.reserve(num_active);
        self.active_cursors.clear();
        self.active_cursors.reserve(num_active);

        let mut offset = 0usize;
        for (dense, &bin) in self.active_bin_ids.iter().enumerate() {
            let count = self.bin_counts[bin] as usize;
            self.bin_dense[bin] = dense as u32;
            self.active_cells.push(DualCell {
                u0: bin % spec.nx,
                v0: bin / spec.nx,
                start: offset,
                end: offset + count,
            });
            self.active_cursors.push(offset);
            offset += count;
        }

        // Recompute bin assignments directly from `xs`/`ys` on pass 2 so particles scatter into
        // `self.binned` in-place without allocating a secondary particle scratch buffer.
        for i in 0..n {
            let qx = xs[i].to_f64() - spec.center_x;
            let qy = ys[i].to_f64() - spec.center_y;
            let u_real = (qx - spec.bbox_min_x) * inv_hx;
            let v_real = (qy - spec.bbox_min_y) * inv_hy;
            let (u0, _, _, _) = stencil_axis(u_real, spec.nx, spec.effective_order);
            let (v0, _, _, _) = stencil_axis(v_real, spec.ny, spec.effective_order);
            let bin = v0 * spec.nx + u0;
            let dense = self.bin_dense[bin] as usize;
            let pos = self.active_cursors[dense];
            self.active_cursors[dense] = pos + 1;
            self.binned[pos] = BinnedParticle {
                orig_idx: i as u32,
                qx,
                qy,
                u_real,
                v_real,
            };
        }

        // Clear only touched bins so next prepare() starts clean in O(active_bins).
        for &bin in &self.active_bin_ids {
            self.bin_counts[bin] = 0;
        }
    }

    fn scatter_g_dual(&mut self, spec: &IbFftGridSpec) {
        let order = spec.effective_order;
        let nx = spec.nx;
        let ny = spec.ny;
        let pad_nx = spec.pad_nx;

        for cell in &self.active_cells {
            let (u_idx, v_idx, len) = cell_grid_indices(cell.u0, cell.v0, order);
            let mut local_1 = [[0.0_f64; 4]; 4];
            let mut local_x = [[0.0_f64; 4]; 4];
            let mut local_y = [[0.0_f64; 4]; 4];

            for p in &self.binned[cell.start..cell.end] {
                let (_, _, wu, _) = stencil_axis(p.u_real, nx, order);
                let (_, _, wv, _) = stencil_axis(p.v_real, ny, order);
                for k in 0..len {
                    let w_k = wv[k];
                    for l in 0..len {
                        let w = w_k * wu[l];
                        local_1[k][l] += w;
                        local_x[k][l] += w * p.qx;
                        local_y[k][l] += w * p.qy;
                    }
                }
            }

            for k in 0..len {
                let row_off = v_idx[k] * pad_nx;
                for l in 0..len {
                    let idx = row_off + u_idx[l];
                    self.grid_1_fft[idx].re += local_1[k][l];
                    self.grid_xy_fft[idx].re += local_x[k][l];
                    self.grid_xy_fft[idx].im += local_y[k][l];
                }
            }
        }
    }
}
