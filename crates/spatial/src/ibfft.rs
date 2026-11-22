//! Interpolation-Based 2D Fast Fourier Transform (`IbFft2d`) with `G_dual` particle binning.
//!
//! Supports $p=1$ bilinear interpolation (default, free of Runge oscillation) and $p=3$ cubic
//! Catmull-Rom / Lagrange interpolation over a bounded adaptive padded grid for non-circular
//! (open-boundary Toeplitz) 2D convolution on [`Fft2dWorkspace`].

use std::marker::PhantomData;

use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};

use crate::fft::{Complex64, Fft2dWorkspace};

/// Polynomial interpolation order per grid axis in [`IbFft2d`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub enum InterpolationOrder {
    /// $p = 1$ bilinear interpolation (`2x2` stencil, non-negative weights, default).
    #[default]
    Bilinear,
    /// $p = 3$ cubic Lagrange / Catmull-Rom interpolation (`4x4` stencil).
    Cubic,
}

impl InterpolationOrder {
    /// Parses numeric polynomial order (`1` -> [`Bilinear`](Self::Bilinear), `3` -> [`Cubic`](Self::Cubic)).
    pub fn from_order(order: usize) -> CoreResult<Self> {
        match order {
            1 => Ok(Self::Bilinear),
            3 => Ok(Self::Cubic),
            _ => Err(CoreError::invalid_input(format!(
                "unsupported interpolation order {order}; expected 1 (bilinear) or 3 (cubic)"
            ))),
        }
    }

    /// Returns the numeric polynomial order (`1` or `3`).
    #[inline]
    pub const fn order(self) -> usize {
        match self {
            Self::Bilinear => 1,
            Self::Cubic => 3,
        }
    }

    /// Returns the stencil width per axis (`2` for bilinear, `4` for cubic).
    #[inline]
    pub const fn stencil_len(self) -> usize {
        match self {
            Self::Bilinear => 2,
            Self::Cubic => 4,
        }
    }
}

/// Configuration for [`IbFft2d`] grid sizing and polynomial order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IbFftConfig {
    /// Interpolation order (`Bilinear` $p=1$ by default).
    pub order: InterpolationOrder,
    /// Nominal interval count used to determine base grid resolution.
    pub intervals: usize,
    /// Minimum interior grid resolution per axis (must be a power of two >= 16).
    pub min_grid_dim: usize,
    /// Maximum interior grid resolution per axis (must be a power of two >= `min_grid_dim`).
    ///
    /// Bounds padded FFT memory to `(2 * max_grid_dim)^2` complex entries (`16 MiB` at `512`),
    /// keeping WebAssembly memory well below the 2-4 GiB ceiling on wide coordinate spans.
    pub max_grid_dim: usize,
}

impl Default for IbFftConfig {
    fn default() -> Self {
        Self {
            order: InterpolationOrder::Bilinear,
            intervals: 25,
            min_grid_dim: 64,
            max_grid_dim: 512,
        }
    }
}

impl IbFftConfig {
    /// Validates grid configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.intervals == 0 {
            return Err(CoreError::invalid_input("intervals must be >= 1"));
        }
        if self.min_grid_dim < 16 || !self.min_grid_dim.is_power_of_two() {
            return Err(CoreError::invalid_input(format!(
                "min_grid_dim must be a power of two >= 16, got {}",
                self.min_grid_dim
            )));
        }
        if self.max_grid_dim < self.min_grid_dim || !self.max_grid_dim.is_power_of_two() {
            return Err(CoreError::invalid_input(format!(
                "max_grid_dim must be a power of two >= min_grid_dim ({}), got {}",
                self.min_grid_dim, self.max_grid_dim
            )));
        }
        Ok(())
    }
}

/// Computed adaptive grid geometry for the current coordinate batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IbFftGridSpec {
    pub center_x: f64,
    pub center_y: f64,
    pub bbox_min_x: f64,
    pub bbox_min_y: f64,
    pub hx: f64,
    pub hy: f64,
    pub nx: usize,
    pub ny: usize,
    pub pad_nx: usize,
    pub pad_ny: usize,
    pub effective_order: InterpolationOrder,
    pub softening_sq: f64,
}

impl IbFftGridSpec {
    pub(crate) fn from_bounds(
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        config: &IbFftConfig,
    ) -> Self {
        let center_x = 0.5 * (min_x + max_x);
        let center_y = 0.5 * (min_y + max_y);

        let span_x = (max_x - min_x).max(1e-4);
        let span_y = (max_y - min_y).max(1e-4);
        let margin_x = span_x * 0.1;
        let margin_y = span_y * 0.1;

        let bbox_min_x = (min_x - center_x) - margin_x;
        let bbox_max_x = (max_x - center_x) + margin_x;
        let bbox_min_y = (min_y - center_y) - margin_y;
        let bbox_max_y = (max_y - center_y) + margin_y;

        let total_span_x = bbox_max_x - bbox_min_x;
        let total_span_y = bbox_max_y - bbox_min_y;

        let order_mul = match config.order {
            InterpolationOrder::Bilinear => 1,
            InterpolationOrder::Cubic => 3,
        };
        let min_cubic_floor = match config.order {
            InterpolationOrder::Bilinear => 16,
            // Ensure cubic 4-point stencils stay strictly inside interior grid indices [1..nx-3].
            InterpolationOrder::Cubic => 32,
        };
        let target_grid = (config.intervals.saturating_mul(order_mul))
            .max(config.min_grid_dim)
            .max(min_cubic_floor);
        let base_dim = target_grid.next_power_of_two().min(config.max_grid_dim);

        let nx = adaptive_dim(total_span_x, base_dim, config.max_grid_dim);
        let ny = adaptive_dim(total_span_y, base_dim, config.max_grid_dim);

        let hx = total_span_x / (nx as f64);
        let hy = total_span_y / (ny as f64);
        let h_max = hx.max(hy);

        // When grid spacing exceeds unit scale, fall back to non-negative bilinear weights and
        // soften the kernel at the grid scale so intra-cell self-aliasing decays smoothly.
        let effective_order = if h_max > 1.0 {
            InterpolationOrder::Bilinear
        } else {
            config.order
        };
        let s = 0.5 * (h_max - 1.0).max(0.0);

        Self {
            center_x,
            center_y,
            bbox_min_x,
            bbox_min_y,
            hx,
            hy,
            nx,
            ny,
            pad_nx: nx * 2,
            pad_ny: ny * 2,
            effective_order,
            softening_sq: s * s,
        }
    }
}

#[inline]
fn adaptive_dim(total_span: f64, base_dim: usize, max_dim: usize) -> usize {
    if total_span.is_finite() && total_span > (base_dim as f64) {
        let needed = (total_span.ceil() as usize).min(max_dim);
        needed.next_power_of_two().min(max_dim).max(base_dim)
    } else {
        base_dim
    }
}

#[inline]
pub(crate) fn cubic_weights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    ]
}

#[inline]
pub(crate) fn stencil_axis(
    u_real: f64,
    dim: usize,
    order: InterpolationOrder,
) -> (usize, [usize; 4], [f64; 4], usize) {
    match order {
        InterpolationOrder::Bilinear => {
            let u0 = (u_real.floor() as isize).clamp(0, (dim as isize) - 2) as usize;
            let t = (u_real - u0 as f64).clamp(0.0, 1.0);
            (u0, [u0, u0 + 1, 0, 0], [1.0 - t, t, 0.0, 0.0], 2)
        }
        InterpolationOrder::Cubic => {
            let u0 = (u_real.floor() as isize).clamp(1, (dim as isize) - 3) as usize;
            let t = (u_real - u0 as f64).clamp(0.0, 1.0);
            let w = cubic_weights(t);
            (u0, [u0 - 1, u0, u0 + 1, u0 + 2], w, 4)
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BinnedParticle {
    pub(crate) orig_idx: u32,
    pub(crate) qx: f64,
    pub(crate) qy: f64,
    pub(crate) u_real: f64,
    pub(crate) v_real: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DualCell {
    pub(crate) u0: usize,
    pub(crate) v0: usize,
    pub(crate) start: usize,
    pub(crate) end: usize,
}

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
            _marker: PhantomData,
        })
    }

    /// Creates an [`IbFft2d`] workspace with default grid sizing and explicit [`InterpolationOrder`].
    pub fn with_order(order: InterpolationOrder) -> Self {
        Self::new(IbFftConfig {
            order,
            ..IbFftConfig::default()
        })
        .expect("default IbFftConfig with valid order is valid")
    }

    /// Returns the configuration of this interpolator.
    #[inline]
    pub fn config(&self) -> IbFftConfig {
        self.config
    }

    /// Returns the adaptive grid geometry computed by the most recent [`IbFft2d::prepare`].
    #[inline]
    pub fn grid_spec(&self) -> Option<IbFftGridSpec> {
        self.spec
    }

    /// Bins particles into `G_dual`, scatters `(1, qx, qy)` onto the zero-padded grid, and
    /// executes the two shared particle forward 2D FFTs (`grid_1_fft` and `grid_xy_fft`).
    ///
    /// Returns the number of forward 2D FFTs executed (`0` when `layout.len() < 2`, `2` otherwise).
    pub fn prepare(&mut self, ws: &mut Fft2dWorkspace, layout: &Layout2d<S>) -> CoreResult<u64> {
        self.prepare_xy(ws, layout.x(), layout.y())
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
        self.spec = Some(spec);

        let pad_len = spec.pad_nx * spec.pad_ny;
        self.grid_1_fft.resize(pad_len, Complex64::ZERO);
        self.grid_1_fft.fill(Complex64::ZERO);
        self.grid_xy_fft.resize(pad_len, Complex64::ZERO);
        self.grid_xy_fft.fill(Complex64::ZERO);
        self.kernel_buf.resize(pad_len, Complex64::ZERO);
        self.conv_buf.resize(pad_len, Complex64::ZERO);

        self.bin_particles_g_dual(xs, ys, &spec);
        self.scatter_g_dual(&spec);

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

        let spec = self.spec.expect("spec must exist when num_points >= 2");
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

        // In-place out-of-order scatter into temporary? Use scratch or copy? Actually we overwrote
        // binned during pass 1; scatter into active_tmp or swap! Keeping a reusable `tmp_binned`
        // avoids overwriting unread entries. Wait: let's store pass-1 particles in `conv_buf`? No,
        // swap with a tiny `binned_scratch` or place directly! Wait: let's check simple in-place
        // scatter using `binned_scratch`: wait, we can't borrow `binned` mutably while reading it.
        // Look how clean `bin_scatter_buf` (Wait: let's add `binned_unsorted` or do cycle sort?
        // Actually `binned_unsorted` reused vector costs only 40 bytes * n once, 0 alloc after!).
        // Wait: even better, compute `bin` per particle during pass 1 in `bin_of_particle` and
        // scatter directly from `xs, ys` into `self.binned` on pass 2 — ZERO second particle Vec!
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

#[inline]
pub(crate) fn cell_grid_indices(
    u0: usize,
    v0: usize,
    order: InterpolationOrder,
) -> ([usize; 4], [usize; 4], usize) {
    match order {
        InterpolationOrder::Bilinear => ([u0, u0 + 1, 0, 0], [v0, v0 + 1, 0, 0], 2),
        InterpolationOrder::Cubic => (
            [u0 - 1, u0, u0 + 1, u0 + 2],
            [v0 - 1, v0, v0 + 1, v0 + 2],
            4,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ibfft_default_is_bilinear_and_matches_exact_cauchy() {
        let mut ws = Fft2dWorkspace::new();
        let mut ibfft = IbFft2d::<f64>::default();
        assert_eq!(ibfft.config().order, InterpolationOrder::Bilinear);

        let layout = Layout2d::from_xy(
            vec![-1.2, -0.4, 0.2, 0.9, 1.5, -0.8, 0.6, 1.1],
            vec![0.7, -0.9, 0.3, -0.2, 0.8, 1.2, -1.1, 0.1],
        )
        .unwrap();
        let n = layout.len();

        let fwd = ibfft.prepare(&mut ws, &layout).unwrap();
        assert_eq!(fwd, 2);

        let mut phi_1 = vec![0.0; n];
        let mut fx = vec![0.0; n];
        let mut fy = vec![0.0; n];
        ibfft
            .evaluate_isotropic_kernel(
                &mut ws,
                |d_sq| 1.0 / (1.0 + d_sq),
                &mut phi_1,
                &mut fx,
                &mut fy,
            )
            .unwrap();
        assert_eq!(ws.fft_2d_call_count(), 5);

        // Compare against exact O(N^2) sum (including K(0) = 1.0 self term in phi_1).
        for i in 0..n {
            let mut exact_phi = 0.0;
            let mut exact_fx = 0.0;
            let mut exact_fy = 0.0;
            for j in 0..n {
                let dx = layout.x()[i] - layout.x()[j];
                let dy = layout.y()[i] - layout.y()[j];
                let k = 1.0 / (1.0 + dx * dx + dy * dy);
                exact_phi += k;
                exact_fx += dx * k;
                exact_fy += dy * k;
            }
            assert!(
                (phi_1[i] - exact_phi).abs() < 5e-3,
                "phi_1[{i}]={} vs {exact_phi}",
                phi_1[i]
            );
            assert!((fx[i] - exact_fx).abs() < 5e-3);
            assert!((fy[i] - exact_fy).abs() < 5e-3);
        }
    }

    #[test]
    fn test_ibfft_cubic_and_bounded_adaptive_grid_size() {
        let mut ws = Fft2dWorkspace::new();
        let mut ibfft = IbFft2d::<f32>::with_order(InterpolationOrder::Cubic);

        // Extreme coordinate span (> 512) must clamp grid to max_grid_dim = 512 and stay finite.
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for k in 0..12 {
            let v = -1200.0_f32 + (k as f32) * 240.0;
            xs.push(v);
            ys.push(v * 0.5);
        }
        let layout = Layout2d::from_xy(xs, ys).unwrap();
        ibfft.prepare(&mut ws, &layout).unwrap();
        let spec = ibfft.grid_spec().unwrap();
        assert_eq!(spec.nx, 512);
        assert_eq!(spec.ny, 512);
        assert_eq!(spec.effective_order, InterpolationOrder::Bilinear);
        assert!(spec.softening_sq > 0.0);

        let mut phi_1 = vec![0.0_f32; layout.len()];
        let mut fx = vec![0.0_f32; layout.len()];
        let mut fy = vec![0.0_f32; layout.len()];
        ibfft
            .evaluate_isotropic_kernel(
                &mut ws,
                |d_sq| 1.0 / (1.0 + spec.softening_sq + d_sq),
                &mut phi_1,
                &mut fx,
                &mut fy,
            )
            .unwrap();
        assert!(phi_1.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(fx.iter().all(|v| v.is_finite()));
        assert!(fy.iter().all(|v| v.is_finite()));

        assert!(matches!(
            InterpolationOrder::from_order(2),
            Err(CoreError::InvalidInput(_))
        ));
    }
}
