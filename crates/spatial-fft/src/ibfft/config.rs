use egraph_core::{CoreError, CoreResult};

/// Polynomial interpolation order per grid axis in [`super::IbFft2d`].
#[non_exhaustive]
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

/// Configuration for [`super::IbFft2d`] grid sizing and polynomial order.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
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
    /// Sets the polynomial interpolation order per grid axis.
    pub fn with_order(mut self, order: InterpolationOrder) -> Self {
        self.order = order;
        self
    }

    /// Sets the nominal interval count used to determine base grid resolution.
    pub fn with_intervals(mut self, intervals: usize) -> Self {
        self.intervals = intervals;
        self
    }

    /// Sets the minimum interior grid resolution per axis.
    pub fn with_min_grid_dim(mut self, min_grid_dim: usize) -> Self {
        self.min_grid_dim = min_grid_dim;
        self
    }

    /// Sets the maximum interior grid resolution per axis.
    pub fn with_max_grid_dim(mut self, max_grid_dim: usize) -> Self {
        self.max_grid_dim = max_grid_dim;
        self
    }

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
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
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
