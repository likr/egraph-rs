mod entropy;
mod kl_repulsion;

pub use entropy::compute_fit_entropy_gradient;
pub use kl_repulsion::compute_fit_kl_repulsion;

/// Evaluates cubic Catmull-Rom / Lagrange interpolation weights for fraction `t` in `[0, 1)`.
#[inline]
fn cubic_weights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;

    [
        -0.5 * t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    ]
}

pub(super) struct ParticleStencil {
    pub(super) len: usize,
    pub(super) u_idx: [usize; 4],
    pub(super) v_idx: [usize; 4],
    pub(super) wu: [f64; 4],
    pub(super) wv: [f64; 4],
}

/// Grid geometry and interpolation mode for 2D FFT kernel evaluation.
pub(super) struct FitGridSpec {
    pub(super) center_x: f64,
    pub(super) center_y: f64,
    pub(super) bbox_min_x: f64,
    pub(super) bbox_min_y: f64,
    pub(super) hx: f64,
    pub(super) hy: f64,
    pub(super) nx: usize,
    pub(super) ny: usize,
    pub(super) use_bilinear: bool,
    pub(super) softening_sq: f64,
}

impl FitGridSpec {
    pub(super) fn from_points(
        points: &[[f64; 2]],
        intervals: usize,
        interpolation_points: usize,
    ) -> Self {
        // Bound adaptive grid expansion to 512x512 (1024x1024 padded = 16 MiB per Complex<f64> buffer)
        // so large spans stay memory-safe while keeping hx, hy <= 1.0 whenever span <= 512.
        const MAX_ADAPTIVE_GRID_DIM: usize = 512;

        let mut min_x = points[0][0];
        let mut max_x = points[0][0];
        let mut min_y = points[0][1];
        let mut max_y = points[0][1];

        for p in points.iter().skip(1) {
            min_x = min_x.min(p[0]);
            max_x = max_x.max(p[0]);
            min_y = min_y.min(p[1]);
            max_y = max_y.max(p[1]);
        }

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

        let target_grid_x = (intervals * interpolation_points).max(16);
        let target_grid_y = (intervals * interpolation_points).max(16);

        let base_nx = target_grid_x.next_power_of_two();
        let base_ny = target_grid_y.next_power_of_two();

        let adaptive_nx = if total_span_x.is_finite() && total_span_x > base_nx as f64 {
            let needed = (total_span_x.ceil() as usize).min(MAX_ADAPTIVE_GRID_DIM);
            needed.next_power_of_two().min(MAX_ADAPTIVE_GRID_DIM)
        } else {
            base_nx
        };
        let adaptive_ny = if total_span_y.is_finite() && total_span_y > base_ny as f64 {
            let needed = (total_span_y.ceil() as usize).min(MAX_ADAPTIVE_GRID_DIM);
            needed.next_power_of_two().min(MAX_ADAPTIVE_GRID_DIM)
        } else {
            base_ny
        };

        let nx = base_nx.max(adaptive_nx);
        let ny = base_ny.max(adaptive_ny);

        let hx = total_span_x / nx as f64;
        let hy = total_span_y / ny as f64;
        let h_max = hx.max(hy);
        // When grid spacing exceeds the unit Cauchy kernel scale, switch to non-negative bilinear
        // interpolation and add grid-scale softening so intra-cell K_2(0) aliasing decays as O(1/h).
        let use_bilinear = h_max > 1.0;
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
            use_bilinear,
            softening_sq: s * s,
        }
    }
}

#[inline]
pub(super) fn build_particle_stencil(px: f64, py: f64, grid: &FitGridSpec) -> ParticleStencil {
    let u_real = (px - grid.bbox_min_x) / grid.hx;
    let v_real = (py - grid.bbox_min_y) / grid.hy;

    if grid.use_bilinear {
        let u0 = (u_real.floor() as isize).clamp(0, (grid.nx - 2) as isize) as usize;
        let v0 = (v_real.floor() as isize).clamp(0, (grid.ny - 2) as isize) as usize;
        let tu = (u_real - u0 as f64).clamp(0.0, 1.0);
        let tv = (v_real - v0 as f64).clamp(0.0, 1.0);
        return ParticleStencil {
            len: 2,
            u_idx: [u0, u0 + 1, 0, 0],
            v_idx: [v0, v0 + 1, 0, 0],
            wu: [1.0 - tu, tu, 0.0, 0.0],
            wv: [1.0 - tv, tv, 0.0, 0.0],
        };
    }

    let u0 = (u_real.floor() as isize).max(1) as usize;
    let v0 = (v_real.floor() as isize).max(1) as usize;

    let wu = cubic_weights(u_real - u_real.floor());
    let wv = cubic_weights(v_real - v_real.floor());

    let u_idx = [
        (u0 - 1).min(grid.nx - 1),
        u0.min(grid.nx - 1),
        (u0 + 1).min(grid.nx - 1),
        (u0 + 2).min(grid.nx - 1),
    ];
    let v_idx = [
        (v0 - 1).min(grid.ny - 1),
        v0.min(grid.ny - 1),
        (v0 + 1).min(grid.ny - 1),
        (v0 + 2).min(grid.ny - 1),
    ];

    ParticleStencil {
        len: 4,
        u_idx,
        v_idx,
        wu,
        wv,
    }
}

#[cfg(test)]
mod tests;
