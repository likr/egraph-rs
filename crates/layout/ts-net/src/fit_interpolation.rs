use crate::fft::{fft_2d, Complex};
use num_traits::Float;
use petgraph_drawing::DrawingValue;

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

struct ParticleStencil {
    len: usize,
    u_idx: [usize; 4],
    v_idx: [usize; 4],
    wu: [f64; 4],
    wv: [f64; 4],
}

/// Grid geometry and interpolation mode for 2D FFT kernel evaluation.
struct FitGridSpec {
    center_x: f64,
    center_y: f64,
    bbox_min_x: f64,
    bbox_min_y: f64,
    hx: f64,
    hy: f64,
    nx: usize,
    ny: usize,
    use_bilinear: bool,
    softening_sq: f64,
}

impl FitGridSpec {
    fn from_points(points: &[[f64; 2]], intervals: usize, interpolation_points: usize) -> Self {
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
fn build_particle_stencil(px: f64, py: f64, grid: &FitGridSpec) -> ParticleStencil {
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

/// Evaluates low-dimensional Student-t KL divergence repulsion forces in O(N) using 2D FFT interpolation.
///
/// Uses `K_1(d) = 1 / (1 + d^2)` for normalization `Z` (subtracting each particle's 4x4 grid self-term)
/// and `K_2(d) = 1 / (1 + d^2)^2` for repulsive forces `(px * phi_1_k2 - phi_x_k2, py * phi_1_k2 - phi_y_k2)`.
/// Computes the FFT and interpolation internals in `f64` to avoid `f32` cancellation in `phi_1_z - self_z`
/// and `px * phi_1_k2 - phi_x_k2`.
/// Returns `(rep_forces, total_z)` where `rep_forces[i] = (F_rep_x_i / Z, F_rep_y_i / Z)`.
pub fn compute_fit_kl_repulsion<S>(
    points: &[[S; 2]],
    intervals: usize,
    interpolation_points: usize,
) -> (Vec<[S; 2]>, S)
where
    S: DrawingValue + Float + Default,
{
    let n = points.len();
    if n < 2 {
        return (vec![[S::zero(), S::zero()]; n], S::one());
    }

    let pts64: Vec<[f64; 2]> = points
        .iter()
        .map(|p| [p[0].to_f64().unwrap(), p[1].to_f64().unwrap()])
        .collect();
    let (rep64, total_z64) = compute_fit_kl_repulsion_f64(&pts64, intervals, interpolation_points);
    let rep_out = rep64
        .into_iter()
        .map(|f| [S::from_f64(f[0]).unwrap(), S::from_f64(f[1]).unwrap()])
        .collect();
    (rep_out, S::from_f64(total_z64).unwrap())
}

fn compute_fit_kl_repulsion_f64(
    points: &[[f64; 2]],
    intervals: usize,
    interpolation_points: usize,
) -> (Vec<[f64; 2]>, f64) {
    let n = points.len();

    // Step 1: Compute adaptive grid geometry and center coordinates to reduce cancellation
    let grid = FitGridSpec::from_points(points, intervals, interpolation_points);
    let pad_nx = grid.nx * 2;
    let pad_ny = grid.ny * 2;

    // Step 2: Spread charges w^(1) = 1 and packed w^(x) + i w^(y) onto (pad_nx x pad_ny) grid
    let mut grid_1 = vec![Complex::zero(); pad_nx * pad_ny];
    let mut grid_xy = vec![Complex::zero(); pad_nx * pad_ny];
    let mut stencils = Vec::with_capacity(n);

    for p in points {
        let qx = p[0] - grid.center_x;
        let qy = p[1] - grid.center_y;
        let st = build_particle_stencil(qx, qy, &grid);

        for (k_idx, &u) in st.u_idx[..st.len].iter().enumerate() {
            let w_k = st.wu[k_idx];
            for (l_idx, &v) in st.v_idx[..st.len].iter().enumerate() {
                let w = w_k * st.wv[l_idx];
                let idx = v * pad_nx + u;
                grid_1[idx].re += w;
                grid_xy[idx].re += w * qx;
                grid_xy[idx].im += w * qy;
            }
        }
        stencils.push(st);
    }

    // Step 3: Pack even real kernels K_1 + i K_2 into one complex array and transform once.
    // Because K_1 and K_2 are real and circularly even, Re(FFT) = FFT(K_1) and Im(FFT) = FFT(K_2).
    let base_denom = 1.0 + grid.softening_sq;
    let mut kernel_fft = vec![Complex::zero(); pad_nx * pad_ny];
    for v in 0..pad_ny {
        let dy = if v < grid.ny {
            v as f64 * grid.hy
        } else {
            (v as f64 - pad_ny as f64) * grid.hy
        };

        for u in 0..pad_nx {
            let dx = if u < grid.nx {
                u as f64 * grid.hx
            } else {
                (u as f64 - pad_nx as f64) * grid.hx
            };

            let dist_sq = dx * dx + dy * dy;
            let k1 = 1.0 / (base_denom + dist_sq);
            let k2 = k1 * k1;
            kernel_fft[v * pad_nx + u] = Complex::new(k1, k2);
        }
    }
    fft_2d(&mut kernel_fft, pad_nx, pad_ny, false);

    // Step 4: 2D FFT convolutions using complex packing (5 FFTs total instead of 9):
    // - grid_1 * (FFT(K_1) + i FFT(K_2)) -> IFFT yields pot_1_z + i pot_1_k2
    // - (grid_x + i grid_y) * FFT(K_2) -> IFFT yields pot_x_k2 + i pot_y_k2
    fft_2d(&mut grid_1, pad_nx, pad_ny, false);
    for (g, &kf) in grid_1.iter_mut().zip(kernel_fft.iter()) {
        *g = *g * kf;
    }
    fft_2d(&mut grid_1, pad_nx, pad_ny, true);

    fft_2d(&mut grid_xy, pad_nx, pad_ny, false);
    for (g, &kf) in grid_xy.iter_mut().zip(kernel_fft.iter()) {
        *g = g.scale(kf.im);
    }
    fft_2d(&mut grid_xy, pad_nx, pad_ny, true);

    // Precompute local 7x7 K_1 stencil offsets for exact self-term subtraction s_i
    let mut k1_local = [[0.0f64; 7]; 7];
    for (du_idx, row) in k1_local.iter_mut().enumerate() {
        let dx = (du_idx as isize - 3) as f64 * grid.hx;
        for (dv_idx, cell) in row.iter_mut().enumerate() {
            let dy = (dv_idx as isize - 3) as f64 * grid.hy;
            *cell = 1.0 / (base_denom + dx * dx + dy * dy);
        }
    }

    // Step 5: Gather potentials at particle positions
    let mut rep_forces = vec![[0.0f64; 2]; n];
    let mut total_z = 0.0f64;

    for (i, p) in points.iter().enumerate() {
        let qx = p[0] - grid.center_x;
        let qy = p[1] - grid.center_y;
        let st = &stencils[i];

        let mut phi_1_z = 0.0f64;
        let mut phi_1_k2 = 0.0f64;
        let mut phi_x_k2 = 0.0f64;
        let mut phi_y_k2 = 0.0f64;

        for (k_idx, &u) in st.u_idx[..st.len].iter().enumerate() {
            let w_k = st.wu[k_idx];
            for (l_idx, &v) in st.v_idx[..st.len].iter().enumerate() {
                let w = w_k * st.wv[l_idx];
                let idx = v * pad_nx + u;
                phi_1_z += w * grid_1[idx].re;
                phi_1_k2 += w * grid_1[idx].im;
                phi_x_k2 += w * grid_xy[idx].re;
                phi_y_k2 += w * grid_xy[idx].im;
            }
        }

        let mut self_z = 0.0f64;
        for (k1, &uk1) in st.u_idx[..st.len].iter().enumerate() {
            for (l1, &vl1) in st.v_idx[..st.len].iter().enumerate() {
                let w1 = st.wu[k1] * st.wv[l1];
                for (k2, &uk2) in st.u_idx[..st.len].iter().enumerate() {
                    let du = (uk2 as isize - uk1 as isize + 3) as usize;
                    for (l2, &vl2) in st.v_idx[..st.len].iter().enumerate() {
                        let dv = (vl2 as isize - vl1 as isize + 3) as usize;
                        self_z += w1 * (st.wu[k2] * st.wv[l2]) * k1_local[du][dv];
                    }
                }
            }
        }

        let f_rep_x = qx * phi_1_k2 - phi_x_k2;
        let f_rep_y = qy * phi_1_k2 - phi_y_k2;
        let z_i = (phi_1_z - self_z).max(0.0);

        rep_forces[i] = [f_rep_x, f_rep_y];
        total_z += z_i;
    }

    total_z = total_z.max(1e-12);

    // Normalize repulsion forces by global Z and enforce exact Cauchy upper bound ||F_rep / Z|| <= 0.5
    // (since d / (1 + d^2) <= 0.5 for all pairwise distances d >= 0).
    for force in &mut rep_forces {
        force[0] /= total_z;
        force[1] /= total_z;
        let norm = (force[0] * force[0] + force[1] * force[1]).sqrt();
        if norm > 0.5 {
            let scale = 0.5 / norm;
            force[0] *= scale;
            force[1] *= scale;
        }
    }

    (rep_forces, total_z)
}

/// Evaluates the entropy repulsion gradient in O(N) using 2D FFT interpolation for L-tsNET.
///
/// Computes `(h_2 - x_i * h_1, h_3 - y_i * h_1)` where `h_1 = sum K_e`, `h_2 = sum x_k K_e`, `h_3 = sum y_k K_e`,
/// and `K_e = 1 / (epsilon_r + ||X_k - X_i||^2)`.
pub fn compute_fit_entropy_gradient<S>(
    points: &[[S; 2]],
    intervals: usize,
    interpolation_points: usize,
    epsilon_r: S,
) -> Vec<[S; 2]>
where
    S: DrawingValue + Float + Default,
{
    let n = points.len();
    if n < 2 {
        return vec![[S::zero(), S::zero()]; n];
    }

    let pts64: Vec<[f64; 2]> = points
        .iter()
        .map(|p| [p[0].to_f64().unwrap(), p[1].to_f64().unwrap()])
        .collect();
    let ent64 = compute_fit_entropy_gradient_f64(
        &pts64,
        intervals,
        interpolation_points,
        epsilon_r.to_f64().unwrap(),
    );
    ent64
        .into_iter()
        .map(|f| [S::from_f64(f[0]).unwrap(), S::from_f64(f[1]).unwrap()])
        .collect()
}

fn compute_fit_entropy_gradient_f64(
    points: &[[f64; 2]],
    intervals: usize,
    interpolation_points: usize,
    epsilon_r: f64,
) -> Vec<[f64; 2]> {
    let n = points.len();

    // Step 1: Compute adaptive grid geometry and center coordinates
    let grid = FitGridSpec::from_points(points, intervals, interpolation_points);
    let pad_nx = grid.nx * 2;
    let pad_ny = grid.ny * 2;

    // Step 2: Spread charges onto padded grid (packing w^(x) + i w^(y))
    let mut grid_1 = vec![Complex::zero(); pad_nx * pad_ny];
    let mut grid_xy = vec![Complex::zero(); pad_nx * pad_ny];
    let mut stencils = Vec::with_capacity(n);

    for p in points {
        let qx = p[0] - grid.center_x;
        let qy = p[1] - grid.center_y;
        let st = build_particle_stencil(qx, qy, &grid);

        for (k_idx, &u) in st.u_idx[..st.len].iter().enumerate() {
            let w_k = st.wu[k_idx];
            for (l_idx, &v) in st.v_idx[..st.len].iter().enumerate() {
                let w = w_k * st.wv[l_idx];
                let idx = v * pad_nx + u;
                grid_1[idx].re += w;
                grid_xy[idx].re += w * qx;
                grid_xy[idx].im += w * qy;
            }
        }
        stencils.push(st);
    }

    // Step 3: Compute Entropy kernel K_e = 1 / (epsilon_r + s^2 + dx^2 + dy^2) on padded 2D grid and its FFT
    let epsilon_eff = epsilon_r + grid.softening_sq;
    let mut kernel_grid = vec![Complex::zero(); pad_nx * pad_ny];
    for v in 0..pad_ny {
        let dy = if v < grid.ny {
            v as f64 * grid.hy
        } else {
            (v as f64 - pad_ny as f64) * grid.hy
        };

        for u in 0..pad_nx {
            let dx = if u < grid.nx {
                u as f64 * grid.hx
            } else {
                (u as f64 - pad_nx as f64) * grid.hx
            };

            let dist_sq = dx * dx + dy * dy;
            let val = 1.0 / (epsilon_eff + dist_sq);
            kernel_grid[v * pad_nx + u] = Complex::new(val, 0.0);
        }
    }

    fft_2d(&mut kernel_grid, pad_nx, pad_ny, false);

    // Step 4: 2D FFT convolution (5 FFTs total instead of 7)
    fft_2d(&mut grid_1, pad_nx, pad_ny, false);
    for (g, &kf) in grid_1.iter_mut().zip(kernel_grid.iter()) {
        *g = *g * kf;
    }
    fft_2d(&mut grid_1, pad_nx, pad_ny, true);

    fft_2d(&mut grid_xy, pad_nx, pad_ny, false);
    for (g, &kf) in grid_xy.iter_mut().zip(kernel_grid.iter()) {
        *g = *g * kf;
    }
    fft_2d(&mut grid_xy, pad_nx, pad_ny, true);

    // Step 5: Gather potentials h_1, h_2, h_3 and compute entropy gradient
    let mut ent_grad = vec![[0.0f64; 2]; n];

    for (i, p) in points.iter().enumerate() {
        let qx = p[0] - grid.center_x;
        let qy = p[1] - grid.center_y;
        let st = &stencils[i];

        let mut h_1 = 0.0f64;
        let mut h_2 = 0.0f64;
        let mut h_3 = 0.0f64;

        for (k_idx, &u) in st.u_idx[..st.len].iter().enumerate() {
            let w_k = st.wu[k_idx];
            for (l_idx, &v) in st.v_idx[..st.len].iter().enumerate() {
                let w = w_k * st.wv[l_idx];
                let idx = v * pad_nx + u;
                h_1 += w * grid_1[idx].re;
                h_2 += w * grid_xy[idx].re;
                h_3 += w * grid_xy[idx].im;
            }
        }

        let f_ent_x = h_2 - qx * h_1;
        let f_ent_y = h_3 - qy * h_1;

        ent_grad[i] = [f_ent_x, f_ent_y];
    }

    ent_grad
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_kl_repulsion(points: &[[f64; 2]]) -> (Vec<[f64; 2]>, f64) {
        let n = points.len();
        let mut forces = vec![[0.0; 2]; n];
        let mut total_z = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = points[i][0] - points[j][0];
                let dy = points[i][1] - points[j][1];
                let w = 1.0 / (1.0 + dx * dx + dy * dy);
                total_z += w;
                let w2 = w * w;
                forces[i][0] += w2 * dx;
                forces[i][1] += w2 * dy;
            }
        }
        let denom = total_z.max(1e-12);
        for f in &mut forces {
            f[0] /= denom;
            f[1] /= denom;
        }
        (forces, total_z)
    }

    #[test]
    fn test_compute_fit_kl_repulsion_matches_exact_small_and_large_span() {
        let small_points = [
            [-0.8, -0.4],
            [-0.5, 0.3],
            [-0.2, -0.6],
            [0.1, 0.5],
            [0.4, -0.1],
            [0.9, 0.7],
            [-1.1, 0.8],
            [1.2, -0.7],
        ];
        let large_points = [
            [-28.0, 11.0],
            [-19.5, -3.5],
            [-7.4, 6.8],
            [0.0, 0.0],
            [1.2, -14.1],
            [10.5, 13.7],
            [18.9, -12.0],
            [26.6, 9.7],
            [-27.5, 10.6],
            [26.1, 9.2],
        ];
        let medium_span_points = [
            [-100.0, -40.0],
            [-95.0, -39.0],
            [-20.0, 60.0],
            [0.0, 0.0],
            [1.5, 0.8],
            [45.0, -70.0],
            [98.0, 82.0],
            [100.0, 81.0],
        ];

        for (label, pts, max_force_err, max_z_rel_err) in [
            ("small_span", small_points.as_slice(), 5e-3, 0.05),
            ("large_span", large_points.as_slice(), 1e-2, 0.15),
            (
                "medium_span_adaptive",
                medium_span_points.as_slice(),
                2e-2,
                0.10,
            ),
        ] {
            let (exact_f, exact_z) = exact_kl_repulsion(pts);
            let (approx_f, approx_z) = compute_fit_kl_repulsion(pts, 25, 3);

            assert!(
                approx_z.is_finite() && approx_z > 0.0,
                "{label}: invalid total_z={approx_z}"
            );
            let z_rel_err = (approx_z - exact_z).abs() / exact_z;
            assert!(
                z_rel_err < max_z_rel_err,
                "{label}: total_z rel_err={z_rel_err} (approx={approx_z}, exact={exact_z})"
            );

            for (i, (af, ef)) in approx_f.iter().zip(exact_f.iter()).enumerate() {
                assert!(af[0].is_finite() && af[1].is_finite());
                let err = ((af[0] - ef[0]).powi(2) + (af[1] - ef[1]).powi(2)).sqrt();
                assert!(
                    err < max_force_err,
                    "{label}[{i}]: force err={err}, approx={af:?}, exact={ef:?}"
                );
            }
        }
    }

    #[test]
    fn test_compute_fit_kl_repulsion_extreme_span_decays_without_saturating() {
        // Points spread across span = 2000 (> MAX_ADAPTIVE_GRID_DIM = 512) with pairs sharing
        // coarse grid cells (separation ~2.0 inside hx ~ 4.7 cells) must not saturate at 0.5.
        let mut pts = Vec::new();
        for k in 0..10 {
            let base = -1000.0 + k as f64 * 200.0;
            pts.push([base, base * 0.5]);
            pts.push([base + 1.8, base * 0.5 + 1.2]);
        }
        let grid = FitGridSpec::from_points(&pts, 25, 3);
        assert_eq!(grid.nx, 512);
        assert!(grid.use_bilinear && grid.softening_sq > 0.0);

        let (rep, total_z) = compute_fit_kl_repulsion(&pts, 25, 3);
        assert!(total_z.is_finite() && total_z > 0.0);
        for (i, f) in rep.iter().enumerate() {
            let norm = (f[0] * f[0] + f[1] * f[1]).sqrt();
            assert!(norm.is_finite(), "non-finite force at {i}");
            assert!(
                norm < 0.25,
                "extreme-span force[{i}] saturated instead of decaying: norm={norm}"
            );
        }
    }
}
