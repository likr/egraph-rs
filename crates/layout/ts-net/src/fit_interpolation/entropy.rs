use super::{build_particle_stencil, FitGridSpec};
use crate::fft::{fft_2d, Complex};
use num_traits::Float;
use petgraph_drawing::DrawingValue;

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
