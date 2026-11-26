use super::{build_particle_stencil, FitGridSpec};
use crate::fft::{fft_2d, Complex};
use num_traits::Float;
use petgraph_drawing::DrawingValue;

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
