use crate::fft::{convolve_2d, fft_2d, Complex};
use num_traits::Float;
use petgraph_drawing::DrawingValue;

/// Evaluates cubic Catmull-Rom / Lagrange interpolation weights for fraction `t` in `[0, 1)`.
#[inline]
fn cubic_weights<S>(t: S) -> [S; 4]
where
    S: DrawingValue + Float + Default,
{
    let half = S::from_f32(0.5).unwrap();
    let one = S::one();
    let two = S::from_f32(2.0).unwrap();
    let three = S::from_f32(3.0).unwrap();
    let five = S::from_f32(5.0).unwrap();

    let t2 = t * t;
    let t3 = t2 * t;

    [
        -half * t + t2 - half * t3,
        one - (five / two) * t2 + (three / two) * t3,
        half * t + two * t2 - (three / two) * t3,
        -half * t2 + half * t3,
    ]
}

/// Evaluates low-dimensional Student-t KL divergence repulsion forces in O(N) using 2D FFT interpolation.
///
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

    // Step 1: Compute bounding box with margin
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

    let span_x = (max_x - min_x).max(S::from_f32(1e-4).unwrap());
    let span_y = (max_y - min_y).max(S::from_f32(1e-4).unwrap());
    let margin_x = span_x * S::from_f32(0.1).unwrap();
    let margin_y = span_y * S::from_f32(0.1).unwrap();

    let bbox_min_x = min_x - margin_x;
    let bbox_max_x = max_x + margin_x;
    let bbox_min_y = min_y - margin_y;
    let bbox_max_y = max_y + margin_y;

    let total_span_x = bbox_max_x - bbox_min_x;
    let total_span_y = bbox_max_y - bbox_min_y;

    let target_grid_x = (intervals * interpolation_points).max(16);
    let target_grid_y = (intervals * interpolation_points).max(16);

    let nx = target_grid_x.next_power_of_two();
    let ny = target_grid_y.next_power_of_two();

    let hx = total_span_x / S::from_usize(nx).unwrap();
    let hy = total_span_y / S::from_usize(ny).unwrap();

    let pad_nx = nx * 2;
    let pad_ny = ny * 2;

    // Step 2: Spread charges w^(1) = 1, w^(x) = x_j, w^(y) = y_j onto (pad_nx x pad_ny) padded grid
    let mut grid_1 = vec![Complex::zero(); pad_nx * pad_ny];
    let mut grid_x = vec![Complex::zero(); pad_nx * pad_ny];
    let mut grid_y = vec![Complex::zero(); pad_nx * pad_ny];

    for p in points {
        let px = p[0];
        let py = p[1];

        let u_real = (px - bbox_min_x) / hx;
        let v_real = (py - bbox_min_y) / hy;

        let u0 = (u_real.floor().to_isize().unwrap_or(0)).max(1) as usize;
        let v0 = (v_real.floor().to_isize().unwrap_or(0)).max(1) as usize;

        let tu = u_real - u_real.floor();
        let tv = v_real - v_real.floor();

        let wu = cubic_weights(tu);
        let wv = cubic_weights(tv);

        for (k_idx, &w_k) in wu.iter().enumerate() {
            let u = (u0 + k_idx - 1).min(nx - 1);
            for (l_idx, &w_l) in wv.iter().enumerate() {
                let v = (v0 + l_idx - 1).min(ny - 1);
                let w = w_k * w_l;
                let idx = v * pad_nx + u;
                grid_1[idx].re += w;
                grid_x[idx].re += w * px;
                grid_y[idx].re += w * py;
            }
        }
    }

    // Step 3: Compute Cauchy kernel on padded 2D grid and its FFT
    let mut kernel_grid = vec![Complex::zero(); pad_nx * pad_ny];
    for v in 0..pad_ny {
        let dy = if v < ny {
            S::from_usize(v).unwrap() * hy
        } else {
            (S::from_usize(v).unwrap() - S::from_usize(pad_ny).unwrap()) * hy
        };

        for u in 0..pad_nx {
            let dx = if u < nx {
                S::from_usize(u).unwrap() * hx
            } else {
                (S::from_usize(u).unwrap() - S::from_usize(pad_nx).unwrap()) * hx
            };

            let dist_sq = dx * dx + dy * dy;
            let val = S::one() / (S::one() + dist_sq);
            kernel_grid[v * pad_nx + u] = Complex::new(val, S::zero());
        }
    }

    fft_2d(&mut kernel_grid, pad_nx, pad_ny, false);

    // Step 4: 2D FFT convolution for the 3 charge fields
    let pot_1 = convolve_2d(&grid_1, &kernel_grid, pad_nx, pad_ny);
    let pot_x = convolve_2d(&grid_x, &kernel_grid, pad_nx, pad_ny);
    let pot_y = convolve_2d(&grid_y, &kernel_grid, pad_nx, pad_ny);

    // Step 5: Gather potentials at particle positions
    let mut rep_forces = vec![[S::zero(), S::zero()]; n];
    let mut total_z = S::zero();

    for (i, p) in points.iter().enumerate() {
        let px = p[0];
        let py = p[1];

        let u_real = (px - bbox_min_x) / hx;
        let v_real = (py - bbox_min_y) / hy;

        let u0 = (u_real.floor().to_isize().unwrap_or(0)).max(1) as usize;
        let v0 = (v_real.floor().to_isize().unwrap_or(0)).max(1) as usize;

        let tu = u_real - u_real.floor();
        let tv = v_real - v_real.floor();

        let wu = cubic_weights(tu);
        let wv = cubic_weights(tv);

        let mut phi_1 = S::zero();
        let mut phi_x = S::zero();
        let mut phi_y = S::zero();

        for (k_idx, &w_k) in wu.iter().enumerate() {
            let u = (u0 + k_idx - 1).min(nx - 1);
            for (l_idx, &w_l) in wv.iter().enumerate() {
                let v = (v0 + l_idx - 1).min(ny - 1);
                let w = w_k * w_l;
                let idx = v * pad_nx + u;
                phi_1 += w * pot_1[idx].re;
                phi_x += w * pot_x[idx].re;
                phi_y += w * pot_y[idx].re;
            }
        }

        let f_rep_x = px * phi_1 - phi_x;
        let f_rep_y = py * phi_1 - phi_y;
        let z_i = (phi_1 - S::one()).max(S::zero());

        rep_forces[i] = [f_rep_x, f_rep_y];
        total_z += z_i;
    }

    total_z = total_z.max(S::from_f32(1e-12).unwrap());

    // Normalize repulsion forces by global Z
    for force in &mut rep_forces {
        force[0] /= total_z;
        force[1] /= total_z;
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

    // Step 1: Compute bounding box with margin
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

    let span_x = (max_x - min_x).max(S::from_f32(1e-4).unwrap());
    let span_y = (max_y - min_y).max(S::from_f32(1e-4).unwrap());
    let margin_x = span_x * S::from_f32(0.1).unwrap();
    let margin_y = span_y * S::from_f32(0.1).unwrap();

    let bbox_min_x = min_x - margin_x;
    let bbox_max_x = max_x + margin_x;
    let bbox_min_y = min_y - margin_y;
    let bbox_max_y = max_y + margin_y;

    let total_span_x = bbox_max_x - bbox_min_x;
    let total_span_y = bbox_max_y - bbox_min_y;

    let target_grid_x = (intervals * interpolation_points).max(16);
    let target_grid_y = (intervals * interpolation_points).max(16);

    let nx = target_grid_x.next_power_of_two();
    let ny = target_grid_y.next_power_of_two();

    let hx = total_span_x / S::from_usize(nx).unwrap();
    let hy = total_span_y / S::from_usize(ny).unwrap();

    let pad_nx = nx * 2;
    let pad_ny = ny * 2;

    // Step 2: Spread charges onto padded grid
    let mut grid_1 = vec![Complex::zero(); pad_nx * pad_ny];
    let mut grid_x = vec![Complex::zero(); pad_nx * pad_ny];
    let mut grid_y = vec![Complex::zero(); pad_nx * pad_ny];

    for p in points {
        let px = p[0];
        let py = p[1];

        let u_real = (px - bbox_min_x) / hx;
        let v_real = (py - bbox_min_y) / hy;

        let u0 = (u_real.floor().to_isize().unwrap_or(0)).max(1) as usize;
        let v0 = (v_real.floor().to_isize().unwrap_or(0)).max(1) as usize;

        let tu = u_real - u_real.floor();
        let tv = v_real - v_real.floor();

        let wu = cubic_weights(tu);
        let wv = cubic_weights(tv);

        for (k_idx, &w_k) in wu.iter().enumerate() {
            let u = (u0 + k_idx - 1).min(nx - 1);
            for (l_idx, &w_l) in wv.iter().enumerate() {
                let v = (v0 + l_idx - 1).min(ny - 1);
                let w = w_k * w_l;
                let idx = v * pad_nx + u;
                grid_1[idx].re += w;
                grid_x[idx].re += w * px;
                grid_y[idx].re += w * py;
            }
        }
    }

    // Step 3: Compute Entropy kernel K_e = 1 / (epsilon_r + dx^2 + dy^2) on padded 2D grid and its FFT
    let mut kernel_grid = vec![Complex::zero(); pad_nx * pad_ny];
    for v in 0..pad_ny {
        let dy = if v < ny {
            S::from_usize(v).unwrap() * hy
        } else {
            (S::from_usize(v).unwrap() - S::from_usize(pad_ny).unwrap()) * hy
        };

        for u in 0..pad_nx {
            let dx = if u < nx {
                S::from_usize(u).unwrap() * hx
            } else {
                (S::from_usize(u).unwrap() - S::from_usize(pad_nx).unwrap()) * hx
            };

            let dist_sq = dx * dx + dy * dy;
            let val = S::one() / (epsilon_r + dist_sq);
            kernel_grid[v * pad_nx + u] = Complex::new(val, S::zero());
        }
    }

    fft_2d(&mut kernel_grid, pad_nx, pad_ny, false);

    // Step 4: 2D FFT convolution
    let pot_1 = convolve_2d(&grid_1, &kernel_grid, pad_nx, pad_ny);
    let pot_x = convolve_2d(&grid_x, &kernel_grid, pad_nx, pad_ny);
    let pot_y = convolve_2d(&grid_y, &kernel_grid, pad_nx, pad_ny);

    // Step 5: Gather potentials h_1, h_2, h_3 and compute entropy gradient
    let mut ent_grad = vec![[S::zero(), S::zero()]; n];

    for (i, p) in points.iter().enumerate() {
        let px = p[0];
        let py = p[1];

        let u_real = (px - bbox_min_x) / hx;
        let v_real = (py - bbox_min_y) / hy;

        let u0 = (u_real.floor().to_isize().unwrap_or(0)).max(1) as usize;
        let v0 = (v_real.floor().to_isize().unwrap_or(0)).max(1) as usize;

        let tu = u_real - u_real.floor();
        let tv = v_real - v_real.floor();

        let wu = cubic_weights(tu);
        let wv = cubic_weights(tv);

        let mut h_1 = S::zero();
        let mut h_2 = S::zero();
        let mut h_3 = S::zero();

        for (k_idx, &w_k) in wu.iter().enumerate() {
            let u = (u0 + k_idx - 1).min(nx - 1);
            for (l_idx, &w_l) in wv.iter().enumerate() {
                let v = (v0 + l_idx - 1).min(ny - 1);
                let w = w_k * w_l;
                let idx = v * pad_nx + u;
                h_1 += w * pot_1[idx].re;
                h_2 += w * pot_x[idx].re;
                h_3 += w * pot_y[idx].re;
            }
        }

        let f_ent_x = h_2 - px * h_1;
        let f_ent_y = h_3 - py * h_1;

        ent_grad[i] = [f_ent_x, f_ent_y];
    }

    ent_grad
}
