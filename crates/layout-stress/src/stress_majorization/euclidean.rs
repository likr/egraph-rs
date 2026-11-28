use egraph_core::{
    CoreError, CoreResult, EuclideanLayout, EuclideanStorageMut, EuclideanStorageRef, Scalar,
    SolverStatus,
};

use super::StressMajorizationState;

#[inline]
fn norm3_f64(dx: f64, dy: f64, dz: f64) -> f64 {
    let m = dx.abs().max(dy.abs()).max(dz.abs());
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    let sx = dx / m;
    let sy = dy / m;
    let sz = dz / m;
    m * (sx * sx + sy * sy + sz * sz).sqrt()
}

#[inline]
fn euclid_norm_diff_flat<S: Scalar>(coords: &[S], d: usize, i: usize, j: usize) -> f64 {
    let base_i = i * d;
    let base_j = j * d;
    let mut sum_sq = 0.0_f64;
    for a in 0..d {
        let diff = coords[base_i + a].to_f64() - coords[base_j + a].to_f64();
        sum_sq += diff * diff;
    }
    if sum_sq.is_finite() {
        return sum_sq.sqrt();
    }
    let mut max_abs = 0.0_f64;
    for a in 0..d {
        let abs_d = (coords[base_i + a].to_f64() - coords[base_j + a].to_f64()).abs();
        if abs_d > max_abs {
            max_abs = abs_d;
        }
    }
    if !max_abs.is_finite() || max_abs == 0.0 {
        return max_abs;
    }
    let mut scaled_sq = 0.0_f64;
    for a in 0..d {
        let r = (coords[base_i + a].to_f64() - coords[base_j + a].to_f64()) / max_abs;
        scaled_sq += r * r;
    }
    max_abs * scaled_sq.sqrt()
}

#[inline]
fn euclid_norm_diff_axis_major(sol: &[f64], n: usize, d: usize, i: usize, j: usize) -> f64 {
    let mut sum_sq = 0.0_f64;
    for a in 0..d {
        let off = a * n;
        let diff = sol[off + i] - sol[off + j];
        sum_sq += diff * diff;
    }
    if sum_sq.is_finite() {
        return sum_sq.sqrt();
    }
    let mut max_abs = 0.0_f64;
    for a in 0..d {
        let off = a * n;
        let abs_d = (sol[off + i] - sol[off + j]).abs();
        if abs_d > max_abs {
            max_abs = abs_d;
        }
    }
    if !max_abs.is_finite() || max_abs == 0.0 {
        return max_abs;
    }
    let mut scaled_sq = 0.0_f64;
    for a in 0..d {
        let off = a * n;
        let r = (sol[off + i] - sol[off + j]) / max_abs;
        scaled_sq += r * r;
    }
    max_abs * scaled_sq.sqrt()
}

pub(super) fn validate_euclidean<L: EuclideanLayout>(
    state: &StressMajorizationState,
    target: &L,
) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "StressMajorizationSolver::validate_target",
        ));
    }
    let dims = target.dims();
    if dims == 0 {
        return Err(CoreError::invalid_input("Layout dims must be >= 1"));
    }
    if let Some(prev) = state.locked_dim
        && prev != dims
    {
        return Err(CoreError::dimension_mismatch(
            prev,
            dims,
            "StressMajorizationSolver::validate_target::dims",
        ));
    }
    target.validate_finite()
}

pub(super) fn compute_stress_euclidean<L: EuclideanLayout>(
    state: &StressMajorizationState,
    layout: &L,
) -> CoreResult<f64> {
    let n = state.num_nodes;
    let total = match layout.as_euclidean_storage() {
        EuclideanStorageRef::Soa2d { x, y } => compute_stress_2d(state, n, x, y),
        EuclideanStorageRef::Soa3d { x, y, z } => compute_stress_3d(state, n, x, y, z),
        EuclideanStorageRef::FlatNd { coords, dims, .. } => {
            compute_stress_nd_slice(state, n, coords, dims)
        }
    };
    if !total.is_finite() {
        return Err(CoreError::non_finite(
            "StressMajorizationSolver::compute_stress",
        ));
    }
    Ok(total)
}

fn compute_stress_2d<S: Scalar>(
    state: &StressMajorizationState,
    n: usize,
    xs: &[S],
    ys: &[S],
) -> f64 {
    let mut total = 0.0_f64;
    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let row_off = i * n;
        for j in (i + 1)..n {
            let w = state.weights[row_off + j];
            if w <= 0.0 {
                continue;
            }
            let d_target = state.dists[row_off + j];
            let norm = (xi - xs[j].to_f64()).hypot(yi - ys[j].to_f64());
            let diff = norm - d_target;
            total += w * diff * diff;
        }
    }
    total
}

fn compute_stress_3d<S: Scalar>(
    state: &StressMajorizationState,
    n: usize,
    xs: &[S],
    ys: &[S],
    zs: &[S],
) -> f64 {
    let mut total = 0.0_f64;
    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let zi = zs[i].to_f64();
        let row_off = i * n;
        for j in (i + 1)..n {
            let w = state.weights[row_off + j];
            if w <= 0.0 {
                continue;
            }
            let d_target = state.dists[row_off + j];
            let norm = norm3_f64(
                xi - xs[j].to_f64(),
                yi - ys[j].to_f64(),
                zi - zs[j].to_f64(),
            );
            let diff = norm - d_target;
            total += w * diff * diff;
        }
    }
    total
}

fn compute_stress_nd_slice<S: Scalar>(
    state: &StressMajorizationState,
    n: usize,
    coords: &[S],
    d: usize,
) -> f64 {
    let mut total = 0.0_f64;
    for i in 0..n {
        let row_off = i * n;
        for j in (i + 1)..n {
            let w = state.weights[row_off + j];
            if w <= 0.0 {
                continue;
            }
            let d_target = state.dists[row_off + j];
            let norm = euclid_norm_diff_flat(coords, d, i, j);
            let diff = norm - d_target;
            total += w * diff * diff;
        }
    }
    total
}

pub(super) fn step_euclidean<L: EuclideanLayout>(
    state: &mut StressMajorizationState,
    layout: &mut L,
) -> CoreResult<()> {
    let d = layout.dims();
    if state.locked_dim.is_none() {
        state.locked_dim = Some(d);
    }
    let n = state.num_nodes;
    if n <= 1 {
        state.prev_stress = Some(0.0);
        state.status = SolverStatus::Converged;
        return Ok(());
    }

    let s_old = match state.prev_stress {
        Some(s) => s,
        None => compute_stress_euclidean(state, layout)?,
    };

    let pcg_max = state.config.pcg_max_iterations.unwrap_or((2 * n).max(16));
    let min_norm = state.config.min_norm;

    match layout.as_euclidean_storage() {
        EuclideanStorageRef::Soa2d { x, y } => {
            if state.rhs_x.len() != n {
                state.rhs_x.resize(n, 0.0);
                state.rhs_y.resize(n, 0.0);
                state.sol_x.resize(n, 0.0);
                state.sol_y.resize(n, 0.0);
            }
            state.rhs_x.fill(0.0);
            state.rhs_y.fill(0.0);
            for i in 0..n {
                state.sol_x[i] = x[i].to_f64();
                state.sol_y[i] = y[i].to_f64();
            }
            for i in 0..n {
                let xi = state.sol_x[i];
                let yi = state.sol_y[i];
                let row_off = i * n;
                for j in (i + 1)..n {
                    let w = state.weights[row_off + j];
                    if w <= 0.0 {
                        continue;
                    }
                    let d_target = state.dists[row_off + j];
                    let dx = xi - state.sol_x[j];
                    let dy = yi - state.sol_y[j];
                    let norm = dx.hypot(dy);
                    if norm >= min_norm && norm.is_finite() {
                        let f = w * d_target / norm;
                        let fx = f * dx;
                        let fy = f * dy;
                        state.rhs_x[i] += fx;
                        state.rhs_x[j] -= fx;
                        state.rhs_y[i] += fy;
                        state.rhs_y[j] -= fy;
                    }
                }
            }
            state.pcg.solve_dense_symmetric(
                &state.l_w,
                &state.rhs_x,
                &mut state.sol_x,
                state.config.pcg_tolerance,
                pcg_max,
            )?;
            state.pcg.solve_dense_symmetric(
                &state.l_w,
                &state.rhs_y,
                &mut state.sol_y,
                state.config.pcg_tolerance,
                pcg_max,
            )?;
            if let EuclideanStorageMut::Soa2d { x: xm, y: ym } = layout.as_euclidean_storage_mut() {
                for i in 0..n {
                    let nx = L::Scalar::from_f64(state.sol_x[i]);
                    let ny = L::Scalar::from_f64(state.sol_y[i]);
                    if nx.is_finite() {
                        xm[i] = nx;
                    }
                    if ny.is_finite() {
                        ym[i] = ny;
                    }
                }
            }
        }
        EuclideanStorageRef::Soa3d { x, y, z } => {
            let needed = n
                .checked_mul(3)
                .ok_or_else(|| CoreError::invalid_input("Layout3d buffer size overflow"))?;
            if state.nd_rhs.len() != needed {
                state.nd_rhs.resize(needed, 0.0);
                state.nd_sol.resize(needed, 0.0);
            }
            state.nd_rhs.fill(0.0);
            for i in 0..n {
                state.nd_sol[i] = x[i].to_f64();
                state.nd_sol[n + i] = y[i].to_f64();
                state.nd_sol[2 * n + i] = z[i].to_f64();
            }
            for i in 0..n {
                let row_off = i * n;
                for j in (i + 1)..n {
                    let w = state.weights[row_off + j];
                    if w <= 0.0 {
                        continue;
                    }
                    let d_target = state.dists[row_off + j];
                    let dx = state.nd_sol[i] - state.nd_sol[j];
                    let dy = state.nd_sol[n + i] - state.nd_sol[n + j];
                    let dz = state.nd_sol[2 * n + i] - state.nd_sol[2 * n + j];
                    let norm = norm3_f64(dx, dy, dz);
                    if norm >= min_norm && norm.is_finite() {
                        let f = w * d_target / norm;
                        let fx = f * dx;
                        let fy = f * dy;
                        let fz = f * dz;
                        state.nd_rhs[i] += fx;
                        state.nd_rhs[j] -= fx;
                        state.nd_rhs[n + i] += fy;
                        state.nd_rhs[n + j] -= fy;
                        state.nd_rhs[2 * n + i] += fz;
                        state.nd_rhs[2 * n + j] -= fz;
                    }
                }
            }
            for a in 0..3 {
                let off = a * n;
                state.pcg.solve_dense_symmetric(
                    &state.l_w,
                    &state.nd_rhs[off..off + n],
                    &mut state.nd_sol[off..off + n],
                    state.config.pcg_tolerance,
                    pcg_max,
                )?;
            }
            if let EuclideanStorageMut::Soa3d {
                x: xm,
                y: ym,
                z: zm,
            } = layout.as_euclidean_storage_mut()
            {
                for i in 0..n {
                    let nx = L::Scalar::from_f64(state.nd_sol[i]);
                    let ny = L::Scalar::from_f64(state.nd_sol[n + i]);
                    let nz = L::Scalar::from_f64(state.nd_sol[2 * n + i]);
                    if nx.is_finite() {
                        xm[i] = nx;
                    }
                    if ny.is_finite() {
                        ym[i] = ny;
                    }
                    if nz.is_finite() {
                        zm[i] = nz;
                    }
                }
            }
        }
        EuclideanStorageRef::FlatNd { coords, dims, .. } => {
            let needed = n
                .checked_mul(dims)
                .ok_or_else(|| CoreError::invalid_input("LayoutNd buffer size overflow"))?;
            if state.nd_rhs.len() != needed {
                state.nd_rhs.resize(needed, 0.0);
                state.nd_sol.resize(needed, 0.0);
            }
            state.nd_rhs.fill(0.0);
            for i in 0..n {
                let base = i * dims;
                for a in 0..dims {
                    state.nd_sol[a * n + i] = coords[base + a].to_f64();
                }
            }
            for i in 0..n {
                let row_off = i * n;
                for j in (i + 1)..n {
                    let w = state.weights[row_off + j];
                    if w <= 0.0 {
                        continue;
                    }
                    let d_target = state.dists[row_off + j];
                    let norm = euclid_norm_diff_axis_major(&state.nd_sol, n, dims, i, j);
                    if norm >= min_norm && norm.is_finite() {
                        let f = w * d_target / norm;
                        for a in 0..dims {
                            let off = a * n;
                            let fa = f * (state.nd_sol[off + i] - state.nd_sol[off + j]);
                            state.nd_rhs[off + i] += fa;
                            state.nd_rhs[off + j] -= fa;
                        }
                    }
                }
            }
            for a in 0..dims {
                let off = a * n;
                state.pcg.solve_dense_symmetric(
                    &state.l_w,
                    &state.nd_rhs[off..off + n],
                    &mut state.nd_sol[off..off + n],
                    state.config.pcg_tolerance,
                    pcg_max,
                )?;
            }
            if let EuclideanStorageMut::FlatNd { coords: cm, .. } =
                layout.as_euclidean_storage_mut()
            {
                for i in 0..n {
                    let base = i * dims;
                    for a in 0..dims {
                        let nv = L::Scalar::from_f64(state.nd_sol[a * n + i]);
                        if nv.is_finite() {
                            cm[base + a] = nv;
                        }
                    }
                }
            }
        }
    }

    let new_stress = compute_stress_euclidean(state, layout)?;
    state.prev_stress = Some(new_stress);
    state.iter += 1;

    let converged = if s_old > 1e-30 {
        (s_old - new_stress).abs() / s_old < state.config.epsilon
    } else {
        new_stress <= 1e-30
    };

    state.status = if converged {
        SolverStatus::Converged
    } else if state.iter >= state.config.max_iterations {
        SolverStatus::Completed
    } else {
        SolverStatus::Running
    };
    Ok(())
}
