use egraph_core::{CoreError, CoreResult, RiemannianLayout, SolverStatus};

use super::KamadaKawaiConfig;

const HESSIAN_DET_TOL: f64 = 1e-12;
const MIN_SEPARATION: f64 = 1e-12;
const MAX_NEWTON_STEP: f64 = 1e6;

#[derive(Debug, Clone)]
pub(super) struct KamadaKawaiState {
    pub(super) config: KamadaKawaiConfig,
    pub(super) num_nodes: usize,
    pub(super) max_iterations: usize,
    pub(super) l: Vec<f64>,
    pub(super) k: Vec<f64>,
    pub(super) log_buf: Vec<f64>,
    pub(super) grad: Vec<f64>,
    pub(super) hessian: Vec<f64>,
    pub(super) delta: Vec<f64>,
    pub(super) locked_dim: Option<usize>,
    pub(super) iter: usize,
    pub(super) status: SolverStatus,
}

#[inline]
fn euclid_norm(vec: &[f64]) -> f64 {
    let mut sum_sq = 0.0_f64;
    for &v in vec {
        sum_sq += v * v;
    }
    if sum_sq.is_finite() {
        return sum_sq.sqrt();
    }
    let mut max_abs = 0.0_f64;
    for &v in vec {
        let av = v.abs();
        if av > max_abs {
            max_abs = av;
        }
    }
    if !max_abs.is_finite() || max_abs == 0.0 {
        return max_abs;
    }
    let mut scaled_sq = 0.0_f64;
    for &v in vec {
        let r = v / max_abs;
        scaled_sq += r * r;
    }
    max_abs * scaled_sq.sqrt()
}

/// Solves `h * out = rhs` (`d x d` symmetric dense row-major) via Cholesky factorization
/// $H = L L^T$. Returns `false` if any pivot is `<= diag_tol` or non-finite, ensuring
/// indefinite or negative-definite Hessians fall back cleanly to steepest descent.
fn solve_spd_cholesky(
    h: &mut [f64],
    rhs: &[f64],
    out: &mut [f64],
    d: usize,
    diag_tol: f64,
) -> bool {
    for j in 0..d {
        let mut diag_sum = h[j * d + j];
        for k in 0..j {
            let l_jk = h[j * d + k];
            diag_sum -= l_jk * l_jk;
        }
        if !diag_sum.is_finite() || diag_sum <= diag_tol {
            return false;
        }
        let l_jj = diag_sum.sqrt();
        h[j * d + j] = l_jj;
        let inv_diag = 1.0 / l_jj;
        for i in (j + 1)..d {
            let mut sum = h[i * d + j];
            for k in 0..j {
                sum -= h[i * d + k] * h[j * d + k];
            }
            let l_ij = sum * inv_diag;
            if !l_ij.is_finite() {
                return false;
            }
            h[i * d + j] = l_ij;
        }
    }

    // Forward substitution: L y = rhs.
    for i in 0..d {
        let mut sum = rhs[i];
        for k in 0..i {
            sum -= h[i * d + k] * out[k];
        }
        let val = sum / h[i * d + i];
        if !val.is_finite() {
            return false;
        }
        out[i] = val;
    }

    // Backward substitution: L^T out = y.
    for i in (0..d).rev() {
        let mut sum = out[i];
        for k in (i + 1)..d {
            sum -= h[k * d + i] * out[k];
        }
        let val = sum / h[i * d + i];
        if !val.is_finite() {
            return false;
        }
        out[i] = val;
    }

    true
}

pub(super) fn validate_riemannian<L: RiemannianLayout>(
    state: &KamadaKawaiState,
    target: &L,
) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "KamadaKawaiSolver::validate_target",
        ));
    }
    let dim = target.tangent_dim();
    if dim == 0 && state.num_nodes > 0 {
        return Err(CoreError::invalid_input("tangent_dim must be >= 1"));
    }
    if let Some(prev) = state.locked_dim
        && prev != dim
    {
        return Err(CoreError::dimension_mismatch(
            prev,
            dim,
            "KamadaKawaiSolver::validate_target::dims",
        ));
    }
    target.validate_finite()
}

fn ensure_scratch(state: &mut KamadaKawaiState, d: usize) -> CoreResult<()> {
    let needed_hessian = d
        .checked_mul(d)
        .ok_or_else(|| CoreError::invalid_input("KamadaKawai Hessian size overflow"))?;
    if state.log_buf.len() != d {
        state.log_buf.resize(d, 0.0);
        state.grad.resize(d, 0.0);
        state.delta.resize(d, 0.0);
    }
    if state.hessian.len() != needed_hessian {
        state.hessian.resize(needed_hessian, 0.0);
    }
    Ok(())
}

fn select_node<L: RiemannianLayout>(
    state: &mut KamadaKawaiState,
    layout: &L,
    d: usize,
    min_sep: f64,
) -> CoreResult<Option<usize>> {
    let n = state.num_nodes;
    let eps_sq = state.config.eps * state.config.eps;
    let mut best_m = 0usize;
    let mut best_delta2 = 0.0_f64;

    for m in 0..n {
        state.grad.fill(0.0);
        let row = m * n;
        for i in 0..n {
            if i == m {
                continue;
            }
            let k_mi = state.k[row + i];
            if k_mi <= 0.0 {
                continue;
            }
            layout.log_map_into(m, i, &mut state.log_buf)?;
            let dist = euclid_norm(&state.log_buf[..d]);
            if dist > min_sep && dist.is_finite() {
                let f = k_mi * (1.0 - state.l[row + i] / dist);
                for a in 0..d {
                    state.grad[a] += f * (-state.log_buf[a]);
                }
            }
        }
        let delta2: f64 = state.grad.iter().map(|&g| g * g).sum();
        if delta2.is_finite() && delta2 > best_delta2 {
            best_delta2 = delta2;
            best_m = m;
        }
    }

    if best_delta2 < eps_sq {
        Ok(None)
    } else {
        Ok(Some(best_m))
    }
}

fn steepest_descent_fallback<L: RiemannianLayout>(
    state: &mut KamadaKawaiState,
    layout: &mut L,
    m: usize,
    d: usize,
    delta2: f64,
) -> CoreResult<()> {
    let g_norm = delta2.sqrt();
    if g_norm > 0.0 && g_norm.is_finite() {
        let step_len = (0.1 / g_norm).min(0.25);
        let mut all_finite = true;
        for a in 0..d {
            let cand = -step_len * state.grad[a];
            if !cand.is_finite() {
                all_finite = false;
                break;
            }
            state.delta[a] = cand;
        }
        if all_finite {
            layout.exp_map(m, &state.delta[..d])?;
        }
    }
    Ok(())
}

pub(super) fn step_unified<L: RiemannianLayout>(
    state: &mut KamadaKawaiState,
    layout: &mut L,
) -> CoreResult<()> {
    let d = layout.tangent_dim();
    if d > 0 && state.locked_dim.is_none() {
        state.locked_dim = Some(d);
    }

    let n = state.num_nodes;
    if n <= 1 {
        state.status = SolverStatus::Converged;
        return Ok(());
    }

    ensure_scratch(state, d)?;
    let min_sep = MIN_SEPARATION;
    let max_newton = MAX_NEWTON_STEP;

    let Some(m) = select_node(state, layout, d, min_sep)? else {
        state.status = SolverStatus::Converged;
        return Ok(());
    };

    let eps_sq = state.config.eps * state.config.eps;
    let row = m * n;

    for _ in 0..state.config.max_inner_iters {
        state.grad.fill(0.0);
        if d == 2 {
            let mut hxx = 0.0_f64;
            let mut hyy = 0.0_f64;
            let mut hxy = 0.0_f64;
            let mut dedx = 0.0_f64;
            let mut dedy = 0.0_f64;
            let mut max_dist_seen = 0.0_f64;

            for i in 0..n {
                if i == m {
                    continue;
                }
                let k_mi = state.k[row + i];
                if k_mi <= 0.0 {
                    continue;
                }
                let l_mi = state.l[row + i];
                layout.log_map_into(m, i, &mut state.log_buf)?;
                let dx = -state.log_buf[0];
                let dy = -state.log_buf[1];
                let dist = dx.hypot(dy);
                if dist <= min_sep || !dist.is_finite() {
                    continue;
                }
                if dist > max_dist_seen {
                    max_dist_seen = dist;
                }
                let d3 = dist * dist * dist;
                if d3 > 0.0 && d3.is_finite() {
                    hxx += k_mi * (1.0 - l_mi * dy * dy / d3);
                    hyy += k_mi * (1.0 - l_mi * dx * dx / d3);
                    hxy += k_mi * l_mi * dx * dy / d3;
                }
                let f = k_mi * (1.0 - l_mi / dist);
                dedx += f * dx;
                dedy += f * dy;
            }

            state.grad[0] = dedx;
            state.grad[1] = dedy;
            let delta2 = dedx * dedx + dedy * dedy;
            if !delta2.is_finite() || delta2 < eps_sq {
                break;
            }

            let det = hxx * hyy - hxy * hxy;
            if !det.is_finite() || det <= HESSIAN_DET_TOL || hxx <= 0.0 || hyy <= 0.0 {
                steepest_descent_fallback(state, layout, m, 2, delta2)?;
                break;
            }

            let mut delta_x = (hyy * dedx - hxy * dedy) / det;
            let mut delta_y = (hxx * dedy - hxy * dedx) / det;
            if !delta_x.is_finite() || !delta_y.is_finite() {
                break;
            }
            let step_cap = if max_dist_seen > 0.0 {
                max_newton.min((0.5 * max_dist_seen).max(0.5))
            } else {
                max_newton
            };
            let step_norm = delta_x.hypot(delta_y);
            if step_norm > step_cap {
                let s = step_cap / step_norm;
                delta_x *= s;
                delta_y *= s;
            }
            state.delta[0] = -delta_x;
            state.delta[1] = -delta_y;
            layout.exp_map(m, &state.delta[..2])?;

            if step_norm <= state.config.eps * 1e-2 {
                break;
            }
        } else {
            state.hessian.fill(0.0);
            for i in 0..n {
                if i == m {
                    continue;
                }
                let k_mi = state.k[row + i];
                if k_mi <= 0.0 {
                    continue;
                }
                let l_mi = state.l[row + i];
                layout.log_map_into(m, i, &mut state.log_buf)?;
                let dist = euclid_norm(&state.log_buf[..d]);
                if dist <= min_sep || !dist.is_finite() {
                    continue;
                }
                let dist2 = dist * dist;
                let d3 = dist2 * dist;
                if d3 > 0.0 && d3.is_finite() {
                    let inv_d3 = l_mi / d3;
                    for a in 0..d {
                        let da = -state.log_buf[a];
                        let h_aa = k_mi * (1.0 - inv_d3 * (dist2 - da * da));
                        state.hessian[a * d + a] += h_aa;
                        for b in (a + 1)..d {
                            let db = -state.log_buf[b];
                            let h_ab = k_mi * inv_d3 * da * db;
                            state.hessian[a * d + b] += h_ab;
                            state.hessian[b * d + a] += h_ab;
                        }
                    }
                }
                let f = k_mi * (1.0 - l_mi / dist);
                for a in 0..d {
                    state.grad[a] += f * (-state.log_buf[a]);
                }
            }

            let delta2: f64 = state.grad.iter().map(|&g| g * g).sum();
            if !delta2.is_finite() || delta2 < eps_sq {
                break;
            }

            if !solve_spd_cholesky(
                &mut state.hessian,
                &state.grad,
                &mut state.delta,
                d,
                HESSIAN_DET_TOL,
            ) {
                steepest_descent_fallback(state, layout, m, d, delta2)?;
                break;
            }

            if !state.delta[..d].iter().all(|v| v.is_finite()) {
                break;
            }
            let step_norm = euclid_norm(&state.delta[..d]);
            if !step_norm.is_finite() {
                break;
            }
            if step_norm > max_newton {
                let s = max_newton / step_norm;
                for v in &mut state.delta[..d] {
                    *v *= s;
                }
            }
            for v in &mut state.delta[..d] {
                *v = -*v;
            }
            layout.exp_map(m, &state.delta[..d])?;

            if step_norm <= state.config.eps * 1e-2 {
                break;
            }
        }
    }

    state.iter += 1;
    state.status = if state.iter >= state.max_iterations {
        SolverStatus::Completed
    } else {
        SolverStatus::Running
    };
    Ok(())
}
