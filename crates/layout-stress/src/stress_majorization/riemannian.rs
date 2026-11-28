use std::any::Any;

use egraph_core::{
    CoreError, CoreResult, ManifoldLayout2d, RiemannianLayout, Scalar, SolverStatus,
};

use crate::manifold_utils::apply_clamped_manifold_step;

use super::StressMajorizationState;

const BACKTRACK_ALPHAS: [f64; 6] = [0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125];

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

pub(super) fn validate_riemannian<L: RiemannianLayout>(
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
            "StressMajorizationSolver::validate_target::dims",
        ));
    }
    target.validate_finite()
}

pub(super) fn validate_manifold<L: RiemannianLayout + 'static>(
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
    let layout = (target as &dyn Any)
        .downcast_ref::<ManifoldLayout2d<L::Scalar>>()
        .ok_or_else(|| {
            CoreError::numerical_breakdown("StressMajorizationSolver target type mismatch")
        })?;
    if let Some(prev) = state.locked_manifold
        && prev != layout.kind()
    {
        return Err(CoreError::invalid_input(
            "StressMajorizationSolver::validate_target: cannot switch ManifoldKind on stateful solver",
        ));
    }
    layout.validate_finite()
}

pub(super) fn compute_stress_riemannian<L: RiemannianLayout>(
    state: &StressMajorizationState,
    layout: &L,
) -> CoreResult<f64> {
    let n = state.num_nodes;
    let mut total = 0.0_f64;
    for i in 0..n {
        let row_off = i * n;
        for j in (i + 1)..n {
            let w = state.weights[row_off + j];
            if w <= 0.0 {
                continue;
            }
            let d_target = state.dists[row_off + j];
            let norm = layout.distance(i, j);
            if !norm.is_finite() {
                return Err(CoreError::non_finite(
                    "StressMajorizationSolver::compute_stress",
                ));
            }
            let diff = norm - d_target;
            total += w * diff * diff;
        }
    }
    if !total.is_finite() {
        return Err(CoreError::non_finite(
            "StressMajorizationSolver::compute_stress",
        ));
    }
    Ok(total)
}

pub(super) fn step_riemannian<L: RiemannianLayout + Clone + 'static>(
    state: &mut StressMajorizationState,
    layout: &mut L,
) -> CoreResult<()> {
    let d = layout.tangent_dim();
    if d > 0 && state.locked_dim.is_none() {
        state.locked_dim = Some(d);
    }

    let scale = if state.is_manifold {
        let m_layout = (layout as &dyn Any)
            .downcast_ref::<ManifoldLayout2d<L::Scalar>>()
            .ok_or_else(|| {
                CoreError::numerical_breakdown("StressMajorizationSolver target type mismatch")
            })?;
        if state.locked_manifold.is_none() {
            state.locked_manifold = Some(m_layout.kind());
        }
        m_layout.scale()
    } else {
        1.0
    };

    let n = state.num_nodes;
    if n <= 1 {
        state.prev_stress = Some(0.0);
        state.status = SolverStatus::Converged;
        return Ok(());
    }

    let s_old = match state.prev_stress {
        Some(s) => s,
        None => compute_stress_riemannian(state, layout)?,
    };

    let grad_len = n
        .checked_mul(d)
        .ok_or_else(|| CoreError::invalid_input("Riemannian gradient buffer overflow"))?;
    if state.nd_rhs.len() != grad_len {
        state.nd_rhs.resize(grad_len, 0.0);
    }
    if state.log_buf.len() != d {
        state.log_buf.resize(d, 0.0);
        state.step_buf.resize(d, 0.0);
    }

    let eff_min_norm = state.config.min_norm * scale.min(1.0);
    for i in 0..n {
        let row_off = i * n;
        let grad_off = i * d;
        state.nd_rhs[grad_off..grad_off + d].fill(0.0);
        let mut diag_w = 0.0_f64;
        for j in 0..n {
            if j == i {
                continue;
            }
            let w = state.weights[row_off + j];
            if w <= 0.0 {
                continue;
            }
            diag_w += w;
            layout.log_map_into(i, j, &mut state.log_buf)?;
            let norm = euclid_norm(&state.log_buf[..d]);
            if norm >= eff_min_norm && norm.is_finite() {
                let d_target = state.dists[row_off + j];
                let coeff = w * (1.0 - d_target / norm);
                if coeff.is_finite() {
                    for a in 0..d {
                        state.nd_rhs[grad_off + a] += coeff * state.log_buf[a];
                    }
                }
            }
        }
        if diag_w > 0.0
            && state.nd_rhs[grad_off..grad_off + d]
                .iter()
                .all(|v| v.is_finite())
        {
            for a in 0..d {
                state.nd_rhs[grad_off + a] /= diag_w;
            }
        } else {
            state.nd_rhs[grad_off..grad_off + d].fill(0.0);
        }
    }

    // Backtrack over damped Jacobi SMACOF steps so curved manifolds never increase stress.
    let mut new_stress = s_old;
    if state.is_manifold {
        if state.sol_x.len() != n {
            state.sol_x.resize(n, 0.0);
            state.sol_y.resize(n, 0.0);
        }
        let m_layout = (layout as &mut dyn Any)
            .downcast_mut::<ManifoldLayout2d<L::Scalar>>()
            .ok_or_else(|| {
                CoreError::numerical_breakdown("StressMajorizationSolver target type mismatch")
            })?;
        for i in 0..n {
            state.sol_x[i] = m_layout.x()[i].to_f64();
            state.sol_y[i] = m_layout.y()[i].to_f64();
        }
        for &alpha in &BACKTRACK_ALPHAS {
            for i in 0..n {
                let vx = alpha * state.nd_rhs[i * 2];
                let vy = alpha * state.nd_rhs[i * 2 + 1];
                if vx != 0.0 || vy != 0.0 {
                    apply_clamped_manifold_step(m_layout, i, vx, vy)?;
                }
            }
            let cand_stress = compute_stress_riemannian(state, m_layout)?;
            if cand_stress <= s_old {
                new_stress = cand_stress;
                break;
            }
            for i in 0..n {
                m_layout.set_point(
                    i,
                    L::Scalar::from_f64(state.sol_x[i]),
                    L::Scalar::from_f64(state.sol_y[i]),
                )?;
            }
        }
    } else {
        let saved = layout.clone();
        for &alpha in &BACKTRACK_ALPHAS {
            for i in 0..n {
                let grad_off = i * d;
                let mut any_nonzero = false;
                for a in 0..d {
                    let v = alpha * state.nd_rhs[grad_off + a];
                    state.step_buf[a] = v;
                    if v != 0.0 {
                        any_nonzero = true;
                    }
                }
                if any_nonzero {
                    layout.exp_map(i, &state.step_buf[..d])?;
                }
            }
            let cand_stress = compute_stress_riemannian(state, layout)?;
            if cand_stress <= s_old {
                new_stress = cand_stress;
                break;
            }
            *layout = saved.clone();
        }
    }

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
