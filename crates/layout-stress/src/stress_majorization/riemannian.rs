use egraph_core::{CoreError, CoreResult, RiemannianLayout};

use super::config::StressMajorizationConfig;
use super::strategy::{GeodesicGradient, StressMajorizationStrategy};

const BACKTRACK_ALPHAS: [f64; 6] = [0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125];

/// Internal workspace state for [`GeodesicGradient`].
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct GeodesicGradientState {
    nd_rhs: Vec<f64>,
    log_buf: Vec<f64>,
    step_buf: Vec<f64>,
    locked_dim: Option<usize>,
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

fn compute_stress_riemannian<L: RiemannianLayout + ?Sized>(
    n: usize,
    dists: &[f64],
    weights: &[f64],
    layout: &L,
) -> CoreResult<f64> {
    let mut total = 0.0_f64;
    for i in 0..n {
        let row_off = i * n;
        for j in (i + 1)..n {
            let w = weights[row_off + j];
            if w <= 0.0 {
                continue;
            }
            let d_target = dists[row_off + j];
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

impl<L: RiemannianLayout + Clone> StressMajorizationStrategy<L> for GeodesicGradient {
    type State = GeodesicGradientState;

    fn init_state(&self, _num_nodes: usize, _weights: &[f64]) -> CoreResult<Self::State> {
        Ok(GeodesicGradientState {
            nd_rhs: Vec::new(),
            log_buf: Vec::new(),
            step_buf: Vec::new(),
            locked_dim: None,
        })
    }

    fn validate_target(&self, state: &Self::State, num_nodes: usize, target: &L) -> CoreResult<()> {
        if target.len() != num_nodes {
            return Err(CoreError::dimension_mismatch(
                num_nodes,
                target.len(),
                "StressMajorizationSolver::validate_target",
            ));
        }
        let dim = target.tangent_dim();
        if dim == 0 && num_nodes > 0 {
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

    fn compute_stress(
        &self,
        _state: &Self::State,
        num_nodes: usize,
        dists: &[f64],
        weights: &[f64],
        layout: &L,
    ) -> CoreResult<f64> {
        compute_stress_riemannian(num_nodes, dists, weights, layout)
    }

    fn step(
        &self,
        state: &mut Self::State,
        num_nodes: usize,
        dists: &[f64],
        weights: &[f64],
        config: &StressMajorizationConfig,
        prev_stress: f64,
        layout: &mut L,
    ) -> CoreResult<Option<f64>> {
        let d = layout.tangent_dim();
        if d > 0 && state.locked_dim.is_none() {
            state.locked_dim = Some(d);
        }

        let n = num_nodes;
        if n <= 1 {
            return Ok(None);
        }

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

        let eff_min_norm = config.min_norm;
        for i in 0..n {
            let row_off = i * n;
            let grad_off = i * d;
            state.nd_rhs[grad_off..grad_off + d].fill(0.0);
            let mut diag_w = 0.0_f64;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let w = weights[row_off + j];
                if w <= 0.0 {
                    continue;
                }
                diag_w += w;
                layout.log_map_into(i, j, &mut state.log_buf)?;
                let norm = euclid_norm(&state.log_buf[..d]);
                if norm >= eff_min_norm && norm.is_finite() {
                    let d_target = dists[row_off + j];
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
        let mut new_stress = prev_stress;
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
            let cand_stress = compute_stress_riemannian(n, dists, weights, layout)?;
            if cand_stress <= prev_stress {
                new_stress = cand_stress;
                break;
            }
            *layout = saved.clone();
        }

        Ok(Some(new_stress))
    }
}
