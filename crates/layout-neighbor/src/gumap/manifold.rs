use std::f64::consts::TAU;

use egraph_core::{CoreResult, ManifoldKind, ManifoldLayout2d, Scalar};

use crate::manifold_utils::apply_clamped_manifold_step;

use super::solver::GuMapState;

const MIN_DIST_SQ_CLAMP: f64 = 1e-8;
const REPULSIVE_DIST_SQ_EPS: f64 = 1e-4;
const GRAD_CLAMP: f64 = 4.0;

fn scatter_zero_manifold_if_needed<S: Scalar>(
    state: &mut GuMapState<S>,
    layout: &mut ManifoldLayout2d<S>,
) -> CoreResult<()> {
    let n = state.num_nodes;
    if state.epoch != 0
        || !layout.x().iter().all(|&v| v == S::ZERO)
        || !layout.y().iter().all(|&v| v == S::ZERO)
    {
        return Ok(());
    }
    let kind = layout.kind();
    for idx in 0..n {
        let (px, py) = match kind {
            ManifoldKind::HyperbolicPoincare | ManifoldKind::HyperbolicLorentz => {
                let r = 0.5 * state.rng.next_f64().sqrt();
                let theta = TAU * state.rng.next_f64();
                (r * theta.cos(), r * theta.sin())
            }
            ManifoldKind::Spherical => {
                let lon = TAU * state.rng.next_f64();
                let u = (2.0 * state.rng.next_f64() - 1.0).clamp(-1.0, 1.0);
                (lon, u.acos())
            }
            ManifoldKind::Torus2d => (state.rng.next_f64(), state.rng.next_f64()),
            _ => {
                let r = 0.5 * state.rng.next_f64().sqrt();
                let theta = TAU * state.rng.next_f64();
                (r * theta.cos(), r * theta.sin())
            }
        };
        layout.set_point(idx, S::from_f64(px), S::from_f64(py))?;
    }
    Ok(())
}

pub(super) fn step_manifold<S: Scalar>(
    state: &mut GuMapState<S>,
    layout: &mut ManifoldLayout2d<S>,
) -> CoreResult<()> {
    if state.locked_manifold.is_none() {
        state.locked_manifold = Some(layout.kind());
    }

    scatter_zero_manifold_if_needed(state, layout)?;

    let n = state.num_nodes;
    let alpha = state.config.learning_rate
        * (1.0 - (state.epoch as f64) / (state.config.iterations as f64));
    let a = state.a;
    let b = state.b;
    let two_b = 2.0 * b;
    let two_ab = 2.0 * a * b;
    let neg_rate = state.config.negative_sample_rate;
    let m = state.edge_src.len();

    for idx in 0..m {
        let p_edge = state.edge_weight[idx] as f64;
        if p_edge <= 0.0 || (p_edge < 1.0 && state.rng.next_f64() > p_edge) {
            continue;
        }

        let i = state.edge_src[idx] as usize;
        let j = state.edge_tgt[idx] as usize;

        let (tix_s, tiy_s) = layout.tangent_delta(i, j)?;
        let (tjx_s, tjy_s) = layout.tangent_delta(j, i)?;
        let tx = tix_s.to_f64();
        let ty = tiy_s.to_f64();
        let d2 = (tx * tx + ty * ty).max(MIN_DIST_SQ_CLAMP);
        let d_2b = d2.powf(b);
        let d_pow = d_2b / d2;
        let raw_grad = -two_ab * d_pow / (1.0 + a * d_2b);
        if raw_grad.is_finite() {
            let grad_coeff = raw_grad.clamp(-GRAD_CLAMP, GRAD_CLAMP);
            let step_coeff = -alpha * grad_coeff;
            apply_clamped_manifold_step(layout, i, step_coeff * tx, step_coeff * ty)?;
            apply_clamped_manifold_step(
                layout,
                j,
                step_coeff * tjx_s.to_f64(),
                step_coeff * tjy_s.to_f64(),
            )?;
        }

        for _ in 0..neg_rate {
            let k = state.rng.gen_range_usize(n);
            if k == i {
                continue;
            }
            let (r_tix_s, r_tiy_s) = layout.tangent_delta(i, k)?;
            let tk = if state.config.symmetric_negative {
                Some(layout.tangent_delta(k, i)?)
            } else {
                None
            };
            let r_tx = r_tix_s.to_f64();
            let r_ty = r_tiy_s.to_f64();
            let r_d2_raw = r_tx * r_tx + r_ty * r_ty;
            let r_d2 = r_d2_raw.max(MIN_DIST_SQ_CLAMP);
            let r_d_2b = r_d2.powf(b);
            let denom = (r_d2_raw + REPULSIVE_DIST_SQ_EPS) * (1.0 + a * r_d_2b);
            let rep_coeff = if denom > 0.0 && denom.is_finite() {
                let raw_rep = two_b / denom;
                if raw_rep.is_finite() {
                    raw_rep.clamp(0.0, GRAD_CLAMP)
                } else {
                    GRAD_CLAMP
                }
            } else {
                0.0
            };
            let rep_step = -alpha * rep_coeff;
            apply_clamped_manifold_step(layout, i, rep_step * r_tx, rep_step * r_ty)?;
            if let Some((tkx_s, tky_s)) = tk {
                apply_clamped_manifold_step(
                    layout,
                    k,
                    rep_step * tkx_s.to_f64(),
                    rep_step * tky_s.to_f64(),
                )?;
            }
        }
    }

    Ok(())
}
