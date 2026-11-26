use std::f64::consts::TAU;

use egraph_core::{
    CoreError, CoreResult, IterativeSolver, ManifoldKind, ManifoldLayout2d, Scalar, SolverStatus,
};

use crate::manifold_utils::apply_clamped_manifold_step;

use super::solver::GuMapSolver;

const MIN_DIST_SQ_CLAMP: f64 = 1e-8;
const REPULSIVE_DIST_SQ_EPS: f64 = 1e-4;
const GRAD_CLAMP: f64 = 4.0;

impl<S: Scalar> GuMapSolver<S> {
    fn scatter_zero_manifold_if_needed(
        &mut self,
        layout: &mut ManifoldLayout2d<S>,
    ) -> CoreResult<()> {
        let n = self.num_nodes;
        if self.epoch != 0
            || !layout.x().iter().all(|&v| v == S::ZERO)
            || !layout.y().iter().all(|&v| v == S::ZERO)
        {
            return Ok(());
        }
        let kind = layout.kind();
        for idx in 0..n {
            let (px, py) = match kind {
                ManifoldKind::HyperbolicPoincare | ManifoldKind::HyperbolicLorentz => {
                    let r = 0.5 * self.rng.next_f64().sqrt();
                    let theta = TAU * self.rng.next_f64();
                    (r * theta.cos(), r * theta.sin())
                }
                ManifoldKind::Spherical => {
                    let lon = TAU * self.rng.next_f64();
                    let u = (2.0 * self.rng.next_f64() - 1.0).clamp(-1.0, 1.0);
                    (lon, u.acos())
                }
                ManifoldKind::Torus2d => (self.rng.next_f64(), self.rng.next_f64()),
                _ => {
                    let r = 0.5 * self.rng.next_f64().sqrt();
                    let theta = TAU * self.rng.next_f64();
                    (r * theta.cos(), r * theta.sin())
                }
            };
            layout.set_point(idx, S::from_f64(px), S::from_f64(py))?;
        }
        Ok(())
    }
}

impl<S: Scalar> IterativeSolver<S, ManifoldLayout2d<S>> for GuMapSolver<S> {
    fn step(&mut self, layout: &mut ManifoldLayout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "GuMapSolver::step",
            ));
        }
        layout.validate_finite()?;
        self.active_layout
            .lock_manifold(layout.kind(), "GuMapSolver")?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n < 2 || self.edge_src.is_empty() {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        self.scatter_zero_manifold_if_needed(layout)?;

        let alpha = self.config.learning_rate
            * (1.0 - (self.epoch as f64) / (self.config.iterations as f64));
        let a = self.a;
        let b = self.b;
        let two_b = 2.0 * b;
        let two_ab = 2.0 * a * b;
        let neg_rate = self.config.negative_sample_rate;
        let m = self.edge_src.len();

        for idx in 0..m {
            let p_edge = self.edge_weight[idx] as f64;
            if p_edge <= 0.0 || (p_edge < 1.0 && self.rng.next_f64() > p_edge) {
                continue;
            }

            let i = self.edge_src[idx] as usize;
            let j = self.edge_tgt[idx] as usize;

            // 1. Attractive geodesic update using pre-step tangent vectors on both endpoints.
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

            // 2. Deterministic negative-sampled geodesic repulsion.
            for _ in 0..neg_rate {
                let k = self.rng.gen_range_usize(n);
                if k == i {
                    continue;
                }
                let (r_tix_s, r_tiy_s) = layout.tangent_delta(i, k)?;
                let tk = if self.config.symmetric_negative {
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

        self.epoch += 1;
        self.status = if self.epoch >= self.config.iterations {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.epoch
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.config.iterations
    }
}
