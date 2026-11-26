use egraph_core::{CoreError, CoreResult, IterativeSolver, Layout2d, Scalar, SolverStatus};

use super::solver::GuMapSolver;

const MIN_DIST_SQ_CLAMP: f64 = 1e-8;
const REPULSIVE_DIST_SQ_EPS: f64 = 1e-4;
const GRAD_CLAMP: f64 = 4.0;

impl<S: Scalar> IterativeSolver<S> for GuMapSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "GuMapSolver::step",
            ));
        }
        layout.validate_finite()?;
        self.active_layout.lock_2d("GuMapSolver")?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n < 2 || self.edge_src.is_empty() {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let (xs, ys) = layout.xy_mut();

        // Match legacy GuMap behavior when an uninitialized all-zero layout is supplied.
        if self.epoch == 0 && xs.iter().all(|&v| v == S::ZERO) && ys.iter().all(|&v| v == S::ZERO) {
            for idx in 0..n {
                xs[idx] = S::from_f64(self.rng.next_f64() * 20.0 - 10.0);
                ys[idx] = S::from_f64(self.rng.next_f64() * 20.0 - 10.0);
            }
        }

        let alpha = S::from_f64(
            self.config.learning_rate
                * (1.0 - (self.epoch as f64) / (self.config.iterations as f64)),
        );
        let a_s = S::from_f64(self.a);
        let b_s = S::from_f64(self.b);
        let two_b_s = S::from_f64(2.0 * self.b);
        let two_ab_s = S::from_f64(2.0 * self.a * self.b);
        let four_s = S::from_f64(GRAD_CLAMP);
        let neg_four_s = -four_s;
        let eps_sq_s = S::from_f64(MIN_DIST_SQ_CLAMP);
        let rep_eps_s = S::from_f64(REPULSIVE_DIST_SQ_EPS);

        let neg_rate = self.config.negative_sample_rate;
        let m = self.edge_src.len();

        for idx in 0..m {
            let p_edge = self.edge_weight[idx] as f64;
            if p_edge <= 0.0 || (p_edge < 1.0 && self.rng.next_f64() > p_edge) {
                continue;
            }

            let i = self.edge_src[idx] as usize;
            let j = self.edge_tgt[idx] as usize;

            // 1. Attractive gradient via d2.powf(b) without sqrt.
            let dx = xs[i] - xs[j];
            let dy = ys[i] - ys[j];
            let d2 = (dx * dx + dy * dy).max(eps_sq_s);
            let d_2b = d2.powf(b_s);
            let d_pow = d_2b / d2;
            let raw_grad = -two_ab_s * d_pow / (S::ONE + a_s * d_2b);
            if raw_grad.is_finite() {
                let grad_coeff = raw_grad.clamp(neg_four_s, four_s);
                let delta_x = alpha * grad_coeff * dx;
                let delta_y = alpha * grad_coeff * dy;
                let new_ix = xs[i] + delta_x;
                let new_iy = ys[i] + delta_y;
                let new_jx = xs[j] - delta_x;
                let new_jy = ys[j] - delta_y;
                if new_ix.is_finite() {
                    xs[i] = new_ix;
                }
                if new_iy.is_finite() {
                    ys[i] = new_iy;
                }
                if new_jx.is_finite() {
                    xs[j] = new_jx;
                }
                if new_jy.is_finite() {
                    ys[j] = new_jy;
                }
            }

            // 2. Deterministic negative sampling via d2.powf(b) without sqrt.
            for _ in 0..neg_rate {
                let k = self.rng.gen_range_usize(n);
                if k == i {
                    continue;
                }
                let r_dx = xs[i] - xs[k];
                let r_dy = ys[i] - ys[k];
                let r_d2_raw = r_dx * r_dx + r_dy * r_dy;
                let r_d2 = r_d2_raw.max(eps_sq_s);
                let r_d_2b = r_d2.powf(b_s);
                let denom = (r_d2_raw + rep_eps_s) * (S::ONE + a_s * r_d_2b);
                let rep_coeff = if denom > S::ZERO && denom.is_finite() {
                    let raw_rep = two_b_s / denom;
                    if raw_rep.is_finite() {
                        raw_rep.clamp(S::ZERO, four_s)
                    } else {
                        four_s
                    }
                } else {
                    S::ZERO
                };
                let r_delta_x = alpha * rep_coeff * r_dx;
                let r_delta_y = alpha * rep_coeff * r_dy;
                let new_ix = xs[i] + r_delta_x;
                let new_iy = ys[i] + r_delta_y;
                if new_ix.is_finite() {
                    xs[i] = new_ix;
                }
                if new_iy.is_finite() {
                    ys[i] = new_iy;
                }
                if self.config.symmetric_negative {
                    let new_kx = xs[k] - r_delta_x;
                    let new_ky = ys[k] - r_delta_y;
                    if new_kx.is_finite() {
                        xs[k] = new_kx;
                    }
                    if new_ky.is_finite() {
                        ys[k] = new_ky;
                    }
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
