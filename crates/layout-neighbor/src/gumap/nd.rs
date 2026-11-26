use egraph_core::{CoreError, CoreResult, IterativeSolver, LayoutNd, Scalar, SolverStatus};

use super::solver::{ActiveLayoutKind, GuMapSolver};

const MIN_DIST_SQ_CLAMP: f64 = 1e-8;
const REPULSIVE_DIST_SQ_EPS: f64 = 1e-4;
const GRAD_CLAMP: f64 = 4.0;

impl<S: Scalar> IterativeSolver<S, LayoutNd<S>> for GuMapSolver<S> {
    fn step(&mut self, layout: &mut LayoutNd<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "GuMapSolver::step",
            ));
        }
        let dims = layout.dims();
        if dims == 0 {
            return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
        }
        layout.validate_finite()?;
        match self.active_layout {
            ActiveLayoutKind::Unset => self.active_layout = ActiveLayoutKind::Nd(dims),
            ActiveLayoutKind::Euclidean2d => {
                return Err(CoreError::invalid_input(
                    "GuMapSolver::step: cannot switch from Layout2d to LayoutNd on stateful solver",
                ));
            }
            ActiveLayoutKind::Nd(prev) if prev != dims => {
                return Err(CoreError::dimension_mismatch(
                    prev,
                    dims,
                    "GuMapSolver::step::dims",
                ));
            }
            ActiveLayoutKind::Nd(_) => {}
        }

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n < 2 || self.edge_src.is_empty() {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let coords = layout.coords_mut();

        // Match 2D GuMap behavior when an uninitialized all-zero layout is supplied.
        if self.epoch == 0 && coords.iter().all(|&v| v == S::ZERO) {
            for val in coords.iter_mut() {
                *val = S::from_f64(self.rng.next_f64() * 20.0 - 10.0);
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
            let i_base = i * dims;
            let j_base = j * dims;

            // 1. Attractive gradient via D-dimensional d2.powf(b) without sqrt.
            let mut d2_raw = S::ZERO;
            for d in 0..dims {
                let diff = coords[i_base + d] - coords[j_base + d];
                d2_raw += diff * diff;
            }
            let d2 = d2_raw.max(eps_sq_s);
            let d_2b = d2.powf(b_s);
            let d_pow = d_2b / d2;
            let raw_grad = -two_ab_s * d_pow / (S::ONE + a_s * d_2b);
            if raw_grad.is_finite() {
                let grad_coeff = raw_grad.clamp(neg_four_s, four_s);
                let step_coeff = alpha * grad_coeff;
                for d in 0..dims {
                    let diff = coords[i_base + d] - coords[j_base + d];
                    let delta = step_coeff * diff;
                    let new_i = coords[i_base + d] + delta;
                    let new_j = coords[j_base + d] - delta;
                    if new_i.is_finite() {
                        coords[i_base + d] = new_i;
                    }
                    if new_j.is_finite() {
                        coords[j_base + d] = new_j;
                    }
                }
            }

            // 2. Deterministic negative sampling via D-dimensional d2.powf(b) without sqrt.
            for _ in 0..neg_rate {
                let k = self.rng.gen_range_usize(n);
                if k == i {
                    continue;
                }
                let k_base = k * dims;
                let mut r_d2_raw = S::ZERO;
                for d in 0..dims {
                    let r_diff = coords[i_base + d] - coords[k_base + d];
                    r_d2_raw += r_diff * r_diff;
                }
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
                let rep_step = alpha * rep_coeff;
                for d in 0..dims {
                    let r_diff = coords[i_base + d] - coords[k_base + d];
                    let r_delta = rep_step * r_diff;
                    let new_i = coords[i_base + d] + r_delta;
                    if new_i.is_finite() {
                        coords[i_base + d] = new_i;
                    }
                    if self.config.symmetric_negative {
                        let new_k = coords[k_base + d] - r_delta;
                        if new_k.is_finite() {
                            coords[k_base + d] = new_k;
                        }
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
