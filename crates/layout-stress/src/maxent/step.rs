use egraph_core::{CoreError, CoreResult, IterativeSolver, Layout2d, Scalar, SolverStatus};

use super::graph_prep::{is_layout_collapsed, norm2_pair};
use super::solver::MaxentStressSolver;

const MIN_NORM_SQ_SUM: f64 = 1e-12;

impl<S: Scalar> IterativeSolver<S> for MaxentStressSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "MaxentStressSolver::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        if self.total_iter_executed == 0 && is_layout_collapsed(layout) {
            self.initialize_collapsed_layout(layout)?;
        }

        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                self.x_cur[i] = xs[i].to_f64();
                self.y_cur[i] = ys[i].to_f64();
            }
        }

        // 1. Majorizing stress RHS: L_{w,d}(x^{(t)}) x^{(t)}.
        self.stress_rhs_x.fill(0.0);
        self.stress_rhs_y.fill(0.0);
        for edge in &self.edges {
            let u = edge.u as usize;
            let v = edge.v as usize;
            let dx = self.x_cur[u] - self.x_cur[v];
            let dy = self.y_cur[u] - self.y_cur[v];
            let norm = dx.hypot(dy);
            if norm > 1e-12 && norm.is_finite() {
                let f = edge.w * edge.d / norm;
                let fx = f * dx;
                let fy = f * dy;
                self.stress_rhs_x[u] += fx;
                self.stress_rhs_y[u] += fy;
                self.stress_rhs_x[v] -= fx;
                self.stress_rhs_y[v] -= fy;
            }
        }

        // 2. Repulsive force b(x) over non-adjacent vertex pairs {i, j} \notin S.
        let eps_sq = self.config.eps_sq;
        let q = self.effective_q;
        let is_q_zero = q == 0.0;
        let exp = 0.5 * (q + 2.0);

        if self.alpha > 0.0 {
            if self.config.use_barnes_hut {
                self.quadtree.build_xy(&self.x_cur, &self.y_cur)?;
                self.quadtree.compute_maxent_repulsion(
                    self.config.theta,
                    q,
                    eps_sq,
                    &mut self.rep_x,
                    &mut self.rep_y,
                )?;
            } else {
                self.rep_x.fill(0.0);
                self.rep_y.fill(0.0);
                for i in 0..n {
                    let xi = self.x_cur[i];
                    let yi = self.y_cur[i];
                    for j in (i + 1)..n {
                        let dx = xi - self.x_cur[j];
                        let dy = yi - self.y_cur[j];
                        let d2 = dx * dx + dy * dy + eps_sq;
                        let mult = if is_q_zero {
                            1.0 / d2
                        } else {
                            1.0 / d2.powf(exp)
                        };
                        let fx = mult * dx;
                        let fy = mult * dy;
                        self.rep_x[i] += fx;
                        self.rep_y[i] += fy;
                        self.rep_x[j] -= fx;
                        self.rep_y[j] -= fy;
                    }
                }
            }

            // Subtract 1-hop S-edge pairs so b(x) sums strictly over {i, j} \notin S.
            for edge in &self.edges {
                let u = edge.u as usize;
                let v = edge.v as usize;
                let dx = self.x_cur[u] - self.x_cur[v];
                let dy = self.y_cur[u] - self.y_cur[v];
                let d2 = dx * dx + dy * dy + eps_sq;
                let mult = if is_q_zero {
                    1.0 / d2
                } else {
                    1.0 / d2.powf(exp)
                };
                let fx = mult * dx;
                let fy = mult * dy;
                self.rep_x[u] -= fx;
                self.rep_y[u] -= fy;
                self.rep_x[v] += fx;
                self.rep_y[v] += fy;
            }

            // Zero isolated vertices and center b(x) per connected component so
            // sum_{i in C_k} b_i == 0 lies strictly in range(L_w).
            self.comp_sum_x.fill(0.0);
            self.comp_sum_y.fill(0.0);
            for i in 0..n {
                let cid = self.comp_id[i] as usize;
                if self.comp_sizes[cid] <= 1 {
                    self.rep_x[i] = 0.0;
                    self.rep_y[i] = 0.0;
                } else {
                    self.comp_sum_x[cid] += self.rep_x[i];
                    self.comp_sum_y[cid] += self.rep_y[i];
                }
            }
            for i in 0..n {
                let cid = self.comp_id[i] as usize;
                let sz = self.comp_sizes[cid];
                if sz > 1 {
                    let inv_sz = 1.0 / (sz as f64);
                    self.rep_x[i] -= self.comp_sum_x[cid] * inv_sz;
                    self.rep_y[i] -= self.comp_sum_y[cid] * inv_sz;
                }
            }

            let norm_stress = norm2_pair(&self.stress_rhs_x, &self.stress_rhs_y);
            let norm_rep = norm2_pair(&self.rep_x, &self.rep_y);
            if norm_rep > 1e-30 {
                let scale = if norm_stress > 0.0 {
                    norm_stress / norm_rep
                } else {
                    1.0
                };
                let coeff = self.alpha * scale;
                for i in 0..n {
                    self.rhs_x[i] = self.stress_rhs_x[i] + coeff * self.rep_x[i];
                    self.rhs_y[i] = self.stress_rhs_y[i] + coeff * self.rep_y[i];
                }
            } else {
                self.rhs_x.copy_from_slice(&self.stress_rhs_x);
                self.rhs_y.copy_from_slice(&self.stress_rhs_y);
            }
        } else {
            self.rhs_x.copy_from_slice(&self.stress_rhs_x);
            self.rhs_y.copy_from_slice(&self.stress_rhs_y);
        }

        // 3. Solve L_w x^{(t+1)} = rhs warm-started from x^{(t)}.
        self.next_x.copy_from_slice(&self.x_cur);
        self.next_y.copy_from_slice(&self.y_cur);
        self.amg
            .solve(&self.rhs_x, &mut self.next_x, &mut self.amg_ws)?;
        self.amg
            .solve(&self.rhs_y, &mut self.next_y, &mut self.amg_ws)?;

        for i in 0..n {
            if self.comp_sizes[self.comp_id[i] as usize] <= 1
                || !self.next_x[i].is_finite()
                || !self.next_y[i].is_finite()
            {
                self.next_x[i] = self.x_cur[i];
                self.next_y[i] = self.y_cur[i];
            }
        }

        let mut diff_sq = 0.0_f64;
        let mut cur_sq = 0.0_f64;
        for i in 0..n {
            let dx = self.next_x[i] - self.x_cur[i];
            let dy = self.next_y[i] - self.y_cur[i];
            diff_sq += dx * dx + dy * dy;
            cur_sq += self.x_cur[i] * self.x_cur[i] + self.y_cur[i] * self.y_cur[i];
        }
        let rel_change = diff_sq.sqrt() / cur_sq.sqrt().max(MIN_NORM_SQ_SUM);

        {
            let (xs_mut, ys_mut) = layout.xy_mut();
            for i in 0..n {
                let nx = S::from_f64(self.next_x[i]);
                let ny = S::from_f64(self.next_y[i]);
                if nx.is_finite() {
                    xs_mut[i] = nx;
                }
                if ny.is_finite() {
                    ys_mut[i] = ny;
                }
            }
        }

        self.stage_step += 1;
        self.total_iter_executed += 1;

        let stage_converged = rel_change < self.config.tol;
        if stage_converged || self.stage_step >= self.config.steps_per_stage {
            self.stage_idx += 1;
            self.stage_step = 0;
            if self.stage_idx >= self.config.stages {
                self.status = if stage_converged {
                    SolverStatus::Converged
                } else {
                    SolverStatus::Completed
                };
            } else {
                self.alpha = (self.alpha * self.config.alpha_decay).max(self.config.alpha_min);
                self.status = SolverStatus::Running;
            }
        } else {
            self.status = SolverStatus::Running;
        }

        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.total_iter_executed
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.total_iterations_budget
    }
}
