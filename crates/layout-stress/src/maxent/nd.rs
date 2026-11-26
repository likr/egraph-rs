use egraph_core::{CoreError, CoreResult, IterativeSolver, LayoutNd, Scalar, SolverStatus};

use super::graph_prep::{is_layout_collapsed_nd, norm2_slice};
use super::solver::MaxentStressSolver;

const MIN_NORM_SQ_SUM: f64 = 1e-12;

#[inline]
fn euclid_sq_diff_flat(coords: &[f64], d: usize, u: usize, v: usize) -> f64 {
    let base_u = u * d;
    let base_v = v * d;
    let mut sum = 0.0_f64;
    for a in 0..d {
        let diff = coords[base_u + a] - coords[base_v + a];
        sum += diff * diff;
    }
    sum
}

#[inline]
fn euclid_norm_diff_flat(coords: &[f64], d: usize, u: usize, v: usize) -> f64 {
    let sum_sq = euclid_sq_diff_flat(coords, d, u, v);
    if sum_sq.is_finite() {
        return sum_sq.sqrt();
    }
    let base_u = u * d;
    let base_v = v * d;
    let mut max_abs = 0.0_f64;
    for a in 0..d {
        let av = (coords[base_u + a] - coords[base_v + a]).abs();
        if av > max_abs {
            max_abs = av;
        }
    }
    if !max_abs.is_finite() || max_abs == 0.0 {
        return max_abs;
    }
    let mut scaled_sq = 0.0_f64;
    for a in 0..d {
        let r = (coords[base_u + a] - coords[base_v + a]) / max_abs;
        scaled_sq += r * r;
    }
    max_abs * scaled_sq.sqrt()
}

impl<S: Scalar> MaxentStressSolver<S> {
    fn ensure_nd_buffers(&mut self, n: usize, d: usize) -> CoreResult<()> {
        let needed = n
            .checked_mul(d)
            .ok_or_else(|| CoreError::invalid_input("LayoutNd buffer size overflow"))?;
        let comp_needed = self
            .comp_sizes
            .len()
            .checked_mul(d)
            .ok_or_else(|| CoreError::invalid_input("component sum buffer size overflow"))?;
        if self.nd_cur.len() != needed {
            self.nd_cur.resize(needed, 0.0);
            self.nd_next.resize(needed, 0.0);
            self.nd_stress_rhs.resize(needed, 0.0);
            self.nd_rep.resize(needed, 0.0);
        }
        if self.nd_comp_sum.len() != comp_needed {
            self.nd_comp_sum.resize(comp_needed, 0.0);
        }
        if self.nd_rhs_axis.len() != n {
            self.nd_rhs_axis.resize(n, 0.0);
            self.nd_next_axis.resize(n, 0.0);
        }
        Ok(())
    }
}

impl<S: Scalar> IterativeSolver<S, LayoutNd<S>> for MaxentStressSolver<S> {
    fn step(&mut self, layout: &mut LayoutNd<S>) -> CoreResult<SolverStatus> {
        let d = layout.dims();
        if d == 0 {
            return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
        }
        if layout.num_nodes() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.num_nodes(),
                "MaxentStressSolver::step",
            ));
        }
        layout.validate_finite()?;
        if let Some(expected_dims) = self.nd_dims
            && d != expected_dims
        {
            return Err(CoreError::dimension_mismatch(
                expected_dims,
                d,
                "MaxentStressSolver::step::dims",
            ));
        }
        if self.config.use_barnes_hut && d != 2 && d != 3 {
            return Err(CoreError::unsupported(format!(
                "MaxentStressSolver Barnes-Hut requires dims == 2 or dims == 3, got {d}; set use_barnes_hut = false for arbitrary dimensions"
            )));
        }
        self.nd_dims = Some(d);

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        if self.total_iter_executed == 0 && is_layout_collapsed_nd(layout) {
            self.initialize_collapsed_layout_nd(layout)?;
        }

        self.ensure_nd_buffers(n, d)?;

        for (dst, &src) in self.nd_cur.iter_mut().zip(layout.coords().iter()) {
            *dst = src.to_f64();
        }

        // 1. Majorizing stress RHS: L_{w,d}(x^{(t)}) x^{(t)}.
        self.nd_stress_rhs.fill(0.0);
        for edge in &self.edges {
            let u = edge.u as usize;
            let v = edge.v as usize;
            let norm = euclid_norm_diff_flat(&self.nd_cur, d, u, v);
            if norm > 1e-12 && norm.is_finite() {
                let f = edge.w * edge.d / norm;
                let base_u = u * d;
                let base_v = v * d;
                for a in 0..d {
                    let fa = f * (self.nd_cur[base_u + a] - self.nd_cur[base_v + a]);
                    self.nd_stress_rhs[base_u + a] += fa;
                    self.nd_stress_rhs[base_v + a] -= fa;
                }
            }
        }

        // 2. Repulsive force b(x) over non-adjacent vertex pairs {i, j} \notin S.
        let eps_sq = self.config.eps_sq;
        let q = self.effective_q;
        let is_q_zero = q == 0.0;
        let exp = 0.5 * (q + 2.0);
        let mut rep_coeff = 0.0_f64;

        if self.alpha > 0.0 {
            if self.config.use_barnes_hut {
                if d == 3 {
                    self.octree.build_flat(&self.nd_cur, n)?;
                    self.octree.compute_maxent_repulsion_flat(
                        self.config.theta,
                        q,
                        eps_sq,
                        &mut self.nd_rep,
                    )?;
                } else {
                    for i in 0..n {
                        self.x_cur[i] = self.nd_cur[i * 2];
                        self.y_cur[i] = self.nd_cur[i * 2 + 1];
                    }
                    self.quadtree.build_xy(&self.x_cur, &self.y_cur)?;
                    self.quadtree.compute_maxent_repulsion(
                        self.config.theta,
                        q,
                        eps_sq,
                        &mut self.rep_x,
                        &mut self.rep_y,
                    )?;
                    for i in 0..n {
                        self.nd_rep[i * 2] = self.rep_x[i];
                        self.nd_rep[i * 2 + 1] = self.rep_y[i];
                    }
                }
            } else {
                self.nd_rep.fill(0.0);
                for i in 0..n {
                    let base_i = i * d;
                    for j in (i + 1)..n {
                        let base_j = j * d;
                        let dist_sq = euclid_sq_diff_flat(&self.nd_cur, d, i, j);
                        let d2 = dist_sq + eps_sq;
                        let mult = if is_q_zero {
                            1.0 / d2
                        } else {
                            1.0 / d2.powf(exp)
                        };
                        for a in 0..d {
                            let fa = mult * (self.nd_cur[base_i + a] - self.nd_cur[base_j + a]);
                            self.nd_rep[base_i + a] += fa;
                            self.nd_rep[base_j + a] -= fa;
                        }
                    }
                }
            }

            // Subtract 1-hop S-edge pairs so b(x) sums strictly over {i, j} \notin S.
            for edge in &self.edges {
                let u = edge.u as usize;
                let v = edge.v as usize;
                let dist_sq = euclid_sq_diff_flat(&self.nd_cur, d, u, v);
                let d2 = dist_sq + eps_sq;
                let mult = if is_q_zero {
                    1.0 / d2
                } else {
                    1.0 / d2.powf(exp)
                };
                let base_u = u * d;
                let base_v = v * d;
                for a in 0..d {
                    let fa = mult * (self.nd_cur[base_u + a] - self.nd_cur[base_v + a]);
                    self.nd_rep[base_u + a] -= fa;
                    self.nd_rep[base_v + a] += fa;
                }
            }

            // Zero isolated vertices and center b(x) per connected component so
            // sum_{i in C_k} b_i == 0 lies strictly in range(L_w).
            self.nd_comp_sum.fill(0.0);
            for i in 0..n {
                let cid = self.comp_id[i] as usize;
                let base_i = i * d;
                if self.comp_sizes[cid] <= 1 {
                    self.nd_rep[base_i..base_i + d].fill(0.0);
                } else {
                    let base_c = cid * d;
                    for a in 0..d {
                        self.nd_comp_sum[base_c + a] += self.nd_rep[base_i + a];
                    }
                }
            }
            for i in 0..n {
                let cid = self.comp_id[i] as usize;
                let sz = self.comp_sizes[cid];
                if sz > 1 {
                    let inv_sz = 1.0 / (sz as f64);
                    let base_i = i * d;
                    let base_c = cid * d;
                    for a in 0..d {
                        self.nd_rep[base_i + a] -= self.nd_comp_sum[base_c + a] * inv_sz;
                    }
                }
            }

            let norm_stress = norm2_slice(&self.nd_stress_rhs);
            let norm_rep = norm2_slice(&self.nd_rep);
            if norm_rep > 1e-30 {
                let scale = if norm_stress > 0.0 {
                    norm_stress / norm_rep
                } else {
                    1.0
                };
                rep_coeff = self.alpha * scale;
            }
        }

        // 3. Solve L_w x_a^{(t+1)} = rhs_a warm-started from x_a^{(t)} per coordinate axis.
        for a in 0..d {
            if rep_coeff != 0.0 {
                for i in 0..n {
                    let idx = i * d + a;
                    self.nd_rhs_axis[i] = self.nd_stress_rhs[idx] + rep_coeff * self.nd_rep[idx];
                    self.nd_next_axis[i] = self.nd_cur[idx];
                }
            } else {
                for i in 0..n {
                    let idx = i * d + a;
                    self.nd_rhs_axis[i] = self.nd_stress_rhs[idx];
                    self.nd_next_axis[i] = self.nd_cur[idx];
                }
            }

            self.amg
                .solve(&self.nd_rhs_axis, &mut self.nd_next_axis, &mut self.amg_ws)?;

            for i in 0..n {
                self.nd_next[i * d + a] = self.nd_next_axis[i];
            }
        }

        for i in 0..n {
            let base = i * d;
            let is_isolated = self.comp_sizes[self.comp_id[i] as usize] <= 1;
            let all_finite =
                !is_isolated && self.nd_next[base..base + d].iter().all(|v| v.is_finite());
            if !all_finite {
                self.nd_next[base..base + d].copy_from_slice(&self.nd_cur[base..base + d]);
            }
        }

        let mut diff_sq = 0.0_f64;
        let mut cur_sq = 0.0_f64;
        for (&next_v, &cur_v) in self.nd_next.iter().zip(self.nd_cur.iter()) {
            let diff = next_v - cur_v;
            diff_sq += diff * diff;
            cur_sq += cur_v * cur_v;
        }
        let rel_change = diff_sq.sqrt() / cur_sq.sqrt().max(MIN_NORM_SQ_SUM);

        {
            let coords_mut = layout.coords_mut();
            for (dst, &val) in coords_mut.iter_mut().zip(self.nd_next.iter()) {
                let sv = S::from_f64(val);
                if sv.is_finite() {
                    *dst = sv;
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
