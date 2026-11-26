use egraph_core::{CoreError, CoreResult, IterativeSolver, Layout2d, Scalar, SolverStatus};

use super::yifan_hu::YifanHuSolver;

pub(super) const PROLONGATION_JITTER_FACTOR: f64 = 0.05;

impl<S: Scalar> YifanHuSolver<S> {
    fn initialize_coarsest_coords(&mut self, layout: &Layout2d<S>) {
        let n = self.num_nodes;
        let xs = layout.x();
        let ys = layout.y();
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for i in 0..n {
            let x = xs[i].to_f64();
            let y = ys[i].to_f64();
            self.active_x[i] = x;
            self.active_y[i] = y;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        let coarsest = self.levels.len() - 1;
        let coarse_n = self.levels[coarsest].graph.num_nodes();
        if (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 {
            let k_coarse = self.levels[coarsest].optimal_distance;
            let spread = k_coarse * (coarse_n as f64).sqrt().max(1.0);
            for i in 0..coarse_n {
                self.active_x[i] = spread * (self.rng.next_f64() - 0.5);
                self.active_y[i] = spread * (self.rng.next_f64() - 0.5);
            }
        } else if coarsest > 0 {
            let map = &self.fine_to_level[coarsest];
            self.fx[..coarse_n].fill(0.0);
            self.fy[..coarse_n].fill(0.0);
            self.rep_x[..coarse_n].fill(0.0);
            for u in 0..n {
                let c = map[u] as usize;
                self.fx[c] += self.active_x[u];
                self.fy[c] += self.active_y[u];
                self.rep_x[c] += 1.0;
            }
            for c in 0..coarse_n {
                let cnt = self.rep_x[c].max(1.0);
                self.active_x[c] = self.fx[c] / cnt;
                self.active_y[c] = self.fy[c] / cnt;
            }
        }
        self.initialized_coords = true;
    }

    fn prolongate_to_finer_level(&mut self) {
        let finer = self.current_level - 1;
        let fine_n = self.levels[finer].graph.num_nodes();
        let k_fine = self.levels[finer].optimal_distance;
        let Some(coarse_of) = self.levels[finer].coarse_of_finer.as_ref() else {
            return;
        };
        let Some(counts) = self.levels[finer].cluster_counts_of_finer.as_ref() else {
            return;
        };

        // Temporarily assemble fine positions in `fx`/`fy` before writing into `active_x`/`active_y`.
        for u in 0..fine_n {
            let c = coarse_of[u] as usize;
            let cx = self.active_x[c];
            let cy = self.active_y[c];
            if counts[c] <= 1 {
                self.fx[u] = cx;
                self.fy[u] = cy;
            } else {
                let angle = std::f64::consts::TAU * self.rng.next_f64();
                let radius =
                    PROLONGATION_JITTER_FACTOR * k_fine * (0.25 + 0.75 * self.rng.next_f64());
                let (s, cs) = angle.sin_cos();
                self.fx[u] = cx + radius * cs;
                self.fy[u] = cy + radius * s;
            }
        }
        self.active_x[..fine_n].copy_from_slice(&self.fx[..fine_n]);
        self.active_y[..fine_n].copy_from_slice(&self.fy[..fine_n]);

        self.current_level = finer;
        self.level_iter = 0;
        self.level_step_len = self.config.step_size * k_fine;
        self.level_progress = 0;
        self.prev_energy = f64::INFINITY;
    }

    fn write_active_coords_to_layout(&self, layout: &mut Layout2d<S>) {
        let (xs_mut, ys_mut) = layout.xy_mut();
        if self.current_level == 0 {
            for i in 0..self.num_nodes {
                let nx = S::from_f64(self.active_x[i]);
                let ny = S::from_f64(self.active_y[i]);
                if nx.is_finite() {
                    xs_mut[i] = nx;
                }
                if ny.is_finite() {
                    ys_mut[i] = ny;
                }
            }
        } else {
            let map = &self.fine_to_level[self.current_level];
            for (u, &c) in map.iter().enumerate() {
                let nx = S::from_f64(self.active_x[c as usize]);
                let ny = S::from_f64(self.active_y[c as usize]);
                if nx.is_finite() {
                    xs_mut[u] = nx;
                }
                if ny.is_finite() {
                    ys_mut[u] = ny;
                }
            }
        }
    }
}

impl<S: Scalar> IterativeSolver<S> for YifanHuSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "YifanHuSolver::step",
            ));
        }
        layout.validate_finite()?;
        if self.active_dims == 0 {
            self.active_dims = 2;
        } else if self.active_dims != 2 {
            return Err(CoreError::dimension_mismatch(
                self.active_dims,
                2,
                "YifanHuSolver::step::dims",
            ));
        }

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        if !self.initialized_coords {
            self.initialize_coarsest_coords(layout);
        } else if self.current_level == 0 {
            // Sync level 0 coordinates from `layout` so external updates are respected.
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                self.active_x[i] = xs[i].to_f64();
                self.active_y[i] = ys[i].to_f64();
            }
        }

        let level = &self.levels[self.current_level];
        let ln = level.graph.num_nodes();
        let k_level = level.optimal_distance;

        if ln <= 1 {
            self.level_iter += 1;
            self.total_iter_executed += 1;
            if self.current_level > 0 {
                self.prolongate_to_finer_level();
                self.write_active_coords_to_layout(layout);
                self.status = SolverStatus::Running;
            } else {
                self.write_active_coords_to_layout(layout);
                self.status = SolverStatus::Converged;
            }
            return Ok(self.status);
        }

        // 1. Repulsion at active level via unit-mass ArenaQuadtree2d Coulomb evaluation.
        self.quadtree.build_weighted_xy(
            &self.active_x[..ln],
            &self.active_y[..ln],
            &self.unit_masses[..ln],
        )?;
        let eps_sq = (1e-8 * k_level * k_level).max(1e-24);
        self.quadtree.compute_coulomb_repulsion(
            self.config.theta,
            eps_sq,
            &mut self.rep_x[..ln],
            &mut self.rep_y[..ln],
        )?;

        let rep_scale = self.config.relative_strength * k_level * k_level;
        for i in 0..ln {
            self.fx[i] = rep_scale * self.rep_x[i];
            self.fy[i] = rep_scale * self.rep_y[i];
        }

        // 2. Quadratic spring attraction (magnitude d^2 / K along unit vector -> (d / K) * delta).
        let level_graph = &self.levels[self.current_level].graph;
        let use_weights = self.current_level == 0 && self.has_explicit_weights;
        for u in 0..ln {
            let ux = self.active_x[u];
            let uy = self.active_y[u];
            let nbrs = level_graph.neighbors(u);
            let weights = level_graph.neighbor_weights(u);
            let mut ax = 0.0_f64;
            let mut ay = 0.0_f64;
            for (pos, &v_u32) in nbrs.iter().enumerate() {
                let v = v_u32 as usize;
                let w = if use_weights {
                    weights.map_or(1.0, |ws| ws[pos])
                } else {
                    1.0
                };
                if w == 0.0 {
                    continue;
                }
                let dx = self.active_x[v] - ux;
                let dy = self.active_y[v] - uy;
                let d = dx.hypot(dy);
                let mult = w * (d / k_level);
                ax += mult * dx;
                ay += mult * dy;
            }
            self.fx[u] += ax;
            self.fy[u] += ay;
        }

        // 3. Compute total squared force energy and adapt step length.
        let mut energy = 0.0_f64;
        for i in 0..ln {
            let fxi = if self.fx[i].is_finite() {
                self.fx[i]
            } else {
                0.0
            };
            let fyi = if self.fy[i].is_finite() {
                self.fy[i]
            } else {
                0.0
            };
            self.fx[i] = fxi;
            self.fy[i] = fyi;
            energy += fxi * fxi + fyi * fyi;
        }
        if !energy.is_finite() {
            energy = f64::MAX;
        }

        if self.level_iter > 0 {
            if energy < self.prev_energy {
                self.level_progress += 1;
                if self.level_progress >= 5 {
                    self.level_progress = 0;
                    self.level_step_len =
                        (self.level_step_len / self.config.step_ratio).min(4.0 * k_level);
                }
            } else {
                self.level_progress = 0;
                self.level_step_len *= self.config.step_ratio;
            }
        }
        self.prev_energy = energy;

        // 4. Displace vertices along normalized force direction by adaptive step length.
        let mut max_disp = 0.0_f64;
        let mut max_force = 0.0_f64;
        let step_len = self.level_step_len;
        for i in 0..ln {
            let fxi = self.fx[i];
            let fyi = self.fy[i];
            let f_norm = fxi.hypot(fyi);
            if f_norm > max_force {
                max_force = f_norm;
            }
            if f_norm > 1e-12 {
                let scale = step_len / f_norm;
                let nx = self.active_x[i] + fxi * scale;
                let ny = self.active_y[i] + fyi * scale;
                if nx.is_finite() && ny.is_finite() {
                    self.active_x[i] = nx;
                    self.active_y[i] = ny;
                    if step_len > max_disp {
                        max_disp = step_len;
                    }
                }
            }
        }

        self.level_iter += 1;
        self.total_iter_executed += 1;

        let level_converged =
            max_disp <= self.config.convergence_tolerance * k_level || max_force <= 1e-12;
        let level_exhausted = self.level_iter >= self.config.max_iterations_per_level;

        if level_converged || level_exhausted {
            if self.current_level > 0 {
                self.prolongate_to_finer_level();
                self.write_active_coords_to_layout(layout);
                self.status = SolverStatus::Running;
            } else {
                self.write_active_coords_to_layout(layout);
                self.status = if level_converged {
                    SolverStatus::Converged
                } else {
                    SolverStatus::Completed
                };
            }
        } else {
            self.write_active_coords_to_layout(layout);
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
        self.max_total_iterations
    }
}
