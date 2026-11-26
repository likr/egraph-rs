use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};

use super::force_atlas2::ForceAtlas2Solver;
use super::overlap_quadtree::overlap_repulsion_pair;

pub(super) const COULOMB_EPS_SQ: f64 = 1e-6;
pub(super) const MIN_FA2_SPEED: f64 = 1e-4;

impl<S: Scalar> IterativeSolver<S> for ForceAtlas2Solver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "ForceAtlas2Solver::step",
            ));
        }
        layout.validate_finite()?;
        if self.active_dims == 0 {
            self.active_dims = 2;
        } else if self.active_dims != 2 {
            return Err(CoreError::dimension_mismatch(
                self.active_dims,
                2,
                "ForceAtlas2Solver::step::dims",
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

        {
            let xs = layout.x();
            let ys = layout.y();
            let mut min_x = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            for i in 0..n {
                let x = xs[i].to_f64();
                let y = ys[i].to_f64();
                self.x_buf[i] = x;
                self.y_buf[i] = y;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }

            // Break completely collapsed initial layouts deterministically so forces are non-zero.
            if self.iter == 0 && (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 {
                let mut rng = DeterministicRng::from_seed_u64(self.config.seed);
                for i in 0..n {
                    self.x_buf[i] += 1e-4 * (rng.next_f64() - 0.5);
                    self.y_buf[i] += 1e-4 * (rng.next_f64() - 0.5);
                }
            }
        }

        // 1. Repulsion: Jacomy (2014) border-to-border `d' = d - r1 - r2` when `prevent_overlap`,
        //    otherwise standard O(N log N) weighted Barnes-Hut or O(N^2) exact.
        let prevent_overlap = self.config.prevent_overlap;
        if prevent_overlap {
            if self.config.use_barnes_hut {
                self.overlap_tree
                    .build(&self.x_buf, &self.y_buf, &self.masses, &self.node_radii);
                self.overlap_tree.compute_repulsion(
                    &self.x_buf,
                    &self.y_buf,
                    &self.masses,
                    &self.node_radii,
                    self.config.theta,
                    self.config.scaling_ratio,
                    self.config.seed,
                    self.iter,
                    &mut self.fx,
                    &mut self.fy,
                );
            } else {
                self.fx.fill(0.0);
                self.fy.fill(0.0);
                let sr = self.config.scaling_ratio;
                let seed = self.config.seed;
                let iter = self.iter;
                for i in 0..n {
                    let xi = self.x_buf[i];
                    let yi = self.y_buf[i];
                    let mi = self.masses[i];
                    let ri = self.node_radii[i];
                    let mut acc_x = 0.0_f64;
                    let mut acc_y = 0.0_f64;
                    for j in (i + 1)..n {
                        let (fx_ij, fy_ij) = overlap_repulsion_pair(
                            xi - self.x_buf[j],
                            yi - self.y_buf[j],
                            mi,
                            self.masses[j],
                            ri + self.node_radii[j],
                            sr,
                            seed,
                            iter,
                            i,
                            j,
                        );
                        acc_x += fx_ij;
                        acc_y += fy_ij;
                        self.fx[j] -= fx_ij;
                        self.fy[j] -= fy_ij;
                    }
                    self.fx[i] += acc_x;
                    self.fy[i] += acc_y;
                }
            }
        } else if self.config.use_barnes_hut {
            self.quadtree
                .build_weighted_xy(&self.x_buf, &self.y_buf, &self.masses)?;
            self.quadtree.compute_coulomb_repulsion(
                self.config.theta,
                COULOMB_EPS_SQ,
                &mut self.rep_x,
                &mut self.rep_y,
            )?;
            let sr = self.config.scaling_ratio;
            for i in 0..n {
                let scale = sr * self.masses[i];
                self.fx[i] = scale * self.rep_x[i];
                self.fy[i] = scale * self.rep_y[i];
            }
        } else {
            self.fx.fill(0.0);
            self.fy.fill(0.0);
            let sr = self.config.scaling_ratio;
            for i in 0..n {
                let xi = self.x_buf[i];
                let yi = self.y_buf[i];
                let mi_sr = sr * self.masses[i];
                let mut acc_x = 0.0_f64;
                let mut acc_y = 0.0_f64;
                for j in (i + 1)..n {
                    let dx = xi - self.x_buf[j];
                    let dy = yi - self.y_buf[j];
                    let denom = dx * dx + dy * dy + COULOMB_EPS_SQ;
                    let mult = (mi_sr * self.masses[j]) / denom;
                    let fx_ij = mult * dx;
                    let fy_ij = mult * dy;
                    acc_x += fx_ij;
                    acc_y += fy_ij;
                    self.fx[j] -= fx_ij;
                    self.fy[j] -= fy_ij;
                }
                self.fx[i] += acc_x;
                self.fy[i] += acc_y;
            }
        }

        // 2. Attraction: linear or LinLog along canonical undirected edges.
        //    When `prevent_overlap` is active, use border distance `d' = d - r_u - r_v` when
        //    `d' > 0` and disable attraction when `d' <= 0` (Jacomy 2014 Eqs. 7-9).
        let linlog = self.config.linlog_mode;
        for idx in 0..self.edges_u.len() {
            let u = self.edges_u[idx] as usize;
            let v = self.edges_v[idx] as usize;
            let w = self.edges_w[idx];
            if w == 0.0 {
                continue;
            }
            let dx = self.x_buf[v] - self.x_buf[u];
            let dy = self.y_buf[v] - self.y_buf[u];
            let factor = if prevent_overlap {
                let d = dx.hypot(dy);
                let d_prime = d - (self.node_radii[u] + self.node_radii[v]);
                if d_prime > 0.0 && d > 1e-12 {
                    if linlog {
                        w * d_prime.ln_1p() / d
                    } else {
                        w * d_prime / d
                    }
                } else {
                    0.0
                }
            } else if linlog {
                let d_sq = dx * dx + dy * dy;
                if d_sq > 1e-24 {
                    let d = d_sq.sqrt();
                    w * d.ln_1p() / d
                } else {
                    0.0
                }
            } else {
                w
            };
            if factor == 0.0 {
                continue;
            }
            let ax = factor * dx;
            let ay = factor * dy;
            self.fx[u] += ax;
            self.fy[u] += ay;
            self.fx[v] -= ax;
            self.fy[v] -= ay;
        }

        // 3. Gravity: standard (constant pull m_i * g) or strong (m_i * g * d).
        if self.config.gravity > 0.0 {
            let g = self.config.gravity;
            if self.config.strong_gravity {
                for i in 0..n {
                    let gf = g * self.masses[i];
                    self.fx[i] -= gf * self.x_buf[i];
                    self.fy[i] -= gf * self.y_buf[i];
                }
            } else {
                for i in 0..n {
                    let d = self.x_buf[i].hypot(self.y_buf[i]);
                    if d > 1e-12 {
                        let gf = (g * self.masses[i]) / d;
                        self.fx[i] -= gf * self.x_buf[i];
                        self.fy[i] -= gf * self.y_buf[i];
                    }
                }
            }
        }

        // 4. Adaptive speed from per-node swing and traction.
        let mut total_swing = 0.0_f64;
        let mut total_traction = 0.0_f64;
        for i in 0..n {
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

            let swg = (fxi - self.prev_fx[i]).hypot(fyi - self.prev_fy[i]);
            let tra = 0.5 * (fxi + self.prev_fx[i]).hypot(fyi + self.prev_fy[i]);
            self.swing_buf[i] = swg;
            let m = self.masses[i];
            total_swing += m * swg;
            total_traction += m * tra;
        }

        let jt = self.config.jitter_tolerance;
        let max_speed = 10.0 * jt.max(1.0);
        let target_speed = if total_swing > 1e-12 {
            (jt * jt * total_traction / total_swing).clamp(MIN_FA2_SPEED, max_speed)
        } else {
            (self.global_speed * 1.5).min(max_speed)
        };
        if self.iter > 0 {
            let max_rise = 0.5 * self.global_speed;
            self.global_speed += (target_speed - self.global_speed).min(max_rise);
        } else {
            self.global_speed = target_speed;
        }
        self.global_speed = self.global_speed.clamp(MIN_FA2_SPEED, max_speed);

        // 5. Apply adaptive displacements back into layout (dividing speed by 10 when
        //    `prevent_overlap` is active per Jacomy 2014 p. 5).
        let speed = self.global_speed;
        let speed_scale = if prevent_overlap { 0.1 } else { 1.0 };
        let max_disp = self.max_displacement;
        let (xs_mut, ys_mut) = layout.xy_mut();
        for i in 0..n {
            let fxi = self.fx[i];
            let fyi = self.fy[i];
            let local_factor = speed_scale * (speed / (1.0 + (speed * self.swing_buf[i]).sqrt()));
            let mut dx = fxi * local_factor;
            let mut dy = fyi * local_factor;
            let disp = dx.hypot(dy);
            if disp > max_disp {
                let cap = max_disp / disp;
                dx *= cap;
                dy *= cap;
            }
            let nx = S::from_f64(self.x_buf[i] + dx);
            let ny = S::from_f64(self.y_buf[i] + dy);
            if nx.is_finite() {
                xs_mut[i] = nx;
            }
            if ny.is_finite() {
                ys_mut[i] = ny;
            }
            self.prev_fx[i] = fxi;
            self.prev_fy[i] = fyi;
        }

        self.iter += 1;
        self.status = if self.iter >= self.config.iterations {
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
        self.iter
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.config.iterations
    }
}
