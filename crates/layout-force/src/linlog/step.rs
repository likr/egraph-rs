use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};

use super::solver::LinLogSolver;

const COULOMB_EPS_SQ: f64 = 1e-6;
const MIN_LINLOG_SPEED: f64 = 1e-4;
const MAX_LINLOG_SPEED: f64 = 10.0;

impl<S: Scalar> IterativeSolver<S> for LinLogSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "LinLogSolver::step",
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

            if self.iter == 0 && (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 {
                let mut rng = DeterministicRng::from_seed_u64(self.config.seed);
                for i in 0..n {
                    self.x_buf[i] += 1e-4 * (rng.next_f64() - 0.5);
                    self.y_buf[i] += 1e-4 * (rng.next_f64() - 0.5);
                }
            }
        }

        let r_exp = self.config.repulsion_exponent;
        let is_coulomb = (r_exp - (-1.0)).abs() <= 1e-12;

        // 1. Repulsion: m_u * m_v * d^r along (p_u - p_v) / d.
        if self.config.use_barnes_hut {
            if is_coulomb {
                self.quadtree
                    .build_weighted_xy(&self.x_buf, &self.y_buf, &self.masses)?;
                self.quadtree.compute_coulomb_repulsion(
                    self.config.theta,
                    COULOMB_EPS_SQ,
                    &mut self.rep_x,
                    &mut self.rep_y,
                )?;
            } else {
                self.gen_quadtree
                    .build(&self.x_buf, &self.y_buf, &self.masses);
                self.gen_quadtree.compute_general_repulsion(
                    &self.x_buf,
                    &self.y_buf,
                    &self.masses,
                    self.config.theta,
                    r_exp,
                    COULOMB_EPS_SQ,
                    &mut self.rep_x,
                    &mut self.rep_y,
                );
            }
            for i in 0..n {
                let mi = self.masses[i];
                self.fx[i] = mi * self.rep_x[i];
                self.fy[i] = mi * self.rep_y[i];
            }
        } else {
            self.fx.fill(0.0);
            self.fy.fill(0.0);
            let pow_exp = 0.5 * (r_exp - 1.0);
            for i in 0..n {
                let xi = self.x_buf[i];
                let yi = self.y_buf[i];
                let mi = self.masses[i];
                let mut acc_x = 0.0_f64;
                let mut acc_y = 0.0_f64;
                for j in (i + 1)..n {
                    let dx = xi - self.x_buf[j];
                    let dy = yi - self.y_buf[j];
                    let d_sq_reg = dx * dx + dy * dy + COULOMB_EPS_SQ;
                    let mult = if is_coulomb {
                        (mi * self.masses[j]) / d_sq_reg
                    } else {
                        (mi * self.masses[j]) * d_sq_reg.powf(pow_exp)
                    };
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

        // 2. Attraction: w_{uv} * d^a along (p_v - p_u) / d (for a = 0: constant magnitude w_{uv}).
        let a_exp = self.config.attraction_exponent;
        let a_is_zero = a_exp.abs() <= 1e-12;
        let a_is_one = (a_exp - 1.0).abs() <= 1e-12;
        let a_hess_scale = a_exp.abs().max(1.0);
        self.att_hess.fill(0.0);

        for idx in 0..self.edges_u.len() {
            let u = self.edges_u[idx] as usize;
            let v = self.edges_v[idx] as usize;
            let w = self.edges_w[idx];
            if w == 0.0 {
                continue;
            }
            let dx = self.x_buf[v] - self.x_buf[u];
            let dy = self.y_buf[v] - self.y_buf[u];
            let d_sq = dx * dx + dy * dy;
            let (factor, hess) = if a_is_zero {
                if d_sq > 1e-24 {
                    let d = d_sq.sqrt();
                    (w / d, w / d)
                } else {
                    (0.0, 0.0)
                }
            } else if a_is_one {
                (w, w)
            } else if a_exp > 0.0 && d_sq <= 1e-24 {
                (0.0, 0.0)
            } else {
                let d_reg = (d_sq + 1e-24).sqrt();
                let f = w * d_reg.powf(a_exp - 1.0);
                (f, a_hess_scale * f)
            };

            let ax = factor * dx;
            let ay = factor * dy;
            self.fx[u] += ax;
            self.fy[u] += ay;
            self.fx[v] -= ax;
            self.fy[v] -= ay;
            self.att_hess[u] += hess;
            self.att_hess[v] += hess;
        }

        // 3. Mass-barycenter gravity (Noack 2007 §3.6).
        if self.config.gravity > 0.0 && self.total_mass > 0.0 {
            let mut bx = 0.0_f64;
            let mut by = 0.0_f64;
            for i in 0..n {
                bx += self.masses[i] * self.x_buf[i];
                by += self.masses[i] * self.y_buf[i];
            }
            bx /= self.total_mass;
            by /= self.total_mass;

            let g = self.config.gravity;
            for i in 0..n {
                let dx = bx - self.x_buf[i];
                let dy = by - self.y_buf[i];
                let d = dx.hypot(dy);
                if d > 1e-12 {
                    let factor = if a_is_zero {
                        (g * self.masses[i]) / d
                    } else if a_is_one {
                        g * self.masses[i]
                    } else {
                        g * self.masses[i] * d.powf(a_exp - 1.0)
                    };
                    self.fx[i] += factor * dx;
                    self.fy[i] += factor * dy;
                    self.att_hess[i] += a_hess_scale * factor;
                }
            }
        }

        // 4. Adaptive swing-traction speed with local curvature damping so constant-pull a=0
        //    and super-linear a>1 forces converge without oscillation or explosion.
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

        let target_speed = if total_swing > 1e-12 {
            (total_traction / total_swing).clamp(MIN_LINLOG_SPEED, MAX_LINLOG_SPEED)
        } else {
            (self.global_speed * 1.5).min(MAX_LINLOG_SPEED)
        };
        if self.iter > 0 {
            let max_rise = 0.5 * self.global_speed;
            self.global_speed += (target_speed - self.global_speed).min(max_rise);
        } else {
            self.global_speed = target_speed;
        }
        self.global_speed = self.global_speed.clamp(MIN_LINLOG_SPEED, MAX_LINLOG_SPEED);

        // 5. Apply displacements back into layout.
        let speed = self.global_speed;
        let max_disp = self.max_displacement;
        let (xs_mut, ys_mut) = layout.xy_mut();
        for i in 0..n {
            let fxi = self.fx[i];
            let fyi = self.fy[i];
            let local_factor =
                speed / (1.0 + (speed * self.swing_buf[i]).sqrt() + 0.5 * self.att_hess[i]);
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
