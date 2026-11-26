use egraph_core::{CoreError, CoreResult, IterativeSolver, Layout2d, Scalar, SolverStatus};
use egraph_spatial_fft::{FitField2d, FitFieldConfig};

use super::config::{MIN_LAMBDA_SUM, TsNetBackend, compute_stage_lr};
use super::solver::TsNetSolver;

const MIN_TOTAL_Z: f64 = 1e-12;
const EXACT_ENTROPY_MIN_DIST: f64 = 1e-12;

impl<S: Scalar> TsNetSolver<S> {
    fn enter_current_stage(&mut self) -> CoreResult<()> {
        let stage = self.stages[self.stage_idx];
        let base_lr = self.config.resolve_learning_rate(self.num_nodes);
        let stage_lr = compute_stage_lr(&stage, self.num_nodes, base_lr, self.config.momentum);
        if stage_lr < self.prev_stage_lr {
            let vel_scale = stage_lr / self.prev_stage_lr;
            for v in &mut self.vel_x {
                *v *= vel_scale;
            }
            for v in &mut self.vel_y {
                *v *= vel_scale;
            }
        }
        self.prev_stage_lr = stage_lr;
        self.active_stage_lr = stage_lr;

        let need_entropy = stage.lambda_r > 0.0;
        if matches!(
            self.resolved_backend,
            TsNetBackend::FftInterpolation | TsNetBackend::LinearFft
        ) {
            let needs_rebuild = match &self.fit_field {
                Some(ff) => ff.config().compute_entropy != need_entropy,
                None => true,
            };
            if needs_rebuild {
                self.fit_field = Some(FitField2d::new(FitFieldConfig {
                    ibfft: self.config.effective_ibfft_for(self.resolved_backend),
                    epsilon_r: self.config.epsilon_r,
                    compute_entropy: need_entropy,
                })?);
            }
        }

        self.stage_entered = true;
        Ok(())
    }

    fn compute_attractions(&mut self, xs: &[S], ys: &[S], exaggeration: f64) {
        let n = self.num_nodes;
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let start = self.indptr[i] as usize;
            let end = self.indptr[i + 1] as usize;
            let mut ax = 0.0_f64;
            let mut ay = 0.0_f64;
            for pos in start..end {
                let j = self.indices[pos] as usize;
                if j == i {
                    continue;
                }
                let dx = xi - xs[j].to_f64();
                let dy = yi - ys[j].to_f64();
                let dist_sq = dx * dx + dy * dy;
                let q_unnorm = 1.0 / (1.0 + dist_sq);
                let mult = (self.p_vals[pos] * exaggeration) * q_unnorm;
                ax += mult * dx;
                ay += mult * dy;
            }
            self.attr_x[i] = ax;
            self.attr_y[i] = ay;
        }
    }

    fn compute_exact_repulsion(&mut self, xs: &[S], ys: &[S], compute_entropy: bool) {
        let n = self.num_nodes;
        let eps_r = self.config.epsilon_r;
        let mut total_z = 0.0_f64;
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let mut z_i = 0.0_f64;
            let mut rfx = 0.0_f64;
            let mut rfy = 0.0_f64;
            let mut efx = 0.0_f64;
            let mut efy = 0.0_f64;

            for j in 0..n {
                if j == i {
                    continue;
                }
                let dx = xi - xs[j].to_f64();
                let dy = yi - ys[j].to_f64();
                let dist_sq = dx * dx + dy * dy;
                let w = 1.0 / (1.0 + dist_sq);
                z_i += w;
                let w2 = w * w;
                rfx += w2 * dx;
                rfy += w2 * dy;
                if compute_entropy {
                    let dist = dist_sq.sqrt();
                    if dist > EXACT_ENTROPY_MIN_DIST {
                        let mult = 1.0 / (dist * (dist + eps_r));
                        efx += mult * dx;
                        efy += mult * dy;
                    }
                }
            }
            total_z += z_i;
            self.rep_x[i] = S::from_f64(rfx);
            self.rep_y[i] = S::from_f64(rfy);
            if compute_entropy {
                self.ent_x[i] = S::from_f64(efx);
                self.ent_y[i] = S::from_f64(efy);
            }
        }

        let denom = total_z.max(MIN_TOTAL_Z);
        for i in 0..n {
            self.rep_x[i] = S::from_f64(self.rep_x[i].to_f64() / denom);
            self.rep_y[i] = S::from_f64(self.rep_y[i].to_f64() / denom);
        }
    }

    fn compute_barnes_hut_repulsion(
        &mut self,
        layout: &Layout2d<S>,
        compute_entropy: bool,
    ) -> CoreResult<()> {
        self.quadtree.build(layout)?;
        let theta_s = S::from_f64(self.config.theta);
        self.quadtree
            .compute_kl_repulsion(theta_s, &mut self.rep_x, &mut self.rep_y)?;
        if compute_entropy {
            let eps_s = S::from_f64(self.config.epsilon_r).max(S::MIN_POSITIVE);
            self.quadtree.compute_entropy_repulsion(
                theta_s,
                eps_s,
                &mut self.ent_x,
                &mut self.ent_y,
            )?;
        }
        Ok(())
    }

    pub(super) fn advance_stage_counters(&mut self) {
        self.iteration += 1;
        self.stage_iter += 1;
        if self.stage_iter < self.stages[self.stage_idx].iterations {
            self.status = SolverStatus::Running;
            return;
        }

        self.stage_iter = 0;
        self.stage_entered = false;
        if let Some(rel_next) = self.stages[self.stage_idx + 1..]
            .iter()
            .position(|s| s.iterations > 0)
        {
            self.stage_idx += 1 + rel_next;
            self.status = SolverStatus::Running;
        } else {
            self.status = SolverStatus::Completed;
        }
    }
}

impl<S: Scalar> IterativeSolver<S> for TsNetSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "TsNetSolver::step",
            ));
        }
        layout.validate_finite()?;
        self.active_layout.lock_2d("TsNetSolver")?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        if self.num_nodes < 2 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        if !self.stage_entered {
            self.enter_current_stage()?;
        }

        let stage = self.stages[self.stage_idx];
        let compute_entropy = stage.lambda_r > 0.0;

        self.compute_attractions(layout.x(), layout.y(), stage.exaggeration);

        match self.resolved_backend {
            TsNetBackend::Exact => {
                self.compute_exact_repulsion(layout.x(), layout.y(), compute_entropy);
            }
            TsNetBackend::BarnesHut => {
                self.compute_barnes_hut_repulsion(layout, compute_entropy)?;
            }
            TsNetBackend::FftInterpolation | TsNetBackend::LinearFft => {
                let fit_field = self.fit_field.as_mut().ok_or_else(|| {
                    CoreError::numerical_breakdown("FitField2d workspace missing")
                })?;
                fit_field.evaluate(layout, &mut self.fit_out)?;
                self.last_fft_stats = Some(self.fit_out.stats);
                self.total_fft_2d_calls += self.fit_out.stats.total_fft_2d_calls;
            }
            TsNetBackend::Auto => {
                return Err(CoreError::numerical_breakdown(
                    "TsNetBackend::Auto was not resolved at construction",
                ));
            }
        }

        let n = self.num_nodes;
        let n_f64 = n as f64;
        let n_sq_f64 = n_f64 * n_f64;
        let lambda_sum = (stage.lambda_kl + stage.lambda_c + stage.lambda_r).max(MIN_LAMBDA_SUM);
        let stage_lr = self.active_stage_lr;
        let momentum = self.config.momentum;
        let use_jacobi = self.config.jacobi_precondition;

        let use_fft_buffers = matches!(
            self.resolved_backend,
            TsNetBackend::FftInterpolation | TsNetBackend::LinearFft
        );

        let mut mean_x = 0.0_f64;
        let mut mean_y = 0.0_f64;
        {
            let xs = layout.x();
            let ys = layout.y();
            let (rep_x_slice, rep_y_slice) = if use_fft_buffers {
                (
                    self.fit_out.kl_repulsion_x.as_slice(),
                    self.fit_out.kl_repulsion_y.as_slice(),
                )
            } else {
                (self.rep_x.as_slice(), self.rep_y.as_slice())
            };
            let (ent_x_slice, ent_y_slice) = if compute_entropy {
                if use_fft_buffers {
                    (
                        self.fit_out.entropy_repulsion_x.as_slice(),
                        self.fit_out.entropy_repulsion_y.as_slice(),
                    )
                } else {
                    (self.ent_x.as_slice(), self.ent_y.as_slice())
                }
            } else {
                (&[][..], &[][..])
            };

            for i in 0..n {
                let xi = xs[i].to_f64();
                let yi = ys[i].to_f64();

                let grad_kl_x = 4.0 * (self.attr_x[i] - rep_x_slice[i].to_f64());
                let grad_kl_y = 4.0 * (self.attr_y[i] - rep_y_slice[i].to_f64());

                let grad_c_x = xi / n_f64;
                let grad_c_y = yi / n_f64;

                let (grad_r_x, grad_r_y) = if compute_entropy {
                    (
                        -ent_x_slice[i].to_f64() / n_sq_f64,
                        -ent_y_slice[i].to_f64() / n_sq_f64,
                    )
                } else {
                    (0.0, 0.0)
                };

                let raw_gx = (stage.lambda_kl * grad_kl_x
                    + stage.lambda_c * grad_c_x
                    + stage.lambda_r * grad_r_x)
                    / lambda_sum;
                let raw_gy = (stage.lambda_kl * grad_kl_y
                    + stage.lambda_c * grad_c_y
                    + stage.lambda_r * grad_r_y)
                    / lambda_sum;
                let gx = if raw_gx.is_finite() { raw_gx } else { 0.0 };
                let gy = if raw_gy.is_finite() { raw_gy } else { 0.0 };

                let lr_i = if use_jacobi {
                    stage_lr * self.inv_jacobi[i]
                } else {
                    stage_lr
                };
                let mut vx = momentum * self.vel_x[i] - lr_i * gx;
                let mut vy = momentum * self.vel_y[i] - lr_i * gy;
                if !vx.is_finite() {
                    vx = 0.0;
                }
                if !vy.is_finite() {
                    vy = 0.0;
                }
                self.vel_x[i] = vx;
                self.vel_y[i] = vy;

                let cand_x = xi + vx;
                let cand_y = yi + vy;
                let new_x = if cand_x.is_finite() { cand_x } else { xi };
                let new_y = if cand_y.is_finite() { cand_y } else { yi };

                // Reuse `attr_x` / `attr_y` as temporary `f64` coordinate buffers before recentering.
                self.attr_x[i] = new_x;
                self.attr_y[i] = new_y;
                mean_x += new_x;
                mean_y += new_y;
            }
        }

        mean_x /= n_f64;
        mean_y /= n_f64;
        let shift_x = if mean_x.is_finite() { mean_x } else { 0.0 };
        let shift_y = if mean_y.is_finite() { mean_y } else { 0.0 };

        let (xs_mut, ys_mut) = layout.xy_mut();
        for i in 0..n {
            let cx = S::from_f64(self.attr_x[i] - shift_x);
            let cy = S::from_f64(self.attr_y[i] - shift_y);
            if cx.is_finite() {
                xs_mut[i] = cx;
            }
            if cy.is_finite() {
                ys_mut[i] = cy;
            }
        }

        self.advance_stage_counters();
        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iteration
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.total_iterations
    }
}
