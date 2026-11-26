use egraph_core::{CoreError, CoreResult, IterativeSolver, LayoutNd, Scalar, SolverStatus};
use egraph_spatial_fft::{FitField2d, FitFieldConfig};
use egraph_spatial_trees::ArenaOctree3d;

use super::config::{MIN_LAMBDA_SUM, TsNetBackend, TsNetStage, compute_stage_lr};
use super::solver::TsNetSolver;

const MIN_TOTAL_Z: f64 = 1e-12;
const EXACT_ENTROPY_MIN_DIST: f64 = 1e-12;

/// Lazily allocated N-D scratch buffers for [`TsNetSolver`] on [`LayoutNd`].
#[derive(Debug, Clone)]
pub(super) struct TsNetNdScratch<S: Scalar> {
    pub(super) attr: Vec<f64>,
    pub(super) vel: Vec<f64>,
    pub(super) rep: Vec<S>,
    pub(super) ent: Vec<S>,
    pub(super) mean: Vec<f64>,
    pub(super) rf_buf: Vec<f64>,
    pub(super) ef_buf: Vec<f64>,
    pub(super) octree: ArenaOctree3d<S>,
    pub(super) x_2d: Vec<S>,
    pub(super) y_2d: Vec<S>,
}

impl<S: Scalar> TsNetSolver<S> {
    fn resolve_nd_backend(&self, dims: usize) -> CoreResult<TsNetBackend> {
        match self.config.backend {
            TsNetBackend::FftInterpolation | TsNetBackend::LinearFft if dims == 2 => {
                Ok(self.resolved_backend)
            }
            TsNetBackend::FftInterpolation | TsNetBackend::LinearFft => {
                Err(CoreError::unsupported(
                    "FFT interpolation backend is only supported in 2D; use Exact or BarnesHut for 3D/N-D LayoutNd",
                ))
            }
            TsNetBackend::BarnesHut if dims == 2 || dims == 3 => Ok(TsNetBackend::BarnesHut),
            TsNetBackend::BarnesHut => Err(CoreError::unsupported(
                "BarnesHut backend is only supported in 2D and 3D; use Exact for other dimensions",
            )),
            TsNetBackend::Exact => Ok(TsNetBackend::Exact),
            TsNetBackend::Auto => match dims {
                2 => Ok(self.resolved_backend),
                // Auto resolves N > 256 to LinearFft at construction; route 3D LayoutNd to ArenaOctree3d.
                3 if self.num_nodes > 256 => Ok(TsNetBackend::BarnesHut),
                _ => Ok(TsNetBackend::Exact),
            },
        }
    }

    fn ensure_nd_scratch(
        &mut self,
        dims: usize,
        effective_backend: TsNetBackend,
    ) -> CoreResult<()> {
        if self.nd_scratch.is_some() {
            return Ok(());
        }
        let n = self.num_nodes;
        let total = n
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("LayoutNd size overflow"))?;
        let any_entropy = self
            .stages
            .iter()
            .any(|s| s.iterations > 0 && s.lambda_r > 0.0);
        let need_nd_rep = effective_backend == TsNetBackend::Exact
            || (effective_backend == TsNetBackend::BarnesHut && dims == 3);
        let rep_len = if need_nd_rep { total } else { 0 };
        let ent_len = if need_nd_rep && any_entropy { total } else { 0 };
        let xy_len = if dims == 2 && effective_backend != TsNetBackend::Exact {
            n
        } else {
            0
        };
        let exact_dims = if effective_backend == TsNetBackend::Exact {
            dims
        } else {
            0
        };

        self.nd_scratch = Some(TsNetNdScratch {
            attr: vec![0.0; total],
            vel: vec![0.0; total],
            rep: vec![S::ZERO; rep_len],
            ent: vec![S::ZERO; ent_len],
            mean: vec![0.0; dims],
            rf_buf: vec![0.0; exact_dims],
            ef_buf: vec![0.0; if any_entropy { exact_dims } else { 0 }],
            octree: ArenaOctree3d::new(),
            x_2d: vec![S::ZERO; xy_len],
            y_2d: vec![S::ZERO; xy_len],
        });
        Ok(())
    }

    fn enter_current_stage_nd(&mut self, effective_backend: TsNetBackend) -> CoreResult<()> {
        let stage = self.stages[self.stage_idx];
        let base_lr = self.config.resolve_learning_rate(self.num_nodes);
        let stage_lr = compute_stage_lr(&stage, self.num_nodes, base_lr, self.config.momentum);
        if stage_lr < self.prev_stage_lr {
            let vel_scale = stage_lr / self.prev_stage_lr;
            if let Some(scratch) = &mut self.nd_scratch {
                for v in &mut scratch.vel {
                    *v *= vel_scale;
                }
            }
        }
        self.prev_stage_lr = stage_lr;
        self.active_stage_lr = stage_lr;

        let need_entropy = stage.lambda_r > 0.0;
        if matches!(
            effective_backend,
            TsNetBackend::FftInterpolation | TsNetBackend::LinearFft
        ) {
            let needs_rebuild = match &self.fit_field {
                Some(ff) => ff.config().compute_entropy != need_entropy,
                None => true,
            };
            if needs_rebuild {
                self.fit_field = Some(FitField2d::new(FitFieldConfig {
                    ibfft: self.config.effective_ibfft_for(effective_backend),
                    epsilon_r: self.config.epsilon_r,
                    compute_entropy: need_entropy,
                })?);
            }
        }

        self.stage_entered = true;
        Ok(())
    }

    fn compute_attractions_nd(&mut self, coords: &[S], dims: usize, exaggeration: f64) {
        let Some(scratch) = &mut self.nd_scratch else {
            return;
        };
        let n = self.num_nodes;
        for i in 0..n {
            let i_base = i * dims;
            scratch.attr[i_base..i_base + dims].fill(0.0);
            let start = self.indptr[i] as usize;
            let end = self.indptr[i + 1] as usize;
            for pos in start..end {
                let j = self.indices[pos] as usize;
                if j == i {
                    continue;
                }
                let j_base = j * dims;
                let mut dist_sq = 0.0_f64;
                for d in 0..dims {
                    let dx = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                    dist_sq += dx * dx;
                }
                let q_unnorm = 1.0 / (1.0 + dist_sq);
                let mult = (self.p_vals[pos] * exaggeration) * q_unnorm;
                for d in 0..dims {
                    let dx = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                    scratch.attr[i_base + d] += mult * dx;
                }
            }
        }
    }

    fn compute_exact_repulsion_nd(&mut self, coords: &[S], dims: usize, compute_entropy: bool) {
        let Some(scratch) = &mut self.nd_scratch else {
            return;
        };
        let n = self.num_nodes;
        let eps_r = self.config.epsilon_r;
        let mut total_z = 0.0_f64;

        for i in 0..n {
            let i_base = i * dims;
            let mut z_i = 0.0_f64;
            scratch.rf_buf.fill(0.0);
            if compute_entropy {
                scratch.ef_buf.fill(0.0);
            }

            for j in 0..n {
                if j == i {
                    continue;
                }
                let j_base = j * dims;
                let mut dist_sq = 0.0_f64;
                for d in 0..dims {
                    let dx = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                    dist_sq += dx * dx;
                }
                let w = 1.0 / (1.0 + dist_sq);
                z_i += w;
                let w2 = w * w;
                let ent_mult = if compute_entropy {
                    let dist = dist_sq.sqrt();
                    if dist > EXACT_ENTROPY_MIN_DIST {
                        Some(1.0 / (dist * (dist + eps_r)))
                    } else {
                        None
                    }
                } else {
                    None
                };

                for d in 0..dims {
                    let dx = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                    scratch.rf_buf[d] += w2 * dx;
                    if let Some(mult) = ent_mult {
                        scratch.ef_buf[d] += mult * dx;
                    }
                }
            }

            total_z += z_i;
            for d in 0..dims {
                scratch.rep[i_base + d] = S::from_f64(scratch.rf_buf[d]);
                if compute_entropy {
                    scratch.ent[i_base + d] = S::from_f64(scratch.ef_buf[d]);
                }
            }
        }

        let denom = total_z.max(MIN_TOTAL_Z);
        for v in &mut scratch.rep {
            *v = S::from_f64(v.to_f64() / denom);
        }
    }

    fn compute_barnes_hut_3d_repulsion(
        &mut self,
        layout: &LayoutNd<S>,
        compute_entropy: bool,
    ) -> CoreResult<()> {
        let scratch = self
            .nd_scratch
            .as_mut()
            .ok_or_else(|| CoreError::numerical_breakdown("TsNetNdScratch workspace missing"))?;
        scratch.octree.build(layout)?;
        let theta_s = S::from_f64(self.config.theta);
        scratch
            .octree
            .compute_kl_repulsion_flat(theta_s, &mut scratch.rep)?;
        if compute_entropy {
            let eps_s = S::from_f64(self.config.epsilon_r).max(S::MIN_POSITIVE);
            scratch
                .octree
                .compute_entropy_repulsion_flat(theta_s, eps_s, &mut scratch.ent)?;
        }
        Ok(())
    }

    fn compute_2d_from_nd(
        &mut self,
        coords: &[S],
        effective_backend: TsNetBackend,
        compute_entropy: bool,
    ) -> CoreResult<()> {
        let n = self.num_nodes;
        let scratch = self
            .nd_scratch
            .as_mut()
            .ok_or_else(|| CoreError::numerical_breakdown("TsNetNdScratch workspace missing"))?;
        for i in 0..n {
            scratch.x_2d[i] = coords[i * 2];
            scratch.y_2d[i] = coords[i * 2 + 1];
        }
        if effective_backend == TsNetBackend::BarnesHut {
            self.quadtree.build_xy(&scratch.x_2d, &scratch.y_2d)?;
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
        } else {
            let fit_field = self
                .fit_field
                .as_mut()
                .ok_or_else(|| CoreError::numerical_breakdown("FitField2d workspace missing"))?;
            fit_field.evaluate_xy(&scratch.x_2d, &scratch.y_2d, &mut self.fit_out)?;
            self.last_fft_stats = Some(self.fit_out.stats);
            self.total_fft_2d_calls += self.fit_out.stats.total_fft_2d_calls;
        }
        Ok(())
    }

    fn apply_nd_update(
        &mut self,
        layout: &mut LayoutNd<S>,
        dims: usize,
        stage: TsNetStage,
        effective_backend: TsNetBackend,
    ) -> CoreResult<()> {
        let n = self.num_nodes;
        let n_f64 = n as f64;
        let n_sq_f64 = n_f64 * n_f64;
        let lambda_sum = (stage.lambda_kl + stage.lambda_c + stage.lambda_r).max(MIN_LAMBDA_SUM);
        let stage_lr = self.active_stage_lr;
        let momentum = self.config.momentum;
        let use_jacobi = self.config.jacobi_precondition;
        let compute_entropy = stage.lambda_r > 0.0;
        let use_nd_flat_rep = effective_backend == TsNetBackend::Exact
            || (effective_backend == TsNetBackend::BarnesHut && dims == 3);
        let use_fft_buffers = matches!(
            effective_backend,
            TsNetBackend::FftInterpolation | TsNetBackend::LinearFft
        );

        let (rep_x_2d, rep_y_2d, ent_x_2d, ent_y_2d) = if use_fft_buffers {
            (
                self.fit_out.kl_repulsion_x.as_slice(),
                self.fit_out.kl_repulsion_y.as_slice(),
                self.fit_out.entropy_repulsion_x.as_slice(),
                self.fit_out.entropy_repulsion_y.as_slice(),
            )
        } else {
            (
                self.rep_x.as_slice(),
                self.rep_y.as_slice(),
                self.ent_x.as_slice(),
                self.ent_y.as_slice(),
            )
        };

        let scratch = self
            .nd_scratch
            .as_mut()
            .ok_or_else(|| CoreError::numerical_breakdown("TsNetNdScratch workspace missing"))?;
        scratch.mean.fill(0.0);

        let coords = layout.coords_mut();
        for i in 0..n {
            let lr_i = if use_jacobi {
                stage_lr * self.inv_jacobi[i]
            } else {
                stage_lr
            };
            let i_base = i * dims;
            for d in 0..dims {
                let idx = i_base + d;
                let x_d = coords[idx].to_f64();
                let rep_d = if use_nd_flat_rep {
                    scratch.rep[idx]
                } else if d == 0 {
                    rep_x_2d[i]
                } else {
                    rep_y_2d[i]
                }
                .to_f64();

                let grad_kl = 4.0 * (scratch.attr[idx] - rep_d);
                let grad_c = x_d / n_f64;
                let grad_r = if compute_entropy {
                    let ent_d = if use_nd_flat_rep {
                        scratch.ent[idx]
                    } else if d == 0 {
                        ent_x_2d[i]
                    } else {
                        ent_y_2d[i]
                    }
                    .to_f64();
                    -ent_d / n_sq_f64
                } else {
                    0.0
                };

                let raw_g =
                    (stage.lambda_kl * grad_kl + stage.lambda_c * grad_c + stage.lambda_r * grad_r)
                        / lambda_sum;
                let g = if raw_g.is_finite() { raw_g } else { 0.0 };

                let mut v = momentum * scratch.vel[idx] - lr_i * g;
                if !v.is_finite() {
                    v = 0.0;
                }
                scratch.vel[idx] = v;

                let cand = x_d + v;
                let new_val = if cand.is_finite() { cand } else { x_d };
                scratch.attr[idx] = new_val;
                scratch.mean[d] += new_val;
            }
        }

        for mean in &mut scratch.mean {
            let m = *mean / n_f64;
            *mean = if m.is_finite() { m } else { 0.0 };
        }

        for i in 0..n {
            let i_base = i * dims;
            for d in 0..dims {
                let idx = i_base + d;
                let centered = S::from_f64(scratch.attr[idx] - scratch.mean[d]);
                if centered.is_finite() {
                    coords[idx] = centered;
                }
            }
        }
        Ok(())
    }
}

impl<S: Scalar> IterativeSolver<S, LayoutNd<S>> for TsNetSolver<S> {
    fn step(&mut self, layout: &mut LayoutNd<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "TsNetSolver::step",
            ));
        }
        let dims = layout.dims();
        if dims == 0 {
            return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
        }
        layout.validate_finite()?;
        self.active_layout
            .lock_nd(dims, "TsNetSolver", "TsNetSolver::step::dims")?;

        let effective_backend = self.resolve_nd_backend(dims)?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        if self.num_nodes < 2 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        self.ensure_nd_scratch(dims, effective_backend)?;

        if !self.stage_entered {
            self.enter_current_stage_nd(effective_backend)?;
        }

        let stage = self.stages[self.stage_idx];
        let compute_entropy = stage.lambda_r > 0.0;

        self.compute_attractions_nd(layout.coords(), dims, stage.exaggeration);

        match effective_backend {
            TsNetBackend::Exact => {
                self.compute_exact_repulsion_nd(layout.coords(), dims, compute_entropy);
            }
            TsNetBackend::BarnesHut if dims == 3 => {
                self.compute_barnes_hut_3d_repulsion(layout, compute_entropy)?;
            }
            TsNetBackend::BarnesHut | TsNetBackend::FftInterpolation | TsNetBackend::LinearFft => {
                self.compute_2d_from_nd(layout.coords(), effective_backend, compute_entropy)?;
            }
            TsNetBackend::Auto => {
                return Err(CoreError::numerical_breakdown(
                    "TsNetBackend::Auto was not resolved for LayoutNd",
                ));
            }
        }

        self.apply_nd_update(layout, dims, stage, effective_backend)?;
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
