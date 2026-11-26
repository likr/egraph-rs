use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};

use super::config::{FieldRepulsionBackend, IbUmapConfig, TFdpConfig};
use super::solver::FieldLayoutSolver;

const MIN_DIST_SQ_CLAMP: f64 = 1e-8;
const REPULSIVE_DIST_SQ_EPS: f64 = 0.05;

impl<S: Scalar> FieldLayoutSolver<S> {
    pub(super) fn step_t_fdp(
        &mut self,
        layout: &mut Layout2d<S>,
        config: TFdpConfig,
        rep_scale: f64,
    ) -> CoreResult<()> {
        let n = self.num_nodes;
        if self.iteration == 0 {
            let (xs, ys) = layout.xy_mut();
            if xs.iter().all(|&v| v == S::ZERO) && ys.iter().all(|&v| v == S::ZERO) {
                for idx in 0..n {
                    xs[idx] = S::from_f64(self.rng.next_f64() * 2.0 - 1.0);
                    ys[idx] = S::from_f64(self.rng.next_f64() * 2.0 - 1.0);
                }
            }
        }

        // 1. Bounded Student-t short-range attractive forces along CSR edges.
        let inv_alpha_sq = 1.0 / (config.alpha * config.alpha);
        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                let xi = xs[i].to_f64();
                let yi = ys[i].to_f64();
                let start = self.indptr[i];
                let end = self.indptr[i + 1];
                let mut fx = 0.0_f64;
                let mut fy = 0.0_f64;
                for pos in start..end {
                    let j = self.indices[pos] as usize;
                    if j == i {
                        continue;
                    }
                    let dx = xi - xs[j].to_f64();
                    let dy = yi - ys[j].to_f64();
                    let d2 = dx * dx + dy * dy;
                    let mult = (self.edge_weights[pos] as f64) / (1.0 + d2 * inv_alpha_sq);
                    fx -= mult * dx;
                    fy -= mult * dy;
                }
                self.attr_x[i] = fx;
                self.attr_y[i] = fy;
            }
        }

        // 2. Global repulsive field evaluation.
        let gamma = config.gamma;
        match config.backend {
            FieldRepulsionBackend::IbFft => {
                let ibfft = self.ibfft.as_mut().ok_or_else(|| {
                    CoreError::numerical_breakdown("IbFft2d workspace missing for FieldLayout")
                })?;
                ibfft.prepare_xy(&mut self.fft_ws, layout.x(), layout.y())?;
                let soft = ibfft.grid_spec().map_or(0.0, |spec| spec.softening_sq);
                ibfft.evaluate_isotropic_kernel(
                    &mut self.fft_ws,
                    |d2| 1.0 / (1.0 + soft + d2).powf(gamma),
                    &mut self.phi_1,
                    &mut self.rep_x,
                    &mut self.rep_y,
                )?;
            }
            FieldRepulsionBackend::BarnesHut => {
                if (gamma - 1.0).abs() <= 1e-12 {
                    self.quadtree.build(layout)?;
                    self.quadtree.compute_coulomb_repulsion(
                        S::from_f64(config.theta),
                        S::ONE,
                        &mut self.rep_x,
                        &mut self.rep_y,
                    )?;
                } else if (gamma - 2.0).abs() <= 1e-12 {
                    self.quadtree.build(layout)?;
                    let total_z = self.quadtree.compute_kl_repulsion(
                        S::from_f64(config.theta),
                        &mut self.rep_x,
                        &mut self.rep_y,
                    )?;
                    let z_f64 = total_z.to_f64();
                    for i in 0..n {
                        self.rep_x[i] = S::from_f64(self.rep_x[i].to_f64() * z_f64);
                        self.rep_y[i] = S::from_f64(self.rep_y[i].to_f64() * z_f64);
                    }
                } else {
                    self.field_quadtree.build_xy(layout.x(), layout.y())?;
                    self.field_quadtree.compute_isotropic_repulsion(
                        config.theta,
                        |d2| 1.0 / (1.0 + d2).powf(gamma),
                        &mut self.rep_x,
                        &mut self.rep_y,
                    );
                }
            }
            FieldRepulsionBackend::Exact => {
                let xs = layout.x();
                let ys = layout.y();
                for i in 0..n {
                    let xi = xs[i].to_f64();
                    let yi = ys[i].to_f64();
                    let mut fx = 0.0_f64;
                    let mut fy = 0.0_f64;
                    for j in 0..n {
                        if j == i {
                            continue;
                        }
                        let dx = xi - xs[j].to_f64();
                        let dy = yi - ys[j].to_f64();
                        let d2 = dx * dx + dy * dy;
                        let k = 1.0 / (1.0 + d2).powf(gamma);
                        fx += k * dx;
                        fy += k * dy;
                    }
                    self.rep_x[i] = S::from_f64(fx);
                    self.rep_y[i] = S::from_f64(fy);
                }
            }
        }

        // 3. Dynamic rho(t) weighting, velocity momentum, bounded step, and recentering.
        let rho = self.current_rho() * rep_scale;
        let step_lr =
            config.learning_rate * (1.0 - (self.iteration as f64) / (self.total_iterations as f64));
        let momentum = config.momentum;
        let max_step = 5.0 * config.alpha.max(1.0) * config.learning_rate.max(1.0);

        let mut mean_x = 0.0_f64;
        let mut mean_y = 0.0_f64;
        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                let xi = xs[i].to_f64();
                let yi = ys[i].to_f64();
                let raw_fx = self.attr_x[i] + rho * self.rep_x[i].to_f64();
                let raw_fy = self.attr_y[i] + rho * self.rep_y[i].to_f64();
                let fx = if raw_fx.is_finite() { raw_fx } else { 0.0 };
                let fy = if raw_fy.is_finite() { raw_fy } else { 0.0 };

                let mut vx = momentum * self.vel_x[i] + step_lr * fx;
                let mut vy = momentum * self.vel_y[i] + step_lr * fy;
                if !vx.is_finite() {
                    vx = 0.0;
                }
                if !vy.is_finite() {
                    vy = 0.0;
                }
                let v_norm = vx.hypot(vy);
                if v_norm > max_step && v_norm > 0.0 {
                    let scale = max_step / v_norm;
                    vx *= scale;
                    vy *= scale;
                }
                self.vel_x[i] = vx;
                self.vel_y[i] = vy;

                let cand_x = xi + vx;
                let cand_y = yi + vy;
                let nx_val = if cand_x.is_finite() { cand_x } else { xi };
                let ny_val = if cand_y.is_finite() { cand_y } else { yi };
                self.next_x[i] = nx_val;
                self.next_y[i] = ny_val;
                mean_x += nx_val;
                mean_y += ny_val;
            }
        }

        let inv_n = 1.0 / (n as f64);
        mean_x *= inv_n;
        mean_y *= inv_n;
        let shift_x = if mean_x.is_finite() { mean_x } else { 0.0 };
        let shift_y = if mean_y.is_finite() { mean_y } else { 0.0 };

        let (xs_mut, ys_mut) = layout.xy_mut();
        for i in 0..n {
            let cx = S::from_f64(self.next_x[i] - shift_x);
            let cy = S::from_f64(self.next_y[i] - shift_y);
            if cx.is_finite() {
                xs_mut[i] = cx;
            }
            if cy.is_finite() {
                ys_mut[i] = cy;
            }
        }
        Ok(())
    }

    pub(super) fn step_ib_umap(
        &mut self,
        layout: &mut Layout2d<S>,
        config: IbUmapConfig,
        a: f64,
        b: f64,
    ) -> CoreResult<()> {
        let n = self.num_nodes;
        if self.iteration == 0 {
            let (xs, ys) = layout.xy_mut();
            if xs.iter().all(|&v| v == S::ZERO) && ys.iter().all(|&v| v == S::ZERO) {
                for idx in 0..n {
                    xs[idx] = S::from_f64(self.rng.next_f64() * 20.0 - 10.0);
                    ys[idx] = S::from_f64(self.rng.next_f64() * 20.0 - 10.0);
                }
            }
        }

        // 1. Attractive UMAP forces along fuzzy simplicial set edges.
        let two_ab = 2.0 * a * b;
        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                let xi = xs[i].to_f64();
                let yi = ys[i].to_f64();
                let start = self.indptr[i];
                let end = self.indptr[i + 1];
                let mut fx = 0.0_f64;
                let mut fy = 0.0_f64;
                for pos in start..end {
                    let j = self.indices[pos] as usize;
                    if j == i {
                        continue;
                    }
                    let p_ij = self.edge_weights[pos] as f64;
                    if p_ij <= 0.0 {
                        continue;
                    }
                    let dx = xi - xs[j].to_f64();
                    let dy = yi - ys[j].to_f64();
                    let d2 = (dx * dx + dy * dy).max(MIN_DIST_SQ_CLAMP);
                    let d_2b = d2.powf(b);
                    let d_pow = d_2b / d2;
                    let coeff = -two_ab * p_ij * d_pow / (1.0 + a * d_2b);
                    if coeff.is_finite() {
                        fx += coeff * dx;
                        fy += coeff * dy;
                    }
                }
                self.attr_x[i] = fx;
                self.attr_y[i] = fy;
            }
        }

        // 2. UMAP repulsive field evaluation: 2 b c_K / ((eps + d^2) (1 + a (d^2)^b)).
        let two_b_ck = 2.0 * b * config.c_k;
        match config.backend {
            FieldRepulsionBackend::IbFft => {
                let ibfft = self.ibfft.as_mut().ok_or_else(|| {
                    CoreError::numerical_breakdown("IbFft2d workspace missing for FieldLayout")
                })?;
                ibfft.prepare_xy(&mut self.fft_ws, layout.x(), layout.y())?;
                let soft = ibfft.grid_spec().map_or(0.0, |spec| spec.softening_sq);
                let eps_eff = REPULSIVE_DIST_SQ_EPS + soft;
                ibfft.evaluate_isotropic_kernel(
                    &mut self.fft_ws,
                    |d2| {
                        let d2_pos = d2.max(0.0);
                        two_b_ck / ((eps_eff + d2_pos) * (1.0 + a * d2_pos.powf(b)))
                    },
                    &mut self.phi_1,
                    &mut self.rep_x,
                    &mut self.rep_y,
                )?;
            }
            FieldRepulsionBackend::BarnesHut => {
                self.field_quadtree.build_xy(layout.x(), layout.y())?;
                self.field_quadtree.compute_isotropic_repulsion(
                    config.theta,
                    |d2| {
                        let d2_pos = d2.max(0.0);
                        two_b_ck / ((REPULSIVE_DIST_SQ_EPS + d2_pos) * (1.0 + a * d2_pos.powf(b)))
                    },
                    &mut self.rep_x,
                    &mut self.rep_y,
                );
            }
            FieldRepulsionBackend::Exact => {
                let xs = layout.x();
                let ys = layout.y();
                for i in 0..n {
                    let xi = xs[i].to_f64();
                    let yi = ys[i].to_f64();
                    let mut fx = 0.0_f64;
                    let mut fy = 0.0_f64;
                    for j in 0..n {
                        if j == i {
                            continue;
                        }
                        let dx = xi - xs[j].to_f64();
                        let dy = yi - ys[j].to_f64();
                        let d2 = dx * dx + dy * dy;
                        let k = two_b_ck / ((REPULSIVE_DIST_SQ_EPS + d2) * (1.0 + a * d2.powf(b)));
                        fx += k * dx;
                        fy += k * dy;
                    }
                    self.rep_x[i] = S::from_f64(fx);
                    self.rep_y[i] = S::from_f64(fy);
                }
            }
        }

        // 3. Hub damping eta_i, per-node step clipping tau_t = tau_scale * alpha_t, and recenter.
        let alpha_t =
            config.learning_rate * (1.0 - (self.iteration as f64) / (self.total_iterations as f64));
        let tau_t = config.tau_scale * alpha_t;
        let inv_n = 1.0 / (n as f64);

        let mut mean_x = 0.0_f64;
        let mut mean_y = 0.0_f64;
        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                let xi = xs[i].to_f64();
                let yi = ys[i].to_f64();
                let raw_fx = self.attr_x[i] + self.rep_x[i].to_f64() * inv_n;
                let raw_fy = self.attr_y[i] + self.rep_y[i].to_f64() * inv_n;
                let fx = if raw_fx.is_finite() { raw_fx } else { 0.0 };
                let fy = if raw_fy.is_finite() { raw_fy } else { 0.0 };

                let step_scale = alpha_t * self.hub_eta[i];
                let mut dx = step_scale * fx;
                let mut dy = step_scale * fy;
                if !dx.is_finite() {
                    dx = 0.0;
                }
                if !dy.is_finite() {
                    dy = 0.0;
                }
                let norm = dx.hypot(dy);
                if norm > tau_t && norm > 0.0 {
                    let clip = tau_t / norm;
                    dx *= clip;
                    dy *= clip;
                }

                let cand_x = xi + dx;
                let cand_y = yi + dy;
                let nx_val = if cand_x.is_finite() { cand_x } else { xi };
                let ny_val = if cand_y.is_finite() { cand_y } else { yi };
                self.next_x[i] = nx_val;
                self.next_y[i] = ny_val;
                mean_x += nx_val;
                mean_y += ny_val;
            }
        }

        mean_x *= inv_n;
        mean_y *= inv_n;
        let shift_x = if mean_x.is_finite() { mean_x } else { 0.0 };
        let shift_y = if mean_y.is_finite() { mean_y } else { 0.0 };

        let (xs_mut, ys_mut) = layout.xy_mut();
        for i in 0..n {
            let cx = S::from_f64(self.next_x[i] - shift_x);
            let cy = S::from_f64(self.next_y[i] - shift_y);
            if cx.is_finite() {
                xs_mut[i] = cx;
            }
            if cy.is_finite() {
                ys_mut[i] = cy;
            }
        }
        Ok(())
    }
}
