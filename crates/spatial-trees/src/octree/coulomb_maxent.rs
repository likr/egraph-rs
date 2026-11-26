use egraph_core::{CoreError, CoreResult, Scalar};

use super::{ArenaOctree3d, TRAVERSAL_STACK_CAP};

impl<S: Scalar> ArenaOctree3d<S> {
    fn eval_weighted_radial_raw<F>(
        &self,
        point_idx: usize,
        theta_sq: f64,
        context: &'static str,
        mut coeff: F,
    ) -> CoreResult<(f64, f64, f64)>
    where
        F: FnMut(f64, f64) -> f64,
    {
        let n = self.point_count();
        if point_idx >= n {
            return Err(CoreError::index_out_of_bounds(point_idx, n, context));
        }
        if !theta_sq.is_finite() || theta_sq < 0.0 {
            return Err(CoreError::invalid_input(
                "theta_sq must be finite and >= 0.0",
            ));
        }
        if self.nodes.is_empty() {
            return Ok((0.0, 0.0, 0.0));
        }

        let x = self.coords_x[point_idx];
        let y = self.coords_y[point_idx];
        let z = self.coords_z[point_idx];
        let own_leaf = self.leaf_of_point[point_idx];

        let mut stack = [0u32; TRAVERSAL_STACK_CAP];
        let mut sp = 1usize;
        stack[0] = 0;

        let mut fx = 0.0_f64;
        let mut fy = 0.0_f64;
        let mut fz = 0.0_f64;

        while sp > 0 {
            sp -= 1;
            let node_idx = stack[sp];
            let node = &self.nodes[node_idx as usize];
            if node.point_count == 0 || node.total_mass == 0.0 {
                continue;
            }

            if node.is_leaf {
                if node_idx == own_leaf {
                    let mut p = node.leaf_head;
                    while p != u32::MAX {
                        let other = p as usize;
                        if other != point_idx {
                            let dx = x - self.coords_x[other];
                            let dy = y - self.coords_y[other];
                            let dz = z - self.coords_z[other];
                            let dist_sq = dx * dx + dy * dy + dz * dz;
                            let mult = coeff(self.masses[other], dist_sq);
                            fx += mult * dx;
                            fy += mult * dy;
                            fz += mult * dz;
                        }
                        p = self.next_in_leaf[other];
                    }
                } else {
                    let dx = x - node.com[0];
                    let dy = y - node.com[1];
                    let dz = z - node.com[2];
                    let dist_sq = dx * dx + dy * dy + dz * dz;
                    let mult = coeff(node.total_mass, dist_sq);
                    fx += mult * dx;
                    fy += mult * dy;
                    fz += mult * dz;
                }
                continue;
            }

            let dx = x - node.com[0];
            let dy = y - node.com[1];
            let dz = z - node.com[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let full_span = 2.0 * node.half_span;
            if (full_span * full_span) < theta_sq * dist_sq {
                let mult = coeff(node.total_mass, dist_sq);
                fx += mult * dx;
                fy += mult * dy;
                fz += mult * dz;
            } else {
                for &child in &node.children {
                    if child != u32::MAX && self.nodes[child as usize].point_count > 0 {
                        stack[sp] = child;
                        sp += 1;
                    }
                }
            }
        }

        Ok((fx, fy, fz))
    }

    fn check_xyz_outputs(
        &self,
        out_x: &[S],
        out_y: &[S],
        out_z: &[S],
        context: &'static str,
    ) -> CoreResult<usize> {
        let n = self.point_count();
        if out_x.len() != n {
            return Err(CoreError::dimension_mismatch(n, out_x.len(), context));
        }
        if out_y.len() != n {
            return Err(CoreError::dimension_mismatch(n, out_y.len(), context));
        }
        if out_z.len() != n {
            return Err(CoreError::dimension_mismatch(n, out_z.len(), context));
        }
        Ok(n)
    }

    fn check_flat_output(&self, out_flat: &[S], context: &'static str) -> CoreResult<usize> {
        let n = self.point_count();
        let expected = n
            .checked_mul(3)
            .ok_or_else(|| CoreError::invalid_input("point_count * 3 overflows usize"))?;
        if out_flat.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                out_flat.len(),
                context,
            ));
        }
        Ok(n)
    }

    /// Evaluates unnormalized Barnes-Hut weighted Coulomb repulsion `(fx, fy, fz)` for vertex `point_idx`:
    /// $\sum_{j \neq i} m_j \frac{x_i - x_j}{d_{ij}^2 + \varepsilon^2}$.
    pub fn eval_point_coulomb_raw(
        &self,
        point_idx: usize,
        theta_sq: f64,
        eps_sq: f64,
    ) -> CoreResult<(f64, f64, f64)> {
        if !eps_sq.is_finite() || eps_sq <= 0.0 {
            return Err(CoreError::invalid_input("eps_sq must be finite and > 0.0"));
        }
        self.eval_weighted_radial_raw(
            point_idx,
            theta_sq,
            "ArenaOctree3d::eval_point_coulomb_raw",
            |mass, dist_sq| mass / (dist_sq + eps_sq),
        )
    }

    /// Evaluates weighted Barnes-Hut Coulomb repulsion across all points into separate `x`/`y`/`z` slices.
    pub fn compute_coulomb_repulsion(
        &self,
        theta: S,
        eps_sq: S,
        out_x: &mut [S],
        out_y: &mut [S],
        out_z: &mut [S],
    ) -> CoreResult<()> {
        let n = self.check_xyz_outputs(
            out_x,
            out_y,
            out_z,
            "ArenaOctree3d::compute_coulomb_repulsion",
        )?;
        let theta_f = theta.to_f64();
        let eps_sq_f = eps_sq.to_f64();
        if !theta_f.is_finite() || theta_f < 0.0 {
            return Err(CoreError::invalid_input("theta must be finite and >= 0.0"));
        }
        if !eps_sq_f.is_finite() || eps_sq_f <= 0.0 {
            return Err(CoreError::invalid_input("eps_sq must be finite and > 0.0"));
        }
        if n < 2 {
            out_x.fill(S::ZERO);
            out_y.fill(S::ZERO);
            out_z.fill(S::ZERO);
            return Ok(());
        }

        let theta_sq = theta_f * theta_f;
        for i in 0..n {
            let (fx, fy, fz) = self.eval_point_coulomb_raw(i, theta_sq, eps_sq_f)?;
            out_x[i] = S::from_f64(fx);
            out_y[i] = S::from_f64(fy);
            out_z[i] = S::from_f64(fz);
        }
        Ok(())
    }

    /// Evaluates weighted Barnes-Hut Coulomb repulsion across all points into flat row-major `[N * 3]`.
    pub fn compute_coulomb_repulsion_flat(
        &self,
        theta: S,
        eps_sq: S,
        out_flat: &mut [S],
    ) -> CoreResult<()> {
        let n =
            self.check_flat_output(out_flat, "ArenaOctree3d::compute_coulomb_repulsion_flat")?;
        let theta_f = theta.to_f64();
        let eps_sq_f = eps_sq.to_f64();
        if !theta_f.is_finite() || theta_f < 0.0 {
            return Err(CoreError::invalid_input("theta must be finite and >= 0.0"));
        }
        if !eps_sq_f.is_finite() || eps_sq_f <= 0.0 {
            return Err(CoreError::invalid_input("eps_sq must be finite and > 0.0"));
        }
        if n < 2 {
            out_flat.fill(S::ZERO);
            return Ok(());
        }

        let theta_sq = theta_f * theta_f;
        for i in 0..n {
            let (fx, fy, fz) = self.eval_point_coulomb_raw(i, theta_sq, eps_sq_f)?;
            let base = i * 3;
            out_flat[base] = S::from_f64(fx);
            out_flat[base + 1] = S::from_f64(fy);
            out_flat[base + 2] = S::from_f64(fz);
        }
        Ok(())
    }

    /// Evaluates unnormalized Barnes-Hut generalized MaxEnt $q$-power repulsion `(fx, fy, fz)` for
    /// vertex `point_idx` ($q > -2.0$):
    /// $\sum_{j \neq i} m_j \frac{x_i - x_j}{(d_{ij}^2 + \varepsilon^2)^{(q + 2)/2}}$.
    pub fn eval_point_maxent_raw(
        &self,
        point_idx: usize,
        theta_sq: f64,
        q: f64,
        eps_sq: f64,
    ) -> CoreResult<(f64, f64, f64)> {
        if !q.is_finite() || q <= -2.0 {
            return Err(CoreError::invalid_input("q must be finite and > -2.0"));
        }
        if !eps_sq.is_finite() || eps_sq <= 0.0 {
            return Err(CoreError::invalid_input("eps_sq must be finite and > 0.0"));
        }
        if q == 0.0 {
            return self.eval_point_coulomb_raw(point_idx, theta_sq, eps_sq);
        }
        let exp = 0.5 * (q + 2.0);
        self.eval_weighted_radial_raw(
            point_idx,
            theta_sq,
            "ArenaOctree3d::eval_point_maxent_raw",
            |mass, dist_sq| mass / (dist_sq + eps_sq).powf(exp),
        )
    }

    /// Evaluates weighted Barnes-Hut generalized MaxEnt $q$-power repulsion ($q > -2.0$) into `x`/`y`/`z` slices.
    pub fn compute_maxent_repulsion(
        &self,
        theta: S,
        q: f64,
        eps_sq: S,
        out_x: &mut [S],
        out_y: &mut [S],
        out_z: &mut [S],
    ) -> CoreResult<()> {
        let n = self.check_xyz_outputs(
            out_x,
            out_y,
            out_z,
            "ArenaOctree3d::compute_maxent_repulsion",
        )?;
        let theta_f = theta.to_f64();
        let eps_sq_f = eps_sq.to_f64();
        if !theta_f.is_finite() || theta_f < 0.0 {
            return Err(CoreError::invalid_input("theta must be finite and >= 0.0"));
        }
        if !q.is_finite() || q <= -2.0 {
            return Err(CoreError::invalid_input("q must be finite and > -2.0"));
        }
        if !eps_sq_f.is_finite() || eps_sq_f <= 0.0 {
            return Err(CoreError::invalid_input("eps_sq must be finite and > 0.0"));
        }
        if q == 0.0 {
            return self.compute_coulomb_repulsion(theta, eps_sq, out_x, out_y, out_z);
        }
        if n < 2 {
            out_x.fill(S::ZERO);
            out_y.fill(S::ZERO);
            out_z.fill(S::ZERO);
            return Ok(());
        }

        let theta_sq = theta_f * theta_f;
        for i in 0..n {
            let (fx, fy, fz) = self.eval_point_maxent_raw(i, theta_sq, q, eps_sq_f)?;
            out_x[i] = S::from_f64(fx);
            out_y[i] = S::from_f64(fy);
            out_z[i] = S::from_f64(fz);
        }
        Ok(())
    }

    /// Evaluates weighted Barnes-Hut generalized MaxEnt $q$-power repulsion ($q > -2.0$) into flat `[N * 3]`.
    pub fn compute_maxent_repulsion_flat(
        &self,
        theta: S,
        q: f64,
        eps_sq: S,
        out_flat: &mut [S],
    ) -> CoreResult<()> {
        let n = self.check_flat_output(out_flat, "ArenaOctree3d::compute_maxent_repulsion_flat")?;
        let theta_f = theta.to_f64();
        let eps_sq_f = eps_sq.to_f64();
        if !theta_f.is_finite() || theta_f < 0.0 {
            return Err(CoreError::invalid_input("theta must be finite and >= 0.0"));
        }
        if !q.is_finite() || q <= -2.0 {
            return Err(CoreError::invalid_input("q must be finite and > -2.0"));
        }
        if !eps_sq_f.is_finite() || eps_sq_f <= 0.0 {
            return Err(CoreError::invalid_input("eps_sq must be finite and > 0.0"));
        }
        if q == 0.0 {
            return self.compute_coulomb_repulsion_flat(theta, eps_sq, out_flat);
        }
        if n < 2 {
            out_flat.fill(S::ZERO);
            return Ok(());
        }

        let theta_sq = theta_f * theta_f;
        for i in 0..n {
            let (fx, fy, fz) = self.eval_point_maxent_raw(i, theta_sq, q, eps_sq_f)?;
            let base = i * 3;
            out_flat[base] = S::from_f64(fx);
            out_flat[base + 1] = S::from_f64(fy);
            out_flat[base + 2] = S::from_f64(fz);
        }
        Ok(())
    }

    /// Evaluates unnormalized Barnes-Hut generalized power-law repulsion `(fx, fy, fz)` for
    /// vertex `point_idx` matching the `LinLog` sign/exponent convention:
    /// $\sum_{j \neq i} m_j (d_{ij}^2 + \varepsilon^2)^{(r - 1)/2} (x_i - x_j)$.
    pub fn eval_point_power_raw(
        &self,
        point_idx: usize,
        theta_sq: f64,
        exp_r: f64,
        eps_sq: f64,
    ) -> CoreResult<(f64, f64, f64)> {
        if !exp_r.is_finite() {
            return Err(CoreError::invalid_input("exp_r must be finite"));
        }
        if !eps_sq.is_finite() || eps_sq <= 0.0 {
            return Err(CoreError::invalid_input("eps_sq must be finite and > 0.0"));
        }
        if (exp_r - (-1.0)).abs() <= 1e-12 {
            return self.eval_point_coulomb_raw(point_idx, theta_sq, eps_sq);
        }
        let pow_exp = 0.5 * (exp_r - 1.0);
        self.eval_weighted_radial_raw(
            point_idx,
            theta_sq,
            "ArenaOctree3d::eval_point_power_raw",
            |mass, dist_sq| mass * (dist_sq + eps_sq).powf(pow_exp),
        )
    }

    /// Evaluates weighted Barnes-Hut generalized power-law repulsion across all points into flat `[N * 3]`
    /// following the `LinLog` convention $\sum_{j \neq i} m_j (d_{ij}^2 + \varepsilon^2)^{(r - 1)/2} (x_i - x_j)$.
    pub fn compute_power_repulsion_flat(
        &self,
        theta: f64,
        exp_r: f64,
        eps_sq: f64,
        out_flat: &mut [S],
    ) -> CoreResult<()> {
        let n = self.check_flat_output(out_flat, "ArenaOctree3d::compute_power_repulsion_flat")?;
        if !theta.is_finite() || theta < 0.0 {
            return Err(CoreError::invalid_input("theta must be finite and >= 0.0"));
        }
        if !exp_r.is_finite() {
            return Err(CoreError::invalid_input("exp_r must be finite"));
        }
        if !eps_sq.is_finite() || eps_sq <= 0.0 {
            return Err(CoreError::invalid_input("eps_sq must be finite and > 0.0"));
        }
        if n < 2 {
            out_flat.fill(S::ZERO);
            return Ok(());
        }

        let theta_sq = theta * theta;
        for i in 0..n {
            let (fx, fy, fz) = self.eval_point_power_raw(i, theta_sq, exp_r, eps_sq)?;
            let base = i * 3;
            out_flat[base] = S::from_f64(fx);
            out_flat[base + 1] = S::from_f64(fy);
            out_flat[base + 2] = S::from_f64(fz);
        }
        Ok(())
    }
}
