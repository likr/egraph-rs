use egraph_core::{CoreError, CoreResult, Scalar};

use super::ArenaQuadtree2d;

impl<S: Scalar> ArenaQuadtree2d<S> {
    /// Evaluates unnormalized Barnes-Hut weighted Coulomb repulsion `(fx, fy)` for vertex `point_idx`:
    /// $\sum_{j \neq i} m_j \frac{x_i - x_j}{d_{ij}^2 + \varepsilon^2}$.
    pub fn eval_point_coulomb_raw(
        &self,
        point_idx: usize,
        theta_sq: f64,
        eps_sq: f64,
    ) -> CoreResult<(f64, f64)> {
        let n = self.point_count();
        if point_idx >= n {
            return Err(CoreError::index_out_of_bounds(
                point_idx,
                n,
                "ArenaQuadtree2d::eval_point_coulomb_raw",
            ));
        }
        if !theta_sq.is_finite() || theta_sq < 0.0 {
            return Err(CoreError::invalid_input(
                "theta_sq must be finite and >= 0.0",
            ));
        }
        if !eps_sq.is_finite() || eps_sq <= 0.0 {
            return Err(CoreError::invalid_input("eps_sq must be finite and > 0.0"));
        }
        if self.nodes.is_empty() {
            return Ok((0.0, 0.0));
        }

        let x = self.coords_x[point_idx];
        let y = self.coords_y[point_idx];
        let own_leaf = self.leaf_of_point[point_idx];

        let mut stack = [0u32; 128];
        let mut sp = 1usize;
        stack[0] = 0;

        let mut fx = 0.0_f64;
        let mut fy = 0.0_f64;

        while sp > 0 {
            sp -= 1;
            let node_idx = stack[sp];
            let node = &self.nodes[node_idx as usize];
            if node.count == 0 || node.total_mass == 0.0 {
                continue;
            }

            if node.first_child == u32::MAX {
                if node_idx == own_leaf {
                    let mut p = node.leaf_head;
                    while p != u32::MAX {
                        let other = p as usize;
                        if other != point_idx {
                            let dx = x - self.coords_x[other];
                            let dy = y - self.coords_y[other];
                            let denom = dx * dx + dy * dy + eps_sq;
                            let mult = self.masses[other] / denom;
                            fx += mult * dx;
                            fy += mult * dy;
                        }
                        p = self.next_in_leaf[other];
                    }
                } else {
                    let dx = x - node.com_x;
                    let dy = y - node.com_y;
                    let denom = dx * dx + dy * dy + eps_sq;
                    let mult = node.total_mass / denom;
                    fx += mult * dx;
                    fy += mult * dy;
                }
                continue;
            }

            let dx = x - node.com_x;
            let dy = y - node.com_y;
            let dist_sq = dx * dx + dy * dy;
            let max_dim = node.width.max(node.height);
            if (max_dim * max_dim) < theta_sq * dist_sq {
                let denom = dist_sq + eps_sq;
                let mult = node.total_mass / denom;
                fx += mult * dx;
                fy += mult * dy;
            } else {
                let base = node.first_child;
                for c in 0..4 {
                    let child = base + c;
                    if self.nodes[child as usize].count > 0 {
                        stack[sp] = child;
                        sp += 1;
                    }
                }
            }
        }

        Ok((fx, fy))
    }

    /// Evaluates weighted Barnes-Hut Coulomb repulsion across all points:
    /// $\sum_{j \neq i} m_j \frac{x_i - x_j}{d_{ij}^2 + \varepsilon^2}$.
    pub fn compute_coulomb_repulsion(
        &self,
        theta: S,
        eps_sq: S,
        out_x: &mut [S],
        out_y: &mut [S],
    ) -> CoreResult<()> {
        let n = self.point_count();
        if out_x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_x.len(),
                "ArenaQuadtree2d::compute_coulomb_repulsion::out_x",
            ));
        }
        if out_y.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_y.len(),
                "ArenaQuadtree2d::compute_coulomb_repulsion::out_y",
            ));
        }
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
            return Ok(());
        }

        let theta_sq = theta_f * theta_f;
        for i in 0..n {
            let (fx, fy) = self.eval_point_coulomb_raw(i, theta_sq, eps_sq_f)?;
            out_x[i] = S::from_f64(fx);
            out_y[i] = S::from_f64(fy);
        }
        Ok(())
    }

    /// Evaluates unnormalized Barnes-Hut generalized Maxent $q$-power repulsion `(fx, fy)` for
    /// vertex `point_idx` ($q > -2.0$):
    /// $\sum_{j \neq i} m_j \frac{x_i - x_j}{(d_{ij}^2 + \varepsilon^2)^{(q + 2)/2}}$.
    ///
    /// When `q == 0.0`, delegates directly to [`ArenaQuadtree2d::eval_point_coulomb_raw`] without
    /// `powf`.
    pub fn eval_point_maxent_raw(
        &self,
        point_idx: usize,
        theta_sq: f64,
        q: f64,
        eps_sq: f64,
    ) -> CoreResult<(f64, f64)> {
        let n = self.point_count();
        if point_idx >= n {
            return Err(CoreError::index_out_of_bounds(
                point_idx,
                n,
                "ArenaQuadtree2d::eval_point_maxent_raw",
            ));
        }
        if !theta_sq.is_finite() || theta_sq < 0.0 {
            return Err(CoreError::invalid_input(
                "theta_sq must be finite and >= 0.0",
            ));
        }
        if !q.is_finite() || q <= -2.0 {
            return Err(CoreError::invalid_input("q must be finite and > -2.0"));
        }
        if !eps_sq.is_finite() || eps_sq <= 0.0 {
            return Err(CoreError::invalid_input("eps_sq must be finite and > 0.0"));
        }
        if q == 0.0 {
            return self.eval_point_coulomb_raw(point_idx, theta_sq, eps_sq);
        }
        if self.nodes.is_empty() {
            return Ok((0.0, 0.0));
        }

        let exp = 0.5 * (q + 2.0);
        let x = self.coords_x[point_idx];
        let y = self.coords_y[point_idx];
        let own_leaf = self.leaf_of_point[point_idx];

        let mut stack = [0u32; 128];
        let mut sp = 1usize;
        stack[0] = 0;

        let mut fx = 0.0_f64;
        let mut fy = 0.0_f64;

        while sp > 0 {
            sp -= 1;
            let node_idx = stack[sp];
            let node = &self.nodes[node_idx as usize];
            if node.count == 0 || node.total_mass == 0.0 {
                continue;
            }

            if node.first_child == u32::MAX {
                if node_idx == own_leaf {
                    let mut p = node.leaf_head;
                    while p != u32::MAX {
                        let other = p as usize;
                        if other != point_idx {
                            let dx = x - self.coords_x[other];
                            let dy = y - self.coords_y[other];
                            let denom = (dx * dx + dy * dy + eps_sq).powf(exp);
                            let mult = self.masses[other] / denom;
                            fx += mult * dx;
                            fy += mult * dy;
                        }
                        p = self.next_in_leaf[other];
                    }
                } else {
                    let dx = x - node.com_x;
                    let dy = y - node.com_y;
                    let denom = (dx * dx + dy * dy + eps_sq).powf(exp);
                    let mult = node.total_mass / denom;
                    fx += mult * dx;
                    fy += mult * dy;
                }
                continue;
            }

            let dx = x - node.com_x;
            let dy = y - node.com_y;
            let dist_sq = dx * dx + dy * dy;
            let max_dim = node.width.max(node.height);
            if (max_dim * max_dim) < theta_sq * dist_sq {
                let denom = (dist_sq + eps_sq).powf(exp);
                let mult = node.total_mass / denom;
                fx += mult * dx;
                fy += mult * dy;
            } else {
                let base = node.first_child;
                for c in 0..4 {
                    let child = base + c;
                    if self.nodes[child as usize].count > 0 {
                        stack[sp] = child;
                        sp += 1;
                    }
                }
            }
        }

        Ok((fx, fy))
    }

    /// Evaluates weighted Barnes-Hut generalized Maxent $q$-power repulsion across all points
    /// ($q > -2.0$):
    /// $\sum_{j \neq i} m_j \frac{x_i - x_j}{(d_{ij}^2 + \varepsilon^2)^{(q + 2)/2}}$.
    pub fn compute_maxent_repulsion(
        &self,
        theta: S,
        q: f64,
        eps_sq: S,
        out_x: &mut [S],
        out_y: &mut [S],
    ) -> CoreResult<()> {
        let n = self.point_count();
        if out_x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_x.len(),
                "ArenaQuadtree2d::compute_maxent_repulsion::out_x",
            ));
        }
        if out_y.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_y.len(),
                "ArenaQuadtree2d::compute_maxent_repulsion::out_y",
            ));
        }
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
            return self.compute_coulomb_repulsion(theta, eps_sq, out_x, out_y);
        }
        if n < 2 {
            out_x.fill(S::ZERO);
            out_y.fill(S::ZERO);
            return Ok(());
        }

        let theta_sq = theta_f * theta_f;
        for i in 0..n {
            let (fx, fy) = self.eval_point_maxent_raw(i, theta_sq, q, eps_sq_f)?;
            out_x[i] = S::from_f64(fx);
            out_y[i] = S::from_f64(fy);
        }
        Ok(())
    }
}
