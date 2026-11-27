use egraph_core::{CoreError, CoreResult, Scalar};

use super::{ArenaQuadtree2d, ENTROPY_MIN_DIST};

impl<S: Scalar> ArenaQuadtree2d<S> {
    /// Evaluates unnormalized Barnes-Hut KL sum `(z_i, raw_fx_i, raw_fy_i)` for vertex `point_idx`.
    ///
    /// Uses an explicit stack array bounded by `MAX_DEPTH` (`3 * 32 + 1 = 97 <= 128`) so
    /// traversal is strictly non-recursive and allocation-free.
    pub fn eval_point_kl_raw(
        &self,
        point_idx: usize,
        theta_sq: f64,
    ) -> CoreResult<(f64, f64, f64)> {
        let n = self.point_count();
        if point_idx >= n {
            return Err(CoreError::index_out_of_bounds(
                point_idx,
                n,
                "ArenaQuadtree2d::eval_point_kl_raw",
            ));
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
        let own_leaf = self.leaf_of_point[point_idx];

        let mut stack = [0u32; 128];
        let mut sp = 1usize;
        stack[0] = 0;

        let mut z_sum = 0.0_f64;
        let mut fx = 0.0_f64;
        let mut fy = 0.0_f64;

        while sp > 0 {
            sp -= 1;
            let node_idx = stack[sp];
            let node = &self.nodes[node_idx as usize];
            if node.count == 0 {
                continue;
            }

            if node.first_child == u32::MAX {
                if node_idx == own_leaf {
                    // Exact per-point leaf summation excludes self while handling sub-ULP neighbors.
                    let mut p = node.leaf_head;
                    while p != u32::MAX {
                        let other = p as usize;
                        if other != point_idx {
                            let dx = x - self.coords_x[other];
                            let dy = y - self.coords_y[other];
                            let w = 1.0 / (1.0 + dx * dx + dy * dy);
                            z_sum += w;
                            let mult = w * w;
                            fx += mult * dx;
                            fy += mult * dy;
                        }
                        p = self.next_in_leaf[other];
                    }
                } else {
                    let dx = x - node.com_x;
                    let dy = y - node.com_y;
                    let w = 1.0 / (1.0 + dx * dx + dy * dy);
                    let count_f = node.count as f64;
                    z_sum += count_f * w;
                    let mult = count_f * w * w;
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
                let w = 1.0 / (1.0 + dist_sq);
                let count_f = node.count as f64;
                z_sum += count_f * w;
                let mult = count_f * w * w;
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

        Ok((z_sum, fx, fy))
    }

    /// Evaluates normalized Barnes-Hut KL repulsive forces across all points and returns `total_z`.
    pub fn compute_kl_repulsion(
        &self,
        theta: S,
        out_x: &mut [S],
        out_y: &mut [S],
    ) -> CoreResult<S> {
        let n = self.point_count();
        if out_x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_x.len(),
                "ArenaQuadtree2d::compute_kl_repulsion::out_x",
            ));
        }
        if out_y.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_y.len(),
                "ArenaQuadtree2d::compute_kl_repulsion::out_y",
            ));
        }
        let theta_f = theta.to_f64();
        if !theta_f.is_finite() || theta_f < 0.0 {
            return Err(CoreError::invalid_input("theta must be finite and >= 0.0"));
        }
        if n < 2 {
            out_x.fill(S::ZERO);
            out_y.fill(S::ZERO);
            return Ok(S::ONE);
        }

        let theta_sq = theta_f * theta_f;
        let mut total_z = 0.0_f64;
        for i in 0..n {
            let (zi, fx, fy) = self.eval_point_kl_raw(i, theta_sq)?;
            total_z += zi;
            out_x[i] = S::from_f64(fx);
            out_y[i] = S::from_f64(fy);
        }

        let denom = total_z.max(1e-12);
        for i in 0..n {
            out_x[i] = S::from_f64(out_x[i].to_f64() / denom);
            out_y[i] = S::from_f64(out_y[i].to_f64() / denom);
        }
        Ok(S::from_f64(denom))
    }

    /// Evaluates normalized Barnes-Hut KL repulsive forces into flat row-major `[N * 2]` and returns `total_z`.
    pub fn compute_kl_repulsion_flat(&self, theta: S, out_flat: &mut [S]) -> CoreResult<S> {
        let n = self.point_count();
        let expected = n
            .checked_mul(2)
            .ok_or_else(|| CoreError::invalid_input("point_count * 2 overflows usize"))?;
        if out_flat.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                out_flat.len(),
                "ArenaQuadtree2d::compute_kl_repulsion_flat",
            ));
        }
        let theta_f = theta.to_f64();
        if !theta_f.is_finite() || theta_f < 0.0 {
            return Err(CoreError::invalid_input("theta must be finite and >= 0.0"));
        }
        if n < 2 {
            out_flat.fill(S::ZERO);
            return Ok(S::ONE);
        }

        let theta_sq = theta_f * theta_f;
        let mut total_z = 0.0_f64;
        for i in 0..n {
            let (zi, fx, fy) = self.eval_point_kl_raw(i, theta_sq)?;
            total_z += zi;
            let base = i * 2;
            out_flat[base] = S::from_f64(fx);
            out_flat[base + 1] = S::from_f64(fy);
        }

        let denom = total_z.max(1e-12);
        for v in out_flat.iter_mut() {
            *v = S::from_f64(v.to_f64() / denom);
        }
        Ok(S::from_f64(denom))
    }

    /// Evaluates unnormalized Barnes-Hut entropy repulsion `(fx, fy)` for vertex `point_idx`:
    /// $\sum_{j \neq i} \frac{y_i - y_j}{\|y_i - y_j\|(\|y_i - y_j\| + \varepsilon_r)}$.
    pub fn eval_point_entropy_raw(
        &self,
        point_idx: usize,
        theta_sq: f64,
        epsilon_r: f64,
    ) -> CoreResult<(f64, f64)> {
        let n = self.point_count();
        if point_idx >= n {
            return Err(CoreError::index_out_of_bounds(
                point_idx,
                n,
                "ArenaQuadtree2d::eval_point_entropy_raw",
            ));
        }
        if !theta_sq.is_finite() || theta_sq < 0.0 {
            return Err(CoreError::invalid_input(
                "theta_sq must be finite and >= 0.0",
            ));
        }
        if !epsilon_r.is_finite() || epsilon_r <= 0.0 {
            return Err(CoreError::invalid_input(
                "epsilon_r must be finite and > 0.0",
            ));
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
            if node.count == 0 {
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
                            let dist = dx.hypot(dy);
                            if dist > ENTROPY_MIN_DIST {
                                let mult = 1.0 / (dist * (dist + epsilon_r));
                                fx += mult * dx;
                                fy += mult * dy;
                            }
                        }
                        p = self.next_in_leaf[other];
                    }
                } else {
                    let dx = x - node.com_x;
                    let dy = y - node.com_y;
                    let dist = dx.hypot(dy);
                    if dist > ENTROPY_MIN_DIST {
                        let mult = (node.count as f64) / (dist * (dist + epsilon_r));
                        fx += mult * dx;
                        fy += mult * dy;
                    }
                }
                continue;
            }

            let dx = x - node.com_x;
            let dy = y - node.com_y;
            let dist_sq = dx * dx + dy * dy;
            let max_dim = node.width.max(node.height);
            if (max_dim * max_dim) < theta_sq * dist_sq {
                let dist = dist_sq.sqrt();
                if dist > ENTROPY_MIN_DIST {
                    let mult = (node.count as f64) / (dist * (dist + epsilon_r));
                    fx += mult * dx;
                    fy += mult * dy;
                }
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

    /// Evaluates Barnes-Hut entropy repulsion forces across all points.
    pub fn compute_entropy_repulsion(
        &self,
        theta: S,
        epsilon_r: S,
        out_x: &mut [S],
        out_y: &mut [S],
    ) -> CoreResult<()> {
        let n = self.point_count();
        if out_x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_x.len(),
                "ArenaQuadtree2d::compute_entropy_repulsion::out_x",
            ));
        }
        if out_y.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_y.len(),
                "ArenaQuadtree2d::compute_entropy_repulsion::out_y",
            ));
        }
        let theta_f = theta.to_f64();
        let eps_f = epsilon_r.to_f64();
        if !theta_f.is_finite() || theta_f < 0.0 {
            return Err(CoreError::invalid_input("theta must be finite and >= 0.0"));
        }
        if !eps_f.is_finite() || eps_f <= 0.0 {
            return Err(CoreError::invalid_input(
                "epsilon_r must be finite and > 0.0",
            ));
        }
        if n < 2 {
            out_x.fill(S::ZERO);
            out_y.fill(S::ZERO);
            return Ok(());
        }

        let theta_sq = theta_f * theta_f;
        for i in 0..n {
            let (fx, fy) = self.eval_point_entropy_raw(i, theta_sq, eps_f)?;
            out_x[i] = S::from_f64(fx);
            out_y[i] = S::from_f64(fy);
        }
        Ok(())
    }

    /// Evaluates Barnes-Hut entropy repulsion forces across all points into flat row-major `[N * 2]`.
    pub fn compute_entropy_repulsion_flat(
        &self,
        theta: S,
        epsilon_r: S,
        out_flat: &mut [S],
    ) -> CoreResult<()> {
        let n = self.point_count();
        let expected = n
            .checked_mul(2)
            .ok_or_else(|| CoreError::invalid_input("point_count * 2 overflows usize"))?;
        if out_flat.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                out_flat.len(),
                "ArenaQuadtree2d::compute_entropy_repulsion_flat",
            ));
        }
        let theta_f = theta.to_f64();
        let eps_f = epsilon_r.to_f64();
        if !theta_f.is_finite() || theta_f < 0.0 {
            return Err(CoreError::invalid_input("theta must be finite and >= 0.0"));
        }
        if !eps_f.is_finite() || eps_f <= 0.0 {
            return Err(CoreError::invalid_input(
                "epsilon_r must be finite and > 0.0",
            ));
        }
        if n < 2 {
            out_flat.fill(S::ZERO);
            return Ok(());
        }

        let theta_sq = theta_f * theta_f;
        for i in 0..n {
            let (fx, fy) = self.eval_point_entropy_raw(i, theta_sq, eps_f)?;
            let base = i * 2;
            out_flat[base] = S::from_f64(fx);
            out_flat[base + 1] = S::from_f64(fy);
        }
        Ok(())
    }
}
