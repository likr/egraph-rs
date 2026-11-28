use egraph_core::{CoreError, CoreResult, Scalar};
use egraph_spatial_trees::Delaunay2d;

use super::solver::PrismOverlapRemoval;
use super::{
    INNER_MAJORIZATION_ITERS, MIN_EDGE_DIST, MIN_WEIGHT_DIST, OVERLAP_MICRO_DIST_RATIO,
    OVERLAP_WEIGHT_BOOST, SSOR_OMEGA,
};

#[derive(Debug, Clone, Copy)]
pub(super) struct PrismEdge {
    pub(super) u: u32,
    pub(super) v: u32,
    pub(super) target_dist: f64,
    pub(super) weight: f64,
}

impl<S: Scalar> PrismOverlapRemoval<S> {
    pub(super) fn jitter_coincident_points(&mut self) -> bool {
        let n = self.num_nodes;
        if n < 2 {
            return false;
        }
        self.sort_pts.clear();
        {
            let xs = self.work_layout.x();
            let ys = self.work_layout.y();
            for i in 0..n {
                self.sort_pts.push((xs[i], ys[i], i as u32));
            }
        }
        self.sort_pts.sort_unstable_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| a.1.total_cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        let mut moved = false;
        let (xs_mut, ys_mut) = self.work_layout.xy_mut();
        let mut i = 0usize;
        while i < n {
            let (base_x, base_y, _) = self.sort_pts[i];
            let mut j = i + 1;
            while j < n {
                let (nx, ny, _) = self.sort_pts[j];
                if (nx - base_x).abs() <= MIN_EDGE_DIST && (ny - base_y).abs() <= MIN_EDGE_DIST {
                    j += 1;
                } else {
                    break;
                }
            }
            if j - i > 1 {
                moved = true;
                let coord_ulp = base_x.abs().max(base_y.abs()) * S::EPSILON.to_f64();
                for k in (i + 1)..j {
                    let idx = self.sort_pts[k].2 as usize;
                    let extent = self.eff_hw_f64[idx].max(self.eff_hh_f64[idx]).max(1e-2);
                    let jitter_scale = (extent * 0.05).max(4.0 * coord_ulp);
                    let angle = self.rng.next_f64() * std::f64::consts::TAU;
                    let radius = jitter_scale * (0.25 + 0.75 * self.rng.next_f64());
                    let (sin_a, cos_a) = angle.sin_cos();
                    xs_mut[idx] += radius * cos_a;
                    ys_mut[idx] += radius * sin_a;
                }
            }
            i = j;
        }
        moved
    }

    pub(super) fn build_proximity_edges(&mut self) -> CoreResult<()> {
        self.raw_edges.clear();

        Delaunay2d::triangulate_edges_into(&self.work_layout, &mut self.raw_edges)?;

        // Merge in any non-Delaunay overlapping rectangle pairs so overlaps always converge to 0.
        self.raw_edges.extend_from_slice(&self.active_overlaps);
        self.raw_edges.sort_unstable();
        self.raw_edges.dedup();
        Ok(())
    }

    pub(super) fn build_laplacian_csr(&mut self) -> CoreResult<()> {
        let n = self.num_nodes;
        self.prism_edges.clear();
        self.l_deg.fill(0);
        self.u_deg.fill(0);

        let s_max = self.config.max_scale_factor;
        // Tiny clearance margin ensures separated boxes cross the exact touching boundary cleanly
        // even under neighbor tension and f32/f64 rounding.
        let sep_eps = 1.0 + 1e-5_f64.max(16.0 * S::EPSILON.to_f64());

        let xs = self.work_layout.x();
        let ys = self.work_layout.y();

        for &(raw_u, raw_v) in &self.raw_edges {
            if raw_u == raw_v {
                continue;
            }
            let (u32_u, u32_v) = if raw_u < raw_v {
                (raw_u, raw_v)
            } else {
                (raw_v, raw_u)
            };
            let u = u32_u as usize;
            let v = u32_v as usize;

            let mut dx = xs[u] - xs[v];
            let mut dy = ys[u] - ys[v];
            let mut dist = dx.hypot(dy);
            let extent = self.eff_hw_f64[u]
                .max(self.eff_hh_f64[u])
                .max(self.eff_hw_f64[v])
                .max(self.eff_hh_f64[v]);

            if dist < MIN_EDGE_DIST {
                let angle = self.rng.next_f64() * std::f64::consts::TAU;
                let (sin_a, cos_a) = angle.sin_cos();
                let coord_ulp = xs[u]
                    .abs()
                    .max(xs[v].abs())
                    .max(ys[u].abs())
                    .max(ys[v].abs())
                    * S::EPSILON.to_f64();
                let eps = (extent.max(1e-2) * 1e-3).max(4.0 * coord_ulp);
                dx = eps * cos_a;
                dy = eps * sin_a;
                dist = eps;
            }

            let req_x = self.eff_hw_f64[u] + self.eff_hw_f64[v];
            let req_y = self.eff_hh_f64[u] + self.eff_hh_f64[v];
            let adx = dx.abs();
            let ady = dy.abs();

            // Check overlap both in f64 and in S (e.g., f32 rounding of `x +/- half_extent`).
            let xu_s = S::from_f64(xs[u]);
            let xv_s = S::from_f64(xs[v]);
            let yu_s = S::from_f64(ys[u]);
            let yv_s = S::from_f64(ys[v]);
            let hw_u_s = self.eff_hw_s[u];
            let hw_v_s = self.eff_hw_s[v];
            let hh_u_s = self.eff_hh_s[u];
            let hh_v_s = self.eff_hh_s[v];
            let overlaps_in_s = (xu_s - hw_u_s) < (xv_s + hw_v_s)
                && (xv_s - hw_v_s) < (xu_s + hw_u_s)
                && (yu_s - hh_u_s) < (yv_s + hh_v_s)
                && (yv_s - hh_v_s) < (yu_s + hh_u_s);

            let (s_ij, is_overlapping) =
                if req_x > 0.0 && req_y > 0.0 && ((adx < req_x && ady < req_y) || overlaps_in_s) {
                    let ulp_margin_x =
                        (xs[u].abs().max(xs[v].abs()) + req_x) * (8.0 * S::EPSILON.to_f64());
                    let ulp_margin_y =
                        (ys[u].abs().max(ys[v].abs()) + req_y) * (8.0 * S::EPSILON.to_f64());
                    let target_x = req_x * sep_eps + ulp_margin_x;
                    let target_y = req_y * sep_eps + ulp_margin_y;
                    let tx = if adx > 1e-12 {
                        target_x / adx
                    } else {
                        f64::INFINITY
                    };
                    let ty = if ady > 1e-12 {
                        target_y / ady
                    } else {
                        f64::INFINITY
                    };
                    (tx.min(ty).max(1.0).min(s_max), true)
                } else {
                    (1.0, false)
                };

            // Collapsed micro-distance guard: overlapping near-coincident vertices (e.g., from
            // t-FDP leaf collapse) start at least `extent * 1e-4` instead of spending ~50 rounds
            // expanding geometrically from ~1e-9, and never smaller than S-coordinate ULP resolution.
            let eff_dist = if is_overlapping {
                let coord_ulp = xs[u]
                    .abs()
                    .max(xs[v].abs())
                    .max(ys[u].abs())
                    .max(ys[v].abs())
                    * S::EPSILON.to_f64();
                dist.max(extent * OVERLAP_MICRO_DIST_RATIO)
                    .max(2.0 * coord_ulp)
            } else {
                dist
            };
            let target_dist = s_ij * eff_dist;
            let d_eff = target_dist.max(MIN_WEIGHT_DIST);
            let mut w = 1.0 / (d_eff * d_eff);
            if is_overlapping {
                w *= OVERLAP_WEIGHT_BOOST;
            }

            self.prism_edges.push(PrismEdge {
                u: u32_u,
                v: u32_v,
                target_dist,
                weight: w,
            });
            // Canonical u < v places (u, v) in U[u] and (v, u) in L[v].
            self.u_deg[u] += 1;
            self.l_deg[v] += 1;
        }

        // Build split L + D + U CSR for symmetric weighted Laplacian L_w.
        self.l_indptr[0] = 0;
        self.u_indptr[0] = 0;
        for i in 0..n {
            self.l_indptr[i + 1] = self.l_indptr[i]
                .checked_add(self.l_deg[i])
                .ok_or_else(|| CoreError::invalid_csr("PRISM lower CSR nnz overflow"))?;
            self.u_indptr[i + 1] = self.u_indptr[i]
                .checked_add(self.u_deg[i])
                .ok_or_else(|| CoreError::invalid_csr("PRISM upper CSR nnz overflow"))?;
        }
        let l_nnz = self.l_indptr[n] as usize;
        let u_nnz = self.u_indptr[n] as usize;
        self.l_indices.resize(l_nnz, 0);
        self.l_values.resize(l_nnz, 0.0);
        self.u_indices.resize(u_nnz, 0);
        self.u_values.resize(u_nnz, 0.0);

        self.l_pos.copy_from_slice(&self.l_indptr[..n]);
        self.u_pos.copy_from_slice(&self.u_indptr[..n]);
        self.diag.fill(0.0);

        for edge in &self.prism_edges {
            let u = edge.u as usize;
            let v = edge.v as usize;
            let w = edge.weight;

            self.diag[u] += w;
            self.diag[v] += w;

            let pu = self.u_pos[u] as usize;
            self.u_indices[pu] = edge.v;
            self.u_values[pu] = -w;
            self.u_pos[u] += 1;

            let pv = self.l_pos[v] as usize;
            self.l_indices[pv] = edge.u;
            self.l_values[pv] = -w;
            self.l_pos[v] += 1;
        }

        for i in 0..n {
            let d = self.diag[i];
            self.diag_inv[i] = if d.is_finite() && d > 1e-30 {
                1.0 / d
            } else {
                0.0
            };
        }

        Ok(())
    }

    pub(super) fn run_majorization_round(&mut self) -> CoreResult<()> {
        let n = self.num_nodes;
        let xs = self.work_layout.x();
        let ys = self.work_layout.y();
        self.x_next.copy_from_slice(xs);
        self.y_next.copy_from_slice(ys);

        let mean_x_before = xs.iter().sum::<f64>() / (n as f64);
        let mean_y_before = ys.iter().sum::<f64>() / (n as f64);

        for _inner in 0..INNER_MAJORIZATION_ITERS {
            self.rhs_x.fill(0.0);
            self.rhs_y.fill(0.0);

            for edge in &self.prism_edges {
                let u = edge.u as usize;
                let v = edge.v as usize;
                let dx = self.x_next[u] - self.x_next[v];
                let dy = self.y_next[u] - self.y_next[v];
                let norm = dx.hypot(dy);
                if norm >= MIN_EDGE_DIST && norm.is_finite() {
                    let f = edge.weight * edge.target_dist / norm;
                    let fx = f * dx;
                    let fy = f * dy;
                    self.rhs_x[u] += fx;
                    self.rhs_x[v] -= fx;
                    self.rhs_y[u] += fy;
                    self.rhs_y[v] -= fy;
                } else {
                    // Deterministic unit fallback if coincident nodes survived sorted jitter.
                    let angle = ((edge.u.wrapping_mul(2_654_435_761)
                        ^ edge.v.wrapping_mul(2_246_822_519))
                        as f64)
                        * 1e-9;
                    let (sin_a, cos_a) = angle.sin_cos();
                    let f = edge.weight * edge.target_dist;
                    let fx = f * cos_a;
                    let fy = f * sin_a;
                    self.rhs_x[u] += fx;
                    self.rhs_x[v] -= fx;
                    self.rhs_y[u] += fy;
                    self.rhs_y[v] -= fy;
                }
            }

            self.pcg.solve_ldu_2rhs_ssor(
                &self.diag,
                &self.diag_inv,
                &self.l_indptr,
                &self.l_indices,
                &self.l_values,
                &self.u_indptr,
                &self.u_indices,
                &self.u_values,
                &self.rhs_x,
                &self.rhs_y,
                &mut self.x_next,
                &mut self.y_next,
                self.config.cg_tolerance,
                self.config.cg_max_iters,
                SSOR_OMEGA,
            )?;
        }

        // Preserve layout centroid to avoid numerical drift across rounds.
        let mean_x_after = self.x_next.iter().sum::<f64>() / (n as f64);
        let mean_y_after = self.y_next.iter().sum::<f64>() / (n as f64);
        let shift_x = mean_x_before - mean_x_after;
        let shift_y = mean_y_before - mean_y_after;
        for i in 0..n {
            self.x_next[i] += shift_x;
            self.y_next[i] += shift_y;
        }

        Ok(())
    }
}
