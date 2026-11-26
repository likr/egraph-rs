use egraph_core::{CoreResult, Layout2d, Scalar};
use egraph_spatial_trees::{KdQueryScratch, RectangleOverlapSweep, StaticKdTree};

use super::config::{ForbidVariant, MIN_EDGE_DIST};
use super::solver::ForbidOverlapRemoval;

impl<S: Scalar> ForbidOverlapRemoval<S> {
    fn jitter_coincident_orig(&mut self) {
        let n = self.num_nodes;
        if n < 2 {
            return;
        }
        self.sort_pts.clear();
        for i in 0..n {
            self.sort_pts
                .push((self.orig_x[i], self.orig_y[i], i as u32));
        }
        self.sort_pts.sort_unstable_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| a.1.total_cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

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
                let coord_ulp = base_x.abs().max(base_y.abs()) * S::EPSILON.to_f64();
                for k in (i + 1)..j {
                    let idx = self.sort_pts[k].2 as usize;
                    let extent = self.eff_hw_f64[idx].max(self.eff_hh_f64[idx]).max(1e-2);
                    let jitter_scale = (extent * 0.05).max(4.0 * coord_ulp).max(1e-5);
                    let angle = self.rng.next_f64() * std::f64::consts::TAU;
                    let radius = jitter_scale * (0.25 + 0.75 * self.rng.next_f64());
                    let (sin_a, cos_a) = angle.sin_cos();
                    self.orig_x[idx] += radius * cos_a;
                    self.orig_y[idx] += radius * sin_a;
                }
            }
            i = j;
        }
    }

    pub(super) fn initialize_from_layout(&mut self, layout: &Layout2d<S>) -> CoreResult<()> {
        let n = self.num_nodes;
        let xs = layout.x();
        let ys = layout.y();
        for i in 0..n {
            self.orig_x[i] = xs[i].to_f64();
            self.orig_y[i] = ys[i].to_f64();
        }
        self.jitter_coincident_orig();

        self.orig_cx = self.orig_x.iter().sum::<f64>() / (n as f64);
        self.orig_cy = self.orig_y.iter().sum::<f64>() / (n as f64);

        {
            let (wx, wy) = self.work_layout.xy_mut();
            wx.copy_from_slice(&self.orig_x);
            wy.copy_from_slice(&self.orig_y);
        }

        // Compute `up_scale = getScalingRatio(X, S)` across all overlapping pairs in perturbed `orig`.
        let overlaps = RectangleOverlapSweep::find_layout_overlaps(
            &self.work_layout,
            &self.eff_hw_f64,
            &self.eff_hh_f64,
        )?;
        let sep_eps = 1.0 + 1e-5_f64.max(16.0 * S::EPSILON.to_f64());
        let mut up_scale = 1.0_f64;
        for (i, j) in overlaps {
            let dx = (self.orig_x[i] - self.orig_x[j]).abs();
            let dy = (self.orig_y[i] - self.orig_y[j]).abs();
            let req_x = self.eff_hw_f64[i] + self.eff_hw_f64[j];
            let req_y = self.eff_hh_f64[i] + self.eff_hh_f64[j];
            if req_x <= 0.0 || req_y <= 0.0 {
                continue;
            }
            let ulp_x = (self.orig_x[i].abs().max(self.orig_x[j].abs()) + req_x)
                * (8.0 * S::EPSILON.to_f64());
            let ulp_y = (self.orig_y[i].abs().max(self.orig_y[j].abs()) + req_y)
                * (8.0 * S::EPSILON.to_f64());
            let tx = if dx > 1e-15 {
                (req_x * sep_eps + ulp_x) / dx
            } else {
                f64::INFINITY
            };
            let ty = if dy > 1e-15 {
                (req_y * sep_eps + ulp_y) / dy
            } else {
                f64::INFINITY
            };
            let ratio = tx.min(ty);
            if ratio.is_finite() && ratio > up_scale {
                up_scale = ratio;
            }
        }

        // Verify `up_scale` produces strictly 0 overlaps in `S` precision and record `best_free`.
        for _ in 0..8 {
            for i in 0..n {
                self.best_free_x[i] = self.orig_cx + up_scale * (self.orig_x[i] - self.orig_cx);
                self.best_free_y[i] = self.orig_cy + up_scale * (self.orig_y[i] - self.orig_cy);
            }
            let check_layout = Layout2d::from_xy(
                self.best_free_x.iter().map(|&v| S::from_f64(v)).collect(),
                self.best_free_y.iter().map(|&v| S::from_f64(v)).collect(),
            )?;
            if !RectangleOverlapSweep::has_layout_overlaps(
                &check_layout,
                &self.eff_hw_s,
                &self.eff_hh_s,
            )? {
                self.has_best_free = true;
                break;
            }
            up_scale *= 1.01;
        }

        self.low_scale = 1.0;
        self.up_scale = up_scale;
        self.cur_scale = 1.0;
        self.prev_scale = 1.0;

        // Precompute static base pair list (`AllPairs` or `SparseForbid` k-NN).
        self.base_pairs.clear();
        match self.config.variant {
            ForbidVariant::Forbid | ForbidVariant::ForbidPrime => {
                self.base_pairs.reserve(n.saturating_mul(n - 1) / 2);
                for i in 0..n {
                    for j in (i + 1)..n {
                        self.base_pairs.push((i as u32, j as u32));
                    }
                }
            }
            ForbidVariant::SparseForbid => {
                let k = self.config.sparse_k.min(n - 1);
                if k > 0 {
                    let tree = StaticKdTree::<2, f64>::from_layout(&self.work_layout)?;
                    let mut scratch = KdQueryScratch::with_capacity(k);
                    let mut knn_out = Vec::with_capacity(k);
                    self.base_pairs.reserve(n * k);
                    for u in 0..n {
                        tree.k_nearest_excluding_self_with_scratch(
                            u,
                            k,
                            &mut scratch,
                            &mut knn_out,
                        )?;
                        let u32_u = u as u32;
                        for &(v, _) in &knn_out {
                            let v32 = v as u32;
                            self.base_pairs.push(if u32_u < v32 {
                                (u32_u, v32)
                            } else {
                                (v32, u32_u)
                            });
                        }
                    }
                    self.base_pairs.sort_unstable();
                    self.base_pairs.dedup();
                }
            }
        }

        self.initialized_state = true;
        Ok(())
    }

    pub(super) fn apply_scale(&mut self) {
        let n = self.num_nodes;
        match self.config.variant {
            ForbidVariant::Forbid => {
                if self.iter == 0 {
                    let (wx, wy) = self.work_layout.xy_mut();
                    wx.copy_from_slice(&self.orig_x);
                    wy.copy_from_slice(&self.orig_y);
                    self.prev_scale = 1.0;
                } else {
                    let factor = if self.prev_scale > 0.0 {
                        self.cur_scale / self.prev_scale
                    } else {
                        self.cur_scale
                    };
                    let (wx, wy) = self.work_layout.xy_mut();
                    let cx = wx.iter().sum::<f64>() / (n as f64);
                    let cy = wy.iter().sum::<f64>() / (n as f64);
                    for i in 0..n {
                        wx[i] = cx + factor * (wx[i] - cx);
                        wy[i] = cy + factor * (wy[i] - cy);
                    }
                    self.prev_scale = self.cur_scale;
                }
            }
            ForbidVariant::ForbidPrime | ForbidVariant::SparseForbid => {
                let s = self.cur_scale;
                let (wx, wy) = self.work_layout.xy_mut();
                for i in 0..n {
                    wx[i] = self.orig_cx + s * (self.orig_x[i] - self.orig_cx);
                    wy[i] = self.orig_cy + s * (self.orig_y[i] - self.orig_cy);
                }
            }
        }
    }

    pub(super) fn copy_work_to_layout(&self, layout: &mut Layout2d<S>) {
        let wx = self.work_layout.x();
        let wy = self.work_layout.y();
        let (xs_mut, ys_mut) = layout.xy_mut();
        for i in 0..self.num_nodes {
            let nx = S::from_f64(wx[i]);
            let ny = S::from_f64(wy[i]);
            if nx.is_finite() {
                xs_mut[i] = nx;
            }
            if ny.is_finite() {
                ys_mut[i] = ny;
            }
        }
    }
}
