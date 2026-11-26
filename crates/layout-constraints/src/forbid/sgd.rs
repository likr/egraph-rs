use egraph_core::{CoreResult, Scalar};
use egraph_spatial_trees::RectangleOverlapSweep;

use super::config::{ForbidVariant, MIN_EDGE_DIST, MIN_TARGET_DIST, SGD2_EPS, SgdPairItem};
use super::solver::ForbidOverlapRemoval;

impl<S: Scalar> ForbidOverlapRemoval<S> {
    pub(super) fn sgd2_stress_opt(&mut self) -> CoreResult<()> {
        let n = self.num_nodes;
        let t_max = self.config.sgd_iterations;
        if n <= 1 || t_max == 0 {
            return Ok(());
        }

        let alpha = -2.0_f64;
        let overlap_exp = alpha * self.config.overlap_weight_k;
        let sep_eps = 1.0 + 1e-6_f64.max(16.0 * S::EPSILON.to_f64());

        for t in 0..t_max {
            // Recompute active overlapping pairs O(X') at the start of each SGD2 iteration.
            let raw_overlaps = RectangleOverlapSweep::find_layout_overlaps(
                &self.work_layout,
                &self.eff_hw_f64,
                &self.eff_hh_f64,
            )?;
            self.active_overlaps.clear();
            self.active_overlaps
                .extend(raw_overlaps.into_iter().map(|(u, v)| (u as u32, v as u32)));

            self.sgd_pairs.clear();
            let mut w_min = f64::INFINITY;
            let mut w_max = 0.0_f64;

            if matches!(self.config.variant, ForbidVariant::SparseForbid) {
                // Merge sorted `base_pairs` and sorted `active_overlaps` cleanly.
                let mut bi = 0usize;
                let mut oi = 0usize;
                while bi < self.base_pairs.len() || oi < self.active_overlaps.len() {
                    let (u, v, is_ov) = if oi >= self.active_overlaps.len() {
                        let p = self.base_pairs[bi];
                        bi += 1;
                        (p.0, p.1, false)
                    } else if bi >= self.base_pairs.len() {
                        let p = self.active_overlaps[oi];
                        oi += 1;
                        (p.0, p.1, true)
                    } else {
                        let pb = self.base_pairs[bi];
                        let po = self.active_overlaps[oi];
                        match pb.cmp(&po) {
                            std::cmp::Ordering::Less => {
                                bi += 1;
                                (pb.0, pb.1, false)
                            }
                            std::cmp::Ordering::Greater => {
                                oi += 1;
                                (po.0, po.1, true)
                            }
                            std::cmp::Ordering::Equal => {
                                bi += 1;
                                oi += 1;
                                (po.0, po.1, true)
                            }
                        }
                    };
                    let ui = u as usize;
                    let vi = v as usize;
                    let (d, w) = if is_ov {
                        let req_x = self.eff_hw_f64[ui] + self.eff_hw_f64[vi];
                        let req_y = self.eff_hh_f64[ui] + self.eff_hh_f64[vi];
                        let d_ov = (req_x * sep_eps)
                            .hypot(req_y * sep_eps)
                            .max(MIN_TARGET_DIST);
                        let w_ov = d_ov.powf(overlap_exp);
                        (d_ov, w_ov)
                    } else {
                        let dx = self.orig_x[ui] - self.orig_x[vi];
                        let dy = self.orig_y[ui] - self.orig_y[vi];
                        let d_orig = dx.hypot(dy).max(MIN_EDGE_DIST);
                        let w_orig = 1.0 / (d_orig * d_orig);
                        (d_orig, w_orig)
                    };
                    if w.is_finite() && w > 0.0 {
                        w_min = w_min.min(w);
                        w_max = w_max.max(w);
                        self.sgd_pairs.push(SgdPairItem {
                            u,
                            v,
                            d,
                            w,
                            is_overlap: is_ov,
                        });
                    }
                }
            } else {
                // Full all-pairs (`Forbid` / `ForbidPrime`): linear merge with sorted `active_overlaps`.
                let mut oi = 0usize;
                for &(u, v) in &self.base_pairs {
                    while oi < self.active_overlaps.len() && self.active_overlaps[oi] < (u, v) {
                        oi += 1;
                    }
                    let is_ov =
                        oi < self.active_overlaps.len() && self.active_overlaps[oi] == (u, v);
                    if is_ov {
                        oi += 1;
                    }
                    let ui = u as usize;
                    let vi = v as usize;
                    let (d, w) = if is_ov {
                        let req_x = self.eff_hw_f64[ui] + self.eff_hw_f64[vi];
                        let req_y = self.eff_hh_f64[ui] + self.eff_hh_f64[vi];
                        let d_ov = (req_x * sep_eps)
                            .hypot(req_y * sep_eps)
                            .max(MIN_TARGET_DIST);
                        let w_ov = d_ov.powf(overlap_exp);
                        (d_ov, w_ov)
                    } else {
                        let dx = self.orig_x[ui] - self.orig_x[vi];
                        let dy = self.orig_y[ui] - self.orig_y[vi];
                        let d_orig = dx.hypot(dy).max(MIN_EDGE_DIST);
                        let w_orig = 1.0 / (d_orig * d_orig);
                        (d_orig, w_orig)
                    };
                    if w.is_finite() && w > 0.0 {
                        w_min = w_min.min(w);
                        w_max = w_max.max(w);
                        self.sgd_pairs.push(SgdPairItem {
                            u,
                            v,
                            d,
                            w,
                            is_overlap: is_ov,
                        });
                    }
                }
            }

            if self.sgd_pairs.is_empty() || !w_min.is_finite() || w_max <= 0.0 {
                break;
            }

            let eta_max = 1.0 / w_min;
            let eta_min = (SGD2_EPS / w_max).min(eta_max);
            let eta = if t_max <= 1 || eta_max <= eta_min {
                eta_max
            } else {
                let b = (eta_max / eta_min).ln() / ((t_max - 1) as f64);
                eta_max * (-b * (t as f64)).exp()
            };

            self.rng.shuffle(&mut self.sgd_pairs);

            let (wx, wy) = self.work_layout.xy_mut();
            let mut max_move = 0.0_f64;
            for item in &self.sgd_pairs {
                let mu = (eta * item.w).min(1.0);
                if !mu.is_finite() || mu <= 0.0 {
                    continue;
                }
                let i = item.u as usize;
                let j = item.v as usize;
                let dx = wx[i] - wx[j];
                let dy = wy[i] - wy[j];
                let norm = dx.hypot(dy);
                if norm <= 1e-12 || !norm.is_finite() {
                    if item.is_overlap {
                        let angle = ((item.u.wrapping_mul(2_654_435_761)
                            ^ item.v.wrapping_mul(2_246_822_519))
                            as f64)
                            * 1e-6;
                        let (sin_a, cos_a) = angle.sin_cos();
                        let step = 0.5 * item.d * mu;
                        let mx = step * cos_a;
                        let my = step * sin_a;
                        wx[i] -= mx;
                        wy[i] -= my;
                        wx[j] += mx;
                        wy[j] += my;
                        max_move = max_move.max(step);
                    }
                    continue;
                }
                let coeff = 0.5 * (norm - item.d) / norm * mu;
                let mx = dx * coeff;
                let my = dy * coeff;
                if mx.is_finite() && my.is_finite() {
                    wx[i] -= mx;
                    wy[i] -= my;
                    wx[j] += mx;
                    wy[j] += my;
                    max_move = max_move.max(mx.abs().max(my.abs()));
                }
            }

            if max_move <= 1e-12 {
                break;
            }
        }

        Ok(())
    }
}
