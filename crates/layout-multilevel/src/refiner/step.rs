use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use egraph_layout_sgd::sgd::{PackedSymPair, SgdScheduleConfig, SgdScheduleKind, SgdSolver};
use egraph_spatial_trees::StaticKdTree;

use super::SpatialIntruderRefiner;
use super::config::SpatialRefinerRoundStats;
use super::helpers::{
    MAX_WEIGHT_RATIO, compute_coords_std, compute_layout_scale, condition_weights_packed,
    fast_hypot_f64, has_common_non_hub_neighbor, push_packed_pair, sample_normal_pair,
};

impl<S: Scalar> IterativeSolver<S> for SpatialIntruderRefiner<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "SpatialIntruderRefiner::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if !self.scale_initialized {
            self.sync_coords_from_layout(layout);
            self.initial_scale = compute_layout_scale(&self.coords_f64, &self.u1, &self.v1);
            self.initial_std = compute_coords_std(&self.coords_f64);
            self.d_floor = (1e-9_f64).max(1e-3 * self.initial_scale);
            self.stats.initial_scale = self.initial_scale;
            self.stats.final_scale = self.initial_scale;
            self.stats.d_floor = self.d_floor;
            self.stats.scale_ratio = 1.0;
            self.scale_initialized = true;
        }

        if n <= 2 || self.u1.is_empty() || self.config.t_max == 0 || self.k_eff_round == 0 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        self.sync_coords_from_layout(layout);

        let r = self.round;
        let round_seed =
            (self.config.seed.wrapping_add((r as u64).wrapping_mul(1009))) & 0xFFFF_FFFF;
        let mut tree = StaticKdTree::<2, f64>::build(&self.coords_f64)?;

        // Single spatial k-NN pass (k_q = ku.max(1)) detects near-coincident nodes simultaneously.
        let tol_coinc = 1e-12 * self.initial_scale;
        let tol_coinc_sq = tol_coinc * tol_coinc;
        self.coincident.fill(false);
        let any_coincident = self.populate_spatial_knn(&tree, tol_coinc_sq, true)?;

        if any_coincident {
            let mut rng_jit = DeterministicRng::from_seed_u64(round_seed);
            let jit_std = 1e-4 * self.initial_scale;
            let (xs_mut, ys_mut) = layout.xy_mut();
            for u in 0..n {
                if self.coincident[u] {
                    let (jx, jy) = sample_normal_pair(&mut rng_jit, jit_std);
                    let nx = self.coords_f64[u][0] + jx;
                    let ny = self.coords_f64[u][1] + jy;
                    if nx.is_finite() && ny.is_finite() {
                        xs_mut[u] = S::from_f64(nx);
                        ys_mut[u] = S::from_f64(ny);
                        self.coords_f64[u] = [xs_mut[u].to_f64(), ys_mut[u].to_f64()];
                    }
                }
            }
            tree = StaticKdTree::<2, f64>::build(&self.coords_f64)?;
            let _ = self.populate_spatial_knn(&tree, tol_coinc_sq, false)?;
        }

        // Emit active pairs directly into a single PackedSymPair buffer ([FN/A2, A1, FP, SEP])
        // without an intermediate f64 pair double buffer.
        let mut packed: Vec<PackedSymPair<S>> = Vec::with_capacity(self.u1.len());
        let mut n_springs = 0usize;
        let mut n_hinges = 0usize;

        let sep_enabled = self.config.min_sep_ratio > 0.0 && self.config.w_sep > 0.0;
        let d_sep_tgt = (self.config.min_sep_ratio * self.initial_scale).max(self.d_floor);
        let mut sep_candidates: Vec<(u32, u32)> = Vec::new();
        let mut fp_covered_sep: Vec<(u32, u32)> = Vec::new();

        // 1. Tier 2: 2-hop False Negatives F- and/or 2-hop Anchors A2.
        let mut n_fn_2hop = 0usize;
        let mut n_anc_2hop = 0usize;
        if self.config.w_fn > 0.0 || self.config.w_anchor2 > 0.0 {
            let mut emit_2hop = |ui: usize, vi: usize, u_u32: u32, v_u32: u32| {
                let dx = self.coords_f64[ui][0] - self.coords_f64[vi][0];
                let dy = self.coords_f64[ui][1] - self.coords_f64[vi][1];
                let d2_curr = fast_hypot_f64(dx, dy);
                if sep_enabled && d2_curr < d_sep_tgt {
                    sep_candidates.push(if u_u32 < v_u32 {
                        (u_u32, v_u32)
                    } else {
                        (v_u32, u_u32)
                    });
                    return;
                }
                let d2_safe = d2_curr.max(self.d_floor);

                let mut emitted_fn = false;
                if self.config.w_fn > 0.0 && !self.in_s_both(ui, vi) {
                    let r_pair_fn = 0.5 * (self.r_knn[ui] + self.r_knn[vi]);
                    let d2_fn_tgt = self
                        .d_floor
                        .max((self.config.fn_shrink * d2_curr).max(0.9 * r_pair_fn));
                    if d2_curr > d2_fn_tgt {
                        emitted_fn = true;
                        let w =
                            (self.config.w_anchor2 + self.config.w_fn) / (d2_fn_tgt * d2_fn_tgt);
                        if push_packed_pair(&mut packed, u_u32, v_u32, false, d2_fn_tgt, w) {
                            n_springs += 1;
                            n_fn_2hop += 1;
                            if self.config.w_anchor2 > 0.0 {
                                n_anc_2hop += 1;
                            }
                        }
                    }
                }
                if !emitted_fn && self.config.w_anchor2 > 0.0 {
                    let w = self.config.w_anchor2 / (d2_safe * d2_safe);
                    if push_packed_pair(&mut packed, u_u32, v_u32, false, d2_safe, w) {
                        n_springs += 1;
                        n_anc_2hop += 1;
                    }
                }
            };

            if let Some(adj2) = &self.adj2 {
                for u in 0..n {
                    let u_u32 = u as u32;
                    for &v_u32 in adj2.neighbors(u) {
                        if u_u32 < v_u32 {
                            emit_2hop(u, v_u32 as usize, u_u32, v_u32);
                        }
                    }
                }
            } else {
                for (&u_u32, &v_u32) in self.u2_hub.iter().zip(self.v2_hub.iter()) {
                    emit_2hop(u_u32 as usize, v_u32 as usize, u_u32, v_u32);
                }
            }
        }

        // 2. Tier 1: 1-hop Anchors A1 (protect 1-hop lengths; gentle w_fn boost for elongated missing 1-hop).
        let mut n_1hop_fn = 0usize;
        let mut n_anc_1hop = 0usize;
        for (&u_u32, &v_u32) in self.u1.iter().zip(self.v1.iter()) {
            let ui = u_u32 as usize;
            let vi = v_u32 as usize;
            let dx = self.coords_f64[ui][0] - self.coords_f64[vi][0];
            let dy = self.coords_f64[ui][1] - self.coords_f64[vi][1];
            let d1_curr = fast_hypot_f64(dx, dy);
            if sep_enabled && d1_curr < d_sep_tgt {
                sep_candidates.push(if u_u32 < v_u32 {
                    (u_u32, v_u32)
                } else {
                    (v_u32, u_u32)
                });
                continue;
            }
            let d1_safe = d1_curr.max(self.d_floor);

            let (d1, w1) = if self.config.w_fn > 0.0 {
                let in_both = self.in_s_both(ui, vi);
                let r_pair_1 = 0.5 * (self.r_knn[ui] + self.r_knn[vi]);
                let d1_fn_tgt = self
                    .d_floor
                    .max((self.config.fn_shrink * d1_curr).max(0.9 * r_pair_1));
                let is_1hop_fn = !in_both && (d1_curr > d1_fn_tgt);
                if is_1hop_fn {
                    n_1hop_fn += 1;
                    (
                        d1_fn_tgt,
                        (self.config.w_anchor + self.config.w_fn) / (d1_fn_tgt * d1_fn_tgt),
                    )
                } else {
                    (d1_safe, self.config.w_anchor / (d1_safe * d1_safe))
                }
            } else {
                (d1_safe, self.config.w_anchor / (d1_safe * d1_safe))
            };

            if push_packed_pair(&mut packed, u_u32, v_u32, false, d1, w1) {
                n_springs += 1;
                n_anc_1hop += 1;
            }
        }

        // 3. Tier 3: False positives F+ (strictly > 2-hop, one-sided hinge d_curr < d_fp_target).
        let mut n_fp = 0usize;
        if self.config.w_fp > 0.0 && !self.s_indices.is_empty() {
            let fp_is_hinge = self.config.hinge_fp;
            let mut fp_candidates: Vec<(u32, u32)> = Vec::new();
            for u in 0..n {
                let ku = self.row_k(u);
                if ku == 0 {
                    continue;
                }
                let s_start = self.s_indptr[u];
                let s_row = &self.s_indices[s_start..s_start + ku];
                self.stamp.clear();
                self.stamp.mark(u);
                let n1_u = self.adj1.neighbors(u);
                for &w in n1_u {
                    self.stamp.mark(w as usize);
                }
                if let Some(adj2) = &self.adj2 {
                    for &w in adj2.neighbors(u) {
                        self.stamp.mark(w as usize);
                    }
                    for &v in s_row {
                        if !self.stamp.is_visited(v as usize) {
                            let u_u32 = u as u32;
                            fp_candidates.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                        }
                    }
                } else {
                    for &v in s_row {
                        let vi = v as usize;
                        if self.stamp.is_visited(vi) {
                            continue;
                        }
                        let n1_v = self.adj1.neighbors(vi);
                        if !has_common_non_hub_neighbor(
                            n1_u,
                            n1_v,
                            &self.stamp,
                            &self.deg,
                            self.non_hub_thresh,
                        ) {
                            let u_u32 = u as u32;
                            fp_candidates.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                        }
                    }
                }
            }

            fp_candidates.sort_unstable();
            fp_candidates.dedup();
            for (u_u32, v_u32) in fp_candidates {
                let ui = u_u32 as usize;
                let vi = v_u32 as usize;
                let r_pair_fp = if self.row_k(ui) == 0 {
                    self.r_knn[vi]
                } else if self.row_k(vi) == 0 {
                    self.r_knn[ui]
                } else {
                    0.5 * (self.r_knn[ui] + self.r_knn[vi])
                };
                let d_fp_tgt = (self.config.margin_scale * r_pair_fp).max(self.d_floor);
                let dx = self.coords_f64[ui][0] - self.coords_f64[vi][0];
                let dy = self.coords_f64[ui][1] - self.coords_f64[vi][1];
                let d_fp_curr = fast_hypot_f64(dx, dy);
                if d_fp_curr < d_fp_tgt {
                    // Defer to the stronger separation hinge when sep_enabled and d_fp_tgt < d_sep_tgt.
                    if sep_enabled && d_fp_curr < d_sep_tgt && d_fp_tgt < d_sep_tgt {
                        sep_candidates.push((u_u32, v_u32));
                        continue;
                    }
                    let w = self.config.w_fp / (d_fp_tgt * d_fp_tgt);
                    if push_packed_pair(&mut packed, u_u32, v_u32, fp_is_hinge, d_fp_tgt, w) {
                        n_fp += 1;
                        if fp_is_hinge {
                            n_hinges += 1;
                        } else {
                            n_springs += 1;
                        }
                        if sep_enabled && d_fp_curr < d_sep_tgt {
                            fp_covered_sep.push((u_u32, v_u32));
                        }
                    }
                }
            }
        }

        // 4. Opt-in short-range minimum separation hinges (cures closed-twin collapse & zero-length edges).
        let mut n_sep = 0usize;
        if sep_enabled {
            if !self.s_indices.is_empty() {
                for u in 0..n {
                    let ku = self.row_k(u);
                    if ku == 0 {
                        continue;
                    }
                    let s_start = self.s_indptr[u];
                    let s_row = &self.s_indices[s_start..s_start + ku];
                    let ux = self.coords_f64[u][0];
                    let uy = self.coords_f64[u][1];
                    let u_u32 = u as u32;
                    for &v in s_row {
                        let vi = v as usize;
                        let d_curr = fast_hypot_f64(
                            ux - self.coords_f64[vi][0],
                            uy - self.coords_f64[vi][1],
                        );
                        if d_curr < d_sep_tgt {
                            sep_candidates.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                        }
                    }
                }
            }
            sep_candidates.sort_unstable();
            sep_candidates.dedup();
            let w_sep_pair = self.config.w_sep / (d_sep_tgt * d_sep_tgt);
            for (u_u32, v_u32) in sep_candidates {
                if !fp_covered_sep.is_empty()
                    && fp_covered_sep.binary_search(&(u_u32, v_u32)).is_ok()
                {
                    continue;
                }
                let ui = u_u32 as usize;
                let vi = v_u32 as usize;
                let d_curr = fast_hypot_f64(
                    self.coords_f64[ui][0] - self.coords_f64[vi][0],
                    self.coords_f64[ui][1] - self.coords_f64[vi][1],
                );
                if d_curr < d_sep_tgt
                    && push_packed_pair(&mut packed, u_u32, v_u32, true, d_sep_tgt, w_sep_pair)
                {
                    n_sep += 1;
                    n_hinges += 1;
                }
            }
        }

        let n_fn = n_fn_2hop + n_1hop_fn;
        let n_anc = n_anc_1hop + n_anc_2hop;
        let n_tot_valid = packed.len();

        // Early stop when zero active intruders/stragglers/collapsed pairs exist so clean layouts remain untouched.
        if (n_fp == 0 && n_fn == 0 && n_sep == 0) || n_tot_valid == 0 {
            self.stats.round_stats.push(SpatialRefinerRoundStats {
                round: r + 1,
                n_anchors: 0,
                n_fn: 0,
                n_fp: 0,
                n_sep: 0,
                n_pairs_total: 0,
                skipped: true,
            });
            self.update_final_scale_stats();
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        condition_weights_packed(&mut packed, MAX_WEIGHT_RATIO);

        let scale_r = self.config.round_decay.powi(r as i32);
        let sched = SgdScheduleConfig {
            t_max: self.config.t_max,
            eps: self.config.eps,
            kind: SgdScheduleKind::Exponential,
            step_scale: scale_r,
        };

        let mut sgd = SgdSolver::<S>::from_packed_symmetric(
            n, packed, n_springs, n_hinges, sched, round_seed,
        )?;
        sgd.run(layout)?;

        self.sync_coords_from_layout(layout);
        self.update_final_scale_stats();

        self.stats.rounds_executed += 1;
        self.stats.n_anchors_total += n_anc;
        self.stats.n_fn_total += n_fn;
        self.stats.n_fp_total += n_fp;
        self.stats.n_sep_total += n_sep;
        self.stats.n_pairs_total += n_tot_valid;
        self.stats.round_stats.push(SpatialRefinerRoundStats {
            round: r + 1,
            n_anchors: n_anc,
            n_fn,
            n_fp,
            n_sep,
            n_pairs_total: n_tot_valid,
            skipped: false,
        });

        self.round += 1;
        if self.config.k_cap_decay < 1.0 {
            let decayed = ((self.current_k_cap as f64) * self.config.k_cap_decay)
                .round()
                .max(1.0) as usize;
            let min_k = self.config.k_cap_min.min(self.k_cap_capacity);
            self.current_k_cap = decayed.clamp(min_k, self.k_cap_capacity);
            self.k_eff_round = if n > 1 {
                self.current_k_cap.min(n - 1)
            } else {
                0
            };
        }
        self.status = if self.round >= self.config.rounds {
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
        self.round
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.config.rounds
    }
}
