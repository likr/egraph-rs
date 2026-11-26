use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use egraph_core::{Layout2d, Scalar};

use super::solver::PfsPrimeOverlapRemoval;
use super::verify::bump_coord_s_to_clear;

#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    key: f64,
    node: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key.total_cmp(&other.key) == Ordering::Equal && self.node == other.node
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key
            .total_cmp(&other.key)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl<S: Scalar> PfsPrimeOverlapRemoval<S> {
    pub(super) fn resolve_residual_overlaps_order_preserving(&self, layout: &mut Layout2d<S>) {
        let n = self.num_nodes;
        if n < 2 {
            return;
        }
        let s_eps = S::EPSILON.to_f64();

        // Assign each node to its initial-X tie group so Phase 1 can identify
        // all pairs sharing `orig_x` in O(1).
        let mut x_order: Vec<usize> = (0..n).collect();
        x_order.sort_unstable_by(|&a, &b| {
            self.orig_x[a]
                .total_cmp(&self.orig_x[b])
                .then_with(|| a.cmp(&b))
        });
        let mut orig_x_group = vec![0usize; n];
        let mut orig_x_has_tie = vec![false; n];
        let mut x_groups: Vec<(usize, usize)> = Vec::new();
        let mut gx_id = 0usize;
        let mut pos = 0usize;
        while pos < n {
            let base = self.orig_x[x_order[pos]];
            let mut end = pos + 1;
            while end < n && self.orig_x[x_order[end]] == base {
                end += 1;
            }
            let is_tie = end - pos > 1;
            for &node in &x_order[pos..end] {
                orig_x_group[node] = gx_id;
                orig_x_has_tie[node] = is_tie;
            }
            x_groups.push((pos, end));
            gx_id += 1;
            pos = end;
        }

        // Phase 1: Global monotonic Y-sweep across `orig_y` groups.
        // Resolves every overlap between pairs with `orig_x[u] == orig_x[v]`
        // (including exact-coincident inputs) while preserving global `orig_y` order.
        let mut y_order: Vec<usize> = (0..n).collect();
        y_order.sort_unstable_by(|&a, &b| {
            self.orig_y[a]
                .total_cmp(&self.orig_y[b])
                .then_with(|| a.cmp(&b))
        });
        let mut y_groups: Vec<(usize, usize)> = Vec::new();
        pos = 0;
        while pos < n {
            let base = self.orig_y[y_order[pos]];
            let mut end = pos + 1;
            while end < n && self.orig_y[y_order[end]] == base {
                end += 1;
            }
            y_groups.push((pos, end));
            pos = end;
        }

        let x_cur_s: Vec<S> = layout.x().to_vec();
        let y_cur: Vec<f64> = layout.y().iter().map(|v| v.to_f64()).collect();
        let mut y_out_s: Vec<S> = layout.y().to_vec();

        // Sort nodes inside each `orig_y` tie group by current `y_cur` so exact-coincident
        // nodes are swept monotonically from bottom to top.
        for &(g_start, g_end) in &y_groups {
            y_order[g_start..g_end]
                .sort_unstable_by(|&a, &b| y_cur[a].total_cmp(&y_cur[b]).then_with(|| a.cmp(&b)));
        }

        let mut suffix_max_hh = vec![0.0_f64; y_groups.len()];
        let mut running_hh = 0.0_f64;
        for g_idx in (0..y_groups.len()).rev() {
            let (gs, ge) = y_groups[g_idx];
            for &node in &y_order[gs..ge] {
                running_hh = running_hh.max(self.eff_hh_req[node]);
            }
            suffix_max_hh[g_idx] = running_hh;
        }

        let mut cum_shift_y = 0.0_f64;
        let mut prev_group_max_y: Option<S> = None;
        let mut active_y: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();

        for (g_idx, &(gs, ge)) in y_groups.iter().enumerate() {
            let mut min_y_cur = f64::INFINITY;
            for &node in &y_order[gs..ge] {
                min_y_cur = min_y_cur.min(y_cur[node]);
            }
            if let Some(pmy) = prev_group_max_y {
                let needed_min = pmy.to_f64();
                if min_y_cur + cum_shift_y < needed_min {
                    cum_shift_y = needed_min - min_y_cur;
                }
            }
            let cand_min = match prev_group_max_y {
                Some(pmy) => (min_y_cur + cum_shift_y).max(pmy.to_f64()),
                None => min_y_cur + cum_shift_y,
            };
            let max_hh_rem = suffix_max_hh[g_idx];
            let pop_threshold =
                cand_min - max_hh_rem - 4.0 * (cand_min.abs() + max_hh_rem) * s_eps - 1e-12;
            while active_y
                .peek()
                .is_some_and(|Reverse(entry)| entry.key <= pop_threshold)
            {
                active_y.pop();
            }

            let mut curr_group_max_y: Option<S> = None;
            for &v in &y_order[gs..ge] {
                let floor_y = match (prev_group_max_y, curr_group_max_y) {
                    (Some(p), Some(c)) => Some(p.max(c)),
                    (Some(p), None) => Some(p),
                    (None, Some(c)) => Some(c),
                    (None, None) => None,
                };
                let mut c_s = floor_y.map_or_else(
                    || S::from_f64(y_cur[v] + cum_shift_y),
                    |fy| S::from_f64(y_cur[v] + cum_shift_y).max(fy),
                );

                if orig_x_has_tie[v] {
                    let xv_s = x_cur_s[v];
                    let xv_f = xv_s.to_f64();
                    let hw_v_s = self.eff_hw_s[v];
                    let hh_v_s = self.eff_hh_s[v];

                    // Re-scan active intervals whenever c_s bumps upward so BinaryHeap
                    // backing-Vec ordering can never skip an earlier interval.
                    loop {
                        let mut bumped = false;
                        for &Reverse(entry) in &active_y {
                            let u = entry.node;
                            if orig_x_group[u] != orig_x_group[v] {
                                continue;
                            }
                            let req_x = self.eff_hw_req[u] + self.eff_hw_req[v];
                            let req_y = self.eff_hh_req[u] + self.eff_hh_req[v];
                            if req_x <= 0.0 || req_y <= 0.0 {
                                continue;
                            }
                            let xu_s = x_cur_s[u];
                            let hw_u_s = self.eff_hw_s[u];
                            let x_ov = (xu_s.to_f64() - xv_f).abs() < req_x
                                || ((xu_s - hw_u_s) < (xv_s + hw_v_s)
                                    && (xv_s - hw_v_s) < (xu_s + hw_u_s));
                            if !x_ov {
                                continue;
                            }

                            let yu_s = y_out_s[u];
                            let yu_f = yu_s.to_f64();
                            let hh_u_s = self.eff_hh_s[u];
                            let y_ov = (c_s.to_f64() - yu_f).abs() < req_y
                                || ((yu_s - hh_u_s) < (c_s + hh_v_s)
                                    && (c_s - hh_v_s) < (yu_s + hh_u_s));
                            if y_ov {
                                let next_c_s =
                                    bump_coord_s_to_clear(c_s, yu_s, hh_u_s, hh_v_s, yu_f, req_y);
                                if next_c_s > c_s {
                                    c_s = next_c_s;
                                    bumped = true;
                                }
                            }
                        }
                        if !bumped {
                            break;
                        }
                    }

                    let shift_needed = c_s.to_f64() - y_cur[v];
                    if shift_needed > cum_shift_y {
                        cum_shift_y = shift_needed;
                    }
                    let y_top_padded = (c_s.to_f64() + self.eff_hh_req[v])
                        .max((c_s + hh_v_s).to_f64())
                        + 4.0 * (c_s.to_f64().abs() + self.eff_hh_req[v]) * s_eps
                        + 1e-12;
                    active_y.push(Reverse(HeapEntry {
                        key: y_top_padded,
                        node: v,
                    }));
                }

                y_out_s[v] = c_s;
                curr_group_max_y = Some(curr_group_max_y.map_or(c_s, |m| m.max(c_s)));
            }
            prev_group_max_y = curr_group_max_y;
        }
        layout.y_mut().copy_from_slice(&y_out_s);

        // Phase 2: Global monotonic X-sweep across `orig_x` groups with Y frozen.
        // Resolves all remaining overlaps (`orig_x[u] != orig_x[v]`) while preserving
        // global `orig_x` order.
        let x_cur: Vec<f64> = layout.x().iter().map(|v| v.to_f64()).collect();
        let mut x_out_s: Vec<S> = layout.x().to_vec();

        for &(g_start, g_end) in &x_groups {
            x_order[g_start..g_end]
                .sort_unstable_by(|&a, &b| x_cur[a].total_cmp(&x_cur[b]).then_with(|| a.cmp(&b)));
        }

        let mut suffix_max_hw = vec![0.0_f64; x_groups.len()];
        let mut running_hw = 0.0_f64;
        for g_idx in (0..x_groups.len()).rev() {
            let (gs, ge) = x_groups[g_idx];
            for &node in &x_order[gs..ge] {
                running_hw = running_hw.max(self.eff_hw_req[node]);
            }
            suffix_max_hw[g_idx] = running_hw;
        }

        let mut cum_shift_x = 0.0_f64;
        let mut prev_group_max_x: Option<S> = None;
        let mut active_x: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();

        for (g_idx, &(gs, ge)) in x_groups.iter().enumerate() {
            let mut min_x_cur = f64::INFINITY;
            for &node in &x_order[gs..ge] {
                min_x_cur = min_x_cur.min(x_cur[node]);
            }
            if let Some(pmx) = prev_group_max_x {
                let needed_min = pmx.to_f64();
                if min_x_cur + cum_shift_x < needed_min {
                    cum_shift_x = needed_min - min_x_cur;
                }
            }
            let cand_min = match prev_group_max_x {
                Some(pmx) => (min_x_cur + cum_shift_x).max(pmx.to_f64()),
                None => min_x_cur + cum_shift_x,
            };
            let max_hw_rem = suffix_max_hw[g_idx];
            let pop_threshold =
                cand_min - max_hw_rem - 4.0 * (cand_min.abs() + max_hw_rem) * s_eps - 1e-12;
            while active_x
                .peek()
                .is_some_and(|Reverse(entry)| entry.key <= pop_threshold)
            {
                active_x.pop();
            }

            let mut curr_group_max_x: Option<S> = None;
            for &v in &x_order[gs..ge] {
                let floor_x = match (prev_group_max_x, curr_group_max_x) {
                    (Some(p), Some(c)) => Some(p.max(c)),
                    (Some(p), None) => Some(p),
                    (None, Some(c)) => Some(c),
                    (None, None) => None,
                };
                let mut c_s = floor_x.map_or_else(
                    || S::from_f64(x_cur[v] + cum_shift_x),
                    |fx| S::from_f64(x_cur[v] + cum_shift_x).max(fx),
                );

                let yv_s = y_out_s[v];
                let yv_f = yv_s.to_f64();
                let hw_v_s = self.eff_hw_s[v];
                let hh_v_s = self.eff_hh_s[v];

                // Re-scan active intervals whenever c_s bumps rightward so BinaryHeap
                // backing-Vec ordering can never skip an earlier interval.
                loop {
                    let mut bumped = false;
                    for &Reverse(entry) in &active_x {
                        let u = entry.node;
                        let req_x = self.eff_hw_req[u] + self.eff_hw_req[v];
                        let req_y = self.eff_hh_req[u] + self.eff_hh_req[v];
                        if req_x <= 0.0 || req_y <= 0.0 {
                            continue;
                        }
                        let yu_s = y_out_s[u];
                        let hh_u_s = self.eff_hh_s[u];
                        let y_ov = (yu_s.to_f64() - yv_f).abs() < req_y
                            || ((yu_s - hh_u_s) < (yv_s + hh_v_s)
                                && (yv_s - hh_v_s) < (yu_s + hh_u_s));
                        if !y_ov {
                            continue;
                        }

                        let xu_s = x_out_s[u];
                        let xu_f = xu_s.to_f64();
                        let hw_u_s = self.eff_hw_s[u];
                        let x_ov = (c_s.to_f64() - xu_f).abs() < req_x
                            || ((xu_s - hw_u_s) < (c_s + hw_v_s)
                                && (c_s - hw_v_s) < (xu_s + hw_u_s));
                        if x_ov {
                            let next_c_s =
                                bump_coord_s_to_clear(c_s, xu_s, hw_u_s, hw_v_s, xu_f, req_x);
                            if next_c_s > c_s {
                                c_s = next_c_s;
                                bumped = true;
                            }
                        }
                    }
                    if !bumped {
                        break;
                    }
                }

                let shift_needed = c_s.to_f64() - x_cur[v];
                if shift_needed > cum_shift_x {
                    cum_shift_x = shift_needed;
                }
                x_out_s[v] = c_s;
                curr_group_max_x = Some(curr_group_max_x.map_or(c_s, |m| m.max(c_s)));

                let x_right_padded = (c_s.to_f64() + self.eff_hw_req[v])
                    .max((c_s + hw_v_s).to_f64())
                    + 4.0 * (c_s.to_f64().abs() + self.eff_hw_req[v]) * s_eps
                    + 1e-12;
                active_x.push(Reverse(HeapEntry {
                    key: x_right_padded,
                    node: v,
                }));
            }
            prev_group_max_x = curr_group_max_x;
        }
        layout.x_mut().copy_from_slice(&x_out_s);
    }
}
