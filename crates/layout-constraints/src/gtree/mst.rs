use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use egraph_core::{CoreError, CoreResult, Scalar};

use super::solver::GTreeOverlapRemoval;

#[derive(Debug, Clone, Copy)]
struct PrimEntry {
    cost: f64,
    t_factor: f64,
    parent: u32,
    child: u32,
}

impl PartialEq for PrimEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost.total_cmp(&other.cost) == Ordering::Equal
            && self.child == other.child
            && self.parent == other.parent
    }
}

impl Eq for PrimEntry {}

impl PartialOrd for PrimEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrimEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost
            .total_cmp(&other.cost)
            .then_with(|| self.child.cmp(&other.child))
            .then_with(|| self.parent.cmp(&other.parent))
    }
}

impl<S: Scalar> GTreeOverlapRemoval<S> {
    #[inline]
    fn edge_cost_and_expansion(&self, u: usize, v: usize) -> (f64, f64, bool) {
        let xs = self.work_layout.x();
        let ys = self.work_layout.y();

        let dx = (xs[u] - xs[v]).abs();
        let dy = (ys[u] - ys[v]).abs();
        let req_x = self.eff_hw_f64[u] + self.eff_hw_f64[v];
        let req_y = self.eff_hh_f64[u] + self.eff_hh_f64[v];

        let xu_s = S::from_f64(xs[u]);
        let xv_s = S::from_f64(xs[v]);
        let yu_s = S::from_f64(ys[u]);
        let yv_s = S::from_f64(ys[v]);
        let overlaps_in_s = (xu_s - self.eff_hw_s[u]) < (xv_s + self.eff_hw_s[v])
            && (xv_s - self.eff_hw_s[v]) < (xu_s + self.eff_hw_s[u])
            && (yu_s - self.eff_hh_s[u]) < (yv_s + self.eff_hh_s[v])
            && (yv_s - self.eff_hh_s[v]) < (yu_s + self.eff_hh_s[u]);

        let has_area = req_x > 0.0 && req_y > 0.0;
        let overlaps_f64 = dx < req_x && dy < req_y;

        if has_area && (overlaps_f64 || overlaps_in_s) {
            let sep_eps = 1.0 + 1e-6_f64.max(16.0 * S::EPSILON.to_f64());
            let ulp_x = (xs[u].abs().max(xs[v].abs()) + req_x) * (8.0 * S::EPSILON.to_f64());
            let ulp_y = (ys[u].abs().max(ys[v].abs()) + req_y) * (8.0 * S::EPSILON.to_f64());
            let target_x = req_x * sep_eps + ulp_x;
            let target_y = req_y * sep_eps + ulp_y;

            let tx = if dx > 1e-15 {
                target_x / dx
            } else {
                f64::INFINITY
            };
            let ty = if dy > 1e-15 {
                target_y / dy
            } else {
                f64::INFINITY
            };
            let t_ij = tx.min(ty).max(1.0);
            let t_ij = if t_ij.is_finite() { t_ij } else { 1e6 };
            let dist = dx.hypot(dy).max(1e-12);
            let cost = -(t_ij - 1.0) * dist;
            (cost, t_ij, true)
        } else {
            let gap_x = (dx - req_x).max(0.0);
            let gap_y = (dy - req_y).max(0.0);
            let cost = gap_x.hypot(gap_y);
            (cost, 1.0, false)
        }
    }

    pub(super) fn run_mst_growth_pass(&mut self, t_cap: Option<f64>) -> CoreResult<bool> {
        let n = self.num_nodes;
        self.deg.fill(0);
        let mut any_edge_overlap = false;

        for &(u, v) in &self.raw_edges {
            if u == v {
                continue;
            }
            self.deg[u as usize] += 1;
            self.deg[v as usize] += 1;
        }

        self.indptr[0] = 0;
        for i in 0..n {
            self.indptr[i + 1] = self.indptr[i]
                .checked_add(self.deg[i])
                .ok_or_else(|| CoreError::invalid_csr("GTree proximity CSR nnz overflow"))?;
        }
        let nnz = self.indptr[n] as usize;
        self.adj_to.resize(nnz, 0);
        self.adj_cost.resize(nnz, 0.0);
        self.adj_t.resize(nnz, 1.0);
        self.pos.copy_from_slice(&self.indptr[..n]);

        for idx in 0..self.raw_edges.len() {
            let (u32_u, u32_v) = self.raw_edges[idx];
            if u32_u == u32_v {
                continue;
            }
            let u = u32_u as usize;
            let v = u32_v as usize;
            let (cost, t_ij, is_ov) = self.edge_cost_and_expansion(u, v);
            if is_ov {
                any_edge_overlap = true;
            }

            let pu = self.pos[u] as usize;
            self.adj_to[pu] = u32_v;
            self.adj_cost[pu] = cost;
            self.adj_t[pu] = t_ij;
            self.pos[u] += 1;

            let pv = self.pos[v] as usize;
            self.adj_to[pv] = u32_u;
            self.adj_cost[pv] = cost;
            self.adj_t[pv] = t_ij;
            self.pos[v] += 1;
        }

        if !any_edge_overlap {
            return Ok(false);
        }

        // Choose deterministic root nearest to the layout centroid.
        let xs = self.work_layout.x();
        let ys = self.work_layout.y();
        let mean_x = xs.iter().sum::<f64>() / (n as f64);
        let mean_y = ys.iter().sum::<f64>() / (n as f64);
        let mut root = 0usize;
        let mut best_r_dist = f64::INFINITY;
        for i in 0..n {
            let d2 = (xs[i] - mean_x).powi(2) + (ys[i] - mean_y).powi(2);
            if d2 < best_r_dist {
                best_r_dist = d2;
                root = i;
            }
        }

        // Prim's MST rooted at `root`.
        self.visited.fill(false);
        self.tree_order.clear();
        let mut heap: BinaryHeap<Reverse<PrimEntry>> = BinaryHeap::new();

        self.visited[root] = true;
        let r_start = self.indptr[root] as usize;
        let r_end = self.indptr[root + 1] as usize;
        for p in r_start..r_end {
            heap.push(Reverse(PrimEntry {
                cost: self.adj_cost[p],
                t_factor: self.adj_t[p],
                parent: root as u32,
                child: self.adj_to[p],
            }));
        }

        while let Some(Reverse(entry)) = heap.pop() {
            let c = entry.child as usize;
            if self.visited[c] {
                continue;
            }
            self.visited[c] = true;
            self.tree_order
                .push((entry.parent, entry.child, entry.t_factor));
            let c_start = self.indptr[c] as usize;
            let c_end = self.indptr[c + 1] as usize;
            for p in c_start..c_end {
                let nb = self.adj_to[p];
                if !self.visited[nb as usize] {
                    heap.push(Reverse(PrimEntry {
                        cost: self.adj_cost[p],
                        t_factor: self.adj_t[p],
                        parent: entry.child,
                        child: nb,
                    }));
                }
            }
        }

        // Grow tree top-down in Prim discovery order.
        self.new_x[root] = xs[root];
        self.new_y[root] = ys[root];
        for &(parent, child, t_ij) in &self.tree_order {
            let p = parent as usize;
            let c = child as usize;
            let t_used = match t_cap {
                Some(cap) => t_ij.min(cap),
                None => t_ij,
            };
            self.new_x[c] = self.new_x[p] + t_used * (xs[c] - xs[p]);
            self.new_y[c] = self.new_y[p] + t_used * (ys[c] - ys[p]);
        }
        // Defensive fallback for any unvisited node (should never happen due to bridging).
        for i in 0..n {
            if !self.visited[i] {
                self.new_x[i] = xs[i];
                self.new_y[i] = ys[i];
            }
        }

        // Preserve centroid to prevent layout drift across passes.
        let mean_x_after = self.new_x.iter().sum::<f64>() / (n as f64);
        let mean_y_after = self.new_y.iter().sum::<f64>() / (n as f64);
        let shift_x = mean_x - mean_x_after;
        let shift_y = mean_y - mean_y_after;
        let (wx_mut, wy_mut) = self.work_layout.xy_mut();
        for i in 0..n {
            wx_mut[i] = self.new_x[i] + shift_x;
            wy_mut[i] = self.new_y[i] + shift_y;
        }

        Ok(true)
    }
}
