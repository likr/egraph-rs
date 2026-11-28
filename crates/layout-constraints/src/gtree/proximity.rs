use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};
use egraph_spatial_trees::Delaunay2d;

use super::config::MIN_EDGE_DIST;
use super::solver::GTreeOverlapRemoval;

#[inline]
fn uf_find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let p = parent[x as usize];
        parent[x as usize] = parent[p as usize];
        x = p;
    }
    x
}

#[inline]
fn uf_union(parent: &mut [u32], rank: &mut [u8], a: u32, b: u32) -> bool {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra == rb {
        return false;
    }
    if rank[ra as usize] < rank[rb as usize] {
        parent[ra as usize] = rb;
    } else if rank[ra as usize] > rank[rb as usize] {
        parent[rb as usize] = ra;
    } else {
        parent[rb as usize] = ra;
        rank[ra as usize] += 1;
    }
    true
}

impl<S: Scalar> GTreeOverlapRemoval<S> {
    pub(super) fn jitter_coincident_points(&mut self) {
        let n = self.num_nodes;
        if n < 2 {
            return;
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
                let coord_ulp = base_x.abs().max(base_y.abs()) * S::EPSILON.to_f64();
                for k in (i + 1)..j {
                    let idx = self.sort_pts[k].2 as usize;
                    let extent = self.eff_hw_f64[idx].max(self.eff_hh_f64[idx]).max(1e-2);
                    let jitter_scale = (extent * 0.05).max(4.0 * coord_ulp).max(1e-5);
                    let angle = self.rng.next_f64() * std::f64::consts::TAU;
                    let radius = jitter_scale * (0.25 + 0.75 * self.rng.next_f64());
                    let (sin_a, cos_a) = angle.sin_cos();
                    xs_mut[idx] += radius * cos_a;
                    ys_mut[idx] += radius * sin_a;
                }
            }
            i = j;
        }
    }

    fn bridge_disconnected_components(&mut self) {
        let n = self.num_nodes;
        if n < 2 {
            return;
        }
        for i in 0..n {
            self.uf_parent[i] = i as u32;
            self.uf_rank[i] = 0;
        }
        let mut merges = 0usize;
        for idx in 0..self.raw_edges.len() {
            let (u, v) = self.raw_edges[idx];
            if uf_union(&mut self.uf_parent, &mut self.uf_rank, u, v) {
                merges += 1;
                if merges + 1 == n {
                    return;
                }
            }
        }

        // Collect one deterministic representative per connected component.
        self.comp_rep.clear();
        let xs = self.work_layout.x();
        let ys = self.work_layout.y();
        for i in 0..n {
            if self.uf_parent[i] as usize == i {
                self.comp_rep.push((i as u32, xs[i], ys[i]));
            }
        }
        let c = self.comp_rep.len();
        if c <= 1 {
            return;
        }

        if c <= 512 {
            // Exact Prim MST across component representatives.
            let mut in_tree = vec![false; c];
            let mut min_dist = vec![f64::INFINITY; c];
            let mut min_src = vec![0usize; c];
            in_tree[0] = true;
            for j in 1..c {
                let dx = self.comp_rep[j].1 - self.comp_rep[0].1;
                let dy = self.comp_rep[j].2 - self.comp_rep[0].2;
                min_dist[j] = dx * dx + dy * dy;
                min_src[j] = 0;
            }
            for _ in 1..c {
                let mut best_j = usize::MAX;
                let mut best_d = f64::INFINITY;
                for j in 0..c {
                    if !in_tree[j] && min_dist[j] < best_d {
                        best_d = min_dist[j];
                        best_j = j;
                    }
                }
                if best_j == usize::MAX {
                    break;
                }
                in_tree[best_j] = true;
                let u = self.comp_rep[min_src[best_j]].0;
                let v = self.comp_rep[best_j].0;
                self.raw_edges.push(if u < v { (u, v) } else { (v, u) });
                for j in 0..c {
                    if !in_tree[j] {
                        let dx = self.comp_rep[j].1 - self.comp_rep[best_j].1;
                        let dy = self.comp_rep[j].2 - self.comp_rep[best_j].2;
                        let d2 = dx * dx + dy * dy;
                        if d2 < min_dist[j] {
                            min_dist[j] = d2;
                            min_src[j] = best_j;
                        }
                    }
                }
            }
        } else {
            // Sort representatives spatially and link consecutive representatives in O(C log C).
            self.comp_rep.sort_unstable_by(|a, b| {
                a.1.total_cmp(&b.1)
                    .then_with(|| a.2.total_cmp(&b.2))
                    .then_with(|| a.0.cmp(&b.0))
            });
            for w in self.comp_rep.windows(2) {
                let u = w[0].0;
                let v = w[1].0;
                self.raw_edges.push(if u < v { (u, v) } else { (v, u) });
            }
        }
    }

    pub(super) fn build_proximity_graph(&mut self) -> CoreResult<()> {
        self.raw_edges.clear();

        Delaunay2d::triangulate_edges_into(&self.work_layout, &mut self.raw_edges)?;

        self.bridge_disconnected_components();
        self.raw_edges.sort_unstable();
        self.raw_edges.dedup();
        Ok(())
    }

    pub(super) fn copy_layout_to_work(&mut self, layout: &Layout2d<S>) {
        let xs = layout.x();
        let ys = layout.y();
        let (wx, wy) = self.work_layout.xy_mut();
        for i in 0..self.num_nodes {
            wx[i] = xs[i].to_f64();
            wy[i] = ys[i].to_f64();
        }
    }

    pub(super) fn copy_work_to_layout(&self, layout: &mut Layout2d<S>) -> CoreResult<()> {
        let wx = self.work_layout.x();
        let wy = self.work_layout.y();
        let (xs_mut, ys_mut) = layout.xy_mut();
        for i in 0..self.num_nodes {
            let nx = S::from_f64(wx[i]);
            let ny = S::from_f64(wy[i]);
            if !nx.is_finite() || !ny.is_finite() {
                return Err(CoreError::numerical_breakdown(
                    "GTree produced non-finite coordinates",
                ));
            }
            xs_mut[i] = nx;
            ys_mut[i] = ny;
        }
        Ok(())
    }
}
