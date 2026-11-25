//! Nachmanson et al. (2016) GTree proximity-MST node overlap removal solver.
//!
//! Implements GTree ("Node Overlap Removal by Growing a Tree", Nachmanson, Nocaj, Bereg,
//! Zhang & Holroyd, *Graph Drawing* 2016) using `Delaunay2d` (under `feature = "delaunay"`) or
//! `StaticKdTree` $k$-NN + connectivity bridging fallback (under `--no-default-features`),
//! combined with Prim's minimum spanning tree and $O(N \log N + |\mathcal{O}|)$
//! [`RectangleOverlapSweep`] Stage-2 cleanup to guarantee strictly `0` overlapping rectangles.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
#[cfg(feature = "delaunay")]
use egraph_spatial::Delaunay2d;
use egraph_spatial::RectangleOverlapSweep;
#[cfg(not(feature = "delaunay"))]
use egraph_spatial::{KdQueryScratch, StaticKdTree};

#[cfg(not(feature = "delaunay"))]
const K_FALLBACK_NEIGHBORS: usize = 12;

const MIN_EDGE_DIST: f64 = 1e-9;
const MAX_STAGE2_FINAL_ITERS: usize = 32;

/// Hyperparameters for [`GTreeOverlapRemoval`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GTreeConfig {
    /// Maximum number of outer GTree iterations (`default: 20`).
    pub max_iterations: usize,
    /// Extra minimum clearance added between rectangle borders (`default: 0.0`).
    pub padding: f64,
    /// Optional compact per-step expansion cap $t_{\max} > 1.0$ (`default: None`).
    pub compact_t_max: Option<f64>,
    /// Explicit deterministic seed used for coincident-point jitter (`default: 42`).
    pub seed: u64,
}

impl Default for GTreeConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            padding: 0.0,
            compact_t_max: None,
            seed: 42,
        }
    }
}

impl GTreeConfig {
    /// Validates all GTree configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.padding.is_finite() {
            return Err(CoreError::non_finite("GTreeConfig::padding"));
        }
        if self.padding < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "GTreeConfig::padding must be >= 0.0, got {}",
                self.padding
            )));
        }
        if let Some(t_max) = self.compact_t_max {
            if !t_max.is_finite() {
                return Err(CoreError::non_finite("GTreeConfig::compact_t_max"));
            }
            if t_max <= 1.0 {
                return Err(CoreError::invalid_input(format!(
                    "GTreeConfig::compact_t_max must be > 1.0, got {t_max}"
                )));
            }
        }
        Ok(())
    }
}

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

/// Cooperative Nachmanson et al. (2016) GTree proximity-MST node overlap removal solver.
///
/// Each [`IterativeSolver::step`] executes one Stage-1 (proximity-graph MST) or Stage-2
/// (global sweep-augmented MST) tree-growth iteration and updates
/// [`GTreeOverlapRemoval::last_overlap_count`]. Reaching `max_iterations` with residual
/// overlaps triggers bounded Stage-2 sweep-MST cleanup and deterministic QPSC fallback so
/// converged layouts strictly have `0` overlapping rectangles.
#[derive(Debug, Clone)]
pub struct GTreeOverlapRemoval<S: Scalar = f64> {
    config: GTreeConfig,
    num_nodes: usize,
    eff_hw_f64: Vec<f64>,
    eff_hh_f64: Vec<f64>,
    eff_hw_s: Vec<S>,
    eff_hh_s: Vec<S>,
    rng: DeterministicRng,
    work_layout: Layout2d<f64>,
    new_x: Vec<f64>,
    new_y: Vec<f64>,
    sort_pts: Vec<(f64, f64, u32)>,
    raw_edges: Vec<(u32, u32)>,
    // Flat CSR adjacency for Prim's MST.
    deg: Vec<u32>,
    indptr: Vec<u32>,
    pos: Vec<u32>,
    adj_to: Vec<u32>,
    adj_cost: Vec<f64>,
    adj_t: Vec<f64>,
    visited: Vec<bool>,
    tree_order: Vec<(u32, u32, f64)>,
    uf_parent: Vec<u32>,
    uf_rank: Vec<u8>,
    comp_rep: Vec<(u32, f64, f64)>,
    #[cfg(not(feature = "delaunay"))]
    kd_scratch: KdQueryScratch,
    #[cfg(not(feature = "delaunay"))]
    knn_out: Vec<(usize, f64)>,
    iter: usize,
    stage2_active: bool,
    last_overlap_count: usize,
    status: SolverStatus,
    _marker: PhantomData<S>,
}

impl<S: Scalar> GTreeOverlapRemoval<S> {
    /// Creates a [`GTreeOverlapRemoval`] solver for rectangular nodes with per-node half-widths
    /// and half-heights.
    pub fn new(half_widths: Vec<S>, half_heights: Vec<S>, config: GTreeConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = half_widths.len();
        if half_heights.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                half_heights.len(),
                "GTreeOverlapRemoval::new::half_heights",
            ));
        }
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "GTreeOverlapRemoval node count exceeds u32::MAX",
            ));
        }

        let pad_half = 0.5 * config.padding;
        let mut eff_hw_f64 = Vec::with_capacity(n);
        let mut eff_hh_f64 = Vec::with_capacity(n);
        let mut eff_hw_s = Vec::with_capacity(n);
        let mut eff_hh_s = Vec::with_capacity(n);

        for i in 0..n {
            let hw = half_widths[i];
            let hh = half_heights[i];
            if !hw.is_finite() || !hh.is_finite() {
                return Err(CoreError::non_finite("GTreeOverlapRemoval::new::extents"));
            }
            if hw < S::ZERO || hh < S::ZERO {
                return Err(CoreError::invalid_input(
                    "half_widths and half_heights must be >= 0",
                ));
            }
            let hw_f = hw.to_f64() + pad_half;
            let hh_f = hh.to_f64() + pad_half;
            let hw_s = S::from_f64(hw_f);
            let hh_s = S::from_f64(hh_f);
            if !hw_s.is_finite() || !hh_s.is_finite() {
                return Err(CoreError::non_finite(
                    "GTreeOverlapRemoval::new::padded_extents",
                ));
            }
            eff_hw_f64.push(hw_f);
            eff_hh_f64.push(hh_f);
            eff_hw_s.push(hw_s);
            eff_hh_s.push(hh_s);
        }

        let status = if config.max_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            num_nodes: n,
            eff_hw_f64,
            eff_hh_f64,
            eff_hw_s,
            eff_hh_s,
            rng: DeterministicRng::from_seed_u64(config.seed),
            work_layout: Layout2d::zeros(n),
            new_x: vec![0.0; n],
            new_y: vec![0.0; n],
            sort_pts: Vec::with_capacity(n),
            raw_edges: Vec::new(),
            deg: vec![0; n],
            indptr: vec![0; n + 1],
            pos: vec![0; n],
            adj_to: Vec::new(),
            adj_cost: Vec::new(),
            adj_t: Vec::new(),
            visited: vec![false; n],
            tree_order: Vec::with_capacity(n.saturating_sub(1)),
            uf_parent: (0..n as u32).collect(),
            uf_rank: vec![0; n],
            comp_rep: Vec::new(),
            #[cfg(not(feature = "delaunay"))]
            kd_scratch: KdQueryScratch::with_capacity(K_FALLBACK_NEIGHBORS),
            #[cfg(not(feature = "delaunay"))]
            knn_out: Vec::with_capacity(K_FALLBACK_NEIGHBORS),
            iter: 0,
            stage2_active: false,
            last_overlap_count: 0,
            status,
            _marker: PhantomData,
        })
    }

    /// Creates a [`GTreeOverlapRemoval`] solver for circular/square nodes with equal half-width
    /// and half-height given by `radii`.
    pub fn from_radii(radii: Vec<S>, config: GTreeConfig) -> CoreResult<Self> {
        let half_heights = radii.clone();
        Self::new(radii, half_heights, config)
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &GTreeConfig {
        &self.config
    }

    /// Returns the number of vertices.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of overlapping rectangle pairs recorded at the most recent step.
    #[inline]
    pub fn last_overlap_count(&self) -> usize {
        self.last_overlap_count
    }

    fn jitter_coincident_points(&mut self) {
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

    fn build_proximity_graph(&mut self) -> CoreResult<()> {
        self.raw_edges.clear();

        #[cfg(feature = "delaunay")]
        {
            Delaunay2d::triangulate_edges_into(&self.work_layout, &mut self.raw_edges)?;
        }

        #[cfg(not(feature = "delaunay"))]
        {
            let n = self.num_nodes;
            let tree = StaticKdTree::<2, f64>::from_layout(&self.work_layout)?;
            let k = K_FALLBACK_NEIGHBORS.min(n.saturating_sub(1));
            for u in 0..n {
                tree.k_nearest_excluding_self_with_scratch(
                    u,
                    k,
                    &mut self.kd_scratch,
                    &mut self.knn_out,
                )?;
                let u32_u = u as u32;
                for &(v, _) in &self.knn_out {
                    let v32 = v as u32;
                    self.raw_edges.push(if u32_u < v32 {
                        (u32_u, v32)
                    } else {
                        (v32, u32_u)
                    });
                }
            }
        }

        self.bridge_disconnected_components();
        self.raw_edges.sort_unstable();
        self.raw_edges.dedup();
        Ok(())
    }

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

    fn run_mst_growth_pass(&mut self, t_cap: Option<f64>) -> CoreResult<bool> {
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

    fn copy_layout_to_work(&mut self, layout: &Layout2d<S>) {
        let xs = layout.x();
        let ys = layout.y();
        let (wx, wy) = self.work_layout.xy_mut();
        for i in 0..self.num_nodes {
            wx[i] = xs[i].to_f64();
            wy[i] = ys[i].to_f64();
        }
    }

    fn copy_work_to_layout(&self, layout: &mut Layout2d<S>) {
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

    fn execute_one_round(
        &mut self,
        layout: &mut Layout2d<S>,
        force_stage2: bool,
        t_cap: Option<f64>,
    ) -> CoreResult<usize> {
        self.copy_layout_to_work(layout);
        self.jitter_coincident_points();
        self.build_proximity_graph()?;

        if !self.stage2_active && !force_stage2 {
            // Stage 1: proximity-graph MST + tree growth.
            let moved = self.run_mst_growth_pass(t_cap)?;
            if moved {
                self.copy_work_to_layout(layout);
                let overlaps = RectangleOverlapSweep::find_layout_overlaps(
                    layout,
                    &self.eff_hw_s,
                    &self.eff_hh_s,
                )?;
                if overlaps.is_empty() {
                    return Ok(0);
                }
                return Ok(overlaps.len());
            }
            // No proximity edge overlaps -> transition into Stage 2.
            self.stage2_active = true;
        }

        // Stage 2: find all remaining overlaps via scan-line sweep and add to G_P.
        let sweep_overlaps =
            RectangleOverlapSweep::find_layout_overlaps(layout, &self.eff_hw_s, &self.eff_hh_s)?;
        if sweep_overlaps.is_empty() {
            return Ok(0);
        }

        for (u, v) in sweep_overlaps {
            let a = u as u32;
            let b = v as u32;
            self.raw_edges.push(if a < b { (a, b) } else { (b, a) });
        }
        self.raw_edges.sort_unstable();
        self.raw_edges.dedup();

        let _ = self.run_mst_growth_pass(t_cap)?;
        self.copy_work_to_layout(layout);
        let remaining =
            RectangleOverlapSweep::find_layout_overlaps(layout, &self.eff_hw_s, &self.eff_hh_s)?;
        Ok(remaining.len())
    }
}

impl<S: Scalar> IterativeSolver<S> for GTreeOverlapRemoval<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "GTreeOverlapRemoval::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.last_overlap_count = 0;
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        if self.iter == 0 {
            let initial_overlaps = RectangleOverlapSweep::find_layout_overlaps(
                layout,
                &self.eff_hw_s,
                &self.eff_hh_s,
            )?;
            if initial_overlaps.is_empty() {
                self.last_overlap_count = 0;
                self.status = SolverStatus::Converged;
                return Ok(self.status);
            }
        }

        let mut ov_count = self.execute_one_round(layout, false, self.config.compact_t_max)?;
        self.iter += 1;

        if ov_count > 0 && self.iter >= self.config.max_iterations {
            // Reached iteration cap (e.g., compact_t_max or complex configuration) with remaining
            // overlaps: run uncapped Stage-2 sweep-MST rounds and QPSC guard before finishing.
            self.stage2_active = true;
            for _ in 0..MAX_STAGE2_FINAL_ITERS {
                if ov_count == 0 {
                    break;
                }
                ov_count = self.execute_one_round(layout, true, None)?;
            }
            if ov_count > 0 {
                crate::constraints::project_rectangle_no_overlap_2d(
                    layout,
                    &self.eff_hw_s,
                    &self.eff_hh_s,
                )?;
                ov_count = RectangleOverlapSweep::find_layout_overlaps(
                    layout,
                    &self.eff_hw_s,
                    &self.eff_hh_s,
                )?
                .len();
            }
        }

        self.last_overlap_count = ov_count;
        self.status = if ov_count == 0 {
            SolverStatus::Converged
        } else if self.iter >= self.config.max_iterations {
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
        self.iter
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.config.max_iterations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_core::DeterministicRng;

    fn assert_no_overlaps<S: Scalar>(layout: &Layout2d<S>, hw: &[S], hh: &[S]) {
        let overlaps = RectangleOverlapSweep::find_layout_overlaps(layout, hw, hh).unwrap();
        assert!(
            overlaps.is_empty(),
            "expected 0 overlaps, found {}: {:?}",
            overlaps.len(),
            &overlaps[..overlaps.len().min(5)]
        );
    }

    #[test]
    fn test_gtree_validation_and_degenerate_inputs() {
        assert!(matches!(
            GTreeConfig {
                padding: -0.1,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            GTreeConfig {
                padding: f64::NAN,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            GTreeConfig {
                compact_t_max: Some(1.0),
                ..Default::default()
            }
            .validate(),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            GTreeConfig {
                compact_t_max: Some(f64::INFINITY),
                ..Default::default()
            }
            .validate(),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            GTreeOverlapRemoval::new(vec![1.0], vec![1.0, 2.0], GTreeConfig::default()),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            GTreeOverlapRemoval::new(vec![-1.0], vec![1.0], GTreeConfig::default()),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            GTreeOverlapRemoval::new(vec![f64::NAN], vec![1.0], GTreeConfig::default()),
            Err(CoreError::NonFiniteValue { .. })
        ));

        // Empty (N=0) and singleton (N=1).
        for n in [0usize, 1] {
            let mut solver =
                GTreeOverlapRemoval::<f64>::from_radii(vec![1.0; n], GTreeConfig::default())
                    .unwrap();
            let mut layout = Layout2d::zeros(n);
            assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Converged);
            assert_eq!(solver.last_overlap_count(), 0);
        }

        // Layout mismatch and non-finite coordinates.
        let mut solver =
            GTreeOverlapRemoval::<f64>::from_radii(vec![1.0; 2], GTreeConfig::default()).unwrap();
        let mut wrong = Layout2d::zeros(3);
        assert!(matches!(
            solver.step(&mut wrong),
            Err(CoreError::DimensionMismatch { .. })
        ));
        let mut bad = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 0.0]).unwrap();
        assert!(matches!(
            solver.step(&mut bad),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn test_gtree_zero_overlaps_f64_f32_and_step_eq_run() {
        let n = 48usize;
        let mut rng = DeterministicRng::from_seed_u64(12345);
        let mut xs64 = Vec::with_capacity(n);
        let mut ys64 = Vec::with_capacity(n);
        let mut hw64 = Vec::with_capacity(n);
        let mut hh64 = Vec::with_capacity(n);
        for i in 0..n {
            // Clustered overlapping layout with varied box aspect ratios.
            let cluster_x = if i % 2 == 0 { 0.0 } else { 20.0 };
            xs64.push(cluster_x + (rng.next_f64() - 0.5) * 2.0);
            ys64.push((rng.next_f64() - 0.5) * 2.0);
            hw64.push(0.8 + (i % 5) as f64 * 0.4);
            hh64.push(0.6 + ((i + 2) % 4) as f64 * 0.35);
        }

        let cfg = GTreeConfig {
            max_iterations: 20,
            padding: 0.2,
            compact_t_max: None,
            seed: 42,
        };
        let pad_half = 0.5 * cfg.padding;
        let eff_hw64: Vec<f64> = hw64.iter().map(|&w| w + pad_half).collect();
        let eff_hh64: Vec<f64> = hh64.iter().map(|&h| h + pad_half).collect();

        let mut l64_run = Layout2d::from_xy(xs64.clone(), ys64.clone()).unwrap();
        let mut l64_step = l64_run.clone();
        let mut s64_run = GTreeOverlapRemoval::<f64>::new(hw64.clone(), hh64.clone(), cfg).unwrap();
        let mut s64_step =
            GTreeOverlapRemoval::<f64>::new(hw64.clone(), hh64.clone(), cfg).unwrap();

        assert_eq!(s64_run.run(&mut l64_run).unwrap(), SolverStatus::Converged);
        assert_eq!(s64_run.last_overlap_count(), 0);
        assert_no_overlaps(&l64_run, &eff_hw64, &eff_hh64);

        s64_step.run_steps(&mut l64_step, 1).unwrap();
        while !s64_step.status().is_finished() {
            s64_step.step(&mut l64_step).unwrap();
        }
        assert_eq!(s64_step.status(), SolverStatus::Converged);
        for i in 0..n {
            assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
            assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
        }

        // Compact mode & f32 precision.
        let cfg_compact = GTreeConfig {
            max_iterations: 10,
            padding: 0.1,
            compact_t_max: Some(1.5),
            seed: 77,
        };
        let xs32: Vec<f32> = xs64.iter().map(|&v| v as f32 + 500.0).collect();
        let ys32: Vec<f32> = ys64.iter().map(|&v| v as f32 - 500.0).collect();
        let hw32: Vec<f32> = hw64.iter().map(|&v| v as f32).collect();
        let hh32: Vec<f32> = hh64.iter().map(|&v| v as f32).collect();
        let eff_hw32: Vec<f32> = hw32
            .iter()
            .map(|&v| (v as f64 + 0.5 * cfg_compact.padding) as f32)
            .collect();
        let eff_hh32: Vec<f32> = hh32
            .iter()
            .map(|&v| (v as f64 + 0.5 * cfg_compact.padding) as f32)
            .collect();

        let mut l32_run = Layout2d::from_xy(xs32, ys32).unwrap();
        let mut l32_step = l32_run.clone();
        let mut s32_run =
            GTreeOverlapRemoval::<f32>::new(hw32.clone(), hh32.clone(), cfg_compact).unwrap();
        let mut s32_step = GTreeOverlapRemoval::<f32>::new(hw32, hh32, cfg_compact).unwrap();

        assert_eq!(s32_run.run(&mut l32_run).unwrap(), SolverStatus::Converged);
        assert_eq!(s32_run.last_overlap_count(), 0);
        assert_no_overlaps(&l32_run, &eff_hw32, &eff_hh32);

        s32_step.run_steps(&mut l32_step, 2).unwrap();
        while !s32_step.status().is_finished() {
            s32_step.step(&mut l32_step).unwrap();
        }
        for i in 0..n {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }

    #[test]
    fn test_gtree_coincident_collinear_and_zero_size_edge_cases() {
        // 1. All coincident points at origin.
        let n = 16usize;
        let hw = vec![1.0; n];
        let hh = vec![0.5; n];
        let mut coincident = Layout2d::zeros(n);
        let mut s_coin =
            GTreeOverlapRemoval::new(hw.clone(), hh.clone(), GTreeConfig::default()).unwrap();
        assert_eq!(
            s_coin.run(&mut coincident).unwrap(),
            SolverStatus::Converged
        );
        assert_eq!(s_coin.last_overlap_count(), 0);
        assert_no_overlaps(&coincident, &hw, &hh);

        // 2. Strictly collinear layout along y = 0.
        let mut collinear =
            Layout2d::from_xy((0..n).map(|i| (i as f64) * 0.25).collect(), vec![0.0; n]).unwrap();
        let mut s_coll =
            GTreeOverlapRemoval::new(hw.clone(), hh.clone(), GTreeConfig::default()).unwrap();
        assert_eq!(s_coll.run(&mut collinear).unwrap(), SolverStatus::Converged);
        assert_eq!(s_coll.last_overlap_count(), 0);
        assert_no_overlaps(&collinear, &hw, &hh);

        // 3. Zero-size nodes (no interior overlap possible).
        let zero_r = vec![0.0; n];
        let mut zero_layout = Layout2d::zeros(n);
        let mut s_zero =
            GTreeOverlapRemoval::from_radii(zero_r.clone(), GTreeConfig::default()).unwrap();
        assert_eq!(
            s_zero.run(&mut zero_layout).unwrap(),
            SolverStatus::Converged
        );
        assert_eq!(s_zero.last_overlap_count(), 0);
        assert_no_overlaps(&zero_layout, &zero_r, &zero_r);

        // 4. Two widely separated overlapping clusters (>12 nodes each, exercises bridging).
        let m = 32usize;
        let mut c_xs = Vec::with_capacity(m);
        let mut c_ys = Vec::with_capacity(m);
        for i in 0..m {
            let base_x = if i < 16 { -1000.0 } else { 1000.0 };
            c_xs.push(base_x + ((i % 4) as f64) * 0.2);
            c_ys.push(((i / 4 % 4) as f64) * 0.2);
        }
        let c_hw = vec![1.0; m];
        let c_hh = vec![1.0; m];
        let mut c_layout = Layout2d::from_xy(c_xs, c_ys).unwrap();
        let mut s_clust =
            GTreeOverlapRemoval::new(c_hw.clone(), c_hh.clone(), GTreeConfig::default()).unwrap();
        assert_eq!(s_clust.run(&mut c_layout).unwrap(), SolverStatus::Converged);
        assert_eq!(s_clust.last_overlap_count(), 0);
        assert_no_overlaps(&c_layout, &c_hw, &c_hh);
    }
}
