//! Cluster-faithful layout refinement (`Cluster-kmeans` and `Cluster-HAC`, Cai et al., 2024).
//!
//! Iteratively improves geometric cluster faithfulness on an existing 2D graph drawing given
//! assigned or ground-truth cluster labels:
//! 1. Computes a 2D geometric partition $\mathcal{C}'$ of cardinality $K = |\mathcal{C}|$ using
//!    either $k$-means++ (`ClusterRefineMethod::Kmeans`) or $O(N^2)$ nearest-neighbor-chain
//!    average-linkage agglomerative clustering (`ClusterRefineMethod::Hac`).
//! 2. Finds the exact maximum-weight one-to-one mapping $\pi: \mathcal{C} \to \mathcal{C}'$ over
//!    the $K \times K$ Jaccard similarity matrix $JS(c_i, c'_j) = |c_i \cap c'_j| / |c_i \cup c'_j|$
//!    via the Hungarian algorithm.
//! 3. Identifies misplaced vertices $V_{\text{move}} = \bigcup_i (c_i \setminus \pi(c_i))$ and
//!    shifts them into their target geometric clusters:
//!    - `Kmeans`: crosses the exact Voronoi cell exit $P_{mj}$ along segment $[\mu_m, \mu_j]$ by
//!      fractional clearance `margin` toward target centroid $\mu_j$.
//!    - `Hac`: steps past the nearest vertex $w \in \pi(c_i)$ by mean edge length $\ell$ (or mean
//!      1-NN distance when no graph edges are supplied).

use egraph_clustering::{KMeansConfig, kmeans_plus_plus};
use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use egraph_spatial::StaticKdTree;

const GEOM_EPS: f64 = 1e-12;

/// Geometric clustering backend used by [`ClusterRefineSolver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ClusterRefineMethod {
    /// Cai et al. (2024) Algorithm 1: $k$-means++ geometric partitioning + Voronoi exit shift.
    #[default]
    Kmeans,
    /// Cai et al. (2024) Algorithm 2: average-linkage (UPGMA) HAC + nearest-target-vertex shift.
    Hac,
}

/// Configuration for [`ClusterRefineSolver`] and [`refine_cluster_layout`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterRefineConfig {
    /// Geometric clustering and vertex shift strategy (`default: ClusterRefineMethod::Kmeans`).
    pub method: ClusterRefineMethod,
    /// Maximum outer refinement rounds (`default: 10`).
    pub max_rounds: usize,
    /// Number of K-Means++ initializations per round (`default: 5`).
    pub kmeans_n_init: usize,
    /// Maximum Lloyd iterations per K-Means++ initialization (`default: 100`).
    pub kmeans_max_iters: usize,
    /// Interpolation factor toward the computed target coordinate (`default: 1.0`).
    pub shift_step: f64,
    /// Fractional clearance past the Voronoi boundary toward the target centroid (`default: 0.1`).
    pub margin: f64,
    /// Deterministic explicit RNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for ClusterRefineConfig {
    fn default() -> Self {
        Self {
            method: ClusterRefineMethod::Kmeans,
            max_rounds: 10,
            kmeans_n_init: 5,
            kmeans_max_iters: 100,
            shift_step: 1.0,
            margin: 0.1,
            seed: 42,
        }
    }
}

impl ClusterRefineConfig {
    /// Validates all cluster refinement hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.kmeans_n_init == 0 {
            return Err(CoreError::invalid_input("kmeans_n_init must be >= 1"));
        }
        if self.kmeans_max_iters == 0 {
            return Err(CoreError::invalid_input("kmeans_max_iters must be >= 1"));
        }
        if !self.shift_step.is_finite() {
            return Err(CoreError::non_finite("ClusterRefineConfig::shift_step"));
        }
        if self.shift_step < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "shift_step must be >= 0.0, got {}",
                self.shift_step
            )));
        }
        if !self.margin.is_finite() {
            return Err(CoreError::non_finite("ClusterRefineConfig::margin"));
        }
        if self.margin < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "margin must be >= 0.0, got {}",
                self.margin
            )));
        }
        Ok(())
    }
}

/// Cooperative cluster-faithful layout refiner implementing [`IterativeSolver`].
///
/// Each [`IterativeSolver::step`] executes one outer round of Cai et al. (2024)
/// `Cluster-kmeans` or `Cluster-HAC`, stopping early with [`SolverStatus::Converged`]
/// when no misplaced vertices remain ($V_{\text{move}} = \emptyset$).
#[derive(Debug, Clone)]
pub struct ClusterRefineSolver<S: Scalar = f64> {
    config: ClusterRefineConfig,
    num_nodes: usize,
    num_clusters: usize,
    gt_labels: Vec<u32>,
    edges: Vec<(u32, u32)>,
    last_moved_count: usize,
    iter: usize,
    status: SolverStatus,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Scalar> ClusterRefineSolver<S> {
    /// Constructs a [`ClusterRefineSolver`] from per-node cluster labels without graph edges.
    ///
    /// When `config.method == ClusterRefineMethod::Hac` and no graph edges are provided, the
    /// step scale $\ell$ falls back to the mean 1-nearest-neighbor Euclidean distance in the
    /// current layout.
    pub fn new(cluster_labels: Vec<u32>, config: ClusterRefineConfig) -> CoreResult<Self> {
        Self::build(cluster_labels, Vec::new(), config)
    }

    /// Constructs a [`ClusterRefineSolver`] with graph topology for computing mean edge length
    /// $\ell$ during [`ClusterRefineMethod::Hac`] shifts.
    pub fn from_graph<W: Scalar>(
        graph: &CsrGraph<W>,
        cluster_labels: Vec<u32>,
        config: ClusterRefineConfig,
    ) -> CoreResult<Self> {
        let n = graph.num_nodes();
        if cluster_labels.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                cluster_labels.len(),
                "ClusterRefineSolver::from_graph::cluster_labels",
            ));
        }
        let edges = extract_unique_undirected_edges(graph);
        Self::build(cluster_labels, edges, config)
    }

    fn build(
        cluster_labels: Vec<u32>,
        edges: Vec<(u32, u32)>,
        config: ClusterRefineConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = cluster_labels.len();
        let (gt_labels, num_clusters) = canonicalize_labels(&cluster_labels);
        let status = if config.max_rounds == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        Ok(Self {
            config,
            num_nodes: n,
            num_clusters,
            gt_labels,
            edges,
            last_moved_count: 0,
            iter: 0,
            status,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the number of misplaced vertices shifted during the most recent round.
    #[inline]
    pub fn last_moved_count(&self) -> usize {
        self.last_moved_count
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of distinct canonical ground-truth clusters $K$.
    #[inline]
    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &ClusterRefineConfig {
        &self.config
    }
}

impl<S: Scalar> IterativeSolver<S> for ClusterRefineSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "ClusterRefineSolver::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        let k = self.num_clusters;
        if n <= 1 || k <= 1 {
            self.last_moved_count = 0;
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let xs_in = layout.x();
        let ys_in = layout.y();
        let mut pts_x = Vec::with_capacity(n);
        let mut pts_y = Vec::with_capacity(n);
        for i in 0..n {
            pts_x.push(xs_in[i].to_f64());
            pts_y.push(ys_in[i].to_f64());
        }

        let geom_labels = match self.config.method {
            ClusterRefineMethod::Kmeans => {
                let mut flat = Vec::with_capacity(n * 2);
                for i in 0..n {
                    flat.push(pts_x[i]);
                    flat.push(pts_y[i]);
                }
                let km_cfg = KMeansConfig {
                    k,
                    max_iters: self.config.kmeans_max_iters,
                    tol: 1e-4,
                    n_init: self.config.kmeans_n_init,
                    seed: self.config.seed.wrapping_add(self.iter as u64),
                };
                let res = kmeans_plus_plus(&flat, n, 2, &km_cfg)?;
                res.labels
            }
            ClusterRefineMethod::Hac => upgma_hac_2d(&pts_x, &pts_y, k)?,
        };

        // Compute optimal 1-to-1 matching pi: gt_cluster -> geom_cluster maximizing Jaccard sum.
        let target_geom = match_clusters_hungarian(&self.gt_labels, &geom_labels, k);

        // Identify misplaced vertices V_move = { u : geom_labels[u] != target_geom[gt_labels[u]] }.
        let mut misplaced = Vec::new();
        for u in 0..n {
            let gt = self.gt_labels[u] as usize;
            let cur_g = geom_labels[u] as usize;
            let tgt_g = target_geom[gt];
            if cur_g != tgt_g {
                misplaced.push((u, cur_g, tgt_g));
            }
        }

        self.last_moved_count = misplaced.len();
        if misplaced.is_empty() {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let mut next_x = pts_x.clone();
        let mut next_y = pts_y.clone();

        match self.config.method {
            ClusterRefineMethod::Kmeans => {
                let centroids = compute_cluster_centroids(&pts_x, &pts_y, &geom_labels, k);
                let step_alpha = self.config.shift_step;
                let margin = self.config.margin;

                for &(u, cur_g, tgt_g) in &misplaced {
                    let mu_j = centroids[tgt_g];
                    let p_mj = voronoi_cell_exit(&centroids, cur_g, tgt_g);
                    let target_x = p_mj.0 + margin * (mu_j.0 - p_mj.0);
                    let target_y = p_mj.1 + margin * (mu_j.1 - p_mj.1);
                    next_x[u] = pts_x[u] + step_alpha * (target_x - pts_x[u]);
                    next_y[u] = pts_y[u] + step_alpha * (target_y - pts_y[u]);
                }
            }
            ClusterRefineMethod::Hac => {
                let ell = compute_hac_step_scale(&pts_x, &pts_y, &self.edges)?;
                let mut members_by_geom: Vec<Vec<usize>> = vec![Vec::new(); k];
                for (u, &g) in geom_labels.iter().enumerate() {
                    members_by_geom[g as usize].push(u);
                }
                let centroids = compute_cluster_centroids(&pts_x, &pts_y, &geom_labels, k);
                let step_alpha = self.config.shift_step;

                for &(u, _cur_g, tgt_g) in &misplaced {
                    let tgt_members = &members_by_geom[tgt_g];
                    if tgt_members.is_empty() {
                        continue;
                    }
                    let ux = pts_x[u];
                    let uy = pts_y[u];
                    let mut best_w = tgt_members[0];
                    let mut best_d2 = f64::INFINITY;
                    for &w in tgt_members {
                        let dx = pts_x[w] - ux;
                        let dy = pts_y[w] - uy;
                        let d2 = dx * dx + dy * dy;
                        if d2 < best_d2 || (d2 == best_d2 && w < best_w) {
                            best_d2 = d2;
                            best_w = w;
                        }
                    }

                    let wx = pts_x[best_w];
                    let wy = pts_y[best_w];
                    let dx = wx - ux;
                    let dy = wy - uy;
                    let dist = dx.hypot(dy);
                    let (dir_x, dir_y) = if dist > GEOM_EPS {
                        (dx / dist, dy / dist)
                    } else {
                        let cx = centroids[tgt_g].0 - wx;
                        let cy = centroids[tgt_g].1 - wy;
                        let cd = cx.hypot(cy);
                        if cd > GEOM_EPS {
                            (cx / cd, cy / cd)
                        } else {
                            deterministic_unit_vector(self.config.seed, self.iter, u, best_w)
                        }
                    };
                    let target_x = wx + ell * dir_x;
                    let target_y = wy + ell * dir_y;
                    next_x[u] = ux + step_alpha * (target_x - ux);
                    next_y[u] = uy + step_alpha * (target_y - uy);
                }
            }
        }

        let (xs_out, ys_out) = layout.xy_mut();
        for i in 0..n {
            xs_out[i] = S::from_f64(next_x[i]);
            ys_out[i] = S::from_f64(next_y[i]);
        }

        self.iter += 1;
        self.status = if self.iter >= self.config.max_rounds {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        Ok(self.status)
    }

    fn status(&self) -> SolverStatus {
        self.status
    }

    fn iteration(&self) -> usize {
        self.iter
    }

    fn total_iterations(&self) -> usize {
        self.config.max_rounds
    }
}

/// Runs [`ClusterRefineSolver`] to completion on `layout` and returns the total number of
/// misplaced-vertex shifts executed across all rounds.
pub fn refine_cluster_layout<S: Scalar>(
    layout: &mut Layout2d<S>,
    cluster_labels: &[u32],
    config: &ClusterRefineConfig,
) -> CoreResult<usize> {
    let mut solver = ClusterRefineSolver::new(cluster_labels.to_vec(), *config)?;
    let mut total_moved = 0usize;
    while !solver.status().is_finished() {
        let st = solver.step(layout)?;
        total_moved += solver.last_moved_count();
        if st.is_finished() {
            break;
        }
    }
    Ok(total_moved)
}

fn extract_unique_undirected_edges<W: Scalar>(graph: &CsrGraph<W>) -> Vec<(u32, u32)> {
    let n = graph.num_nodes();
    let mut edges = Vec::new();
    if !graph.is_directed() {
        for u in 0..n {
            let u32_u = u as u32;
            for &v_u32 in graph.neighbors(u) {
                if u32_u < v_u32 {
                    edges.push((u32_u, v_u32));
                }
            }
        }
    } else {
        for u in 0..n {
            let u32_u = u as u32;
            for &v_u32 in graph.neighbors(u) {
                if u32_u != v_u32 {
                    edges.push(if u32_u < v_u32 {
                        (u32_u, v_u32)
                    } else {
                        (v_u32, u32_u)
                    });
                }
            }
        }
        edges.sort_unstable();
        edges.dedup();
    }
    edges
}

/// Maps arbitrary `u32` cluster labels into contiguous `0..k` IDs in first-appearance order.
fn canonicalize_labels(labels: &[u32]) -> (Vec<u32>, usize) {
    let n = labels.len();
    if n == 0 {
        return (Vec::new(), 0);
    }
    let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut out = vec![0u32; n];
    let fast_limit = n.max(64).saturating_mul(4).min(1_000_000);
    if max_label < fast_limit {
        let mut remap = vec![u32::MAX; max_label + 1];
        let mut next_id = 0u32;
        for (i, &lbl) in labels.iter().enumerate() {
            let idx = lbl as usize;
            if remap[idx] == u32::MAX {
                remap[idx] = next_id;
                next_id += 1;
            }
            out[i] = remap[idx];
        }
        return (out, next_id as usize);
    }

    let mut pairs: Vec<(u32, u32)> = labels
        .iter()
        .enumerate()
        .map(|(i, &lbl)| (lbl, i as u32))
        .collect();
    pairs.sort_unstable();

    let mut groups: Vec<(u32, usize, usize)> = Vec::new();
    let mut start = 0usize;
    while start < n {
        let key = pairs[start].0;
        let mut end = start + 1;
        while end < n && pairs[end].0 == key {
            end += 1;
        }
        groups.push((pairs[start].1, start, end));
        start = end;
    }
    groups.sort_unstable_by_key(|&(first_node, _, _)| first_node);

    let k = groups.len();
    for (cid, &(_, g_start, g_end)) in groups.iter().enumerate() {
        let cid_u32 = cid as u32;
        for &(_, node_idx) in &pairs[g_start..g_end] {
            out[node_idx as usize] = cid_u32;
        }
    }
    (out, k)
}

/// Exact $O(N^2)$ nearest-neighbor-chain UPGMA (average-linkage) clustering cut at `k` clusters.
fn upgma_hac_2d(xs: &[f64], ys: &[f64], k: usize) -> CoreResult<Vec<u32>> {
    let n = xs.len();
    if k == 0 || k > n {
        return Err(CoreError::invalid_input(format!(
            "upgma_hac_2d requires 1 <= k <= n, got k={k}, n={n}"
        )));
    }
    if k == n {
        return Ok((0..n as u32).collect());
    }
    if k == 1 {
        return Ok(vec![0u32; n]);
    }

    let mat_len = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("HAC distance matrix size overflow"))?;
    let mut dist = vec![0.0_f64; mat_len];
    for i in 0..n {
        let xi = xs[i];
        let yi = ys[i];
        for j in (i + 1)..n {
            let d = (xi - xs[j]).hypot(ys[j] - yi);
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }

    let mut active = vec![true; n];
    let mut sizes = vec![1usize; n];
    let mut min_node: Vec<usize> = (0..n).collect();
    let member_head = (0..n).collect::<Vec<usize>>();
    let mut member_tail = (0..n).collect::<Vec<usize>>();
    let mut member_next = vec![usize::MAX; n];
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    let mut remaining = n;

    while remaining > k {
        if chain.is_empty() {
            let first = active
                .iter()
                .position(|&a| a)
                .ok_or_else(|| CoreError::invalid_input("HAC internal state exhausted"))?;
            chain.push(first);
        }

        let u = *chain.last().unwrap();
        let prev = if chain.len() >= 2 {
            Some(chain[chain.len() - 2])
        } else {
            None
        };

        // Find nearest active cluster to `u` with deterministic tie-breaking by `min_node`.
        let mut best_v = usize::MAX;
        let mut best_d = f64::INFINITY;
        let mut best_min = usize::MAX;
        let row_off = u * n;
        for v in 0..n {
            if !active[v] || v == u {
                continue;
            }
            let d = dist[row_off + v];
            let mv = min_node[v];
            if d < best_d || (d == best_d && mv < best_min) {
                best_d = d;
                best_v = v;
                best_min = mv;
            }
        }

        if best_v == usize::MAX {
            break;
        }

        // Check reciprocal nearest neighbor: either `best_v == prev`, or `prev` is active and
        // tied in distance with `best_v` (exact equilateral tie check).
        let is_reciprocal =
            prev == Some(best_v) || prev.is_some_and(|p| active[p] && dist[row_off + p] == best_d);

        if is_reciprocal {
            let p = prev.unwrap();
            chain.pop();
            chain.pop();

            // Merge `u` and `p` into the slot with smaller index `a` (deactivating `b`).
            let (a, b) = if u < p { (u, p) } else { (p, u) };
            let sa = sizes[a] as f64;
            let sb = sizes[b] as f64;
            let sab = sa + sb;

            // Lance-Williams average-linkage update:
            // d(A U B, C) = (|A| * d(A, C) + |B| * d(B, C)) / (|A| + |B|).
            for c in 0..n {
                if !active[c] || c == a || c == b {
                    continue;
                }
                let d_new = (sa * dist[a * n + c] + sb * dist[b * n + c]) / sab;
                dist[a * n + c] = d_new;
                dist[c * n + a] = d_new;
            }

            sizes[a] += sizes[b];
            min_node[a] = min_node[a].min(min_node[b]);
            member_next[member_tail[a]] = member_head[b];
            member_tail[a] = member_tail[b];
            active[b] = false;
            remaining -= 1;
        } else {
            chain.push(best_v);
        }
    }

    let mut raw_labels = vec![0u32; n];
    let mut cid = 0u32;
    for c in 0..n {
        if !active[c] {
            continue;
        }
        let mut cur = member_head[c];
        while cur != usize::MAX {
            raw_labels[cur] = cid;
            cur = member_next[cur];
        }
        cid += 1;
    }

    let (canon, _) = canonicalize_labels(&raw_labels);
    Ok(canon)
}

/// Computes exact maximum-weight bipartite matching over the $K \times K$ Jaccard matrix
/// between ground-truth clusters (`gt_labels`) and geometric clusters (`geom_labels`).
fn match_clusters_hungarian(gt_labels: &[u32], geom_labels: &[u32], k: usize) -> Vec<usize> {
    let n = gt_labels.len();
    let mut inter = vec![0usize; k * k];
    let mut gt_sizes = vec![0usize; k];
    let mut geom_sizes = vec![0usize; k];

    for idx in 0..n {
        let i = gt_labels[idx] as usize;
        let j = geom_labels[idx] as usize;
        gt_sizes[i] += 1;
        geom_sizes[j] += 1;
        inter[i * k + j] += 1;
    }

    let mut cost = vec![1.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let ov = inter[i * k + j];
            let uni = gt_sizes[i] + geom_sizes[j] - ov;
            let js = if uni > 0 {
                (ov as f64) / (uni as f64)
            } else {
                0.0
            };
            cost[i * k + j] = 1.0 - js;
        }
    }

    // 1-indexed O(K^3) Hungarian algorithm minimizing sum_i cost[i, match[i]].
    let mut u = vec![0.0_f64; k + 1];
    let mut v = vec![0.0_f64; k + 1];
    let mut p = vec![0usize; k + 1];
    let mut way = vec![0usize; k + 1];
    let mut minv = vec![0.0_f64; k + 1];
    let mut used = vec![false; k + 1];

    for i in 1..=k {
        p[0] = i;
        let mut j0 = 0usize;
        minv.fill(f64::INFINITY);
        used.fill(false);

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=k {
                if !used[j] {
                    let cur = cost[(i0 - 1) * k + (j - 1)] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=k {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut row_to_col = vec![0usize; k];
    for j in 1..=k {
        if p[j] > 0 {
            row_to_col[p[j] - 1] = j - 1;
        }
    }
    row_to_col
}

fn compute_cluster_centroids(xs: &[f64], ys: &[f64], labels: &[u32], k: usize) -> Vec<(f64, f64)> {
    let mut sum_x = vec![0.0_f64; k];
    let mut sum_y = vec![0.0_f64; k];
    let mut counts = vec![0usize; k];
    for (i, &lbl) in labels.iter().enumerate() {
        let c = lbl as usize;
        sum_x[c] += xs[i];
        sum_y[c] += ys[i];
        counts[c] += 1;
    }
    let mut centroids = Vec::with_capacity(k);
    for c in 0..k {
        if counts[c] > 0 {
            let inv = 1.0 / (counts[c] as f64);
            centroids.push((sum_x[c] * inv, sum_y[c] * inv));
        } else {
            centroids.push((0.0, 0.0));
        }
    }
    centroids
}

/// Computes the exact point where segment $[\mu_m, \mu_j]$ exits the Voronoi cell of $\mu_m$,
/// falling back to the midpoint $\frac{1}{2}(\mu_m + \mu_j)$ if degenerate.
fn voronoi_cell_exit(centroids: &[(f64, f64)], m: usize, j: usize) -> (f64, f64) {
    let mu_m = centroids[m];
    let mu_j = centroids[j];
    let vx = mu_j.0 - mu_m.0;
    let vy = mu_j.1 - mu_m.1;
    let v_sq = vx * vx + vy * vy;
    if v_sq <= GEOM_EPS * GEOM_EPS {
        return (0.5 * (mu_m.0 + mu_j.0), 0.5 * (mu_m.1 + mu_j.1));
    }

    // Half-plane for site k != m: <(mu_m + t*v) - (mu_m + mu_k)/2, mu_k - mu_m> <= 0
    // <=> t * <v, d_k> <= 0.5 * ||d_k||^2, where d_k = mu_k - mu_m.
    let mut t_exit = 0.5_f64;
    for (c_idx, &mu_k) in centroids.iter().enumerate() {
        if c_idx == m {
            continue;
        }
        let dkx = mu_k.0 - mu_m.0;
        let dky = mu_k.1 - mu_m.1;
        let dk_sq = dkx * dkx + dky * dky;
        let dot = vx * dkx + vy * dky;
        if dot > GEOM_EPS && dk_sq > 0.0 {
            let t_k = (0.5 * dk_sq) / dot;
            if t_k >= 0.0 && t_k < t_exit {
                t_exit = t_k;
            }
        }
    }

    if !t_exit.is_finite() || t_exit <= 0.0 {
        t_exit = 0.5;
    }
    (mu_m.0 + t_exit * vx, mu_m.1 + t_exit * vy)
}

/// Computes $\ell$ for `Cluster-HAC`: mean graph edge length in the current layout when edges
/// are present, otherwise mean 1-NN Euclidean distance (falling back to layout scale).
fn compute_hac_step_scale(xs: &[f64], ys: &[f64], edges: &[(u32, u32)]) -> CoreResult<f64> {
    if !edges.is_empty() {
        let mut sum = 0.0_f64;
        for &(u, v) in edges {
            let ui = u as usize;
            let vi = v as usize;
            sum += (xs[ui] - xs[vi]).hypot(ys[ui] - ys[vi]);
        }
        let mean_edge = sum / (edges.len() as f64);
        if mean_edge > GEOM_EPS && mean_edge.is_finite() {
            return Ok(mean_edge);
        }
    }

    let n = xs.len();
    if n >= 2 {
        let mut pts = Vec::with_capacity(n);
        for i in 0..n {
            pts.push([xs[i], ys[i]]);
        }
        let tree = StaticKdTree::<2, f64>::build(&pts)?;
        let mut nn_sum = 0.0_f64;
        let mut nn_cnt = 0usize;
        for i in 0..n {
            if let Some((_, d)) = tree.nearest_excluding_self(i)? {
                nn_sum += d;
                nn_cnt += 1;
            }
        }
        if nn_cnt > 0 {
            let mean_nn = nn_sum / (nn_cnt as f64);
            if mean_nn > GEOM_EPS && mean_nn.is_finite() {
                return Ok(mean_nn);
            }
        }
    }

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for i in 0..n {
        min_x = min_x.min(xs[i]);
        max_x = max_x.max(xs[i]);
        min_y = min_y.min(ys[i]);
        max_y = max_y.max(ys[i]);
    }
    let diag = (max_x - min_x).hypot(max_y - min_y);
    Ok(if diag > GEOM_EPS { diag * 0.1 } else { 1.0 })
}

#[inline]
fn deterministic_unit_vector(seed: u64, iter: usize, u: usize, w: usize) -> (f64, f64) {
    let mut z = seed
        .wrapping_add((iter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((u as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add((w as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let frac = ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
    let angle = frac * std::f64::consts::TAU;
    (angle.cos(), angle.sin())
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_core::DeterministicRng;

    fn compute_cq(gt_labels: &[u32], geom_labels: &[u32], k: usize) -> f64 {
        let n = gt_labels.len();
        if n == 0 {
            return 1.0;
        }
        let matching = match_clusters_hungarian(gt_labels, geom_labels, k);
        let mut hits = 0usize;
        for i in 0..n {
            if geom_labels[i] as usize == matching[gt_labels[i] as usize] {
                hits += 1;
            }
        }
        (hits as f64) / (n as f64)
    }

    fn kmeans_cq(layout: &Layout2d<f64>, gt_labels: &[u32], k: usize) -> f64 {
        let n = layout.len();
        let mut flat = Vec::with_capacity(n * 2);
        for i in 0..n {
            flat.push(layout.x()[i]);
            flat.push(layout.y()[i]);
        }
        let km = kmeans_plus_plus(
            &flat,
            n,
            2,
            &KMeansConfig {
                k,
                n_init: 5,
                max_iters: 100,
                tol: 1e-4,
                seed: 7,
            },
        )
        .unwrap();
        let (canon_gt, _) = canonicalize_labels(gt_labels);
        compute_cq(&canon_gt, &km.labels, k)
    }

    fn hac_cq(layout: &Layout2d<f64>, gt_labels: &[u32], k: usize) -> f64 {
        let geom = upgma_hac_2d(layout.x(), layout.y(), k).unwrap();
        let (canon_gt, _) = canonicalize_labels(gt_labels);
        compute_cq(&canon_gt, &geom, k)
    }

    #[test]
    fn test_cluster_kmeans_and_hac_improve_planted_overlapping_clusters() {
        // 3 planted clusters of 10 nodes each (30 nodes), with several overlapping misplaced nodes.
        let mut rng = DeterministicRng::from_seed_u64(101);
        let centers = [(0.0_f64, 0.0_f64), (10.0, 0.0), (5.0, 8.66)];
        let mut xs = Vec::with_capacity(30);
        let mut ys = Vec::with_capacity(30);
        let mut labels = Vec::with_capacity(30);

        for (cid, &(cx, cy)) in centers.iter().enumerate() {
            for idx in 0..10 {
                // Intentionally misplace 3 nodes per cluster into neighboring cluster regions.
                let (bx, by) = if idx < 3 {
                    let wrong = (cid + 1) % 3;
                    centers[wrong]
                } else {
                    (cx, cy)
                };
                xs.push(bx + (rng.next_f64() - 0.5) * 0.8);
                ys.push(by + (rng.next_f64() - 0.5) * 0.8);
                labels.push((cid as u32) * 10 + 5); // Non-contiguous labels test canonicalization.
            }
        }

        let mut edges = Vec::new();
        for cid in 0..3 {
            let base = cid * 10;
            for i in 0..10 {
                for j in (i + 1)..10 {
                    edges.push(((base + i) as u32, (base + j) as u32));
                }
            }
        }
        let graph = CsrGraph::<f64>::from_edges(30, &edges, None, false, false).unwrap();

        let orig = Layout2d::from_xy(xs, ys).unwrap();
        let orig_cq_km = kmeans_cq(&orig, &labels, 3);
        let orig_cq_hac = hac_cq(&orig, &labels, 3);
        assert!(
            orig_cq_km < 0.85,
            "expected overlapping initial CQ < 0.85, got {orig_cq_km}"
        );
        assert!(
            orig_cq_hac < 0.85,
            "expected overlapping initial HAC CQ < 0.85, got {orig_cq_hac}"
        );

        // Cluster-kmeans improvement + step == run bit-exactness.
        let km_cfg = ClusterRefineConfig {
            method: ClusterRefineMethod::Kmeans,
            max_rounds: 10,
            ..Default::default()
        };
        let mut km_batch = orig.clone();
        let mut km_step = orig.clone();
        let mut s_batch = ClusterRefineSolver::<f64>::new(labels.clone(), km_cfg).unwrap();
        let mut s_step = ClusterRefineSolver::<f64>::new(labels.clone(), km_cfg).unwrap();
        let st_batch = s_batch.run(&mut km_batch).unwrap();
        while !s_step.status().is_finished() {
            s_step.step(&mut km_step).unwrap();
        }
        assert_eq!(st_batch, s_step.status());
        assert_eq!(s_batch.iteration(), s_step.iteration());
        assert_eq!(km_batch, km_step);

        let refined_cq_km = kmeans_cq(&km_batch, &labels, 3);
        assert!(
            refined_cq_km > orig_cq_km,
            "Cluster-kmeans should improve CQ: {orig_cq_km} -> {refined_cq_km}"
        );

        // Cluster-HAC improvement via from_graph + step == run bit-exactness.
        let hac_cfg = ClusterRefineConfig {
            method: ClusterRefineMethod::Hac,
            max_rounds: 10,
            ..Default::default()
        };
        let mut hac_batch = orig.clone();
        let mut hac_step = orig.clone();
        let mut sh_batch =
            ClusterRefineSolver::<f64>::from_graph(&graph, labels.clone(), hac_cfg).unwrap();
        let mut sh_step =
            ClusterRefineSolver::<f64>::from_graph(&graph, labels.clone(), hac_cfg).unwrap();
        let sth_batch = sh_batch.run(&mut hac_batch).unwrap();
        while !sh_step.status().is_finished() {
            sh_step.step(&mut hac_step).unwrap();
        }
        assert_eq!(sth_batch, sh_step.status());
        assert_eq!(hac_batch, hac_step);

        let refined_cq_hac = hac_cq(&hac_batch, &labels, 3);
        assert!(
            refined_cq_hac > orig_cq_hac,
            "Cluster-HAC should improve CQ: {orig_cq_hac} -> {refined_cq_hac}"
        );

        // Also test f32 bit-exact step == run and refine_cluster_layout convenience fn.
        let orig_f32 = Layout2d::from_xy(
            orig.x().iter().map(|&v| v as f32).collect(),
            orig.y().iter().map(|&v| v as f32).collect(),
        )
        .unwrap();
        let mut f32_batch = orig_f32.clone();
        let mut f32_step = orig_f32.clone();
        let mut s32_b = ClusterRefineSolver::<f32>::new(labels.clone(), km_cfg).unwrap();
        let mut s32_s = ClusterRefineSolver::<f32>::new(labels.clone(), km_cfg).unwrap();
        s32_b.run(&mut f32_batch).unwrap();
        while !s32_s.status().is_finished() {
            s32_s.step(&mut f32_step).unwrap();
        }
        assert_eq!(f32_batch, f32_step);

        let mut convenience_layout = orig.clone();
        let moved = refine_cluster_layout(&mut convenience_layout, &labels, &km_cfg).unwrap();
        assert!(moved > 0);
        assert_eq!(convenience_layout, km_batch);
    }

    #[test]
    fn test_upgma_hac_matches_exact_naive_and_handles_ties() {
        // Test exact UPGMA against naive O(N^3) implementation on points with ties.
        let xs = [0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0];
        let ys = [0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0];
        let labels = upgma_hac_2d(&xs, &ys, 2).unwrap();
        // Left cluster {0,1,2,3} vs right cluster {4,5,6}.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[4], labels[6]);
        assert_ne!(labels[0], labels[4]);

        // Collinear equidistant points (strict tie-breaking must not cycle in NN-chain).
        let eq_x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let eq_y = [0.0, 0.0, 0.0, 0.0, 0.0];
        let eq_labels = upgma_hac_2d(&eq_x, &eq_y, 2).unwrap();
        assert_eq!(eq_labels.len(), 5);
        let distinct: std::collections::BTreeSet<u32> = eq_labels.iter().copied().collect();
        assert_eq!(distinct.len(), 2);
    }

    #[test]
    fn test_voronoi_exit_uses_intermediate_cell_when_non_adjacent() {
        // 3 centroids along x-axis: c0=(0,0), c1=(4,0), c2=(10,0).
        // Segment [c0, c2] exits c0's Voronoi cell at bisector(c0, c1) = (2, 0), NOT midpoint (5, 0).
        let centroids = [(0.0, 0.0), (4.0, 0.0), (10.0, 0.0)];
        let exit = voronoi_cell_exit(&centroids, 0, 2);
        assert!((exit.0 - 2.0).abs() < 1e-12);
        assert!(exit.1.abs() < 1e-12);
    }

    #[test]
    fn test_cluster_refine_degenerate_and_validation() {
        // Empty layout
        let mut l0 = Layout2d::<f64>::zeros(0);
        let mut s0 =
            ClusterRefineSolver::<f64>::new(vec![], ClusterRefineConfig::default()).unwrap();
        assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);
        assert_eq!(s0.last_moved_count(), 0);

        // Single node
        let mut l1 = Layout2d::<f64>::from_xy(vec![1.0], vec![2.0]).unwrap();
        let mut s1 =
            ClusterRefineSolver::<f64>::new(vec![7], ClusterRefineConfig::default()).unwrap();
        assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
        assert_eq!(s1.last_moved_count(), 0);

        // Single cluster (K = 1)
        let mut lk1 = Layout2d::<f64>::from_xy(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]).unwrap();
        let mut sk1 =
            ClusterRefineSolver::<f64>::new(vec![3, 3, 3], ClusterRefineConfig::default()).unwrap();
        assert_eq!(sk1.run(&mut lk1).unwrap(), SolverStatus::Converged);
        assert_eq!(sk1.last_moved_count(), 0);

        // Already perfectly separated -> early Converged with 0 moves
        let mut l_sep =
            Layout2d::<f64>::from_xy(vec![0.0, 0.1, 20.0, 20.1], vec![0.0, 0.1, 0.0, 0.1]).unwrap();
        let mut s_sep =
            ClusterRefineSolver::<f64>::new(vec![0, 0, 1, 1], ClusterRefineConfig::default())
                .unwrap();
        assert_eq!(s_sep.run(&mut l_sep).unwrap(), SolverStatus::Converged);
        assert_eq!(s_sep.last_moved_count(), 0);
        assert_eq!(s_sep.iteration(), 0);

        // Validation errors
        assert!(matches!(
            ClusterRefineConfig {
                kmeans_n_init: 0,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            ClusterRefineConfig {
                shift_step: f64::NAN,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            ClusterRefineConfig {
                margin: -0.1,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::InvalidInput(_))
        ));
    }
}
