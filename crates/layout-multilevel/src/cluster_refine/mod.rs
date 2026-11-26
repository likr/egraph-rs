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

mod config;
mod geometry;
mod hac;
#[cfg(test)]
mod tests;

pub use config::{ClusterRefineConfig, ClusterRefineMethod};

use egraph_clustering::{KMeansConfig, kmeans_plus_plus};
use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use geometry::{
    GEOM_EPS, canonicalize_labels, compute_cluster_centroids, compute_hac_step_scale,
    deterministic_unit_vector, extract_unique_undirected_edges, voronoi_cell_exit,
};
use hac::{match_clusters_hungarian, upgma_hac_2d};

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
