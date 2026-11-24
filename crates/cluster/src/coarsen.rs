//! Graph coarsening and multilevel layout uncoarsening/projection primitives.

use crate::louvain::{LouvainConfig, louvain};
use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, Scalar};

/// Configuration for multi-level Louvain coarsening up to a coarse-node budget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoarseLouvainConfig {
    /// Target upper bound on the number of coarsest super-nodes (`default: 10_000`).
    ///
    /// Coarsening stops once `num_clusters <= max_coarse_nodes`, or earlier at the
    /// connected-component floor if the graph has more than `max_coarse_nodes` disconnected
    /// components.
    pub max_coarse_nodes: usize,
    /// Initial Louvain modularity resolution $\gamma \ge 0.0$ (`default: 1.0`).
    pub resolution: f64,
    /// Multiplicative resolution decay factor in `(0.0, 1.0)` applied when a round stalls
    /// above `max_coarse_nodes` (`default: 0.5`).
    pub resolution_decay: f64,
    /// Maximum number of inner Louvain passes per outer coarsening round (`default: 100`).
    pub max_passes: usize,
    /// Minimum modularity gain threshold for inner Louvain passes (`default: 1e-7`).
    pub min_modularity_gain: f64,
    /// Maximum number of outer Louvain contraction rounds (`default: 16`).
    pub max_outer_rounds: usize,
    /// Explicit deterministic PRNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for CoarseLouvainConfig {
    fn default() -> Self {
        Self {
            max_coarse_nodes: 10_000,
            resolution: 1.0,
            resolution_decay: 0.5,
            max_passes: 100,
            min_modularity_gain: 1e-7,
            max_outer_rounds: 16,
            seed: 42,
        }
    }
}

impl CoarseLouvainConfig {
    /// Validates all multi-level Louvain coarsening hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.max_coarse_nodes == 0 {
            return Err(CoreError::invalid_input(
                "CoarseLouvainConfig::max_coarse_nodes must be >= 1",
            ));
        }
        if !self.resolution.is_finite() {
            return Err(CoreError::non_finite("CoarseLouvainConfig::resolution"));
        }
        if self.resolution < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "CoarseLouvainConfig::resolution must be >= 0.0, got {}",
                self.resolution
            )));
        }
        if !self.resolution_decay.is_finite() {
            return Err(CoreError::non_finite(
                "CoarseLouvainConfig::resolution_decay",
            ));
        }
        if self.resolution_decay <= 0.0 || self.resolution_decay >= 1.0 {
            return Err(CoreError::invalid_input(format!(
                "CoarseLouvainConfig::resolution_decay must be in (0.0, 1.0), got {}",
                self.resolution_decay
            )));
        }
        if self.max_passes == 0 {
            return Err(CoreError::invalid_input(
                "CoarseLouvainConfig::max_passes must be >= 1",
            ));
        }
        if !self.min_modularity_gain.is_finite() {
            return Err(CoreError::non_finite(
                "CoarseLouvainConfig::min_modularity_gain",
            ));
        }
        if self.min_modularity_gain < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "CoarseLouvainConfig::min_modularity_gain must be >= 0.0, got {}",
                self.min_modularity_gain
            )));
        }
        if self.max_outer_rounds == 0 {
            return Err(CoreError::invalid_input(
                "CoarseLouvainConfig::max_outer_rounds must be >= 1",
            ));
        }
        Ok(())
    }
}

/// Single contraction level recorded by [`louvain_coarsen`].
#[derive(Debug, Clone, PartialEq)]
pub struct CoarsenedHierarchyLevel {
    /// Contracted super-graph produced at this round.
    pub coarse: CoarsenedGraph,
    /// Louvain resolution $\gamma$ used at this round.
    pub resolution: f64,
    /// Modularity $Q_\gamma$ achieved at this round.
    pub modularity: f64,
}

/// Multi-level Louvain coarsening hierarchy from fine (`levels[0]`) to coarsest (`coarsest`).
#[derive(Debug, Clone, PartialEq)]
pub struct CoarsenedHierarchy {
    /// Ordered outer contraction levels (`0..num_levels`).
    pub levels: Vec<CoarsenedHierarchyLevel>,
    /// Coarsest super-graph with direct original-to-coarsest mapping and cumulative sizes.
    pub coarsest: CoarsenedGraph,
}

/// Result of contracting vertex clusters via [`coarsen_graph`].
#[derive(Debug, Clone, PartialEq)]
pub struct CoarsenedGraph {
    /// Weighted undirected super-graph over `0..num_clusters`.
    pub graph: CsrGraph<f64>,
    /// Compacted cluster ID in `0..num_clusters` for each original vertex.
    pub node_to_coarse: Vec<u32>,
    /// Number of original vertices belonging to each coarse cluster.
    pub coarse_sizes: Vec<u32>,
    /// Number of distinct compacted clusters.
    pub num_clusters: usize,
}

/// Compacts arbitrary `u32` labels into canonical `0..num_clusters` in first-appearance order
/// without hash maps.
fn compact_cluster_labels(labels: &[u32]) -> (Vec<u32>, Vec<u32>, usize) {
    let n = labels.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), 0);
    }

    let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut node_to_coarse = vec![0u32; n];

    // Use a flat lookup array when labels are bounded; otherwise sort (label, node) pairs.
    let fast_limit = n.max(64).saturating_mul(4).min(1_000_000);
    if max_label < fast_limit {
        let mut remap = vec![u32::MAX; max_label + 1];
        let mut next_id = 0u32;
        for (i, &l) in labels.iter().enumerate() {
            let idx = l as usize;
            if remap[idx] == u32::MAX {
                remap[idx] = next_id;
                next_id += 1;
            }
            node_to_coarse[i] = remap[idx];
        }
        let num_clusters = next_id as usize;
        let mut coarse_sizes = vec![0u32; num_clusters];
        for &c in &node_to_coarse {
            coarse_sizes[c as usize] += 1;
        }
        return (node_to_coarse, coarse_sizes, num_clusters);
    }

    let mut pairs: Vec<(u32, u32)> = labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i as u32))
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

    // Order clusters by their smallest (first-appearing) original vertex index.
    groups.sort_unstable_by_key(|&(first_node, _, _)| first_node);

    let num_clusters = groups.len();
    let mut coarse_sizes = Vec::with_capacity(num_clusters);
    for (cid, &(_, g_start, g_end)) in groups.iter().enumerate() {
        let cid_u32 = cid as u32;
        coarse_sizes.push((g_end - g_start) as u32);
        for &(_, orig_node) in &pairs[g_start..g_end] {
            node_to_coarse[orig_node as usize] = cid_u32;
        }
    }
    (node_to_coarse, coarse_sizes, num_clusters)
}

/// Coarsens an undirected [`CsrGraph`] by contracting vertices sharing a cluster label.
///
/// Arbitrary `labels` values are canonically compacted to `0..num_clusters` in ascending
/// order of first appearance. Parallel inter-cluster edges are summed into a single
/// weighted undirected edge in `CoarsenedGraph::graph`, and intra-cluster edges are omitted.
pub fn coarsen_graph<W: Scalar>(graph: &CsrGraph<W>, labels: &[u32]) -> CoreResult<CoarsenedGraph> {
    if graph.is_directed() {
        return Err(CoreError::invalid_input(
            "coarsen_graph requires an undirected graph",
        ));
    }
    let n = graph.num_nodes();
    if labels.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            labels.len(),
            "coarsen_graph::labels",
        ));
    }
    if n == 0 {
        let empty = CsrGraph::from_csr(&[0], &[], None, false, false)?;
        return Ok(CoarsenedGraph {
            graph: empty,
            node_to_coarse: Vec::new(),
            coarse_sizes: Vec::new(),
            num_clusters: 0,
        });
    }

    let weights_opt = graph.weights();
    if let Some(ws) = weights_opt {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite("coarsen_graph::graph::weights"));
            }
            if wf < 0.0 {
                return Err(CoreError::invalid_input(
                    "coarsen_graph edge weights must be non-negative",
                ));
            }
        }
    }

    let (node_to_coarse, coarse_sizes, num_clusters) = compact_cluster_labels(labels);

    let mut inter_edges: Vec<(u32, u32, f64)> = Vec::new();
    for u in 0..n {
        let cu = node_to_coarse[u];
        let start = graph.indptr()[u] as usize;
        let end = graph.indptr()[u + 1] as usize;
        for pos in start..end {
            let v = graph.indices()[pos] as usize;
            // Visit each undirected edge once in canonical endpoint order.
            if v <= u {
                continue;
            }
            let cv = node_to_coarse[v];
            if cu == cv {
                continue;
            }
            let w = match weights_opt {
                Some(ws) => ws[pos].to_f64(),
                None => 1.0,
            };
            let (a, b) = if cu < cv { (cu, cv) } else { (cv, cu) };
            inter_edges.push((a, b, w));
        }
    }

    inter_edges.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));

    let mut coarse_pairs: Vec<(u32, u32)> = Vec::new();
    let mut coarse_weights: Vec<f64> = Vec::new();
    for &(a, b, w) in &inter_edges {
        if let Some(&last_pair) = coarse_pairs.last()
            && last_pair == (a, b)
        {
            if let Some(last_w) = coarse_weights.last_mut() {
                *last_w += w;
            }
        } else {
            coarse_pairs.push((a, b));
            coarse_weights.push(w);
        }
    }

    let coarse_graph = CsrGraph::from_edges(
        num_clusters,
        &coarse_pairs,
        Some(&coarse_weights),
        false,
        false,
    )?;

    Ok(CoarsenedGraph {
        graph: coarse_graph,
        node_to_coarse,
        coarse_sizes,
        num_clusters,
    })
}

/// Projects a 2D layout of coarse super-nodes onto the fine vertices specified by
/// `node_to_coarse`, adding deterministic disk offsets within `spread_radius` for
/// multi-member clusters to avoid exact coordinate overlap.
pub fn project_clustered_layout<S: Scalar>(
    coarse_layout: &Layout2d<S>,
    node_to_coarse: &[u32],
    spread_radius: S,
    seed: u64,
) -> CoreResult<Layout2d<S>> {
    let r = spread_radius.to_f64();
    if !r.is_finite() {
        return Err(CoreError::non_finite(
            "project_clustered_layout::spread_radius",
        ));
    }
    if r < 0.0 {
        return Err(CoreError::invalid_input(
            "spread_radius must be non-negative",
        ));
    }
    coarse_layout.validate_finite()?;

    let n = node_to_coarse.len();
    if n == 0 {
        return Ok(Layout2d::zeros(0));
    }

    let num_clusters = coarse_layout.len();
    let mut cluster_sizes = vec![0usize; num_clusters];
    for &c in node_to_coarse {
        let cid = c as usize;
        if cid >= num_clusters {
            return Err(CoreError::index_out_of_bounds(
                cid,
                num_clusters,
                "project_clustered_layout::node_to_coarse",
            ));
        }
        cluster_sizes[cid] += 1;
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let cx = coarse_layout.x();
    let cy = coarse_layout.y();
    let mut fine_x = Vec::with_capacity(n);
    let mut fine_y = Vec::with_capacity(n);

    for &c in node_to_coarse {
        let cid = c as usize;
        let base_x = cx[cid].to_f64();
        let base_y = cy[cid].to_f64();
        if r > 0.0 && cluster_sizes[cid] > 1 {
            let angle = std::f64::consts::TAU * rng.next_f64();
            let rad = r * rng.next_f64().sqrt();
            let (sin_a, cos_a) = angle.sin_cos();
            fine_x.push(S::from_f64(base_x + rad * cos_a));
            fine_y.push(S::from_f64(base_y + rad * sin_a));
        } else {
            fine_x.push(S::from_f64(base_x));
            fine_y.push(S::from_f64(base_y));
        }
    }

    Layout2d::from_xy(fine_x, fine_y)
}

/// Runs multi-level Louvain coarsening with resolution decay until
/// `coarsest.num_clusters <= config.max_coarse_nodes` or the connected-component floor is reached.
pub fn louvain_coarsen<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &CoarseLouvainConfig,
) -> CoreResult<CoarsenedHierarchy> {
    config.validate()?;
    if graph.is_directed() {
        return Err(CoreError::invalid_input(
            "louvain_coarsen requires an undirected graph",
        ));
    }

    let n = graph.num_nodes();
    if n == 0 {
        let coarsest = coarsen_graph(graph, &[])?;
        return Ok(CoarsenedHierarchy {
            levels: Vec::new(),
            coarsest,
        });
    }

    if n == 1 || graph.nnz() == 0 {
        let labels: Vec<u32> = (0..n as u32).collect();
        let coarse = coarsen_graph(graph, &labels)?;
        let level = CoarsenedHierarchyLevel {
            coarse: coarse.clone(),
            resolution: config.resolution,
            modularity: 0.0,
        };
        return Ok(CoarsenedHierarchy {
            levels: vec![level],
            coarsest: coarse,
        });
    }

    let mut cur_res = config.resolution;
    let mut cur_min_gain = config.min_modularity_gain;
    let l0_cfg = LouvainConfig {
        max_passes: config.max_passes,
        resolution: cur_res,
        min_modularity_gain: cur_min_gain,
        seed: config.seed,
    };
    let res0 = louvain(graph, &l0_cfg)?;
    let coarse0 = coarsen_graph(graph, &res0.labels)?;
    let mut fine_to_coarsest = coarse0.node_to_coarse.clone();
    let mut cumulative_sizes = coarse0.coarse_sizes.clone();
    let mut num_clusters = coarse0.num_clusters;
    let mut current_graph = coarse0.graph.clone();
    let mut levels = vec![CoarsenedHierarchyLevel {
        coarse: coarse0,
        resolution: cur_res,
        modularity: res0.modularity,
    }];

    let mut stalled_prev = num_clusters == n;
    for round in 1..config.max_outer_rounds {
        if num_clusters <= config.max_coarse_nodes || current_graph.num_edges() == 0 {
            break;
        }
        if stalled_prev {
            cur_res *= config.resolution_decay;
            if cur_res < 1e-6 {
                cur_res = 0.0;
                cur_min_gain = 0.0;
            }
        }
        let l_cfg = LouvainConfig {
            max_passes: config.max_passes,
            resolution: cur_res,
            min_modularity_gain: cur_min_gain,
            seed: config.seed.wrapping_add(round as u64),
        };
        let res = louvain(&current_graph, &l_cfg)?;
        if res.num_communities >= current_graph.num_nodes() {
            if cur_res == 0.0 && cur_min_gain == 0.0 {
                break;
            }
            stalled_prev = true;
            continue;
        }

        let coarse = coarsen_graph(&current_graph, &res.labels)?;
        for cid in &mut fine_to_coarsest {
            *cid = coarse.node_to_coarse[*cid as usize];
        }
        num_clusters = coarse.num_clusters;
        cumulative_sizes.clear();
        cumulative_sizes.resize(num_clusters, 0);
        for &cid in &fine_to_coarsest {
            cumulative_sizes[cid as usize] += 1;
        }
        current_graph = coarse.graph.clone();
        levels.push(CoarsenedHierarchyLevel {
            coarse,
            resolution: cur_res,
            modularity: res.modularity,
        });
        stalled_prev = false;
    }

    Ok(CoarsenedHierarchy {
        levels,
        coarsest: CoarsenedGraph {
            graph: current_graph,
            node_to_coarse: fine_to_coarsest,
            coarse_sizes: cumulative_sizes,
            num_clusters,
        },
    })
}

/// Projects a 2D layout of coarse super-nodes onto fine vertices using per-cluster
/// reserved radii (`jitter_scale * cluster_radii[cid]`) for deterministic initial disk jitter.
pub fn project_clustered_layout_with_radii<S: Scalar>(
    coarse_layout: &Layout2d<S>,
    node_to_coarse: &[u32],
    cluster_radii: &[f64],
    jitter_scale: f64,
    seed: u64,
) -> CoreResult<Layout2d<S>> {
    if !jitter_scale.is_finite() {
        return Err(CoreError::non_finite(
            "project_clustered_layout_with_radii::jitter_scale",
        ));
    }
    if jitter_scale < 0.0 {
        return Err(CoreError::invalid_input(
            "jitter_scale must be non-negative",
        ));
    }
    coarse_layout.validate_finite()?;

    let num_clusters = coarse_layout.len();
    if cluster_radii.len() != num_clusters {
        return Err(CoreError::dimension_mismatch(
            num_clusters,
            cluster_radii.len(),
            "project_clustered_layout_with_radii::cluster_radii",
        ));
    }
    for &rad in cluster_radii {
        if !rad.is_finite() {
            return Err(CoreError::non_finite(
                "project_clustered_layout_with_radii::cluster_radii",
            ));
        }
        if rad < 0.0 {
            return Err(CoreError::invalid_input(
                "cluster_radii entries must be non-negative",
            ));
        }
    }

    let n = node_to_coarse.len();
    if n == 0 {
        return Ok(Layout2d::zeros(0));
    }

    let mut cluster_sizes = vec![0usize; num_clusters];
    for &c in node_to_coarse {
        let cid = c as usize;
        if cid >= num_clusters {
            return Err(CoreError::index_out_of_bounds(
                cid,
                num_clusters,
                "project_clustered_layout_with_radii::node_to_coarse",
            ));
        }
        cluster_sizes[cid] += 1;
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let cx = coarse_layout.x();
    let cy = coarse_layout.y();
    let mut fine_x = Vec::with_capacity(n);
    let mut fine_y = Vec::with_capacity(n);

    for &c in node_to_coarse {
        let cid = c as usize;
        let base_x = cx[cid].to_f64();
        let base_y = cy[cid].to_f64();
        let r = jitter_scale * cluster_radii[cid];
        if r > 0.0 && cluster_sizes[cid] > 1 {
            let angle = std::f64::consts::TAU * rng.next_f64();
            let rad = r * rng.next_f64().sqrt();
            let (sin_a, cos_a) = angle.sin_cos();
            fine_x.push(S::from_f64(base_x + rad * cos_a));
            fine_y.push(S::from_f64(base_y + rad * sin_a));
        } else {
            fine_x.push(S::from_f64(base_x));
            fine_y.push(S::from_f64(base_y));
        }
    }

    Layout2d::from_xy(fine_x, fine_y)
}

/// Aligns fine-level cluster centroids toward coarse super-node positions via rigid
/// per-cluster translation $x_i \leftarrow x_i + \gamma_{\mathrm{tether}} (x_C - \bar{x}_C)$
/// without rigid disk containment, preserving anisotropic intra-cluster geometry.
pub fn align_clustered_centroids<S: Scalar>(
    fine_layout: &mut Layout2d<S>,
    coarse_layout: &Layout2d<S>,
    node_to_coarse: &[u32],
    gamma_tether: f64,
) -> CoreResult<()> {
    if !gamma_tether.is_finite() {
        return Err(CoreError::non_finite(
            "align_clustered_centroids::gamma_tether",
        ));
    }
    if !(0.0..=1.0).contains(&gamma_tether) {
        return Err(CoreError::invalid_input(format!(
            "gamma_tether must be in [0.0, 1.0], got {gamma_tether}"
        )));
    }
    fine_layout.validate_finite()?;
    coarse_layout.validate_finite()?;

    let n = fine_layout.len();
    if node_to_coarse.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            node_to_coarse.len(),
            "align_clustered_centroids::node_to_coarse",
        ));
    }
    let num_clusters = coarse_layout.len();
    for &c in node_to_coarse {
        let cid = c as usize;
        if cid >= num_clusters {
            return Err(CoreError::index_out_of_bounds(
                cid,
                num_clusters,
                "align_clustered_centroids::node_to_coarse",
            ));
        }
    }
    if n == 0 || gamma_tether == 0.0 {
        return Ok(());
    }

    let mut sum_x = vec![0.0_f64; num_clusters];
    let mut sum_y = vec![0.0_f64; num_clusters];
    let mut counts = vec![0usize; num_clusters];

    {
        let fx = fine_layout.x();
        let fy = fine_layout.y();
        for i in 0..n {
            let cid = node_to_coarse[i] as usize;
            sum_x[cid] += fx[i].to_f64();
            sum_y[cid] += fy[i].to_f64();
            counts[cid] += 1;
        }
    }

    let cx = coarse_layout.x();
    let cy = coarse_layout.y();
    let mut shift_x = vec![0.0_f64; num_clusters];
    let mut shift_y = vec![0.0_f64; num_clusters];
    for cid in 0..num_clusters {
        if counts[cid] > 0 {
            let inv_cnt = 1.0 / (counts[cid] as f64);
            let mean_x = sum_x[cid] * inv_cnt;
            let mean_y = sum_y[cid] * inv_cnt;
            shift_x[cid] = gamma_tether * (cx[cid].to_f64() - mean_x);
            shift_y[cid] = gamma_tether * (cy[cid].to_f64() - mean_y);
        }
    }

    let (fx, fy) = fine_layout.xy_mut();
    for i in 0..n {
        let cid = node_to_coarse[i] as usize;
        fx[i] = S::from_f64(fx[i].to_f64() + shift_x[cid]);
        fy[i] = S::from_f64(fy[i].to_f64() + shift_y[cid]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coarsen_and_project_roundtrip() {
        // Two triangles {0, 1, 2} and {3, 4, 5} linked by two edges (0, 3) and (2, 4).
        let g = CsrGraph::<f64>::from_edges(
            6,
            &[
                (0, 1),
                (1, 2),
                (0, 2),
                (3, 4),
                (4, 5),
                (3, 5),
                (0, 3),
                (2, 4),
            ],
            None,
            false,
            false,
        )
        .unwrap();

        // Non-contiguous large labels verify canonical compaction without HashMap.
        let labels = [100, 100, 100, u32::MAX - 5, u32::MAX - 5, u32::MAX - 5];
        let coarse = coarsen_graph(&g, &labels).unwrap();
        assert_eq!(coarse.num_clusters, 2);
        assert_eq!(coarse.node_to_coarse, vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(coarse.coarse_sizes, vec![3, 3]);
        assert_eq!(coarse.graph.num_nodes(), 2);
        assert_eq!(coarse.graph.num_edges(), 1);
        assert_eq!(coarse.graph.neighbors(0), &[1]);
        assert_eq!(coarse.graph.neighbor_weights(0), Some(&[2.0][..]));

        let coarse_layout =
            Layout2d::from_xy(vec![-5.0_f64, 5.0_f64], vec![1.0_f64, -1.0_f64]).unwrap();
        let fine1 =
            project_clustered_layout(&coarse_layout, &coarse.node_to_coarse, 0.5, 42).unwrap();
        let fine2 =
            project_clustered_layout(&coarse_layout, &coarse.node_to_coarse, 0.5, 42).unwrap();
        assert_eq!(fine1, fine2);
        assert_eq!(fine1.len(), 6);

        for (i, &cid_u32) in coarse.node_to_coarse.iter().enumerate() {
            let cid = cid_u32 as usize;
            let dx = fine1.x()[i] - coarse_layout.x()[cid];
            let dy = fine1.y()[i] - coarse_layout.y()[cid];
            assert!(dx.hypot(dy) <= 0.5 + 1e-12);
        }

        // Zero spread radius recovers exact cluster centers.
        let exact =
            project_clustered_layout(&coarse_layout, &coarse.node_to_coarse, 0.0, 42).unwrap();
        assert_eq!(exact.x(), &[-5.0, -5.0, -5.0, 5.0, 5.0, 5.0]);
        assert_eq!(exact.y(), &[1.0, 1.0, 1.0, -1.0, -1.0, -1.0]);

        // Weighted f32 coarsening + singleton cluster preservation in f32 projection.
        let wg = CsrGraph::from_edges(
            4,
            &[(0, 1), (0, 2), (1, 2), (2, 3)],
            Some(&[2.0_f32, 3.0, 4.0, 1.5]),
            false,
            false,
        )
        .unwrap();
        let wc = coarsen_graph(&wg, &[7, 7, 9, 7]).unwrap();
        assert_eq!(wc.num_clusters, 2);
        assert_eq!(wc.node_to_coarse, vec![0, 0, 1, 0]);
        assert_eq!(wc.coarse_sizes, vec![3, 1]);
        assert_eq!(wc.graph.neighbor_weights(0), Some(&[8.5_f64][..]));

        let cl_f32 = Layout2d::from_xy(vec![1.0_f32, 4.0_f32], vec![2.0_f32, -3.0_f32]).unwrap();
        let fine_f32 = project_clustered_layout(&cl_f32, &wc.node_to_coarse, 0.25_f32, 99).unwrap();
        assert!(fine_f32.is_all_finite());
        // Singleton cluster (node 2 -> coarse 1) sits directly at coarse center.
        assert_eq!(fine_f32.x()[2], 4.0_f32);
        assert_eq!(fine_f32.y()[2], -3.0_f32);
    }

    #[test]
    fn test_coarsen_and_project_errors_and_empty() {
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let ce = coarsen_graph(&empty, &[]).unwrap();
        assert_eq!(ce.num_clusters, 0);
        assert_eq!(ce.graph.num_nodes(), 0);
        let pe = project_clustered_layout(&Layout2d::<f32>::zeros(0), &[], 0.1_f32, 1).unwrap();
        assert!(pe.is_empty());

        let directed = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, true, false).unwrap();
        assert!(matches!(
            coarsen_graph(&directed, &[0, 1]),
            Err(CoreError::InvalidInput(_))
        ));

        let g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
        assert!(matches!(
            coarsen_graph(&g, &[0]),
            Err(CoreError::DimensionMismatch { .. })
        ));

        let cl = Layout2d::from_xy(vec![0.0_f64, 1.0], vec![0.0, 1.0]).unwrap();
        assert!(matches!(
            project_clustered_layout(&cl, &[0, 2], 0.5, 42),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
        assert!(matches!(
            project_clustered_layout(&cl, &[0, 1], -0.1, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            project_clustered_layout(&cl, &[0, 1], f64::NAN, 42),
            Err(CoreError::NonFiniteValue { .. })
        ));
        let bad_cl = Layout2d::from_xy(vec![f64::NAN, 1.0], vec![0.0, 1.0]).unwrap();
        assert!(matches!(
            project_clustered_layout(&bad_cl, &[0, 1], 0.5, 42),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn test_louvain_coarsen_hierarchy_and_disconnected_floor() {
        // Connected barbell of 4 cliques of size 5 (20 nodes) contracted to <= 6 nodes.
        let mut edges = Vec::new();
        for c in 0..4 {
            let off = c * 5;
            for i in 0..5 {
                for j in (i + 1)..5 {
                    edges.push(((off + i) as u32, (off + j) as u32));
                }
            }
        }
        edges.extend([(4, 5), (9, 10), (14, 15)]);
        let g = CsrGraph::<f64>::from_edges(20, &edges, None, false, false).unwrap();

        let cfg = CoarseLouvainConfig {
            max_coarse_nodes: 3,
            resolution: 1.0,
            resolution_decay: 0.5,
            max_passes: 50,
            min_modularity_gain: 1e-7,
            max_outer_rounds: 16,
            seed: 42,
        };
        let h1 = louvain_coarsen(&g, &cfg).unwrap();
        let h2 = louvain_coarsen(&g, &cfg).unwrap();
        assert_eq!(h1, h2);
        assert!(h1.coarsest.num_clusters <= 3);
        assert_eq!(h1.coarsest.node_to_coarse.len(), 20);
        let sum_sizes: u32 = h1.coarsest.coarse_sizes.iter().sum();
        assert_eq!(sum_sizes, 20);

        // Disconnected graph with 4 isolated triangles (> max_coarse_nodes = 2):
        // must stop cleanly at the connected-component floor (4 clusters) without looping.
        let mut disc_edges = Vec::new();
        for c in 0..4 {
            let off = (c * 3) as u32;
            disc_edges.extend([(off, off + 1), (off + 1, off + 2), (off, off + 2)]);
        }
        let disc = CsrGraph::<f64>::from_edges(12, &disc_edges, None, false, false).unwrap();
        let disc_cfg = CoarseLouvainConfig {
            max_coarse_nodes: 2,
            ..CoarseLouvainConfig::default()
        };
        let hd = louvain_coarsen(&disc, &disc_cfg).unwrap();
        assert_eq!(hd.coarsest.num_clusters, 4);
        assert_eq!(hd.coarsest.graph.num_edges(), 0);

        // Empty and edgeless edge cases.
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        assert_eq!(
            louvain_coarsen(&empty, &cfg).unwrap().coarsest.num_clusters,
            0
        );
        let edgeless = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
        assert_eq!(
            louvain_coarsen(&edgeless, &cfg)
                .unwrap()
                .coarsest
                .num_clusters,
            5
        );
    }

    #[test]
    fn test_project_with_radii_and_centroid_alignment_preserves_filament() {
        // Pitfall #7 oracle: cluster 0 is an elongated horizontal filament (aspect ratio 10:1),
        // cluster 1 is a singleton. Rigid translation must shift cluster 0 centroid to (5.0, -3.0)
        // while preserving its > 2.0 aspect ratio identically.
        let coarse = Layout2d::from_xy(vec![5.0_f64, -5.0], vec![-3.0, 2.0]).unwrap();
        let n2c = [0u32, 0, 0, 0, 0, 1];

        let mut fine = Layout2d::from_xy(
            vec![0.0, 2.5, 5.0, 7.5, 10.0, 0.0],
            vec![0.0, 0.25, 0.0, -0.25, 0.0, 0.0],
        )
        .unwrap();
        align_clustered_centroids(&mut fine, &coarse, &n2c, 1.0).unwrap();

        let mean_x: f64 = fine.x()[..5].iter().sum::<f64>() / 5.0;
        let mean_y: f64 = fine.y()[..5].iter().sum::<f64>() / 5.0;
        assert!((mean_x - 5.0).abs() < 1e-12);
        assert!((mean_y - (-3.0)).abs() < 1e-12);
        assert!((fine.x()[5] - (-5.0)).abs() < 1e-12);
        assert!((fine.y()[5] - 2.0).abs() < 1e-12);

        let span_x = fine.x()[..5]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            - fine.x()[..5].iter().copied().fold(f64::INFINITY, f64::min);
        let span_y = fine.y()[..5]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            - fine.y()[..5].iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            span_x / span_y > 10.0,
            "filament distorted: span_x={span_x}, span_y={span_y}"
        );

        // Radius-scaled projection respects per-cluster radius bounds.
        let proj = project_clustered_layout_with_radii(&coarse, &n2c, &[2.0, 4.0], 0.5, 7).unwrap();
        for i in 0..5 {
            let dist = (proj.x()[i] - 5.0).hypot(proj.y()[i] - (-3.0));
            assert!(dist <= 1.0 + 1e-12);
        }
        assert_eq!(proj.x()[5], -5.0);
        assert_eq!(proj.y()[5], 2.0);
    }
}
