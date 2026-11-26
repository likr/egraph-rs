use egraph_clustering::{KMeansConfig, kmeans_plus_plus};
use egraph_core::{CoreError, CoreResult, LayoutNd, Scalar};
use egraph_spatial_trees::{DynamicKdTree, KdQueryScratch};

use super::agreement::partition_agreement;
use super::cluster_quality::upgma_hac_labels;
use super::nd_proximity::{
    ProximityModeNd, build_dynamic_kdtree_f64, euclidean_dist_nd, is_gong_directed_edge_nd,
    proximity_edges_nd_with_tree, validate_nd_labels_and_tree,
};
use super::separation::{dir_cpt_from_degree_counts, dir_cpt_from_undirected_edges};
use super::{PartitionAgreement, compact_labels, validate_layout_nd_labels};

fn normalized_coords_for_cq_nd<S: Scalar>(layout: &LayoutNd<S>) -> Option<Vec<f64>> {
    let n = layout.num_nodes();
    let d = layout.dims();
    if n < 2 {
        return None;
    }
    let raw = layout.coords();
    let mut mins = vec![f64::INFINITY; d];
    let mut maxs = vec![f64::NEG_INFINITY; d];
    let mut means = vec![0.0_f64; d];
    for i in 0..n {
        let base = i * d;
        for axis in 0..d {
            let v = raw[base + axis].to_f64();
            mins[axis] = mins[axis].min(v);
            maxs[axis] = maxs[axis].max(v);
            means[axis] += v;
        }
    }
    let mut span = 0.0_f64;
    for axis in 0..d {
        span = span.max(maxs[axis] - mins[axis]);
        means[axis] /= n as f64;
    }
    if span <= 1e-15 {
        return None;
    }

    let mut sum_sq = 0.0_f64;
    for i in 0..n {
        let base = i * d;
        for axis in 0..d {
            let diff = raw[base + axis].to_f64() - means[axis];
            sum_sq += diff * diff;
        }
    }
    let rms = (sum_sq / n as f64).sqrt();
    if rms <= 1e-15 {
        return None;
    }
    let inv_rms = 1.0 / rms;
    let mut flat = Vec::with_capacity(n * d);
    for i in 0..n {
        let base = i * d;
        for axis in 0..d {
            flat.push((raw[base + axis].to_f64() - means[axis]) * inv_rms);
        }
    }
    Some(flat)
}

/// Evaluates Cluster Quality via $k$-means++ clustering on centered, RMS-normalized $N$-D
/// layout coordinates (`CQ-kmeans`), returning [`PartitionAgreement`] against `labels`.
///
/// Returns [`PartitionAgreement::default()`] on degenerate cases (`n < 2`, `k < 2`, `k >= n`,
/// or collapsed coordinates with `span <= 1e-15` or `rms <= 1e-15`).
pub fn cq_kmeans_nd<S: Scalar>(
    layout: &LayoutNd<S>,
    labels: &[u32],
    n_init: usize,
    max_iters: usize,
    seed: u64,
) -> CoreResult<PartitionAgreement> {
    if n_init == 0 {
        return Err(CoreError::invalid_input(
            "cq_kmeans_nd requires n_init >= 1",
        ));
    }
    if max_iters == 0 {
        return Err(CoreError::invalid_input(
            "cq_kmeans_nd requires max_iters >= 1",
        ));
    }
    validate_layout_nd_labels(layout, labels, "cq_kmeans_nd")?;

    let n = layout.num_nodes();
    let d = layout.dims();
    let (_, _, k) = compact_labels(labels);
    if n < 2 || k < 2 || k >= n {
        return Ok(PartitionAgreement::default());
    }
    let Some(points_flat) = normalized_coords_for_cq_nd(layout) else {
        return Ok(PartitionAgreement::default());
    };

    let cfg = KMeansConfig {
        k,
        max_iterations: max_iters,
        tol: 1e-4,
        n_init,
        seed,
    };
    let res = kmeans_plus_plus(&points_flat, n, d, &cfg)?;
    partition_agreement(labels, &res.labels)
}

/// Evaluates Cluster Quality via exact $O(N^2)$ Murtagh NN-chain average-linkage (UPGMA)
/// hierarchical agglomerative clustering on centered, RMS-normalized $N$-D layout coordinates
/// (`CQ-HAC`), returning [`PartitionAgreement`] against `labels`.
///
/// Returns [`PartitionAgreement::default()`] on degenerate cases (`n < 2`, `k < 2`, `k >= n`,
/// or collapsed coordinates).
pub fn cq_hac_nd<S: Scalar>(
    layout: &LayoutNd<S>,
    labels: &[u32],
) -> CoreResult<PartitionAgreement> {
    validate_layout_nd_labels(layout, labels, "cq_hac_nd")?;

    let n = layout.num_nodes();
    let d = layout.dims();
    let (_, _, k) = compact_labels(labels);
    if n < 2 || k < 2 || k >= n {
        return Ok(PartitionAgreement::default());
    }
    let Some(points_flat) = normalized_coords_for_cq_nd(layout) else {
        return Ok(PartitionAgreement::default());
    };

    let matrix_len = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("cq_hac_nd N x N matrix size overflow"))?;
    let mut dist = vec![0.0_f64; matrix_len];
    for i in 0..n {
        let pi = &points_flat[i * d..(i + 1) * d];
        for j in (i + 1)..n {
            let pj = &points_flat[j * d..(j + 1) * d];
            let d_ij = euclidean_dist_nd(pi, pj);
            dist[i * n + j] = d_ij;
            dist[j * n + i] = d_ij;
        }
    }

    let pred_labels = upgma_hac_labels(n, k, &mut dist);
    partition_agreement(labels, &pred_labels)
}

/// Computes Aupetit & Sedlmair (2016) `SepMe` $k$-Nearest Neighbor Graph (`KNNG DIR CPT`)
/// on an $N$-D [`LayoutNd`] in `[0.0, 1.0]`.
pub fn sepme_knng_nd<S: Scalar>(layout: &LayoutNd<S>, labels: &[u32], k: usize) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input("sepme_knng_nd requires k >= 1"));
    }
    validate_layout_nd_labels(layout, labels, "sepme_knng_nd")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    if layout.num_nodes() == 1 {
        return Ok(1.0);
    }
    let tree = build_dynamic_kdtree_f64(layout)?;
    sepme_knng_nd_with_tree(layout, labels, k, &tree)
}

#[doc(hidden)]
pub fn sepme_knng_nd_with_tree<S: Scalar>(
    layout: &LayoutNd<S>,
    labels: &[u32],
    k: usize,
    tree: &DynamicKdTree<f64>,
) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input("sepme_knng_nd requires k >= 1"));
    }
    validate_nd_labels_and_tree(layout, labels, tree, "sepme_knng_nd_with_tree")?;

    let n = layout.num_nodes();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }

    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let k_eff = k.min(n - 1);
    let mut scratch = KdQueryScratch::with_capacity(k_eff);
    let mut nbrs = Vec::with_capacity(k_eff);
    let mut deg = vec![0usize; n];
    let mut same = vec![0usize; n];

    for i in 0..n {
        tree.k_nearest_excluding_self_with_scratch(i, k_eff, &mut scratch, &mut nbrs)?;
        deg[i] = nbrs.len();
        let ci = comp[i];
        same[i] = nbrs.iter().filter(|&&(j, _)| comp[j] == ci).count();
    }

    Ok(dir_cpt_from_degree_counts(
        &comp,
        &class_sizes,
        k_classes,
        &deg,
        &same,
    ))
}

/// Computes Aupetit & Sedlmair (2016) `SepMe` Relative Neighborhood Graph (`RNG DIR CPT`)
/// on an $N$-D [`LayoutNd`] in `[0.0, 1.0]`.
pub fn sepme_rng_nd<S: Scalar>(layout: &LayoutNd<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_nd_labels(layout, labels, "sepme_rng_nd")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    if layout.num_nodes() == 1 {
        return Ok(1.0);
    }
    let tree = build_dynamic_kdtree_f64(layout)?;
    sepme_rng_nd_with_tree(layout, labels, &tree)
}

#[doc(hidden)]
pub fn sepme_rng_nd_with_tree<S: Scalar>(
    layout: &LayoutNd<S>,
    labels: &[u32],
    tree: &DynamicKdTree<f64>,
) -> CoreResult<f64> {
    validate_nd_labels_and_tree(layout, labels, tree, "sepme_rng_nd_with_tree")?;
    let n = layout.num_nodes();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }
    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let edges = proximity_edges_nd_with_tree(tree, ProximityModeNd::RelativeNeighborhood)?;
    Ok(dir_cpt_from_undirected_edges(
        n,
        &comp,
        &class_sizes,
        k_classes,
        &edges,
    ))
}

/// Computes Aupetit & Sedlmair (2016) `SepMe` Gabriel Graph (`GAB DIR CPT`) on an $N$-D
/// [`LayoutNd`] in `[0.0, 1.0]`.
pub fn sepme_gab_nd<S: Scalar>(layout: &LayoutNd<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_nd_labels(layout, labels, "sepme_gab_nd")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    if layout.num_nodes() == 1 {
        return Ok(1.0);
    }
    let tree = build_dynamic_kdtree_f64(layout)?;
    sepme_gab_nd_with_tree(layout, labels, &tree)
}

#[doc(hidden)]
pub fn sepme_gab_nd_with_tree<S: Scalar>(
    layout: &LayoutNd<S>,
    labels: &[u32],
    tree: &DynamicKdTree<f64>,
) -> CoreResult<f64> {
    validate_nd_labels_and_tree(layout, labels, tree, "sepme_gab_nd_with_tree")?;
    let n = layout.num_nodes();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }
    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let edges = proximity_edges_nd_with_tree(tree, ProximityModeNd::Gabriel)?;
    Ok(dir_cpt_from_undirected_edges(
        n,
        &comp,
        &class_sizes,
        k_classes,
        &edges,
    ))
}

/// Computes Aupetit & Sedlmair (2016) `SepMe` $\gamma$-Observable Neighbor Graph (`GONG DIR CPT`,
/// default `gamma = 0.35`) on an $N$-D [`LayoutNd`] in `[0.0, 1.0]`.
pub fn sepme_gong_nd<S: Scalar>(
    layout: &LayoutNd<S>,
    labels: &[u32],
    gamma: f64,
) -> CoreResult<f64> {
    if !gamma.is_finite() {
        return Err(CoreError::non_finite("sepme_gong_nd::gamma"));
    }
    if !(0.0..=1.0).contains(&gamma) {
        return Err(CoreError::invalid_input(
            "sepme_gong_nd requires gamma in [0.0, 1.0]",
        ));
    }
    validate_layout_nd_labels(layout, labels, "sepme_gong_nd")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    if layout.num_nodes() == 1 {
        return Ok(1.0);
    }
    let tree = build_dynamic_kdtree_f64(layout)?;
    sepme_gong_nd_with_tree(layout, labels, gamma, &tree)
}

#[doc(hidden)]
pub fn sepme_gong_nd_with_tree<S: Scalar>(
    layout: &LayoutNd<S>,
    labels: &[u32],
    gamma: f64,
    tree: &DynamicKdTree<f64>,
) -> CoreResult<f64> {
    if !gamma.is_finite() {
        return Err(CoreError::non_finite("sepme_gong_nd::gamma"));
    }
    if !(0.0..=1.0).contains(&gamma) {
        return Err(CoreError::invalid_input(
            "sepme_gong_nd requires gamma in [0.0, 1.0]",
        ));
    }
    validate_nd_labels_and_tree(layout, labels, tree, "sepme_gong_nd_with_tree")?;

    let n = layout.num_nodes();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }

    let d = layout.dims();
    let coords = tree.coords_by_id();
    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let mut deg_out = vec![0usize; n];
    let mut same_out = vec![0usize; n];
    let mut mid_buf = vec![0.0_f64; d];
    let mut scratch = KdQueryScratch::with_capacity(0);
    let mut witness_buf = Vec::new();

    if gamma <= 0.5 {
        let gg_edges = proximity_edges_nd_with_tree(tree, ProximityModeNd::Gabriel)?;
        for (u, v) in gg_edges {
            let ui = u as usize;
            let vi = v as usize;
            let same_class = comp[ui] == comp[vi];
            if is_gong_directed_edge_nd(
                ui,
                vi,
                gamma,
                coords,
                d,
                tree,
                &mut mid_buf,
                &mut scratch,
                &mut witness_buf,
            )? {
                deg_out[ui] += 1;
                if same_class {
                    same_out[ui] += 1;
                }
            }
            if is_gong_directed_edge_nd(
                vi,
                ui,
                gamma,
                coords,
                d,
                tree,
                &mut mid_buf,
                &mut scratch,
                &mut witness_buf,
            )? {
                deg_out[vi] += 1;
                if same_class {
                    same_out[vi] += 1;
                }
            }
        }
    } else {
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if is_gong_directed_edge_nd(
                    i,
                    j,
                    gamma,
                    coords,
                    d,
                    tree,
                    &mut mid_buf,
                    &mut scratch,
                    &mut witness_buf,
                )? {
                    deg_out[i] += 1;
                    if comp[i] == comp[j] {
                        same_out[i] += 1;
                    }
                }
            }
        }
    }

    Ok(dir_cpt_from_degree_counts(
        &comp,
        &class_sizes,
        k_classes,
        &deg_out,
        &same_out,
    ))
}

/// Computes Paulovich et al. (2008) / Espadoto et al. (2021) Neighborhood Hit (`M_NH`, default
/// `k = 7`) on an $N$-D [`LayoutNd`] in `[0.0, 1.0]`, micro-averaged over all `N` vertices (`N < 2`
/// returns `0.0`).
pub fn neighborhood_hit_nd<S: Scalar>(
    layout: &LayoutNd<S>,
    labels: &[u32],
    k: usize,
) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input(
            "neighborhood_hit_nd requires k >= 1",
        ));
    }
    validate_layout_nd_labels(layout, labels, "neighborhood_hit_nd")?;
    if layout.num_nodes() < 2 {
        return Ok(0.0);
    }
    let tree = build_dynamic_kdtree_f64(layout)?;
    neighborhood_hit_nd_with_tree(layout, labels, k, &tree)
}

#[doc(hidden)]
pub fn neighborhood_hit_nd_with_tree<S: Scalar>(
    layout: &LayoutNd<S>,
    labels: &[u32],
    k: usize,
    tree: &DynamicKdTree<f64>,
) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input(
            "neighborhood_hit_nd requires k >= 1",
        ));
    }
    validate_nd_labels_and_tree(layout, labels, tree, "neighborhood_hit_nd_with_tree")?;

    let n = layout.num_nodes();
    if n < 2 {
        return Ok(0.0);
    }
    let k_eff = k.min(n - 1);
    let mut scratch = KdQueryScratch::with_capacity(k_eff);
    let mut nbrs = Vec::with_capacity(k_eff);
    let mut total_same = 0usize;

    for i in 0..n {
        tree.k_nearest_excluding_self_with_scratch(i, k_eff, &mut scratch, &mut nbrs)?;
        let li = labels[i];
        total_same += nbrs.iter().filter(|&&(j, _)| labels[j] == li).count();
    }

    Ok(total_same as f64 / (k_eff * n) as f64)
}
