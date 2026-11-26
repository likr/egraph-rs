//! Cluster-aware, visual separation, Sprawlter-NN, and NOR baseline evaluation metrics.
//!
//! Gated behind the `eval-baselines` Cargo feature to keep the minimal `egraph-metrics-cluster`
//! dependency graph free of `egraph-clustering` when baseline clustering metrics are not needed.
//!
//! All coordinates and extents are accumulated in `f64` internally regardless of layout
//! scalar type `S: Scalar`, all algorithms are strictly single-threaded and deterministic,
//! and all normalized score families (`DSC`, `DC`, `CDM`, `HDM`, `SepMe`, `M_NH`, `PDOE`)
//! are returned on `[0.0, 1.0]` (`paper_percent = 100.0 * rust_score`).

mod agreement;
mod cluster_quality;
mod consistency;
mod internal_scores;
mod nor;
mod separation;
mod sprawlter;
#[cfg(test)]
mod tests;

use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};

pub use agreement::partition_agreement;
pub use cluster_quality::{cq_hac, cq_kmeans};
pub use consistency::{class_density_measure, distance_consistency, distribution_consistency};
pub use internal_scores::{
    calinski_harabasz_score, davies_bouldin_score, pairwise_distance_overlap_error,
    silhouette_score,
};
pub use nor::{nor_metrics, nor_metrics_without_graph};
pub use separation::{neighborhood_hit, sepme_gab, sepme_gong, sepme_knng, sepme_rng};
#[doc(hidden)]
pub use separation::{
    neighborhood_hit_with_tree, sepme_gab_with_tree, sepme_gong_with_tree, sepme_knng_with_tree,
    sepme_rng_with_tree,
};
pub use sprawlter::{sprawlter_nn, sprawlter_nn_radii};

/// Partition agreement metrics comparing two discrete node labelings.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PartitionAgreement {
    /// Adjusted Rand Index (`sklearn.metrics.adjusted_rand_score`).
    pub ari: f64,
    /// Adjusted Mutual Information with arithmetic normalization (`nats`, `sklearn` default).
    pub ami: f64,
    /// Cluster homogeneity in `[0.0, 1.0]`.
    pub homogeneity: f64,
    /// Cluster completeness in `[0.0, 1.0]`.
    pub completeness: f64,
    /// Harmonic mean V-measure ($\beta = 1.0$) in `[0.0, 1.0]`.
    pub v_measure: f64,
    /// Fowlkes-Mallows Index in `[0.0, 1.0]`.
    pub fmi: f64,
}

/// Chen et al. (2020) Node Overlap Removal (NOR) 5-metric evaluation summary.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct NorMetrics {
    /// Orthogonal Ordering: Normalized Number of Inversions in `[0.0, 1.0]` (lower is better).
    pub oo_nni: f64,
    /// Spread Minimization: Convex Hull Area Ratio using $4N$ box corners (optimal `1.0`).
    pub sp_ch_a: f64,
    /// Global Shape: Bounding Box Improved Aspect Ratio $\ge 1.0$ using $4N$ box corners (optimal `1.0`).
    pub gs_bb_iar: f64,
    /// Node Movement: Translation-invariant Distance Moved Improved MSE (lower is better).
    pub nm_dm_imse: f64,
    /// Edge Length Preservation: Relative Standard Deviation of edge length ratios (lower is better).
    pub el_rsd: f64,
}

#[inline]
pub(super) fn canonical_zero(v: f64) -> f64 {
    if v == 0.0 { 0.0 } else { v }
}

#[inline]
pub(super) fn validate_layout_labels<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    context: &'static str,
) -> CoreResult<()> {
    if layout.len() != labels.len() {
        return Err(CoreError::dimension_mismatch(
            layout.len(),
            labels.len(),
            context,
        ));
    }
    layout.validate_finite()?;
    Ok(())
}

/// Compacts arbitrary `u32` labels into canonical `0..num_clusters` in ascending order
/// of first appearance without hash maps.
pub(super) fn compact_labels(labels: &[u32]) -> (Vec<u32>, Vec<usize>, usize) {
    let n = labels.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), 0);
    }

    let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut compacted = vec![0u32; n];
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
            compacted[i] = remap[idx];
        }
        let num_clusters = next_id as usize;
        let mut sizes = vec![0usize; num_clusters];
        for &c in &compacted {
            sizes[c as usize] += 1;
        }
        return (compacted, sizes, num_clusters);
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
    groups.sort_unstable_by_key(|&(first_node, _, _)| first_node);

    let num_clusters = groups.len();
    let mut sizes = Vec::with_capacity(num_clusters);
    for (cid, &(_, g_start, g_end)) in groups.iter().enumerate() {
        let cid_u32 = cid as u32;
        sizes.push(g_end - g_start);
        for &(_, orig_node) in &pairs[g_start..g_end] {
            compacted[orig_node as usize] = cid_u32;
        }
    }
    (compacted, sizes, num_clusters)
}

#[inline]
pub(super) fn cross_2d(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

pub(super) fn convex_hull_ccw(mut pts: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    for p in &mut pts {
        p[0] = canonical_zero(p[0]);
        p[1] = canonical_zero(p[1]);
    }
    pts.sort_unstable_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    pts.dedup_by(|b, a| a[0] == b[0] && a[1] == b[1]);

    if pts.len() <= 2 {
        return pts;
    }

    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(pts.len() + 1);
    for &p in &pts {
        while hull.len() >= 2 && cross_2d(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower_len = hull.len();
    for &p in pts.iter().rev().skip(1) {
        while hull.len() > lower_len
            && cross_2d(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
        {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}
