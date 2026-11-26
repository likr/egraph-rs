//! Community coherence, cluster area faithfulness, and cluster-evaluation metrics for `egraph-rs`.
//!
//! `egraph-metrics-cluster` provides single-threaded, pure-Rust cluster evaluation primitives:
//! - [`community`]: Spatial community coherence (`adjusted_patch_nig`,
//!   `adjusted_patch_nig_multiscale`).
//! - [`cluster_area`]: Grid-based Cluster Area Faithfulness (`G-CAF`) combining
//!   resolution, adjusted spatial purity, area proportion, and 8-connected coherence.
//! - [`cluster_baselines`] *(feature `eval-baselines`)*: External/internal cluster
//!   evaluation, visual separation, Sprawlter-NN, and Chen (2020) NOR baseline metrics.

pub mod cluster_area;
#[cfg(feature = "eval-baselines")]
pub mod cluster_baselines;
pub mod community;

pub use cluster_area::{
    ClusterAreaConfig, ClusterAreaDetail, ClusterAreaReport, cluster_area_faithfulness,
    cluster_area_faithfulness_detail, cluster_area_faithfulness_detail_with_weights,
    cluster_area_faithfulness_with_weights,
};
#[cfg(feature = "eval-baselines")]
pub use cluster_baselines::{
    NorMetrics, PartitionAgreement, calinski_harabasz_score, calinski_harabasz_score_manifold,
    calinski_harabasz_score_nd, class_density_measure, cq_hac, cq_hac_manifold, cq_hac_nd,
    cq_kmeans, cq_kmeans_manifold, cq_kmeans_nd, davies_bouldin_score,
    davies_bouldin_score_manifold, davies_bouldin_score_nd, distance_consistency,
    distance_consistency_manifold, distance_consistency_nd, distribution_consistency,
    distribution_consistency_manifold, distribution_consistency_nd, histogram_density_measure,
    neighborhood_hit, neighborhood_hit_manifold, neighborhood_hit_nd, nor_metrics,
    nor_metrics_without_graph, pairwise_distance_overlap_error,
    pairwise_distance_overlap_error_manifold, pairwise_distance_overlap_error_nd,
    partition_agreement, sepme_gab, sepme_gab_manifold, sepme_gab_nd, sepme_gong,
    sepme_gong_manifold, sepme_gong_nd, sepme_knng, sepme_knng_manifold, sepme_knng_nd, sepme_rng,
    sepme_rng_manifold, sepme_rng_nd, silhouette_score, silhouette_score_manifold,
    silhouette_score_nd, sprawlter_nn, sprawlter_nn_radii,
};
pub use community::{
    AdjustedPatchNigDetail, adjusted_patch_nig, adjusted_patch_nig_detail,
    adjusted_patch_nig_multiscale,
};

/// Returns the crate version.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn metrics_cluster_crate_version_non_empty() {
        assert!(!crate_version().is_empty());
    }
}
