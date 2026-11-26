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
    NorMetrics, PartitionAgreement, calinski_harabasz_score, class_density_measure, cq_hac,
    cq_kmeans, davies_bouldin_score, distance_consistency, distribution_consistency,
    histogram_density_measure, neighborhood_hit, nor_metrics, nor_metrics_without_graph,
    pairwise_distance_overlap_error, partition_agreement, sepme_gab, sepme_gong, sepme_knng,
    sepme_rng, silhouette_score, sprawlter_nn, sprawlter_nn_radii,
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
