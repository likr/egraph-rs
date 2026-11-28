//! Community coherence, cluster area faithfulness, and cluster-evaluation metrics for `egraph-rs`.
//!
//! `egraph-metrics-cluster` provides single-threaded, pure-Rust cluster evaluation primitives:
//! - Community coherence (`AdjustedPatchNigDetail`, `adjusted_patch_nig`,
//!   `adjusted_patch_nig_detail`, `adjusted_patch_nig_multiscale`).
//! - Grid-based Cluster Area Faithfulness (`ClusterAreaConfig`, `ClusterAreaReport`,
//!   `ClusterAreaDetail`, `cluster_area_faithfulness*`).
//! - Cluster baselines *(feature `eval-baselines`)*: External/internal cluster evaluation,
//!   visual separation (`SepMe`, `Neighborhood Hit`), `Sprawlter-NN`, and Chen (2020) `NOR`
//!   baseline metrics unified across [`egraph_core::MetricSpace`],
//!   [`egraph_core::EuclideanLayout`], and [`egraph_core::ManifoldLayout2d`].

#![cfg_attr(docsrs, feature(doc_cfg))]

mod cluster_area;
#[cfg(feature = "eval-baselines")]
mod cluster_baselines;
mod community;

pub use cluster_area::{
    ClusterAreaConfig, ClusterAreaDetail, ClusterAreaReport, cluster_area_faithfulness,
    cluster_area_faithfulness_detail, cluster_area_faithfulness_detail_with_weights,
    cluster_area_faithfulness_with_weights,
};
#[cfg(feature = "eval-baselines")]
#[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
pub use cluster_baselines::{
    NorMetrics, PartitionAgreement, calinski_harabasz_score, class_density_measure, cq_hac,
    cq_kmeans, davies_bouldin_score, distance_consistency, distribution_consistency,
    histogram_density_measure, neighborhood_hit, neighborhood_hit_with_index, nor_metrics,
    nor_metrics_without_graph, pairwise_distance_overlap_error, partition_agreement, sepme_gab,
    sepme_gab_with_index, sepme_gong, sepme_gong_with_index, sepme_knng, sepme_knng_with_index,
    sepme_rng, sepme_rng_with_index, silhouette_score, sprawlter_nn, sprawlter_nn_radii,
};
pub use community::{
    AdjustedPatchNigDetail, adjusted_patch_nig, adjusted_patch_nig_detail,
    adjusted_patch_nig_multiscale,
};
