//! Unified quality evaluation metrics and selective evaluation dispatcher for `egraph-rs`.
//!
//! `egraph-metrics` re-exports all layout geometry, stress, crossing, neighborhood,
//! proximity, triplet, community, and cluster evaluation primitives alongside the
//! target-selective metric dispatcher ([`evaluate_metrics`], [`MetricTarget`],
//! [`MetricConfig`], [`MetricInputs`], [`MetricReport`], [`ClusterBaselineConfig`],
//! and [`ClusterBaselineReport`]).

mod dispatcher;

pub use dispatcher::{
    ClusterBaselineConfig, ClusterBaselineReport, MetricConfig, MetricInputs, MetricReport,
    MetricTarget, evaluate_metrics, evaluate_metrics_with_communities,
    evaluate_metrics_with_community_weights, evaluate_metrics_with_inputs,
};
pub use egraph_metrics_cluster::{
    AdjustedPatchNigDetail, ClusterAreaConfig, ClusterAreaDetail, ClusterAreaReport,
    adjusted_patch_nig, adjusted_patch_nig_detail, adjusted_patch_nig_multiscale,
    cluster_area_faithfulness, cluster_area_faithfulness_detail,
    cluster_area_faithfulness_detail_with_weights, cluster_area_faithfulness_with_weights,
};
#[cfg(feature = "eval-baselines")]
pub use egraph_metrics_cluster::{
    NorMetrics, PartitionAgreement, calinski_harabasz_score, class_density_measure, cq_hac,
    cq_kmeans, davies_bouldin_score, distance_consistency, distribution_consistency,
    histogram_density_measure, neighborhood_hit, neighborhood_hit_with_index, nor_metrics,
    nor_metrics_without_graph, pairwise_distance_overlap_error, partition_agreement, sepme_gab,
    sepme_gab_with_index, sepme_gong, sepme_gong_with_index, sepme_knng, sepme_knng_with_index,
    sepme_rng, sepme_rng_with_index, silhouette_score, sprawlter_nn, sprawlter_nn_radii,
};
pub use egraph_metrics_geometry::{
    BfsScratch, CrossingPair, CrossingSummary, LayoutDegeneracy, MinimumAngleSummary,
    PrismDistortion, RtaEvaluation, StressEvaluation, TorusCrossing, TorusLayoutView, TorusView,
    all_pairs_bfs, all_pairs_bfs_into, angular_resolution, aspect_ratio, bfs_single_source,
    crossing_angle, crossing_angle_from_pairs, crossing_edges, crossing_edges_hyperbolic,
    crossing_edges_spherical, crossing_edges_torus, crossing_edges_torus_from_edges,
    crossing_number, crossing_number_from_pairs, crossing_summary, crosslessness,
    crosslessness_from_count, edge_angle, gabriel_graph_edges, gabriel_graph_edges_with_index,
    gabriel_graph_preservation, gabriel_graph_preservation_with_index, gabriel_graph_property,
    gabriel_graph_property_with_index, geodesic_distortion, hyperbolic_crossing_number,
    ideal_edge_lengths, ideal_edge_lengths_from_distances, ideal_edge_lengths_scaled,
    ideal_edge_lengths_with_unit, landmark_stratified_rta, landmark_stratified_rta_detail,
    landmark_stratified_rta_from_landmarks, landmark_stratified_rta_from_landmarks_f64,
    layout_degeneracy, layout_degeneracy_with_index, max_crossing_pairs, minimum_angle,
    minimum_angle_summary, neighborhood_preservation_1hop,
    neighborhood_preservation_1hop_with_index, neighborhood_preservation_2hop,
    neighborhood_preservation_2hop_with_index, node_resolution, node_resolution_with_index,
    prism_distortion, prism_distortion_with_edges, relative_neighborhood_edges,
    relative_neighborhood_edges_with_index, relative_neighborhood_preservation,
    relative_neighborhood_preservation_with_index, sampled_crossing_number,
    sampled_crossing_number_with_rng, sampled_crosslessness,
    sampled_neighborhood_preservation_1hop, sampled_neighborhood_preservation_1hop_with_index,
    sampled_neighborhood_preservation_2hop, sampled_neighborhood_preservation_2hop_with_index,
    sampled_scale_normalized_stress, sampled_scale_normalized_stress_detail,
    scale_normalized_stress, scale_normalized_stress_detail,
    scale_normalized_stress_from_distances, scale_normalized_stress_from_distances_detail,
    scale_normalized_stress_from_pairs, spherical_crossing_number, torus_crossing_number,
};
