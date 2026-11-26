//! Quality evaluation metrics and selective evaluation dispatcher for `egraph-rs`.
//!
//! `egraph-metrics` re-exports all layout geometry, stress, crossing, neighborhood,
//! proximity, triplet, community, and cluster evaluation modules together with the
//! target-selective [`dispatcher`]:
//! - [`stress`]: Reusable `VisitStamp`-backed BFS shortest paths, closed-form
//!   Scale-Normalized Stress (`SNS`), and $O(kN)$ pivot-sampled `SNS`.
//! - [`crossings`]: Exact sweep-line crossing counts/angles, torus crossings, and
//!   deterministic sampled crossing estimation.
//! - [`neighborhood`]: `StaticKdTree`-accelerated 1-hop and 2-hop neighborhood preservation.
//! - [`proximity`]: Gabriel graph property, hinge node resolution, and Eades et al. (2017)
//!   unweighted shape-based `S_RNG` / `S_GG` preservation metrics.
//! - [`geometry`]: Aspect ratio, angular resolution, edge angles, ideal edge lengths,
//!   and PRISM distortion ($\sigma_{\mathrm{dist}}, \sigma_{\mathrm{disp}}$).
//! - [`rta`]: $O(N)$-RAM streaming Landmark-Stratified Random Triplet Accuracy
//!   (`landmark_stratified_rta`).
//! - [`community`]: Spatial community coherence (`adjusted_patch_nig`).
//! - [`cluster_area`]: Grid-based Cluster Area Faithfulness (`G-CAF`) combining
//!   resolution, adjusted spatial purity, area proportion, and 8-connected coherence.
//! - [`cluster_baselines`] *(feature `eval-baselines`)*: External/internal cluster
//!   evaluation, visual separation, Sprawlter-NN, and Chen (2020) NOR baseline metrics.
//! - [`dispatcher`]: Target-selective evaluation avoiding unrequested work (`MetricTarget`,
//!   `MetricReport`, `evaluate_metrics`, `evaluate_metrics_with_communities`,
//!   `evaluate_metrics_with_community_weights`).

#[cfg(feature = "eval-baselines")]
pub use egraph_metrics_cluster::cluster_baselines;
pub use egraph_metrics_cluster::{cluster_area, community};
pub use egraph_metrics_geometry::{crossings, geometry, neighborhood, proximity, rta, stress};

pub mod dispatcher;

#[cfg(feature = "eval-baselines")]
pub use dispatcher::ClusterBaselineConfig;
pub use dispatcher::{
    MetricConfig, MetricInputs, MetricLayout, MetricReport, MetricTarget, evaluate_metrics,
    evaluate_metrics_nd, evaluate_metrics_nd_with_communities,
    evaluate_metrics_nd_with_community_weights, evaluate_metrics_nd_with_inputs,
    evaluate_metrics_with_communities, evaluate_metrics_with_community_weights,
    evaluate_metrics_with_inputs,
};
#[cfg(feature = "eval-baselines")]
pub use egraph_metrics_cluster::cluster_baselines::{
    NorMetrics, PartitionAgreement, calinski_harabasz_score, calinski_harabasz_score_nd,
    class_density_measure, cq_hac, cq_hac_nd, cq_kmeans, cq_kmeans_nd, davies_bouldin_score,
    davies_bouldin_score_nd, distance_consistency, distance_consistency_nd,
    distribution_consistency, distribution_consistency_nd, histogram_density_measure,
    neighborhood_hit, neighborhood_hit_nd, nor_metrics, nor_metrics_without_graph,
    pairwise_distance_overlap_error, pairwise_distance_overlap_error_nd, partition_agreement,
    sepme_gab, sepme_gab_nd, sepme_gong, sepme_gong_nd, sepme_knng, sepme_knng_nd, sepme_rng,
    sepme_rng_nd, silhouette_score, silhouette_score_nd, sprawlter_nn, sprawlter_nn_radii,
};
pub use egraph_metrics_cluster::{
    AdjustedPatchNigDetail, ClusterAreaConfig, ClusterAreaDetail, ClusterAreaReport,
    adjusted_patch_nig, adjusted_patch_nig_detail, adjusted_patch_nig_multiscale,
    cluster_area_faithfulness, cluster_area_faithfulness_detail,
    cluster_area_faithfulness_detail_with_weights, cluster_area_faithfulness_with_weights,
};
pub use egraph_metrics_geometry::{
    BfsScratch, CrossingPair, CrossingSummary, LayoutDegeneracy, MinimumAngleSummary,
    PrismDistortion, RtaEvaluation, StressEvaluation, TorusCrossing, all_pairs_bfs,
    all_pairs_bfs_into, angular_resolution, aspect_ratio, bfs_single_source, crossing_angle,
    crossing_angle_from_pairs, crossing_edges, crossing_edges_torus,
    crossing_edges_torus_from_edges, crossing_number, crossing_number_from_pairs, crossing_summary,
    crosslessness, crosslessness_from_count, edge_angle, gabriel_graph_edges,
    gabriel_graph_edges_nd, gabriel_graph_edges_with_tree, gabriel_graph_preservation,
    gabriel_graph_preservation_nd, gabriel_graph_preservation_with_tree, gabriel_graph_property,
    gabriel_graph_property_nd, gabriel_graph_property_with_tree, ideal_edge_lengths,
    ideal_edge_lengths_from_distances, ideal_edge_lengths_from_distances_nd, ideal_edge_lengths_nd,
    ideal_edge_lengths_scaled, ideal_edge_lengths_scaled_nd, ideal_edge_lengths_with_unit,
    ideal_edge_lengths_with_unit_nd, landmark_stratified_rta, landmark_stratified_rta_detail,
    landmark_stratified_rta_detail_nd, landmark_stratified_rta_from_landmarks,
    landmark_stratified_rta_from_landmarks_f64, landmark_stratified_rta_from_landmarks_f64_nd,
    landmark_stratified_rta_from_landmarks_nd, landmark_stratified_rta_nd, layout_degeneracy,
    layout_degeneracy_nd, max_crossing_pairs, minimum_angle, minimum_angle_summary,
    neighborhood_preservation_1hop, neighborhood_preservation_1hop_nd,
    neighborhood_preservation_1hop_with_tree, neighborhood_preservation_2hop,
    neighborhood_preservation_2hop_nd, neighborhood_preservation_2hop_with_tree, node_resolution,
    node_resolution_nd, node_resolution_with_tree, prism_distortion, prism_distortion_with_edges,
    relative_neighborhood_edges, relative_neighborhood_edges_nd,
    relative_neighborhood_edges_with_tree, relative_neighborhood_preservation,
    relative_neighborhood_preservation_nd, relative_neighborhood_preservation_with_tree,
    sampled_crossing_number, sampled_crossing_number_with_rng, sampled_crosslessness,
    sampled_neighborhood_preservation_1hop, sampled_neighborhood_preservation_1hop_nd,
    sampled_neighborhood_preservation_1hop_with_tree, sampled_neighborhood_preservation_2hop,
    sampled_neighborhood_preservation_2hop_nd, sampled_neighborhood_preservation_2hop_with_tree,
    sampled_scale_normalized_stress, sampled_scale_normalized_stress_detail,
    sampled_scale_normalized_stress_detail_nd, sampled_scale_normalized_stress_nd,
    scale_normalized_stress, scale_normalized_stress_detail, scale_normalized_stress_detail_nd,
    scale_normalized_stress_from_distances, scale_normalized_stress_from_distances_detail,
    scale_normalized_stress_from_distances_detail_nd, scale_normalized_stress_from_distances_nd,
    scale_normalized_stress_from_pairs, scale_normalized_stress_from_pairs_nd,
    scale_normalized_stress_nd,
};
