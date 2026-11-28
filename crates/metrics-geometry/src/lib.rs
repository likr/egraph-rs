//! Geometric, stress, crossing, neighborhood, proximity, and triplet graph-layout metrics for `egraph-rs`.
//!
//! `egraph-metrics-geometry` provides single-threaded, pure-Rust layout evaluation primitives:
//! - Stress (`BfsScratch`, `StressEvaluation`, `scale_normalized_stress*`,
//!   `sampled_scale_normalized_stress*`, `geodesic_distortion`): Reusable `VisitStamp`-backed BFS
//!   shortest paths, closed-form Scale-Normalized Stress (`SNS`), and $O(kN)$ pivot-sampled `SNS`
//!   across arbitrary [`egraph_core::MetricSpace`] layouts.
//! - Crossings (`CrossingPair`, `CrossingSummary`, `TorusCrossing`, `TorusView`,
//!   `TorusLayoutView`, `crossing_*`, `crosslessness*`): Exact sweep-line crossing counts/angles,
//!   non-Euclidean geodesic crossings, and deterministic sampled crossing estimation.
//! - Neighborhood (`neighborhood_preservation_*`, `sampled_neighborhood_preservation_*`):
//!   [`egraph_spatial_trees::NearestNeighborIndex`]-accelerated 1-hop and 2-hop neighborhood
//!   preservation across arbitrary [`egraph_core::MetricSpace`] layouts.
//! - Proximity (`gabriel_graph_*`, `relative_neighborhood_*`, `node_resolution*`):
//!   Gabriel graph property, hinge node resolution, and Eades et al. (2017) unweighted shape-based
//!   `S_RNG` / `S_GG` preservation metrics.
//! - Geometry (`LayoutDegeneracy`, `MinimumAngleSummary`, `PrismDistortion`, `aspect_ratio`,
//!   `angular_resolution`, `edge_angle`, `ideal_edge_lengths*`, `layout_degeneracy*`,
//!   `minimum_angle*`, `prism_distortion*`).
//! - Random Triplet Accuracy (`RtaEvaluation`, `landmark_stratified_rta*`): $O(N)$-RAM streaming
//!   Landmark-Stratified Random Triplet Accuracy across arbitrary [`egraph_core::MetricSpace`]
//!   layouts.

mod crossings;
mod geometry;
mod neighborhood;
mod proximity;
mod rta;
mod spatial_index;
mod stress;

use egraph_core::CsrGraph;

pub use crossings::{
    CrossingPair, CrossingSummary, TorusCrossing, TorusLayoutView, TorusView, crossing_angle,
    crossing_angle_from_pairs, crossing_edges, crossing_edges_hyperbolic, crossing_edges_spherical,
    crossing_edges_torus, crossing_edges_torus_from_edges, crossing_number,
    crossing_number_from_pairs, crossing_summary, crosslessness, crosslessness_from_count,
    hyperbolic_crossing_number, max_crossing_pairs, sampled_crossing_number,
    sampled_crossing_number_with_rng, sampled_crosslessness, spherical_crossing_number,
    torus_crossing_number,
};
pub use geometry::{
    LayoutDegeneracy, MinimumAngleSummary, PrismDistortion, angular_resolution, aspect_ratio,
    edge_angle, ideal_edge_lengths, ideal_edge_lengths_from_distances, ideal_edge_lengths_scaled,
    ideal_edge_lengths_with_unit, layout_degeneracy, layout_degeneracy_with_index, minimum_angle,
    minimum_angle_summary, prism_distortion, prism_distortion_with_edges,
};
pub use neighborhood::{
    neighborhood_preservation_1hop, neighborhood_preservation_1hop_with_index,
    neighborhood_preservation_2hop, neighborhood_preservation_2hop_with_index,
    sampled_neighborhood_preservation_1hop, sampled_neighborhood_preservation_1hop_with_index,
    sampled_neighborhood_preservation_2hop, sampled_neighborhood_preservation_2hop_with_index,
};
pub use proximity::{
    gabriel_graph_edges, gabriel_graph_edges_with_index, gabriel_graph_preservation,
    gabriel_graph_preservation_with_index, gabriel_graph_property,
    gabriel_graph_property_with_index, node_resolution, node_resolution_with_index,
    relative_neighborhood_edges, relative_neighborhood_edges_with_index,
    relative_neighborhood_preservation, relative_neighborhood_preservation_with_index,
};
pub use rta::{
    RtaEvaluation, landmark_stratified_rta, landmark_stratified_rta_detail,
    landmark_stratified_rta_from_landmarks, landmark_stratified_rta_from_landmarks_f64,
};
pub use stress::{
    BfsScratch, StressEvaluation, all_pairs_bfs, all_pairs_bfs_into, bfs_single_source,
    geodesic_distortion, sampled_scale_normalized_stress, sampled_scale_normalized_stress_detail,
    scale_normalized_stress, scale_normalized_stress_detail,
    scale_normalized_stress_from_distances, scale_normalized_stress_from_distances_detail,
    scale_normalized_stress_from_pairs,
};

/// Builds a deduplicated undirected adjacency list (`adj[u]` sorted, self-loops removed).
pub(crate) fn build_undirected_adjacency<W>(graph: &CsrGraph<W>) -> Vec<Vec<u32>> {
    let n = graph.num_nodes();
    let mut adj = vec![Vec::new(); n];
    for u in 0..n {
        let u32_u = u as u32;
        for &v in graph.neighbors(u) {
            if v != u32_u {
                adj[u].push(v);
                if graph.is_directed() {
                    adj[v as usize].push(u32_u);
                }
            }
        }
    }
    for row in &mut adj {
        row.sort_unstable();
        row.dedup();
    }
    adj
}

/// Extracts unique graph edges `(u, v)`.
///
/// For undirected graphs, returns canonical pairs with `u < v`. For directed graphs,
/// returns every stored `(u, v)` edge (ignoring self-loops).
pub(crate) fn extract_unique_edges<W>(graph: &CsrGraph<W>) -> Vec<(u32, u32)> {
    let n = graph.num_nodes();
    let directed = graph.is_directed();
    let mut edges = Vec::with_capacity(graph.num_edges());
    for u in 0..n {
        let u32_u = u as u32;
        for &v in graph.neighbors(u) {
            if v != u32_u && (directed || u32_u < v) {
                edges.push((u32_u, v));
            }
        }
    }
    edges
}
