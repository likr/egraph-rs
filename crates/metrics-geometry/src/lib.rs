//! Geometric, stress, crossing, neighborhood, proximity, and triplet graph-layout metrics for `egraph-rs`.
//!
//! `egraph-metrics-geometry` provides single-threaded, pure-Rust layout evaluation primitives:
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

pub mod crossings;
pub mod geometry;
mod nd_index;
pub mod neighborhood;
pub mod proximity;
pub mod rta;
pub mod stress;

use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, ManifoldLayout2d, Scalar};
use egraph_spatial_trees::StaticKdTree;

#[doc(hidden)]
pub use nd_index::{
    NdSpatialIndex, build_nd_spatial_index, euclidean_diameter_nd, validate_graph_layout_nd,
    validate_index_layout_nd, validate_layout_nd,
};

pub use crossings::{
    CrossingPair, CrossingSummary, TorusCrossing, TorusLayoutView, crossing_angle,
    crossing_angle_from_pairs, crossing_edges, crossing_edges_hyperbolic, crossing_edges_spherical,
    crossing_edges_torus, crossing_edges_torus_from_edges, crossing_number,
    crossing_number_from_pairs, crossing_summary, crosslessness, crosslessness_from_count,
    hyperbolic_crossing_number, max_crossing_pairs, sampled_crossing_number,
    sampled_crossing_number_with_rng, sampled_crosslessness, spherical_crossing_number,
    torus_crossing_number,
};
#[doc(hidden)]
pub use geometry::layout_degeneracy_nd_with_index;
pub use geometry::{
    LayoutDegeneracy, MinimumAngleSummary, PrismDistortion, angular_resolution, aspect_ratio,
    edge_angle, ideal_edge_lengths, ideal_edge_lengths_from_distances,
    ideal_edge_lengths_from_distances_manifold, ideal_edge_lengths_from_distances_nd,
    ideal_edge_lengths_manifold, ideal_edge_lengths_nd, ideal_edge_lengths_scaled,
    ideal_edge_lengths_scaled_manifold, ideal_edge_lengths_scaled_nd, ideal_edge_lengths_with_unit,
    ideal_edge_lengths_with_unit_manifold, ideal_edge_lengths_with_unit_nd, layout_degeneracy,
    layout_degeneracy_manifold, layout_degeneracy_nd, minimum_angle, minimum_angle_summary,
    prism_distortion, prism_distortion_with_edges,
};
#[doc(hidden)]
pub use neighborhood::{
    k_nearest_manifold_excluding_self_into, neighborhood_preservation_1hop_nd_with_index,
    neighborhood_preservation_2hop_nd_with_index,
    sampled_neighborhood_preservation_1hop_nd_with_index,
    sampled_neighborhood_preservation_2hop_nd_with_index,
};
pub use neighborhood::{
    neighborhood_preservation_1hop, neighborhood_preservation_1hop_manifold,
    neighborhood_preservation_1hop_nd, neighborhood_preservation_1hop_with_tree,
    neighborhood_preservation_2hop, neighborhood_preservation_2hop_manifold,
    neighborhood_preservation_2hop_nd, neighborhood_preservation_2hop_with_tree,
    sampled_neighborhood_preservation_1hop, sampled_neighborhood_preservation_1hop_manifold,
    sampled_neighborhood_preservation_1hop_nd, sampled_neighborhood_preservation_1hop_with_tree,
    sampled_neighborhood_preservation_2hop, sampled_neighborhood_preservation_2hop_manifold,
    sampled_neighborhood_preservation_2hop_nd, sampled_neighborhood_preservation_2hop_with_tree,
};
pub use proximity::{
    gabriel_graph_edges, gabriel_graph_edges_nd, gabriel_graph_edges_with_tree,
    gabriel_graph_preservation, gabriel_graph_preservation_manifold, gabriel_graph_preservation_nd,
    gabriel_graph_preservation_with_tree, gabriel_graph_property, gabriel_graph_property_manifold,
    gabriel_graph_property_nd, gabriel_graph_property_with_tree, node_resolution,
    node_resolution_manifold, node_resolution_nd, node_resolution_with_tree,
    relative_neighborhood_edges, relative_neighborhood_edges_nd,
    relative_neighborhood_edges_with_tree, relative_neighborhood_preservation,
    relative_neighborhood_preservation_manifold, relative_neighborhood_preservation_nd,
    relative_neighborhood_preservation_with_tree,
};
#[doc(hidden)]
pub use proximity::{
    gabriel_graph_edges_manifold, gabriel_graph_edges_nd_with_index,
    gabriel_graph_preservation_nd_with_index, gabriel_graph_property_nd_with_index,
    manifold_geodesic_diameter, manifold_geodesic_interpolate_point,
    manifold_point_distance_to_node, node_resolution_nd_with_index,
    relative_neighborhood_edges_manifold, relative_neighborhood_edges_nd_with_index,
    relative_neighborhood_preservation_nd_with_index,
};
pub use rta::{
    RtaEvaluation, landmark_stratified_rta, landmark_stratified_rta_detail,
    landmark_stratified_rta_detail_manifold, landmark_stratified_rta_detail_nd,
    landmark_stratified_rta_from_landmarks, landmark_stratified_rta_from_landmarks_f64,
    landmark_stratified_rta_from_landmarks_f64_manifold,
    landmark_stratified_rta_from_landmarks_f64_nd, landmark_stratified_rta_from_landmarks_manifold,
    landmark_stratified_rta_from_landmarks_nd, landmark_stratified_rta_manifold,
    landmark_stratified_rta_nd,
};
pub use stress::{
    BfsScratch, StressEvaluation, all_pairs_bfs, all_pairs_bfs_into, bfs_single_source,
    geodesic_distortion, sampled_scale_normalized_stress, sampled_scale_normalized_stress_detail,
    sampled_scale_normalized_stress_detail_manifold, sampled_scale_normalized_stress_detail_nd,
    sampled_scale_normalized_stress_manifold, sampled_scale_normalized_stress_nd,
    scale_normalized_stress, scale_normalized_stress_detail,
    scale_normalized_stress_detail_manifold, scale_normalized_stress_detail_nd,
    scale_normalized_stress_from_distances, scale_normalized_stress_from_distances_detail,
    scale_normalized_stress_from_distances_detail_manifold,
    scale_normalized_stress_from_distances_detail_nd,
    scale_normalized_stress_from_distances_manifold, scale_normalized_stress_from_distances_nd,
    scale_normalized_stress_from_pairs, scale_normalized_stress_from_pairs_manifold,
    scale_normalized_stress_from_pairs_nd, scale_normalized_stress_manifold,
    scale_normalized_stress_nd,
};

#[doc(hidden)]
#[inline]
pub fn validate_graph_layout<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    context: &'static str,
) -> CoreResult<()> {
    if graph.num_nodes() != layout.len() {
        return Err(CoreError::dimension_mismatch(
            graph.num_nodes(),
            layout.len(),
            context,
        ));
    }
    layout.validate_finite()?;
    Ok(())
}

#[doc(hidden)]
#[inline]
pub fn validate_graph_manifold_layout<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
    context: &'static str,
) -> CoreResult<()> {
    if graph.num_nodes() != layout.len() {
        return Err(CoreError::dimension_mismatch(
            graph.num_nodes(),
            layout.len(),
            context,
        ));
    }
    layout.validate_finite()?;
    Ok(())
}

#[doc(hidden)]
#[inline]
pub fn validate_tree_layout<S: Scalar>(
    tree: &StaticKdTree<2, f64>,
    layout: &Layout2d<S>,
    context: &'static str,
) -> CoreResult<()> {
    if tree.len() != layout.len() {
        return Err(CoreError::dimension_mismatch(
            layout.len(),
            tree.len(),
            context,
        ));
    }
    Ok(())
}

#[doc(hidden)]
pub fn build_kdtree_f64<S: Scalar>(layout: &Layout2d<S>) -> CoreResult<StaticKdTree<2, f64>> {
    layout.validate_finite()?;
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        points.push([xs[i].to_f64(), ys[i].to_f64()]);
    }
    StaticKdTree::<2, f64>::build(&points)
}

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
#[doc(hidden)]
pub fn extract_unique_edges<W>(graph: &CsrGraph<W>) -> Vec<(u32, u32)> {
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

/// Returns the crate version.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn metrics_geometry_crate_version_non_empty() {
        assert!(!crate_version().is_empty());
    }
}
