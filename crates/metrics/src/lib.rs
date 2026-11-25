//! Quality metrics for graph layouts built on `egraph-core` and `egraph-spatial`.
//!
//! `egraph-metrics` provides single-threaded, pure-Rust layout evaluation primitives:
//! - [`stress`]: Reusable `VisitStamp`-backed BFS shortest paths, closed-form
//!   Scale-Normalized Stress (`SNS`), and $O(kN)$ pivot-sampled `SNS`.
//! - [`crossings`]: Exact sweep-line crossing counts/angles, torus crossings, and
//!   deterministic sampled crossing estimation.
//! - [`neighborhood`]: `StaticKdTree`-accelerated 1-hop and 2-hop neighborhood preservation.
//! - [`proximity`]: Gabriel graph property, hinge node resolution, and shape-based
//!   `dRNG` / `dGG` preservation metrics.
//! - [`geometry`]: Aspect ratio, angular resolution, edge angles, ideal edge lengths,
//!   and PRISM distortion ($\sigma_{\mathrm{dist}}, \sigma_{\mathrm{disp}}$).
//! - [`community`]: Spatial community coherence (`adjusted_patch_nig`,
//!   `adjusted_patch_nig_multiscale`).
//! - [`cluster_area`]: Grid-based Cluster Area Faithfulness (`G-CAF`) combining
//!   resolution, adjusted spatial purity, area proportion, and 8-connected coherence.
//! - [`cluster_baselines`] *(feature `eval-baselines`)*: External/internal cluster
//!   evaluation, visual separation, Sprawlter-NN, and Chen (2020) NOR baseline metrics.
//! - [`rta`]: $O(N)$-RAM streaming Landmark-Stratified Random Triplet Accuracy
//!   (`landmark_stratified_rta`).
//! - [`dispatcher`]: Selective metric evaluation (`evaluate_metrics`,
//!   `evaluate_metrics_with_communities`, `evaluate_metrics_with_community_weights`)
//!   executing only explicitly requested targets.

pub mod cluster_area;
#[cfg(feature = "eval-baselines")]
pub mod cluster_baselines;
pub mod community;
pub mod crossings;
pub mod dispatcher;
pub mod geometry;
pub mod neighborhood;
pub mod proximity;
pub mod rta;
pub mod stress;

use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar};
use egraph_spatial::StaticKdTree;

pub use cluster_area::{
    ClusterAreaConfig, ClusterAreaDetail, ClusterAreaReport, cluster_area_faithfulness,
    cluster_area_faithfulness_detail, cluster_area_faithfulness_detail_with_weights,
    cluster_area_faithfulness_with_weights,
};
#[cfg(feature = "eval-baselines")]
pub use cluster_baselines::*;
pub use community::{adjusted_patch_nig, adjusted_patch_nig_multiscale};
pub use crossings::{
    CrossingPair, CrossingSummary, TorusCrossing, crossing_angle, crossing_angle_from_pairs,
    crossing_edges, crossing_edges_torus, crossing_edges_torus_from_edges, crossing_number,
    crossing_number_from_pairs, crossing_summary, crosslessness, crosslessness_from_count,
    max_crossing_pairs, sampled_crossing_number, sampled_crossing_number_with_rng,
    sampled_crosslessness,
};
#[cfg(feature = "eval-baselines")]
pub use dispatcher::ClusterBaselineConfig;
pub use dispatcher::{
    MetricConfig, MetricReport, MetricTarget, evaluate_metrics, evaluate_metrics_with_communities,
    evaluate_metrics_with_community_weights,
};
pub use geometry::{
    LayoutDegeneracy, MinimumAngleSummary, PrismDistortion, angular_resolution, aspect_ratio,
    edge_angle, ideal_edge_lengths, ideal_edge_lengths_from_distances,
    ideal_edge_lengths_with_unit, layout_degeneracy, minimum_angle, minimum_angle_summary,
    prism_distortion, prism_distortion_with_edges,
};
pub use neighborhood::{
    neighborhood_preservation_1hop, neighborhood_preservation_1hop_with_tree,
    neighborhood_preservation_2hop, neighborhood_preservation_2hop_with_tree,
    sampled_neighborhood_preservation_1hop, sampled_neighborhood_preservation_1hop_with_tree,
    sampled_neighborhood_preservation_2hop, sampled_neighborhood_preservation_2hop_with_tree,
};
pub use proximity::{
    gabriel_graph_edges, gabriel_graph_edges_with_tree, gabriel_graph_preservation,
    gabriel_graph_preservation_with_tree, gabriel_graph_property, gabriel_graph_property_with_tree,
    node_resolution, node_resolution_with_tree, relative_neighborhood_edges,
    relative_neighborhood_edges_with_tree, relative_neighborhood_preservation,
    relative_neighborhood_preservation_with_tree,
};
pub use rta::{
    RtaEvaluation, landmark_stratified_rta, landmark_stratified_rta_detail,
    landmark_stratified_rta_from_landmarks,
};
pub use stress::{
    BfsScratch, StressEvaluation, all_pairs_bfs, all_pairs_bfs_into, bfs_single_source,
    sampled_scale_normalized_stress, sampled_scale_normalized_stress_detail,
    scale_normalized_stress, scale_normalized_stress_detail,
    scale_normalized_stress_from_distances, scale_normalized_stress_from_distances_detail,
    scale_normalized_stress_from_pairs,
};

#[inline]
pub(crate) fn validate_graph_layout<W, S: Scalar>(
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

#[inline]
pub(crate) fn validate_tree_layout<S: Scalar>(
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

pub(crate) fn build_kdtree_f64<S: Scalar>(
    layout: &Layout2d<S>,
) -> CoreResult<StaticKdTree<2, f64>> {
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
