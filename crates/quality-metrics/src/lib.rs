//! # Quality Metrics for Graph Layouts
//!
//! This crate provides implementations of various metrics for evaluating the quality of
//! graph layouts in 2D Euclidean space. These metrics can be used to quantitatively assess
//! different aspects of a graph drawing, such as how well it preserves the graph's structure,
//! its aesthetic qualities, and its readability.
//!
//! ## Available Metrics
//!
//! The following metrics are available:
//!
//! - **Stress**: Measures how well the layout preserves pairwise distances between nodes
//! - **Ideal Edge Lengths**: Evaluates how well edge lengths in the drawing match their ideal lengths
//! - **Neighborhood Preservation**: Assesses how well the layout preserves local neighborhoods
//! - **Crossing Number**: Counts the number of edge crossings in the layout
//! - **Crossing Angle**: Measures the angles at which edges cross (larger angles are better)
//! - **Aspect Ratio**: Evaluates the balance between width and height of the drawing
//! - **Angular Resolution**: Measures the angles between adjacent edges (larger angles are better)
//! - **Node Resolution**: Assesses how well nodes are distributed in the drawing space
//! - **Gabriel Graph Property**: Evaluates how well the layout satisfies the Gabriel graph condition
//!
//! ## Usage
//!
//! ```
//! use petgraph::prelude::*;
//! use petgraph_algorithm_shortest_path::warshall_floyd;
//! use petgraph_drawing::DrawingEuclidean2d;
//! use petgraph_quality_metrics::{quality_metrics, QualityMetric};
//!
//! // Create a graph and its layout
//! let mut graph = Graph::<(), ()>::new();
//! let a = graph.add_node(());
//! let b = graph.add_node(());
//! let c = graph.add_node(());
//! graph.add_edge(a, b, ());
//! graph.add_edge(b, c, ());
//! graph.add_edge(c, a, ());
//!
//! // Create a drawing
//! let drawing = DrawingEuclidean2d::initial_placement(&graph);
//!
//! // Calculate all-pairs shortest path distances
//! let d = warshall_floyd(&graph, |_| 1.0);
//!
//! // Compute all quality metrics
//! let metrics = quality_metrics(&graph, &drawing, &d);
//!
//! // Print the metrics
//! for (metric, value) in metrics {
//!     println!("{}: {}", metric.name(), value);
//! }
//! ```

mod angular_resolution;
mod aspect_ratio;
mod edge_angle;
mod edge_crossings;
mod gabriel_graph_property;
mod ideal_edge_lengths;
mod neighborhood_preservation;
mod node_resolution;
mod stress;

use petgraph::visit::{IntoEdgeReferences, IntoNeighbors, IntoNodeIdentifiers, NodeIndexable};
use petgraph_algorithm_shortest_path::FullDistanceMatrix;
use petgraph_drawing::{DrawingEuclidean2d, DrawingIndex};

pub use angular_resolution::angular_resolution;
pub use aspect_ratio::aspect_ratio;
pub use edge_crossings::{
    crossing_angle, crossing_angle_with_crossing_edges, crossing_edges, crossing_edges_torus,
    crossing_number, crossing_number_with_crossing_edges, CrossingEdges,
};
pub use gabriel_graph_property::gabriel_graph_property;
pub use ideal_edge_lengths::ideal_edge_lengths;
pub use neighborhood_preservation::neighborhood_preservation;
pub use node_resolution::node_resolution;
pub use stress::stress;

/// Specifies whether a quality metric should be maximized or minimized.
///
/// Some metrics are better when their values are higher (e.g., neighborhood preservation),
/// while others are better when their values are lower (e.g., stress).
#[derive(Clone, Copy)]
pub enum Sense {
    /// The metric should be maximized (higher values are better).
    Maximize,
    /// The metric should be minimized (lower values are better).
    Minimize,
}

/// Represents the different types of quality metrics available for evaluating graph layouts.
///
/// Each metric evaluates a different aspect of a graph layout, such as its fidelity to the
/// graph structure, aesthetic qualities, or readability.
#[derive(Clone, Copy)]
pub enum QualityMetric {
    /// Measures how well the layout preserves pairwise distances between nodes.
    ///
    /// A lower stress value indicates that the Euclidean distances in the layout
    /// better match the graph-theoretical distances. This metric is calculated as
    /// the sum of squared relative differences between these distances.
    Stress,

    /// Evaluates how well edge lengths in the drawing match their ideal lengths.
    ///
    /// This metric measures the sum of squared relative differences between the actual
    /// edge lengths in the layout and the ideal lengths defined by the graph structure.
    /// Lower values indicate better preservation of edge length proportions.
    IdealEdgeLengths,

    /// Assesses how well the layout preserves local neighborhoods.
    ///
    /// This metric calculates the ratio of nodes that are both graph-theoretical
    /// neighbors and spatial neighbors in the layout. Higher values indicate better
    /// preservation of the graph's neighborhood structure.
    NeighborhoodPreservation,

    /// Counts the number of edge crossings in the layout.
    ///
    /// Edge crossings can reduce the readability of a graph drawing. A lower
    /// crossing number is generally preferred for clearer visualizations.
    CrossingNumber,

    /// Measures the angles at which edges cross.
    ///
    /// When edges must cross, it's preferable that they cross at angles close to 90 degrees
    /// for better readability. Higher values of this metric indicate more perpendicular crossings.
    CrossingAngle,

    /// Evaluates the balance between width and height of the drawing.
    ///
    /// This metric uses the ratio of the smaller to the larger eigenvalue of the
    /// covariance matrix of node positions. Values closer to 1 indicate a more
    /// balanced, circular layout rather than a highly elongated one.
    AspectRatio,

    /// Measures the angles between adjacent edges.
    ///
    /// Larger angles between edges incident to the same node improve readability.
    /// This metric sums up a function of these angles, with higher values indicating
    /// better angular resolution.
    AngularResolution,

    /// Assesses how well nodes are distributed in the drawing space.
    ///
    /// This metric evaluates whether nodes are too close to each other, which can
    /// hamper readability. Higher values indicate better spacing between nodes.
    NodeResolution,

    /// Evaluates how well the layout satisfies the Gabriel graph condition.
    ///
    /// A Gabriel graph has the property that for any edge, the disk with the edge as
    /// diameter contains no other nodes. This metric measures violations of this
    /// condition, with lower values indicating better adherence.
    GabrielGraphProperty,
}

impl QualityMetric {
    /// Returns a string representation of the metric name.
    ///
    /// This method provides a consistent, kebab-case string identifier for each metric,
    /// which can be used for display or logging purposes.
    ///
    /// # Returns
    ///
    /// A `String` containing the metric's name.
    pub fn name(&self) -> String {
        match self {
            QualityMetric::Stress => "stress".into(),
            QualityMetric::IdealEdgeLengths => "ideal-edge-lengths".into(),
            QualityMetric::NeighborhoodPreservation => "neighborhood-preservation".into(),
            QualityMetric::CrossingNumber => "crossing-number".into(),
            QualityMetric::CrossingAngle => "crossing-angle".into(),
            QualityMetric::AspectRatio => "aspect-ratio".into(),
            QualityMetric::AngularResolution => "angular-resolution".into(),
            QualityMetric::NodeResolution => "node-resolution".into(),
            QualityMetric::GabrielGraphProperty => "gabriel-graph-property".into(),
        }
    }

    /// Returns whether the metric should be maximized or minimized.
    ///
    /// This method indicates the optimization direction for each metric. For some metrics,
    /// higher values represent better layouts (Maximize), while for others, lower values
    /// are preferable (Minimize).
    ///
    /// # Returns
    ///
    /// A `Sense` enum value indicating whether the metric should be maximized or minimized.
    pub fn sense(&self) -> Sense {
        match self {
            QualityMetric::NeighborhoodPreservation => Sense::Maximize,
            QualityMetric::CrossingAngle => Sense::Maximize,
            QualityMetric::AspectRatio => Sense::Maximize,
            QualityMetric::AngularResolution => Sense::Maximize,
            QualityMetric::NodeResolution => Sense::Maximize,
            _ => Sense::Minimize,
        }
    }
}

/// Calculates all available quality metrics for a given graph layout.
///
/// This function computes a comprehensive set of metrics to evaluate different aspects
/// of a graph drawing, such as its fidelity to the graph structure, aesthetic qualities,
/// and readability.
///
/// # Parameters
///
/// * `graph`: The graph structure to evaluate
/// * `drawing`: The 2D Euclidean layout of the graph
/// * `d`: The full distance matrix containing shortest path distances between all node pairs
///
/// # Returns
///
/// A vector of tuples, each containing a `QualityMetric` enum value and its corresponding
/// computed value as an `f32`.
///
/// # Type Parameters
///
/// * `G`: A graph type that implements the required traits
pub fn quality_metrics<G>(
    graph: G,
    drawing: &DrawingEuclidean2d<G::NodeId, f32>,
    d: &FullDistanceMatrix<G::NodeId, f32>,
) -> Vec<(QualityMetric, f32)>
where
    G: IntoEdgeReferences + IntoNeighbors + IntoNodeIdentifiers + NodeIndexable,
    G::NodeId: DrawingIndex,
{
    quality_metrics_with_targets(
        graph,
        drawing,
        d,
        &[
            QualityMetric::Stress,
            QualityMetric::IdealEdgeLengths,
            QualityMetric::NeighborhoodPreservation,
            QualityMetric::CrossingNumber,
            QualityMetric::CrossingAngle,
            QualityMetric::AspectRatio,
            QualityMetric::AngularResolution,
            QualityMetric::NodeResolution,
            QualityMetric::GabrielGraphProperty,
        ],
    )
}

/// Calculates only the specified quality metrics for a given graph layout.
///
/// This function computes a selected subset of metrics to evaluate specific aspects
/// of a graph drawing, as specified by the `targets` parameter.
///
/// # Parameters
///
/// * `graph`: The graph structure to evaluate
/// * `drawing`: The 2D Euclidean layout of the graph
/// * `d`: The full distance matrix containing shortest path distances between all node pairs
/// * `targets`: A slice of `QualityMetric` enum values specifying which metrics to compute
///
/// # Returns
///
/// A vector of tuples, each containing a `QualityMetric` enum value and its corresponding
/// computed value as an `f32`. Only the metrics specified in `targets` are included.
///
/// # Type Parameters
///
/// * `G`: A graph type that implements the required traits
pub fn quality_metrics_with_targets<G>(
    graph: G,
    drawing: &DrawingEuclidean2d<G::NodeId, f32>,
    d: &FullDistanceMatrix<G::NodeId, f32>,
    targets: &[QualityMetric],
) -> Vec<(QualityMetric, f32)>
where
    G: IntoEdgeReferences + IntoNeighbors + IntoNodeIdentifiers + NodeIndexable,
    G::NodeId: DrawingIndex,
{
    let crossing_edges = crossing_edges(graph, drawing);
    targets
        .iter()
        .map(|&t| {
            let v = match t {
                QualityMetric::Stress => stress(drawing, d),
                QualityMetric::IdealEdgeLengths => ideal_edge_lengths(graph, drawing, d),
                QualityMetric::NeighborhoodPreservation => {
                    neighborhood_preservation(graph, drawing)
                }
                QualityMetric::CrossingNumber => {
                    crossing_number_with_crossing_edges(&crossing_edges)
                }
                QualityMetric::CrossingAngle => crossing_angle_with_crossing_edges(&crossing_edges),
                QualityMetric::AspectRatio => aspect_ratio(drawing),
                QualityMetric::AngularResolution => angular_resolution(graph, drawing),
                QualityMetric::NodeResolution => node_resolution(drawing),
                QualityMetric::GabrielGraphProperty => gabriel_graph_property(graph, drawing),
            };
            (t, v)
        })
        .collect::<Vec<_>>()
}
