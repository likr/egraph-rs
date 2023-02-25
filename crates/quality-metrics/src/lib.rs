mod angular_resolution;
mod aspect_ratio;
mod edge_angle;
mod edge_crossings;
mod gabriel_graph_property;
mod ideal_edge_lengths;
mod neighborhood_preservation;
mod node_resolution;
mod stress;

use ndarray::prelude::*;
use petgraph::graph::{Graph, IndexType};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::Coordinates;

pub use angular_resolution::angular_resolution;
pub use aspect_ratio::aspect_ratio;
pub use edge_crossings::{
    crossing_angle, crossing_angle_with_crossing_edges, crossing_edges, crossing_number,
    crossing_number_with_crossing_edges,
};
pub use gabriel_graph_property::gabriel_graph_property;
pub use ideal_edge_lengths::ideal_edge_lengths;
pub use neighborhood_preservation::neighborhood_preservation;
pub use node_resolution::node_resolution;
pub use stress::stress;

#[derive(Clone, Copy)]
pub enum Sense {
    Maximize,
    Minimize,
}

#[derive(Clone, Copy)]
pub enum QualityMetric {
    Stress,
    IdealEdgeLengths,
    NeighborhoodPreservation,
    CrossingNumber,
    CrossingAngle,
    AspectRatio,
    AngularResolution,
    NodeResolution,
    GabrielGraphProperty,
}

impl QualityMetric {
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

pub fn quality_metrics<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    coordinates: &Coordinates<Ix>,
    d: &Array2<f32>,
) -> Vec<(QualityMetric, f32)> {
    quality_metrics_with_targets(
        graph,
        coordinates,
        d,
        &vec![
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

pub fn quality_metrics_with_targets<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    coordinates: &Coordinates<Ix>,
    d: &Array2<f32>,
    targets: &[QualityMetric],
) -> Vec<(QualityMetric, f32)> {
    let crossing_edges = crossing_edges(graph, coordinates);
    targets
        .iter()
        .map(|&t| {
            let v = match t {
                QualityMetric::Stress => stress(coordinates, d),
                QualityMetric::IdealEdgeLengths => ideal_edge_lengths(graph, coordinates, d),
                QualityMetric::NeighborhoodPreservation => {
                    neighborhood_preservation(graph, coordinates)
                }
                QualityMetric::CrossingNumber => {
                    crossing_number_with_crossing_edges(&crossing_edges)
                }
                QualityMetric::CrossingAngle => {
                    crossing_angle_with_crossing_edges(graph, coordinates, &crossing_edges)
                }
                QualityMetric::AspectRatio => aspect_ratio(coordinates),
                QualityMetric::AngularResolution => angular_resolution(graph, coordinates),
                QualityMetric::NodeResolution => node_resolution(graph, coordinates),
                QualityMetric::GabrielGraphProperty => gabriel_graph_property(graph, coordinates),
            };
            (t, v)
        })
        .collect::<Vec<_>>()
}
