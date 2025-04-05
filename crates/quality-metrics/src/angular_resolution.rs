use crate::edge_angle::edge_angle;
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers};
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex, MetricEuclidean2d};

/// Calculates the angular resolution metric for a graph layout.
///
/// Angular resolution measures how well the angles between edges connected to the same node
/// are distributed. Higher values of this metric indicate better readability, as edges
/// are more evenly spaced around their common node.
///
/// This implementation calculates a sum of exponential functions of negative angles
/// between adjacent edges. Smaller angles contribute larger values to the sum,
/// making the metric increase as angular resolution worsens.
///
/// # Parameters
///
/// * `graph`: The graph structure to evaluate
/// * `drawing`: The 2D Euclidean layout of the graph
///
/// # Returns
///
/// An `f32` value representing the angular resolution metric. Lower values indicate
/// better angular resolution.
///
/// # Type Parameters
///
/// * `G`: A graph type that implements the required traits
pub fn angular_resolution<G>(graph: G, drawing: &DrawingEuclidean2d<G::NodeId, f32>) -> f32
where
    G: IntoNodeIdentifiers + IntoNeighbors,
    G::NodeId: DrawingIndex,
{
    let mut s = 0.;
    for u in graph.node_identifiers() {
        let MetricEuclidean2d(x0, y0) = drawing.position(u).unwrap();
        let neighbors = graph.neighbors(u).collect::<Vec<_>>();
        let n = neighbors.len();
        for i in 1..n {
            let v = neighbors[i];
            let MetricEuclidean2d(x1, y1) = drawing.position(v).unwrap();
            for neighbor in neighbors.iter().take(i) {
                let w = *neighbor;
                let MetricEuclidean2d(x2, y2) = drawing.position(w).unwrap();
                if let Some(angle) = edge_angle(x1 - x0, y1 - y0, x2 - x0, y2 - y0) {
                    s += (-angle).exp()
                }
            }
        }
    }
    s
}
