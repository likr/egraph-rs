use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph_drawing::{
    Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue, MetricEuclidean2d,
};

/// Evaluates how well a graph layout adheres to the Gabriel graph property.
///
/// A Gabriel graph has the property that for any edge, the disk with the edge as
/// diameter contains no other nodes. This metric measures violations of this condition.
///
/// For each edge in the graph, this function computes the disk with that edge as its
/// diameter, and then calculates how much each node violates this property by being
/// inside the disk. The metric is the sum of squared violations.
///
/// # Parameters
///
/// * `graph`: The graph structure to evaluate
/// * `drawing`: The 2D Euclidean layout of the graph
///
/// # Returns
///
/// An `S` value representing the Gabriel graph property violation. Lower values
/// indicate better adherence to the Gabriel graph property (fewer violations).
///
/// # Type Parameters
///
/// * `G`: A graph type that implements the required traits
pub fn gabriel_graph_property<G, S>(graph: G, drawing: &DrawingEuclidean2d<G::NodeId, S>) -> S
where
    G: IntoEdgeReferences,
    G::NodeId: DrawingIndex,
    S: DrawingValue,
{
    let n = drawing.len();
    let mut s = S::zero();
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
        let MetricEuclidean2d(x1, y1) = *drawing.position(u).unwrap();
        let MetricEuclidean2d(x2, y2) = *drawing.position(v).unwrap();
        let cx = (x1 + x2) / (2.).into();
        let cy = (y1 + y2) / (2.).into();
        let r = (x1 - x2).hypot(y1 - y2) / (2.).into();
        for i in 0..n {
            s += (r - (drawing.raw_entry(i).0 - cx).hypot(drawing.raw_entry(i).1 - cy))
                .max(S::zero())
                .powi(2);
        }
    }
    s
}
