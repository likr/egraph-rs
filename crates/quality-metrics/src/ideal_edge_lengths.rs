use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph_algorithm_shortest_path::{DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::{Delta, Drawing, DrawingIndex, DrawingValue, Metric};

/// Evaluates how well edge lengths in a drawing match their ideal lengths.
///
/// This metric measures the sum of squared relative differences between the actual
/// edge lengths in the layout and the ideal lengths defined by the graph structure.
/// The ideal length of an edge is typically derived from the graph-theoretical
/// distance between its endpoints.
///
/// A lower value indicates better preservation of edge length proportions, meaning
/// the visual distances in the drawing better reflect the underlying graph structure.
///
/// # Parameters
///
/// * `graph`: The graph structure to evaluate
/// * `drawing`: The layout of the graph
/// * `d`: The full distance matrix containing shortest path distances between all node pairs
///
/// # Returns
///
/// A value of type `S` representing the ideal edge lengths metric. Lower values
/// indicate better adherence to ideal edge lengths.
///
/// # Type Parameters
///
/// * `G`: A graph type that implements the required traits
/// * `Diff`: A type for representing differences between metric values
/// * `D`: A drawing type
/// * `N`: Node ID type
/// * `M`: Metric type used in the drawing
/// * `S`: Numeric type for distance calculations
pub fn ideal_edge_lengths<G, Diff, D, N, M, S>(
    graph: G,
    drawing: &D,
    d: &FullDistanceMatrix<N, S>,
) -> S
where
    G: IntoEdgeReferences<NodeId = N>,
    D: Drawing<Item = M, Index = N>,
    Diff: Delta<S = S>,
    N: Copy + DrawingIndex,
    M: Copy + Metric<D = Diff>,
    S: DrawingValue,
{
    let mut s = S::zero();
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
        let delta = drawing.delta(drawing.index(u), drawing.index(v));
        let l = d.get(u, v).unwrap();
        s += ((delta.norm() - l) / l).powi(2);
    }
    s
}
