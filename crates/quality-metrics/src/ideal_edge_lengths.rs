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
        let Some(l) = d.get(u, v) else {
            continue;
        };
        if !l.is_finite() || l <= S::zero() {
            continue;
        }
        let delta = drawing.delta(drawing.index(u), drawing.index(v));
        let norm = delta.norm();
        if !norm.is_finite() {
            continue;
        }
        s += ((norm - l) / l).powi(2);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use petgraph::graph::{NodeIndex, UnGraph};
    use petgraph_drawing::DrawingEuclidean2d;

    #[test]
    fn test_ideal_edge_lengths_skips_zero_and_nonfinite_target_distances() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        g.add_edge(n0, n1, ());
        let d = DrawingEuclidean2d::<NodeIndex, f64>::initial_placement(&g);

        let dm_zero = FullDistanceMatrix::from_array(&g, Array2::zeros((2, 2)));
        assert_eq!(ideal_edge_lengths(&g, &d, &dm_zero), 0.0);

        let dm_nan = FullDistanceMatrix::from_array(&g, Array2::from_elem((2, 2), f64::NAN));
        assert_eq!(ideal_edge_lengths(&g, &d, &dm_nan), 0.0);
    }
}
