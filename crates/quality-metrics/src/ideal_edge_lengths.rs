use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph_algorithm_shortest_path::{DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::{Delta, Drawing, DrawingIndex, DrawingValue, Metric};

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
