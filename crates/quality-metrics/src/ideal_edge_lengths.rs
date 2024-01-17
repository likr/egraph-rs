use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeIdentifiers};
use petgraph_algorithm_shortest_path::{DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::{Drawing2D, DrawingIndex, Tuple2D};

pub fn ideal_edge_lengths<G>(
    graph: G,
    coordinates: &Drawing2D<G::NodeId, f32>,
    d: &FullDistanceMatrix<G::NodeId, f32>,
) -> f32
where
    G: IntoEdgeReferences + IntoNodeIdentifiers,
    G::NodeId: DrawingIndex,
{
    let mut s = 0.;
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
        let Tuple2D(x1, y1) = coordinates.position(u).unwrap();
        let Tuple2D(x2, y2) = coordinates.position(v).unwrap();
        let l = d.get(u, v).unwrap();
        s += (((x1 - x2).hypot(y1 - y2) - l) / l).powi(2);
    }
    s
}
