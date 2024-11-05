use crate::Sgd;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers};
use petgraph_algorithm_shortest_path::{all_sources_dijkstra, DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::{DrawingIndex, DrawingValue};

pub struct FullSgd<S> {
    node_pairs: Vec<(usize, usize, S, S, S)>,
}

impl<S> FullSgd<S> {
    pub fn new<G, F>(graph: G, length: F) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        let d = all_sources_dijkstra(graph, length);
        Self::new_with_distance_matrix(&d)
    }

    pub fn new_with_distance_matrix<N>(d: &FullDistanceMatrix<N, S>) -> Self
    where
        N: DrawingIndex,
        S: DrawingValue,
    {
        let n = d.shape().0;
        let mut node_pairs = vec![];
        for j in 1..n {
            for i in 0..j {
                let dij = d.get_by_index(i, j);
                let wij = S::one() / (dij * dij);
                node_pairs.push((i, j, dij, wij, wij));
            }
        }
        FullSgd { node_pairs }
    }
}

impl<S> Sgd<S> for FullSgd<S> {
    fn node_pairs(&self) -> &Vec<(usize, usize, S, S, S)> {
        &self.node_pairs
    }

    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, S, S, S)> {
        &mut self.node_pairs
    }
}
