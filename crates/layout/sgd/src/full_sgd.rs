use crate::Sgd;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers};
use petgraph_algorithm_shortest_path::{all_sources_dijkstra, DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::DrawingIndex;

pub struct FullSgd {
    node_pairs: Vec<(usize, usize, f32, f32)>,
}

impl FullSgd {
    pub fn new<G, F>(graph: G, length: F) -> FullSgd
    where
        G: IntoEdges + IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> f32,
    {
        let d = all_sources_dijkstra(graph, length);
        Self::new_with_distance_matrix(&d)
    }

    pub fn new_with_distance_matrix<N>(d: &FullDistanceMatrix<N, f32>) -> FullSgd
    where
        N: DrawingIndex,
    {
        let n = d.shape().0;
        let mut node_pairs = vec![];
        for j in 1..n {
            for i in 0..j {
                let dij = d.get_by_index(i, j);
                let wij = 1. / (dij * dij);
                node_pairs.push((i, j, dij, wij));
                node_pairs.push((j, i, dij, wij));
            }
        }
        FullSgd { node_pairs }
    }
}

impl Sgd for FullSgd {
    fn node_pairs(&self) -> &Vec<(usize, usize, f32, f32)> {
        &self.node_pairs
    }

    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, f32, f32)> {
        &mut self.node_pairs
    }
}
