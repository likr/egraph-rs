use crate::Sgd;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers};
use petgraph_algorithm_shortest_path::{all_sources_dijkstra, DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::{DrawingIndex, DrawingValue};

/// Builder for creating Full SGD instances.
///
/// This builder computes the shortest-path distances between all pairs of nodes
/// in the graph and creates an SGD instance that optimizes the layout to match
/// these distances geometrically. It considers all possible node pairs during
/// the optimization process, which makes it accurate but potentially slow for large graphs.
pub struct FullSgd;

impl Default for FullSgd {
    fn default() -> Self {
        Self::new()
    }
}

impl FullSgd {
    pub fn new() -> Self {
        Self
    }

    /// Creates a new SGD instance from a graph using all-pairs shortest paths.
    ///
    /// This method computes the all-pairs shortest path distances for the input graph
    /// and uses them as target distances for the layout optimization.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    ///
    /// # Returns
    /// A new SGD instance configured with all node pairs
    pub fn build<G, F, S>(&self, graph: G, length: F) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        let d = all_sources_dijkstra(graph, length);
        self.build_with_distance_matrix(&d)
    }

    /// Creates a new SGD instance from a pre-computed distance matrix.
    ///
    /// This method is useful when you already have a distance matrix computed
    /// or want to use custom distances rather than shortest-path distances.
    ///
    /// # Parameters
    /// * `d` - A full distance matrix containing distances between all node pairs
    ///
    /// # Returns
    /// A new SGD instance configured with all node pairs
    pub fn build_with_distance_matrix<N, S>(&self, d: &FullDistanceMatrix<N, S>) -> Sgd<S>
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
                node_pairs.push((i, j, dij, dij, wij, wij));
            }
        }
        Sgd::new(node_pairs)
    }
}
