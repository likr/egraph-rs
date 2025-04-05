use crate::Sgd;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers};
use petgraph_algorithm_shortest_path::{all_sources_dijkstra, DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::{DrawingIndex, DrawingValue};

/// Full Stochastic Gradient Descent (SGD) implementation for graph layout.
///
/// This implementation computes the shortest-path distances between all pairs of nodes
/// in the graph and optimizes the layout to match these distances geometrically.
/// It considers all possible node pairs during the optimization process, which
/// makes it accurate but potentially slow for large graphs.
pub struct FullSgd<S> {
    /// List of node pairs to be considered during layout optimization.
    /// Each tuple contains (i, j, distance_ij, distance_ji, weight_ij, weight_ji)
    node_pairs: Vec<(usize, usize, S, S, S, S)>,
}

impl<S> FullSgd<S> {
    /// Creates a new FullSgd instance from a graph.
    ///
    /// This constructor computes the all-pairs shortest path distances for the input graph
    /// and uses them as target distances for the layout optimization.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    ///
    /// # Returns
    /// A new FullSgd instance configured with all node pairs
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

    /// Creates a new FullSgd instance from a pre-computed distance matrix.
    ///
    /// This constructor is useful when you already have a distance matrix computed
    /// or want to use custom distances rather than shortest-path distances.
    ///
    /// # Parameters
    /// * `d` - A full distance matrix containing distances between all node pairs
    ///
    /// # Returns
    /// A new FullSgd instance configured with all node pairs
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
                node_pairs.push((i, j, dij, dij, wij, wij));
            }
        }
        FullSgd { node_pairs }
    }
}

/// Implementation of the Sgd trait for FullSgd
///
/// This provides the core SGD functionality for the full graph layout algorithm,
/// allowing it to work with the common SGD framework.
impl<S> Sgd<S> for FullSgd<S> {
    /// Returns a reference to the node pairs data structure.
    ///
    /// This implementation uses a full complement of node pairs where each tuple
    /// represents a pair of nodes with their associated target distances and weights.
    fn node_pairs(&self) -> &Vec<(usize, usize, S, S, S, S)> {
        &self.node_pairs
    }

    /// Returns a mutable reference to the node pairs data structure.
    ///
    /// This allows the algorithm to modify the node pairs during execution,
    /// such as updating distances or weights based on current layout.
    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, S, S, S, S)> {
        &mut self.node_pairs
    }
}
