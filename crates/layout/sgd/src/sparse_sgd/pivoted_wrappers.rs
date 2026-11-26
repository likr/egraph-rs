use super::SparseSgd;
use crate::Sgd;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers, NodeIndexable};
use petgraph_algorithm_shortest_path::PivotedDistanceMatrix;
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_linalg_kernel::Distance;
use std::collections::HashSet;

impl SparseSgd {
    /// Creates a new SGD instance with pre-selected pivot nodes and a pre-computed distance matrix.
    ///
    /// This is the most low-level method, useful when you have already computed
    /// the distance matrix or want to provide a custom distance matrix.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `pivot` - A slice of node IDs to use as pivot nodes
    /// * `distance_matrix` - A pre-computed distance matrix from pivots to all nodes
    ///
    /// # Returns
    /// A new SGD instance configured with the specified pivot nodes and distances
    pub fn build_with_pivoted_neg_log_distance<G, F, K, S>(
        &self,
        graph: G,
        length: F,
        distance_matrix: &petgraph_linalg_kernel::PivotedNegLogDistance<G::NodeId, S, K>,
    ) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
        K: petgraph_linalg_kernel::PivotedKernel<S>,
    {
        self.try_build_with_pivoted_neg_log_distance(graph, length, distance_matrix)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_build_with_pivoted_neg_log_distance<G, F, K, S>(
        &self,
        graph: G,
        length: F,
        distance_matrix: &petgraph_linalg_kernel::PivotedNegLogDistance<G::NodeId, S, K>,
    ) -> Result<Sgd<S>, String>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
        K: petgraph_linalg_kernel::PivotedKernel<S>,
    {
        self.try_build_with_pivoted_kernel_distance(
            graph,
            length,
            distance_matrix.pivots(),
            distance_matrix.shape().0,
            distance_matrix,
        )
    }

    pub fn build_with_pivoted_neg_log_sim_distance<G, F, K, S>(
        &self,
        graph: G,
        length: F,
        distance_matrix: &petgraph_linalg_kernel::PivotedNegLogSimDistance<G::NodeId, S, K>,
    ) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
        K: petgraph_linalg_kernel::PivotedKernel<S>,
    {
        self.try_build_with_pivoted_neg_log_sim_distance(graph, length, distance_matrix)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_build_with_pivoted_neg_log_sim_distance<G, F, K, S>(
        &self,
        graph: G,
        length: F,
        distance_matrix: &petgraph_linalg_kernel::PivotedNegLogSimDistance<G::NodeId, S, K>,
    ) -> Result<Sgd<S>, String>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
        K: petgraph_linalg_kernel::PivotedKernel<S>,
    {
        self.try_build_with_pivoted_kernel_distance(
            graph,
            length,
            distance_matrix.pivots(),
            distance_matrix.shape().0,
            distance_matrix,
        )
    }

    fn try_build_with_pivoted_kernel_distance<G, F, S>(
        &self,
        graph: G,
        length: F,
        pivots: &[usize],
        dist_node_count: usize,
        distance_matrix: &dyn Distance<G::NodeId, S>,
    ) -> Result<Sgd<S>, String>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        let nodes = graph.node_identifiers().collect::<Vec<_>>();
        let n = nodes.len();
        if dist_node_count != n {
            return Err(format!(
                "distance matrix node count ({dist_node_count}) must match graph node count ({n})"
            ));
        }
        let mut pivot_nodes = Vec::with_capacity(pivots.len());
        for &idx in pivots {
            let Some(&node) = nodes.get(idx) else {
                return Err(format!(
                    "pivot index {idx} is out of bounds for graph with {n} nodes"
                ));
            };
            pivot_nodes.push(node);
        }
        Ok(self.build_with_pivot_and_distance_matrix(graph, length, &pivot_nodes, distance_matrix))
    }

    pub fn build_with_pivoted_distance_matrix<G, F, S>(
        &self,
        graph: G,
        length: F,
        distance_matrix: &PivotedDistanceMatrix<G::NodeId, S>,
    ) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        self.try_build_with_pivoted_distance_matrix(graph, length, distance_matrix)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_build_with_pivoted_distance_matrix<G, F, S>(
        &self,
        graph: G,
        length: F,
        distance_matrix: &PivotedDistanceMatrix<G::NodeId, S>,
    ) -> Result<Sgd<S>, String>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        let nodes = graph.node_identifiers().collect::<Vec<_>>();
        let n = nodes.len();
        let dist_n = distance_matrix.shape().1;
        if dist_n != n {
            return Err(format!(
                "pivoted distance matrix node count ({dist_n}) must match graph node count ({n})"
            ));
        }
        let node_set: HashSet<G::NodeId> = nodes.into_iter().collect();
        for &p in distance_matrix.pivots() {
            if !node_set.contains(&p) {
                return Err("pivoted distance matrix contains pivot not in graph".to_string());
            }
        }
        Ok(self.build_with_pivot_and_distance_matrix(
            graph,
            length,
            distance_matrix.pivots(),
            distance_matrix,
        ))
    }
}
