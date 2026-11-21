use petgraph::visit::{EdgeIndexable, EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeIndexable};
use std::collections::HashSet;

/// A calculator for weighted edge lengths based on node degrees and common neighbors.
///
/// This struct implements the weighted edge length calculation algorithm that computes
/// edge weights based on the formula: `degree(u) + degree(v) - 2 * common_neighbors`
/// where `u` and `v` are the endpoints of an edge.
///
/// The algorithm pre-computes neighbor sets for all nodes and edge endpoints
/// to enable efficient calculation of common neighbors for any edge.
pub struct WeightedEdgeLength {
    neighbors: Vec<HashSet<usize>>,
    edge_endpoints: Vec<Option<(usize, usize)>>,
}

impl WeightedEdgeLength {
    /// Creates a new `WeightedEdgeLength` calculator for the given graph.
    ///
    /// This constructor pre-computes neighbor sets for all nodes and edge endpoints,
    /// enabling efficient edge weight calculations later.
    ///
    /// # Arguments
    /// * `graph` - The graph to analyze
    ///
    /// # Returns
    /// A new `WeightedEdgeLength` instance ready for edge weight calculations
    pub fn new<G>(graph: G) -> Self
    where
        G: IntoNodeIdentifiers + IntoEdges + NodeIndexable + EdgeIndexable,
        G::NodeId: Copy,
    {
        let node_count = graph.node_bound();
        let edge_bound = graph.edge_bound();
        let mut neighbors = vec![HashSet::new(); node_count];
        let mut edge_endpoints = vec![None; edge_bound];

        // Index endpoints by edge ID so self-loops and sparse edge indices stay aligned.
        for node_id in graph.node_identifiers() {
            let node_index = NodeIndexable::to_index(&graph, node_id);
            for edge in graph.edges(node_id) {
                let target_index = NodeIndexable::to_index(&graph, edge.target());
                // Exclude self-loops from neighbor sets so they do not inflate common neighbors
                // or collapse distinct-endpoint weights to zero.
                if node_index != target_index {
                    neighbors[node_index].insert(target_index);
                    // For undirected behavior, also add the reverse edge
                    neighbors[target_index].insert(node_index);
                }

                let edge_index = EdgeIndexable::to_index(&graph, edge.id());
                let source_index = NodeIndexable::to_index(&graph, edge.source());
                if edge_index < edge_endpoints.len() {
                    let pair = if source_index <= target_index {
                        (source_index, target_index)
                    } else {
                        (target_index, source_index)
                    };
                    edge_endpoints[edge_index] = Some(pair);
                }
            }
        }

        Self {
            neighbors,
            edge_endpoints,
        }
    }

    /// Calculates the weighted length for an edge by its index, returning `None`
    /// if `edge_index` is out of bounds or does not correspond to an active edge.
    pub fn try_edge_weight(&self, edge_index: usize) -> Option<usize> {
        let (u, v) = (*self.edge_endpoints.get(edge_index)?)?;
        if u == v {
            return Some(0);
        }

        // Ensure u has fewer or equal neighbors than v for efficiency
        let (u, v) = if self.neighbors[u].len() > self.neighbors[v].len() {
            (v, u)
        } else {
            (u, v)
        };

        // Count common neighbors
        let mut common_neighbors = 0;
        for &neighbor in &self.neighbors[u] {
            if self.neighbors[v].contains(&neighbor) {
                common_neighbors += 1;
            }
        }

        Some(self.neighbors[u].len() + self.neighbors[v].len() - 2 * common_neighbors)
    }

    /// Calculates the weighted length for an edge by its index.
    ///
    /// The weight is calculated using the formula:
    /// `degree(u) + degree(v) - 2 * common_neighbors`
    ///
    /// where u and v are the node indices of the edge endpoints.
    ///
    /// # Arguments
    /// * `edge_index` - Index of the edge
    ///
    /// # Returns
    /// The calculated edge weight
    ///
    /// # Panics
    /// Panics if `edge_index` is out of bounds or does not correspond to an active edge.
    pub fn edge_weight(&self, edge_index: usize) -> usize {
        self.try_edge_weight(edge_index)
            .unwrap_or_else(|| panic!("Edge index {} out of bounds or absent", edge_index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;

    #[test]
    fn test_simple_graph() {
        let mut graph = Graph::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, ());
        graph.add_edge(n1, n2, ());

        let weighted_length = WeightedEdgeLength::new(&graph);

        // Edge 0: between n0 and n1
        // degree(n0) = 1, degree(n1) = 2, common_neighbors = 0
        // weight = 1 + 2 - 2 * 0 = 3
        assert_eq!(weighted_length.edge_weight(0), 3);

        // Edge 1: between n1 and n2
        // degree(n1) = 2, degree(n2) = 1, common_neighbors = 0
        // weight = 2 + 1 - 2 * 0 = 3
        assert_eq!(weighted_length.edge_weight(1), 3);
    }

    #[test]
    fn test_triangle_graph() {
        let mut graph = Graph::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, ());
        graph.add_edge(n1, n2, ());
        graph.add_edge(n0, n2, ()); // Note: edge order matters for indexing

        let weighted_length = WeightedEdgeLength::new(&graph);

        // All edges in triangle
        // degree(each node) = 2, common_neighbors = 1 for each edge
        // weight = 2 + 2 - 2 * 1 = 2
        assert_eq!(weighted_length.edge_weight(0), 2); // n0-n1
        assert_eq!(weighted_length.edge_weight(1), 2); // n1-n2
        assert_eq!(weighted_length.edge_weight(2), 2); // n0-n2
        assert_eq!(weighted_length.try_edge_weight(99), None);
    }

    #[test]
    fn test_self_loop_and_oob() {
        let mut graph = Graph::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        let e_loop = graph.add_edge(n0, n0, ());
        let e01 = graph.add_edge(n0, n1, ());
        let e12 = graph.add_edge(n1, n2, ());

        let weighted_length = WeightedEdgeLength::new(&graph);

        assert_eq!(weighted_length.try_edge_weight(e_loop.index()), Some(0));
        assert_eq!(weighted_length.try_edge_weight(e01.index()), Some(3));
        assert_eq!(weighted_length.try_edge_weight(e12.index()), Some(3));
        assert_eq!(weighted_length.try_edge_weight(99), None);
    }

    #[test]
    fn test_removed_edges_preserve_edge_id_indexing() {
        let mut graph = petgraph::stable_graph::StableUnGraph::<(), ()>::default();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());

        let e01 = graph.add_edge(n0, n1, ());
        let e12 = graph.add_edge(n1, n2, ());
        let e23 = graph.add_edge(n2, n3, ());
        graph.remove_edge(e12);

        let weighted_length = WeightedEdgeLength::new(&graph);

        assert_eq!(weighted_length.try_edge_weight(e01.index()), Some(2));
        assert_eq!(weighted_length.try_edge_weight(e12.index()), None);
        assert_eq!(weighted_length.try_edge_weight(e23.index()), Some(2));
        assert_eq!(weighted_length.try_edge_weight(99), None);
    }
}
