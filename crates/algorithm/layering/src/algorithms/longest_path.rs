use petgraph::graph::{IndexType, NodeIndex};
use petgraph::{Directed, EdgeDirection, Graph};
use std::collections::HashMap;

use crate::LayeringAlgorithm;

/// Performs a depth-first search to assign layer values to nodes in a directed graph.
///
/// This function recursively traverses the graph, assigning layer values to nodes
/// based on their maximum path length from a source node.
///
/// # Arguments
///
/// * `graph` - The directed graph to traverse
/// * `layers` - A map from node indices to their layer values
/// * `u` - The current node being visited
/// * `depth` - The current depth in the traversal
fn dfs_layer<N, E, Ix: IndexType>(
    graph: &Graph<N, E, Directed, Ix>,
    layers: &mut HashMap<NodeIndex<Ix>, usize>,
    u: NodeIndex<Ix>,
    depth: usize,
) {
    for v in graph.neighbors(u) {
        if let std::collections::hash_map::Entry::Vacant(e) = layers.entry(v) {
            e.insert(depth + 1);
        } else {
            let layer = layers.get_mut(&v).unwrap();
            if *layer <= depth {
                *layer = depth + 1
            }
        }
        dfs_layer(graph, layers, v, depth + 1);
    }
}

/// Implementation of the longest path algorithm for layer assignment.
///
/// This algorithm assigns layers to nodes in a directed acyclic graph (DAG) based on
/// the longest path from any source node (node with no incoming edges). It ensures
/// that for every edge (u,v), node v is placed in a layer greater than node u.
///
/// This is one of the simplest layer assignment algorithms for hierarchical graph layouts
/// and is commonly used in the Sugiyama framework.
pub struct LongestPath;

impl Default for LongestPath {
    fn default() -> Self {
        Self::new()
    }
}

impl LongestPath {
    /// Creates a new instance of the LongestPath algorithm.
    pub fn new() -> Self {
        LongestPath
    }

    /// Assigns layers to nodes using the longest path algorithm.
    ///
    /// This function assigns layers to nodes in a directed acyclic graph by:
    /// 1. Starting with nodes that have no incoming edges (sources)
    /// 2. Traversing the graph from these sources
    /// 3. Assigning each node to a layer based on the longest path from any source
    ///
    /// # Arguments
    ///
    /// * `graph` - A reference to a directed acyclic graph
    ///
    /// # Returns
    ///
    /// A HashMap mapping each node index to its assigned layer (starting from 0).
    ///
    /// # Examples
    ///
    /// ```
    /// use petgraph::Graph;
    /// use petgraph_algorithm_layering::algorithms::longest_path::LongestPath;
    /// use petgraph_algorithm_layering::LayeringAlgorithm;
    ///
    /// let mut graph = Graph::new();
    /// let a = graph.add_node("a");
    /// let b = graph.add_node("b");
    /// let c = graph.add_node("c");
    /// graph.add_edge(a, b, "");
    /// graph.add_edge(b, c, "");
    ///
    /// let longest_path = LongestPath::new();
    /// let layers = longest_path.assign_layers(&graph);
    ///
    /// assert_eq!(*layers.get(&a).unwrap(), 0);
    /// assert_eq!(*layers.get(&b).unwrap(), 1);
    /// assert_eq!(*layers.get(&c).unwrap(), 2);
    /// ```
    pub fn assign_layers<N, E, Ix: IndexType>(
        &self,
        graph: &Graph<N, E, Directed, Ix>,
    ) -> HashMap<NodeIndex<Ix>, usize> {
        let mut result = HashMap::new();

        // Start with source nodes (nodes with no incoming edges)
        for u in graph.externals(EdgeDirection::Incoming) {
            result.insert(u, 0);
            dfs_layer(graph, &mut result, u, 0);
        }

        result
    }
}

impl<N, E, Ix: IndexType> LayeringAlgorithm<N, E, Ix> for LongestPath {
    fn assign_layers(&self, graph: &Graph<N, E, Directed, Ix>) -> HashMap<NodeIndex<Ix>, usize> {
        self.assign_layers(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;

    #[test]
    fn test_longest_path() {
        let mut graph = Graph::<&str, &str>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");
        let e = graph.add_node("e");
        graph.add_edge(a, b, "");
        graph.add_edge(b, c, "");
        graph.add_edge(d, c, "");
        graph.add_edge(d, e, "");

        let algorithm = LongestPath::new();
        let layers = algorithm.assign_layers(&graph);

        assert_eq!(*layers.get(&a).unwrap(), 0);
        assert_eq!(*layers.get(&b).unwrap(), 1);
        assert_eq!(*layers.get(&c).unwrap(), 2);
        assert_eq!(*layers.get(&d).unwrap(), 0);
        assert_eq!(*layers.get(&e).unwrap(), 1);
    }

    #[test]
    fn test_longest_path_complex() {
        let mut graph = Graph::<&str, &str>::new();
        let a = graph.add_node("a"); // Layer 0
        let b = graph.add_node("b"); // Layer 1
        let c = graph.add_node("c"); // Layer 2
        let d = graph.add_node("d"); // Layer 0
        let e = graph.add_node("e"); // Layer 1
        let f = graph.add_node("f"); // Layer 2

        // Create a more complex graph with multiple paths
        graph.add_edge(a, b, "");
        graph.add_edge(b, c, "");
        graph.add_edge(d, e, "");
        graph.add_edge(e, f, "");
        graph.add_edge(a, e, ""); // Cross-connection

        let algorithm = LongestPath::new();
        let layers = algorithm.assign_layers(&graph);

        assert_eq!(*layers.get(&a).unwrap(), 0);
        assert_eq!(*layers.get(&b).unwrap(), 1);
        assert_eq!(*layers.get(&c).unwrap(), 2);
        assert_eq!(*layers.get(&d).unwrap(), 0);
        assert_eq!(*layers.get(&e).unwrap(), 1);
        assert_eq!(*layers.get(&f).unwrap(), 2);
    }
}
