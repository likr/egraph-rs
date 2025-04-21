//! # Graph Layering Algorithms
//!
//! This crate provides algorithms for assigning layers to nodes in directed graphs,
//! which is a fundamental operation in hierarchical graph layouts.
//!
//! The main components of this crate are:
//!
//! * **Cycle Removal**: Functions to detect and remove cycles in directed graphs
//! * **Layer Assignment**: Algorithms to assign layers to nodes, ensuring proper hierarchical arrangement
//!
//! ## Usage Example
//!
//! ```
//! use petgraph::Graph;
//! use petgraph_algorithm_layering::{cycle, LongestPath, LayeringAlgorithm};
//!
//! // Create a directed graph with a cycle
//! let mut graph = Graph::new();
//! let n1 = graph.add_node("Node 1");
//! let n2 = graph.add_node("Node 2");
//! let n3 = graph.add_node("Node 3");
//! graph.add_edge(n1, n2, "Edge 1-2");
//! graph.add_edge(n2, n3, "Edge 2-3");
//! graph.add_edge(n3, n1, "Edge 3-1"); // Creates a cycle
//!
//! // Remove cycles to create a directed acyclic graph (DAG)
//! let mut acyclic_graph = graph.map(|_, n| n, |_, e| e);
//! cycle::remove_cycle(&mut acyclic_graph);
//!
//! // Assign layers to nodes using the longest path algorithm
//! let longest_path = LongestPath::new();
//! let layers = longest_path.assign_layers(&acyclic_graph);
//!
//! // Each node is now assigned to a layer (0, 1, 2, etc.)
//! println!("Node 1 layer: {}", layers.get(&n1).unwrap());
//! println!("Node 2 layer: {}", layers.get(&n2).unwrap());
//! println!("Node 3 layer: {}", layers.get(&n3).unwrap());
//! ```

pub mod algorithms;
pub mod cycle;

use petgraph::graph::{IndexType, NodeIndex};
use petgraph::{Directed, Graph};
use std::collections::HashMap;

// Re-export commonly used types and functions
pub use algorithms::LongestPath;
pub use cycle::{cycle_edges, remove_cycle};

/// A trait for algorithms that assign layers to nodes in a directed graph.
///
/// This trait defines a common interface for different layering algorithms,
/// which are a key component in hierarchical graph layout methods like
/// the Sugiyama framework.
///
/// Implementations of this trait should assign layer values (starting from 0)
/// to nodes in the graph, typically ensuring that if there's an edge (u,v),
/// then layer(v) > layer(u).
pub trait LayeringAlgorithm<N, E, Ix: IndexType> {
    /// Assigns layers to nodes in the given directed graph.
    ///
    /// # Arguments
    ///
    /// * `graph` - A reference to a directed graph
    ///
    /// # Returns
    ///
    /// A HashMap mapping each node index to its assigned layer (starting from 0).
    fn assign_layers(&self, graph: &Graph<N, E, Directed, Ix>) -> HashMap<NodeIndex<Ix>, usize>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::LongestPath;
    use petgraph::Graph;

    #[test]
    fn test_layering_algorithm_trait() {
        // Create a simple directed graph
        let mut graph = Graph::<(), ()>::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());

        // Use the LongestPath algorithm through the trait
        let algorithm: Box<dyn LayeringAlgorithm<(), (), _>> = Box::new(LongestPath::new());
        let layers = algorithm.assign_layers(&graph);

        // Verify the layer assignments
        assert_eq!(*layers.get(&a).unwrap(), 0);
        assert_eq!(*layers.get(&b).unwrap(), 1);
        assert_eq!(*layers.get(&c).unwrap(), 2);
    }

    #[test]
    fn test_cycle_removal_integration() {
        // Create a directed graph with a cycle
        let mut graph = Graph::<(), ()>::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, a, ()); // Creates a cycle

        // Clone the graph and remove cycles
        let mut acyclic_graph = graph.map(|_, n| n, |_, e| e);
        cycle::remove_cycle(&mut acyclic_graph);

        // Assign layers using the longest path algorithm
        let algorithm = LongestPath::new();
        let layers = algorithm.assign_layers(&acyclic_graph);

        // Verify all nodes have layer assignments
        assert!(layers.contains_key(&a));
        assert!(layers.contains_key(&b));
        assert!(layers.contains_key(&c));

        // Verify the layer relationships after cycle removal
        // Each node should have a unique layer
        // Verify each node has been assigned a layer
        assert!(layers.get(&a).is_some());
        assert!(layers.get(&b).is_some());
        assert!(layers.get(&c).is_some());

        // Check that the remaining edges respect the layer ordering
        for edge in acyclic_graph.edge_indices() {
            let (source, target) = acyclic_graph.edge_endpoints(edge).unwrap();
            let source_layer = *layers.get(&source).unwrap();
            let target_layer = *layers.get(&target).unwrap();
            assert!(
                source_layer < target_layer,
                "Edge ({:?},{:?}) violates layering: {} >= {}",
                source,
                target,
                source_layer,
                target_layer
            );
        }
    }
}
