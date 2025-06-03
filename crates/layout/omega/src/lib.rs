//! Omega SGD layout algorithm for graphs using spectral coordinates.
//!
//! This crate provides the `Omega` struct which implements the `Sgd` trait from
//! `petgraph-layout-sgd`. The Omega algorithm differs from FullSgd and SparseSgd
//! in how it constructs node pairs for the SGD optimization process.
//!
//! The Omega algorithm:
//! 1. Computes the smallest d non-zero eigenvalues and eigenvectors of the graph Laplacian
//! 2. Creates d-dimensional coordinates by dividing eigenvectors by sqrt of eigenvalues  
//! 3. Adds edge-based node pairs using Euclidean distances from coordinates
//! 4. Adds k random node pairs per node using Euclidean distances (avoiding duplicates)
//!
//! # Example
//!
//! ```rust
//! use petgraph::{Graph, graph::NodeIndex};
//! use petgraph_layout_omega::Omega;
//! use petgraph_layout_sgd::{Scheduler, SchedulerExponential, Sgd};
//! use petgraph_drawing::DrawingEuclidean2d;
//! use rand::thread_rng;
//!
//! // Create a graph
//! let mut graph = Graph::new_undirected();
//! let a = graph.add_node(());
//! let b = graph.add_node(());
//! let c = graph.add_node(());
//! graph.add_edge(a, b, ());
//! graph.add_edge(b, c, ());
//! graph.add_edge(c, a, ());
//!
//! // Create Omega instance
//! let mut rng = thread_rng();
//! let d = 2; // Number of spectral dimensions
//! let k = 5; // Number of random pairs per node
//! let mut omega = Omega::new(&graph, |_| 1.0, d, k, &mut rng);
//!
//! // Use with SGD framework
//! let mut drawing: DrawingEuclidean2d<NodeIndex, f32> = DrawingEuclidean2d::initial_placement(&graph);
//! let mut scheduler = omega.scheduler::<SchedulerExponential<f32>>(1000, 0.1);
//!
//! scheduler.run(&mut |eta| {
//!     omega.shuffle(&mut rng);
//!     omega.apply(&mut drawing, eta);
//! });
//! ```
//!
//! # Computational Complexity
//!
//! - Eigenvalue computation: O(d(|V| + |E|))
//! - Coordinate generation: O(d|V|)
//! - Edge-based pairs: O(|E|)
//! - Random pairs: O(k|V|)
//! - Total: O(d(|V| + |E|) + k|V|)

mod eigenvalue;
mod omega;

pub use eigenvalue::EigenSolver;
pub use omega::Omega;

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;
    use petgraph_layout_sgd::Sgd;
    use rand::thread_rng;

    #[test]
    fn test_omega_basic_functionality() {
        // Create a simple triangle graph
        let mut graph = Graph::new_undirected();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, a, ());

        // Create Omega instance
        let mut rng = thread_rng();
        let d = 2; // Number of spectral dimensions
        let k = 1; // Number of random pairs per node
        let omega = Omega::new(&graph, |_| 1.0f32, d, k, &mut rng);

        // Verify that node pairs were created
        assert!(
            !omega.node_pairs().is_empty(),
            "Omega should create node pairs"
        );

        // Should have at least the 3 edges from the triangle
        assert!(
            omega.node_pairs().len() >= 3,
            "Should have at least 3 node pairs from edges"
        );

        // All node pairs should have positive distances and weights
        for &(i, j, dij, dji, wij, wji) in omega.node_pairs() {
            assert!(i != j, "Node pairs should be between different nodes");
            assert!(dij > 0.0, "Distance ij should be positive");
            assert!(dji > 0.0, "Distance ji should be positive");
            assert!(wij > 0.0, "Weight ij should be positive");
            assert!(wji > 0.0, "Weight ji should be positive");
        }
    }

    #[test]
    fn test_eigenvalue_solver() {
        // Create a simple path graph
        let mut graph = Graph::new_undirected();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());

        let solver = EigenSolver::<f32>::default();
        let (eigenvalues, eigenvectors) = solver.compute_smallest_eigenvalues(&graph, 2);

        // Debug output
        println!("Found {} eigenvalues: {:?}", eigenvalues.len(), eigenvalues);
        for (i, eigenvalue) in eigenvalues.iter().enumerate() {
            println!("Eigenvalue {}: {}", i, eigenvalue);
        }

        // Should compute 2 eigenvalues and eigenvectors
        assert_eq!(eigenvalues.len(), 2, "Should compute 2 eigenvalues");
        assert_eq!(eigenvectors.len(), 2, "Should compute 2 eigenvectors");

        // Each eigenvector should have the same length as the number of nodes
        for eigenvector in &eigenvectors {
            assert_eq!(eigenvector.len(), 3, "Eigenvector should have 3 components");
        }

        // Eigenvalues should be positive (non-zero for connected graph)
        for &eigenvalue in &eigenvalues {
            assert!(eigenvalue > 0.0, "Eigenvalues should be positive");
        }
    }
}
