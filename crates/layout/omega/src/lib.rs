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
//! use petgraph_layout_sgd::{Scheduler, SchedulerExponential};
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
//! // Create Omega instance using builder pattern
//! let mut rng = thread_rng();
//! let mut omega = Omega::new()
//!     .d(2)        // Number of spectral dimensions
//!     .k(5)        // Number of random pairs per node
//!     .min_dist(1e-3) // Minimum distance between node pairs
//!     .build(&graph, |_| 1.0, &mut rng);
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

pub use eigenvalue::{compute_smallest_eigenvalues, LaplacianStructure};
pub use omega::Omega;

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;
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

        // Create Omega instance with builder
        let mut rng = thread_rng();
        let omega = Omega::new()
            .d(2) // Number of spectral dimensions
            .k(1) // Number of random pairs per node
            .min_dist(1e-3f32) // Minimum distance between node pairs
            .build(&graph, |_| 1.0f32, &mut rng);

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
    fn test_min_dist_functionality() {
        // Create a graph with nodes that will be very close in spectral coordinates
        let mut graph = Graph::new_undirected();
        let a = graph.add_node(());
        let b = graph.add_node(());
        graph.add_edge(a, b, ());

        // Create Omega instance with a specific min_dist
        let mut rng = thread_rng();
        let omega = Omega::new()
            .d(1) // Single dimension for simplicity
            .k(0) // No random pairs to isolate edge pairs
            .min_dist(0.5f32) // Set a minimum distance
            .build(&graph, |_| 1.0f32, &mut rng);

        // Verify that all distances are at least min_dist
        for &(i, j, dij, dji, _wij, _wji) in omega.node_pairs() {
            assert!(i != j, "Node pairs should be between different nodes");
            assert!(
                dij >= 0.5f32,
                "Distance ij should be at least min_dist: {} >= {}",
                dij,
                0.5f32
            );
            assert!(
                dji >= 0.5f32,
                "Distance ji should be at least min_dist: {} >= {}",
                dji,
                0.5f32
            );
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

        // Create LaplacianStructure
        let laplacian = LaplacianStructure::new(&graph, |_| 1.0f32);
        let mut rng = thread_rng();
        let (eigenvalues, eigenvectors) =
            compute_smallest_eigenvalues(&laplacian, 2, 1000, 100, 1e-4f32, 1e-4f32, &mut rng);

        // Debug output
        println!("Found {} eigenvalues: {:?}", eigenvalues.len(), eigenvalues);
        for i in 0..eigenvalues.len() {
            println!("Eigenvalue {}: {}", i, eigenvalues[i]);
        }

        // Should compute n_target + 1 eigenvalues and eigenvectors (2 requested = 3 total)
        assert_eq!(eigenvalues.len(), 3, "Should compute 3 eigenvalues");
        assert_eq!(eigenvectors.ncols(), 3, "Should compute 3 eigenvectors");

        // Each eigenvector should have the same length as the number of nodes
        for i in 0..eigenvectors.ncols() {
            let eigenvector = eigenvectors.column(i);
            assert_eq!(eigenvector.len(), 3, "Eigenvector should have 3 components");
        }

        // First eigenvalue should be zero (constant eigenvector), rest should be positive
        assert!(
            eigenvalues[0].abs() < 1e-6,
            "First eigenvalue should be approximately zero"
        );
        for i in 1..eigenvalues.len() {
            assert!(
                eigenvalues[i] > 0.0,
                "Non-zero eigenvalues should be positive"
            );
        }
    }
}
