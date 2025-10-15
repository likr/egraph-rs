//! # Omega SGD Layout Algorithm
//!
//! This crate provides the `Omega` struct for creating SGD instances from
//! precomputed spectral embeddings. The Omega algorithm generates node pairs
//! for SGD optimization based on distances in the spectral embedding space.
//!
//! ## Workflow
//!
//! 1. Use `RdMds` from `petgraph-linalg-rdmds` to compute spectral embeddings
//! 2. Use `Omega` to create an SGD instance from the embeddings
//! 3. Run the SGD optimization to refine the layout
//!
//! ## Example
//!
//! ```rust
//! use petgraph::Graph;
//! use petgraph_linalg_rdmds::RdMds;
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
//! // Step 1: Compute spectral embedding with RdMds
//! let mut rng = thread_rng();
//! let rdmds = RdMds::new().d(2).shift(1e-3f32);
//! let embedding = rdmds.embedding(&graph, |_| 1.0f32, &mut rng);
//!
//! // Step 2: Create SGD instance from embedding
//! let omega = Omega::new().k(5).min_dist(1e-3f32);
//! let mut sgd = omega.build(&graph, &embedding, &mut rng);
//!
//! // Step 3: Run SGD optimization
//! let mut drawing = DrawingEuclidean2d::initial_placement(&graph);
//! let mut scheduler = sgd.scheduler::<SchedulerExponential<f32>>(1000, 0.1);
//!
//! scheduler.run(&mut |eta| {
//!     sgd.shuffle(&mut rng);
//!     sgd.apply(&mut drawing, eta);
//! });
//! ```
//!
//! ## Computational Complexity
//!
//! - Node pair generation: O(|E| + k|V|)
//!
//! where k is the number of random pairs per node, |V| is the number of vertices,
//! and |E| is the number of edges.

mod omega;

pub use omega::Omega;

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;
    use petgraph_linalg_rdmds::RdMds;
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

        // Compute embedding with RdMds
        let mut rng = thread_rng();
        let mut rdmds = RdMds::new();
        let embedding = rdmds
            .d(2)
            .shift(1e-3f32)
            .embedding(&graph, |_| 1.0f32, &mut rng);

        // Create Omega instance and build SGD
        let mut omega = Omega::new();
        let sgd = omega
            .k(1)
            .min_dist(1e-3f32)
            .build(&graph, &embedding, &mut rng);

        // Verify that node pairs were created
        assert!(
            !sgd.node_pairs().is_empty(),
            "Omega should create node pairs"
        );

        // Should have at least the 3 edges from the triangle
        assert!(
            sgd.node_pairs().len() >= 3,
            "Should have at least 3 node pairs from edges"
        );

        // All node pairs should have positive distances and weights
        for &(i, j, dij, dji, wij, wji) in sgd.node_pairs() {
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

        // Compute embedding with RdMds
        let mut rng = thread_rng();
        let mut rdmds = RdMds::new();
        let embedding = rdmds
            .d(2)
            .shift(1e-3f32)
            .embedding(&graph, |_| 1.0f32, &mut rng);

        // Create Omega instance with a specific min_dist
        let mut omega = Omega::new();
        let sgd = omega
            .k(0)
            .min_dist(0.5f32)
            .build(&graph, &embedding, &mut rng);

        // Verify that all distances are at least min_dist
        for &(i, j, dij, dji, _wij, _wji) in sgd.node_pairs() {
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
}
