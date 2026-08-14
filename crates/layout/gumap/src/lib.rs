//! # GUMAP (Graph UMAP) Layout Algorithm
//!
//! This crate provides the `GuMap` layout algorithm, which adapts Uniform Manifold Approximation
//! and Projection (UMAP) to graph drawing by constructing high-dimensional fuzzy simplicial sets
//! from graph distance matrices and optimizing cross-entropy loss with stochastic gradient descent (SGD).
//!
//! ## References
//!
//! - McInnes, L., Healy, J., & Melville, J. (2018).
//!   UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.

pub mod curve_fit;
pub mod fuzzy_simplicial_set;
mod gumap;
mod gumap_builder;

pub use gumap::GuMap;
pub use gumap_builder::GuMapBuilder;

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;
    use petgraph_algorithm_shortest_path::all_sources_dijkstra;
    use petgraph_drawing::DrawingEuclidean2d;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_gumap_builder_defaults() {
        let builder = GuMapBuilder::<f32>::new();
        let gumap = builder.build().expect("default builder should be valid");
        assert_eq!(gumap.k, 15);
        assert_eq!(gumap.min_dist, 0.1);
        assert_eq!(gumap.spread, 1.0);
        assert_eq!(gumap.n_epochs, 300);
        assert_eq!(gumap.negative_sample_rate, 5);
        assert_eq!(gumap.learning_rate, 1.0);
        assert!((gumap.a - 1.577).abs() < 0.05);
        assert!((gumap.b - 0.895).abs() < 0.05);
    }

    #[test]
    fn test_gumap_builder_validation() {
        assert!(GuMapBuilder::<f32>::new().k(0).build().is_err());
        assert!(GuMapBuilder::<f32>::new().min_dist(0.0).build().is_err());
        assert!(GuMapBuilder::<f32>::new().spread(0.0).build().is_err());
        assert!(GuMapBuilder::<f32>::new().n_epochs(0).build().is_err());
        assert!(GuMapBuilder::<f32>::new()
            .negative_sample_rate(0)
            .build()
            .is_err());
        assert!(GuMapBuilder::<f32>::new()
            .learning_rate(0.0)
            .build()
            .is_err());
        assert!(GuMapBuilder::<f32>::new().sigma_iters(0).build().is_err());
        assert!(GuMapBuilder::<f32>::new()
            .sigma_tolerance(0.0)
            .build()
            .is_err());
    }

    #[test]
    fn test_gumap_basic() {
        let mut graph = Graph::new_undirected();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, d, ());
        graph.add_edge(d, a, ());

        let mut coordinates = DrawingEuclidean2d::initial_placement(&graph);
        let distance_matrix = all_sources_dijkstra(&graph, |_| 1.0f32);
        let mut rng = StdRng::seed_from_u64(42);

        let gumap = GuMapBuilder::new()
            .k(3)
            .n_epochs(50)
            .build()
            .expect("valid configuration");

        gumap.run(&mut coordinates, &distance_matrix, &mut rng);

        for &u in &[a, b, c, d] {
            let x = coordinates.x(u).unwrap();
            let y = coordinates.y(u).unwrap();
            assert!(x.is_finite());
            assert!(y.is_finite());
        }
    }

    #[test]
    fn test_gumap_via_default_constructor() {
        let mut graph = Graph::new_undirected();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());

        let mut coordinates = DrawingEuclidean2d::initial_placement(&graph);
        let distance_matrix = all_sources_dijkstra(&graph, |_| 1.0f32);
        let mut rng = StdRng::seed_from_u64(42);

        let gumap = GuMap::<f32>::new();
        gumap.run(&mut coordinates, &distance_matrix, &mut rng);

        for &u in &[a, b, c] {
            let x = coordinates.x(u).unwrap();
            let y = coordinates.y(u).unwrap();
            assert!(x.is_finite());
            assert!(y.is_finite());
        }
    }
}
