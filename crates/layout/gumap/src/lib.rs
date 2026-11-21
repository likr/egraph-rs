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

pub use fuzzy_simplicial_set::{
    compute_fuzzy_simplicial_set_from_csr, compute_fuzzy_simplicial_set_from_knn,
    try_compute_fuzzy_simplicial_set_from_knn, validate_csr_knn, SparseEdge,
};
pub use gumap::GuMap;
pub use gumap_builder::GuMapBuilder;

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;
    use petgraph::Graph;
    use petgraph_algorithm_shortest_path::all_sources_dijkstra;
    use petgraph_drawing::{Drawing, DrawingEuclidean2d};
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

    #[test]
    fn test_fuzzy_simplicial_set_from_knn_self_loop_dedup_unsorted_and_isolated() {
        // Row 0 has self-loop (0, 0.0), duplicates of node 1 with smaller distance 0.25 later,
        // and out-of-order neighbors; Row 1 has uniform distances; Row 2 has m == 1; Row 3 is isolated.
        let knn_messy: Vec<Vec<(usize, f64)>> = vec![
            vec![(0, 0.0), (2, 1.5), (1, 0.8), (1, 0.25)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 1.5)],
            vec![],
        ];
        let knn_clean: Vec<Vec<(usize, f64)>> = vec![
            vec![(1, 0.25), (2, 1.5)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 1.5)],
            vec![],
        ];

        let edges_messy = compute_fuzzy_simplicial_set_from_knn(4, &knn_messy, 64, 1e-5);
        let edges_clean = compute_fuzzy_simplicial_set_from_knn(4, &knn_clean, 64, 1e-5);
        assert_eq!(edges_messy.len(), edges_clean.len());
        for (a, b) in edges_messy.iter().zip(edges_clean.iter()) {
            assert_eq!(a.source, b.source);
            assert_eq!(a.target, b.target);
            assert_eq!(a.weight.to_bits(), b.weight.to_bits());
            assert!(a.weight.is_finite());
        }

        // Uniform distance row (1) and singleton row (2) must produce weight 1.0 without NaN
        let edge_1_2 = edges_clean
            .iter()
            .find(|e| e.source == 1 && e.target == 2)
            .expect("edge (1, 2) must exist");
        assert!((edge_1_2.weight - 1.0).abs() < 1e-12);
        let edge_0_2 = edges_clean
            .iter()
            .find(|e| e.source == 0 && e.target == 2)
            .expect("edge (0, 2) must exist");
        assert!((edge_0_2.weight - 1.0).abs() < 1e-12);

        // CSR must match row-wise Vec<Vec<_>> bit-for-bit
        let indptr = vec![0usize, 4, 6, 7, 7];
        let indices = vec![0, 2, 1, 1, 0, 2, 0];
        let distances: Vec<f64> = vec![0.0, 1.5, 0.8, 0.25, 1.0, 1.0, 1.5];
        let edges_csr =
            compute_fuzzy_simplicial_set_from_csr(4, &indptr, &indices, &distances, 64, 1e-5)
                .unwrap();
        assert_eq!(edges_csr.len(), edges_clean.len());
        for (a, b) in edges_csr.iter().zip(edges_clean.iter()) {
            assert_eq!(a.source, b.source);
            assert_eq!(a.target, b.target);
            assert_eq!(a.weight.to_bits(), b.weight.to_bits());
        }
    }

    #[test]
    fn test_gumap_run_with_knn_matches_run_on_cycle() {
        let n = 6;
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
        for i in 0..n {
            graph.add_edge(nodes[i], nodes[(i + 1) % n], ());
        }
        let dm = all_sources_dijkstra(&graph, |_| 1.0_f64);

        // Construct exact k=4 neighbor lists matching all_sources_dijkstra on C_6
        let knn: Vec<Vec<(usize, f64)>> = (0..n)
            .map(|i| {
                vec![
                    ((i + 1) % n, 1.0),
                    ((i + n - 1) % n, 1.0),
                    ((i + 2) % n, 2.0),
                    ((i + n - 2) % n, 2.0),
                ]
            })
            .collect();
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::with_capacity(n * 4);
        let mut distances = Vec::with_capacity(n * 4);
        indptr.push(0);
        for row in &knn {
            for &(j, d) in row {
                indices.push(j);
                distances.push(d);
            }
            indptr.push(indices.len());
        }

        let gumap = GuMapBuilder::<f64>::new()
            .k(4)
            .n_epochs(40)
            .build()
            .unwrap();

        let mut d_run = DrawingEuclidean2d::initial_placement(&graph);
        let mut d_knn = DrawingEuclidean2d::initial_placement(&graph);
        let mut d_csr = DrawingEuclidean2d::initial_placement(&graph);
        // Copy exact initial placement to all three drawings
        for i in 0..n {
            let entry = *d_run.raw_entry(i);
            *d_knn.raw_entry_mut(i) = entry;
            *d_csr.raw_entry_mut(i) = entry;
        }

        gumap.run(&mut d_run, &dm, &mut StdRng::seed_from_u64(42));
        gumap.run_with_knn(&mut d_knn, &knn, &mut StdRng::seed_from_u64(42));
        gumap
            .try_run_with_csr_knn(
                &mut d_csr,
                &indptr,
                &indices,
                &distances,
                &mut StdRng::seed_from_u64(42),
            )
            .unwrap();

        for &u in &nodes {
            assert_eq!(d_run.x(u).unwrap().to_bits(), d_knn.x(u).unwrap().to_bits());
            assert_eq!(d_run.y(u).unwrap().to_bits(), d_knn.y(u).unwrap().to_bits());
            assert_eq!(d_run.x(u).unwrap().to_bits(), d_csr.x(u).unwrap().to_bits());
            assert_eq!(d_run.y(u).unwrap().to_bits(), d_csr.y(u).unwrap().to_bits());
        }
    }

    #[test]
    fn test_gumap_run_with_knn_two_cluster_and_edge_cases() {
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let n = 10;
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
        let mut knn = vec![Vec::new(); n];
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    knn[i].push((j, 0.2));
                    knn[5 + i].push((5 + j, 0.2));
                }
            }
            knn[i].push((5 + i, 5.0));
            knn[5 + i].push((i, 5.0));
        }

        let mut drawing = DrawingEuclidean2d::initial_placement(&graph);
        // Zero-initialized coordinates trigger random [-10, 10) init inside run_from_edges
        for i in 0..n {
            drawing.raw_entry_mut(i).0 = 0.0_f64;
            drawing.raw_entry_mut(i).1 = 0.0_f64;
        }
        let gumap = GuMapBuilder::<f64>::new()
            .k(5)
            .n_epochs(150)
            .build()
            .unwrap();
        gumap
            .try_run_with_knn(&mut drawing, &knn, &mut StdRng::seed_from_u64(42))
            .unwrap();

        let mut intra = 0.0_f64;
        let mut intra_cnt = 0usize;
        let mut inter = 0.0_f64;
        let mut inter_cnt = 0usize;
        for (i, &ni) in nodes.iter().enumerate() {
            let xi = drawing.x(ni).unwrap();
            let yi = drawing.y(ni).unwrap();
            assert!(xi.is_finite() && yi.is_finite());
            for (j, &nj) in nodes.iter().enumerate().skip(i + 1) {
                let xj = drawing.x(nj).unwrap();
                let yj = drawing.y(nj).unwrap();
                let dist = ((xi - xj).powi(2) + (yi - yj).powi(2)).sqrt();
                if (i < 5) == (j < 5) {
                    intra += dist;
                    intra_cnt += 1;
                } else {
                    inter += dist;
                    inter_cnt += 1;
                }
            }
        }
        assert!((intra / (intra_cnt as f64)) < (inter / (inter_cnt as f64)));

        // Edge cases: n = 0, n = 1, empty k-NN, and validation errors even for n < 2
        let g0 = UnGraph::<(), ()>::new_undirected();
        let mut d0 = DrawingEuclidean2d::<petgraph::graph::NodeIndex, f64>::initial_placement(&g0);
        let mut rng = StdRng::seed_from_u64(42);
        gumap.try_run_with_knn(&mut d0, &[], &mut rng).unwrap();
        gumap
            .try_run_with_csr_knn(&mut d0, &[0], &[], &[], &mut rng)
            .unwrap();
        assert!(gumap
            .try_run_with_knn(&mut d0, &[vec![]], &mut rng)
            .is_err());
        assert!(gumap
            .try_run_with_csr_knn(&mut d0, &[], &[], &[], &mut rng)
            .is_err());

        let mut g1 = UnGraph::<(), ()>::new_undirected();
        g1.add_node(());
        let mut d1 = DrawingEuclidean2d::<petgraph::graph::NodeIndex, f64>::initial_placement(&g1);
        gumap
            .try_run_with_knn(&mut d1, &[vec![]], &mut rng)
            .unwrap();
        assert!(gumap
            .try_run_with_knn(&mut d1, &[vec![(1, 0.5)]], &mut rng)
            .is_err());
        assert!(gumap
            .try_run_with_knn(&mut d1, &[vec![(0, -0.1)]], &mut rng)
            .is_err());
        assert!(gumap
            .try_run_with_knn(&mut d1, &[vec![(0, f64::NAN)]], &mut rng)
            .is_err());
        assert!(gumap
            .try_run_with_knn(&mut d1, &[vec![(0, f64::INFINITY)]], &mut rng)
            .is_err());
    }

    #[test]
    fn test_gumap_builder_rejects_nan_and_inf() {
        assert!(GuMapBuilder::<f64>::new()
            .min_dist(f64::NAN)
            .build()
            .is_err());
        assert!(GuMapBuilder::<f64>::new().spread(f64::NAN).build().is_err());
        assert!(GuMapBuilder::<f64>::new()
            .learning_rate(f64::NAN)
            .build()
            .is_err());
        assert!(GuMapBuilder::<f64>::new()
            .learning_rate(f64::INFINITY)
            .build()
            .is_err());
        assert!(GuMapBuilder::<f64>::new()
            .a(f64::NAN)
            .b(1.0)
            .build()
            .is_err());
        assert!(GuMapBuilder::<f64>::new()
            .a(1.0)
            .b(f64::INFINITY)
            .build()
            .is_err());
    }
}
