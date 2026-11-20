//! # tsNET, BH-tsNET, FIt-tsNET & L-tsNET Graph Layout Algorithms
//!
//! This crate provides the `TsNet`, `BhTsNet`, `FitTsNet`, and `LTsNet` layout algorithms:
//! - `TsNet`: Full-matrix t-SNE-based graph layout optimizing KL divergence, early compression, and entropy.
//! - `BhTsNet`: Accelerated O(N log N) Barnes-Hut tsNET combining Partial BFS and Quadtree spatial approximations.
//! - `FitTsNet`: Accelerated O(N log N) Fast Interpolation tsNET combining Partial BFS, 2D FFT interpolation for KL divergence, and Quadtree for entropy.
//! - `LTsNet`: Linear-time O(N) L-tsNET extending 2D FFT interpolation to both KL divergence and entropy gradients.
//!
//! ## References
//!
//! - Kruiger, J. F., Rauber, P. E., Martins, R. M., Kerren, A., Kobourov, S., & Telea, A. C. (2017).
//!   Graph Layouts by t-SNE. *Computer Graphics Forum*, 36(3), 283-294.
//! - Meidiana, A., Hong, S.-H., & Ma, K.-L. (2025).
//!   BH-tsNET, FIt-tsNET, L-tsNET: Fast tsNET Algorithms for Large Graph Drawing. *arXiv:2509.19785*.

pub mod bh_ts_net;
pub mod bh_ts_net_builder;
pub mod fft;
pub mod fit_interpolation;
pub mod fit_ts_net;
pub mod fit_ts_net_builder;
pub mod l_ts_net;
pub mod l_ts_net_builder;
pub mod partial_bfs;
pub mod quadtree;
mod ts_net;
mod ts_net_builder;

pub use bh_ts_net::BhTsNet;
pub use bh_ts_net_builder::BhTsNetBuilder;
pub use fit_ts_net::FitTsNet;
pub use fit_ts_net_builder::FitTsNetBuilder;
pub use l_ts_net::LTsNet;
pub use l_ts_net_builder::LTsNetBuilder;
pub use partial_bfs::{
    compute_knn_probabilities, compute_knn_probabilities_with_params,
    compute_partial_bfs_probabilities, SparseP,
};
pub use ts_net::TsNet;
pub use ts_net_builder::TsNetBuilder;

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;
    use petgraph_algorithm_shortest_path::all_sources_dijkstra;
    use petgraph_drawing::DrawingEuclidean2d;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_ts_net_builder_defaults() {
        let builder = TsNetBuilder::<f32>::new();
        let ts_net = builder.build().expect("default builder should be valid");
        assert_eq!(ts_net.perplexity, 40.0);
        assert_eq!(ts_net.n_epochs, 5000);
        assert_eq!(ts_net.initial_lr, 50.0);
        assert_eq!(ts_net.final_lr, 50.0);
        assert_eq!(ts_net.lr_switch, None);
        assert_eq!(ts_net.initial_momentum, 0.5);
        assert_eq!(ts_net.final_momentum, 0.5);
        assert_eq!(ts_net.momentum_switch, None);
        assert_eq!(ts_net.initial_l_kl, Some(1.0));
        assert_eq!(ts_net.final_l_kl, Some(1.0));
        assert_eq!(ts_net.l_kl_switch, None);
        assert_eq!(ts_net.initial_l_c, Some(1.2));
        assert_eq!(ts_net.final_l_c, Some(0.01));
        assert_eq!(ts_net.l_c_switch, None);
        assert_eq!(ts_net.initial_l_r, Some(0.0));
        assert_eq!(ts_net.final_l_r, Some(0.6));
        assert_eq!(ts_net.l_r_switch, None);
        assert_eq!(ts_net.r_eps, 0.05);
        assert_eq!(ts_net.sigma_iters, 50);
        assert_eq!(ts_net.autostop, 1e-7);
        assert_eq!(ts_net.window_size, 40);
    }

    #[test]
    fn test_ts_net_builder_new_star() {
        let builder = TsNetBuilder::<f32>::new_star();
        let ts_net = builder.build().expect("star builder should be valid");
        assert_eq!(ts_net.initial_l_c, Some(0.1));
        assert_eq!(ts_net.perplexity, 40.0);
        assert_eq!(ts_net.n_epochs, 5000);
        assert_eq!(ts_net.initial_l_r, Some(0.0));
        assert_eq!(ts_net.final_l_r, Some(0.6));
    }

    #[test]
    fn test_ts_net_builder_validation() {
        assert!(TsNetBuilder::<f32>::new().perplexity(-1.0).build().is_err());
        assert!(TsNetBuilder::<f32>::new().n_epochs(0).build().is_err());
        assert!(TsNetBuilder::<f32>::new().initial_lr(0.0).build().is_err());
        assert!(TsNetBuilder::<f32>::new().final_lr(0.0).build().is_err());
        assert!(TsNetBuilder::<f32>::new()
            .learning_rate(0.0)
            .build()
            .is_err());
        assert!(TsNetBuilder::<f32>::new()
            .initial_momentum(1.0)
            .build()
            .is_err());
        assert!(TsNetBuilder::<f32>::new()
            .initial_momentum(-0.1)
            .build()
            .is_err());
        assert!(TsNetBuilder::<f32>::new()
            .final_momentum(1.0)
            .build()
            .is_err());
        assert!(TsNetBuilder::<f32>::new().momentum(1.0).build().is_err());
        assert!(TsNetBuilder::<f32>::new()
            .initial_l_kl(Some(-0.1))
            .build()
            .is_err());
        assert!(TsNetBuilder::<f32>::new()
            .final_l_kl(Some(-0.1))
            .build()
            .is_err());
        assert!(TsNetBuilder::<f32>::new()
            .initial_l_c(Some(-0.1))
            .build()
            .is_err());
        assert!(TsNetBuilder::<f32>::new()
            .final_l_c(Some(-0.1))
            .build()
            .is_err());
        assert!(TsNetBuilder::<f32>::new()
            .initial_l_r(Some(-0.1))
            .build()
            .is_err());
        assert!(TsNetBuilder::<f32>::new()
            .final_l_r(Some(-0.1))
            .build()
            .is_err());
        assert!(TsNetBuilder::<f32>::new().r_eps(0.0).build().is_err());
        assert!(TsNetBuilder::<f32>::new().sigma_iters(0).build().is_err());
        assert!(TsNetBuilder::<f32>::new().autostop(-1.0).build().is_err());
        assert!(TsNetBuilder::<f32>::new().window_size(0).build().is_err());
    }

    #[test]
    fn test_bh_ts_net_builder_defaults() {
        let builder = BhTsNetBuilder::<f32>::new();
        let bh_ts_net = builder.build().expect("default builder should be valid");
        assert_eq!(bh_ts_net.perplexity, 40.0);
        assert_eq!(bh_ts_net.theta, 0.5);
        assert_eq!(bh_ts_net.k, 120);
        assert_eq!(bh_ts_net.iterations_stage1, 250);
        assert_eq!(bh_ts_net.exaggeration, 4.0);
        assert_eq!(bh_ts_net.iterations_stage2, 250);
        assert_eq!(bh_ts_net.lambda_c_stage2, 1.2);
        assert_eq!(bh_ts_net.iterations_stage3, 250);
        assert_eq!(bh_ts_net.lambda_c_stage3, 0.01);
        assert_eq!(bh_ts_net.lambda_r_stage3, 0.6);
        assert_eq!(bh_ts_net.learning_rate, 200.0);
        assert_eq!(bh_ts_net.momentum, 0.8);
        assert_eq!(bh_ts_net.power, 2.0);
        assert_eq!(bh_ts_net.epsilon_r, 0.05);
    }

    #[test]
    fn test_bh_ts_net_builder_validation() {
        assert!(BhTsNetBuilder::<f32>::new()
            .perplexity(-1.0)
            .build()
            .is_err());
        assert!(BhTsNetBuilder::<f32>::new().theta(0.0).build().is_err());
        assert!(BhTsNetBuilder::<f32>::new().k(0).build().is_err());
        assert!(BhTsNetBuilder::<f32>::new()
            .learning_rate(0.0)
            .build()
            .is_err());
        assert!(BhTsNetBuilder::<f32>::new().momentum(1.0).build().is_err());
    }

    #[test]
    fn test_fit_ts_net_builder_defaults() {
        let builder = FitTsNetBuilder::<f32>::new();
        let fit_ts_net = builder.build().expect("default builder should be valid");
        assert_eq!(fit_ts_net.intervals, 25);
        assert_eq!(fit_ts_net.interpolation_points, 3);
        assert_eq!(fit_ts_net.perplexity, 40.0);
        assert_eq!(fit_ts_net.theta, 0.5);
        assert_eq!(fit_ts_net.k, 120);
        assert_eq!(fit_ts_net.iterations_stage1, 250);
        assert_eq!(fit_ts_net.exaggeration, 4.0);
        assert_eq!(fit_ts_net.iterations_stage2, 250);
        assert_eq!(fit_ts_net.lambda_c_stage2, 1.2);
        assert_eq!(fit_ts_net.iterations_stage3, 250);
        assert_eq!(fit_ts_net.lambda_c_stage3, 0.01);
        assert_eq!(fit_ts_net.lambda_r_stage3, 0.6);
        assert_eq!(fit_ts_net.learning_rate, 200.0);
        assert_eq!(fit_ts_net.momentum, 0.8);
        assert_eq!(fit_ts_net.power, 2.0);
        assert_eq!(fit_ts_net.epsilon_r, 0.05);
    }

    #[test]
    fn test_fit_ts_net_builder_validation() {
        assert!(FitTsNetBuilder::<f32>::new().intervals(0).build().is_err());
        assert!(FitTsNetBuilder::<f32>::new()
            .interpolation_points(0)
            .build()
            .is_err());
        assert!(FitTsNetBuilder::<f32>::new()
            .perplexity(-1.0)
            .build()
            .is_err());
        assert!(FitTsNetBuilder::<f32>::new().theta(0.0).build().is_err());
        assert!(FitTsNetBuilder::<f32>::new().k(0).build().is_err());
        assert!(FitTsNetBuilder::<f32>::new()
            .learning_rate(0.0)
            .build()
            .is_err());
        assert!(FitTsNetBuilder::<f32>::new().momentum(1.0).build().is_err());
    }

    #[test]
    fn test_l_ts_net_builder_defaults() {
        let builder = LTsNetBuilder::<f32>::new();
        let l_ts_net = builder.build().expect("default builder should be valid");
        assert_eq!(l_ts_net.intervals, 25);
        assert_eq!(l_ts_net.interpolation_points, 3);
        assert_eq!(l_ts_net.perplexity, 40.0);
        assert_eq!(l_ts_net.k, 120);
        assert_eq!(l_ts_net.iterations_stage1, 250);
        assert_eq!(l_ts_net.exaggeration, 4.0);
        assert_eq!(l_ts_net.iterations_stage2, 250);
        assert_eq!(l_ts_net.lambda_c_stage2, 1.2);
        assert_eq!(l_ts_net.iterations_stage3, 250);
        assert_eq!(l_ts_net.lambda_c_stage3, 0.01);
        assert_eq!(l_ts_net.lambda_r_stage3, 0.6);
        assert_eq!(l_ts_net.learning_rate, 200.0);
        assert_eq!(l_ts_net.momentum, 0.8);
        assert_eq!(l_ts_net.power, 2.0);
        assert_eq!(l_ts_net.epsilon_r, 0.05);
    }

    #[test]
    fn test_l_ts_net_builder_validation() {
        assert!(LTsNetBuilder::<f32>::new().intervals(0).build().is_err());
        assert!(LTsNetBuilder::<f32>::new()
            .interpolation_points(0)
            .build()
            .is_err());
        assert!(LTsNetBuilder::<f32>::new()
            .perplexity(-1.0)
            .build()
            .is_err());
        assert!(LTsNetBuilder::<f32>::new().k(0).build().is_err());
        assert!(LTsNetBuilder::<f32>::new()
            .learning_rate(0.0)
            .build()
            .is_err());
        assert!(LTsNetBuilder::<f32>::new().momentum(1.0).build().is_err());
    }

    #[test]
    fn test_fft_roundtrip() {
        use crate::fft::{fft_1d, fft_2d, Complex};

        // 1D test
        let mut data = vec![
            Complex::new(1.0f32, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let original = data.clone();
        fft_1d(&mut data, false);
        fft_1d(&mut data, true);
        for (orig, res) in original.iter().zip(data.iter()) {
            assert!((orig.re - res.re).abs() < 1e-5);
            assert!((orig.im - res.im).abs() < 1e-5);
        }

        // 2D test
        let mut data_2d = vec![Complex::new(0.0f32, 0.0); 16];
        data_2d[0] = Complex::new(1.0, 0.0);
        data_2d[5] = Complex::new(2.5, 0.0);
        let orig_2d = data_2d.clone();
        fft_2d(&mut data_2d, 4, 4, false);
        fft_2d(&mut data_2d, 4, 4, true);
        for (orig, res) in orig_2d.iter().zip(data_2d.iter()) {
            assert!((orig.re - res.re).abs() < 1e-5);
            assert!((orig.im - res.im).abs() < 1e-5);
        }
    }

    #[test]
    fn test_ts_net_basic() {
        let mut graph = Graph::new_undirected();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, a, ());

        let mut coordinates = DrawingEuclidean2d::initial_placement(&graph);
        let distance_matrix = all_sources_dijkstra(&graph, |_| 1.0f32);

        let ts_net = TsNetBuilder::new()
            .learning_rate(2.0f32)
            .n_epochs(30)
            .build()
            .expect("valid builder configuration");

        ts_net.run(&mut coordinates, &distance_matrix);

        for &u in &[a, b, c] {
            let x = coordinates.x(u).unwrap();
            let y = coordinates.y(u).unwrap();
            assert!(x.is_finite());
            assert!(y.is_finite());
        }
    }

    #[test]
    fn test_bh_ts_net_basic() {
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
        let mut rng = StdRng::seed_from_u64(42);

        let bh_ts_net = BhTsNetBuilder::new()
            .perplexity(2.0f32)
            .k(3)
            .learning_rate(2.0f32)
            .iterations_stage1(10)
            .iterations_stage2(10)
            .iterations_stage3(10)
            .build()
            .expect("valid builder configuration");

        bh_ts_net.run(&mut coordinates, &graph, &mut rng);

        for &u in &[a, b, c, d] {
            let x = coordinates.x(u).unwrap();
            let y = coordinates.y(u).unwrap();
            assert!(x.is_finite());
            assert!(y.is_finite());
        }
    }

    #[test]
    fn test_fit_ts_net_basic() {
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
        let mut rng = StdRng::seed_from_u64(42);

        let fit_ts_net = FitTsNetBuilder::new()
            .intervals(10)
            .interpolation_points(3)
            .perplexity(2.0f32)
            .k(3)
            .learning_rate(2.0f32)
            .iterations_stage1(10)
            .iterations_stage2(10)
            .iterations_stage3(10)
            .build()
            .expect("valid builder configuration");

        fit_ts_net.run(&mut coordinates, &graph, &mut rng);

        for &u in &[a, b, c, d] {
            let x = coordinates.x(u).unwrap();
            let y = coordinates.y(u).unwrap();
            assert!(x.is_finite());
            assert!(y.is_finite());
        }
    }

    #[test]
    fn test_l_ts_net_basic() {
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
        let mut rng = StdRng::seed_from_u64(42);

        let l_ts_net = LTsNetBuilder::new()
            .intervals(10)
            .interpolation_points(3)
            .perplexity(2.0f32)
            .k(3)
            .learning_rate(2.0f32)
            .iterations_stage1(10)
            .iterations_stage2(10)
            .iterations_stage3(10)
            .build()
            .expect("valid builder configuration");

        l_ts_net.run(&mut coordinates, &graph, &mut rng);

        for &u in &[a, b, c, d] {
            let x = coordinates.x(u).unwrap();
            let y = coordinates.y(u).unwrap();
            assert!(x.is_finite());
            assert!(y.is_finite());
        }
    }

    #[test]
    fn test_ts_net_via_default_constructor() {
        let mut graph = Graph::new_undirected();
        let a = graph.add_node(());
        let b = graph.add_node(());
        graph.add_edge(a, b, ());

        let mut coordinates = DrawingEuclidean2d::initial_placement(&graph);
        let distance_matrix = all_sources_dijkstra(&graph, |_| 1.0f32);

        let ts_net = TsNet::<f32>::new();
        ts_net.run(&mut coordinates, &distance_matrix);

        for &u in &[a, b] {
            let x = coordinates.x(u).unwrap();
            let y = coordinates.y(u).unwrap();
            assert!(x.is_finite());
            assert!(y.is_finite());
        }
    }

    fn assert_sparse_p_symmetric_and_normalized(sparse_p: &SparseP<f64>, expected_sum: f64) {
        let n = sparse_p.len();
        let mut total = 0.0;
        for (i, row) in sparse_p.iter().enumerate() {
            let mut prev = None;
            for &(j, p_ij) in row {
                assert_ne!(i, j, "self-loop found in sparse_p[{i}]");
                assert!(j < n, "out-of-range neighbor {j} in sparse_p[{i}]");
                assert!(p_ij.is_finite() && p_ij > 0.0);
                if let Some(p) = prev {
                    assert!(p < j, "duplicate or unsorted neighbor {j} in sparse_p[{i}]");
                }
                prev = Some(j);
                let p_ji = sparse_p[j]
                    .iter()
                    .find(|&&(col, _)| col == i)
                    .map(|&(_, v)| v)
                    .expect("missing symmetric reverse entry");
                assert!(
                    (p_ij - p_ji).abs() < 1e-12,
                    "asymmetry at ({i}, {j}): {p_ij} vs {p_ji}"
                );
                total += p_ij;
            }
        }
        assert!(
            (total - expected_sum).abs() < 1e-6,
            "expected probability sum {expected_sum}, got {total}"
        );
    }

    #[test]
    fn test_compute_knn_probabilities_directed_symmetric_sum() {
        // Asymmetric directed k-NN graph: node 0 -> (1, 2), node 1 -> (2, 3), etc.
        let knn: Vec<Vec<(usize, f64)>> = vec![
            vec![(1, 0.2), (2, 1.0), (3, 2.5)],
            vec![(2, 0.3), (3, 1.1), (0, 2.0)],
            vec![(3, 0.4), (4, 0.9), (1, 1.8)],
            vec![(4, 0.25), (5, 1.2), (2, 2.1)],
            vec![(5, 0.35), (0, 1.5), (3, 2.2)],
            vec![(0, 0.45), (1, 1.4), (4, 2.4)],
        ];
        let sparse_p = compute_knn_probabilities(6, &knn, 2.0);
        assert_sparse_p_symmetric_and_normalized(&sparse_p, 1.0);
    }

    #[test]
    fn test_compute_knn_probabilities_filters_and_edge_cases() {
        // Empty and tiny n
        assert!(compute_knn_probabilities::<f64>(0, &[], 2.0).is_empty());
        assert_eq!(
            compute_knn_probabilities::<f64>(1, &[vec![(0, 1.0)]], 2.0),
            vec![Vec::<(usize, f64)>::new()]
        );

        // Self-loops, out-of-range, duplicates (keeping min distance), non-finite, and empty row
        let knn: Vec<Vec<(usize, f64)>> = vec![
            vec![
                (0, 0.0),           // self-loop -> filtered
                (99, 0.1),          // out of range -> filtered
                (1, 5.0),           // duplicate neighbor 1 -> min(5.0, 0.5) = 0.5
                (1, 0.5),           // duplicate with smaller distance
                (2, f64::NAN),      // non-finite -> filtered
                (3, f64::INFINITY), // non-finite -> filtered
                (2, 1.5),
            ],
            vec![(0, 0.5), (2, 1.0)],
            vec![(1, 1.0), (0, 1.5)],
            vec![], // empty row
        ];
        let sparse_p = compute_knn_probabilities(4, &knn, 1.5);
        // 3 out of 4 rows have neighbors => sum = 3 / 4 = 0.75
        assert_sparse_p_symmetric_and_normalized(&sparse_p, 0.75);

        // Verify duplicate neighbor 1 used distance 0.5 (so p_{1|0} > p_{2|0})
        let p01 = sparse_p[0].iter().find(|&&(j, _)| j == 1).unwrap().1;
        let p02 = sparse_p[0].iter().find(|&&(j, _)| j == 2).unwrap().1;
        assert!(p01 > p02);

        // Extreme tiny scale (1e-10) should still converge cleanly without underflow
        let tiny_knn: Vec<Vec<(usize, f64)>> = vec![
            vec![(1, 1e-10), (2, 1e-9), (3, 1e-8)],
            vec![(0, 1e-10), (2, 1e-9), (3, 1e-8)],
            vec![(0, 1e-9), (1, 1e-9), (3, 1e-10)],
            vec![(2, 1e-10), (1, 1e-8), (0, 1e-8)],
        ];
        let tiny_p = compute_knn_probabilities(4, &tiny_knn, 2.0);
        assert_sparse_p_symmetric_and_normalized(&tiny_p, 1.0);
    }

    #[test]
    fn test_partial_bfs_matches_knn_probabilities() {
        let mut graph = Graph::new_undirected();
        let nodes: Vec<_> = (0..6).map(|_| graph.add_node(())).collect();
        for i in 0..6 {
            graph.add_edge(nodes[i], nodes[(i + 1) % 6], ());
        }
        // k = 4 visits hop-1 (2 nodes) and hop-2 (2 nodes) with zero ties on C_6
        let mut rng = StdRng::seed_from_u64(42);
        let bfs_p = compute_partial_bfs_probabilities(&graph, 2.5f64, 4, 2.0, 50, 1e-5, &mut rng);

        let exact_knn: Vec<Vec<(usize, f64)>> = (0..6)
            .map(|i| {
                vec![
                    ((i + 1) % 6, 1.0),
                    ((i + 5) % 6, 1.0),
                    ((i + 2) % 6, 2.0),
                    ((i + 4) % 6, 2.0),
                ]
            })
            .collect();
        let knn_p = compute_knn_probabilities(6, &exact_knn, 2.5f64);

        assert_sparse_p_symmetric_and_normalized(&bfs_p, 1.0);
        assert_eq!(bfs_p.len(), knn_p.len());
        for i in 0..6 {
            assert_eq!(bfs_p[i].len(), knn_p[i].len());
            for (&(j_a, p_a), &(j_b, p_b)) in bfs_p[i].iter().zip(knn_p[i].iter()) {
                assert_eq!(j_a, j_b);
                assert!((p_a - p_b).abs() < 1e-12);
            }
        }
    }

    type ClusterKnnFixture = (
        petgraph::Graph<(), (), petgraph::Undirected>,
        Vec<petgraph::graph::NodeIndex>,
        Vec<Vec<(usize, f32)>>,
    );

    fn build_two_cluster_knn(n_per_cluster: usize) -> ClusterKnnFixture {
        let n = 2 * n_per_cluster;
        let mut graph = Graph::new_undirected();
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
        let mut knn = vec![Vec::new(); n];
        for (i, row) in knn.iter_mut().enumerate() {
            let same_start = if i < n_per_cluster { 0 } else { n_per_cluster };
            for j in 0..n {
                if i == j {
                    continue;
                }
                let in_same = j >= same_start && j < same_start + n_per_cluster;
                let dist = if in_same { 0.1f32 } else { 5.0f32 };
                row.push((j, dist));
            }
        }
        (graph, nodes, knn)
    }

    #[test]
    fn test_bh_fit_l_ts_net_run_with_knn() {
        let (graph, nodes, knn) = build_two_cluster_knn(4);

        let bh = BhTsNetBuilder::new()
            .perplexity(2.5f32)
            .learning_rate(5.0f32)
            .iterations_stage1(15)
            .iterations_stage2(15)
            .iterations_stage3(15)
            .build()
            .unwrap();
        let mut bh_drawing = DrawingEuclidean2d::initial_placement(&graph);
        bh.run_with_knn(&mut bh_drawing, &knn);
        for &u in &nodes {
            assert!(bh_drawing.x(u).unwrap().is_finite());
            assert!(bh_drawing.y(u).unwrap().is_finite());
        }

        let fit = FitTsNetBuilder::new()
            .intervals(10)
            .interpolation_points(3)
            .perplexity(2.5f32)
            .learning_rate(5.0f32)
            .iterations_stage1(15)
            .iterations_stage2(15)
            .iterations_stage3(15)
            .build()
            .unwrap();
        let mut fit_drawing = DrawingEuclidean2d::initial_placement(&graph);
        fit.run_with_knn(&mut fit_drawing, &knn);
        for &u in &nodes {
            assert!(fit_drawing.x(u).unwrap().is_finite());
            assert!(fit_drawing.y(u).unwrap().is_finite());
        }

        let l = LTsNetBuilder::new()
            .intervals(10)
            .interpolation_points(3)
            .perplexity(2.5f32)
            .learning_rate(5.0f32)
            .iterations_stage1(15)
            .iterations_stage2(15)
            .iterations_stage3(15)
            .build()
            .unwrap();
        let mut l_drawing = DrawingEuclidean2d::initial_placement(&graph);
        l.run_with_knn(&mut l_drawing, &knn);
        for &u in &nodes {
            assert!(l_drawing.x(u).unwrap().is_finite());
            assert!(l_drawing.y(u).unwrap().is_finite());
        }
    }

    #[test]
    fn test_run_with_knn_matches_run_on_cycle() {
        let mut graph = Graph::new_undirected();
        let nodes: Vec<_> = (0..6).map(|_| graph.add_node(())).collect();
        for i in 0..6 {
            graph.add_edge(nodes[i], nodes[(i + 1) % 6], ());
        }
        let knn: Vec<Vec<(usize, f32)>> = (0..6)
            .map(|i| {
                vec![
                    ((i + 1) % 6, 1.0),
                    ((i + 5) % 6, 1.0),
                    ((i + 2) % 6, 2.0),
                    ((i + 4) % 6, 2.0),
                ]
            })
            .collect();

        let bh = BhTsNetBuilder::new()
            .perplexity(2.5f32)
            .k(4)
            .learning_rate(2.0f32)
            .iterations_stage1(10)
            .iterations_stage2(10)
            .iterations_stage3(10)
            .build()
            .unwrap();
        let mut d_run = DrawingEuclidean2d::initial_placement(&graph);
        let mut d_knn = DrawingEuclidean2d::initial_placement(&graph);
        let mut rng = StdRng::seed_from_u64(42);
        bh.run(&mut d_run, &graph, &mut rng);
        bh.run_with_knn(&mut d_knn, &knn);

        for &u in &nodes {
            assert!((d_run.x(u).unwrap() - d_knn.x(u).unwrap()).abs() < 1e-4);
            assert!((d_run.y(u).unwrap() - d_knn.y(u).unwrap()).abs() < 1e-4);
        }

        let fit = FitTsNetBuilder::new()
            .intervals(10)
            .interpolation_points(3)
            .perplexity(2.5f32)
            .k(4)
            .learning_rate(2.0f32)
            .iterations_stage1(10)
            .iterations_stage2(10)
            .iterations_stage3(10)
            .build()
            .unwrap();
        let mut fit_run = DrawingEuclidean2d::initial_placement(&graph);
        let mut fit_knn = DrawingEuclidean2d::initial_placement(&graph);
        let mut rng_fit = StdRng::seed_from_u64(42);
        fit.run(&mut fit_run, &graph, &mut rng_fit);
        fit.run_with_knn(&mut fit_knn, &knn);

        for &u in &nodes {
            assert!((fit_run.x(u).unwrap() - fit_knn.x(u).unwrap()).abs() < 1e-4);
            assert!((fit_run.y(u).unwrap() - fit_knn.y(u).unwrap()).abs() < 1e-4);
        }

        let l = LTsNetBuilder::new()
            .intervals(10)
            .interpolation_points(3)
            .perplexity(2.5f32)
            .k(4)
            .learning_rate(2.0f32)
            .iterations_stage1(10)
            .iterations_stage2(10)
            .iterations_stage3(10)
            .build()
            .unwrap();
        let mut l_run = DrawingEuclidean2d::initial_placement(&graph);
        let mut l_knn = DrawingEuclidean2d::initial_placement(&graph);
        let mut rng_l = StdRng::seed_from_u64(42);
        l.run(&mut l_run, &graph, &mut rng_l);
        l.run_with_knn(&mut l_knn, &knn);

        for &u in &nodes {
            assert!((l_run.x(u).unwrap() - l_knn.x(u).unwrap()).abs() < 1e-4);
            assert!((l_run.y(u).unwrap() - l_knn.y(u).unwrap()).abs() < 1e-4);
        }

        // Single-node graph edge case should be a clean no-op
        let mut g1: Graph<(), (), petgraph::Undirected> = Graph::new_undirected();
        let n0 = g1.add_node(());
        let mut d1 = DrawingEuclidean2d::<_, f32>::initial_placement(&g1);
        bh.run_with_knn(&mut d1, &[vec![]]);
        fit.run_with_knn(&mut d1, &[vec![]]);
        l.run_with_knn(&mut d1, &[vec![]]);
        assert!(d1.x(n0).unwrap().is_finite());
        assert!(d1.y(n0).unwrap().is_finite());
    }

    #[test]
    fn test_fit_and_l_ts_net_default_250_iters_bounded() {
        let (graph, nodes, knn) = build_two_cluster_knn(5);

        let fit = FitTsNetBuilder::new().perplexity(3.0f32).build().unwrap();
        assert_eq!(fit.iterations_stage1, 250);
        assert_eq!(fit.iterations_stage2, 250);
        assert_eq!(fit.iterations_stage3, 250);

        let mut fit_drawing = DrawingEuclidean2d::initial_placement(&graph);
        fit.run_with_knn(&mut fit_drawing, &knn);
        for &u in &nodes {
            let x = fit_drawing.x(u).unwrap();
            let y = fit_drawing.y(u).unwrap();
            assert!(x.is_finite() && x.abs() < 1e3, "FitTsNet unbounded x={x}");
            assert!(y.is_finite() && y.abs() < 1e3, "FitTsNet unbounded y={y}");
        }

        let l = LTsNetBuilder::new().perplexity(3.0f32).build().unwrap();
        let mut l_drawing = DrawingEuclidean2d::initial_placement(&graph);
        l.run_with_knn(&mut l_drawing, &knn);
        for &u in &nodes {
            let x = l_drawing.x(u).unwrap();
            let y = l_drawing.y(u).unwrap();
            assert!(x.is_finite() && x.abs() < 1e3, "LTsNet unbounded x={x}");
            assert!(y.is_finite() && y.abs() < 1e3, "LTsNet unbounded y={y}");
        }
    }

    fn evaluate_layout_quality(
        drawing: &DrawingEuclidean2d<petgraph::graph::NodeIndex, f64>,
        nodes: &[petgraph::graph::NodeIndex],
        edges: &[(usize, usize)],
    ) -> (f64, f64, f64) {
        let n = nodes.len();
        let mut max_abs = 0.0f64;
        let pts: Vec<[f64; 2]> = nodes
            .iter()
            .map(|&u| {
                let x = drawing.x(u).unwrap();
                let y = drawing.y(u).unwrap();
                assert!(x.is_finite() && y.is_finite(), "non-finite coordinate");
                max_abs = max_abs.max(x.abs()).max(y.abs());
                [x, y]
            })
            .collect();
        let mean_edge = edges
            .iter()
            .map(|&(u, v)| {
                let dx = pts[u][0] - pts[v][0];
                let dy = pts[u][1] - pts[v][1];
                (dx * dx + dy * dy).sqrt()
            })
            .sum::<f64>()
            / edges.len() as f64;
        let mut total_all = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dx = pts[i][0] - pts[j][0];
                    let dy = pts[i][1] - pts[j][1];
                    total_all += (dx * dx + dy * dy).sqrt();
                }
            }
        }
        let mean_all = total_all / (n * (n - 1)) as f64;
        (max_abs, mean_edge, mean_all)
    }

    #[test]
    fn test_bh_fit_l_ts_net_grid_5x5_and_cycle_30_compact_and_coherent() {
        // grid_5x5 (n=25)
        let mut grid_graph = Graph::new_undirected();
        let grid_nodes: Vec<_> = (0..25).map(|_| grid_graph.add_node(())).collect();
        let mut grid_edges = Vec::new();
        for r in 0..5 {
            for c in 0..5 {
                let u = r * 5 + c;
                if c + 1 < 5 {
                    grid_graph.add_edge(grid_nodes[u], grid_nodes[u + 1], ());
                    grid_edges.push((u, u + 1));
                }
                if r + 1 < 5 {
                    grid_graph.add_edge(grid_nodes[u], grid_nodes[u + 5], ());
                    grid_edges.push((u, u + 5));
                }
            }
        }

        // cycle_30 (n=30)
        let mut cycle_graph = Graph::new_undirected();
        let cycle_nodes: Vec<_> = (0..30).map(|_| cycle_graph.add_node(())).collect();
        let mut cycle_edges = Vec::new();
        for i in 0..30 {
            let j = (i + 1) % 30;
            cycle_graph.add_edge(cycle_nodes[i], cycle_nodes[j], ());
            cycle_edges.push((i, j));
        }

        for (label, graph, nodes, edges, perp, large_init) in [
            (
                "grid_5x5",
                &grid_graph,
                &grid_nodes,
                &grid_edges,
                8.0f64,
                false,
            ),
            (
                "cycle_30",
                &cycle_graph,
                &cycle_nodes,
                &cycle_edges,
                10.0f64,
                false,
            ),
            (
                "cycle_30_large_init",
                &cycle_graph,
                &cycle_nodes,
                &cycle_edges,
                10.0f64,
                true,
            ),
        ] {
            let init_drawing = |g: &Graph<(), (), petgraph::Undirected>| {
                let mut d = DrawingEuclidean2d::<_, f64>::initial_placement(g);
                if large_init {
                    for (idx, &u) in nodes.iter().enumerate() {
                        d.set_x(u, (idx as f64 - 14.5) * 1000.0);
                        d.set_y(u, ((idx % 5) as f64 - 2.0) * 500.0);
                    }
                }
                d
            };

            let bh = BhTsNetBuilder::new().perplexity(perp).build().unwrap();
            let mut d_bh = init_drawing(graph);
            bh.run(&mut d_bh, graph, &mut StdRng::seed_from_u64(42));
            let (max_abs, me, ma) = evaluate_layout_quality(&d_bh, nodes, edges);
            assert!(
                max_abs < 100.0 && me < 0.6 * ma,
                "{label} BhTsNet failed: max_abs={max_abs:.2}, edge={me:.2}, all={ma:.2}"
            );

            let fit = FitTsNetBuilder::new().perplexity(perp).build().unwrap();
            let mut d_fit = init_drawing(graph);
            fit.run(&mut d_fit, graph, &mut StdRng::seed_from_u64(42));
            let (max_abs, me, ma) = evaluate_layout_quality(&d_fit, nodes, edges);
            assert!(
                max_abs < 100.0 && me < 0.6 * ma,
                "{label} FitTsNet failed: max_abs={max_abs:.2}, edge={me:.2}, all={ma:.2}"
            );

            let l = LTsNetBuilder::new().perplexity(perp).build().unwrap();
            let mut d_l = init_drawing(graph);
            l.run(&mut d_l, graph, &mut StdRng::seed_from_u64(42));
            let (max_abs, me, ma) = evaluate_layout_quality(&d_l, nodes, edges);
            assert!(
                max_abs < 100.0 && me < 0.6 * ma,
                "{label} LTsNet failed: max_abs={max_abs:.2}, edge={me:.2}, all={ma:.2}"
            );
        }
    }
}
