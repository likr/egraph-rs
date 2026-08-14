//! # tsNET, BH-tsNET & FIt-tsNET Graph Layout Algorithms
//!
//! This crate provides the `TsNet`, `BhTsNet`, and `FitTsNet` layout algorithms:
//! - `TsNet`: Full-matrix t-SNE-based graph layout optimizing KL divergence, early compression, and entropy.
//! - `BhTsNet`: Accelerated O(N log N) Barnes-Hut tsNET combining Partial BFS and Quadtree spatial approximations.
//! - `FitTsNet`: Accelerated O(N log N) Fast Interpolation tsNET combining Partial BFS, 2D FFT interpolation for KL divergence, and Quadtree for entropy.
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
pub mod partial_bfs;
pub mod quadtree;
mod ts_net;
mod ts_net_builder;

pub use bh_ts_net::BhTsNet;
pub use bh_ts_net_builder::BhTsNetBuilder;
pub use fit_ts_net::FitTsNet;
pub use fit_ts_net_builder::FitTsNetBuilder;
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
        assert_eq!(ts_net.perplexity, 30.0);
        assert_eq!(ts_net.iterations_stage1, 250);
        assert_eq!(ts_net.exaggeration, 4.0);
        assert_eq!(ts_net.iterations_stage2, 250);
        assert_eq!(ts_net.lambda_c_stage2, 1.2);
        assert_eq!(ts_net.iterations_stage3, 250);
        assert_eq!(ts_net.lambda_c_stage3, 0.01);
        assert_eq!(ts_net.lambda_r_stage3, 0.6);
        assert_eq!(ts_net.learning_rate, 200.0);
        assert_eq!(ts_net.momentum, 0.8);
        assert_eq!(ts_net.power, 2.0);
        assert_eq!(ts_net.epsilon_r, 0.05);
    }

    #[test]
    fn test_ts_net_builder_validation() {
        assert!(TsNetBuilder::<f32>::new().perplexity(-1.0).build().is_err());
        assert!(TsNetBuilder::<f32>::new()
            .learning_rate(0.0)
            .build()
            .is_err());
        assert!(TsNetBuilder::<f32>::new().momentum(1.0).build().is_err());
        assert!(TsNetBuilder::<f32>::new().momentum(-0.1).build().is_err());
        assert!(TsNetBuilder::<f32>::new().power(0.0).build().is_err());
        assert!(TsNetBuilder::<f32>::new().epsilon_r(0.0).build().is_err());
        assert!(TsNetBuilder::<f32>::new()
            .exaggeration(0.5)
            .build()
            .is_err());
        assert!(TsNetBuilder::<f32>::new().sigma_iters(0).build().is_err());
        assert!(TsNetBuilder::<f32>::new()
            .sigma_tolerance(0.0)
            .build()
            .is_err());
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
            .iterations_stage1(10)
            .iterations_stage2(10)
            .iterations_stage3(10)
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
}
