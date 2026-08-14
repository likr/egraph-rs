//! # tsNET Graph Layout Algorithm
//!
//! This crate provides the `TsNet` layout algorithm, which optimizes graph layouts
//! using a t-SNE-like objective function with early compression and entropy repulsion.
//!
//! ## References
//!
//! Kruiger, J. F., Rauber, P. E., Martins, R. M., Kerren, A., Kobourov, S., & Telea, A. C. (2017).
//! Graph Layouts by t-SNE. *Computer Graphics Forum*, 36(3), 283-294.

mod ts_net;
mod ts_net_builder;

pub use ts_net::TsNet;
pub use ts_net_builder::TsNetBuilder;

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;
    use petgraph_algorithm_shortest_path::all_sources_dijkstra;
    use petgraph_drawing::DrawingEuclidean2d;

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

        // Verify that the coordinates are finite numbers
        for &u in &[a, b, c] {
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
