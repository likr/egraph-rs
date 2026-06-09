//! # tsNET Graph Layout Algorithm
//!
//! This crate provides the `TsNet` layout algorithm, which optimizes graph layouts
//! using a t-SNE-like objective function with early compression and entropy repulsion.

mod ts_net;

pub use ts_net::TsNet;

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;
    use petgraph_algorithm_shortest_path::all_sources_dijkstra;
    use petgraph_drawing::DrawingEuclidean2d;

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

        let mut ts_net = TsNet::new();
        ts_net.learning_rate(2.0f32);
        ts_net.run(&mut coordinates, &distance_matrix);

        // Verify that the coordinates are valid numbers
        for &u in &[a, b, c] {
            let x = coordinates.x(u).unwrap();
            let y = coordinates.y(u).unwrap();
            assert!(x.is_finite());
            assert!(y.is_finite());
        }
    }
}
