use egraph_dataset::dataset_1138_bus;
use ndarray::prelude::*;
use petgraph::{graph::node_index, prelude::*};
use petgraph_algorithm_shortest_path::*;

fn run<F>(f: F)
where
    F: Fn(&UnGraph<(), ()>) -> Array2<f32>,
{
    let graph: UnGraph<(), ()> = dataset_1138_bus();
    let actual = f(&graph);
    let expected = petgraph::algo::floyd_warshall(&graph, |_| 1.).unwrap();
    let n = graph.node_count();
    for u in 0..n {
        for v in 0..n {
            assert_eq!(
                actual[[u, v]],
                expected[&(node_index(u), node_index(v))],
                "d[{}, {}]",
                u,
                v
            );
        }
    }
}

#[test]
fn test_warshall_floyd() {
    run(|graph| warshall_floyd(graph, &mut |_| 1.));
}

#[test]
fn test_all_sources_dijkstra() {
    run(|graph| all_sources_dijkstra(graph, &mut |_| 1.));
}
