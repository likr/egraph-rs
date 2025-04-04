use egraph_dataset::dataset_qh882;
use petgraph::prelude::*;
use petgraph_algorithm_shortest_path::*;

fn run<F, D>(f: F)
where
    F: Fn(&UnGraph<(), ()>) -> D,
    D: DistanceMatrix<NodeIndex, f32>,
{
    let graph: UnGraph<(), ()> = dataset_qh882();
    let actual = f(&graph);
    let expected = petgraph::algo::floyd_warshall(&graph, |_| 1.).unwrap();
    for u in graph.node_indices() {
        for v in graph.node_indices() {
            assert_eq!(
                actual.get(u, v).unwrap(),
                expected[&(u, v)],
                "d[{:?}, {:?}]",
                u,
                v
            );
        }
    }
}

#[test]
fn test_all_sources_bfs() {
    run(|graph| all_sources_bfs(graph, 1.));
}

#[test]
fn test_all_sources_dijkstra() {
    run(|graph| all_sources_dijkstra(graph, &mut |_| 1.));
}

#[test]
fn test_warshall_floyd() {
    run(|graph| warshall_floyd(graph, &mut |_| 1.));
}
