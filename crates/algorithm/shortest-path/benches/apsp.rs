use criterion::{criterion_group, criterion_main, Criterion};
use egraph_dataset::dataset_1138_bus;
use petgraph::prelude::*;
use petgraph_algorithm_shortest_path::*;

fn criterion_benchmark(c: &mut Criterion) {
    let graph: UnGraph<(), ()> = dataset_1138_bus();
    let mut group = c.benchmark_group("1138_bus");
    group.bench_with_input("all_sources_bfs", &graph, |bench, graph| {
        bench.iter(|| {
            let _ = all_sources_bfs(graph, 30.);
        });
    });
    group.bench_with_input("all_sources_dijkstra", &graph, |bench, graph| {
        bench.iter(|| {
            let _ = all_sources_dijkstra(graph, &mut |_| 30.);
        });
    });
    group.bench_with_input("warshall_floyd", &graph, |bench, graph| {
        bench.iter(|| {
            let _ = warshall_floyd(graph, &mut |_| 30.);
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
