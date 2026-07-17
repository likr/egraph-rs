use criterion::{criterion_group, criterion_main, Criterion};
use egraph_dataset::{dataset_1138_bus, dataset_lesmis};
use petgraph::prelude::*;
use petgraph_algorithm_shortest_path::*;

fn criterion_benchmark(c: &mut Criterion) {
    // 1. Benchmark on medium-sized sparse graph (Les Miserables - 77 nodes)
    // This will run very quickly and allow highly accurate measurements
    let graph_lesmis: UnGraph<(), ()> = dataset_lesmis();
    let mut group_lesmis = c.benchmark_group("lesmis");
    group_lesmis.bench_with_input("all_sources_bfs", &graph_lesmis, |bench, graph| {
        bench.iter(|| {
            let _ = all_sources_bfs(graph, 30.);
        });
    });
    group_lesmis.bench_with_input("all_sources_dijkstra", &graph_lesmis, |bench, graph| {
        bench.iter(|| {
            let _ = all_sources_dijkstra(graph, &mut |_| 30.);
        });
    });
    group_lesmis.bench_with_input("warshall_floyd", &graph_lesmis, |bench, graph| {
        bench.iter(|| {
            let _ = warshall_floyd(graph, &mut |_| 30.);
        });
    });
    group_lesmis.finish();

    // 2. Benchmark on larger graph (1138_bus - 1138 nodes)
    // We restrict sample size to 10 to prevent long runs and timeouts
    let graph_bus: UnGraph<(), ()> = dataset_1138_bus();
    let mut group_bus = c.benchmark_group("1138_bus");
    group_bus.sample_size(10);
    group_bus.bench_with_input("all_sources_bfs", &graph_bus, |bench, graph| {
        bench.iter(|| {
            let _ = all_sources_bfs(graph, 30.);
        });
    });
    group_bus.bench_with_input("all_sources_dijkstra", &graph_bus, |bench, graph| {
        bench.iter(|| {
            let _ = all_sources_dijkstra(graph, &mut |_| 30.);
        });
    });
    group_bus.bench_with_input("warshall_floyd", &graph_bus, |bench, graph| {
        bench.iter(|| {
            let _ = warshall_floyd(graph, &mut |_| 30.);
        });
    });
    group_bus.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
