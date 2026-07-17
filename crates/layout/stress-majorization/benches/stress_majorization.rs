use criterion::{criterion_group, criterion_main, Criterion};
use egraph_dataset::dataset_lesmis;
use petgraph::prelude::*;
use petgraph_drawing::DrawingEuclidean2d;
use petgraph_layout_stress_majorization::StressMajorization;

fn criterion_benchmark(c: &mut Criterion) {
    let graph: UnGraph<(), ()> = dataset_lesmis();
    let initial_drawing = DrawingEuclidean2d::<NodeIndex, f64>::initial_placement(&graph);

    let mut group = c.benchmark_group("stress_majorization");
    group.bench_function("lesmis", |bench| {
        bench.iter(|| {
            let mut drawing = initial_drawing.clone();
            let mut sm = StressMajorization::new(&graph, &drawing, |_| 1.0);
            sm.max_iterations = 30;
            sm.run(&mut drawing);
        });
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
