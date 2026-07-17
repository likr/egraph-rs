use criterion::{criterion_group, criterion_main, Criterion};
use egraph_dataset::dataset_lesmis;
use petgraph::prelude::*;
use petgraph_drawing::DrawingEuclidean2d;
use petgraph_layout_kamada_kawai::KamadaKawai;

fn criterion_benchmark(c: &mut Criterion) {
    let graph: UnGraph<(), ()> = dataset_lesmis();
    let initial_drawing = DrawingEuclidean2d::<NodeIndex, f64>::initial_placement(&graph);

    let mut group = c.benchmark_group("kamada_kawai");
    group.bench_function("lesmis", |bench| {
        bench.iter(|| {
            let mut drawing = initial_drawing.clone();
            let kk = KamadaKawai::new(&graph, |_| 1.0);
            kk.run(&mut drawing);
        });
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
