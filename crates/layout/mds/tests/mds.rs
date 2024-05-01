use egraph_dataset::dataset_1138_bus;
use petgraph::prelude::*;
use petgraph_layout_mds::{ClassicalMds, PivotMds};

#[test]
fn test_classical_mds_2d() {
    let graph: UnGraph<(), ()> = dataset_1138_bus();
    let mds = ClassicalMds::new(&graph, |_| 1.);
    let drawing = mds.run_2d();
    for u in graph.node_indices() {
        assert!(drawing.x(u).unwrap().is_finite());
        assert!(drawing.y(u).unwrap().is_finite());
    }
}

#[test]
fn test_classical_mds_3d() {
    let graph: UnGraph<(), ()> = dataset_1138_bus();
    let mds = ClassicalMds::new(&graph, |_| 1.);
    let drawing = mds.run(3);
    for u in graph.node_indices() {
        assert!(drawing.get(u, 0).unwrap().is_finite());
        assert!(drawing.get(u, 1).unwrap().is_finite());
        assert!(drawing.get(u, 2).unwrap().is_finite());
    }
}

#[test]
fn test_pivot_mds_2d() {
    let graph: UnGraph<(), ()> = dataset_1138_bus();
    let pivot = graph.node_indices().take(50).collect::<Vec<_>>();
    let mds = PivotMds::new(&graph, |_| 1., &pivot);
    let drawing = mds.run_2d();
    for u in graph.node_indices() {
        assert!(drawing.x(u).unwrap().is_finite());
        assert!(drawing.y(u).unwrap().is_finite());
    }
}

#[test]
fn test_pivot_mds_3d() {
    let graph: UnGraph<(), ()> = dataset_1138_bus();
    let pivot = graph.node_indices().take(50).collect::<Vec<_>>();
    let mds = PivotMds::new(&graph, |_| 1., &pivot);
    let drawing = mds.run(3);
    for u in graph.node_indices() {
        assert!(drawing.get(u, 0).unwrap().is_finite());
        assert!(drawing.get(u, 1).unwrap().is_finite());
        assert!(drawing.get(u, 2).unwrap().is_finite());
    }
}
