use egraph::prelude::*;

#[test]
fn facade_core_smoke() {
    let graph =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], None, false, false)
            .expect("valid cycle graph");
    assert_eq!(graph.num_nodes(), 4);
    assert_eq!(graph.num_edges(), 4);

    let layout = Layout2d::<f64>::from_xy(vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 0.0, 1.0, 1.0])
        .expect("valid layout");
    assert!(layout.is_all_finite());

    let manifold = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, 4);
    assert_eq!(manifold.len(), 4);
}

#[cfg(feature = "layout")]
#[test]
fn facade_layout_smoke() {
    let graph = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)],
        None,
        false,
        false,
    )
    .expect("valid graph");
    let mut layout = pivot_mds::<f64, f64>(&graph, &PivotMdsConfig::default()).expect("pivot_mds");
    let mut solver = TsNetSolver::<f64>::from_graph_pure_2hop(
        &graph,
        TsNetConfig {
            iterations_stage1: 2,
            iterations_stage2: 2,
            iterations_stage3: 2,
            ..TsNetConfig::pure_2hop()
        },
    )
    .expect("tsnet pure_2hop");
    let status = solver.run(&mut layout).expect("tsnet run");
    assert!(status.is_finished());
    assert!(layout.is_all_finite());
}

#[cfg(all(feature = "metrics", feature = "clustering", feature = "dataset"))]
#[test]
fn facade_metrics_clustering_dataset_smoke() {
    let graph = egraph::dataset::dataset_karate_csr();
    assert_eq!(graph.num_nodes(), 34);

    let mut layout =
        pivot_mds::<f64, f64>(&graph, &PivotMdsConfig::default()).expect("pivot_mds on karate");
    let louvain_res = louvain(&graph, &LouvainConfig::default()).expect("louvain");
    assert_eq!(louvain_res.labels.len(), 34);

    let cfg = MetricConfig {
        sns_num_pivots: 4,
        ..MetricConfig::default()
    };
    let report = evaluate_metrics(
        &graph,
        &layout,
        &[
            MetricTarget::ScaleNormalizedStress,
            MetricTarget::NeighborhoodPreservation2Hop,
        ],
        &cfg,
    )
    .expect("evaluate_metrics");
    assert!(report.scale_normalized_stress.is_some());
    assert!(report.neighborhood_preservation_2hop.is_some());

    normalize_layout_2d(&mut layout, 1.0, 42).expect("normalize_layout_2d");
    assert!(layout.is_all_finite());
}

#[cfg(feature = "petgraph")]
#[test]
fn facade_petgraph_smoke() {
    let graph =
        CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).expect("valid graph");
    let pg = to_petgraph_ungraph(&graph).expect("to_petgraph_ungraph");
    let roundtrip = from_petgraph_ungraph(&pg).expect("from_petgraph_ungraph");
    assert_eq!(roundtrip.num_nodes(), 3);
    assert_eq!(roundtrip.num_edges(), 2);
}
