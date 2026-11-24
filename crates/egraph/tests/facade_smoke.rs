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

#[cfg(feature = "linalg")]
#[test]
fn facade_amg_pcg_smoke() {
    let graph = CsrGraph::<f64>::from_edges(
        5,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
        None,
        false,
        false,
    )
    .expect("valid cycle graph");
    let solver = AmgPcgSolver::from_regularized_sym_laplacian(&graph, 2.0, AmgPcgConfig::default())
        .expect("amg pcg solver");
    let (x, res) = solver
        .solve_vec(&[1.0, -1.0, 0.5, -0.5, 0.0])
        .expect("solve");
    assert!(res.converged);
    assert!(x.iter().all(|v| v.is_finite()));
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

#[cfg(all(feature = "linalg", feature = "layout"))]
#[test]
fn facade_resistance_curvature_manifold_smoke() {
    // 1. Barbell graph -> HyperbolicPoincare + SgdSolver::full + run_manifold
    let mut edges = Vec::new();
    for u in 0..5u32 {
        for v in (u + 1)..5u32 {
            edges.push((u, v));
            edges.push((u + 5, v + 5));
        }
    }
    edges.push((4, 5));
    let barbell = CsrGraph::<f64>::from_edges(10, &edges, None, false, false).expect("barbell");
    let flow_hyp = resistance_curvature_flow(
        &barbell,
        &ResistanceCurvatureConfig {
            steps: 2,
            eta: 0.2,
            ..Default::default()
        },
    )
    .expect("barbell rcf");
    let length_graph = flow_hyp.to_length_graph().expect("length graph");
    let init_euclid = pivot_mds::<f64, f64>(&length_graph, &PivotMdsConfig::default())
        .expect("pivot_mds on length_graph");

    let mut ml_hyp = ManifoldLayout2d::<f64>::from_layout_scaled(
        ManifoldKind::HyperbolicPoincare,
        &init_euclid,
        0.7,
    )
    .expect("from_layout_scaled hyperbolic");
    let mut sgd_hyp = SgdSolver::<f64>::full(
        &length_graph,
        SgdScheduleConfig {
            t_max: 8,
            ..Default::default()
        },
        42,
    )
    .expect("sgd full hyp");
    let st_hyp = sgd_hyp.run_manifold(&mut ml_hyp).expect("run_manifold hyp");
    assert!(st_hyp.is_finished());
    assert!(ml_hyp.as_layout().is_all_finite());

    // 2. Spherical manifold with random_init + run_manifold
    let mut ml_sph = ManifoldLayout2d::<f64>::random_init(ManifoldKind::Spherical, 10, 0.5, 42)
        .expect("random_init spherical");
    let mut sgd_sph = SgdSolver::<f64>::full(
        &length_graph,
        SgdScheduleConfig {
            t_max: 8,
            step_scale: 0.1,
            ..Default::default()
        },
        42,
    )
    .expect("sgd full sph");
    let st_sph = sgd_sph.run_manifold(&mut ml_sph).expect("run_manifold sph");
    assert!(st_sph.is_finished());
    assert!(ml_sph.as_layout().is_all_finite());

    // 3. Torus2d on T_{8x8} + run_manifold
    let mut t_edges = Vec::new();
    for r in 0..8u32 {
        for c in 0..8u32 {
            let u = r * 8 + c;
            t_edges.push((u, r * 8 + ((c + 1) % 8)));
            t_edges.push((u, ((r + 1) % 8) * 8 + c));
        }
    }
    let torus = CsrGraph::<f64>::from_edges(64, &t_edges, None, false, true).expect("torus");
    let flow_tor = resistance_curvature_flow(
        &torus,
        &ResistanceCurvatureConfig {
            steps: 1,
            ..Default::default()
        },
    )
    .expect("torus rcf");
    assert_eq!(flow_tor.suggested_geometry, SuggestedGeometry::Torus2d);
    assert_eq!(
        flow_tor.suggested_manifold_kind(),
        Some(ManifoldKind::Torus2d)
    );
    let mut ml_tor = ManifoldLayout2d::<f64>::random_init(ManifoldKind::Torus2d, 64, 0.4, 42)
        .expect("random_init torus");
    let mut sgd_tor = SgdSolver::<f64>::full(
        &flow_tor.to_length_graph().expect("torus length graph"),
        SgdScheduleConfig {
            t_max: 8,
            ..Default::default()
        },
        42,
    )
    .expect("sgd full tor");
    let st_tor = sgd_tor.run_manifold(&mut ml_tor).expect("run_manifold tor");
    assert!(st_tor.is_finished());
    assert!(ml_tor.as_layout().is_all_finite());
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
