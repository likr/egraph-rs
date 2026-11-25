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

    let caf_report = evaluate_metrics_with_communities(
        &graph,
        &layout,
        &[MetricTarget::ClusterAreaFaithfulness],
        &cfg,
        Some(&louvain_res.labels),
    )
    .expect("evaluate_metrics_with_communities G-CAF");
    let caf_val = caf_report
        .cluster_area_faithfulness
        .expect("cluster_area_faithfulness should be populated");
    assert!(caf_val.is_finite() && (0.0..=1.0).contains(&caf_val));

    let caf_detail = cluster_area_faithfulness_detail(
        &layout,
        &louvain_res.labels,
        &ClusterAreaConfig::default(),
    )
    .expect("cluster_area_faithfulness_detail");
    assert_eq!(caf_detail.report.num_nodes, 34);
    assert_eq!(caf_detail.cluster_ids.len(), caf_detail.report.num_clusters);
    assert!((caf_detail.report.caf - caf_val).abs() < 1e-12);
    assert!(
        caf_detail.report.spatial_coherence.is_finite()
            && (0.0..=1.0).contains(&caf_detail.report.spatial_coherence)
    );

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

#[cfg(all(
    feature = "layout",
    feature = "baselines",
    feature = "metrics",
    feature = "eval-baselines",
    feature = "clustering",
    feature = "dataset"
))]
#[test]
fn facade_baselines_smoke() {
    let graph = egraph::dataset::dataset_karate_csr();
    let leiden_res = leiden(&graph, &LeidenConfig::default()).expect("leiden");
    assert_eq!(leiden_res.labels.len(), graph.num_nodes());

    let mut layout = Layout2d::<f64>::zeros(graph.num_nodes());
    let mut linlog = LinLogSolver::<f64>::new(
        &graph,
        LinLogConfig {
            iterations: 5,
            ..LinLogConfig::default()
        },
    )
    .expect("linlog");
    assert!(linlog.run(&mut layout).expect("linlog run").is_finished());

    let mut fr = FruchtermanReingoldSolver::<f64>::new(
        &graph,
        FruchtermanReingoldConfig {
            iterations: 5,
            ..FruchtermanReingoldConfig::default()
        },
    )
    .expect("fr");
    assert!(fr.run(&mut layout).expect("fr run").is_finished());

    let backbone = extract_quadrilateral_backbone(&graph, &BackboneConfig::default())
        .expect("extract_quadrilateral_backbone");
    assert_eq!(backbone.simmelian_weights.len(), graph.num_edges());
    let mut bb_solver = BackboneSolver::<f64>::new(
        &graph,
        BackboneConfig {
            iterations: 5,
            ..BackboneConfig::default()
        },
    )
    .expect("backbone solver");
    assert!(bb_solver.run(&mut layout).expect("bb run").is_finished());

    let mut refine = ClusterRefineSolver::<f64>::new(
        leiden_res.labels.clone(),
        ClusterRefineConfig {
            max_rounds: 2,
            ..ClusterRefineConfig::default()
        },
    )
    .expect("cluster_refine");
    assert!(refine.run(&mut layout).expect("refine run").is_finished());

    let boxes = squarified_treemap(&[1.0, 2.0, 3.0], 0.0, 0.0, 100.0, 100.0).expect("treemap");
    assert_eq!(boxes.len(), 3);
    let mut gib = GroupInABoxSolver::<f64>::new(
        &graph,
        leiden_res.labels.clone(),
        GroupInABoxConfig {
            intra_iterations: 5,
            ..GroupInABoxConfig::default()
        },
    )
    .expect("gib");
    assert!(gib.run(&mut layout).expect("gib run").is_finished());

    let radii = vec![0.25_f64; graph.num_nodes()];
    let mut gtree = GTreeOverlapRemoval::<f64>::from_radii(radii.clone(), GTreeConfig::default())
        .expect("gtree");
    assert!(gtree.run(&mut layout).expect("gtree run").is_finished());
    assert_eq!(gtree.last_overlap_count(), 0);

    let mut forbid = ForbidOverlapRemoval::<f64>::from_radii(
        radii,
        ForbidConfig {
            max_scale_iters: 5,
            sgd_iterations: 5,
            ..ForbidConfig::default()
        },
    )
    .expect("forbid");
    assert!(forbid.run(&mut layout).expect("forbid run").is_finished());
    assert_eq!(forbid.last_overlap_count(), 0);

    let agr = partition_agreement(&leiden_res.labels, &leiden_res.labels).expect("agreement");
    assert!((agr.ari - 1.0).abs() < 1e-12);
    let dsc = distance_consistency(&layout, &leiden_res.labels).expect("dsc");
    assert!(dsc.is_finite());

    let report = evaluate_metrics_with_communities(
        &graph,
        &layout,
        &[MetricTarget::Dsc, MetricTarget::SprawlterNn],
        &MetricConfig::default(),
        Some(&leiden_res.labels),
    )
    .expect("eval-baselines dispatcher");
    assert!(report.dsc.is_some());
    assert!(report.sprawlter_nn.is_some());
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

#[cfg(all(feature = "linalg", feature = "layout"))]
#[test]
fn facade_w2_core_smoke() {
    let graph = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)],
        None,
        false,
        false,
    )
    .expect("valid graph");

    let hde_cfg = HdeConfig {
        dimensions: 4,
        ..HdeConfig::default()
    };
    let hde_2d = hde::<f64, f64>(&graph, &hde_cfg).expect("hde");
    assert_eq!(hde_2d.len(), 6);
    assert!(hde_2d.is_all_finite());
    let hde_piv =
        hde_with_pivots::<f64, f64>(&graph, &[0, 2, 4], &hde_cfg).expect("hde_with_pivots");
    assert!(hde_piv.is_all_finite());
    let hde_nd_cfg = HdeConfig {
        dimensions: 4,
        output_dims: 3,
        ..HdeConfig::default()
    };
    let nd_1 = hde_nd::<f64, f64>(&graph, &hde_nd_cfg).expect("hde_nd");
    let nd_2 = hde_nd_with_pivots::<f64, f64>(&graph, &[0, 2, 4], &hde_nd_cfg)
        .expect("hde_nd_with_pivots");
    assert_eq!(nd_1.dims(), 3);
    assert_eq!(nd_2.dims(), 3);

    let mut layout = hde_2d;
    let mut ortmann = SgdSolver::<f64>::ortmann_sparse_stress(
        &graph,
        OrtmannSparseStressConfig {
            num_pivots: 3,
            pivot_strategy: OrtmannPivotStrategy::MaxMinSp,
            iterations: 5,
            ..OrtmannSparseStressConfig::default()
        },
    )
    .expect("ortmann");
    assert!(ortmann.run(&mut layout).expect("ortmann run").is_finished());

    let mut lv = SgdSolver::<f64>::largevis(
        &graph,
        LargeVisConfig {
            iterations: 4,
            ..LargeVisConfig::default()
        },
    )
    .expect("largevis");
    assert!(lv.run(&mut layout).expect("largevis run").is_finished());

    let mut dr = SgdSolver::<f64>::drgraph(
        &graph,
        DrGraphConfig {
            iterations: 4,
            ..DrGraphConfig::default()
        },
    )
    .expect("drgraph");
    assert!(dr.run(&mut layout).expect("drgraph run").is_finished());

    let mut cne = SgdSolver::<f64>::graph_cne(
        &graph,
        GraphCneConfig {
            objective: GraphCneObjective::InfoNce,
            iterations: 4,
            ..GraphCneConfig::default()
        },
    )
    .expect("graph_cne");
    assert!(cne.run(&mut layout).expect("graph_cne run").is_finished());
}

#[cfg(all(feature = "layout", feature = "baselines", feature = "dataset"))]
#[test]
fn facade_w2_baselines_smoke() {
    let graph = egraph::dataset::dataset_karate_csr();
    let mut layout = Layout2d::<f64>::zeros(graph.num_nodes());

    let mut maxent = MaxentStressSolver::<f64>::new(
        &graph,
        MaxentStressConfig {
            stages: 2,
            steps_per_stage: 3,
            ..MaxentStressConfig::default()
        },
    )
    .expect("maxent");
    assert!(maxent.run(&mut layout).expect("maxent run").is_finished());
    assert!(maxent.current_alpha().is_finite());
    assert!(maxent.effective_q().is_finite());

    let mut grip = GripSolver::<f64>::new(
        &graph,
        GripConfig {
            rounds_min: 1,
            rounds_max: 2,
            ..GripConfig::default()
        },
    )
    .expect("grip");
    assert!(grip.run(&mut layout).expect("grip run").is_finished());

    let mut gem = GemSolver::<f64>::new(
        &graph,
        GemConfig {
            max_rounds: Some(4),
            ..GemConfig::default()
        },
    )
    .expect("gem");
    assert!(gem.run(&mut layout).expect("gem run").is_finished());

    let mut sel = SelLayoutSolver::<f64>::new(
        &graph,
        SelConfig {
            mapping: SelWeightMapping::Sigmoid,
            layout_backend: SelLayoutBackend::Fr,
            iterations: 4,
            ..SelConfig::default()
        },
    )
    .expect("sel");
    assert_eq!(sel.modulated_weights().len(), graph.num_edges());
    assert!(sel.se_target().is_finite());
    assert!(sel.run(&mut layout).expect("sel run").is_finished());

    let mut sfc = SfcLayoutSolver::<f64>::new(
        &graph,
        SfcConfig {
            curve: SfcCurveKind::Hilbert,
            ..SfcConfig::default()
        },
    )
    .expect("sfc");
    assert_eq!(sfc.vertex_order().len(), graph.num_nodes());
    assert!(sfc.run(&mut layout).expect("sfc run").is_finished());

    let radii = vec![0.25_f64; graph.num_nodes()];
    let mut pfsp = PfsPrimeOverlapRemoval::<f64>::from_radii(radii, PfsPrimeConfig::default())
        .expect("pfs_prime");
    assert!(pfsp.run(&mut layout).expect("pfs_prime run").is_finished());
    assert_eq!(pfsp.last_overlap_count(), 0);
}
