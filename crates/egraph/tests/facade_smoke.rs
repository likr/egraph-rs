use egraph::prelude::*;

#[test]
fn facade_core_smoke() {
    let graph =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], None, false, false)
            .expect("valid cycle graph");
    assert_eq!(graph.num_nodes(), 4);
    assert_eq!(graph.num_edges(), 4);

    let mut builder = CsrGraphBuilder::<()>::new(false);
    let a = builder.add_node().expect("add_node a");
    let b = builder.add_node().expect("add_node b");
    let c = builder.add_node().expect("add_node c");
    builder.add_edge(a, b).expect("add_edge ab");
    builder.add_edge(b, c).expect("add_edge bc");
    let built = builder.build_unweighted(false).expect("builder graph");
    assert_eq!(built.num_nodes(), 3);
    assert_eq!(built.num_edges(), 2);

    let layout = Layout2d::<f64>::from_xy(vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 0.0, 1.0, 1.0])
        .expect("valid layout");
    assert!(layout.is_all_finite());

    let l3 = Layout3d::<f64>::init(4, 42, 1.0).expect("valid 3d layout");
    assert_eq!(l3.len(), 4);
    assert!(l3.is_all_finite());

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
fn facade_layout_and_typed_backends_smoke() {
    let graph = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)],
        None,
        false,
        false,
    )
    .expect("valid graph");
    let mut layout: Layout2d<f64> = PivotMds::<Layout2d<f64>>::new(&graph)
        .generate()
        .expect("pivot_mds");
    let stages = vec![
        TsNetStage::new().with_iterations(2).with_exaggeration(4.0),
        TsNetStage::new().with_iterations(2),
        TsNetStage::new().with_iterations(2),
    ];
    let mut solver = TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(
        &graph,
        TsNetConfig::pure_2hop().with_stages(stages.clone()),
    )
    .expect("tsnet pure_2hop");
    let status = solver.run(&mut layout).expect("tsnet run");
    assert!(status.is_finished());
    assert!(layout.is_all_finite());

    // Layout3d with PivotMds + explicit neighbor::BarnesHut and default ForceAtlas2Solver<Layout3d<f64>>.
    let mut l3: Layout3d<f64> = PivotMds::<Layout3d<f64>>::new(&graph)
        .generate()
        .expect("pivot_mds 3d");
    let mut tsnet_3d =
        TsNetSolver::<Layout3d<f64>, egraph::layout::neighbor::BarnesHut>::new_with_backend(
            &graph,
            TsNetConfig::pure_2hop().with_stages(stages),
            egraph::layout::neighbor::BarnesHut::new(0.5),
        )
        .expect("tsnet 3d bh");
    assert!(tsnet_3d.run(&mut l3).expect("tsnet 3d run").is_finished());
    assert!(l3.is_all_finite());

    let mut fa2_3d = ForceAtlas2Solver::<Layout3d<f64>>::new(
        &graph,
        ForceAtlas2Config::new().with_iterations(4).with_seed(42),
    )
    .expect("fa2 3d");
    assert!(fa2_3d.run(&mut l3).expect("fa2 3d run").is_finished());
    assert!(l3.is_all_finite());

    // pacmap_graph_with_backbone convenience wrapper + cooperative IterativeSolver execution.
    let pm_cfg = PacmapGraphConfig::new()
        .with_phase_iterations([2, 2, 2])
        .with_num_landmarks(4)
        .with_seed(42);
    let bb_cfg = CoarseBackboneConfig::new()
        .with_coarse_pivots(4)
        .with_coarse_t_max(8)
        .with_seed(42);
    let (mut pm_solver, mut bb_init, bb_res) =
        pacmap_graph_with_backbone::<f64, f64>(&graph, pm_cfg, &bb_cfg)
            .expect("pacmap_graph_with_backbone");
    assert_eq!(bb_res.fine_layout.len(), graph.num_nodes());
    assert!(pm_solver.run(&mut bb_init).expect("pm run").is_finished());
    assert!(bb_init.is_all_finite());
}

#[cfg(feature = "edge-routing")]
#[test]
fn facade_edge_routing_polylines_smoke() {
    let graph =
        CsrGraph::<()>::from_unweighted_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], false, false)
            .expect("cycle");
    let layout = Layout2d::<f64>::from_xy(vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 0.0, 1.0, 1.0])
        .expect("layout");
    let mut polylines =
        EdgePolylines::straight_line_from_graph(&graph, &layout).expect("polylines");
    let mut fdeb = FdebSolver::<Layout2d<f64>>::new(
        &polylines,
        FdebConfig::new().with_cycles(2).with_steps_per_cycle(3),
    )
    .expect("fdeb");
    let st = fdeb.run(&mut polylines).expect("fdeb run");
    assert!(st.is_finished());
    assert_eq!(polylines.num_edges(), 4);
    assert!(polylines.validate_finite().is_ok());
}

#[cfg(all(
    feature = "metrics",
    feature = "clustering",
    feature = "dataset-embedded"
))]
#[test]
fn facade_metrics_clustering_dataset_smoke() {
    let graph = egraph::dataset::dataset_karate();
    assert_eq!(graph.num_nodes(), 34);

    let mut layout: Layout2d<f64> = PivotMds::<Layout2d<f64>>::new(&graph)
        .generate()
        .expect("pivot_mds on karate");
    let louvain_res =
        egraph::clustering::louvain(&graph, &LouvainConfig::default()).expect("louvain");
    assert_eq!(louvain_res.labels.len(), 34);

    let cfg = MetricConfig::new()
        .with_sns_num_pivots(4)
        .with_np_num_samples(16)
        .with_np_seed(77);
    let report = evaluate_metrics(
        &graph,
        &layout,
        &[
            MetricTarget::ScaleNormalizedStress,
            MetricTarget::NeighborhoodPreservation2Hop,
            MetricTarget::SampledNp2,
        ],
        &cfg,
    )
    .expect("evaluate_metrics");
    assert!(report.scale_normalized_stress.is_some());
    assert!(report.neighborhood_preservation_2hop.is_some());
    assert!(report.sampled_np2.is_some());

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

    let caf_detail = egraph::metrics::cluster_area_faithfulness_detail(
        &layout,
        &louvain_res.labels,
        &egraph::metrics::ClusterAreaConfig::default(),
    )
    .expect("cluster_area_faithfulness_detail");
    assert_eq!(caf_detail.report.num_nodes, 34);
    assert_eq!(caf_detail.cluster_ids.len(), caf_detail.report.num_clusters);
    assert!((caf_detail.report.caf - caf_val).abs() < 1e-12);
    assert!(
        caf_detail.report.spatial_coherence.is_finite()
            && (0.0..=1.0).contains(&caf_detail.report.spatial_coherence)
    );

    egraph::linalg::normalize_layout_2d(&mut layout, 1.0, 42).expect("normalize_layout_2d");
    assert!(layout.is_all_finite());
}

#[cfg(all(feature = "linalg", feature = "layout"))]
#[test]
fn facade_resistance_curvature_manifold_smoke() {
    let mut edges = Vec::new();
    for u in 0..5u32 {
        for v in (u + 1)..5u32 {
            edges.push((u, v));
            edges.push((u + 5, v + 5));
        }
    }
    edges.push((4, 5));
    let barbell = CsrGraph::<f64>::from_edges(10, &edges, None, false, false).expect("barbell");
    let flow_hyp = egraph::linalg::resistance_curvature_flow(
        &barbell,
        &egraph::linalg::ResistanceCurvatureConfig::new()
            .with_steps(2)
            .with_eta(0.2),
    )
    .expect("barbell rcf");
    let length_graph = flow_hyp.to_length_graph().expect("length graph");
    let init_euclid: Layout2d<f64> = PivotMds::<Layout2d<f64>>::new(&length_graph)
        .generate()
        .expect("pivot_mds on length_graph");

    let mut ml_hyp = ManifoldLayout2d::<f64>::from_layout_scaled(
        ManifoldKind::HyperbolicPoincare,
        &init_euclid,
        0.7,
    )
    .expect("from_layout_scaled hyperbolic");
    let mut sgd_hyp =
        SgdSolver::<ManifoldLayout2d<f64>>::full(&length_graph, SgdScheduleConfig::new(8), 42)
            .expect("sgd full hyp");
    let st_hyp = sgd_hyp.run(&mut ml_hyp).expect("run hyp");
    assert!(st_hyp.is_finished());
    assert!(ml_hyp.as_layout().is_all_finite());

    let mut ml_sph = ManifoldLayout2d::<f64>::random_init(ManifoldKind::Spherical, 10, 0.5, 42)
        .expect("random_init spherical");
    let mut sgd_sph = SgdSolver::<ManifoldLayout2d<f64>>::full(
        &length_graph,
        SgdScheduleConfig::new(8).with_step_scale(0.1),
        42,
    )
    .expect("sgd full sph");
    let st_sph = sgd_sph.run(&mut ml_sph).expect("run sph");
    assert!(st_sph.is_finished());
    assert!(ml_sph.as_layout().is_all_finite());

    let mut t_edges = Vec::new();
    for r in 0..8u32 {
        for c in 0..8u32 {
            let u = r * 8 + c;
            t_edges.push((u, r * 8 + ((c + 1) % 8)));
            t_edges.push((u, ((r + 1) % 8) * 8 + c));
        }
    }
    let torus = CsrGraph::<f64>::from_edges(64, &t_edges, None, false, true).expect("torus");
    let flow_tor = egraph::linalg::resistance_curvature_flow(
        &torus,
        &egraph::linalg::ResistanceCurvatureConfig::new().with_steps(1),
    )
    .expect("torus rcf");
    assert_eq!(
        flow_tor.suggested_geometry,
        egraph::linalg::SuggestedGeometry::Torus2d
    );
    assert_eq!(
        flow_tor.suggested_manifold_kind(),
        Some(ManifoldKind::Torus2d)
    );
    let mut ml_tor = ManifoldLayout2d::<f64>::random_init(ManifoldKind::Torus2d, 64, 0.4, 42)
        .expect("random_init torus");
    let mut sgd_tor = SgdSolver::<ManifoldLayout2d<f64>>::full(
        &flow_tor.to_length_graph().expect("torus length graph"),
        SgdScheduleConfig::new(8),
        42,
    )
    .expect("sgd full tor");
    let st_tor = sgd_tor.run(&mut ml_tor).expect("run tor");
    assert!(st_tor.is_finished());
    assert!(ml_tor.as_layout().is_all_finite());
}

#[cfg(all(
    feature = "layout",
    feature = "baselines",
    feature = "metrics",
    feature = "eval-baselines",
    feature = "clustering",
    feature = "dataset-embedded"
))]
#[test]
fn facade_baselines_smoke() {
    let graph = egraph::dataset::dataset_karate();
    let leiden_res = egraph::clustering::leiden(&graph, &LeidenConfig::default()).expect("leiden");
    assert_eq!(leiden_res.labels.len(), graph.num_nodes());

    let mut layout = Layout2d::<f64>::zeros(graph.num_nodes());
    let mut linlog =
        LinLogSolver::<Layout2d<f64>>::new(&graph, LinLogConfig::new().with_iterations(5))
            .expect("linlog");
    assert!(linlog.run(&mut layout).expect("linlog run").is_finished());

    let mut fr = FruchtermanReingoldSolver::<Layout2d<f64>>::new(
        &graph,
        FruchtermanReingoldConfig::new().with_iterations(5),
    )
    .expect("fr");
    assert!(fr.run(&mut layout).expect("fr run").is_finished());

    let backbone =
        egraph::layout::extract_quadrilateral_backbone(&graph, &BackboneConfig::default())
            .expect("extract_quadrilateral_backbone");
    assert_eq!(backbone.simmelian_weights.len(), graph.num_edges());
    let mut bb_solver =
        BackboneSolver::<f64>::new(&graph, BackboneConfig::new().with_iterations(5))
            .expect("backbone solver");
    assert!(bb_solver.run(&mut layout).expect("bb run").is_finished());

    let mut refine = ClusterRefineSolver::<f64>::new(
        &leiden_res.labels,
        ClusterRefineConfig::new().with_max_iterations(2),
    )
    .expect("cluster_refine");
    assert!(refine.run(&mut layout).expect("refine run").is_finished());

    let boxes = egraph::spatial::squarified_treemap(&[1.0, 2.0, 3.0], 0.0, 0.0, 100.0, 100.0)
        .expect("treemap");
    assert_eq!(boxes.len(), 3);
    let mut gib = GroupInABoxSolver::<f64>::new(
        &graph,
        &leiden_res.labels,
        GroupInABoxConfig::new().with_intra_iterations(5),
    )
    .expect("gib");
    assert!(gib.run(&mut layout).expect("gib run").is_finished());

    let radii = vec![0.25_f64; graph.num_nodes()];
    let mut gtree =
        GTreeOverlapRemoval::<f64>::from_radii(&radii, GTreeConfig::default()).expect("gtree");
    assert!(gtree.run(&mut layout).expect("gtree run").is_finished());
    assert_eq!(gtree.last_overlap_count(), 0);

    let mut forbid = ForbidOverlapRemoval::<f64>::from_radii(
        &radii,
        ForbidConfig::new()
            .with_max_scale_iters(5)
            .with_sgd_iterations(5),
    )
    .expect("forbid");
    assert!(forbid.run(&mut layout).expect("forbid run").is_finished());
    assert_eq!(forbid.last_overlap_count(), 0);

    let agr = egraph::metrics::partition_agreement(&leiden_res.labels, &leiden_res.labels)
        .expect("agreement");
    assert!((agr.ari - 1.0).abs() < 1e-12);
    let dsc = egraph::metrics::distance_consistency(&layout, &leiden_res.labels).expect("dsc");
    assert!(dsc.is_finite());

    let extents = NodeExtents2d::from_radii(&radii).expect("extents");
    let inputs = MetricInputs::new()
        .with_labels(Some(&leiden_res.labels))
        .with_sprawlter_extents(Some(extents));
    let report = evaluate_metrics_with_inputs(
        &graph,
        &layout,
        &[MetricTarget::DistanceConsistency, MetricTarget::SprawlterNn],
        &MetricConfig::default(),
        &inputs,
    )
    .expect("eval-baselines dispatcher");
    let baselines: &ClusterBaselineReport = report
        .baseline_metrics()
        .expect("baseline_metrics populated");
    assert!(baselines.distance_consistency.is_some());
    assert!(baselines.sprawlter_nn.is_some());
}

#[cfg(all(feature = "linalg", feature = "layout"))]
#[test]
fn facade_hde_and_embedding_solvers_smoke() {
    let graph = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)],
        None,
        false,
        false,
    )
    .expect("valid graph");

    let hde_cfg = HdeConfig::new().with_dimensions(4);
    let hde_2d: Layout2d<f64> = Hde::<Layout2d<f64>>::from_config(&graph, hde_cfg.clone())
        .generate()
        .expect("hde");
    assert_eq!(hde_2d.len(), 6);
    assert!(hde_2d.is_all_finite());
    let hde_piv: Layout2d<f64> = Hde::<Layout2d<f64>>::from_config(&graph, hde_cfg.clone())
        .with_pivots(&[0, 2, 4])
        .generate()
        .expect("hde with pivots");
    assert!(hde_piv.is_all_finite());
    let nd_1: LayoutNd<f64> =
        Hde::<LayoutNd<f64>>::from_config(&graph, hde_cfg.with_output_dims(Some(3)))
            .generate()
            .expect("hde nd");
    assert_eq!(nd_1.dims(), 3);

    let mut layout = hde_2d;
    let mut ortmann = SgdSolver::<Layout2d<f64>>::ortmann_sparse_stress(
        &graph,
        egraph::layout::OrtmannSparseStressConfig::new()
            .with_num_pivots(3)
            .with_pivot_strategy(egraph::layout::OrtmannPivotStrategy::MaxMinSp)
            .with_iterations(5),
    )
    .expect("ortmann");
    assert!(ortmann.run(&mut layout).expect("ortmann run").is_finished());

    let mut lv =
        LargeVisSolver::<Layout2d<f64>>::new(&graph, LargeVisConfig::new().with_iterations(4))
            .expect("largevis");
    assert!(lv.run(&mut layout).expect("largevis run").is_finished());

    let mut dr =
        DrGraphSolver::<Layout2d<f64>>::new(&graph, DrGraphConfig::new().with_iterations(4))
            .expect("drgraph");
    assert!(dr.run(&mut layout).expect("drgraph run").is_finished());

    let mut cne = GraphCneSolver::<Layout2d<f64>>::new(
        &graph,
        GraphCneConfig::new()
            .with_objective(egraph::layout::GraphCneObjective::InfoNce)
            .with_iterations(4),
    )
    .expect("graph_cne");
    assert!(cne.run(&mut layout).expect("graph_cne run").is_finished());
}

#[cfg(all(
    feature = "layout",
    feature = "baselines",
    feature = "dataset-embedded"
))]
#[test]
fn facade_multilevel_and_stress_baselines_smoke() {
    let graph = egraph::dataset::dataset_karate();
    let mut layout = Layout2d::<f64>::zeros(graph.num_nodes());

    let mut maxent =
        MaxentStressSolver::<Layout2d<f64>, egraph::layout::stress::Exact>::new_with_backend(
            &graph,
            MaxentStressConfig::new()
                .with_stages(2)
                .with_steps_per_stage(3),
            egraph::layout::stress::Exact::new(),
        )
        .expect("maxent");
    assert!(maxent.run(&mut layout).expect("maxent run").is_finished());
    assert!(maxent.current_alpha().is_finite());
    assert!(maxent.effective_q().is_finite());

    let mut grip = GripSolver::<f64>::new(
        &graph,
        GripConfig::new().with_rounds_min(1).with_rounds_max(2),
    )
    .expect("grip");
    assert!(grip.run(&mut layout).expect("grip run").is_finished());

    let mut gem =
        GemSolver::<f64>::new(&graph, GemConfig::new().with_max_iterations(Some(4))).expect("gem");
    assert!(gem.run(&mut layout).expect("gem run").is_finished());

    let mut sel = SelLayoutSolver::<f64>::new(
        &graph,
        SelConfig::new()
            .with_mapping(egraph::layout::SelWeightMapping::Sigmoid)
            .with_layout_backend(egraph::layout::SelLayoutBackend::Fr)
            .with_iterations(4),
    )
    .expect("sel");
    assert_eq!(sel.modulated_weights().len(), graph.num_edges());
    assert!(sel.se_target().is_finite());
    assert!(sel.run(&mut layout).expect("sel run").is_finished());

    let mut sfc = SfcLayout::<f64>::from_config(
        &graph,
        SfcConfig::new().with_curve(egraph::layout::SfcCurveKind::Hilbert),
    );
    let sfc_layout = sfc.generate().expect("sfc generate");
    assert_eq!(sfc.vertex_order().len(), graph.num_nodes());
    assert_eq!(sfc_layout.len(), graph.num_nodes());

    let radii = vec![0.25_f64; graph.num_nodes()];
    let mut pfsp = PfsPrimeOverlapRemoval::<f64>::from_radii(&radii, PfsPrimeConfig::default())
        .expect("pfs_prime");
    pfsp.apply(&mut layout).expect("pfs_prime apply");
    assert_eq!(pfsp.last_overlap_count(), 0);
}
