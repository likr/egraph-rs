use egraph::prelude::*;

#[test]
fn core_parity_and_no_default_features_smoke() {
    let graph =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], None, false, false)
            .expect("valid cycle graph");
    assert_eq!(graph.edges().len(), 4);
    assert!(graph.has_edge(0, 1));
    assert_eq!(graph.edge_weight(0, 1), None);
    assert!(graph.is_connected());

    let layout = Layout2d::<f64>::init(4, 42, 1.0).expect("layout init");
    assert_eq!(layout.copy().len(), 4);

    let mut nd = LayoutNd::<f64>::init(4, 3, 42, 1.0).expect("layout_nd init");
    nd.center_and_normalize_std().expect("normalize nd");
    assert!(nd.validate_finite().is_ok());

    let ml =
        ManifoldLayout2d::<f64>::init(ManifoldKind::Torus2d, 4, 42, 0.5).expect("manifold init");
    assert_eq!(ml.copy().len(), 4);
}

#[cfg(feature = "spatial")]
#[test]
fn spatial_screened_poisson_smoke() {
    let cfg = ScreenedPoissonConfig::default();
    assert_eq!(cfg.grid_size, 512);
    let mut field = ScreenedPoissonField2d::<f64>::new(cfg).expect("field");
    let layout = Layout2d::<f64>::init(8, 7, 1.0).expect("layout");
    let mut out = ScreenedPoissonOutput::<f64>::default();
    field.evaluate(&layout, &mut out).expect("evaluate");
    assert!(field.density_grid().is_some());
    assert_eq!(field.density_grid_dim(), 512);
}

#[cfg(feature = "linalg")]
#[test]
fn linalg_primitives_smoke() {
    let graph =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], None, false, false)
            .expect("graph");
    let mut dws = DijkstraWorkspace::new(4);
    let d0 = dijkstra_single_source(&graph, 0).expect("dijkstra_single_source");
    assert_eq!(d0.len(), 4);
    let mut apsp = vec![0.0_f64; 16];
    all_pairs_dijkstra_into(&graph, &mut apsp, &mut dws).expect("all_pairs_dijkstra_into");
    let apsp2 = all_pairs_dijkstra(&graph).expect("all_pairs_dijkstra");
    assert_eq!(apsp, apsp2);

    let _bws = BfsWorkspace::new(4);
    let lap = random_walk_laplacian(&graph);
    assert_eq!(lap.num_nodes(), 4);
    let (sym, _inv_sqrt_deg) = symmetric_normalized_laplacian(&graph);
    let shifted = shifted_normalized_laplacian(&sym, 2.0).expect("shifted");
    assert_eq!(shifted.num_nodes(), 4);
    let stats = spectral_moment_stats(&graph);
    assert!(stats.sigma_lambda.is_finite());
    let dog_terms = build_regularized_dog_terms(0.5, 0.1).expect("dog terms");
    assert_eq!(dog_terms.len(), 2);

    let eig: EigenResult =
        smallest_sym_laplacian_eigenpairs(&graph, 2, 1e-6, 16, 42).expect("lanczos");
    assert_eq!(eig.lambdas.len(), 2);
    let (jac_vals, _jac_vecs) =
        symmetric_eigen_jacobi(&[2.0, -1.0, -1.0, 2.0], 2, 32, 1e-10).expect("jacobi");
    assert_eq!(jac_vals.len(), 2);

    let cmds = classical_mds::<f64, f64>(&graph, 2).expect("classical_mds");
    assert_eq!(cmds.dims(), 2);
    let cmds_d =
        classical_mds_from_distances::<f64>(&apsp, 4, 2).expect("classical_mds_from_distances");
    assert_eq!(cmds_d.dims(), 2);
    let pmds =
        pivot_mds_with_pivots::<f64, f64>(&graph, &[0, 2], None, 2).expect("pivot_mds_with_pivots");
    assert_eq!(pmds.len(), 4);
    let pmds_d =
        pivot_mds_from_distances::<f64>(&[0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0], 4, 2, None, 42)
            .expect("pivot_mds_from_distances");
    assert_eq!(pmds_d.len(), 4);

    let er =
        compute_effective_resistances(&graph, &EffectiveResistanceConfig::default()).expect("er");
    let sug = classify_geometry_from_curvature_with_sampling(
        er.num_edges,
        0.0,
        0.01,
        0.0,
        0.0,
        -0.2,
        0.2,
        0.05,
        er.used_exact,
        128,
        false,
    );
    let _ = format!("{:?}", sug);
}

#[cfg(feature = "neighborhood")]
#[test]
fn neighborhood_exports_smoke() {
    let graph = CsrGraph::<f64>::from_edges(
        5,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
        None,
        false,
        false,
    )
    .expect("graph");
    let pairs = extract_sparse_1hop_2hop_pairs_with_mode(
        &graph,
        4,
        TwoHopSamplingMode::TwoStepRandomWalk,
        None,
        42,
    )
    .expect("pairs");
    assert!(!pairs.u1.is_empty());

    let probs = graph_12hop_row_normalized_probabilities_with_options(
        &graph,
        0.2,
        Some(4),
        0.0,
        TwoHopSamplingMode::TwoStepRandomWalk,
        None,
        42,
    )
    .expect("probs");
    assert_eq!(probs.num_nodes(), 5);

    let (rhgh, _stats): (RhghLandmarks, RhghLandmarkStats) =
        compute_pivoted_rhgh_landmarks_result(&graph, &RhghLandmarkConfig::default())
            .expect("rhgh");
    assert_eq!(rhgh.num_nodes, 5);
}

#[cfg(feature = "clustering")]
#[test]
fn clustering_result_types_smoke() {
    let graph =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], None, false, false)
            .expect("graph");
    let lp: LabelPropagationResult =
        label_propagation(&graph, &LabelPropagationConfig::default()).expect("lp");
    assert_eq!(lp.labels.len(), 4);
    let sc: SpectralClusteringResult =
        spectral_clustering(&graph, &SpectralClusteringConfig::default()).expect("sc");
    assert_eq!(sc.labels.len(), 4);
}

#[cfg(feature = "dataset")]
#[test]
fn dataset_catalog_and_synthetic_smoke() {
    let names = list_datasets();
    assert!(!names.is_empty());
    let catalog: &[DatasetInfo] = dataset_catalog();
    assert_eq!(catalog.len(), names.len());
    assert!(dataset_info(names[0]).is_some());

    assert_eq!(
        generate_torus_grid(3, 3, false).expect("torus").num_nodes(),
        9
    );
    assert_eq!(
        generate_layered_dag(3, 2, 0.5, 42)
            .expect("dag")
            .num_nodes(),
        6
    );
    assert_eq!(
        generate_sbm_graph(&[3, 3], 0.8, 0.1, 42)
            .expect("sbm")
            .0
            .num_nodes(),
        6
    );
    assert_eq!(
        generate_weighted_graph(5, 2, 1.0, 3.0, 42)
            .expect("weighted")
            .num_nodes(),
        5
    );
}

#[cfg(feature = "layout")]
#[test]
fn layout_twin_declutter_and_transitive_reduction_smoke() {
    let mut layout = Layout2d::<f64>::from_xy(vec![0.0, 0.0, 1.0, 1.0], vec![0.0, 0.0, 1.0, 1.0])
        .expect("layout");
    let cfg = TwinDeclutterConfig::default();
    let passes = declutter_twins_2d(&mut layout, &cfg).expect("declutter");
    assert!(passes >= 1);

    let mut declutterer = TwinDeclutterer::<f64>::new(cfg).expect("declutterer");
    let st = declutterer.step(&mut layout).expect("step");
    assert!(st.is_finished());

    let dag =
        CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2), (0, 2)], None, true, false).expect("dag");
    let reduced = transitive_reduction(&dag).expect("transitive_reduction");
    assert_eq!(reduced.num_edges(), 2);

    let _ = std::any::type_name::<FieldFftStats>();
}

#[cfg(feature = "metrics")]
#[test]
fn metrics_exports_and_inputs_smoke() {
    let graph =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], None, false, false)
            .expect("graph");
    let layout = Layout2d::<f64>::from_xy(vec![0.0, 1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0])
        .expect("layout");
    let orig = Layout2d::<f64>::init(4, 1, 1.0).expect("orig");
    let labels = [0_u32, 0, 1, 1];
    let inputs = MetricInputs {
        labels: Some(&labels),
        original_layout: Some(&orig),
        ..MetricInputs::default()
    };
    let report = evaluate_metrics_with_inputs(
        &graph,
        &layout,
        &[
            MetricTarget::CrossingEdges,
            MetricTarget::PrismDistortion,
            MetricTarget::AdjustedPatchNig,
        ],
        &MetricConfig::default(),
        &inputs,
    )
    .expect("evaluate_metrics_with_inputs");
    assert!(report.crossing_edges.is_some());
    assert!(report.prism_sigma_dist.is_some());
    assert!(report.adjusted_patch_nig.is_some());

    let nig: AdjustedPatchNigDetail =
        adjusted_patch_nig_detail(&layout, &labels, 4, 32, 0.05, 42).expect("nig detail");
    assert!(nig.nig.is_finite());

    let lms = vec![0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 2.0];
    let rta = landmark_stratified_rta_from_landmarks_f64(&layout, &[0, 1], &lms, 16, 42)
        .expect("rta f64");
    assert!(rta.rta.is_finite());

    let mut scratch = BfsScratch::new(4);
    let mut dists = vec![0.0_f64; 16];
    all_pairs_bfs_into(&graph, 1.0, &mut dists, &mut scratch).expect("bfs_into");
    let stress: StressEvaluation =
        scale_normalized_stress_detail(&graph, &layout).expect("stress detail");
    assert!(stress.normalized_stress.is_finite());
    let _: CrossingSummary = crossing_summary(&graph, &layout).expect("crossing_summary");
    let _: PrismDistortion = prism_distortion(&orig, &layout).expect("prism_distortion");
    let _ = crossing_angle(&graph, &layout).expect("crossing_angle");
    let _ = edge_angle(1.0, 0.0, 0.0, 1.0).expect("edge_angle");
    let _ = ideal_edge_lengths_from_distances(&graph, &layout, &dists).expect("iel");
    let _ = sampled_scale_normalized_stress_detail(&graph, &layout, 2, 42).expect("sampled_sns");
    let _ = scale_normalized_stress_from_distances_detail(&layout, &dists, None).expect("sns_dist");
    let _ = scale_normalized_stress_from_pairs(&layout, &[(0, 1, 1.0, 1.0)]).expect("sns_pairs");
    let _ = crossing_edges_torus_from_edges(&[(0, 1), (2, 3)], &layout).expect("torus");
    let _ = std::any::type_name::<CrossingPair>();
    let _ = std::any::type_name::<TorusCrossing>();
    let _ = crossing_number_from_pairs(&[]);
    let _ = crossing_angle_from_pairs(&[]);
}

#[cfg(all(feature = "metrics", feature = "eval-baselines"))]
#[test]
fn metrics_eval_baselines_histogram_smoke() {
    let layout = Layout2d::<f64>::init(8, 12, 1.0).expect("layout");
    let labels = [0_u32, 0, 0, 0, 1, 1, 1, 1];
    let hdm = histogram_density_measure(&layout, &labels, 8).expect("hdm");
    assert!(hdm.is_finite());
}
