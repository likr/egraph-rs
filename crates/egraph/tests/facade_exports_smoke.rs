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
    assert_eq!(layout.clone().len(), 4);

    let mut l3 = Layout3d::<f64>::init(4, 42, 1.0).expect("layout3d init");
    l3.center_and_normalize_std().expect("normalize 3d");
    assert!(l3.validate_finite().is_ok());

    let mut nd = LayoutNd::<f64>::init(4, 3, 42, 1.0).expect("layout_nd init");
    nd.center_and_normalize_std().expect("normalize nd");
    assert!(nd.validate_finite().is_ok());

    let ml =
        ManifoldLayout2d::<f64>::init(ManifoldKind::Torus2d, 4, 42, 0.5).expect("manifold init");
    assert_eq!(ml.clone().len(), 4);
}

#[cfg(feature = "spatial")]
#[test]
fn spatial_screened_poisson_and_index_smoke() {
    let cfg = egraph::spatial::ScreenedPoissonConfig::default();
    assert_eq!(cfg.grid_size, 512);
    let mut field = egraph::spatial::ScreenedPoissonField2d::<f64>::new(cfg).expect("field");
    let layout = Layout2d::<f64>::init(8, 7, 1.0).expect("layout");
    let mut out = egraph::spatial::ScreenedPoissonOutput::<f64>::default();
    field.evaluate(&layout, &mut out).expect("evaluate");
    assert!(field.density_grid().is_some());
    assert_eq!(field.density_grid_dim(), 512);

    let tree = egraph::spatial::StaticKdTree::<2, f64>::from_layout(&layout).expect("kdtree");
    let index: &dyn NearestNeighborIndex = &tree;
    assert_eq!(index.dims(), 2);
    assert_eq!(index.len(), 8);
}

#[cfg(feature = "linalg")]
#[test]
fn linalg_primitives_smoke() {
    let graph =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], None, false, false)
            .expect("graph");
    let mut dws = egraph::linalg::DijkstraWorkspace::new(4);
    let d0 = egraph::linalg::dijkstra_single_source(&graph, 0).expect("dijkstra_single_source");
    assert_eq!(d0.len(), 4);
    let mut apsp = vec![0.0_f64; 16];
    egraph::linalg::all_pairs_dijkstra_into(&graph, &mut apsp, &mut dws)
        .expect("all_pairs_dijkstra_into");
    let apsp2 = egraph::linalg::all_pairs_dijkstra(&graph).expect("all_pairs_dijkstra");
    assert_eq!(apsp, apsp2);

    let _bws = egraph::linalg::BfsWorkspace::new(4);
    let lap = egraph::linalg::random_walk_laplacian(&graph);
    assert_eq!(lap.num_nodes(), 4);
    let (sym, _inv_sqrt_deg) = egraph::linalg::symmetric_normalized_laplacian(&graph);
    let shifted = egraph::linalg::shifted_normalized_laplacian(&sym, 2.0).expect("shifted");
    assert_eq!(shifted.num_nodes(), 4);
    let stats = egraph::linalg::spectral_moment_stats(&graph);
    assert!(stats.sigma_lambda.is_finite());
    let dog_terms = egraph::linalg::build_regularized_dog_terms(0.5, 0.1).expect("dog terms");
    assert_eq!(dog_terms.len(), 2);

    let eig: egraph::linalg::EigenResult =
        egraph::linalg::smallest_sym_laplacian_eigenpairs(&graph, 2, 1e-6, 16, 42)
            .expect("lanczos");
    assert_eq!(eig.lambdas.len(), 2);
    let (jac_vals, _jac_vecs) =
        egraph::linalg::symmetric_eigen_jacobi(&[2.0, -1.0, -1.0, 2.0], 2, 32, 1e-10)
            .expect("jacobi");
    assert_eq!(jac_vals.len(), 2);

    let cmds: Layout2d<f64> = ClassicalMds::<Layout2d<f64>>::new(&graph)
        .generate()
        .expect("classical_mds");
    assert_eq!(cmds.len(), 4);
    let cmds_d: Layout2d<f64> = ClassicalMds::<Layout2d<f64>>::from_distances(&apsp, 4)
        .generate()
        .expect("classical_mds_from_distances");
    assert_eq!(cmds_d.len(), 4);
    let pmds: Layout2d<f64> = PivotMds::<Layout2d<f64>>::new(&graph)
        .with_pivots(&[0, 2])
        .generate()
        .expect("pivot_mds_with_pivots");
    assert_eq!(pmds.len(), 4);
    let pmds_d: Layout2d<f64> =
        PivotMds::<Layout2d<f64>>::from_distances(&[0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0], 4, 2)
            .generate()
            .expect("pivot_mds_from_distances");
    assert_eq!(pmds_d.len(), 4);
    let kspec: Layout3d<f64> = KorenSpectral::<Layout3d<f64>>::new(&graph)
        .generate()
        .expect("koren_spectral 3d");
    assert_eq!(kspec.len(), 4);
    let rd: Layout2d<f64> = RdMds::<Layout2d<f64>>::new(&graph)
        .generate()
        .expect("rd_mds");
    assert_eq!(rd.len(), 4);

    let er = egraph::linalg::compute_effective_resistances(
        &graph,
        &egraph::linalg::EffectiveResistanceConfig::default(),
    )
    .expect("er");
    let sug = egraph::linalg::classify_geometry_from_curvature_with_sampling(
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
    let pairs = egraph::neighborhood::extract_sparse_1hop_2hop_pairs_with_mode(
        &graph,
        4,
        egraph::neighborhood::TwoHopSamplingMode::TwoStepRandomWalk,
        None,
        42,
    )
    .expect("pairs");
    assert!(!pairs.u1().is_empty());

    let probs = egraph::neighborhood::graph_12hop_row_normalized_probabilities_with_options(
        &graph,
        0.2,
        Some(4),
        0.0,
        egraph::neighborhood::TwoHopSamplingMode::TwoStepRandomWalk,
        None,
        42,
    )
    .expect("probs");
    assert_eq!(probs.num_nodes(), 5);

    let (rhgh, _stats): (
        egraph::neighborhood::RhghLandmarks,
        egraph::neighborhood::RhghLandmarkStats,
    ) = egraph::neighborhood::compute_pivoted_rhgh_landmarks_result(
        &graph,
        &egraph::neighborhood::RhghLandmarkConfig::default(),
    )
    .expect("rhgh");
    assert_eq!(rhgh.num_nodes(), 5);
}

#[cfg(feature = "clustering")]
#[test]
fn clustering_result_types_and_traits_smoke() {
    let graph =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], None, false, false)
            .expect("graph");
    let lp: egraph::clustering::LabelPropagationResult = egraph::clustering::label_propagation(
        &graph,
        &egraph::clustering::LabelPropagationConfig::default(),
    )
    .expect("lp");
    assert_eq!(lp.labels.len(), 4);
    let sc: egraph::clustering::SpectralClusteringResult = egraph::clustering::spectral_clustering(
        &graph,
        &egraph::clustering::SpectralClusteringConfig::default(),
    )
    .expect("sc");
    assert_eq!(sc.labels.len(), 4);

    let detector = Louvain::new(LouvainConfig::default());
    let detected: ClusterResult = detector.detect(&graph).expect("louvain detect");
    assert_eq!(detected.labels.len(), 4);
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
        egraph::dataset::generate_torus_grid(3, 3, false)
            .expect("torus")
            .num_nodes(),
        9
    );
    assert_eq!(
        egraph::dataset::generate_layered_dag(3, 2, 0.5, 42)
            .expect("dag")
            .num_nodes(),
        6
    );
    assert_eq!(
        egraph::dataset::generate_sbm_graph(&[3, 3], 0.8, 0.1, 42)
            .expect("sbm")
            .0
            .num_nodes(),
        6
    );
    assert_eq!(
        egraph::dataset::generate_weighted_graph(5, 2, 1.0, 3.0, 42)
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
    let passes = egraph::layout::declutter_twins_2d(&mut layout, &cfg).expect("declutter");
    assert!(passes >= 1);

    let mut declutterer = TwinDeclutterer::<f64>::new(cfg).expect("declutterer");
    declutterer.apply(&mut layout).expect("apply");

    let dag_unweighted =
        CsrGraph::<()>::from_unweighted_edges(3, &[(0, 1), (1, 2), (0, 2)], true, false)
            .expect("unweighted dag");
    let reduced_u: CsrGraph<()> =
        egraph::layout::transitive_reduction(&dag_unweighted).expect("transitive_reduction");
    assert_eq!(reduced_u.num_edges(), 2);

    let dag_weighted = CsrGraph::<f64>::from_edges(
        3,
        &[(0, 1), (1, 2), (0, 2)],
        Some(&[1.5, 2.5, 9.0]),
        true,
        false,
    )
    .expect("weighted dag");
    let reduced_w: CsrGraph<f64> = egraph::layout::transitive_reduction_weighted(&dag_weighted)
        .expect("transitive_reduction_weighted");
    assert_eq!(reduced_w.num_edges(), 2);
    assert_eq!(reduced_w.weights().expect("weights"), &[1.5, 2.5]);

    let _ = std::any::type_name::<egraph::layout::FieldFftStats>();
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
    let extents = NodeExtents2d::uniform(4, 0.1, 0.1).expect("extents");
    let inputs = MetricInputs::new()
        .with_labels(Some(&labels))
        .with_original_layout(Some(&orig))
        .with_sprawlter_extents(Some(extents));
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

    let nig: egraph::metrics::AdjustedPatchNigDetail =
        egraph::metrics::adjusted_patch_nig_detail(&layout, &labels, 4, 32, 0.05, 42)
            .expect("nig detail");
    assert!(nig.nig.is_finite());

    let lms = vec![0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 2.0];
    let rta =
        egraph::metrics::landmark_stratified_rta_from_landmarks_f64(&layout, &[0, 1], &lms, 16, 42)
            .expect("rta f64");
    assert!(rta.rta.is_finite());

    let mut scratch = egraph::metrics::BfsScratch::new(4);
    let mut dists = vec![0.0_f64; 16];
    egraph::metrics::all_pairs_bfs_into(&graph, 1.0, &mut dists, &mut scratch).expect("bfs_into");
    let stress: egraph::metrics::StressEvaluation =
        egraph::metrics::scale_normalized_stress_detail(&graph, &layout).expect("stress detail");
    assert!(stress.normalized_stress.is_finite());
    let _: egraph::metrics::CrossingSummary =
        egraph::metrics::crossing_summary(&graph, &layout).expect("crossing_summary");
    let _: egraph::metrics::PrismDistortion =
        egraph::metrics::prism_distortion(&orig, &layout).expect("prism_distortion");
    let _ = egraph::metrics::crossing_angle(&graph, &layout).expect("crossing_angle");
    let _ = egraph::metrics::edge_angle(1.0, 0.0, 0.0, 1.0).expect("edge_angle");
    let _ =
        egraph::metrics::ideal_edge_lengths_from_distances(&graph, &layout, &dists).expect("iel");
    let _ = egraph::metrics::sampled_scale_normalized_stress_detail(&graph, &layout, 2, 42)
        .expect("sampled_sns");
    let _ = egraph::metrics::scale_normalized_stress_from_distances_detail(&layout, &dists, None)
        .expect("sns_dist");
    let _ = egraph::metrics::scale_normalized_stress_from_pairs(&layout, &[(0, 1, 1.0, 1.0)])
        .expect("sns_pairs");

    let torus_view = TorusView::new(&layout, 1.0, 1.0).expect("torus view");
    let _ = egraph::metrics::crossing_edges_torus_from_edges(&[(0, 1), (2, 3)], &torus_view)
        .expect("torus");
    let _ = std::any::type_name::<egraph::metrics::CrossingPair>();
    let _ = std::any::type_name::<egraph::metrics::TorusCrossing>();
    let _ = egraph::metrics::crossing_number_from_pairs(&[]);
    let _ = egraph::metrics::crossing_angle_from_pairs(&[]);
}

#[cfg(all(feature = "metrics", feature = "eval-baselines"))]
#[test]
fn metrics_eval_baselines_histogram_smoke() {
    let layout = Layout2d::<f64>::init(8, 12, 1.0).expect("layout");
    let labels = [0_u32, 0, 0, 0, 1, 1, 1, 1];
    let hdm = egraph::metrics::histogram_density_measure(&layout, &labels, 8).expect("hdm");
    assert!(hdm.is_finite());
}
