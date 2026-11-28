use egraph::prelude::*;

fn grid_2x3() -> CsrGraph<f64> {
    CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)],
        None,
        false,
        false,
    )
    .expect("valid 2x3 grid")
}

#[test]
fn core_manifold_metric_lorentz_and_torus_smoke() {
    const {
        assert!(egraph::core::POINCARE_EPS_F64 > 0.0 && egraph::core::POINCARE_EPS_F32 > 0.0);
    }
    let metric = ManifoldMetric::default();
    assert_eq!((metric.scale, metric.aspect), (1.0, 1.0));

    let g = grid_2x3();
    let mut lor = ManifoldLayout2d::<f64>::init(ManifoldKind::HyperbolicLorentz, 6, 42, 1.5)
        .expect("lorentz");
    lor.set_scale(2.0).expect("set_scale");
    assert_eq!(lor.scale(), 2.0);
    lor.rescale_geodesic(1.0).expect("rescale_geodesic");
    assert_eq!(lor.scale(), 1.0);
    let poincare = lor.to_poincare_layout().expect("to_poincare_layout");
    assert_eq!(poincare.kind(), ManifoldKind::HyperbolicPoincare);
    for i in 0..poincare.len() {
        assert!(poincare.x()[i].hypot(poincare.y()[i]) < 1.0);
    }

    let mut torus =
        ManifoldLayout2d::<f64>::torus_spectral_phase_init(&g, 42).expect("spectral_phase_init");
    assert_eq!(torus.kind(), ManifoldKind::Torus2d);
    torus.set_torus_extents(2.0, 1.0).expect("torus_extents");
    assert_eq!((torus.aspect(), torus.scale()), (2.0, 1.0));
    let panned = torus.torus_auto_pan(&g).expect("torus_auto_pan");
    assert_eq!(panned.metric(), torus.metric());
}

#[cfg(feature = "spatial")]
#[test]
fn spatial_non_euclidean_crossings_smoke() {
    let g4 = CsrGraph::<f64>::from_edges(4, &[(0, 2), (1, 3)], None, false, false).expect("g4");
    let mut torus = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.1, 0.1, 0.4, 0.4],
        vec![0.1, 0.4, 0.4, 0.1],
    )
    .expect("torus");
    torus.set_torus_extents(2.0, 1.0).expect("extents");
    assert_eq!((torus.aspect(), torus.scale()), (2.0, 1.0));
    assert_eq!(
        egraph::spatial::torus_crossing_number(&g4, &torus).expect("torus_cnt"),
        1
    );
    let t_pairs: Vec<egraph::spatial::TorusCrossing> =
        egraph::spatial::torus_crossing_edges(&g4, &torus).expect("torus_edges");
    assert_eq!(t_pairs.len(), 1);
    assert_eq!(
        egraph::spatial::torus_crossing_edges_from_edges(&[(0, 2), (1, 3)], &torus)
            .expect("from_edges")
            .len(),
        1
    );
    assert!(
        !egraph::spatial::torus_edge_segments(0.1, 0.1, 0.9, 0.9)
            .expect("segs")
            .is_empty()
    );
    assert!(
        !egraph::spatial::torus_layout_edge_segments(&g4, &torus)
            .expect("layout_segs")
            .is_empty()
    );

    // Explicit TorusView wrapper over Layout2d.
    let euclid_torus = Layout2d::<f64>::from_xy(vec![0.1, 0.1, 0.4, 0.4], vec![0.1, 0.4, 0.4, 0.1])
        .expect("euclid");
    let torus_view = TorusView::new(&euclid_torus, 2.0, 1.0).expect("TorusView");
    assert_eq!(
        egraph::spatial::torus_crossing_number(&g4, &torus_view).expect("view cnt"),
        1
    );

    let sphere = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Spherical,
        vec![0.0, -0.4, 0.0, 0.4],
        vec![1.2, 1.5, 1.8, 1.5],
    )
    .expect("sphere");
    assert_eq!(
        egraph::spatial::spherical_crossing_number(&g4, &sphere).expect("sph_cnt"),
        1
    );
    assert_eq!(
        egraph::spatial::crossing_edges_spherical(&g4, &sphere)
            .expect("sph_edges")
            .len(),
        1
    );

    let poincare = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::HyperbolicPoincare,
        vec![-0.4, 0.0, 0.4, 0.0],
        vec![0.0, -0.4, 0.0, 0.4],
    )
    .expect("poincare");
    assert_eq!(
        egraph::spatial::hyperbolic_crossing_number(&g4, &poincare).expect("hyp_cnt"),
        1
    );
    assert_eq!(
        egraph::spatial::crossing_edges_hyperbolic(&g4, &poincare)
            .expect("hyp_edges")
            .len(),
        1
    );

    let lorentz = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::HyperbolicLorentz,
        vec![-1.0, 0.0, 1.0, 0.0],
        vec![0.0, -1.0, 0.0, 1.0],
    )
    .expect("lorentz");
    assert_eq!(
        egraph::spatial::hyperbolic_crossing_number(&g4, &lorentz).expect("lor_cnt"),
        1
    );
}

#[cfg(feature = "layout")]
fn run_manifold_solver(
    solver: &mut impl IterativeSolver<Target = ManifoldLayout2d<f64>>,
    layout: &mut ManifoldLayout2d<f64>,
    label: &str,
) {
    let st = solver
        .run(layout)
        .unwrap_or_else(|e| panic!("{label} run failed: {e}"));
    assert!(st.is_finished(), "{label}");
    assert!(layout.validate_finite().is_ok(), "{label}");
}

#[cfg(feature = "layout")]
fn run_windowed_sgd(mut solver: SgdSolver<ManifoldLayout2d<f64>>, kind: ManifoldKind, n: usize) {
    let mut ml = ManifoldLayout2d::<f64>::init(kind, n, 7, 0.35).expect("init");
    solver.run_schedule_window(&mut ml, 0, 3).expect("window");
    assert!(ml.validate_finite().is_ok());
}

#[cfg(feature = "layout")]
fn build_sgd_mode_smoke(
    mode: usize,
    g: &CsrGraph<f64>,
    landmarks: &egraph::neighborhood::DogVaradhanLandmarks,
    emb: &[f64],
    sched: SgdScheduleConfig,
) -> (&'static str, SgdSolver<ManifoldLayout2d<f64>>) {
    let n = g.num_nodes();
    let dog_cfg = egraph::neighborhood::DogVaradhanConfig::new()
        .with_n_landmarks(4)
        .with_seed(42);
    let sb_cfg = SparseBarrierConfig::new()
        .with_num_pivots(4)
        .with_schedule(sched.clone())
        .with_seed(42);
    let mut pair_buf = SgdPairBuffer::<f64>::new();
    pair_buf.push_spring(0, 1, 1.0, 1.0).expect("push_spring");
    pair_buf.push_hinge(0, 5, 2.0, 0.5).expect("push_hinge");
    match mode {
        0 => ("full", SgdSolver::full(g, sched, 42).expect("full")),
        1 => {
            let u = [1.0, 1.0];
            let s = SgdSolver::from_pairs(n, &[0, 1], &[1, 2], &u, &u, &u, &u, sched, 42);
            ("pairs", s.expect("pairs"))
        }
        2 => {
            let o = [1.0];
            let s = SgdSolver::from_pairs_with_hinges(
                n,
                &[0],
                &[1],
                &o,
                &o,
                &o,
                &o,
                &[0],
                &[5],
                &[2.0],
                &[2.0],
                &[0.5],
                &[0.5],
                sched,
                42,
            );
            ("hinges", s.expect("hinges"))
        }
        3 => (
            "pair_buffer",
            SgdSolver::from_pair_buffer(n, pair_buf, sched, 42).expect("pair_buffer"),
        ),
        4 => ("rp", SgdSolver::random_pair(g, 3, sched, 42).expect("rp")),
        5 => (
            "omega",
            SgdSolver::omega(g, emb, 2, 3, 1e-3, sched, 42).expect("omega"),
        ),
        6 => (
            "sbl",
            SgdSolver::sparse_barrier_from_landmarks(g, landmarks, None, &sb_cfg)
                .expect("sbl")
                .0,
        ),
        7 => (
            "sbd",
            SgdSolver::sparse_barrier_dog(g, &dog_cfg, &sb_cfg)
                .expect("sbd")
                .0,
        ),
        8 => (
            "sn",
            SgdSolver::streaming_negatives(
                g,
                egraph::layout::StreamingNegativeConfig::default(),
                sched,
                42,
            )
            .expect("sn"),
        ),
        9 => (
            "ortmann",
            SgdSolver::ortmann_sparse_stress(
                g,
                egraph::layout::OrtmannSparseStressConfig::new()
                    .with_num_pivots(4)
                    .with_iterations(6)
                    .with_seed(42),
            )
            .expect("ortmann"),
        ),
        _ => (
            "empty",
            SgdSolver::from_pair_buffer(n, SgdPairBuffer::new(), sched, 42).expect("empty"),
        ),
    }
}

#[cfg(feature = "layout")]
#[test]
fn manifold_all_solvers_and_separated_embedding_solvers_smoke() {
    let g = grid_2x3();
    let n = g.num_nodes();
    let sched = SgdScheduleConfig::new(6);
    let dog_cfg = egraph::neighborhood::DogVaradhanConfig::new()
        .with_n_landmarks(4)
        .with_seed(42);
    let (landmarks, _) =
        egraph::neighborhood::compute_regularized_dog_varadhan_landmarks(&g, &dog_cfg)
            .expect("dog");
    let spec: Layout2d<f64> = KorenSpectral::<Layout2d<f64>>::new(&g)
        .generate()
        .expect("spec");
    let emb: Vec<f64> = (0..n).flat_map(|i| [spec.x()[i], spec.y()[i]]).collect();
    for mode in 0..11 {
        let (label, mut solver) = build_sgd_mode_smoke(mode, &g, &landmarks, &emb, sched.clone());
        let mut l =
            ManifoldLayout2d::<f64>::init(ManifoldKind::HyperbolicLorentz, n, 42, 0.35).unwrap();
        run_manifold_solver(&mut solver, &mut l, label);
    }

    // Separated Sgd2Solver + 5 layout-embedding solvers on ManifoldLayout2d.
    let base_hyp =
        ManifoldLayout2d::<f64>::init(ManifoldKind::HyperbolicLorentz, n, 42, 0.35).unwrap();
    let mut sgd2 = Sgd2Solver::<ManifoldLayout2d<f64>>::new(
        &g,
        Sgd2Config::new()
            .with_iterations(6)
            .with_batch_size(4)
            .with_seed(42),
    )
    .expect("sgd2");
    run_manifold_solver(&mut sgd2, &mut base_hyp.clone(), "sgd2");

    let mut snap = SnapTFdpSolver::<ManifoldLayout2d<f64>>::new(
        &g,
        SnapTFdpConfig::new().with_iterations(6).with_seed(42),
    )
    .expect("snap");
    run_manifold_solver(&mut snap, &mut base_hyp.clone(), "snap");

    let mut pacmap = PacmapGraphSolver::<ManifoldLayout2d<f64>>::new(
        &g,
        PacmapGraphConfig::new()
            .with_phase_iterations([2, 2, 2])
            .with_num_landmarks(4)
            .with_seed(42),
    )
    .expect("pacmap");
    run_manifold_solver(&mut pacmap, &mut base_hyp.clone(), "pacmap");

    let mut lv = LargeVisSolver::<ManifoldLayout2d<f64>>::new(
        &g,
        LargeVisConfig::new()
            .with_iterations(6)
            .with_negative_samples(2)
            .with_seed(42),
    )
    .expect("largevis");
    run_manifold_solver(&mut lv, &mut base_hyp.clone(), "largevis");

    let mut dr = DrGraphSolver::<ManifoldLayout2d<f64>>::new(
        &g,
        DrGraphConfig::new()
            .with_k_order(2)
            .with_iterations(6)
            .with_negative_samples(2)
            .with_coarsen_stop_ratio(0.85)
            .with_seed(42),
    )
    .expect("drgraph");
    run_manifold_solver(&mut dr, &mut base_hyp.clone(), "drgraph");

    let mut cne = GraphCneSolver::<ManifoldLayout2d<f64>>::new(
        &g,
        GraphCneConfig::new()
            .with_iterations(6)
            .with_negative_samples(2)
            .with_seed(42),
    )
    .expect("graph_cne");
    run_manifold_solver(&mut cne, &mut base_hyp.clone(), "graph_cne");

    run_windowed_sgd(
        SgdSolver::full(&g, sched, 42).expect("win"),
        ManifoldKind::Spherical,
        n,
    );

    let sph_full_cfg = SgdScheduleConfig::new(12).with_step_scale(1.0);
    assert_eq!(sph_full_cfg.max_mu, None);
    let mut l_full = ManifoldLayout2d::<f64>::init(ManifoldKind::Spherical, n, 11, 1.0).expect("s");
    let mut l_cap = l_full.clone();
    let mut s_full = SgdSolver::full(&g, sph_full_cfg.clone(), 11).expect("s_full");
    let mut s_cap = SgdSolver::full(&g, sph_full_cfg.with_max_mu(Some(0.1)), 11).expect("s_cap");
    run_manifold_solver(&mut s_full, &mut l_full, "sph_full");
    run_manifold_solver(&mut s_cap, &mut l_cap, "sph_cap");
    assert_ne!(l_full.x(), l_cap.x());

    let base = ManifoldLayout2d::<f64>::init(ManifoldKind::Torus2d, n, 42, 0.4).expect("torus");
    let (mut l_sm, mut l_kk, mut l_ts, mut l_gm) = (base.clone(), base.clone(), base.clone(), base);
    let mut sm = StressMajorizationSolver::<ManifoldLayout2d<f64>>::from_graph(
        &g,
        StressMajorizationConfig::new().with_max_iterations(5),
    )
    .expect("sm");
    run_manifold_solver(&mut sm, &mut l_sm, "stress_majorization");
    assert!(sm.compute_stress(&l_sm).expect("stress").is_finite());

    let mut kk = KamadaKawaiSolver::<ManifoldLayout2d<f64>>::from_graph(
        &g,
        KamadaKawaiConfig::new().with_max_iterations(Some(5)),
    )
    .expect("kk");
    run_manifold_solver(&mut kk, &mut l_kk, "kamada_kawai");

    let stages = vec![
        TsNetStage::new().with_iterations(2).with_exaggeration(4.0),
        TsNetStage::new().with_iterations(2),
        TsNetStage::new().with_iterations(2),
    ];
    let mut ts = TsNetSolver::<ManifoldLayout2d<f64>>::from_graph_pure_2hop(
        &g,
        TsNetConfig::pure_2hop().with_stages(stages),
    )
    .expect("tsnet");
    run_manifold_solver(&mut ts, &mut l_ts, "ts_net");

    let mut gm = GuMapSolver::<ManifoldLayout2d<f64>>::new(
        &g,
        GuMapConfig::new().with_iterations(5).with_seed(42),
    )
    .expect("gumap");
    run_manifold_solver(&mut gm, &mut l_gm, "gumap");
}

#[cfg(all(feature = "layout", feature = "baselines"))]
#[test]
fn manifold_baselines_fr_solver_smoke() {
    let g = grid_2x3();
    let mut ml = ManifoldLayout2d::<f64>::init(ManifoldKind::Spherical, 6, 42, 1.0).expect("sph");
    let mut fr = FruchtermanReingoldSolver::<ManifoldLayout2d<f64>>::new(
        &g,
        FruchtermanReingoldConfig::new()
            .with_iterations(6)
            .with_seed(42),
    )
    .expect("fr");
    run_manifold_solver(&mut fr, &mut ml, "fruchterman_reingold");
}

#[cfg(feature = "metrics")]
#[test]
fn manifold_metrics_unified_and_dispatcher_smoke() {
    let g = grid_2x3();
    let mut torus = ManifoldLayout2d::<f64>::torus_spectral_phase_init(&g, 42).expect("torus");
    torus.set_torus_extents(2.0, 1.5).expect("extents");
    let cfg = MetricConfig::new()
        .with_sns_num_pivots(2)
        .with_np_num_samples(16)
        .with_np_seed(77)
        .with_rta_num_landmarks(3)
        .with_rta_num_triplets(128)
        .with_rta_min_degree(1);
    let rep = evaluate_metrics(
        &g,
        &torus,
        &[
            MetricTarget::GeodesicDistortion,
            MetricTarget::ScaleNormalizedStress,
            MetricTarget::SampledScaleNormalizedStress,
            MetricTarget::TorusCrossingNumber,
            MetricTarget::CrossingEdgesTorus,
            MetricTarget::NeighborhoodPreservation1Hop,
            MetricTarget::NeighborhoodPreservation2Hop,
            MetricTarget::SampledNp1,
            MetricTarget::SampledNp2,
            MetricTarget::GabrielGraphProperty,
            MetricTarget::NodeResolution,
            MetricTarget::RelativeNeighborhoodPreservation,
            MetricTarget::GabrielGraphPreservation,
            MetricTarget::LayoutDegeneracy,
            MetricTarget::IdealEdgeLengths,
            MetricTarget::LandmarkStratifiedRta,
        ],
        &cfg,
    )
    .expect("evaluate_metrics");
    assert_eq!(
        rep.geodesic_distortion,
        Some(egraph::metrics::geodesic_distortion(&g, &torus).expect("gd"))
    );
    assert_eq!(
        rep.scale_normalized_stress,
        Some(egraph::metrics::scale_normalized_stress(&g, &torus).expect("sns"))
    );
    assert_eq!(
        rep.torus_crossing_number,
        Some(egraph::metrics::torus_crossing_number(&g, &torus).expect("tc"))
    );
    assert_eq!(
        rep.neighborhood_preservation_2hop,
        Some(egraph::metrics::neighborhood_preservation_2hop(&g, &torus).expect("np2"))
    );
    assert!(matches!(
        evaluate_metrics(&g, &torus, &[MetricTarget::CrossingNumber], &cfg),
        Err(CoreError::UnsupportedOperation(_))
    ));
}

#[cfg(all(feature = "metrics", feature = "eval-baselines"))]
#[test]
fn manifold_metrics_eval_baselines_unified_smoke() {
    let g = grid_2x3();
    let ml =
        ManifoldLayout2d::<f64>::init(ManifoldKind::HyperbolicPoincare, 6, 42, 0.4).expect("p");
    let labels = [0_u32, 0, 0, 1, 1, 1];
    let rep = evaluate_metrics_with_communities(
        &g,
        &ml,
        &[
            MetricTarget::DistanceConsistency,
            MetricTarget::DistributionConsistency,
            MetricTarget::SepmeKnng,
            MetricTarget::SepmeRng,
            MetricTarget::SepmeGab,
            MetricTarget::SepmeGong,
            MetricTarget::NeighborhoodHit,
            MetricTarget::SilhouetteScore,
            MetricTarget::CalinskiHarabaszScore,
            MetricTarget::DaviesBouldinScore,
            MetricTarget::CqKmeansAri,
            MetricTarget::CqHacHom,
            MetricTarget::PairwiseDistanceOverlapError,
        ],
        &MetricConfig::default(),
        Some(&labels),
    )
    .expect("eval-baselines");
    let bl = rep.baseline_metrics().expect("baseline_metrics");
    assert_eq!(
        bl.distance_consistency,
        Some(egraph::metrics::distance_consistency(&ml, &labels).expect("dsc"))
    );
    assert_eq!(
        bl.silhouette_score,
        Some(egraph::metrics::silhouette_score(&ml, &labels).expect("sil"))
    );
    assert!(
        egraph::metrics::calinski_harabasz_score(&ml, &labels)
            .expect("ch")
            .is_finite()
    );
    assert!(
        egraph::metrics::davies_bouldin_score(&ml, &labels)
            .expect("db")
            .is_finite()
    );
    assert!(
        egraph::metrics::cq_kmeans(&ml, &labels, 2, 10, 42)
            .expect("km")
            .ari
            .is_finite()
    );
    assert!(
        egraph::metrics::cq_hac(&ml, &labels)
            .expect("hac")
            .homogeneity
            .is_finite()
    );
}
