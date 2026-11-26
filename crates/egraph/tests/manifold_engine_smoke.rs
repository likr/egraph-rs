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
        assert!(POINCARE_EPS_F64 > 0.0 && POINCARE_EPS_F32 > 0.0);
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
    assert_eq!(
        TorusLayoutView::torus_extents(&torus).expect("extents"),
        (2.0, 1.0)
    );
    assert_eq!(torus_crossing_number(&g4, &torus).expect("torus_cnt"), 1);
    let t_pairs: Vec<TorusCrossing> = torus_crossing_edges(&g4, &torus).expect("torus_edges");
    assert_eq!(t_pairs.len(), 1);
    assert_eq!(
        torus_crossing_edges_from_edges(&[(0, 2), (1, 3)], &torus)
            .expect("from_edges")
            .len(),
        1
    );
    assert!(
        !torus_edge_segments(0.1, 0.1, 0.9, 0.9)
            .expect("segs")
            .is_empty()
    );
    assert!(
        !torus_layout_edge_segments(&g4, &torus)
            .expect("layout_segs")
            .is_empty()
    );

    let sphere = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Spherical,
        vec![0.0, -0.4, 0.0, 0.4],
        vec![1.2, 1.5, 1.8, 1.5],
    )
    .expect("sphere");
    assert_eq!(spherical_crossing_number(&g4, &sphere).expect("sph_cnt"), 1);
    assert_eq!(
        crossing_edges_spherical(&g4, &sphere)
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
        hyperbolic_crossing_number(&g4, &poincare).expect("hyp_cnt"),
        1
    );
    assert_eq!(
        crossing_edges_hyperbolic(&g4, &poincare)
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
        hyperbolic_crossing_number(&g4, &lorentz).expect("lor_cnt"),
        1
    );
}

#[cfg(feature = "layout")]
fn run_manifold_solver(
    solver: &mut impl IterativeSolver<f64, ManifoldLayout2d<f64>>,
    layout: &mut ManifoldLayout2d<f64>,
    label: &str,
) {
    let st = IterativeSolver::<f64, ManifoldLayout2d<f64>>::run(solver, layout)
        .unwrap_or_else(|e| panic!("{label} run failed: {e}"));
    assert!(st.is_finished(), "{label}");
    assert!(layout.validate_finite().is_ok(), "{label}");
}

#[cfg(feature = "layout")]
fn run_windowed_sgd(mut solver: SgdSolver<f64>, kind: ManifoldKind, n: usize) {
    let mut ml = ManifoldLayout2d::<f64>::init(kind, n, 7, 0.35).expect("init");
    fn advance<L: SgdScheduleWindowLayout<f64>>(s: &mut SgdSolver<f64>, l: &mut L) {
        s.run_schedule_window(l, 0, 3).expect("window");
    }
    advance(&mut solver, &mut ml);
    assert!(ml.validate_finite().is_ok());
}

#[cfg(feature = "layout")]
fn build_sgd_mode_smoke(
    mode: usize,
    g: &CsrGraph<f64>,
    landmarks: &DogVaradhanLandmarks,
    emb: &[f64],
    sched: SgdScheduleConfig,
) -> (&'static str, SgdSolver<f64>) {
    let n = g.num_nodes();
    let dog_cfg = DogVaradhanConfig {
        n_landmarks: 4,
        seed: 42,
        ..DogVaradhanConfig::default()
    };
    let sb_cfg = SparseBarrierConfig {
        num_pivots: 4,
        schedule: sched,
        seed: 42,
        ..SparseBarrierConfig::default()
    };
    let (u, o) = (vec![1.0, 1.0], vec![1.0]);
    match mode {
        0 => ("full", SgdSolver::full(g, sched, 42).expect("full")),
        1 => (
            "pairs",
            SgdSolver::from_pairs(
                n,
                vec![0, 1],
                vec![1, 2],
                u.clone(),
                u.clone(),
                u.clone(),
                u,
                sched,
                42,
            )
            .expect("pairs"),
        ),
        2 => (
            "hinges",
            SgdSolver::from_pairs_with_hinges(
                n,
                vec![0],
                vec![1],
                o.clone(),
                o.clone(),
                o.clone(),
                o,
                vec![0],
                vec![5],
                vec![2.0],
                vec![2.0],
                vec![0.5],
                vec![0.5],
                sched,
                42,
            )
            .expect("hinges"),
        ),
        3 => (
            "packed",
            SgdSolver::from_packed_symmetric(
                n,
                vec![egraph::layout::sgd::PackedSymPair::new(
                    0, 1, false, 1.0, 1.0,
                )],
                1,
                0,
                sched,
                42,
            )
            .expect("packed"),
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
        9 => {
            let mut c = OrtmannSparseStressConfig::default();
            (c.num_pivots, c.iterations, c.seed) = (4, 6, 42);
            (
                "ortmann",
                SgdSolver::ortmann_sparse_stress(g, c).expect("o"),
            )
        }
        10 => (
            "empty",
            SgdSolver::from_packed_symmetric(n, Vec::new(), 0, 0, sched, 42).expect("empty"),
        ),
        11 => {
            let mut c = SnapTFdpConfig::default();
            (c.iterations, c.seed) = (6, 42);
            ("snap", SgdSolver::snap_t_fdp(g, c).expect("snap"))
        }
        12 => {
            let mut c = PacmapGraphConfig::default();
            (c.phase_iterations, c.num_landmarks, c.seed) = ([2, 2, 2], 4, 42);
            ("pacmap", SgdSolver::pacmap_graph(g, c).expect("pacmap"))
        }
        13 => {
            let mut c = Sgd2Config::default();
            (c.iterations, c.batch_size, c.seed) = (6, 4, 42);
            ("sgd2", SgdSolver::sgd2(g, c).expect("sgd2"))
        }
        14 => {
            let mut c = LargeVisConfig::default();
            (c.iterations, c.negative_samples, c.seed) = (6, 2, 42);
            ("largevis", SgdSolver::largevis(g, c).expect("largevis"))
        }
        15 => {
            let mut c = DrGraphConfig::default();
            (
                c.k_order,
                c.iterations,
                c.negative_samples,
                c.coarsen_stop_ratio,
                c.seed,
            ) = (2, 6, 2, 0.85, 42);
            ("drgraph", SgdSolver::drgraph(g, c).expect("drgraph"))
        }
        _ => {
            let mut c = GraphCneConfig::default();
            (c.iterations, c.negative_samples, c.seed) = (6, 2, 42);
            ("graph_cne", SgdSolver::graph_cne(g, c).expect("graph_cne"))
        }
    }
}

#[cfg(feature = "layout")]
#[test]
fn manifold_all_solvers_and_17_sgd_modes_smoke() {
    let g = grid_2x3();
    let n = g.num_nodes();
    let sched = SgdScheduleConfig::new(6);
    let dog_cfg = DogVaradhanConfig {
        n_landmarks: 4,
        seed: 42,
        ..DogVaradhanConfig::default()
    };
    let (landmarks, _) = compute_regularized_dog_varadhan_landmarks(&g, &dog_cfg).expect("dog");
    let spec = koren_spectral_init::<f64, f64>(&g, Some(1.0), 1e-6, 100, 42).expect("spec");
    let emb: Vec<f64> = (0..n).flat_map(|i| [spec.x()[i], spec.y()[i]]).collect();
    for mode in 0..17 {
        let (label, mut solver) = build_sgd_mode_smoke(mode, &g, &landmarks, &emb, sched);
        assert!(solver.supports_manifold(), "{label}");
        let mut l =
            ManifoldLayout2d::<f64>::init(ManifoldKind::HyperbolicLorentz, n, 42, 0.35).unwrap();
        run_manifold_solver(&mut solver, &mut l, label);
    }
    run_windowed_sgd(
        SgdSolver::full(&g, sched, 42).expect("win"),
        ManifoldKind::Spherical,
        n,
    );

    let sph_full_cfg = SgdScheduleConfig::new(12).with_step_scale(1.0);
    assert_eq!(sph_full_cfg.max_mu, None);
    let mut l_full = ManifoldLayout2d::<f64>::init(ManifoldKind::Spherical, n, 11, 1.0).expect("s");
    let mut l_cap = l_full.clone();
    let mut s_full = SgdSolver::full(&g, sph_full_cfg, 11).expect("s_full");
    let mut s_cap = SgdSolver::full(&g, sph_full_cfg.with_max_mu(Some(0.1)), 11).expect("s_cap");
    run_manifold_solver(&mut s_full, &mut l_full, "sph_full");
    run_manifold_solver(&mut s_cap, &mut l_cap, "sph_cap");
    assert_ne!(l_full.x(), l_cap.x());

    let base = ManifoldLayout2d::<f64>::init(ManifoldKind::Torus2d, n, 42, 0.4).expect("torus");
    let (mut l_sm, mut l_kk, mut l_ts, mut l_gm) = (base.clone(), base.clone(), base.clone(), base);
    let (mut c_sm, mut c_kk) = (
        StressMajorizationConfig::default(),
        KamadaKawaiConfig::default(),
    );
    (c_sm.max_iterations, c_kk.max_iterations) = (5, Some(5));
    let mut sm = StressMajorizationSolver::<f64>::from_graph(&g, c_sm).expect("sm");
    run_manifold_solver(&mut sm, &mut l_sm, "stress_majorization");
    assert!(
        sm.compute_stress_manifold(&l_sm)
            .expect("stress")
            .is_finite()
    );

    let mut kk = KamadaKawaiSolver::<f64>::from_graph(&g, c_kk).expect("kk");
    run_manifold_solver(&mut kk, &mut l_kk, "kamada_kawai");

    let mut c_ts = TsNetConfig::pure_2hop();
    (c_ts.backend, c_ts.stage_iterations) = (TsNetBackend::Exact, [2, 2, 2]);
    let mut ts = TsNetSolver::<f64>::from_graph_pure_2hop(&g, c_ts).expect("tsnet");
    run_manifold_solver(&mut ts, &mut l_ts, "ts_net");

    let mut c_gm = GuMapConfig::default();
    (c_gm.iterations, c_gm.seed) = (5, 42);
    let mut gm = GuMapSolver::<f64>::new(&g, c_gm).expect("gumap");
    run_manifold_solver(&mut gm, &mut l_gm, "gumap");
}

#[cfg(all(feature = "layout", feature = "baselines"))]
#[test]
fn manifold_baselines_fr_solver_smoke() {
    let g = grid_2x3();
    let mut ml = ManifoldLayout2d::<f64>::init(ManifoldKind::Spherical, 6, 42, 1.0).expect("sph");
    let mut c_fr = FruchtermanReingoldConfig::default();
    (c_fr.iterations, c_fr.seed) = (6, 42);
    let mut fr = FruchtermanReingoldSolver::<f64>::new(&g, c_fr).expect("fr");
    run_manifold_solver(&mut fr, &mut ml, "fruchterman_reingold");
}

#[cfg(feature = "metrics")]
#[test]
fn manifold_metrics_prelude_and_dispatcher_smoke() {
    let g = grid_2x3();
    let mut torus = ManifoldLayout2d::<f64>::torus_spectral_phase_init(&g, 42).expect("torus");
    torus.set_torus_extents(2.0, 1.5).expect("extents");
    let cfg = MetricConfig {
        sns_num_pivots: 2,
        crossing_num_samples: 16,
        rta_num_landmarks: 3,
        rta_num_triplets: 128,
        rta_min_degree: 1,
        ..MetricConfig::default()
    };
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
        Some(geodesic_distortion(&g, &torus).expect("gd"))
    );
    assert_eq!(
        rep.scale_normalized_stress,
        Some(scale_normalized_stress_manifold(&g, &torus).expect("sns"))
    );
    assert_eq!(
        rep.torus_crossing_number,
        Some(torus_crossing_number(&g, &torus).expect("tc"))
    );
    assert_eq!(
        rep.neighborhood_preservation_2hop,
        Some(neighborhood_preservation_2hop_manifold(&g, &torus).expect("np2"))
    );
    assert!(
        egraph::metrics::geodesic_distortion(&g, &torus)
            .expect("mod gd")
            .is_finite()
    );
    assert!(matches!(
        evaluate_metrics(&g, &torus, &[MetricTarget::CrossingNumber], &cfg),
        Err(CoreError::UnsupportedOperation(_))
    ));
}

#[cfg(all(feature = "metrics", feature = "eval-baselines"))]
#[test]
fn manifold_metrics_eval_baselines_prelude_smoke() {
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
    assert_eq!(
        rep.distance_consistency,
        Some(distance_consistency_manifold(&ml, &labels).expect("dsc"))
    );
    assert_eq!(
        rep.silhouette_score,
        Some(silhouette_score_manifold(&ml, &labels).expect("sil"))
    );
    assert!(
        calinski_harabasz_score_manifold(&ml, &labels)
            .expect("ch")
            .is_finite()
    );
    assert!(
        davies_bouldin_score_manifold(&ml, &labels)
            .expect("db")
            .is_finite()
    );
    assert!(
        cq_kmeans_manifold(&ml, &labels, 2, 10, 42)
            .expect("km")
            .ari
            .is_finite()
    );
    assert!(
        cq_hac_manifold(&ml, &labels)
            .expect("hac")
            .homogeneity
            .is_finite()
    );
}
