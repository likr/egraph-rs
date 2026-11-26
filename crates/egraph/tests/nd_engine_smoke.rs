use egraph::prelude::*;

#[test]
fn core_nd_layout_prelude_smoke() {
    let graph =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], None, false, false)
            .expect("valid cycle");
    assert_eq!(graph.num_nodes(), 4);
    let mut layout = LayoutNd::<f64>::init(4, 3, 42, 1.0).expect("3d layout");
    layout.center_and_normalize_std().expect("normalize");
    assert!(layout.is_all_finite());
}

#[cfg(any(feature = "layout", feature = "metrics"))]
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

#[cfg(feature = "spatial")]
#[test]
fn spatial_nd_trees_reexports_smoke() {
    let layout = LayoutNd::<f64>::init(6, 3, 42, 1.0).expect("3d layout");
    let mut octree: egraph::spatial::ArenaOctree3d<f64> = ArenaOctree3d::<f64>::new();
    octree.build(&layout).expect("octree build");
    assert_eq!(octree.point_count(), 6);
    let mut kl_flat = vec![0.0_f64; 18];
    let z_sum = octree
        .compute_kl_repulsion_flat(0.5, &mut kl_flat)
        .expect("kl repulsion");
    assert!(z_sum.is_finite() && z_sum > 0.0);

    let dkd: egraph::spatial::DynamicKdTree<f64> =
        DynamicKdTree::<f64>::from_layout_nd(&layout).expect("dynamic kdtree");
    assert_eq!(dkd.len(), 6);
    assert_eq!(dkd.dims(), 3);
    let nn = dkd.nearest_excluding_self(0).expect("nearest");
    assert!(nn.is_some());
}

#[cfg(feature = "layout")]
#[test]
fn all_core_nd_solvers_3d_smoke() {
    let graph = grid_2x3();
    let base_layout = LayoutNd::<f64>::init(6, 3, 42, 1.0).expect("3d layout");

    // 1. SgdSolver (full)
    {
        let mut l = base_layout.copy();
        let mut sgd_cfg = SgdScheduleConfig::default();
        sgd_cfg.iterations = 5;
        let mut sgd = SgdSolver::<f64>::full(&graph, sgd_cfg, 42).expect("sgd full");
        assert!(
            IterativeSolver::run(&mut sgd, &mut l)
                .expect("sgd run")
                .is_finished()
        );
        assert!(l.is_all_finite());
    }

    // 2. StressMajorizationSolver
    {
        let mut l = base_layout.copy();
        let mut sm = StressMajorizationSolver::<f64>::from_graph(
            &graph,
            StressMajorizationConfig {
                max_iterations: 5,
                ..StressMajorizationConfig::default()
            },
        )
        .expect("stress majorization");
        assert!(
            IterativeSolver::run(&mut sm, &mut l)
                .expect("sm run")
                .is_finished()
        );
        assert!(l.is_all_finite());
    }

    // 3. KamadaKawaiSolver
    {
        let mut l = base_layout.copy();
        let mut kk = KamadaKawaiSolver::<f64>::from_graph(
            &graph,
            KamadaKawaiConfig {
                max_iterations: Some(5),
                ..KamadaKawaiConfig::default()
            },
        )
        .expect("kamada kawai");
        assert!(
            IterativeSolver::run(&mut kk, &mut l)
                .expect("kk run")
                .is_finished()
        );
        assert!(l.is_all_finite());
    }

    // 4. GuMapSolver
    {
        let mut l = base_layout.copy();
        let mut gm = GuMapSolver::<f64>::new(
            &graph,
            GuMapConfig {
                iterations: 5,
                seed: 42,
                ..GuMapConfig::default()
            },
        )
        .expect("gumap");
        assert!(
            IterativeSolver::run(&mut gm, &mut l)
                .expect("gumap run")
                .is_finished()
        );
        assert!(l.is_all_finite());
    }

    // 5. TsNetSolver (Exact + 3D BarnesHut)
    for backend in [TsNetBackend::Exact, TsNetBackend::BarnesHut] {
        let mut l = base_layout.copy();
        let mut tsnet = TsNetSolver::<f64>::from_graph_pure_2hop(
            &graph,
            TsNetConfig {
                backend,
                stage_iterations: [2, 2, 2],
                ..TsNetConfig::pure_2hop()
            },
        )
        .expect("tsnet");
        assert!(
            IterativeSolver::run(&mut tsnet, &mut l)
                .expect("tsnet run")
                .is_finished()
        );
        assert!(l.is_all_finite());
    }

    // 6. ForceAtlas2Solver
    {
        let mut l = base_layout.copy();
        let mut fa2 = ForceAtlas2Solver::<f64>::new(
            &graph,
            ForceAtlas2Config {
                iterations: 5,
                seed: 42,
                ..ForceAtlas2Config::default()
            },
        )
        .expect("fa2");
        assert!(
            IterativeSolver::run(&mut fa2, &mut l)
                .expect("fa2 run")
                .is_finished()
        );
        assert!(l.is_all_finite());
    }

    // 7. YifanHuSolver
    {
        let mut l = base_layout.copy();
        let mut yh = YifanHuSolver::<f64>::new(
            &graph,
            YifanHuConfig {
                max_iterations_per_level: 5,
                seed: 42,
                ..YifanHuConfig::default()
            },
        )
        .expect("yifan hu");
        assert!(
            IterativeSolver::run(&mut yh, &mut l)
                .expect("yh run")
                .is_finished()
        );
        assert!(l.is_all_finite());
    }

    // 8. FdebSolver (3D edge bundling)
    {
        let mut l = base_layout.copy();
        let mut fdeb = FdebSolver::<f64>::new(
            &graph,
            FdebConfig {
                cycles: 2,
                steps_per_cycle: 4,
                ..FdebConfig::default()
            },
        )
        .expect("fdeb");
        assert!(
            IterativeSolver::run(&mut fdeb, &mut l)
                .expect("fdeb run")
                .is_finished()
        );
        assert_eq!(fdeb.point_dims(), 3);
        assert!(!fdeb.points_nd().is_empty());
    }
}

#[cfg(all(feature = "layout", feature = "baselines"))]
#[test]
fn baselines_nd_solvers_3d_smoke() {
    let graph = grid_2x3();
    let base_layout = LayoutNd::<f64>::init(6, 3, 42, 1.0).expect("3d layout");

    // 9. MaxentStressSolver
    {
        let mut l = base_layout.copy();
        let mut maxent = MaxentStressSolver::<f64>::new(
            &graph,
            MaxentStressConfig {
                stages: 2,
                steps_per_stage: 3,
                ..MaxentStressConfig::default()
            },
        )
        .expect("maxent");
        assert!(
            IterativeSolver::run(&mut maxent, &mut l)
                .expect("maxent run")
                .is_finished()
        );
        assert!(l.is_all_finite());
    }

    // 10. FruchtermanReingoldSolver
    {
        let mut l = base_layout.copy();
        let mut fr = FruchtermanReingoldSolver::<f64>::new(
            &graph,
            FruchtermanReingoldConfig {
                iterations: 5,
                seed: 42,
                ..FruchtermanReingoldConfig::default()
            },
        )
        .expect("fr");
        assert!(
            IterativeSolver::run(&mut fr, &mut l)
                .expect("fr run")
                .is_finished()
        );
        assert!(l.is_all_finite());
    }

    // 11. LinLogSolver
    {
        let mut l = base_layout.copy();
        let mut ll = LinLogSolver::<f64>::new(
            &graph,
            LinLogConfig {
                iterations: 5,
                seed: 42,
                ..LinLogConfig::default()
            },
        )
        .expect("linlog");
        assert!(
            IterativeSolver::run(&mut ll, &mut l)
                .expect("linlog run")
                .is_finished()
        );
        assert!(l.is_all_finite());
    }
}

#[cfg(feature = "metrics")]
#[test]
fn metrics_nd_prelude_and_dispatcher_smoke() {
    let graph = grid_2x3();
    let layout = LayoutNd::<f64>::init(6, 3, 99, 1.0).expect("3d layout");
    let cfg = MetricConfig {
        sns_num_pivots: 2,
        crossing_num_samples: 16,
        rta_num_landmarks: 3,
        rta_num_triplets: 128,
        rta_min_degree: 1,
        ..MetricConfig::default()
    };
    let targets = [
        MetricTarget::ScaleNormalizedStress,
        MetricTarget::SampledScaleNormalizedStress,
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
    ];
    let rep = evaluate_metrics(&graph, &layout, &targets, &cfg).expect("evaluate_metrics nd");
    let rep_explicit =
        evaluate_metrics_nd(&graph, &layout, &targets, &cfg).expect("evaluate_metrics_nd");
    assert_eq!(rep, rep_explicit);
    assert_eq!(
        rep.scale_normalized_stress,
        Some(scale_normalized_stress_nd(&graph, &layout).expect("sns nd"))
    );
    assert_eq!(
        rep.neighborhood_preservation_2hop,
        Some(neighborhood_preservation_2hop_nd(&graph, &layout).expect("np2 nd"))
    );
    assert_eq!(
        rep.node_resolution,
        Some(node_resolution_nd(&layout).expect("nr nd"))
    );
    assert_eq!(
        rep.landmark_stratified_rta,
        Some(
            landmark_stratified_rta_nd(
                &graph,
                &layout,
                cfg.rta_num_landmarks,
                cfg.rta_num_triplets,
                cfg.rta_min_degree,
                cfg.rta_seed,
            )
            .expect("rta nd")
        )
    );
    let _ = layout_degeneracy_nd(&graph, &layout).expect("degeneracy nd");
    let _ = ideal_edge_lengths_nd(&graph, &layout).expect("iel nd");
    let _ = sampled_scale_normalized_stress_detail_nd(&graph, &layout, 2, 42).expect("ssns");
    let _ = scale_normalized_stress_detail_nd(&graph, &layout).expect("sns detail");
    let _ = gabriel_graph_edges_nd(&layout).expect("gg edges");
    let _ = relative_neighborhood_edges_nd(&layout).expect("rng edges");
    assert!(matches!(
        evaluate_metrics(&graph, &layout, &[MetricTarget::CrossingNumber], &cfg),
        Err(CoreError::UnsupportedOperation(_))
    ));
}

#[cfg(all(feature = "metrics", feature = "eval-baselines"))]
#[test]
fn metrics_nd_eval_baselines_prelude_smoke() {
    let graph = grid_2x3();
    let layout = LayoutNd::<f64>::init(6, 3, 99, 1.0).expect("3d layout");
    let labels = [0_u32, 0, 0, 1, 1, 1];
    let rep = evaluate_metrics_with_communities(
        &graph,
        &layout,
        &[
            MetricTarget::DistanceConsistency,
            MetricTarget::DistributionConsistency,
            MetricTarget::SepmeKnng,
            MetricTarget::SilhouetteScore,
            MetricTarget::CqKmeansAri,
        ],
        &MetricConfig::default(),
        Some(&labels),
    )
    .expect("eval-baselines nd");
    assert_eq!(
        rep.distance_consistency,
        Some(distance_consistency_nd(&layout, &labels).expect("dsc nd"))
    );
    assert_eq!(
        rep.silhouette_score,
        Some(silhouette_score_nd(&layout, &labels).expect("sil nd"))
    );
    let _ = calinski_harabasz_score_nd(&layout, &labels).expect("ch nd");
    let _ = davies_bouldin_score_nd(&layout, &labels).expect("db nd");
    let _ = cq_hac_nd(&layout, &labels).expect("cq hac nd");
    let _ = neighborhood_hit_nd(&layout, &labels, 2).expect("nh nd");
    let _ = sepme_rng_nd(&layout, &labels).expect("rng nd");
    let _ = sepme_gab_nd(&layout, &labels).expect("gab nd");
    let _ = sepme_gong_nd(&layout, &labels, 0.35).expect("gong nd");
    let _ = pairwise_distance_overlap_error_nd(&layout, &labels, 16).expect("pdoe nd");
}
