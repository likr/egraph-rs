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

#[cfg(any(feature = "layout", feature = "edge-routing", feature = "metrics"))]
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
    let mut octree = ArenaOctree3d::<f64>::new();
    octree.build(&layout).expect("octree build");
    assert_eq!(octree.point_count(), 6);
    let mut kl_flat = vec![0.0_f64; 18];
    let z_sum = octree
        .compute_kl_repulsion_flat(0.5, &mut kl_flat)
        .expect("kl repulsion");
    assert!(z_sum.is_finite() && z_sum > 0.0);

    let dkd =
        egraph::spatial::DynamicKdTree::<f64>::from_layout_nd(&layout).expect("dynamic kdtree");
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
        let mut l = base_layout.clone();
        let sgd_cfg = SgdScheduleConfig::new(5);
        let mut sgd = SgdSolver::<LayoutNd<f64>>::full(&graph, sgd_cfg, 42).expect("sgd full");
        assert!(sgd.run(&mut l).expect("sgd run").is_finished());
        assert!(l.is_all_finite());
    }

    // 2. StressMajorizationSolver
    {
        let mut l = base_layout.clone();
        let mut sm = StressMajorizationSolver::<LayoutNd<f64>>::from_graph(
            &graph,
            StressMajorizationConfig::new().with_max_iterations(5),
        )
        .expect("stress majorization");
        assert!(sm.run(&mut l).expect("sm run").is_finished());
        assert!(l.is_all_finite());
    }

    // 3. KamadaKawaiSolver
    {
        let mut l = base_layout.clone();
        let mut kk = KamadaKawaiSolver::<LayoutNd<f64>>::from_graph(
            &graph,
            KamadaKawaiConfig::new().with_max_iterations(Some(5)),
        )
        .expect("kamada kawai");
        assert!(kk.run(&mut l).expect("kk run").is_finished());
        assert!(l.is_all_finite());
    }

    // 4. GuMapSolver
    {
        let mut l = base_layout.clone();
        let mut gm = GuMapSolver::<LayoutNd<f64>>::new(
            &graph,
            GuMapConfig::new().with_iterations(5).with_seed(42),
        )
        .expect("gumap");
        assert!(gm.run(&mut l).expect("gumap run").is_finished());
        assert!(l.is_all_finite());
    }

    // 5. TsNetSolver (Exact on LayoutNd + BarnesHut on Layout3d)
    {
        let stages = vec![
            TsNetStage::new().with_iterations(2).with_exaggeration(4.0),
            TsNetStage::new().with_iterations(2),
            TsNetStage::new().with_iterations(2),
        ];
        let mut l_nd = base_layout.clone();
        let mut tsnet_nd = TsNetSolver::<LayoutNd<f64>>::from_graph_pure_2hop(
            &graph,
            TsNetConfig::pure_2hop().with_stages(stages.clone()),
        )
        .expect("tsnet nd");
        assert!(tsnet_nd.run(&mut l_nd).expect("tsnet nd run").is_finished());
        assert!(l_nd.is_all_finite());

        let mut l_3d = Layout3d::<f64>::init(6, 42, 1.0).expect("3d");
        let mut tsnet_3d =
            TsNetSolver::<Layout3d<f64>, egraph::layout::neighbor::BarnesHut>::from_graph_pure_2hop_with_backend(
                &graph,
                TsNetConfig::pure_2hop().with_stages(stages),
                egraph::layout::neighbor::BarnesHut::new(0.5),
            )
            .expect("tsnet 3d");
        assert!(tsnet_3d.run(&mut l_3d).expect("tsnet 3d run").is_finished());
        assert!(l_3d.is_all_finite());
    }

    // 6. ForceAtlas2Solver (Exact default on LayoutNd)
    {
        let mut l = base_layout.clone();
        let mut fa2 = ForceAtlas2Solver::<LayoutNd<f64>>::new(
            &graph,
            ForceAtlas2Config::new().with_iterations(5).with_seed(42),
        )
        .expect("fa2");
        assert!(fa2.run(&mut l).expect("fa2 run").is_finished());
        assert!(l.is_all_finite());
    }

    // 7. YifanHuSolver (Exact default on LayoutNd)
    {
        let mut l = base_layout;
        let mut yh = YifanHuSolver::<LayoutNd<f64>>::new(
            &graph,
            YifanHuConfig::new()
                .with_max_iterations_per_level(5)
                .with_seed(42),
        )
        .expect("yifan hu");
        assert!(yh.run(&mut l).expect("yh run").is_finished());
        assert!(l.is_all_finite());
    }
}

#[cfg(feature = "edge-routing")]
#[test]
fn edge_routing_nd_polylines_smoke() {
    let graph = grid_2x3();
    let base_layout = LayoutNd::<f64>::init(6, 3, 42, 1.0).expect("3d layout");
    let mut polylines =
        EdgePolylines::straight_line_from_graph(&graph, &base_layout).expect("polylines");
    let mut fdeb = FdebSolver::<LayoutNd<f64>>::new(
        &polylines,
        FdebConfig::new().with_cycles(2).with_steps_per_cycle(4),
    )
    .expect("fdeb");
    assert!(fdeb.run(&mut polylines).expect("fdeb run").is_finished());
    assert_eq!(fdeb.dims(), 3);
    assert!(!polylines.points().coords().is_empty());
}

#[cfg(all(feature = "layout", feature = "baselines"))]
#[test]
fn baselines_nd_solvers_3d_smoke() {
    let graph = grid_2x3();
    let base_layout = LayoutNd::<f64>::init(6, 3, 42, 1.0).expect("3d layout");

    // 9. MaxentStressSolver
    {
        let mut l = base_layout.clone();
        let mut maxent = MaxentStressSolver::<LayoutNd<f64>>::new(
            &graph,
            MaxentStressConfig::new()
                .with_stages(2)
                .with_steps_per_stage(3),
        )
        .expect("maxent");
        assert!(maxent.run(&mut l).expect("maxent run").is_finished());
        assert!(l.is_all_finite());
    }

    // 10. FruchtermanReingoldSolver
    {
        let mut l = base_layout.clone();
        let mut fr = FruchtermanReingoldSolver::<LayoutNd<f64>>::new(
            &graph,
            FruchtermanReingoldConfig::new()
                .with_iterations(5)
                .with_seed(42),
        )
        .expect("fr");
        assert!(fr.run(&mut l).expect("fr run").is_finished());
        assert!(l.is_all_finite());
    }

    // 11. LinLogSolver
    {
        let mut l = base_layout;
        let mut ll = LinLogSolver::<LayoutNd<f64>>::new(
            &graph,
            LinLogConfig::new().with_iterations(5).with_seed(42),
        )
        .expect("linlog");
        assert!(ll.run(&mut l).expect("linlog run").is_finished());
        assert!(l.is_all_finite());
    }
}

#[cfg(feature = "metrics")]
#[test]
fn metrics_nd_unified_and_dispatcher_smoke() {
    let graph = grid_2x3();
    let layout = LayoutNd::<f64>::init(6, 3, 99, 1.0).expect("3d layout");
    let cfg = MetricConfig::new()
        .with_sns_num_pivots(2)
        .with_np_num_samples(16)
        .with_np_seed(77)
        .with_rta_num_landmarks(3)
        .with_rta_num_triplets(128)
        .with_rta_min_degree(1);
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
    assert_eq!(
        rep.scale_normalized_stress,
        Some(egraph::metrics::scale_normalized_stress(&graph, &layout).expect("sns"))
    );
    assert_eq!(
        rep.neighborhood_preservation_2hop,
        Some(egraph::metrics::neighborhood_preservation_2hop(&graph, &layout).expect("np2"))
    );
    assert_eq!(
        rep.node_resolution,
        Some(egraph::metrics::node_resolution(&layout).expect("nr"))
    );
    assert_eq!(
        rep.landmark_stratified_rta,
        Some(
            egraph::metrics::landmark_stratified_rta(
                &graph,
                &layout,
                cfg.rta_num_landmarks,
                cfg.rta_num_triplets,
                cfg.rta_min_degree,
                cfg.rta_seed,
            )
            .expect("rta")
        )
    );
    let _ = egraph::metrics::layout_degeneracy(&graph, &layout).expect("degeneracy");
    let _ = egraph::metrics::ideal_edge_lengths(&graph, &layout).expect("iel");
    let _ = egraph::metrics::sampled_scale_normalized_stress_detail(&graph, &layout, 2, 42)
        .expect("ssns");
    let _ = egraph::metrics::scale_normalized_stress_detail(&graph, &layout).expect("sns detail");
    let _ = egraph::metrics::gabriel_graph_edges(&layout).expect("gg edges");
    let _ = egraph::metrics::relative_neighborhood_edges(&layout).expect("rng edges");
    assert!(matches!(
        evaluate_metrics(&graph, &layout, &[MetricTarget::CrossingNumber], &cfg),
        Err(CoreError::UnsupportedOperation(_))
    ));
}

#[cfg(all(feature = "metrics", feature = "eval-baselines"))]
#[test]
fn metrics_nd_eval_baselines_unified_smoke() {
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
    let bl = rep.baseline_metrics().expect("baseline_metrics");
    assert_eq!(
        bl.distance_consistency,
        Some(egraph::metrics::distance_consistency(&layout, &labels).expect("dsc"))
    );
    assert_eq!(
        bl.silhouette_score,
        Some(egraph::metrics::silhouette_score(&layout, &labels).expect("sil"))
    );
    let _ = egraph::metrics::calinski_harabasz_score(&layout, &labels).expect("ch");
    let _ = egraph::metrics::davies_bouldin_score(&layout, &labels).expect("db");
    let _ = egraph::metrics::cq_hac(&layout, &labels).expect("cq hac");
    let _ = egraph::metrics::neighborhood_hit(&layout, &labels, 2).expect("nh");
    let _ = egraph::metrics::sepme_rng(&layout, &labels).expect("rng");
    let _ = egraph::metrics::sepme_gab(&layout, &labels).expect("gab");
    let _ = egraph::metrics::sepme_gong(&layout, &labels, 0.35).expect("gong");
    let _ = egraph::metrics::pairwise_distance_overlap_error(&layout, &labels, 16).expect("pdoe");
}
