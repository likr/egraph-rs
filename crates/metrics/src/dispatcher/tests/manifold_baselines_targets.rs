use super::super::*;
use crate::ClusterAreaConfig;
use crate::cluster_baselines::{
    calinski_harabasz_score_manifold, cq_hac_manifold, cq_kmeans_manifold,
    davies_bouldin_score_manifold, distance_consistency_manifold,
    distribution_consistency_manifold, neighborhood_hit_manifold,
    pairwise_distance_overlap_error_manifold, sepme_gab_manifold, sepme_gong_manifold,
    sepme_knng_manifold, sepme_rng_manifold, silhouette_score_manifold,
};
use egraph_core::{CoreError, CsrGraph, ManifoldKind, ManifoldLayout2d, Scalar};

const BASELINE_MANIFOLD_TARGETS: [MetricTarget; 15] = [
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
    MetricTarget::CqKmeansAmi,
    MetricTarget::CqHacHom,
    MetricTarget::CqHacAmi,
    MetricTarget::PairwiseDistanceOverlapError,
];

fn baseline_test_cfg() -> MetricConfig {
    MetricConfig {
        sns_num_pivots: 3,
        sns_seed: 11,
        crossing_num_samples: 64,
        crossing_seed: 11,
        ideal_edge_length: 1.5,
        rta_num_landmarks: 4,
        rta_num_triplets: 4096,
        rta_min_degree: 1,
        rta_seed: 11,
        patch_nig_radius: 8,
        patch_nig_canvas_size: 256,
        patch_nig_pad_frac: 0.04,
        patch_nig_seed: 11,
        cluster_area: ClusterAreaConfig::default(),
        cluster_baselines: ClusterBaselineConfig::default(),
    }
}

fn make_sample_graph_6() -> CsrGraph<f64> {
    CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)],
        None,
        false,
        false,
    )
    .unwrap()
}

fn make_clustered_manifold_layout<S: Scalar>(
    kind: ManifoldKind,
    scale: f64,
    aspect: f64,
) -> ManifoldLayout2d<S> {
    let mut layout = match kind {
        ManifoldKind::Torus2d => ManifoldLayout2d::<S>::from_xy(
            kind,
            vec![
                S::from_f64(0.10),
                S::from_f64(0.15),
                S::from_f64(0.12),
                S::from_f64(0.75),
                S::from_f64(0.80),
                S::from_f64(0.77),
            ],
            vec![
                S::from_f64(0.10),
                S::from_f64(0.14),
                S::from_f64(0.08),
                S::from_f64(0.75),
                S::from_f64(0.79),
                S::from_f64(0.72),
            ],
        )
        .unwrap(),
        ManifoldKind::Spherical => ManifoldLayout2d::<S>::from_xy(
            kind,
            vec![
                S::from_f64(0.20),
                S::from_f64(0.28),
                S::from_f64(0.24),
                S::from_f64(2.40),
                S::from_f64(2.48),
                S::from_f64(2.44),
            ],
            vec![
                S::from_f64(0.60),
                S::from_f64(0.66),
                S::from_f64(0.55),
                S::from_f64(2.10),
                S::from_f64(2.16),
                S::from_f64(2.04),
            ],
        )
        .unwrap(),
        ManifoldKind::HyperbolicPoincare => ManifoldLayout2d::<S>::from_xy(
            kind,
            vec![
                S::from_f64(-0.45),
                S::from_f64(-0.40),
                S::from_f64(-0.42),
                S::from_f64(0.45),
                S::from_f64(0.40),
                S::from_f64(0.42),
            ],
            vec![
                S::from_f64(0.10),
                S::from_f64(0.14),
                S::from_f64(0.06),
                S::from_f64(-0.10),
                S::from_f64(-0.14),
                S::from_f64(-0.06),
            ],
        )
        .unwrap(),
        ManifoldKind::HyperbolicLorentz => ManifoldLayout2d::<S>::from_xy(
            kind,
            vec![
                S::from_f64(-1.5),
                S::from_f64(-1.3),
                S::from_f64(-1.4),
                S::from_f64(1.5),
                S::from_f64(1.3),
                S::from_f64(1.4),
            ],
            vec![
                S::from_f64(0.2),
                S::from_f64(0.4),
                S::from_f64(0.1),
                S::from_f64(-0.2),
                S::from_f64(-0.4),
                S::from_f64(-0.1),
            ],
        )
        .unwrap(),
        _ => unreachable!(),
    };
    layout.set_scale(scale).unwrap();
    if kind == ManifoldKind::Torus2d {
        layout.set_aspect(aspect).unwrap();
    }
    layout
}

fn assert_baseline_parity<S: Scalar>(kind: ManifoldKind, scale: f64, aspect: f64) {
    let g = make_sample_graph_6();
    let labels = [0_u32, 0, 0, 1, 1, 1];
    let cfg = baseline_test_cfg();
    let layout = make_clustered_manifold_layout::<S>(kind, scale, aspect);

    let rep = evaluate_metrics_with_communities(
        &g,
        &layout,
        &BASELINE_MANIFOLD_TARGETS,
        &cfg,
        Some(&labels),
    )
    .unwrap();
    let rep_cw = evaluate_metrics_with_community_weights(
        &g,
        &layout,
        &BASELINE_MANIFOLD_TARGETS,
        &cfg,
        Some(&labels),
        None,
        None,
    )
    .unwrap();
    assert_eq!(rep, rep_cw);

    let cb = &cfg.cluster_baselines;
    assert_eq!(
        rep.distance_consistency,
        Some(distance_consistency_manifold(&layout, &labels).unwrap())
    );
    assert_eq!(
        rep.distribution_consistency,
        Some(distribution_consistency_manifold(&layout, &labels, cb.dc_grid_size).unwrap())
    );
    assert_eq!(
        rep.sepme_knng,
        Some(sepme_knng_manifold(&layout, &labels, cb.sepme_knng_k).unwrap())
    );
    assert_eq!(
        rep.sepme_rng,
        Some(sepme_rng_manifold(&layout, &labels).unwrap())
    );
    assert_eq!(
        rep.sepme_gab,
        Some(sepme_gab_manifold(&layout, &labels).unwrap())
    );
    assert_eq!(
        rep.sepme_gong,
        Some(sepme_gong_manifold(&layout, &labels, cb.gong_gamma).unwrap())
    );
    assert_eq!(
        rep.neighborhood_hit,
        Some(neighborhood_hit_manifold(&layout, &labels, cb.neighborhood_hit_k).unwrap())
    );
    assert_eq!(
        rep.silhouette_score,
        Some(silhouette_score_manifold(&layout, &labels).unwrap())
    );
    assert_eq!(
        rep.calinski_harabasz_score,
        Some(calinski_harabasz_score_manifold(&layout, &labels).unwrap())
    );
    assert_eq!(
        rep.davies_bouldin_score,
        Some(davies_bouldin_score_manifold(&layout, &labels).unwrap())
    );
    let km = cq_kmeans_manifold(
        &layout,
        &labels,
        cb.kmeans_n_init,
        cb.kmeans_max_iters,
        cb.seed,
    )
    .unwrap();
    assert_eq!(rep.cq_kmeans_ari, Some(km.ari));
    assert_eq!(rep.cq_kmeans_ami, Some(km.ami));
    let hac = cq_hac_manifold(&layout, &labels).unwrap();
    assert_eq!(rep.cq_hac_hom, Some(hac.homogeneity));
    assert_eq!(rep.cq_hac_ami, Some(hac.ami));
    assert_eq!(
        rep.pairwise_distance_overlap_error,
        Some(pairwise_distance_overlap_error_manifold(&layout, &labels, cb.num_bins).unwrap())
    );
}

#[test]
fn test_manifold_dispatcher_eval_baselines_all_kinds_and_errors() {
    let g = make_sample_graph_6();
    let labels = [0_u32, 0, 0, 1, 1, 1];
    let cfg = baseline_test_cfg();

    for &kind in &[
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        assert_baseline_parity::<f64>(kind, 2.0, 1.5);
        assert_baseline_parity::<f32>(kind, 1.25, 0.8);
    }

    // Selective single target leaves all other baseline fields as None.
    let ml = make_clustered_manifold_layout::<f64>(ManifoldKind::HyperbolicPoincare, 1.0, 1.0);
    let subset = evaluate_metrics_with_communities(
        &g,
        &ml,
        &[MetricTarget::SilhouetteScore],
        &cfg,
        Some(&labels),
    )
    .unwrap();
    assert!(subset.silhouette_score.is_some());
    assert!(subset.distance_consistency.is_none());
    assert!(subset.sepme_knng.is_none());
    assert!(subset.cq_kmeans_ari.is_none());

    // 2D-Euclidean-only eval-baselines targets return UnsupportedOperation.
    for &target in &[
        MetricTarget::ClassDensityMeasure,
        MetricTarget::HistogramDensityMeasure,
        MetricTarget::SprawlterNn,
        MetricTarget::NorMetrics,
    ] {
        assert!(
            matches!(
                evaluate_metrics_with_communities(&g, &ml, &[target], &cfg, Some(&labels)),
                Err(CoreError::UnsupportedOperation(_))
            ),
            "expected UnsupportedOperation for {target:?} on ManifoldLayout2d"
        );
    }

    // Missing labels on manifold baseline targets -> InvalidInput.
    for &target in &BASELINE_MANIFOLD_TARGETS {
        assert!(
            matches!(
                evaluate_metrics(&g, &ml, &[target], &cfg),
                Err(CoreError::InvalidInput(_))
            ),
            "expected InvalidInput when labels is None for {target:?}"
        );
    }
}
