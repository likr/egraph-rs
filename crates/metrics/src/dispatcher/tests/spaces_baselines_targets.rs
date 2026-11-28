use super::super::*;
use crate::{
    calinski_harabasz_score, cq_hac, cq_kmeans, davies_bouldin_score, distance_consistency,
    distribution_consistency, neighborhood_hit, pairwise_distance_overlap_error, sepme_gab,
    sepme_gong, sepme_knng, sepme_rng, silhouette_score,
};
use egraph_core::{
    CoreError, CoreResult, CsrGraph, Layout2d, Layout3d, LayoutNd, LayoutStorage, ManifoldKind,
    ManifoldLayout2d, MetricSpace, Scalar,
};

const GENERAL_BASELINE_TARGETS: [MetricTarget; 15] = [
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
    MetricConfig::new()
        .with_sns_num_pivots(3)
        .with_sns_seed(11)
        .with_np_num_samples(64)
        .with_np_seed(11)
        .with_baseline_config(ClusterBaselineConfig::new())
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

fn assert_general_baseline_parity<L: MetricSpace + 'static>(
    g: &CsrGraph<f64>,
    layout: &L,
    labels: &[u32],
    cfg: &MetricConfig,
) {
    let rep =
        evaluate_metrics_with_communities(g, layout, &GENERAL_BASELINE_TARGETS, cfg, Some(labels))
            .unwrap();
    let rep_cw = evaluate_metrics_with_community_weights(
        g,
        layout,
        &GENERAL_BASELINE_TARGETS,
        cfg,
        Some(labels),
        None,
        None,
    )
    .unwrap();
    assert_eq!(rep, rep_cw);

    let br = rep.baseline_metrics().unwrap();
    let cb = cfg.baseline_config();
    assert_eq!(
        br.distance_consistency,
        Some(distance_consistency(layout, labels).unwrap())
    );
    assert_eq!(
        br.distribution_consistency,
        Some(distribution_consistency(layout, labels, cb.dc_grid_size).unwrap())
    );
    assert_eq!(
        br.sepme_knng,
        Some(sepme_knng(layout, labels, cb.sepme_knng_k).unwrap())
    );
    assert_eq!(br.sepme_rng, Some(sepme_rng(layout, labels).unwrap()));
    assert_eq!(br.sepme_gab, Some(sepme_gab(layout, labels).unwrap()));
    assert_eq!(
        br.sepme_gong,
        Some(sepme_gong(layout, labels, cb.gong_gamma).unwrap())
    );
    assert_eq!(
        br.neighborhood_hit,
        Some(neighborhood_hit(layout, labels, cb.neighborhood_hit_k).unwrap())
    );
    assert_eq!(
        br.silhouette_score,
        Some(silhouette_score(layout, labels).unwrap())
    );
    assert_eq!(
        br.calinski_harabasz_score,
        Some(calinski_harabasz_score(layout, labels).unwrap())
    );
    assert_eq!(
        br.davies_bouldin_score,
        Some(davies_bouldin_score(layout, labels).unwrap())
    );
    let km = cq_kmeans(
        layout,
        labels,
        cb.kmeans_n_init,
        cb.kmeans_max_iters,
        cb.seed,
    )
    .unwrap();
    assert_eq!(br.cq_kmeans_ari, Some(km.ari));
    assert_eq!(br.cq_kmeans_ami, Some(km.ami));
    let hac = cq_hac(layout, labels).unwrap();
    assert_eq!(br.cq_hac_hom, Some(hac.homogeneity));
    assert_eq!(br.cq_hac_ami, Some(hac.ami));
    assert_eq!(
        br.pairwise_distance_overlap_error,
        Some(pairwise_distance_overlap_error(layout, labels, cb.num_bins).unwrap())
    );
}

#[test]
fn test_euclidean_3d_nd_eval_baselines_parity_and_errors() {
    let g = make_sample_graph_6();
    let labels = [0_u32, 0, 0, 1, 1, 1];
    let cfg = baseline_test_cfg();

    let xs = vec![0.0, 0.2, 0.1, 5.0, 5.2, 5.1];
    let ys = vec![0.0, 0.1, -0.1, 5.0, 5.1, 4.9];
    let l2d = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
    let mut flat2 = Vec::with_capacity(12);
    for i in 0..6 {
        flat2.push(xs[i]);
        flat2.push(ys[i]);
    }
    let lnd2 = LayoutNd::from_flat(6, 2, &flat2).unwrap();
    assert_eq!(
        evaluate_metrics_with_communities(&g, &l2d, &GENERAL_BASELINE_TARGETS, &cfg, Some(&labels))
            .unwrap(),
        evaluate_metrics_with_communities(
            &g,
            &lnd2,
            &GENERAL_BASELINE_TARGETS,
            &cfg,
            Some(&labels)
        )
        .unwrap()
    );

    let l3d = Layout3d::from_xyz(
        vec![0.0, 0.2, 0.1, 6.0, 6.2, 6.1],
        vec![0.0, 0.1, -0.1, 6.0, 6.1, 5.9],
        vec![0.0, 0.3, 0.2, 6.0, 6.3, 6.2],
    )
    .unwrap();
    assert_general_baseline_parity(&g, &l3d, &labels, &cfg);

    for &dims in &[3usize, 10] {
        let mut flat = Vec::with_capacity(6 * dims);
        for i in 0..6 {
            let base = if i < 3 { 0.0 } else { 6.0 };
            for d in 0..dims {
                flat.push(base + ((i * 5 + d * 2) as f64) * 0.1);
            }
        }
        let lnd = LayoutNd::from_flat(6, dims, &flat).unwrap();
        assert_general_baseline_parity(&g, &lnd, &labels, &cfg);
    }

    for &target in &[
        MetricTarget::ClassDensityMeasure,
        MetricTarget::HistogramDensityMeasure,
        MetricTarget::SprawlterNn,
        MetricTarget::NorMetrics,
    ] {
        assert!(matches!(
            evaluate_metrics_with_communities(&g, &l3d, &[target], &cfg, Some(&labels)),
            Err(CoreError::UnsupportedOperation(_))
        ));
    }

    for &target in &[
        MetricTarget::DistanceConsistency,
        MetricTarget::SepmeKnng,
        MetricTarget::SilhouetteScore,
        MetricTarget::CqKmeansAri,
    ] {
        assert!(matches!(
            evaluate_metrics(&g, &l3d, &[target], &cfg),
            Err(CoreError::InvalidInput(_))
        ));
    }
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

#[test]
fn test_manifold_eval_baselines_all_kinds_and_errors() {
    let g = make_sample_graph_6();
    let labels = [0_u32, 0, 0, 1, 1, 1];
    let cfg = baseline_test_cfg();

    for &kind in &[
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        let m64 = make_clustered_manifold_layout::<f64>(kind, 2.0, 1.5);
        assert_general_baseline_parity(&g, &m64, &labels, &cfg);
        let m32 = make_clustered_manifold_layout::<f32>(kind, 1.25, 0.8);
        assert_general_baseline_parity(&g, &m32, &labels, &cfg);
    }

    let ml = make_clustered_manifold_layout::<f64>(ManifoldKind::HyperbolicPoincare, 1.0, 1.0);
    let subset = evaluate_metrics_with_communities(
        &g,
        &ml,
        &[MetricTarget::SilhouetteScore],
        &cfg,
        Some(&labels),
    )
    .unwrap();
    let sub_br = subset.baseline_metrics().unwrap();
    assert!(sub_br.silhouette_score.is_some());
    assert!(sub_br.distance_consistency.is_none());
    assert!(sub_br.sepme_knng.is_none());
    assert!(sub_br.cq_kmeans_ari.is_none());

    for &target in &[
        MetricTarget::ClassDensityMeasure,
        MetricTarget::HistogramDensityMeasure,
        MetricTarget::SprawlterNn,
        MetricTarget::NorMetrics,
    ] {
        assert!(matches!(
            evaluate_metrics_with_communities(&g, &ml, &[target], &cfg, Some(&labels)),
            Err(CoreError::UnsupportedOperation(_))
        ));
    }

    for &target in &GENERAL_BASELINE_TARGETS {
        assert!(matches!(
            evaluate_metrics(&g, &ml, &[target], &cfg),
            Err(CoreError::InvalidInput(_))
        ));
    }
}

struct ExternalClusterMetricSpace {
    pts: Vec<[f64; 3]>,
}

impl LayoutStorage for ExternalClusterMetricSpace {
    type Scalar = f64;

    fn len(&self) -> usize {
        self.pts.len()
    }

    fn validate_finite(&self) -> CoreResult<()> {
        if self.pts.iter().all(|p| p.iter().all(|v| v.is_finite())) {
            Ok(())
        } else {
            Err(CoreError::non_finite("ExternalClusterMetricSpace"))
        }
    }
}

impl MetricSpace for ExternalClusterMetricSpace {
    fn distance(&self, i: usize, j: usize) -> f64 {
        let a = self.pts[i];
        let b = self.pts[j];
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

#[test]
fn test_external_metric_space_eval_baselines() {
    let g = make_sample_graph_6();
    let labels = [0_u32, 0, 0, 1, 1, 1];
    let cfg = baseline_test_cfg();
    let space = ExternalClusterMetricSpace {
        pts: vec![
            [0.0, 0.0, 0.0],
            [0.2, 0.1, 0.0],
            [0.1, -0.1, 0.2],
            [5.0, 5.0, 5.0],
            [5.2, 5.1, 5.0],
            [5.1, 4.9, 5.2],
        ],
    };

    let dist_targets = [
        MetricTarget::SepmeKnng,
        MetricTarget::SepmeRng,
        MetricTarget::SepmeGab,
        MetricTarget::SepmeGong,
        MetricTarget::NeighborhoodHit,
        MetricTarget::SilhouetteScore,
        MetricTarget::CqHacHom,
        MetricTarget::CqHacAmi,
        MetricTarget::PairwiseDistanceOverlapError,
    ];
    let rep =
        evaluate_metrics_with_communities(&g, &space, &dist_targets, &cfg, Some(&labels)).unwrap();
    let br = rep.baseline_metrics().unwrap();
    let cb = cfg.baseline_config();
    assert_eq!(
        br.sepme_knng,
        Some(sepme_knng(&space, &labels, cb.sepme_knng_k).unwrap())
    );
    assert_eq!(br.sepme_rng, Some(sepme_rng(&space, &labels).unwrap()));
    assert_eq!(br.sepme_gab, Some(sepme_gab(&space, &labels).unwrap()));
    assert_eq!(
        br.sepme_gong,
        Some(sepme_gong(&space, &labels, cb.gong_gamma).unwrap())
    );
    assert_eq!(
        br.neighborhood_hit,
        Some(neighborhood_hit(&space, &labels, cb.neighborhood_hit_k).unwrap())
    );
    assert_eq!(
        br.silhouette_score,
        Some(silhouette_score(&space, &labels).unwrap())
    );
    let hac = cq_hac(&space, &labels).unwrap();
    assert_eq!(br.cq_hac_hom, Some(hac.homogeneity));
    assert_eq!(br.cq_hac_ami, Some(hac.ami));
    assert_eq!(
        br.pairwise_distance_overlap_error,
        Some(pairwise_distance_overlap_error(&space, &labels, cb.num_bins).unwrap())
    );

    for &target in &[
        MetricTarget::DistanceConsistency,
        MetricTarget::DistributionConsistency,
        MetricTarget::CalinskiHarabaszScore,
        MetricTarget::DaviesBouldinScore,
        MetricTarget::CqKmeansAri,
    ] {
        assert!(matches!(
            evaluate_metrics_with_communities(&g, &space, &[target], &cfg, Some(&labels)),
            Err(CoreError::UnsupportedOperation(_))
        ));
    }
}
