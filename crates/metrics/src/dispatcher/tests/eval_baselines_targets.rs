use super::super::*;
use crate::cluster_baselines::{
    calinski_harabasz_score, class_density_measure, cq_hac, cq_kmeans, davies_bouldin_score,
    distance_consistency, distribution_consistency, neighborhood_hit,
    pairwise_distance_overlap_error, sepme_gab, sepme_gong, sepme_knng, sepme_rng,
    silhouette_score, sprawlter_nn_radii,
};
use egraph_core::{CoreError, CsrGraph, Layout2d};

#[test]
fn test_eval_baselines_dispatcher_targets_and_lazy_validation() {
    let g = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)],
        None,
        false,
        false,
    )
    .unwrap();
    let layout = Layout2d::from_xy(
        vec![0.0, 0.2, 0.1, 5.0, 5.2, 5.1],
        vec![0.0, 0.1, -0.1, 5.0, 5.1, 4.9],
    )
    .unwrap();
    let communities = [0_u32, 0, 0, 1, 1, 1];
    let cfg = MetricConfig::default();

    // SprawlterNn works without community_labels; all other eval-baselines fields remain None.
    let sprawl_only = evaluate_metrics(&g, &layout, &[MetricTarget::SprawlterNn], &cfg).unwrap();
    assert!(sprawl_only.sprawlter_nn.is_some());
    assert!(sprawl_only.dsc.is_none());
    assert!(sprawl_only.cq_kmeans_ari.is_none());

    // Label-dependent targets error when community_labels is None.
    for &target in &[
        MetricTarget::Dsc,
        MetricTarget::Dc,
        MetricTarget::Cdm,
        MetricTarget::Hdm,
        MetricTarget::SepmeKnng,
        MetricTarget::SepmeRng,
        MetricTarget::SepmeGab,
        MetricTarget::SepmeGong,
        MetricTarget::NeighborhoodHit,
        MetricTarget::Silhouette,
        MetricTarget::CalinskiHarabasz,
        MetricTarget::DaviesBouldin,
        MetricTarget::CqKmeansAri,
        MetricTarget::CqKmeansAmi,
        MetricTarget::CqHacHom,
        MetricTarget::CqHacAmi,
        MetricTarget::PairwiseDistanceOverlapError,
    ] {
        assert!(
            matches!(
                evaluate_metrics(&g, &layout, &[target], &cfg),
                Err(CoreError::InvalidInput(_))
            ),
            "expected InvalidInput when community_labels is None for {target:?}"
        );
    }

    // Request all 18 eval-baselines targets and verify exact parity with standalone calls.
    let baseline_targets = [
        MetricTarget::Dsc,
        MetricTarget::Dc,
        MetricTarget::Cdm,
        MetricTarget::Hdm,
        MetricTarget::SepmeKnng,
        MetricTarget::SepmeRng,
        MetricTarget::SepmeGab,
        MetricTarget::SepmeGong,
        MetricTarget::NeighborhoodHit,
        MetricTarget::Silhouette,
        MetricTarget::CalinskiHarabasz,
        MetricTarget::DaviesBouldin,
        MetricTarget::CqKmeansAri,
        MetricTarget::CqKmeansAmi,
        MetricTarget::CqHacHom,
        MetricTarget::CqHacAmi,
        MetricTarget::PairwiseDistanceOverlapError,
        MetricTarget::SprawlterNn,
    ];
    let rep =
        evaluate_metrics_with_communities(&g, &layout, &baseline_targets, &cfg, Some(&communities))
            .unwrap();

    let cb = &cfg.cluster_baselines;
    assert_eq!(
        rep.dsc,
        Some(distance_consistency(&layout, &communities).unwrap())
    );
    assert_eq!(
        rep.dc,
        Some(distribution_consistency(&layout, &communities, cb.dc_grid_size).unwrap())
    );
    let (cdm, hdm) =
        class_density_measure(&layout, &communities, cb.cdm_grid_size, cb.cdm_bandwidth).unwrap();
    assert_eq!(rep.cdm, Some(cdm));
    assert_eq!(rep.hdm, Some(hdm));
    assert_eq!(
        rep.sepme_knng,
        Some(sepme_knng(&layout, &communities, cb.sepme_knng_k).unwrap())
    );
    assert_eq!(
        rep.sepme_rng,
        Some(sepme_rng(&layout, &communities).unwrap())
    );
    assert_eq!(
        rep.sepme_gab,
        Some(sepme_gab(&layout, &communities).unwrap())
    );
    assert_eq!(
        rep.sepme_gong,
        Some(sepme_gong(&layout, &communities, cb.gong_gamma).unwrap())
    );
    assert_eq!(
        rep.neighborhood_hit,
        Some(neighborhood_hit(&layout, &communities, cb.neighborhood_hit_k).unwrap())
    );
    assert_eq!(
        rep.silhouette,
        Some(silhouette_score(&layout, &communities).unwrap())
    );
    assert_eq!(
        rep.calinski_harabasz,
        Some(calinski_harabasz_score(&layout, &communities).unwrap())
    );
    assert_eq!(
        rep.davies_bouldin,
        Some(davies_bouldin_score(&layout, &communities).unwrap())
    );
    let km = cq_kmeans(
        &layout,
        &communities,
        cb.kmeans_n_init,
        cb.kmeans_max_iters,
        cb.seed,
    )
    .unwrap();
    assert_eq!(rep.cq_kmeans_ari, Some(km.ari));
    assert_eq!(rep.cq_kmeans_ami, Some(km.ami));
    let hac = cq_hac(&layout, &communities).unwrap();
    assert_eq!(rep.cq_hac_hom, Some(hac.homogeneity));
    assert_eq!(rep.cq_hac_ami, Some(hac.ami));
    assert_eq!(
        rep.pairwise_distance_overlap_error,
        Some(pairwise_distance_overlap_error(&layout, &communities, cb.num_bins).unwrap())
    );
    let radii = vec![cb.sprawlter_node_radius; layout.len()];
    assert_eq!(
        rep.sprawlter_nn,
        Some(sprawlter_nn_radii(&layout, &radii).unwrap())
    );

    // Lazy config validation: invalid baseline config ignored when baseline targets not requested.
    let bad_cb_cfg = MetricConfig {
        cluster_baselines: ClusterBaselineConfig {
            gong_gamma: 2.0,
            ..cfg.cluster_baselines
        },
        ..cfg
    };
    assert!(bad_cb_cfg.validate().is_err());
    assert!(evaluate_metrics(&g, &layout, &[MetricTarget::AspectRatio], &bad_cb_cfg).is_ok());
    assert!(matches!(
        evaluate_metrics_with_communities(
            &g,
            &layout,
            &[MetricTarget::SepmeGong],
            &bad_cb_cfg,
            Some(&communities),
        ),
        Err(CoreError::InvalidInput(_))
    ));
}
