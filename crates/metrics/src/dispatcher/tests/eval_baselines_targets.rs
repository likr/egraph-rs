use super::super::*;
use crate::{
    calinski_harabasz_score, class_density_measure, cq_hac, cq_kmeans, davies_bouldin_score,
    distance_consistency, distribution_consistency, histogram_density_measure, neighborhood_hit,
    nor_metrics, pairwise_distance_overlap_error, sepme_gab, sepme_gong, sepme_knng, sepme_rng,
    silhouette_score, sprawlter_nn, sprawlter_nn_radii,
};
use egraph_core::{CoreError, CsrGraph, Layout2d, NodeExtents2d, NodeShape2d};

#[test]
fn test_eval_baselines_dispatcher_targets_and_extents() {
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

    // Non-baseline request keeps baseline_metrics() as None.
    let non_base = evaluate_metrics(&g, &layout, &[MetricTarget::AspectRatio], &cfg).unwrap();
    assert!(non_base.baseline_metrics().is_none());

    // SprawlterNn works without labels; all other baseline fields remain None.
    let sprawl_only = evaluate_metrics(&g, &layout, &[MetricTarget::SprawlterNn], &cfg).unwrap();
    let sprawl_br = sprawl_only.baseline_metrics().unwrap();
    assert!(sprawl_br.sprawlter_nn.is_some());
    assert!(sprawl_br.distance_consistency.is_none());
    assert!(sprawl_br.cq_kmeans_ari.is_none());
    assert!(sprawl_br.nor_oo_nni.is_none());

    // Label-dependent targets error when labels is None.
    for &target in &[
        MetricTarget::DistanceConsistency,
        MetricTarget::DistributionConsistency,
        MetricTarget::ClassDensityMeasure,
        MetricTarget::HistogramDensityMeasure,
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
    ] {
        assert!(
            matches!(
                evaluate_metrics(&g, &layout, &[target], &cfg),
                Err(CoreError::InvalidInput(_))
            ),
            "expected InvalidInput when labels is None for {target:?}"
        );
    }

    // Request all eval-baselines targets and verify exact parity with standalone calls.
    let baseline_targets = [
        MetricTarget::DistanceConsistency,
        MetricTarget::DistributionConsistency,
        MetricTarget::ClassDensityMeasure,
        MetricTarget::HistogramDensityMeasure,
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
        MetricTarget::SprawlterNn,
    ];
    let rep =
        evaluate_metrics_with_communities(&g, &layout, &baseline_targets, &cfg, Some(&communities))
            .unwrap();
    let br = rep.baseline_metrics().unwrap();
    let cb = cfg.baseline_config();

    assert_eq!(
        br.distance_consistency,
        Some(distance_consistency(&layout, &communities).unwrap())
    );
    assert_eq!(
        br.distribution_consistency,
        Some(distribution_consistency(&layout, &communities, cb.dc_grid_size).unwrap())
    );
    let (cdm, hdm) =
        class_density_measure(&layout, &communities, cb.cdm_grid_size, cb.cdm_bandwidth).unwrap();
    assert_eq!(br.class_density_measure, Some(cdm));
    assert_eq!(br.histogram_density_measure, Some(hdm));
    assert_eq!(
        br.sepme_knng,
        Some(sepme_knng(&layout, &communities, cb.sepme_knng_k).unwrap())
    );
    assert_eq!(
        br.sepme_rng,
        Some(sepme_rng(&layout, &communities).unwrap())
    );
    assert_eq!(
        br.sepme_gab,
        Some(sepme_gab(&layout, &communities).unwrap())
    );
    assert_eq!(
        br.sepme_gong,
        Some(sepme_gong(&layout, &communities, cb.gong_gamma).unwrap())
    );
    assert_eq!(
        br.neighborhood_hit,
        Some(neighborhood_hit(&layout, &communities, cb.neighborhood_hit_k).unwrap())
    );
    assert_eq!(
        br.silhouette_score,
        Some(silhouette_score(&layout, &communities).unwrap())
    );
    assert_eq!(
        br.calinski_harabasz_score,
        Some(calinski_harabasz_score(&layout, &communities).unwrap())
    );
    assert_eq!(
        br.davies_bouldin_score,
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
    assert_eq!(br.cq_kmeans_ari, Some(km.ari));
    assert_eq!(br.cq_kmeans_ami, Some(km.ami));
    let hac = cq_hac(&layout, &communities).unwrap();
    assert_eq!(br.cq_hac_hom, Some(hac.homogeneity));
    assert_eq!(br.cq_hac_ami, Some(hac.ami));
    assert_eq!(
        br.pairwise_distance_overlap_error,
        Some(pairwise_distance_overlap_error(&layout, &communities, cb.num_bins).unwrap())
    );
    let radii = vec![cb.sprawlter_node_radius; layout.len()];
    assert_eq!(
        br.sprawlter_nn,
        Some(sprawlter_nn_radii(&layout, &radii).unwrap())
    );

    // Standalone HistogramDensityMeasure without ClassDensityMeasure.
    let hdm_only = evaluate_metrics_with_communities(
        &g,
        &layout,
        &[MetricTarget::HistogramDensityMeasure],
        &cfg,
        Some(&communities),
    )
    .unwrap();
    let hdm_br = hdm_only.baseline_metrics().unwrap();
    assert_eq!(
        hdm_br.histogram_density_measure,
        Some(histogram_density_measure(&layout, &communities, cb.cdm_grid_size).unwrap())
    );
    assert!(hdm_br.class_density_measure.is_none());

    // NorMetrics + per-node NodeExtents2d via MetricInputs builders.
    let adjusted = Layout2d::from_xy(
        vec![-0.5, 0.5, 0.1, 4.5, 5.5, 5.1],
        vec![0.0, 0.2, -0.5, 5.0, 5.2, 4.5],
    )
    .unwrap();
    let hw = [0.2, 0.3, 0.25, 0.4, 0.2, 0.35];
    let hh = [0.15, 0.25, 0.2, 0.3, 0.15, 0.25];
    let extents = NodeExtents2d::new(&hw, &hh).unwrap();
    let inputs = MetricInputs::new()
        .with_original_layout(Some(&layout))
        .with_sprawlter_extents(Some(extents));
    let rep_nor = evaluate_metrics_with_inputs(
        &g,
        &adjusted,
        &[MetricTarget::NorMetrics, MetricTarget::SprawlterNn],
        &cfg,
        &inputs,
    )
    .unwrap();
    let nor_br = rep_nor.baseline_metrics().unwrap();
    let exp_nor = nor_metrics(&g, &layout, &adjusted, extents).unwrap();
    assert_eq!(nor_br.nor_oo_nni, Some(exp_nor.oo_nni));
    assert_eq!(nor_br.nor_sp_ch_a, Some(exp_nor.sp_ch_a));
    assert_eq!(nor_br.nor_gs_bb_iar, Some(exp_nor.gs_bb_iar));
    assert_eq!(nor_br.nor_nm_dm_imse, Some(exp_nor.nm_dm_imse));
    assert_eq!(nor_br.nor_el_rsd, Some(exp_nor.el_rsd));
    assert_eq!(
        nor_br.sprawlter_nn,
        Some(sprawlter_nn(&adjusted, extents).unwrap())
    );

    // Heterogeneous NodeShape2d extents also flow cleanly through MetricInputs.
    let shapes = [
        NodeShape2d::rectangle(0.2, 0.15).unwrap(),
        NodeShape2d::circle(0.25).unwrap(),
        NodeShape2d::rectangle(0.25, 0.2).unwrap(),
        NodeShape2d::circle(0.3).unwrap(),
        NodeShape2d::rectangle(0.2, 0.15).unwrap(),
        NodeShape2d::circle(0.35).unwrap(),
    ];
    let shape_extents = NodeExtents2d::from_shapes(&shapes).unwrap();
    let shape_inputs = MetricInputs::new().with_sprawlter_extents(Some(shape_extents));
    let rep_shape = evaluate_metrics_with_inputs(
        &g,
        &adjusted,
        &[MetricTarget::SprawlterNn],
        &cfg,
        &shape_inputs,
    )
    .unwrap();
    assert_eq!(
        rep_shape.baseline_metrics().unwrap().sprawlter_nn,
        Some(sprawlter_nn(&adjusted, shape_extents).unwrap())
    );

    assert!(matches!(
        evaluate_metrics(&g, &adjusted, &[MetricTarget::NorMetrics], &cfg),
        Err(CoreError::InvalidInput(_))
    ));

    // Lazy baseline config validation: invalid ClusterBaselineConfig ignored unless baseline targets requested.
    let bad_cb_cfg =
        MetricConfig::new().with_baseline_config(ClusterBaselineConfig::new().with_gong_gamma(2.0));
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
