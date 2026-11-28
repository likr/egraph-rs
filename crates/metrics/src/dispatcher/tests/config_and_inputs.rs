use super::super::*;
use crate::{
    ClusterAreaConfig, crossing_angle, crossing_edges, crossing_edges_torus, crosslessness,
    prism_distortion, sampled_crossing_number, sampled_neighborhood_preservation_1hop,
    sampled_neighborhood_preservation_2hop,
};
use egraph_core::{CoreError, CsrGraph, Layout2d, ManifoldKind, ManifoldLayout2d, NodeExtents2d};

#[test]
fn test_config_and_inputs_builders_feature_additive() {
    let cb = ClusterBaselineConfig::new()
        .with_sepme_knng_k(3)
        .with_neighborhood_hit_k(5)
        .with_dc_grid_size(12)
        .with_cdm_grid_size(32)
        .with_cdm_bandwidth(Some(0.25))
        .with_num_bins(50)
        .with_gong_gamma(0.4)
        .with_kmeans_n_init(3)
        .with_kmeans_max_iters(50)
        .with_seed(99)
        .with_sprawlter_node_radius(1.5);
    assert!(cb.validate().is_ok());
    assert_eq!(cb.sepme_knng_k, 3);
    assert_eq!(cb.neighborhood_hit_k, 5);
    assert_eq!(cb.dc_grid_size, 12);
    assert_eq!(cb.cdm_grid_size, 32);
    assert_eq!(cb.cdm_bandwidth, Some(0.25));
    assert_eq!(cb.num_bins, 50);
    assert_eq!(cb.gong_gamma, 0.4);
    assert_eq!(cb.kmeans_n_init, 3);
    assert_eq!(cb.kmeans_max_iters, 50);
    assert_eq!(cb.seed, 99);
    assert_eq!(cb.sprawlter_node_radius, 1.5);

    let cfg = MetricConfig::new()
        .with_sns_num_pivots(16)
        .with_sns_seed(7)
        .with_crossing_num_samples(128)
        .with_crossing_seed(8)
        .with_np_num_samples(256)
        .with_np_seed(9)
        .with_ideal_edge_length(2.0)
        .with_rta_num_landmarks(8)
        .with_rta_num_triplets(1000)
        .with_rta_min_degree(1)
        .with_rta_seed(10)
        .with_patch_nig_radius(4)
        .with_patch_nig_canvas_size(512)
        .with_patch_nig_pad_frac(0.05)
        .with_patch_nig_seed(11)
        .with_cluster_area(ClusterAreaConfig::new().with_g_max(64))
        .with_baseline_config(cb.clone());
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.baseline_config(), &cb);
    assert_eq!(cfg.np_num_samples, 256);
    assert_eq!(cfg.np_seed, 9);
    assert_eq!(cfg.crossing_num_samples, 128);
    assert_eq!(cfg.crossing_seed, 8);

    let empty_report = MetricReport::new();
    assert!(empty_report.baseline_metrics().is_none());
    let empty_br = ClusterBaselineReport::new();
    assert!(empty_br.silhouette_score.is_none());

    let layout = Layout2d::from_xy(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
    let labels = [0_u32, 1];
    let nw = [1.0, 2.0];
    let cw = [1.0, 2.0];
    let ext = NodeExtents2d::uniform(2, 0.5, 0.5).unwrap();
    let inputs = MetricInputs::new()
        .with_labels(Some(&labels))
        .with_node_weights(Some(&nw))
        .with_cluster_weights(Some(&cw))
        .with_original_layout(Some(&layout))
        .with_sprawlter_extents(Some(ext));
    assert_eq!(inputs.labels, Some(&labels[..]));
    assert_eq!(inputs.node_weights, Some(&nw[..]));
    assert_eq!(inputs.cluster_weights, Some(&cw[..]));
    assert!(inputs.original_layout.is_some());
    assert!(inputs.sprawlter_extents.is_some());
    assert!(format!("{inputs:?}").contains("MetricInputs"));
}

#[test]
fn test_np_sampling_config_independence_and_lazy_validation() {
    let g = CsrGraph::<f64>::from_edges(
        6,
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (0, 3),
            (1, 4),
        ],
        None,
        false,
        false,
    )
    .unwrap();
    let layout = Layout2d::from_xy(
        vec![0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
        vec![0.0, 0.5, 0.0, 1.0, 1.5, 1.0],
    )
    .unwrap();
    let communities = [0_u32, 0, 0, 1, 1, 1];
    let cfg = MetricConfig::new()
        .with_sns_num_pivots(2)
        .with_np_num_samples(64)
        .with_crossing_num_samples(64);

    let cfg_a = cfg
        .clone()
        .with_np_num_samples(2)
        .with_np_seed(123)
        .with_crossing_num_samples(10)
        .with_crossing_seed(777);
    let cfg_b = cfg
        .clone()
        .with_np_num_samples(2)
        .with_np_seed(123)
        .with_crossing_num_samples(500)
        .with_crossing_seed(999);
    let rep_a = evaluate_metrics(
        &g,
        &layout,
        &[MetricTarget::SampledNp1, MetricTarget::SampledNp2],
        &cfg_a,
    )
    .unwrap();
    let rep_b = evaluate_metrics(
        &g,
        &layout,
        &[MetricTarget::SampledNp1, MetricTarget::SampledNp2],
        &cfg_b,
    )
    .unwrap();
    assert_eq!(rep_a, rep_b);
    assert_eq!(
        rep_a.sampled_np1,
        Some(sampled_neighborhood_preservation_1hop(&g, &layout, 2, 123).unwrap())
    );
    assert_eq!(
        rep_a.sampled_np2,
        Some(sampled_neighborhood_preservation_2hop(&g, &layout, 2, 123).unwrap())
    );

    let cfg_c = cfg
        .clone()
        .with_np_num_samples(4)
        .with_np_seed(888)
        .with_crossing_num_samples(16)
        .with_crossing_seed(55);
    let cfg_d = cfg
        .clone()
        .with_np_num_samples(1)
        .with_np_seed(1)
        .with_crossing_num_samples(16)
        .with_crossing_seed(55);
    let rep_c = evaluate_metrics(
        &g,
        &layout,
        &[
            MetricTarget::SampledCrossingNumber,
            MetricTarget::SampledCrosslessness,
        ],
        &cfg_c,
    )
    .unwrap();
    let rep_d = evaluate_metrics(
        &g,
        &layout,
        &[
            MetricTarget::SampledCrossingNumber,
            MetricTarget::SampledCrosslessness,
        ],
        &cfg_d,
    )
    .unwrap();
    assert_eq!(rep_c, rep_d);
    assert_eq!(
        rep_c.sampled_crossing_number,
        Some(sampled_crossing_number(&g, &layout, 16, 55).unwrap())
    );

    let bad_np_cfg = cfg.clone().with_np_num_samples(0);
    assert!(bad_np_cfg.validate().is_err());
    assert!(
        evaluate_metrics(
            &g,
            &layout,
            &[MetricTarget::SampledCrossingNumber],
            &bad_np_cfg
        )
        .is_ok()
    );
    assert!(matches!(
        evaluate_metrics(&g, &layout, &[MetricTarget::SampledNp1], &bad_np_cfg),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        evaluate_metrics(&g, &layout, &[MetricTarget::SampledNp2], &bad_np_cfg),
        Err(CoreError::InvalidInput(_))
    ));

    let bad_cross_cfg = cfg.clone().with_crossing_num_samples(0);
    assert!(bad_cross_cfg.validate().is_err());
    assert!(evaluate_metrics(&g, &layout, &[MetricTarget::SampledNp1], &bad_cross_cfg).is_ok());
    assert!(matches!(
        evaluate_metrics(
            &g,
            &layout,
            &[MetricTarget::SampledCrossingNumber],
            &bad_cross_cfg
        ),
        Err(CoreError::InvalidInput(_))
    ));

    let bad_caf_cfg = cfg
        .clone()
        .with_cluster_area(ClusterAreaConfig::new().with_g_max(4));
    assert!(bad_caf_cfg.validate().is_err());
    assert!(evaluate_metrics(&g, &layout, &[MetricTarget::AspectRatio], &bad_caf_cfg).is_ok());
    assert!(matches!(
        evaluate_metrics_with_communities(
            &g,
            &layout,
            &[MetricTarget::ClusterAreaFaithfulness],
            &bad_caf_cfg,
            Some(&communities),
        ),
        Err(CoreError::InvalidInput(_))
    ));

    let bad_sns_cfg = cfg.clone().with_sns_num_pivots(0);
    assert!(bad_sns_cfg.validate().is_err());
    assert!(evaluate_metrics(&g, &layout, &[MetricTarget::AspectRatio], &bad_sns_cfg).is_ok());
    assert!(matches!(
        evaluate_metrics(
            &g,
            &layout,
            &[MetricTarget::SampledScaleNormalizedStress],
            &bad_sns_cfg
        ),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_prism_and_torus_dispatcher_separation() {
    let g = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)],
        None,
        false,
        false,
    )
    .unwrap();
    let layout = Layout2d::from_xy(vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    let perturbed_l =
        Layout2d::from_xy(vec![0.0, 2.0, 1.0, 0.0], vec![0.0, 0.0, 1.5, 1.0]).unwrap();
    let cfg = MetricConfig::default();

    let inputs = MetricInputs::new().with_original_layout(Some(&layout));
    let rep_ext = evaluate_metrics_with_inputs(
        &g,
        &perturbed_l,
        &[
            MetricTarget::CrossingNumber,
            MetricTarget::CrossingEdges,
            MetricTarget::CrossingAngle,
            MetricTarget::Crosslessness,
            MetricTarget::PrismDistortion,
        ],
        &cfg,
        &inputs,
    )
    .unwrap();
    let expected_edges = crossing_edges(&g, &perturbed_l).unwrap();
    assert_eq!(rep_ext.crossing_number, Some(expected_edges.len()));
    assert_eq!(
        rep_ext.crossing_edges.as_deref(),
        Some(expected_edges.as_slice())
    );
    assert!(
        (rep_ext.crossing_angle.unwrap() - crossing_angle(&g, &perturbed_l).unwrap()).abs() < 1e-12
    );
    assert!(
        (rep_ext.crosslessness.unwrap() - crosslessness(&g, &perturbed_l).unwrap()).abs() < 1e-12
    );
    let expected_prism = prism_distortion(&layout, &perturbed_l).unwrap();
    assert_eq!(rep_ext.prism_sigma_dist, Some(expected_prism.sigma_dist));
    assert_eq!(rep_ext.prism_sigma_disp, Some(expected_prism.sigma_disp));
    assert!(matches!(
        evaluate_metrics(&g, &perturbed_l, &[MetricTarget::PrismDistortion], &cfg),
        Err(CoreError::InvalidInput(_))
    ));

    assert!(matches!(
        evaluate_metrics(&g, &layout, &[MetricTarget::TorusCrossingNumber], &cfg),
        Err(CoreError::UnsupportedOperation(_))
    ));
    let torus = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.1, 0.9, 0.2, 0.8],
        vec![0.1, 0.9, 0.8, 0.2],
    )
    .unwrap();
    let expected_torus = crossing_edges_torus(&g, &torus).unwrap();
    let rep_torus = evaluate_metrics(
        &g,
        &torus,
        &[
            MetricTarget::TorusCrossingNumber,
            MetricTarget::CrossingEdgesTorus,
        ],
        &cfg,
    )
    .unwrap();
    assert_eq!(rep_torus.torus_crossing_number, Some(expected_torus.len()));
    assert_eq!(
        rep_torus.crossing_edges_torus.as_deref(),
        Some(expected_torus.as_slice())
    );
}
