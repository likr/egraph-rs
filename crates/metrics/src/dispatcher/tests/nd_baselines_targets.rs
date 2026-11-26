use super::super::*;
use crate::ClusterAreaConfig;
use crate::cluster_baselines::{
    calinski_harabasz_score_nd, cq_hac_nd, cq_kmeans_nd, davies_bouldin_score_nd,
    distance_consistency_nd, distribution_consistency_nd, neighborhood_hit_nd,
    pairwise_distance_overlap_error_nd, sepme_gab_nd, sepme_gong_nd, sepme_knng_nd, sepme_rng_nd,
    silhouette_score_nd,
};
use egraph_core::{CoreError, CsrGraph, Layout2d, LayoutNd};

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

#[test]
fn test_nd_dispatcher_eval_baselines_parity_and_errors() {
    let g = make_sample_graph_6();
    let labels = [0_u32, 0, 0, 1, 1, 1];
    let cfg = baseline_test_cfg();

    let baseline_nd_targets = [
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

    // dims == 2 parity against Layout2d.
    let xs = vec![0.0, 0.2, 0.1, 5.0, 5.2, 5.1];
    let ys = vec![0.0, 0.1, -0.1, 5.0, 5.1, 4.9];
    let l2d = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
    let mut flat2 = Vec::with_capacity(12);
    for i in 0..6 {
        flat2.push(xs[i]);
        flat2.push(ys[i]);
    }
    let lnd2 = LayoutNd::from_flat(6, 2, flat2).unwrap();
    let rep2d =
        evaluate_metrics_with_communities(&g, &l2d, &baseline_nd_targets, &cfg, Some(&labels))
            .unwrap();
    let repnd2 =
        evaluate_metrics_with_communities(&g, &lnd2, &baseline_nd_targets, &cfg, Some(&labels))
            .unwrap();
    assert_eq!(repnd2, rep2d);

    // 3D and 10D parity with standalone *_nd functions + community_weights wrapper.
    for &dims in &[3usize, 10] {
        let mut flat = Vec::with_capacity(6 * dims);
        for i in 0..6 {
            let base = if i < 3 { 0.0 } else { 6.0 };
            for d in 0..dims {
                flat.push(base + ((i * 5 + d * 2) as f64) * 0.1);
            }
        }
        let lnd = LayoutNd::from_flat(6, dims, flat).unwrap();
        let rep = evaluate_metrics_nd_with_communities(
            &g,
            &lnd,
            &baseline_nd_targets,
            &cfg,
            Some(&labels),
        )
        .unwrap();
        let rep_cw = evaluate_metrics_with_community_weights(
            &g,
            &lnd,
            &baseline_nd_targets,
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
            Some(distance_consistency_nd(&lnd, &labels).unwrap())
        );
        assert_eq!(
            rep.distribution_consistency,
            Some(distribution_consistency_nd(&lnd, &labels, cb.dc_grid_size).unwrap())
        );
        assert_eq!(
            rep.sepme_knng,
            Some(sepme_knng_nd(&lnd, &labels, cb.sepme_knng_k).unwrap())
        );
        assert_eq!(rep.sepme_rng, Some(sepme_rng_nd(&lnd, &labels).unwrap()));
        assert_eq!(rep.sepme_gab, Some(sepme_gab_nd(&lnd, &labels).unwrap()));
        assert_eq!(
            rep.sepme_gong,
            Some(sepme_gong_nd(&lnd, &labels, cb.gong_gamma).unwrap())
        );
        assert_eq!(
            rep.neighborhood_hit,
            Some(neighborhood_hit_nd(&lnd, &labels, cb.neighborhood_hit_k).unwrap())
        );
        assert_eq!(
            rep.silhouette_score,
            Some(silhouette_score_nd(&lnd, &labels).unwrap())
        );
        assert_eq!(
            rep.calinski_harabasz_score,
            Some(calinski_harabasz_score_nd(&lnd, &labels).unwrap())
        );
        assert_eq!(
            rep.davies_bouldin_score,
            Some(davies_bouldin_score_nd(&lnd, &labels).unwrap())
        );
        let km = cq_kmeans_nd(
            &lnd,
            &labels,
            cb.kmeans_n_init,
            cb.kmeans_max_iters,
            cb.seed,
        )
        .unwrap();
        assert_eq!(rep.cq_kmeans_ari, Some(km.ari));
        assert_eq!(rep.cq_kmeans_ami, Some(km.ami));
        let hac = cq_hac_nd(&lnd, &labels).unwrap();
        assert_eq!(rep.cq_hac_hom, Some(hac.homogeneity));
        assert_eq!(rep.cq_hac_ami, Some(hac.ami));
        assert_eq!(
            rep.pairwise_distance_overlap_error,
            Some(pairwise_distance_overlap_error_nd(&lnd, &labels, cb.num_bins).unwrap())
        );
    }

    // 2D-only eval-baselines targets return UnsupportedOperation.
    let l3 = LayoutNd::<f64>::init(6, 3, 19, 1.0).unwrap();
    for &target in &[
        MetricTarget::ClassDensityMeasure,
        MetricTarget::HistogramDensityMeasure,
        MetricTarget::SprawlterNn,
        MetricTarget::NorMetrics,
    ] {
        assert!(matches!(
            evaluate_metrics_with_communities(&g, &l3, &[target], &cfg, Some(&labels)),
            Err(CoreError::UnsupportedOperation(_))
        ));
    }

    // Missing labels on N-D baseline target -> InvalidInput.
    for &target in &[
        MetricTarget::DistanceConsistency,
        MetricTarget::SepmeKnng,
        MetricTarget::SilhouetteScore,
        MetricTarget::CqKmeansAri,
    ] {
        assert!(matches!(
            evaluate_metrics(&g, &l3, &[target], &cfg),
            Err(CoreError::InvalidInput(_))
        ));
    }
}
