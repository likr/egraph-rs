//! Pure-Rust unit tests for layout quality metric dispatch helpers.

use egraph::metrics::MetricTarget;

use super::dispatcher::parse_metric_target;

#[test]
fn test_metric_target_aliases() {
    assert_eq!(
        parse_metric_target("sns").unwrap(),
        MetricTarget::ScaleNormalizedStress
    );
    assert_eq!(
        parse_metric_target("ScaleNormalizedStress").unwrap(),
        MetricTarget::ScaleNormalizedStress
    );
    assert_eq!(
        parse_metric_target("sampled_sns").unwrap(),
        MetricTarget::SampledScaleNormalizedStress
    );
    assert_eq!(
        parse_metric_target("np1").unwrap(),
        MetricTarget::NeighborhoodPreservation1Hop
    );
    assert_eq!(
        parse_metric_target("np2").unwrap(),
        MetricTarget::NeighborhoodPreservation2Hop
    );
    assert_eq!(
        parse_metric_target("drng").unwrap(),
        MetricTarget::RelativeNeighborhoodPreservation
    );
    assert_eq!(
        parse_metric_target("dgg").unwrap(),
        MetricTarget::GabrielGraphPreservation
    );
    for alias in [
        "cluster_area_faithfulness",
        "cluster-area-faithfulness",
        "ClusterAreaFaithfulness",
        "cluster_area",
        "gcaf",
        "g-caf",
    ] {
        assert_eq!(
            parse_metric_target(alias).unwrap(),
            MetricTarget::ClusterAreaFaithfulness
        );
    }
    assert!(parse_metric_target("not_a_metric").is_err());
}

#[test]
fn test_cluster_baseline_metric_target_aliases() {
    let cases = [
        ("dsc", MetricTarget::DistanceConsistency),
        ("distance_consistency", MetricTarget::DistanceConsistency),
        ("dc", MetricTarget::DistributionConsistency),
        (
            "distribution_consistency",
            MetricTarget::DistributionConsistency,
        ),
        ("cdm", MetricTarget::ClassDensityMeasure),
        ("class_density_measure", MetricTarget::ClassDensityMeasure),
        ("hdm", MetricTarget::HistogramDensityMeasure),
        ("2d-hdm", MetricTarget::HistogramDensityMeasure),
        (
            "histogram_density_measure",
            MetricTarget::HistogramDensityMeasure,
        ),
        ("sepme_knng", MetricTarget::SepmeKnng),
        ("sepme-knng", MetricTarget::SepmeKnng),
        ("sepme_knng2", MetricTarget::SepmeKnng),
        ("sepme-knng2", MetricTarget::SepmeKnng),
        ("sepme_rng", MetricTarget::SepmeRng),
        ("sepme_gab", MetricTarget::SepmeGab),
        ("sepme_gong", MetricTarget::SepmeGong),
        ("neighborhood_hit", MetricTarget::NeighborhoodHit),
        ("m_nh", MetricTarget::NeighborhoodHit),
        ("silhouette", MetricTarget::SilhouetteScore),
        ("silhouette_score", MetricTarget::SilhouetteScore),
        ("calinski_harabasz", MetricTarget::CalinskiHarabaszScore),
        ("ch", MetricTarget::CalinskiHarabaszScore),
        ("davies_bouldin", MetricTarget::DaviesBouldinScore),
        ("db", MetricTarget::DaviesBouldinScore),
        ("cq_kmeans_ari", MetricTarget::CqKmeansAri),
        ("cq_kmeans_ami", MetricTarget::CqKmeansAmi),
        ("cq_hac_hom", MetricTarget::CqHacHom),
        ("cq_hac_ami", MetricTarget::CqHacAmi),
        (
            "pairwise_distance_overlap_error",
            MetricTarget::PairwiseDistanceOverlapError,
        ),
        ("pdoe", MetricTarget::PairwiseDistanceOverlapError),
        ("sprawlter_nn", MetricTarget::SprawlterNn),
    ];
    for (raw, expected) in cases {
        assert_eq!(parse_metric_target(raw).unwrap(), expected, "raw={raw}");
    }
}
