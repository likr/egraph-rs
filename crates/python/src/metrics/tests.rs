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
        ("dsc", MetricTarget::Dsc),
        ("distance_consistency", MetricTarget::Dsc),
        ("dc", MetricTarget::Dc),
        ("distribution_consistency", MetricTarget::Dc),
        ("cdm", MetricTarget::Cdm),
        ("class_density_measure", MetricTarget::Cdm),
        ("hdm", MetricTarget::Hdm),
        ("2d-hdm", MetricTarget::Hdm),
        ("histogram_density_measure", MetricTarget::Hdm),
        ("sepme_knng", MetricTarget::SepmeKnng),
        ("sepme-knng", MetricTarget::SepmeKnng),
        ("sepme_knng2", MetricTarget::SepmeKnng),
        ("sepme-knng2", MetricTarget::SepmeKnng),
        ("sepme_rng", MetricTarget::SepmeRng),
        ("sepme_gab", MetricTarget::SepmeGab),
        ("sepme_gong", MetricTarget::SepmeGong),
        ("neighborhood_hit", MetricTarget::NeighborhoodHit),
        ("m_nh", MetricTarget::NeighborhoodHit),
        ("silhouette", MetricTarget::Silhouette),
        ("silhouette_score", MetricTarget::Silhouette),
        ("calinski_harabasz", MetricTarget::CalinskiHarabasz),
        ("ch", MetricTarget::CalinskiHarabasz),
        ("davies_bouldin", MetricTarget::DaviesBouldin),
        ("db", MetricTarget::DaviesBouldin),
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
