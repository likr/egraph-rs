use egraph_core::{CoreError, CoreResult, ManifoldLayout2d, Scalar};
use egraph_metrics_cluster::cluster_baselines::{
    calinski_harabasz_score_manifold, cq_hac_manifold, cq_kmeans_manifold,
    davies_bouldin_score_manifold, distance_consistency_manifold,
    distribution_consistency_manifold, neighborhood_hit_manifold,
    pairwise_distance_overlap_error_manifold, sepme_gab_manifold, sepme_gong_manifold,
    sepme_knng_manifold, sepme_rng_manifold, silhouette_score_manifold,
};

use super::config::{MetricConfig, MetricInputs};
use super::report::{MetricReport, RequestedFlags};

fn require_labels<'a>(
    inputs: &'a MetricInputs<'a, impl Scalar>,
    target: &str,
) -> CoreResult<&'a [u32]> {
    inputs
        .labels
        .ok_or_else(|| CoreError::invalid_input(format!("{target} target requires labels")))
}

pub(super) fn evaluate_eval_baseline_targets_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    flags: &RequestedFlags,
    config: &MetricConfig,
    inputs: &MetricInputs<'_, S>,
    report: &mut MetricReport,
) -> CoreResult<()> {
    if flags.distance_consistency {
        let labels = require_labels(inputs, "DistanceConsistency")?;
        report.distance_consistency = Some(distance_consistency_manifold(layout, labels)?);
    }
    if flags.distribution_consistency {
        let labels = require_labels(inputs, "DistributionConsistency")?;
        report.distribution_consistency = Some(distribution_consistency_manifold(
            layout,
            labels,
            config.cluster_baselines.dc_grid_size,
        )?);
    }
    if flags.sepme_knng {
        let labels = require_labels(inputs, "SepmeKnng")?;
        report.sepme_knng = Some(sepme_knng_manifold(
            layout,
            labels,
            config.cluster_baselines.sepme_knng_k,
        )?);
    }
    if flags.sepme_rng {
        let labels = require_labels(inputs, "SepmeRng")?;
        report.sepme_rng = Some(sepme_rng_manifold(layout, labels)?);
    }
    if flags.sepme_gab {
        let labels = require_labels(inputs, "SepmeGab")?;
        report.sepme_gab = Some(sepme_gab_manifold(layout, labels)?);
    }
    if flags.sepme_gong {
        let labels = require_labels(inputs, "SepmeGong")?;
        report.sepme_gong = Some(sepme_gong_manifold(
            layout,
            labels,
            config.cluster_baselines.gong_gamma,
        )?);
    }
    if flags.neighborhood_hit {
        let labels = require_labels(inputs, "NeighborhoodHit")?;
        report.neighborhood_hit = Some(neighborhood_hit_manifold(
            layout,
            labels,
            config.cluster_baselines.neighborhood_hit_k,
        )?);
    }
    if flags.silhouette_score {
        let labels = require_labels(inputs, "SilhouetteScore")?;
        report.silhouette_score = Some(silhouette_score_manifold(layout, labels)?);
    }
    if flags.calinski_harabasz_score {
        let labels = require_labels(inputs, "CalinskiHarabaszScore")?;
        report.calinski_harabasz_score = Some(calinski_harabasz_score_manifold(layout, labels)?);
    }
    if flags.davies_bouldin_score {
        let labels = require_labels(inputs, "DaviesBouldinScore")?;
        report.davies_bouldin_score = Some(davies_bouldin_score_manifold(layout, labels)?);
    }
    if flags.cq_kmeans_ari || flags.cq_kmeans_ami {
        let labels = require_labels(inputs, "CqKmeans")?;
        let pa = cq_kmeans_manifold(
            layout,
            labels,
            config.cluster_baselines.kmeans_n_init,
            config.cluster_baselines.kmeans_max_iters,
            config.cluster_baselines.seed,
        )?;
        if flags.cq_kmeans_ari {
            report.cq_kmeans_ari = Some(pa.ari);
        }
        if flags.cq_kmeans_ami {
            report.cq_kmeans_ami = Some(pa.ami);
        }
    }
    if flags.cq_hac_hom || flags.cq_hac_ami {
        let labels = require_labels(inputs, "CqHac")?;
        let pa = cq_hac_manifold(layout, labels)?;
        if flags.cq_hac_hom {
            report.cq_hac_hom = Some(pa.homogeneity);
        }
        if flags.cq_hac_ami {
            report.cq_hac_ami = Some(pa.ami);
        }
    }
    if flags.pairwise_distance_overlap_error {
        let labels = require_labels(inputs, "PairwiseDistanceOverlapError")?;
        report.pairwise_distance_overlap_error = Some(pairwise_distance_overlap_error_manifold(
            layout,
            labels,
            config.cluster_baselines.num_bins,
        )?);
    }

    Ok(())
}
