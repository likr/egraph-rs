use egraph_core::{CoreError, CoreResult, LayoutNd, Scalar};
use egraph_metrics_cluster::cluster_baselines::{
    build_dynamic_kdtree_f64, calinski_harabasz_score_nd, cq_hac_nd, cq_kmeans_nd,
    davies_bouldin_score_nd, distance_consistency_nd, distribution_consistency_nd,
    neighborhood_hit_nd_with_tree, pairwise_distance_overlap_error_nd, sepme_gab_nd_with_tree,
    sepme_gong_nd_with_tree, sepme_knng_nd_with_tree, sepme_rng_nd_with_tree, silhouette_score_nd,
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

#[inline]
fn needs_dynamic_kdtree(flags: &RequestedFlags) -> bool {
    flags.sepme_knng
        || flags.sepme_rng
        || flags.sepme_gab
        || flags.sepme_gong
        || flags.neighborhood_hit
}

fn require_labels_for_kdtree<'a>(
    flags: &RequestedFlags,
    inputs: &'a MetricInputs<'a, impl Scalar>,
) -> CoreResult<&'a [u32]> {
    let target = if flags.sepme_knng {
        "SepmeKnng"
    } else if flags.sepme_rng {
        "SepmeRng"
    } else if flags.sepme_gab {
        "SepmeGab"
    } else if flags.sepme_gong {
        "SepmeGong"
    } else {
        "NeighborhoodHit"
    };
    require_labels(inputs, target)
}

pub(super) fn evaluate_eval_baseline_targets_nd<S: Scalar>(
    layout: &LayoutNd<S>,
    flags: &RequestedFlags,
    config: &MetricConfig,
    inputs: &MetricInputs<'_, S>,
    report: &mut MetricReport,
) -> CoreResult<()> {
    if flags.distance_consistency {
        let labels = require_labels(inputs, "DistanceConsistency")?;
        report.distance_consistency = Some(distance_consistency_nd(layout, labels)?);
    }
    if flags.distribution_consistency {
        let labels = require_labels(inputs, "DistributionConsistency")?;
        report.distribution_consistency = Some(distribution_consistency_nd(
            layout,
            labels,
            config.cluster_baselines.dc_grid_size,
        )?);
    }

    if needs_dynamic_kdtree(flags) {
        let labels = require_labels_for_kdtree(flags, inputs)?;
        let tree = build_dynamic_kdtree_f64(layout)?;
        if flags.sepme_knng {
            report.sepme_knng = Some(sepme_knng_nd_with_tree(
                layout,
                labels,
                config.cluster_baselines.sepme_knng_k,
                &tree,
            )?);
        }
        if flags.sepme_rng {
            report.sepme_rng = Some(sepme_rng_nd_with_tree(layout, labels, &tree)?);
        }
        if flags.sepme_gab {
            report.sepme_gab = Some(sepme_gab_nd_with_tree(layout, labels, &tree)?);
        }
        if flags.sepme_gong {
            report.sepme_gong = Some(sepme_gong_nd_with_tree(
                layout,
                labels,
                config.cluster_baselines.gong_gamma,
                &tree,
            )?);
        }
        if flags.neighborhood_hit {
            report.neighborhood_hit = Some(neighborhood_hit_nd_with_tree(
                layout,
                labels,
                config.cluster_baselines.neighborhood_hit_k,
                &tree,
            )?);
        }
    }

    if flags.silhouette_score {
        let labels = require_labels(inputs, "SilhouetteScore")?;
        report.silhouette_score = Some(silhouette_score_nd(layout, labels)?);
    }
    if flags.calinski_harabasz_score {
        let labels = require_labels(inputs, "CalinskiHarabaszScore")?;
        report.calinski_harabasz_score = Some(calinski_harabasz_score_nd(layout, labels)?);
    }
    if flags.davies_bouldin_score {
        let labels = require_labels(inputs, "DaviesBouldinScore")?;
        report.davies_bouldin_score = Some(davies_bouldin_score_nd(layout, labels)?);
    }
    if flags.cq_kmeans_ari || flags.cq_kmeans_ami {
        let labels = require_labels(inputs, "CqKmeans")?;
        let pa = cq_kmeans_nd(
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
        let pa = cq_hac_nd(layout, labels)?;
        if flags.cq_hac_hom {
            report.cq_hac_hom = Some(pa.homogeneity);
        }
        if flags.cq_hac_ami {
            report.cq_hac_ami = Some(pa.ami);
        }
    }
    if flags.pairwise_distance_overlap_error {
        let labels = require_labels(inputs, "PairwiseDistanceOverlapError")?;
        report.pairwise_distance_overlap_error = Some(pairwise_distance_overlap_error_nd(
            layout,
            labels,
            config.cluster_baselines.num_bins,
        )?);
    }

    Ok(())
}
