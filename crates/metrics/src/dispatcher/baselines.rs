use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar};
use egraph_metrics_cluster::cluster_baselines::{
    calinski_harabasz_score, class_density_measure, cq_hac, cq_kmeans, davies_bouldin_score,
    distance_consistency, distribution_consistency, histogram_density_measure, nor_metrics,
    pairwise_distance_overlap_error, silhouette_score, sprawlter_nn,
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

fn resolve_node_half_extents<'a, S: Scalar>(
    n: usize,
    config: &MetricConfig,
    inputs: &'a MetricInputs<'a, S>,
    scratch: &'a mut Vec<S>,
) -> CoreResult<(&'a [S], &'a [S])> {
    match (inputs.sprawlter_half_widths, inputs.sprawlter_half_heights) {
        (Some(hw), Some(hh)) => Ok((hw, hh)),
        (Some(_), None) | (None, Some(_)) => Err(CoreError::invalid_input(
            "sprawlter_half_widths and sprawlter_half_heights must be provided together",
        )),
        (None, None) => {
            let r_f64 = config.cluster_baselines.sprawlter_node_radius;
            if !r_f64.is_finite() {
                return Err(CoreError::non_finite(
                    "ClusterBaselineConfig::sprawlter_node_radius",
                ));
            }
            if r_f64 <= 0.0 {
                return Err(CoreError::invalid_input(
                    "ClusterBaselineConfig::sprawlter_node_radius must be strictly positive",
                ));
            }
            scratch.resize(n, S::from_f64(r_f64));
            Ok((scratch.as_slice(), scratch.as_slice()))
        }
    }
}

pub(super) fn evaluate_eval_baseline_targets<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    flags: &RequestedFlags,
    config: &MetricConfig,
    inputs: &MetricInputs<'_, S>,
    report: &mut MetricReport,
) -> CoreResult<()> {
    if flags.distance_consistency {
        let labels = require_labels(inputs, "DistanceConsistency")?;
        report.distance_consistency = Some(distance_consistency(layout, labels)?);
    }
    if flags.distribution_consistency {
        let labels = require_labels(inputs, "DistributionConsistency")?;
        report.distribution_consistency = Some(distribution_consistency(
            layout,
            labels,
            config.cluster_baselines.dc_grid_size,
        )?);
    }
    if flags.class_density_measure {
        let labels = require_labels(inputs, "ClassDensityMeasure")?;
        let (cdm, hdm) = class_density_measure(
            layout,
            labels,
            config.cluster_baselines.cdm_grid_size,
            config.cluster_baselines.cdm_bandwidth,
        )?;
        report.class_density_measure = Some(cdm);
        if flags.histogram_density_measure {
            report.histogram_density_measure = Some(hdm);
        }
    } else if flags.histogram_density_measure {
        let labels = require_labels(inputs, "HistogramDensityMeasure")?;
        report.histogram_density_measure = Some(histogram_density_measure(
            layout,
            labels,
            config.cluster_baselines.cdm_grid_size,
        )?);
    }
    if flags.silhouette_score {
        let labels = require_labels(inputs, "SilhouetteScore")?;
        report.silhouette_score = Some(silhouette_score(layout, labels)?);
    }
    if flags.calinski_harabasz_score {
        let labels = require_labels(inputs, "CalinskiHarabaszScore")?;
        report.calinski_harabasz_score = Some(calinski_harabasz_score(layout, labels)?);
    }
    if flags.davies_bouldin_score {
        let labels = require_labels(inputs, "DaviesBouldinScore")?;
        report.davies_bouldin_score = Some(davies_bouldin_score(layout, labels)?);
    }
    if flags.cq_kmeans_ari || flags.cq_kmeans_ami {
        let labels = require_labels(inputs, "CqKmeans")?;
        let pa = cq_kmeans(
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
        let pa = cq_hac(layout, labels)?;
        if flags.cq_hac_hom {
            report.cq_hac_hom = Some(pa.homogeneity);
        }
        if flags.cq_hac_ami {
            report.cq_hac_ami = Some(pa.ami);
        }
    }
    if flags.pairwise_distance_overlap_error {
        let labels = require_labels(inputs, "PairwiseDistanceOverlapError")?;
        report.pairwise_distance_overlap_error = Some(pairwise_distance_overlap_error(
            layout,
            labels,
            config.cluster_baselines.num_bins,
        )?);
    }
    if flags.sprawlter_nn || flags.nor_metrics {
        let mut uniform_radii = Vec::new();
        let (hw, hh) = resolve_node_half_extents(layout.len(), config, inputs, &mut uniform_radii)?;
        if flags.sprawlter_nn {
            report.sprawlter_nn = Some(sprawlter_nn(layout, hw, hh)?);
        }
        if flags.nor_metrics {
            let orig = inputs.original_layout.ok_or_else(|| {
                CoreError::invalid_input("NorMetrics target requires original_layout")
            })?;
            let nor = nor_metrics(graph, orig, layout, hw, hh)?;
            report.nor_oo_nni = Some(nor.oo_nni);
            report.nor_sp_ch_a = Some(nor.sp_ch_a);
            report.nor_gs_bb_iar = Some(nor.gs_bb_iar);
            report.nor_nm_dm_imse = Some(nor.nm_dm_imse);
            report.nor_el_rsd = Some(nor.el_rsd);
        }
    }
    Ok(())
}
