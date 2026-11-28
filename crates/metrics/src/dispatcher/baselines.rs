use std::any::Any;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, EuclideanLayout, Layout2d, MetricSpace, NodeExtents2d, Scalar,
};
use egraph_metrics_cluster::{
    calinski_harabasz_score, class_density_measure, cq_hac, cq_kmeans, davies_bouldin_score,
    distance_consistency, distribution_consistency, histogram_density_measure, neighborhood_hit,
    neighborhood_hit_with_index, nor_metrics, pairwise_distance_overlap_error, sepme_gab,
    sepme_gab_with_index, sepme_gong, sepme_gong_with_index, sepme_knng, sepme_knng_with_index,
    sepme_rng, sepme_rng_with_index, silhouette_score, sprawlter_nn,
};
use egraph_spatial_trees::NearestNeighborIndex;

use super::config::{ClusterBaselineConfig, MetricConfig};
use super::inputs::MetricInputs;
use super::report::{ClusterBaselineReport, MetricReport, RequestedFlags};
use super::spatial::SpaceClass;

fn require_labels<'a, L: MetricSpace + ?Sized>(
    inputs: &'a MetricInputs<'a, L>,
    target: &str,
) -> CoreResult<&'a [u32]> {
    inputs
        .labels
        .ok_or_else(|| CoreError::invalid_input(format!("{target} target requires labels")))
}

fn eval_euclidean_separation_targets<E: EuclideanLayout + 'static, L: MetricSpace + ?Sized>(
    layout: &E,
    index: &dyn NearestNeighborIndex,
    flags: &RequestedFlags,
    cb: &ClusterBaselineConfig,
    inputs: &MetricInputs<'_, L>,
    br: &mut ClusterBaselineReport,
) -> CoreResult<()> {
    if flags.sepme_knng {
        let labels = require_labels(inputs, "SepmeKnng")?;
        br.sepme_knng = Some(sepme_knng_with_index(
            layout,
            labels,
            cb.sepme_knng_k,
            index,
        )?);
    }
    if flags.sepme_rng {
        let labels = require_labels(inputs, "SepmeRng")?;
        br.sepme_rng = Some(sepme_rng_with_index(layout, labels, index)?);
    }
    if flags.sepme_gab {
        let labels = require_labels(inputs, "SepmeGab")?;
        br.sepme_gab = Some(sepme_gab_with_index(layout, labels, index)?);
    }
    if flags.sepme_gong {
        let labels = require_labels(inputs, "SepmeGong")?;
        br.sepme_gong = Some(sepme_gong_with_index(layout, labels, cb.gong_gamma, index)?);
    }
    if flags.neighborhood_hit {
        let labels = require_labels(inputs, "NeighborhoodHit")?;
        br.neighborhood_hit = Some(neighborhood_hit_with_index(
            layout,
            labels,
            cb.neighborhood_hit_k,
            index,
        )?);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_eval_baseline_targets<W, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
    space: &SpaceClass<'_, L::Scalar>,
    spatial_index: Option<&dyn NearestNeighborIndex>,
    flags: &RequestedFlags,
    config: &MetricConfig,
    inputs: &MetricInputs<'_, L>,
    report: &mut MetricReport,
) -> CoreResult<()> {
    if !flags.has_any_baseline() {
        return Ok(());
    }

    let cb = config.baseline_config();
    let mut br = ClusterBaselineReport::new();

    if flags.distance_consistency {
        let labels = require_labels(inputs, "DistanceConsistency")?;
        br.distance_consistency = Some(distance_consistency(layout, labels)?);
    }
    if flags.distribution_consistency {
        let labels = require_labels(inputs, "DistributionConsistency")?;
        br.distribution_consistency =
            Some(distribution_consistency(layout, labels, cb.dc_grid_size)?);
    }

    if let SpaceClass::Euclidean2d(l2) = space {
        if flags.class_density_measure {
            let labels = require_labels(inputs, "ClassDensityMeasure")?;
            let (cdm, hdm) =
                class_density_measure(*l2, labels, cb.cdm_grid_size, cb.cdm_bandwidth)?;
            br.class_density_measure = Some(cdm);
            if flags.histogram_density_measure {
                br.histogram_density_measure = Some(hdm);
            }
        } else if flags.histogram_density_measure {
            let labels = require_labels(inputs, "HistogramDensityMeasure")?;
            br.histogram_density_measure =
                Some(histogram_density_measure(*l2, labels, cb.cdm_grid_size)?);
        }
    }

    if let Some(index) = spatial_index {
        match space {
            SpaceClass::Euclidean2d(l2) => {
                eval_euclidean_separation_targets(*l2, index, flags, cb, inputs, &mut br)?;
            }
            SpaceClass::Euclidean3d(l3) => {
                eval_euclidean_separation_targets(*l3, index, flags, cb, inputs, &mut br)?;
            }
            SpaceClass::EuclideanNd(lnd) => {
                eval_euclidean_separation_targets(*lnd, index, flags, cb, inputs, &mut br)?;
            }
            SpaceClass::Manifold(_) | SpaceClass::ExternalMetric => {}
        }
    } else {
        if flags.sepme_knng {
            let labels = require_labels(inputs, "SepmeKnng")?;
            br.sepme_knng = Some(sepme_knng(layout, labels, cb.sepme_knng_k)?);
        }
        if flags.sepme_rng {
            let labels = require_labels(inputs, "SepmeRng")?;
            br.sepme_rng = Some(sepme_rng(layout, labels)?);
        }
        if flags.sepme_gab {
            let labels = require_labels(inputs, "SepmeGab")?;
            br.sepme_gab = Some(sepme_gab(layout, labels)?);
        }
        if flags.sepme_gong {
            let labels = require_labels(inputs, "SepmeGong")?;
            br.sepme_gong = Some(sepme_gong(layout, labels, cb.gong_gamma)?);
        }
        if flags.neighborhood_hit {
            let labels = require_labels(inputs, "NeighborhoodHit")?;
            br.neighborhood_hit = Some(neighborhood_hit(layout, labels, cb.neighborhood_hit_k)?);
        }
    }

    if flags.silhouette_score {
        let labels = require_labels(inputs, "SilhouetteScore")?;
        br.silhouette_score = Some(silhouette_score(layout, labels)?);
    }
    if flags.calinski_harabasz_score {
        let labels = require_labels(inputs, "CalinskiHarabaszScore")?;
        br.calinski_harabasz_score = Some(calinski_harabasz_score(layout, labels)?);
    }
    if flags.davies_bouldin_score {
        let labels = require_labels(inputs, "DaviesBouldinScore")?;
        br.davies_bouldin_score = Some(davies_bouldin_score(layout, labels)?);
    }
    if flags.cq_kmeans_ari || flags.cq_kmeans_ami {
        let labels = require_labels(inputs, "CqKmeans")?;
        let pa = cq_kmeans(
            layout,
            labels,
            cb.kmeans_n_init,
            cb.kmeans_max_iters,
            cb.seed,
        )?;
        if flags.cq_kmeans_ari {
            br.cq_kmeans_ari = Some(pa.ari);
        }
        if flags.cq_kmeans_ami {
            br.cq_kmeans_ami = Some(pa.ami);
        }
    }
    if flags.cq_hac_hom || flags.cq_hac_ami {
        let labels = require_labels(inputs, "CqHac")?;
        let pa = cq_hac(layout, labels)?;
        if flags.cq_hac_hom {
            br.cq_hac_hom = Some(pa.homogeneity);
        }
        if flags.cq_hac_ami {
            br.cq_hac_ami = Some(pa.ami);
        }
    }
    if flags.pairwise_distance_overlap_error {
        let labels = require_labels(inputs, "PairwiseDistanceOverlapError")?;
        br.pairwise_distance_overlap_error = Some(pairwise_distance_overlap_error(
            layout,
            labels,
            cb.num_bins,
        )?);
    }

    if (flags.sprawlter_nn || flags.nor_metrics)
        && let SpaceClass::Euclidean2d(l2) = space
    {
        let extents = match inputs.sprawlter_extents {
            Some(ext) => ext,
            None => {
                let r_f64 = cb.sprawlter_node_radius;
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
                let r = L::Scalar::from_f64(r_f64);
                NodeExtents2d::uniform(l2.len(), r, r)?
            }
        };
        if flags.sprawlter_nn {
            br.sprawlter_nn = Some(sprawlter_nn(*l2, extents)?);
        }
        if flags.nor_metrics {
            let orig = inputs
                .original_layout
                .and_then(|o| (o as &dyn Any).downcast_ref::<Layout2d<L::Scalar>>())
                .ok_or_else(|| {
                    CoreError::invalid_input("NorMetrics target requires original_layout")
                })?;
            let nor = nor_metrics(graph, orig, *l2, extents)?;
            br.nor_oo_nni = Some(nor.oo_nni);
            br.nor_sp_ch_a = Some(nor.sp_ch_a);
            br.nor_gs_bb_iar = Some(nor.gs_bb_iar);
            br.nor_nm_dm_imse = Some(nor.nm_dm_imse);
            br.nor_el_rsd = Some(nor.el_rsd);
        }
    }

    report.set_baselines(br);
    Ok(())
}
