//! Field registry and lookup helpers for `MetricReport`.

use egraph::metrics::MetricReport;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use super::nd_manifold_helpers::{crossing_pairs_to_py_dicts, torus_crossings_to_py_dicts};

pub(super) const CANONICAL_KEYS: [&str; 53] = [
    "scale_normalized_stress",
    "sampled_scale_normalized_stress",
    "geodesic_distortion",
    "crossing_number",
    "crossing_edges",
    "crossing_angle",
    "sampled_crossing_number",
    "crosslessness",
    "sampled_crosslessness",
    "torus_crossing_number",
    "crossing_edges_torus",
    "neighborhood_preservation_1hop",
    "neighborhood_preservation_2hop",
    "sampled_np1",
    "sampled_np2",
    "gabriel_graph_property",
    "node_resolution",
    "relative_neighborhood_preservation",
    "gabriel_graph_preservation",
    "aspect_ratio",
    "angular_resolution",
    "minimum_angle",
    "minimum_angle_a1",
    "layout_degeneracy",
    "ideal_edge_lengths",
    "prism_sigma_dist",
    "prism_sigma_disp",
    "adjusted_patch_nig",
    "cluster_area_faithfulness",
    "landmark_stratified_rta",
    "distance_consistency",
    "distribution_consistency",
    "class_density_measure",
    "histogram_density_measure",
    "sepme_knng",
    "sepme_rng",
    "sepme_gab",
    "sepme_gong",
    "neighborhood_hit",
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "cq_kmeans_ari",
    "cq_kmeans_ami",
    "cq_hac_hom",
    "cq_hac_ami",
    "pairwise_distance_overlap_error",
    "sprawlter_nn",
    "nor_oo_nni",
    "nor_sp_ch_a",
    "nor_gs_bb_iar",
    "nor_nm_dm_imse",
    "nor_el_rsd",
];

fn opt_f64_py(py: Python<'_>, val: Option<f64>) -> PyResult<Option<Py<PyAny>>> {
    val.map(|v| v.into_py_any(py)).transpose()
}

fn opt_usize_py(py: Python<'_>, val: Option<usize>) -> PyResult<Option<Py<PyAny>>> {
    val.map(|v| v.into_py_any(py)).transpose()
}

pub(super) fn lookup_field(
    py: Python<'_>,
    r: &MetricReport,
    key: &str,
) -> PyResult<Option<Option<Py<PyAny>>>> {
    let val = match key {
        "scale_normalized_stress" => opt_f64_py(py, r.scale_normalized_stress)?,
        "sampled_scale_normalized_stress" => opt_f64_py(py, r.sampled_scale_normalized_stress)?,
        "geodesic_distortion" => opt_f64_py(py, r.geodesic_distortion)?,
        "crossing_number" => opt_usize_py(py, r.crossing_number)?,
        "crossing_edges" => match r.crossing_edges.as_ref() {
            Some(pairs) => Some(crossing_pairs_to_py_dicts(py, pairs.clone())?.into_py_any(py)?),
            None => None,
        },
        "crossing_angle" => opt_f64_py(py, r.crossing_angle)?,
        "sampled_crossing_number" => opt_f64_py(py, r.sampled_crossing_number)?,
        "crosslessness" => opt_f64_py(py, r.crosslessness)?,
        "sampled_crosslessness" => opt_f64_py(py, r.sampled_crosslessness)?,
        "torus_crossing_number" => opt_usize_py(py, r.torus_crossing_number)?,
        "crossing_edges_torus" => match r.crossing_edges_torus.as_ref() {
            Some(pairs) => Some(torus_crossings_to_py_dicts(py, pairs.clone())?.into_py_any(py)?),
            None => None,
        },
        "neighborhood_preservation_1hop" => opt_f64_py(py, r.neighborhood_preservation_1hop)?,
        "neighborhood_preservation_2hop" => opt_f64_py(py, r.neighborhood_preservation_2hop)?,
        "sampled_np1" => opt_f64_py(py, r.sampled_np1)?,
        "sampled_np2" => opt_f64_py(py, r.sampled_np2)?,
        "gabriel_graph_property" => opt_f64_py(py, r.gabriel_graph_property)?,
        "node_resolution" => opt_f64_py(py, r.node_resolution)?,
        "relative_neighborhood_preservation" => {
            opt_f64_py(py, r.relative_neighborhood_preservation)?
        }
        "gabriel_graph_preservation" => opt_f64_py(py, r.gabriel_graph_preservation)?,
        "aspect_ratio" => opt_f64_py(py, r.aspect_ratio)?,
        "angular_resolution" => opt_f64_py(py, r.angular_resolution)?,
        "minimum_angle" => opt_f64_py(py, r.minimum_angle)?,
        "minimum_angle_a1" => opt_f64_py(py, r.minimum_angle_a1)?,
        "layout_degeneracy" => opt_f64_py(py, r.layout_degeneracy)?,
        "ideal_edge_lengths" => opt_f64_py(py, r.ideal_edge_lengths)?,
        "prism_sigma_dist" => opt_f64_py(py, r.prism_sigma_dist)?,
        "prism_sigma_disp" => opt_f64_py(py, r.prism_sigma_disp)?,
        "adjusted_patch_nig" => opt_f64_py(py, r.adjusted_patch_nig)?,
        "cluster_area_faithfulness" => opt_f64_py(py, r.cluster_area_faithfulness)?,
        "landmark_stratified_rta" => opt_f64_py(py, r.landmark_stratified_rta)?,
        "distance_consistency" => opt_f64_py(
            py,
            r.baseline_metrics().and_then(|b| b.distance_consistency),
        )?,
        "distribution_consistency" => opt_f64_py(
            py,
            r.baseline_metrics()
                .and_then(|b| b.distribution_consistency),
        )?,
        "class_density_measure" => opt_f64_py(
            py,
            r.baseline_metrics().and_then(|b| b.class_density_measure),
        )?,
        "histogram_density_measure" => opt_f64_py(
            py,
            r.baseline_metrics()
                .and_then(|b| b.histogram_density_measure),
        )?,
        "sepme_knng" => opt_f64_py(py, r.baseline_metrics().and_then(|b| b.sepme_knng))?,
        "sepme_rng" => opt_f64_py(py, r.baseline_metrics().and_then(|b| b.sepme_rng))?,
        "sepme_gab" => opt_f64_py(py, r.baseline_metrics().and_then(|b| b.sepme_gab))?,
        "sepme_gong" => opt_f64_py(py, r.baseline_metrics().and_then(|b| b.sepme_gong))?,
        "neighborhood_hit" => {
            opt_f64_py(py, r.baseline_metrics().and_then(|b| b.neighborhood_hit))?
        }
        "silhouette_score" => {
            opt_f64_py(py, r.baseline_metrics().and_then(|b| b.silhouette_score))?
        }
        "calinski_harabasz_score" => opt_f64_py(
            py,
            r.baseline_metrics().and_then(|b| b.calinski_harabasz_score),
        )?,
        "davies_bouldin_score" => opt_f64_py(
            py,
            r.baseline_metrics().and_then(|b| b.davies_bouldin_score),
        )?,
        "cq_kmeans_ari" => opt_f64_py(py, r.baseline_metrics().and_then(|b| b.cq_kmeans_ari))?,
        "cq_kmeans_ami" => opt_f64_py(py, r.baseline_metrics().and_then(|b| b.cq_kmeans_ami))?,
        "cq_hac_hom" => opt_f64_py(py, r.baseline_metrics().and_then(|b| b.cq_hac_hom))?,
        "cq_hac_ami" => opt_f64_py(py, r.baseline_metrics().and_then(|b| b.cq_hac_ami))?,
        "pairwise_distance_overlap_error" => opt_f64_py(
            py,
            r.baseline_metrics()
                .and_then(|b| b.pairwise_distance_overlap_error),
        )?,
        "sprawlter_nn" => opt_f64_py(py, r.baseline_metrics().and_then(|b| b.sprawlter_nn))?,
        "nor_oo_nni" => opt_f64_py(py, r.baseline_metrics().and_then(|b| b.nor_oo_nni))?,
        "nor_sp_ch_a" => opt_f64_py(py, r.baseline_metrics().and_then(|b| b.nor_sp_ch_a))?,
        "nor_gs_bb_iar" => opt_f64_py(py, r.baseline_metrics().and_then(|b| b.nor_gs_bb_iar))?,
        "nor_nm_dm_imse" => opt_f64_py(py, r.baseline_metrics().and_then(|b| b.nor_nm_dm_imse))?,
        "nor_el_rsd" => opt_f64_py(py, r.baseline_metrics().and_then(|b| b.nor_el_rsd))?,
        _ => return Ok(None),
    };
    Ok(Some(val))
}

pub(super) fn is_field_present(r: &MetricReport, key: &str) -> Option<bool> {
    Some(match key {
        "scale_normalized_stress" => r.scale_normalized_stress.is_some(),
        "sampled_scale_normalized_stress" => r.sampled_scale_normalized_stress.is_some(),
        "geodesic_distortion" => r.geodesic_distortion.is_some(),
        "crossing_number" => r.crossing_number.is_some(),
        "crossing_edges" => r.crossing_edges.is_some(),
        "crossing_angle" => r.crossing_angle.is_some(),
        "sampled_crossing_number" => r.sampled_crossing_number.is_some(),
        "crosslessness" => r.crosslessness.is_some(),
        "sampled_crosslessness" => r.sampled_crosslessness.is_some(),
        "torus_crossing_number" => r.torus_crossing_number.is_some(),
        "crossing_edges_torus" => r.crossing_edges_torus.is_some(),
        "neighborhood_preservation_1hop" => r.neighborhood_preservation_1hop.is_some(),
        "neighborhood_preservation_2hop" => r.neighborhood_preservation_2hop.is_some(),
        "sampled_np1" => r.sampled_np1.is_some(),
        "sampled_np2" => r.sampled_np2.is_some(),
        "gabriel_graph_property" => r.gabriel_graph_property.is_some(),
        "node_resolution" => r.node_resolution.is_some(),
        "relative_neighborhood_preservation" => r.relative_neighborhood_preservation.is_some(),
        "gabriel_graph_preservation" => r.gabriel_graph_preservation.is_some(),
        "aspect_ratio" => r.aspect_ratio.is_some(),
        "angular_resolution" => r.angular_resolution.is_some(),
        "minimum_angle" => r.minimum_angle.is_some(),
        "minimum_angle_a1" => r.minimum_angle_a1.is_some(),
        "layout_degeneracy" => r.layout_degeneracy.is_some(),
        "ideal_edge_lengths" => r.ideal_edge_lengths.is_some(),
        "prism_sigma_dist" => r.prism_sigma_dist.is_some(),
        "prism_sigma_disp" => r.prism_sigma_disp.is_some(),
        "adjusted_patch_nig" => r.adjusted_patch_nig.is_some(),
        "cluster_area_faithfulness" => r.cluster_area_faithfulness.is_some(),
        "landmark_stratified_rta" => r.landmark_stratified_rta.is_some(),
        "distance_consistency" => r
            .baseline_metrics()
            .is_some_and(|b| b.distance_consistency.is_some()),
        "distribution_consistency" => r
            .baseline_metrics()
            .is_some_and(|b| b.distribution_consistency.is_some()),
        "class_density_measure" => r
            .baseline_metrics()
            .is_some_and(|b| b.class_density_measure.is_some()),
        "histogram_density_measure" => r
            .baseline_metrics()
            .is_some_and(|b| b.histogram_density_measure.is_some()),
        "sepme_knng" => r.baseline_metrics().is_some_and(|b| b.sepme_knng.is_some()),
        "sepme_rng" => r.baseline_metrics().is_some_and(|b| b.sepme_rng.is_some()),
        "sepme_gab" => r.baseline_metrics().is_some_and(|b| b.sepme_gab.is_some()),
        "sepme_gong" => r.baseline_metrics().is_some_and(|b| b.sepme_gong.is_some()),
        "neighborhood_hit" => r
            .baseline_metrics()
            .is_some_and(|b| b.neighborhood_hit.is_some()),
        "silhouette_score" => r
            .baseline_metrics()
            .is_some_and(|b| b.silhouette_score.is_some()),
        "calinski_harabasz_score" => r
            .baseline_metrics()
            .is_some_and(|b| b.calinski_harabasz_score.is_some()),
        "davies_bouldin_score" => r
            .baseline_metrics()
            .is_some_and(|b| b.davies_bouldin_score.is_some()),
        "cq_kmeans_ari" => r
            .baseline_metrics()
            .is_some_and(|b| b.cq_kmeans_ari.is_some()),
        "cq_kmeans_ami" => r
            .baseline_metrics()
            .is_some_and(|b| b.cq_kmeans_ami.is_some()),
        "cq_hac_hom" => r.baseline_metrics().is_some_and(|b| b.cq_hac_hom.is_some()),
        "cq_hac_ami" => r.baseline_metrics().is_some_and(|b| b.cq_hac_ami.is_some()),
        "pairwise_distance_overlap_error" => r
            .baseline_metrics()
            .is_some_and(|b| b.pairwise_distance_overlap_error.is_some()),
        "sprawlter_nn" => r
            .baseline_metrics()
            .is_some_and(|b| b.sprawlter_nn.is_some()),
        "nor_oo_nni" => r.baseline_metrics().is_some_and(|b| b.nor_oo_nni.is_some()),
        "nor_sp_ch_a" => r
            .baseline_metrics()
            .is_some_and(|b| b.nor_sp_ch_a.is_some()),
        "nor_gs_bb_iar" => r
            .baseline_metrics()
            .is_some_and(|b| b.nor_gs_bb_iar.is_some()),
        "nor_nm_dm_imse" => r
            .baseline_metrics()
            .is_some_and(|b| b.nor_nm_dm_imse.is_some()),
        "nor_el_rsd" => r.baseline_metrics().is_some_and(|b| b.nor_el_rsd.is_some()),
        _ => return None,
    })
}
