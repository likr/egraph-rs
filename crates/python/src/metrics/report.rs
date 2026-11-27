//! Typed selective metric evaluation result (`MetricReport`).

use egraph::metrics::MetricReport;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::nd_manifold_helpers::{crossing_pairs_to_py_dicts, torus_crossings_to_py_dicts};

const CANONICAL_KEYS: [&str; 53] = [
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

fn lookup_field(
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
        "distance_consistency" => opt_f64_py(py, r.distance_consistency)?,
        "distribution_consistency" => opt_f64_py(py, r.distribution_consistency)?,
        "class_density_measure" => opt_f64_py(py, r.class_density_measure)?,
        "histogram_density_measure" => opt_f64_py(py, r.histogram_density_measure)?,
        "sepme_knng" => opt_f64_py(py, r.sepme_knng)?,
        "sepme_rng" => opt_f64_py(py, r.sepme_rng)?,
        "sepme_gab" => opt_f64_py(py, r.sepme_gab)?,
        "sepme_gong" => opt_f64_py(py, r.sepme_gong)?,
        "neighborhood_hit" => opt_f64_py(py, r.neighborhood_hit)?,
        "silhouette_score" => opt_f64_py(py, r.silhouette_score)?,
        "calinski_harabasz_score" => opt_f64_py(py, r.calinski_harabasz_score)?,
        "davies_bouldin_score" => opt_f64_py(py, r.davies_bouldin_score)?,
        "cq_kmeans_ari" => opt_f64_py(py, r.cq_kmeans_ari)?,
        "cq_kmeans_ami" => opt_f64_py(py, r.cq_kmeans_ami)?,
        "cq_hac_hom" => opt_f64_py(py, r.cq_hac_hom)?,
        "cq_hac_ami" => opt_f64_py(py, r.cq_hac_ami)?,
        "pairwise_distance_overlap_error" => opt_f64_py(py, r.pairwise_distance_overlap_error)?,
        "sprawlter_nn" => opt_f64_py(py, r.sprawlter_nn)?,
        "nor_oo_nni" => opt_f64_py(py, r.nor_oo_nni)?,
        "nor_sp_ch_a" => opt_f64_py(py, r.nor_sp_ch_a)?,
        "nor_gs_bb_iar" => opt_f64_py(py, r.nor_gs_bb_iar)?,
        "nor_nm_dm_imse" => opt_f64_py(py, r.nor_nm_dm_imse)?,
        "nor_el_rsd" => opt_f64_py(py, r.nor_el_rsd)?,
        _ => return Ok(None),
    };
    Ok(Some(val))
}

fn is_field_present(r: &MetricReport, key: &str) -> Option<bool> {
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
        "distance_consistency" => r.distance_consistency.is_some(),
        "distribution_consistency" => r.distribution_consistency.is_some(),
        "class_density_measure" => r.class_density_measure.is_some(),
        "histogram_density_measure" => r.histogram_density_measure.is_some(),
        "sepme_knng" => r.sepme_knng.is_some(),
        "sepme_rng" => r.sepme_rng.is_some(),
        "sepme_gab" => r.sepme_gab.is_some(),
        "sepme_gong" => r.sepme_gong.is_some(),
        "neighborhood_hit" => r.neighborhood_hit.is_some(),
        "silhouette_score" => r.silhouette_score.is_some(),
        "calinski_harabasz_score" => r.calinski_harabasz_score.is_some(),
        "davies_bouldin_score" => r.davies_bouldin_score.is_some(),
        "cq_kmeans_ari" => r.cq_kmeans_ari.is_some(),
        "cq_kmeans_ami" => r.cq_kmeans_ami.is_some(),
        "cq_hac_hom" => r.cq_hac_hom.is_some(),
        "cq_hac_ami" => r.cq_hac_ami.is_some(),
        "pairwise_distance_overlap_error" => r.pairwise_distance_overlap_error.is_some(),
        "sprawlter_nn" => r.sprawlter_nn.is_some(),
        "nor_oo_nni" => r.nor_oo_nni.is_some(),
        "nor_sp_ch_a" => r.nor_sp_ch_a.is_some(),
        "nor_gs_bb_iar" => r.nor_gs_bb_iar.is_some(),
        "nor_nm_dm_imse" => r.nor_nm_dm_imse.is_some(),
        "nor_el_rsd" => r.nor_el_rsd.is_some(),
        _ => return None,
    })
}

/// Typed selective metric evaluation report returned by `evaluate_metrics`.
///
/// Unrequested fields evaluate to `None`.
#[pyclass(name = "MetricReport", module = "egraph._core", frozen, from_py_object)]
#[derive(Debug, Clone)]
pub struct PyMetricReport {
    pub(crate) inner: MetricReport,
}

#[pymethods]
impl PyMetricReport {
    #[getter]
    fn scale_normalized_stress(&self) -> Option<f64> {
        self.inner.scale_normalized_stress
    }
    #[getter]
    fn sampled_scale_normalized_stress(&self) -> Option<f64> {
        self.inner.sampled_scale_normalized_stress
    }
    #[getter]
    fn geodesic_distortion(&self) -> Option<f64> {
        self.inner.geodesic_distortion
    }
    #[getter]
    fn crossing_number(&self) -> Option<usize> {
        self.inner.crossing_number
    }
    #[getter]
    fn crossing_edges(&self, py: Python<'_>) -> PyResult<Option<Vec<Py<PyDict>>>> {
        self.inner
            .crossing_edges
            .as_ref()
            .map(|p| crossing_pairs_to_py_dicts(py, p.clone()))
            .transpose()
    }
    #[getter]
    fn crossing_angle(&self) -> Option<f64> {
        self.inner.crossing_angle
    }
    #[getter]
    fn sampled_crossing_number(&self) -> Option<f64> {
        self.inner.sampled_crossing_number
    }
    #[getter]
    fn crosslessness(&self) -> Option<f64> {
        self.inner.crosslessness
    }
    #[getter]
    fn sampled_crosslessness(&self) -> Option<f64> {
        self.inner.sampled_crosslessness
    }
    #[getter]
    fn torus_crossing_number(&self) -> Option<usize> {
        self.inner.torus_crossing_number
    }
    #[getter]
    fn crossing_edges_torus(&self, py: Python<'_>) -> PyResult<Option<Vec<Py<PyDict>>>> {
        self.inner
            .crossing_edges_torus
            .as_ref()
            .map(|p| torus_crossings_to_py_dicts(py, p.clone()))
            .transpose()
    }
    #[getter]
    fn neighborhood_preservation_1hop(&self) -> Option<f64> {
        self.inner.neighborhood_preservation_1hop
    }
    #[getter]
    fn neighborhood_preservation_2hop(&self) -> Option<f64> {
        self.inner.neighborhood_preservation_2hop
    }
    #[getter]
    fn sampled_np1(&self) -> Option<f64> {
        self.inner.sampled_np1
    }
    #[getter]
    fn sampled_np2(&self) -> Option<f64> {
        self.inner.sampled_np2
    }
    #[getter]
    fn gabriel_graph_property(&self) -> Option<f64> {
        self.inner.gabriel_graph_property
    }
    #[getter]
    fn node_resolution(&self) -> Option<f64> {
        self.inner.node_resolution
    }
    #[getter]
    fn relative_neighborhood_preservation(&self) -> Option<f64> {
        self.inner.relative_neighborhood_preservation
    }
    #[getter]
    fn gabriel_graph_preservation(&self) -> Option<f64> {
        self.inner.gabriel_graph_preservation
    }
    #[getter]
    fn aspect_ratio(&self) -> Option<f64> {
        self.inner.aspect_ratio
    }
    #[getter]
    fn angular_resolution(&self) -> Option<f64> {
        self.inner.angular_resolution
    }
    #[getter]
    fn minimum_angle(&self) -> Option<f64> {
        self.inner.minimum_angle
    }
    #[getter]
    fn minimum_angle_a1(&self) -> Option<f64> {
        self.inner.minimum_angle_a1
    }
    #[getter]
    fn layout_degeneracy(&self) -> Option<f64> {
        self.inner.layout_degeneracy
    }
    #[getter]
    fn ideal_edge_lengths(&self) -> Option<f64> {
        self.inner.ideal_edge_lengths
    }
    #[getter]
    fn prism_sigma_dist(&self) -> Option<f64> {
        self.inner.prism_sigma_dist
    }
    #[getter]
    fn prism_sigma_disp(&self) -> Option<f64> {
        self.inner.prism_sigma_disp
    }
    #[getter]
    fn adjusted_patch_nig(&self) -> Option<f64> {
        self.inner.adjusted_patch_nig
    }
    #[getter]
    fn cluster_area_faithfulness(&self) -> Option<f64> {
        self.inner.cluster_area_faithfulness
    }
    #[getter]
    fn landmark_stratified_rta(&self) -> Option<f64> {
        self.inner.landmark_stratified_rta
    }
    #[getter]
    fn distance_consistency(&self) -> Option<f64> {
        self.inner.distance_consistency
    }
    #[getter]
    fn distribution_consistency(&self) -> Option<f64> {
        self.inner.distribution_consistency
    }
    #[getter]
    fn class_density_measure(&self) -> Option<f64> {
        self.inner.class_density_measure
    }
    #[getter]
    fn histogram_density_measure(&self) -> Option<f64> {
        self.inner.histogram_density_measure
    }
    #[getter]
    fn sepme_knng(&self) -> Option<f64> {
        self.inner.sepme_knng
    }
    #[getter]
    fn sepme_rng(&self) -> Option<f64> {
        self.inner.sepme_rng
    }
    #[getter]
    fn sepme_gab(&self) -> Option<f64> {
        self.inner.sepme_gab
    }
    #[getter]
    fn sepme_gong(&self) -> Option<f64> {
        self.inner.sepme_gong
    }
    #[getter]
    fn neighborhood_hit(&self) -> Option<f64> {
        self.inner.neighborhood_hit
    }
    #[getter]
    fn silhouette_score(&self) -> Option<f64> {
        self.inner.silhouette_score
    }
    #[getter]
    fn calinski_harabasz_score(&self) -> Option<f64> {
        self.inner.calinski_harabasz_score
    }
    #[getter]
    fn davies_bouldin_score(&self) -> Option<f64> {
        self.inner.davies_bouldin_score
    }
    #[getter]
    fn cq_kmeans_ari(&self) -> Option<f64> {
        self.inner.cq_kmeans_ari
    }
    #[getter]
    fn cq_kmeans_ami(&self) -> Option<f64> {
        self.inner.cq_kmeans_ami
    }
    #[getter]
    fn cq_hac_hom(&self) -> Option<f64> {
        self.inner.cq_hac_hom
    }
    #[getter]
    fn cq_hac_ami(&self) -> Option<f64> {
        self.inner.cq_hac_ami
    }
    #[getter]
    fn pairwise_distance_overlap_error(&self) -> Option<f64> {
        self.inner.pairwise_distance_overlap_error
    }
    #[getter]
    fn sprawlter_nn(&self) -> Option<f64> {
        self.inner.sprawlter_nn
    }
    #[getter]
    fn nor_oo_nni(&self) -> Option<f64> {
        self.inner.nor_oo_nni
    }
    #[getter]
    fn nor_sp_ch_a(&self) -> Option<f64> {
        self.inner.nor_sp_ch_a
    }
    #[getter]
    fn nor_gs_bb_iar(&self) -> Option<f64> {
        self.inner.nor_gs_bb_iar
    }
    #[getter]
    fn nor_nm_dm_imse(&self) -> Option<f64> {
        self.inner.nor_nm_dm_imse
    }
    #[getter]
    fn nor_el_rsd(&self) -> Option<f64> {
        self.inner.nor_el_rsd
    }

    /// Return evaluated metric values as a dictionary keyed by canonical snake_case field names.
    #[pyo3(signature = (include_none = false), text_signature = "(self, include_none=False)")]
    fn to_dict(&self, py: Python<'_>, include_none: bool) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for &key in &CANONICAL_KEYS {
            if let Some(opt_val) = lookup_field(py, &self.inner, key)? {
                match opt_val {
                    Some(v) => dict.set_item(key, v)?,
                    None if include_none => dict.set_item(key, py.None())?,
                    None => {}
                }
            }
        }
        Ok(dict.unbind())
    }

    /// Return the list of evaluated canonical field names in declaration order.
    fn keys(&self) -> Vec<&'static str> {
        CANONICAL_KEYS
            .iter()
            .copied()
            .filter(|k| is_field_present(&self.inner, k) == Some(true))
            .collect()
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match lookup_field(py, &self.inner, key)? {
            Some(Some(val)) => Ok(val),
            Some(None) => Ok(py.None()),
            None => Err(PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __contains__(&self, key: &str) -> bool {
        is_field_present(&self.inner, key) == Some(true)
    }

    fn __len__(&self) -> usize {
        CANONICAL_KEYS
            .iter()
            .filter(|k| is_field_present(&self.inner, k) == Some(true))
            .count()
    }

    fn __repr__(&self) -> String {
        let count = self.__len__();
        format!("MetricReport(evaluated={count})")
    }
}
