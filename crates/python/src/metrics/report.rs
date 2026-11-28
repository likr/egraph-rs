//! Typed selective metric evaluation result (`MetricReport`).

use egraph::metrics::MetricReport;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::nd_manifold_helpers::{crossing_pairs_to_py_dicts, torus_crossings_to_py_dicts};
use super::report_fields::{CANONICAL_KEYS, is_field_present, lookup_field};

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
        self.inner
            .baseline_metrics()
            .and_then(|b| b.distance_consistency)
    }
    #[getter]
    fn distribution_consistency(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.distribution_consistency)
    }
    #[getter]
    fn class_density_measure(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.class_density_measure)
    }
    #[getter]
    fn histogram_density_measure(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.histogram_density_measure)
    }
    #[getter]
    fn sepme_knng(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.sepme_knng)
    }
    #[getter]
    fn sepme_rng(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.sepme_rng)
    }
    #[getter]
    fn sepme_gab(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.sepme_gab)
    }
    #[getter]
    fn sepme_gong(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.sepme_gong)
    }
    #[getter]
    fn neighborhood_hit(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.neighborhood_hit)
    }
    #[getter]
    fn silhouette_score(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.silhouette_score)
    }
    #[getter]
    fn calinski_harabasz_score(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.calinski_harabasz_score)
    }
    #[getter]
    fn davies_bouldin_score(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.davies_bouldin_score)
    }
    #[getter]
    fn cq_kmeans_ari(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.cq_kmeans_ari)
    }
    #[getter]
    fn cq_kmeans_ami(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.cq_kmeans_ami)
    }
    #[getter]
    fn cq_hac_hom(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.cq_hac_hom)
    }
    #[getter]
    fn cq_hac_ami(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.cq_hac_ami)
    }
    #[getter]
    fn pairwise_distance_overlap_error(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.pairwise_distance_overlap_error)
    }
    #[getter]
    fn sprawlter_nn(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.sprawlter_nn)
    }
    #[getter]
    fn nor_oo_nni(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.nor_oo_nni)
    }
    #[getter]
    fn nor_sp_ch_a(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.nor_sp_ch_a)
    }
    #[getter]
    fn nor_gs_bb_iar(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.nor_gs_bb_iar)
    }
    #[getter]
    fn nor_nm_dm_imse(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.nor_nm_dm_imse)
    }
    #[getter]
    fn nor_el_rsd(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.nor_el_rsd)
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
