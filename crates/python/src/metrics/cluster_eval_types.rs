//! Frozen structured result classes for `adjusted_patch_nig_detail` and
//! `cluster_area_faithfulness_detail`.

use egraph::metrics::{AdjustedPatchNigDetail, ClusterAreaDetail, ClusterAreaReport};
use numpy::PyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const NIG_DETAIL_KEYS: [&str; 7] = [
    "nig",
    "h_obs",
    "h_null",
    "singleton_patch_ratio",
    "num_nonempty_patches",
    "recommended_patch_radius",
    "is_degenerate",
];

const CAF_SUMMARY_KEYS: [&str; 12] = [
    "caf",
    "purity_adj",
    "prop_vis",
    "res_density",
    "res_deconv",
    "res_combined",
    "spatial_coherence",
    "in_frac",
    "effective_boundary_r95",
    "g_eff",
    "num_clusters",
    "num_nodes",
];

const CAF_PER_CLUSTER_KEYS: [&str; 10] = [
    "cluster_ids",
    "counts",
    "p",
    "q",
    "purity",
    "resolution",
    "res_density_per_cluster",
    "res_deconv_per_cluster",
    "caf_per_cluster",
    "coherence",
];

/// Frozen diagnostic report returned by `adjusted_patch_nig_detail`.
#[pyclass(
    name = "AdjustedPatchNigDetail",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyAdjustedPatchNigDetail {
    inner: AdjustedPatchNigDetail,
}

impl PyAdjustedPatchNigDetail {
    pub(super) fn from_detail(inner: AdjustedPatchNigDetail) -> Self {
        Self { inner }
    }

    fn lookup(&self, py: Python<'_>, key: &str) -> PyResult<Option<Py<PyAny>>> {
        match key {
            "nig" => Ok(Some(self.inner.nig.into_py_any(py)?)),
            "h_obs" => Ok(Some(self.inner.h_obs.into_py_any(py)?)),
            "h_null" => Ok(Some(self.inner.h_null.into_py_any(py)?)),
            "singleton_patch_ratio" => Ok(Some(self.inner.singleton_patch_ratio.into_py_any(py)?)),
            "num_nonempty_patches" => Ok(Some(self.inner.num_nonempty_patches.into_py_any(py)?)),
            "recommended_patch_radius" => {
                Ok(Some(self.inner.recommended_patch_radius.into_py_any(py)?))
            }
            "is_degenerate" => Ok(Some(self.inner.is_degenerate.into_py_any(py)?)),
            _ => Ok(None),
        }
    }
}

#[pymethods]
impl PyAdjustedPatchNigDetail {
    #[getter]
    fn nig(&self) -> f64 {
        self.inner.nig
    }
    #[getter]
    fn h_obs(&self) -> f64 {
        self.inner.h_obs
    }
    #[getter]
    fn h_null(&self) -> f64 {
        self.inner.h_null
    }
    #[getter]
    fn singleton_patch_ratio(&self) -> f64 {
        self.inner.singleton_patch_ratio
    }
    #[getter]
    fn num_nonempty_patches(&self) -> usize {
        self.inner.num_nonempty_patches
    }
    #[getter]
    fn recommended_patch_radius(&self) -> usize {
        self.inner.recommended_patch_radius
    }
    #[getter]
    fn is_degenerate(&self) -> bool {
        self.inner.is_degenerate
    }

    /// Return all diagnostics as a Python dictionary.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for k in NIG_DETAIL_KEYS {
            if let Some(v) = self.lookup(py, k)? {
                dict.set_item(k, v)?;
            }
        }
        Ok(dict.unbind())
    }

    fn keys(&self) -> Vec<&'static str> {
        NIG_DETAIL_KEYS.to_vec()
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        self.lookup(py, key)?
            .ok_or_else(|| PyKeyError::new_err(key.to_owned()))
    }

    fn __contains__(&self, key: &str) -> bool {
        NIG_DETAIL_KEYS.contains(&key)
    }

    fn __repr__(&self) -> String {
        format!(
            "AdjustedPatchNigDetail(nig={:.6}, h_obs={:.6}, h_null={:.6}, is_degenerate={})",
            self.inner.nig, self.inner.h_obs, self.inner.h_null, self.inner.is_degenerate,
        )
    }
}

/// Frozen `G-CAF` summary and optional per-cluster diagnostics returned by
/// `cluster_area_faithfulness_detail`.
#[pyclass(
    name = "ClusterAreaReport",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyClusterAreaReport {
    report: ClusterAreaReport,
    detail: Option<ClusterAreaDetail>,
}

impl PyClusterAreaReport {
    pub(super) fn from_report(report: ClusterAreaReport) -> Self {
        Self {
            report,
            detail: None,
        }
    }

    pub(super) fn from_detail(detail: ClusterAreaDetail) -> Self {
        Self {
            report: detail.report.clone(),
            detail: Some(detail),
        }
    }

    fn lookup(&self, py: Python<'_>, key: &str) -> PyResult<Option<Py<PyAny>>> {
        match key {
            "caf" => Ok(Some(self.report.caf.into_py_any(py)?)),
            "purity_adj" => Ok(Some(self.report.purity_adj.into_py_any(py)?)),
            "prop_vis" => Ok(Some(self.report.prop_vis.into_py_any(py)?)),
            "res_density" => Ok(Some(self.report.res_density.into_py_any(py)?)),
            "res_deconv" => Ok(Some(self.report.res_deconv.into_py_any(py)?)),
            "res_combined" => Ok(Some(self.report.res_combined.into_py_any(py)?)),
            "spatial_coherence" => Ok(Some(self.report.spatial_coherence.into_py_any(py)?)),
            "in_frac" => Ok(Some(self.report.in_frac.into_py_any(py)?)),
            "effective_boundary_r95" => {
                Ok(Some(self.report.effective_boundary_r95.into_py_any(py)?))
            }
            "g_eff" => Ok(Some(self.report.g_eff.into_py_any(py)?)),
            "num_clusters" => Ok(Some(self.report.num_clusters.into_py_any(py)?)),
            "num_nodes" => Ok(Some(self.report.num_nodes.into_py_any(py)?)),
            "cluster_ids" => Ok(self
                .detail
                .as_ref()
                .map(|d| PyArray1::from_slice(py, &d.cluster_ids).into_any().unbind())),
            "counts" => Ok(self
                .detail
                .as_ref()
                .map(|d| PyArray1::from_slice(py, &d.counts).into_any().unbind())),
            "p" => Ok(self
                .detail
                .as_ref()
                .map(|d| PyArray1::from_slice(py, &d.p).into_any().unbind())),
            "q" => Ok(self
                .detail
                .as_ref()
                .map(|d| PyArray1::from_slice(py, &d.q).into_any().unbind())),
            "purity" => Ok(self
                .detail
                .as_ref()
                .map(|d| PyArray1::from_slice(py, &d.purity).into_any().unbind())),
            "resolution" => Ok(self
                .detail
                .as_ref()
                .map(|d| PyArray1::from_slice(py, &d.resolution).into_any().unbind())),
            "res_density_per_cluster" => Ok(self
                .detail
                .as_ref()
                .map(|d| PyArray1::from_slice(py, &d.res_density).into_any().unbind())),
            "res_deconv_per_cluster" => Ok(self
                .detail
                .as_ref()
                .map(|d| PyArray1::from_slice(py, &d.res_deconv).into_any().unbind())),
            "caf_per_cluster" => Ok(self.detail.as_ref().map(|d| {
                PyArray1::from_slice(py, &d.caf_per_cluster)
                    .into_any()
                    .unbind()
            })),
            "coherence" => Ok(self
                .detail
                .as_ref()
                .map(|d| PyArray1::from_slice(py, &d.coherence).into_any().unbind())),
            _ => Ok(None),
        }
    }
}

#[pymethods]
impl PyClusterAreaReport {
    #[getter]
    fn caf(&self) -> f64 {
        self.report.caf
    }
    #[getter]
    fn purity_adj(&self) -> f64 {
        self.report.purity_adj
    }
    #[getter]
    fn prop_vis(&self) -> f64 {
        self.report.prop_vis
    }
    #[getter]
    fn res_density(&self) -> f64 {
        self.report.res_density
    }
    #[getter]
    fn res_deconv(&self) -> f64 {
        self.report.res_deconv
    }
    #[getter]
    fn res_combined(&self) -> f64 {
        self.report.res_combined
    }
    #[getter]
    fn spatial_coherence(&self) -> f64 {
        self.report.spatial_coherence
    }
    #[getter]
    fn in_frac(&self) -> f64 {
        self.report.in_frac
    }
    #[getter]
    fn effective_boundary_r95(&self) -> f64 {
        self.report.effective_boundary_r95
    }
    #[getter]
    fn g_eff(&self) -> usize {
        self.report.g_eff
    }
    #[getter]
    fn num_clusters(&self) -> usize {
        self.report.num_clusters
    }
    #[getter]
    fn num_nodes(&self) -> usize {
        self.report.num_nodes
    }

    #[getter]
    fn cluster_ids<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<u32>>> {
        self.detail
            .as_ref()
            .map(|d| PyArray1::from_slice(py, &d.cluster_ids))
    }
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<usize>>> {
        self.detail
            .as_ref()
            .map(|d| PyArray1::from_slice(py, &d.counts))
    }
    #[getter]
    fn p<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.detail.as_ref().map(|d| PyArray1::from_slice(py, &d.p))
    }
    #[getter]
    fn q<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.detail.as_ref().map(|d| PyArray1::from_slice(py, &d.q))
    }
    #[getter]
    fn purity<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.detail
            .as_ref()
            .map(|d| PyArray1::from_slice(py, &d.purity))
    }
    #[getter]
    fn resolution<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.detail
            .as_ref()
            .map(|d| PyArray1::from_slice(py, &d.resolution))
    }
    #[getter]
    fn res_density_per_cluster<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.detail
            .as_ref()
            .map(|d| PyArray1::from_slice(py, &d.res_density))
    }
    #[getter]
    fn res_deconv_per_cluster<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.detail
            .as_ref()
            .map(|d| PyArray1::from_slice(py, &d.res_deconv))
    }
    #[getter]
    fn caf_per_cluster<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.detail
            .as_ref()
            .map(|d| PyArray1::from_slice(py, &d.caf_per_cluster))
    }
    #[getter]
    fn coherence<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.detail
            .as_ref()
            .map(|d| PyArray1::from_slice(py, &d.coherence))
    }

    /// Return present fields as a Python dictionary.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for k in self.keys() {
            if let Some(v) = self.lookup(py, k)? {
                dict.set_item(k, v)?;
            }
        }
        Ok(dict.unbind())
    }

    fn keys(&self) -> Vec<&'static str> {
        let mut keys = CAF_SUMMARY_KEYS.to_vec();
        if self.detail.is_some() {
            keys.extend_from_slice(&CAF_PER_CLUSTER_KEYS);
        }
        keys
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        self.lookup(py, key)?
            .ok_or_else(|| PyKeyError::new_err(key.to_owned()))
    }

    fn __contains__(&self, key: &str) -> bool {
        CAF_SUMMARY_KEYS.contains(&key)
            || (self.detail.is_some() && CAF_PER_CLUSTER_KEYS.contains(&key))
    }

    fn __repr__(&self) -> String {
        format!(
            "ClusterAreaReport(caf={:.6}, purity_adj={:.6}, prop_vis={:.6}, num_clusters={}, num_nodes={})",
            self.report.caf,
            self.report.purity_adj,
            self.report.prop_vis,
            self.report.num_clusters,
            self.report.num_nodes,
        )
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAdjustedPatchNigDetail>()?;
    m.add_class::<PyClusterAreaReport>()?;
    Ok(())
}
