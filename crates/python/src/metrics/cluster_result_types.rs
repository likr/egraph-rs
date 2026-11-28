//! Frozen structured clustering and NOR metric result classes (`PartitionAgreement`, `NorMetrics`).

use egraph::metrics::{NorMetrics, PartitionAgreement};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Structured clustering partition agreement summary returned by `partition_agreement`, `cq_kmeans`, `cq_hac`.
#[pyclass(
    name = "PartitionAgreement",
    module = "egraph._core",
    frozen,
    eq,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyPartitionAgreement {
    pub(crate) inner: PartitionAgreement,
}

impl PartialEq for PyPartitionAgreement {
    fn eq(&self, other: &Self) -> bool {
        self.inner.ari == other.inner.ari
            && self.inner.ami == other.inner.ami
            && self.inner.homogeneity == other.inner.homogeneity
            && self.inner.completeness == other.inner.completeness
            && self.inner.v_measure == other.inner.v_measure
            && self.inner.fmi == other.inner.fmi
    }
}

#[pymethods]
impl PyPartitionAgreement {
    #[getter]
    fn ari(&self) -> f64 {
        self.inner.ari
    }
    #[getter]
    fn ami(&self) -> f64 {
        self.inner.ami
    }
    #[getter]
    fn homogeneity(&self) -> f64 {
        self.inner.homogeneity
    }
    #[getter]
    fn completeness(&self) -> f64 {
        self.inner.completeness
    }
    #[getter]
    fn v_measure(&self) -> f64 {
        self.inner.v_measure
    }
    #[getter]
    fn fmi(&self) -> f64 {
        self.inner.fmi
    }
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("ari", self.inner.ari)?;
        d.set_item("ami", self.inner.ami)?;
        d.set_item("homogeneity", self.inner.homogeneity)?;
        d.set_item("completeness", self.inner.completeness)?;
        d.set_item("v_measure", self.inner.v_measure)?;
        d.set_item("fmi", self.inner.fmi)?;
        Ok(d.unbind())
    }
    fn keys(&self) -> Vec<&'static str> {
        vec![
            "ari",
            "ami",
            "homogeneity",
            "completeness",
            "v_measure",
            "fmi",
        ]
    }
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "ari" => self.inner.ari.into_py_any(py),
            "ami" => self.inner.ami.into_py_any(py),
            "homogeneity" => self.inner.homogeneity.into_py_any(py),
            "completeness" => self.inner.completeness.into_py_any(py),
            "v_measure" => self.inner.v_measure.into_py_any(py),
            "fmi" => self.inner.fmi.into_py_any(py),
            _ => Err(PyKeyError::new_err(key.to_owned())),
        }
    }
    fn __contains__(&self, key: &str) -> bool {
        matches!(
            key,
            "ari" | "ami" | "homogeneity" | "completeness" | "v_measure" | "fmi"
        )
    }
    fn __repr__(&self) -> String {
        format!(
            "PartitionAgreement(ari={:.6}, ami={:.6})",
            self.inner.ari, self.inner.ami
        )
    }
}

/// Structured Node Overlap Removal (NOR) 5-metric summary returned by `nor_metrics`.
#[pyclass(
    name = "NorMetrics",
    module = "egraph._core",
    frozen,
    eq,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyNorMetrics {
    pub(crate) inner: NorMetrics,
}

impl PartialEq for PyNorMetrics {
    fn eq(&self, other: &Self) -> bool {
        self.inner.oo_nni == other.inner.oo_nni
            && self.inner.sp_ch_a == other.inner.sp_ch_a
            && self.inner.gs_bb_iar == other.inner.gs_bb_iar
            && self.inner.nm_dm_imse == other.inner.nm_dm_imse
            && self.inner.el_rsd == other.inner.el_rsd
    }
}

#[pymethods]
impl PyNorMetrics {
    #[getter]
    fn oo_nni(&self) -> f64 {
        self.inner.oo_nni
    }
    #[getter]
    fn sp_ch_a(&self) -> f64 {
        self.inner.sp_ch_a
    }
    #[getter]
    fn gs_bb_iar(&self) -> f64 {
        self.inner.gs_bb_iar
    }
    #[getter]
    fn nm_dm_imse(&self) -> f64 {
        self.inner.nm_dm_imse
    }
    #[getter]
    fn el_rsd(&self) -> f64 {
        self.inner.el_rsd
    }
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("oo_nni", self.inner.oo_nni)?;
        d.set_item("sp_ch_a", self.inner.sp_ch_a)?;
        d.set_item("gs_bb_iar", self.inner.gs_bb_iar)?;
        d.set_item("nm_dm_imse", self.inner.nm_dm_imse)?;
        d.set_item("el_rsd", self.inner.el_rsd)?;
        Ok(d.unbind())
    }
    fn keys(&self) -> Vec<&'static str> {
        vec!["oo_nni", "sp_ch_a", "gs_bb_iar", "nm_dm_imse", "el_rsd"]
    }
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "oo_nni" => self.inner.oo_nni.into_py_any(py),
            "sp_ch_a" => self.inner.sp_ch_a.into_py_any(py),
            "gs_bb_iar" => self.inner.gs_bb_iar.into_py_any(py),
            "nm_dm_imse" => self.inner.nm_dm_imse.into_py_any(py),
            "el_rsd" => self.inner.el_rsd.into_py_any(py),
            _ => Err(PyKeyError::new_err(key.to_owned())),
        }
    }
    fn __contains__(&self, key: &str) -> bool {
        matches!(
            key,
            "oo_nni" | "sp_ch_a" | "gs_bb_iar" | "nm_dm_imse" | "el_rsd"
        )
    }
    fn __repr__(&self) -> String {
        format!(
            "NorMetrics(oo_nni={:.6}, el_rsd={:.6})",
            self.inner.oo_nni, self.inner.el_rsd
        )
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPartitionAgreement>()?;
    m.add_class::<PyNorMetrics>()?;
    Ok(())
}
