//! Frozen diagnostics result classes for `SpatialIntruderRefiner`.

use egraph::layout::{SpatialRefinerRoundStats, SpatialRefinerStats};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Per-round diagnostics for `SpatialIntruderRefiner`.
#[pyclass(
    name = "SpatialRefinerRoundStats",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySpatialRefinerRoundStats {
    inner: SpatialRefinerRoundStats,
}

impl PySpatialRefinerRoundStats {
    pub(crate) fn from_inner(inner: SpatialRefinerRoundStats) -> Self {
        Self { inner }
    }

    fn dict_impl(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let rd = PyDict::new(py);
        rd.set_item("round", self.inner.round)?;
        rd.set_item("n_anchors", self.inner.n_anchors)?;
        rd.set_item("n_fn", self.inner.n_fn)?;
        rd.set_item("n_fp", self.inner.n_fp)?;
        rd.set_item("n_sep", self.inner.n_sep)?;
        rd.set_item("n_pairs_total", self.inner.n_pairs_total)?;
        rd.set_item("skipped", self.inner.skipped)?;
        Ok(rd.unbind())
    }
}

#[pymethods]
impl PySpatialRefinerRoundStats {
    #[getter]
    fn round(&self) -> usize {
        self.inner.round
    }

    #[getter]
    fn n_anchors(&self) -> usize {
        self.inner.n_anchors
    }

    #[getter]
    fn n_fn(&self) -> usize {
        self.inner.n_fn
    }

    #[getter]
    fn n_fp(&self) -> usize {
        self.inner.n_fp
    }

    #[getter]
    fn n_sep(&self) -> usize {
        self.inner.n_sep
    }

    #[getter]
    fn n_pairs_total(&self) -> usize {
        self.inner.n_pairs_total
    }

    #[getter]
    fn skipped(&self) -> bool {
        self.inner.skipped
    }

    #[pyo3(signature = (), text_signature = "($self)")]
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.dict_impl(py)
    }

    #[pyo3(signature = (), text_signature = "($self)")]
    fn keys(&self) -> Vec<&'static str> {
        vec![
            "round",
            "n_anchors",
            "n_fn",
            "n_fp",
            "n_sep",
            "n_pairs_total",
            "skipped",
        ]
    }

    fn __contains__(&self, key: &str) -> bool {
        matches!(
            key,
            "round" | "n_anchors" | "n_fn" | "n_fp" | "n_sep" | "n_pairs_total" | "skipped"
        )
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "round" => Ok(self.inner.round.into_pyobject(py)?.into_any().unbind()),
            "n_anchors" => Ok(self.inner.n_anchors.into_pyobject(py)?.into_any().unbind()),
            "n_fn" => Ok(self.inner.n_fn.into_pyobject(py)?.into_any().unbind()),
            "n_fp" => Ok(self.inner.n_fp.into_pyobject(py)?.into_any().unbind()),
            "n_sep" => Ok(self.inner.n_sep.into_pyobject(py)?.into_any().unbind()),
            "n_pairs_total" => Ok(self
                .inner
                .n_pairs_total
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "skipped" => Ok(self
                .inner
                .skipped
                .into_pyobject(py)?
                .to_owned()
                .into_any()
                .unbind()),
            _ => Err(PyKeyError::new_err(format!(
                "unknown SpatialRefinerRoundStats key {key:?}"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SpatialRefinerRoundStats(round={}, n_anchors={}, n_fn={}, n_fp={}, skipped={})",
            self.inner.round,
            self.inner.n_anchors,
            self.inner.n_fn,
            self.inner.n_fp,
            self.inner.skipped
        )
    }
}

/// Cumulative and per-round diagnostics for `SpatialIntruderRefiner`.
#[pyclass(
    name = "SpatialRefinerStats",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySpatialRefinerStats {
    inner: SpatialRefinerStats,
}

impl PySpatialRefinerStats {
    pub(crate) fn from_inner(inner: &SpatialRefinerStats) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}

#[pymethods]
impl PySpatialRefinerStats {
    #[getter]
    fn rounds_executed(&self) -> usize {
        self.inner.rounds_executed
    }

    #[getter]
    fn n_anchors_total(&self) -> usize {
        self.inner.n_anchors_total
    }

    #[getter]
    fn n_fn_total(&self) -> usize {
        self.inner.n_fn_total
    }

    #[getter]
    fn n_fp_total(&self) -> usize {
        self.inner.n_fp_total
    }

    #[getter]
    fn n_sep_total(&self) -> usize {
        self.inner.n_sep_total
    }

    #[getter]
    fn n_pairs_total(&self) -> usize {
        self.inner.n_pairs_total
    }

    #[getter]
    fn initial_scale(&self) -> f64 {
        self.inner.initial_scale
    }

    #[getter]
    fn final_scale(&self) -> f64 {
        self.inner.final_scale
    }

    #[getter]
    fn d_floor(&self) -> f64 {
        self.inner.d_floor
    }

    #[getter]
    fn scale_ratio(&self) -> f64 {
        self.inner.scale_ratio
    }

    #[getter]
    fn used_hub_fallback(&self) -> bool {
        self.inner.used_hub_fallback
    }

    #[getter]
    fn hinge_fp(&self) -> bool {
        self.inner.hinge_fp
    }

    #[getter]
    fn round_stats(&self) -> Vec<PySpatialRefinerRoundStats> {
        self.inner
            .round_stats
            .iter()
            .cloned()
            .map(PySpatialRefinerRoundStats::from_inner)
            .collect()
    }

    #[pyo3(signature = (), text_signature = "($self)")]
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("rounds_executed", self.inner.rounds_executed)?;
        dict.set_item("n_anchors_total", self.inner.n_anchors_total)?;
        dict.set_item("n_fn_total", self.inner.n_fn_total)?;
        dict.set_item("n_fp_total", self.inner.n_fp_total)?;
        dict.set_item("n_sep_total", self.inner.n_sep_total)?;
        dict.set_item("n_pairs_total", self.inner.n_pairs_total)?;
        dict.set_item("initial_scale", self.inner.initial_scale)?;
        dict.set_item("final_scale", self.inner.final_scale)?;
        dict.set_item("d_floor", self.inner.d_floor)?;
        dict.set_item("scale_ratio", self.inner.scale_ratio)?;
        dict.set_item("used_hub_fallback", self.inner.used_hub_fallback)?;
        dict.set_item("hinge_fp", self.inner.hinge_fp)?;
        let rounds_list = PyList::empty(py);
        for r in &self.inner.round_stats {
            rounds_list.append(PySpatialRefinerRoundStats::from_inner(r.clone()).dict_impl(py)?)?;
        }
        dict.set_item("round_stats", rounds_list)?;
        Ok(dict.unbind())
    }

    #[pyo3(signature = (), text_signature = "($self)")]
    fn keys(&self) -> Vec<&'static str> {
        vec![
            "rounds_executed",
            "n_anchors_total",
            "n_fn_total",
            "n_fp_total",
            "n_sep_total",
            "n_pairs_total",
            "initial_scale",
            "final_scale",
            "d_floor",
            "scale_ratio",
            "used_hub_fallback",
            "hinge_fp",
            "round_stats",
        ]
    }

    fn __contains__(&self, key: &str) -> bool {
        matches!(
            key,
            "rounds_executed"
                | "n_anchors_total"
                | "n_fn_total"
                | "n_fp_total"
                | "n_sep_total"
                | "n_pairs_total"
                | "initial_scale"
                | "final_scale"
                | "d_floor"
                | "scale_ratio"
                | "used_hub_fallback"
                | "hinge_fp"
                | "round_stats"
        )
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "rounds_executed" => Ok(self
                .inner
                .rounds_executed
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "n_anchors_total" => Ok(self
                .inner
                .n_anchors_total
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "n_fn_total" => Ok(self.inner.n_fn_total.into_pyobject(py)?.into_any().unbind()),
            "n_fp_total" => Ok(self.inner.n_fp_total.into_pyobject(py)?.into_any().unbind()),
            "n_sep_total" => Ok(self
                .inner
                .n_sep_total
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "n_pairs_total" => Ok(self
                .inner
                .n_pairs_total
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "initial_scale" => Ok(self
                .inner
                .initial_scale
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "final_scale" => Ok(self
                .inner
                .final_scale
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "d_floor" => Ok(self.inner.d_floor.into_pyobject(py)?.into_any().unbind()),
            "scale_ratio" => Ok(self
                .inner
                .scale_ratio
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "used_hub_fallback" => Ok(self
                .inner
                .used_hub_fallback
                .into_pyobject(py)?
                .to_owned()
                .into_any()
                .unbind()),
            "hinge_fp" => Ok(self
                .inner
                .hinge_fp
                .into_pyobject(py)?
                .to_owned()
                .into_any()
                .unbind()),
            "round_stats" => Ok(self.round_stats().into_pyobject(py)?.into_any().unbind()),
            _ => Err(PyKeyError::new_err(format!(
                "unknown SpatialRefinerStats key {key:?}"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SpatialRefinerStats(rounds_executed={}, n_anchors_total={}, n_fn_total={}, n_fp_total={}, scale_ratio={:.4})",
            self.inner.rounds_executed,
            self.inner.n_anchors_total,
            self.inner.n_fn_total,
            self.inner.n_fp_total,
            self.inner.scale_ratio
        )
    }
}
