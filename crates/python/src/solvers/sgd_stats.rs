//! Frozen diagnostic result classes emitted by Sparse Barrier and DoG SGD constructors.

use egraph::layout::SparseBarrierStats;
use egraph::neighborhood::DogVaradhanStats;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Construction diagnostics emitted by `SgdSolver.sparse_barrier_from_landmarks` and `sparse_barrier_dog`.
#[pyclass(
    name = "SparseBarrierStats",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PySparseBarrierStats {
    inner: SparseBarrierStats,
}

impl PySparseBarrierStats {
    pub(crate) fn from_core(inner: SparseBarrierStats) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySparseBarrierStats {
    #[getter]
    fn n_pairs_1hop(&self) -> usize {
        self.inner.n_pairs_1hop
    }
    #[getter]
    fn n_pairs_2hop(&self) -> usize {
        self.inner.n_pairs_2hop
    }
    #[getter]
    fn n_pairs_non2(&self) -> usize {
        self.inner.n_pairs_non2
    }
    #[getter]
    fn n_pairs_hinge(&self) -> usize {
        self.inner.n_pairs_hinge
    }
    #[getter]
    fn n_pairs_two_sided(&self) -> usize {
        self.inner.n_pairs_two_sided
    }
    #[getter]
    fn perplexity_val(&self) -> f64 {
        self.inner.perplexity_val
    }
    #[getter]
    fn per_node_neg_used(&self) -> usize {
        self.inner.per_node_neg_used
    }
    #[getter]
    fn perplexity_source(&self) -> &'static str {
        self.inner.perplexity_source
    }
    #[getter]
    fn d_floor(&self) -> f64 {
        self.inner.d_floor
    }
    #[getter]
    fn gamma_eff(&self) -> f64 {
        self.inner.gamma_eff
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("n_pairs_1hop", self.inner.n_pairs_1hop)?;
        d.set_item("n_pairs_2hop", self.inner.n_pairs_2hop)?;
        d.set_item("n_pairs_non2", self.inner.n_pairs_non2)?;
        d.set_item("n_pairs_hinge", self.inner.n_pairs_hinge)?;
        d.set_item("n_pairs_two_sided", self.inner.n_pairs_two_sided)?;
        d.set_item("perplexity_val", self.inner.perplexity_val)?;
        d.set_item("per_node_neg_used", self.inner.per_node_neg_used)?;
        d.set_item("perplexity_source", self.inner.perplexity_source)?;
        d.set_item("d_floor", self.inner.d_floor)?;
        d.set_item("gamma_eff", self.inner.gamma_eff)?;
        Ok(d.unbind())
    }

    fn keys(&self) -> Vec<&'static str> {
        vec![
            "n_pairs_1hop",
            "n_pairs_2hop",
            "n_pairs_non2",
            "n_pairs_hinge",
            "n_pairs_two_sided",
            "perplexity_val",
            "per_node_neg_used",
            "perplexity_source",
            "d_floor",
            "gamma_eff",
        ]
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "n_pairs_1hop" => self.inner.n_pairs_1hop.into_py_any(py),
            "n_pairs_2hop" => self.inner.n_pairs_2hop.into_py_any(py),
            "n_pairs_non2" => self.inner.n_pairs_non2.into_py_any(py),
            "n_pairs_hinge" => self.inner.n_pairs_hinge.into_py_any(py),
            "n_pairs_two_sided" => self.inner.n_pairs_two_sided.into_py_any(py),
            "perplexity_val" => self.inner.perplexity_val.into_py_any(py),
            "per_node_neg_used" => self.inner.per_node_neg_used.into_py_any(py),
            "perplexity_source" => self.inner.perplexity_source.into_py_any(py),
            "d_floor" => self.inner.d_floor.into_py_any(py),
            "gamma_eff" => self.inner.gamma_eff.into_py_any(py),
            _ => Err(PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __contains__(&self, key: &str) -> bool {
        matches!(
            key,
            "n_pairs_1hop"
                | "n_pairs_2hop"
                | "n_pairs_non2"
                | "n_pairs_hinge"
                | "n_pairs_two_sided"
                | "perplexity_val"
                | "per_node_neg_used"
                | "perplexity_source"
                | "d_floor"
                | "gamma_eff"
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "SparseBarrierStats(1hop={}, 2hop={}, non2={}, hinge={}, two_sided={}, source='{}')",
            self.inner.n_pairs_1hop,
            self.inner.n_pairs_2hop,
            self.inner.n_pairs_non2,
            self.inner.n_pairs_hinge,
            self.inner.n_pairs_two_sided,
            self.inner.perplexity_source,
        )
    }
}

/// Regularized Difference-of-Gaussians (DoG) Varadhan landmark diagnostics.
#[pyclass(
    name = "DogVaradhanStats",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyDogVaradhanStats {
    inner: DogVaradhanStats,
}

impl PyDogVaradhanStats {
    pub(crate) fn from_core(inner: DogVaradhanStats) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDogVaradhanStats {
    #[getter]
    fn t1(&self) -> f64 {
        self.inner.t1
    }
    #[getter]
    fn t2(&self) -> f64 {
        self.inner.t2
    }
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha
    }
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree
    }
    #[getter]
    fn delta_psd(&self) -> f64 {
        self.inner.delta_psd
    }
    #[getter]
    fn h_min(&self) -> f64 {
        self.inner.h_min
    }
    #[getter]
    fn k_min(&self) -> f64 {
        self.inner.k_min
    }
    #[getter]
    fn k_med(&self) -> f64 {
        self.inner.k_med
    }
    #[getter]
    fn beta_auto(&self) -> f64 {
        self.inner.beta_auto
    }
    #[getter]
    fn beta_eff(&self) -> f64 {
        self.inner.beta_eff
    }
    #[getter]
    fn n_landmarks(&self) -> usize {
        self.inner.n_landmarks
    }
    #[getter]
    fn t_coarse(&self) -> f64 {
        self.inner.t_coarse
    }
    #[getter]
    fn used_coarse_scaffold(&self) -> bool {
        self.inner.used_coarse_scaffold
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("t1", self.inner.t1)?;
        d.set_item("t2", self.inner.t2)?;
        d.set_item("alpha", self.inner.alpha)?;
        d.set_item("degree", self.inner.degree)?;
        d.set_item("delta_psd", self.inner.delta_psd)?;
        d.set_item("h_min", self.inner.h_min)?;
        d.set_item("k_min", self.inner.k_min)?;
        d.set_item("k_med", self.inner.k_med)?;
        d.set_item("beta_auto", self.inner.beta_auto)?;
        d.set_item("beta_eff", self.inner.beta_eff)?;
        d.set_item("n_landmarks", self.inner.n_landmarks)?;
        d.set_item("t_coarse", self.inner.t_coarse)?;
        d.set_item("used_coarse_scaffold", self.inner.used_coarse_scaffold)?;
        Ok(d.unbind())
    }

    fn keys(&self) -> Vec<&'static str> {
        vec![
            "t1",
            "t2",
            "alpha",
            "degree",
            "delta_psd",
            "h_min",
            "k_min",
            "k_med",
            "beta_auto",
            "beta_eff",
            "n_landmarks",
            "t_coarse",
            "used_coarse_scaffold",
        ]
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "t1" => self.inner.t1.into_py_any(py),
            "t2" => self.inner.t2.into_py_any(py),
            "alpha" => self.inner.alpha.into_py_any(py),
            "degree" => self.inner.degree.into_py_any(py),
            "delta_psd" => self.inner.delta_psd.into_py_any(py),
            "h_min" => self.inner.h_min.into_py_any(py),
            "k_min" => self.inner.k_min.into_py_any(py),
            "k_med" => self.inner.k_med.into_py_any(py),
            "beta_auto" => self.inner.beta_auto.into_py_any(py),
            "beta_eff" => self.inner.beta_eff.into_py_any(py),
            "n_landmarks" => self.inner.n_landmarks.into_py_any(py),
            "t_coarse" => self.inner.t_coarse.into_py_any(py),
            "used_coarse_scaffold" => self.inner.used_coarse_scaffold.into_py_any(py),
            _ => Err(PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __contains__(&self, key: &str) -> bool {
        matches!(
            key,
            "t1" | "t2"
                | "alpha"
                | "degree"
                | "delta_psd"
                | "h_min"
                | "k_min"
                | "k_med"
                | "beta_auto"
                | "beta_eff"
                | "n_landmarks"
                | "t_coarse"
                | "used_coarse_scaffold"
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "DogVaradhanStats(t1={:.4}, t2={:.4}, degree={}, n_landmarks={}, used_coarse_scaffold={})",
            self.inner.t1,
            self.inner.t2,
            self.inner.degree,
            self.inner.n_landmarks,
            self.inner.used_coarse_scaffold
        )
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySparseBarrierStats>()?;
    m.add_class::<PyDogVaradhanStats>()?;
    Ok(())
}
