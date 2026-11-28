//! Pivoted Resolvent Heat-Geodesic Hybrid (`RhghLandmarks`) Python bindings.

use egraph::neighborhood::{
    RhghLandmarkConfig, RhghLandmarkStats, RhghLandmarks,
    compute_pivoted_rhgh_landmarks_result as nb_compute_pivoted_rhgh_landmarks_result,
};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::dog_varadhan::PyDogVaradhanLandmarks;
use crate::csr_graph::PyCsrGraph;
use crate::errors::invalid_input_err;
use crate::ffi_convert::{StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d};

/// Frozen telemetry emitted by `compute_pivoted_rhgh_landmarks`.
#[pyclass(
    name = "RhghLandmarkStats",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyRhghLandmarkStats {
    inner: RhghLandmarkStats,
}

#[pymethods]
impl PyRhghLandmarkStats {
    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma
    }
    #[getter]
    fn p_exp(&self) -> f64 {
        self.inner.p_exp
    }
    #[getter]
    fn deg_cv(&self) -> f64 {
        self.inner.deg_cv
    }
    #[getter]
    fn leaf_fraction(&self) -> f64 {
        self.inner.leaf_fraction
    }
    #[getter]
    fn mean_degree(&self) -> f64 {
        self.inner.mean_degree
    }
    #[getter]
    fn exp2(&self) -> f64 {
        self.inner.exp2
    }
    #[getter]
    fn s_sw(&self) -> f64 {
        self.inner.s_sw
    }
    #[getter]
    fn n_landmarks(&self) -> usize {
        self.inner.n_landmarks
    }
    #[getter]
    fn num_probes(&self) -> usize {
        self.inner.num_probes
    }
    #[getter]
    fn max_pcg_iters(&self) -> usize {
        self.inner.max_pcg_iters
    }
    #[getter]
    fn total_pcg_iters(&self) -> usize {
        self.inner.total_pcg_iters
    }
    #[getter]
    fn all_converged(&self) -> bool {
        self.inner.all_converged
    }
    #[getter]
    fn edge_med_scale(&self) -> f64 {
        self.inner.edge_med_scale
    }

    #[pyo3(signature = (), text_signature = "($self)")]
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("gamma", self.inner.gamma)?;
        dict.set_item("p_exp", self.inner.p_exp)?;
        dict.set_item("deg_cv", self.inner.deg_cv)?;
        dict.set_item("leaf_fraction", self.inner.leaf_fraction)?;
        dict.set_item("mean_degree", self.inner.mean_degree)?;
        dict.set_item("exp2", self.inner.exp2)?;
        dict.set_item("s_sw", self.inner.s_sw)?;
        dict.set_item("n_landmarks", self.inner.n_landmarks)?;
        dict.set_item("num_probes", self.inner.num_probes)?;
        dict.set_item("max_pcg_iters", self.inner.max_pcg_iters)?;
        dict.set_item("total_pcg_iters", self.inner.total_pcg_iters)?;
        dict.set_item("all_converged", self.inner.all_converged)?;
        dict.set_item("edge_med_scale", self.inner.edge_med_scale)?;
        Ok(dict.unbind())
    }

    #[pyo3(signature = (), text_signature = "($self)")]
    fn keys(&self) -> Vec<&'static str> {
        vec![
            "gamma",
            "p_exp",
            "deg_cv",
            "leaf_fraction",
            "mean_degree",
            "exp2",
            "s_sw",
            "n_landmarks",
            "num_probes",
            "max_pcg_iters",
            "total_pcg_iters",
            "all_converged",
            "edge_med_scale",
        ]
    }

    fn __contains__(&self, key: &str) -> bool {
        matches!(
            key,
            "gamma"
                | "p_exp"
                | "deg_cv"
                | "leaf_fraction"
                | "mean_degree"
                | "exp2"
                | "s_sw"
                | "n_landmarks"
                | "num_probes"
                | "max_pcg_iters"
                | "total_pcg_iters"
                | "all_converged"
                | "edge_med_scale"
        )
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "gamma" => self.inner.gamma.into_py_any(py),
            "p_exp" => self.inner.p_exp.into_py_any(py),
            "deg_cv" => self.inner.deg_cv.into_py_any(py),
            "leaf_fraction" => self.inner.leaf_fraction.into_py_any(py),
            "mean_degree" => self.inner.mean_degree.into_py_any(py),
            "exp2" => self.inner.exp2.into_py_any(py),
            "s_sw" => self.inner.s_sw.into_py_any(py),
            "n_landmarks" => self.inner.n_landmarks.into_py_any(py),
            "num_probes" => self.inner.num_probes.into_py_any(py),
            "max_pcg_iters" => self.inner.max_pcg_iters.into_py_any(py),
            "total_pcg_iters" => self.inner.total_pcg_iters.into_py_any(py),
            "all_converged" => self.inner.all_converged.into_py_any(py),
            "edge_med_scale" => self.inner.edge_med_scale.into_py_any(py),
            _ => Err(PyKeyError::new_err(format!(
                "unknown RhghLandmarkStats key {key:?}"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RhghLandmarkStats(n_landmarks={}, gamma={:.4}, p_exp={:.4}, all_converged={})",
            self.inner.n_landmarks, self.inner.gamma, self.inner.p_exp, self.inner.all_converged
        )
    }
}

/// Frozen output of `compute_pivoted_rhgh_landmarks`.
#[pyclass(
    name = "RhghLandmarksResult",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyRhghLandmarksResult {
    landmarks: PyRhghLandmarks,
    stats: PyRhghLandmarkStats,
}

#[pymethods]
impl PyRhghLandmarksResult {
    #[getter]
    fn landmarks(&self) -> PyRhghLandmarks {
        self.landmarks.clone()
    }

    #[getter]
    fn stats(&self) -> PyRhghLandmarkStats {
        self.stats.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RhghLandmarksResult(num_nodes={}, num_landmarks={})",
            self.landmarks.inner.num_nodes(),
            self.landmarks.inner.num_landmarks()
        )
    }
}

/// Pivoted Resolvent Heat-Geodesic Hybrid (`RHGH`) landmark distance table and stats.
#[pyclass(name = "RhghLandmarks", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PyRhghLandmarks {
    inner: RhghLandmarks,
    stats: RhghLandmarkStats,
}

#[pymethods]
impl PyRhghLandmarks {
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    #[getter]
    fn num_landmarks(&self) -> usize {
        self.inner.num_landmarks()
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.stats.gamma
    }

    #[getter]
    fn p_exp(&self) -> f64 {
        self.stats.p_exp
    }

    #[getter]
    fn s_sw(&self) -> f64 {
        self.stats.s_sw
    }

    #[getter]
    fn exp2(&self) -> f64 {
        self.stats.exp2
    }

    #[getter]
    fn all_converged(&self) -> bool {
        self.stats.all_converged
    }

    #[getter]
    fn edge_med_scale(&self) -> f64 {
        self.stats.edge_med_scale
    }

    /// Return a `uint32` NumPy copy of selected landmark vertex IDs.
    fn landmarks<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, self.inner.landmarks())
    }

    /// Return the `(num_nodes, num_landmarks)` `float64` RHGH distance table.
    fn table<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = Array2::from_shape_vec(
            (self.inner.num_nodes(), self.inner.num_landmarks()),
            self.inner.d_rhgh().to_vec(),
        )
        .map_err(|e| invalid_input_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, arr))
    }

    /// Evaluate landmark ALT lower-bound distances for pairs `(u[i], v[i])`.
    #[pyo3(signature = (u, v), text_signature = "($self, u, v)")]
    fn alt_distances<'py>(
        &self,
        py: Python<'py>,
        u: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let u_vec = extract_u32_1d(u, "u")?;
        let v_vec = extract_u32_1d(v, "v")?;
        let out = py
            .detach(|| self.inner.alt_pair_distances(&u_vec, &v_vec))
            .map_err(core_error_to_py)?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// Convert this `RhghLandmarks` table into a `DogVaradhanLandmarks` wrapper.
    fn to_dog_landmarks(&self) -> PyDogVaradhanLandmarks {
        PyDogVaradhanLandmarks::from_rhgh_parts(self.inner.to_dog_landmarks(), &self.stats)
    }

    fn __len__(&self) -> usize {
        self.inner.num_nodes()
    }

    fn __repr__(&self) -> String {
        format!(
            "RhghLandmarks(num_nodes={}, num_landmarks={}, all_converged={})",
            self.inner.num_nodes(),
            self.inner.num_landmarks(),
            if self.stats.all_converged {
                "True"
            } else {
                "False"
            }
        )
    }
}

/// Compute pure-Rust pivoted Resolvent Heat-Geodesic Hybrid (`RHGH`) landmark tables.
///
/// Returns `(RhghLandmarks, stats_dict)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (
        graph,
        *,
        gamma = None,
        p_exp = None,
        num_landmarks = StrictUsize(64),
        min_degree = StrictUsize(2),
        num_probes = StrictUsize(64),
        rel_tol = StrictF64(1e-12),
        max_iterations = StrictUsize(500),
        normalize_1hop = true,
        seed = StrictU64(42)
    ),
    text_signature = "(graph, *, gamma=None, p_exp=None, num_landmarks=64, min_degree=2, num_probes=64, rel_tol=1e-12, max_iterations=500, normalize_1hop=True, seed=42)"
)]
fn compute_pivoted_rhgh_landmarks(
    py: Python<'_>,
    graph: &PyCsrGraph,
    gamma: Option<StrictF64>,
    p_exp: Option<StrictF64>,
    num_landmarks: StrictUsize,
    min_degree: StrictUsize,
    num_probes: StrictUsize,
    rel_tol: StrictF64,
    max_iterations: StrictUsize,
    normalize_1hop: bool,
    seed: StrictU64,
) -> PyResult<PyRhghLandmarksResult> {
    let cfg = RhghLandmarkConfig::new()
        .with_gamma(gamma.map(|v| v.0))
        .with_p_exp(p_exp.map(|v| v.0))
        .with_n_landmarks(num_landmarks.0)
        .with_min_degree(min_degree.0)
        .with_num_probes(num_probes.0)
        .with_rel_tol(rel_tol.0)
        .with_max_iterations(max_iterations.0)
        .with_normalize_1hop(normalize_1hop)
        .with_seed(seed.0);
    let g = graph.inner();
    let (inner, st) = py
        .detach(|| nb_compute_pivoted_rhgh_landmarks_result(&*g, &cfg))
        .map_err(core_error_to_py)?;

    Ok(PyRhghLandmarksResult {
        landmarks: PyRhghLandmarks {
            inner,
            stats: st.clone(),
        },
        stats: PyRhghLandmarkStats { inner: st },
    })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRhghLandmarks>()?;
    m.add_class::<PyRhghLandmarkStats>()?;
    m.add_class::<PyRhghLandmarksResult>()?;
    m.add_function(wrap_pyfunction!(compute_pivoted_rhgh_landmarks, m)?)?;
    Ok(())
}
