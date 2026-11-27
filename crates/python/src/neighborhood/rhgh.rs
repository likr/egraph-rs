//! Pivoted Resolvent Heat-Geodesic Hybrid (`RhghLandmarks`) Python bindings.

use egraph::neighborhood::{
    RhghLandmarkConfig, RhghLandmarkStats, RhghLandmarks,
    compute_pivoted_rhgh_landmarks_result as nb_compute_pivoted_rhgh_landmarks_result,
};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::dog_varadhan::PyDogVaradhanLandmarks;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d};

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
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
) -> PyResult<(PyRhghLandmarks, Py<PyDict>)> {
    let cfg = RhghLandmarkConfig {
        gamma: gamma.map(|v| v.0),
        p_exp: p_exp.map(|v| v.0),
        n_landmarks: num_landmarks.0,
        min_degree: min_degree.0,
        num_probes: num_probes.0,
        rel_tol: rel_tol.0,
        max_iterations: max_iterations.0,
        normalize_1hop,
        seed: seed.0,
    };
    let g = graph.inner();
    let (inner, st) = py
        .detach(|| nb_compute_pivoted_rhgh_landmarks_result(&*g, &cfg))
        .map_err(core_error_to_py)?;

    let dict = PyDict::new(py);
    dict.set_item("gamma", st.gamma)?;
    dict.set_item("p_exp", st.p_exp)?;
    dict.set_item("deg_cv", st.deg_cv)?;
    dict.set_item("leaf_fraction", st.leaf_fraction)?;
    dict.set_item("mean_degree", st.mean_degree)?;
    dict.set_item("exp2", st.exp2)?;
    dict.set_item("s_sw", st.s_sw)?;
    dict.set_item("n_landmarks", st.n_landmarks)?;
    dict.set_item("num_probes", st.num_probes)?;
    dict.set_item("max_pcg_iters", st.max_pcg_iters)?;
    dict.set_item("total_pcg_iters", st.total_pcg_iters)?;
    dict.set_item("all_converged", st.all_converged)?;
    dict.set_item("edge_med_scale", st.edge_med_scale)?;

    Ok((PyRhghLandmarks { inner, stats: st }, dict.unbind()))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRhghLandmarks>()?;
    m.add_function(wrap_pyfunction!(compute_pivoted_rhgh_landmarks, m)?)?;
    Ok(())
}
