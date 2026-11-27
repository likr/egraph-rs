//! Effective edge resistance and Foster diagnostics bindings.

use egraph::linalg::{
    AmgPcgConfig, EffectiveResistanceConfig, EffectiveResistanceResult,
    compute_effective_resistances as linalg_compute_effective_resistances,
};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{StrictF64, StrictU64, StrictUsize, core_error_to_py};

/// Effective edge resistances and Foster's theorem diagnostics for canonical simple undirected edges.
#[pyclass(
    name = "EffectiveResistanceResult",
    module = "egraph._core",
    from_py_object
)]
#[derive(Clone)]
pub struct PyEffectiveResistanceResult {
    inner: EffectiveResistanceResult,
}

#[pymethods]
impl PyEffectiveResistanceResult {
    /// Number of vertices in the graph.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes
    }

    /// Number of canonical undirected simple edges (`u < v`).
    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges
    }

    /// Number of connected components (including isolated vertices).
    #[getter]
    fn num_components(&self) -> usize {
        self.inner.num_components
    }

    /// Observed Foster sum `sum_e w_e * R_e`.
    #[getter]
    fn foster_sum(&self) -> f64 {
        self.inner.foster_sum
    }

    /// Exact theoretical Foster sum `num_nodes - num_components`.
    #[getter]
    fn expected_foster(&self) -> f64 {
        self.inner.expected_foster
    }

    /// Relative Foster error `|foster_sum - expected_foster| / max(1, expected_foster)`.
    #[getter]
    fn foster_rel_err(&self) -> f64 {
        self.inner.foster_rel_err
    }

    /// Whether the exact dense pseudoinverse solver path was used.
    #[getter]
    fn used_exact(&self) -> bool {
        self.inner.used_exact
    }

    /// Total AMG-PCG iterations executed across all JL RHS columns.
    #[getter]
    fn amg_total_iterations(&self) -> usize {
        self.inner.amg_total_iterations
    }

    /// Maximum relative residual across all JL RHS columns.
    #[getter]
    fn amg_max_relative_residual(&self) -> f64 {
        self.inner.amg_max_relative_residual
    }

    /// Whether every JL PCG column converged within tolerance.
    #[getter]
    fn amg_all_converged(&self) -> bool {
        self.inner.amg_all_converged
    }

    /// Return a `uint32` NumPy copy of canonical edge source endpoints `u < v`.
    fn edge_u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, &self.inner.edge_u)
    }

    /// Return a `uint32` NumPy copy of canonical edge target endpoints `v > u`.
    fn edge_v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, &self.inner.edge_v)
    }

    /// Return a `float64` NumPy copy of canonical edge lengths `l_e`.
    fn edge_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.edge_lengths)
    }

    /// Return a `float64` NumPy copy of canonical edge conductances `w_e = 1 / l_e`.
    fn conductances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.conductances)
    }

    /// Return a `float64` NumPy copy of effective resistances `R_e`.
    fn effective_resistances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.effective_resistances)
    }

    /// Return scalar Foster and solver diagnostics as a Python dictionary.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("num_nodes", self.inner.num_nodes)?;
        dict.set_item("num_edges", self.inner.num_edges)?;
        dict.set_item("num_components", self.inner.num_components)?;
        dict.set_item("foster_sum", self.inner.foster_sum)?;
        dict.set_item("expected_foster", self.inner.expected_foster)?;
        dict.set_item("foster_rel_err", self.inner.foster_rel_err)?;
        dict.set_item("used_exact", self.inner.used_exact)?;
        dict.set_item("amg_total_iterations", self.inner.amg_total_iterations)?;
        dict.set_item(
            "amg_max_relative_residual",
            self.inner.amg_max_relative_residual,
        )?;
        dict.set_item("amg_all_converged", self.inner.amg_all_converged)?;
        Ok(dict.unbind())
    }

    /// Return `num_edges`.
    fn __len__(&self) -> usize {
        self.inner.num_edges
    }

    /// Return a concise string representation of the `EffectiveResistanceResult`.
    fn __repr__(&self) -> String {
        format!(
            "EffectiveResistanceResult(num_nodes={}, num_edges={}, num_components={}, foster_rel_err={:.3e}, used_exact={})",
            self.inner.num_nodes,
            self.inner.num_edges,
            self.inner.num_components,
            self.inner.foster_rel_err,
            if self.inner.used_exact {
                "True"
            } else {
                "False"
            },
        )
    }
}

/// Compute effective edge resistances and Foster's theorem diagnostics via exact pseudoinverse
/// (for small graphs) or Spielman-Srivastava JL projection solved with `AmgPcgSolver`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (graph, *, jl_dim = StrictUsize(128), seed = StrictU64(42), weights_are_lengths = true, exact = None, exact_max_nodes = StrictUsize(256), eig_eps = StrictF64(1e-12), rel_tol = StrictF64(1e-8), max_iterations = StrictUsize(500)),
    text_signature = "(graph, *, jl_dim=128, seed=42, weights_are_lengths=True, exact=None, exact_max_nodes=256, eig_eps=1e-12, rel_tol=1e-8, max_iterations=500)"
)]
fn compute_effective_resistances(
    py: Python<'_>,
    graph: &PyCsrGraph,
    jl_dim: StrictUsize,
    seed: StrictU64,
    weights_are_lengths: bool,
    exact: Option<bool>,
    exact_max_nodes: StrictUsize,
    eig_eps: StrictF64,
    rel_tol: StrictF64,
    max_iterations: StrictUsize,
) -> PyResult<PyEffectiveResistanceResult> {
    let cfg = EffectiveResistanceConfig {
        jl_dim: jl_dim.0,
        seed: seed.0,
        weights_are_lengths,
        exact,
        exact_max_nodes: exact_max_nodes.0,
        eig_eps: eig_eps.0,
        amg_config: AmgPcgConfig {
            rel_tol: rel_tol.0,
            max_iterations: max_iterations.0,
            eig_eps: eig_eps.0,
            seed: seed.0,
            ..AmgPcgConfig::default()
        },
    };
    let g = graph.inner();
    let res = py
        .detach(|| linalg_compute_effective_resistances(&*g, &cfg))
        .map_err(core_error_to_py)?;
    Ok(PyEffectiveResistanceResult { inner: res })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEffectiveResistanceResult>()?;
    m.add_function(wrap_pyfunction!(compute_effective_resistances, m)?)?;
    Ok(())
}
