//! Devriendt-Lambiotte resistance curvature flow bindings.

use egraph::linalg::{
    AmgPcgConfig, EffectiveResistanceConfig, ResistanceCurvatureConfig,
    ResistanceCurvatureFlowResult, ResistanceCurvatureStep,
    resistance_curvature_flow as linalg_resistance_curvature_flow,
};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{StrictF64, StrictU64, StrictUsize, core_error_to_py};

fn step_to_py_dict(py: Python<'_>, step: &ResistanceCurvatureStep) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("step", step.step)?;
    dict.set_item("foster_sum", step.foster_sum)?;
    dict.set_item("expected_foster", step.expected_foster)?;
    dict.set_item("foster_rel_err", step.foster_rel_err)?;
    dict.set_item("mean_link_curvature", step.mean_link_curvature)?;
    dict.set_item("std_link_curvature", step.std_link_curvature)?;
    dict.set_item("min_link_curvature", step.min_link_curvature)?;
    dict.set_item("max_link_curvature", step.max_link_curvature)?;
    dict.set_item("mean_vertex_curvature", step.mean_vertex_curvature)?;
    dict.set_item("min_vertex_curvature", step.min_vertex_curvature)?;
    dict.set_item("max_vertex_curvature", step.max_vertex_curvature)?;
    dict.set_item("used_exact", step.used_exact)?;
    dict.set_item("amg_total_iterations", step.amg_total_iterations)?;
    dict.set_item("amg_max_relative_residual", step.amg_max_relative_residual)?;
    Ok(dict.unbind())
}

/// Devriendt-Lambiotte resistance curvature flow result with reweighted graphs and geometry classification.
#[pyclass(
    name = "ResistanceCurvatureFlowResult",
    module = "egraph._core",
    from_py_object
)]
#[derive(Clone)]
pub struct PyResistanceCurvatureFlowResult {
    inner: ResistanceCurvatureFlowResult,
}

#[pymethods]
impl PyResistanceCurvatureFlowResult {
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

    /// Final Foster sum `sum_e w_e * R_e`.
    #[getter]
    fn foster_sum(&self) -> f64 {
        self.inner.foster_sum
    }

    /// Exact theoretical Foster sum `num_nodes - num_components`.
    #[getter]
    fn expected_foster(&self) -> f64 {
        self.inner.expected_foster
    }

    /// Final relative Foster error.
    #[getter]
    fn foster_rel_err(&self) -> f64 {
        self.inner.foster_rel_err
    }

    /// Mean final link resistance curvature.
    #[getter]
    fn mean_link_curvature(&self) -> f64 {
        self.inner.mean_link_curvature
    }

    /// Population standard deviation of final link resistance curvatures.
    #[getter]
    fn std_link_curvature(&self) -> f64 {
        self.inner.std_link_curvature
    }

    /// Minimum final link resistance curvature.
    #[getter]
    fn min_link_curvature(&self) -> f64 {
        self.inner.min_link_curvature
    }

    /// Maximum final link resistance curvature.
    #[getter]
    fn max_link_curvature(&self) -> f64 {
        self.inner.max_link_curvature
    }

    /// Mean final vertex resistance curvature.
    #[getter]
    fn mean_vertex_curvature(&self) -> f64 {
        self.inner.mean_vertex_curvature
    }

    /// Minimum final vertex resistance curvature.
    #[getter]
    fn min_vertex_curvature(&self) -> f64 {
        self.inner.min_vertex_curvature
    }

    /// Maximum final vertex resistance curvature.
    #[getter]
    fn max_vertex_curvature(&self) -> f64 {
        self.inner.max_vertex_curvature
    }

    /// Whether the final evaluation used the exact dense pseudoinverse solver.
    #[getter]
    fn used_exact(&self) -> bool {
        self.inner.used_exact
    }

    /// Total AMG-PCG iterations executed in the final evaluation.
    #[getter]
    fn amg_total_iterations(&self) -> usize {
        self.inner.amg_total_iterations
    }

    /// Maximum relative PCG residual in the final evaluation.
    #[getter]
    fn amg_max_relative_residual(&self) -> f64 {
        self.inner.amg_max_relative_residual
    }

    /// Whether every JL PCG column converged in the final evaluation.
    #[getter]
    fn amg_all_converged(&self) -> bool {
        self.inner.amg_all_converged
    }

    /// Suggested 2D geometry (`"hyperbolic"`, `"spherical"`, `"torus"`, or `"euclidean"`).
    #[getter]
    fn suggested_geometry(&self) -> &'static str {
        self.inner.suggested_geometry.as_str()
    }

    /// Suggested `ManifoldLayout2d` kind (`"hyperbolic"`, `"spherical"`, `"torus"`), or `None` for Euclidean.
    #[getter]
    fn suggested_manifold(&self) -> Option<&'static str> {
        self.inner
            .suggested_manifold_kind()
            .map(|_| self.inner.suggested_geometry.as_str())
    }

    /// Return a `uint32` NumPy copy of canonical edge source endpoints `u < v`.
    fn edge_u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, &self.inner.edge_u)
    }

    /// Return a `uint32` NumPy copy of canonical edge target endpoints `v > u`.
    fn edge_v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, &self.inner.edge_v)
    }

    /// Return a `float64` NumPy copy of final normalized edge lengths `l_e`.
    fn edge_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.edge_lengths)
    }

    /// Return a `float64` NumPy copy of final edge conductances `w_e = 1 / l_e`.
    fn conductances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.conductances)
    }

    /// Return a `float64` NumPy copy of final effective resistances `R_e`.
    fn effective_resistances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.effective_resistances)
    }

    /// Return a `float64` NumPy copy of final vertex resistance curvatures `p_i`.
    fn vertex_curvatures<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.vertex_curvatures)
    }

    /// Return a `float64` NumPy copy of final link resistance curvatures `kappa_res(u, v)`.
    fn link_curvatures<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.link_curvatures)
    }

    /// Return per-step diagnostics across `0..=steps` as a list of dictionaries.
    fn step_history(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        self.inner
            .step_history
            .iter()
            .map(|step| step_to_py_dict(py, step))
            .collect()
    }

    /// Build a weighted undirected `CsrGraph` storing the final edge lengths `l_e`.
    fn length_graph(&self, py: Python<'_>) -> PyResult<PyCsrGraph> {
        let g = py
            .detach(|| self.inner.to_length_graph())
            .map_err(core_error_to_py)?;
        Ok(PyCsrGraph::from_csr_graph(g))
    }

    /// Build a weighted undirected `CsrGraph` storing the final conductances `w_e = 1 / l_e`.
    fn conductance_graph(&self, py: Python<'_>) -> PyResult<PyCsrGraph> {
        let g = py
            .detach(|| self.inner.to_conductance_graph())
            .map_err(core_error_to_py)?;
        Ok(PyCsrGraph::from_csr_graph(g))
    }

    /// Return scalar curvature, Foster, and geometry diagnostics as a Python dictionary.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("num_nodes", self.inner.num_nodes)?;
        dict.set_item("num_edges", self.inner.num_edges)?;
        dict.set_item("num_components", self.inner.num_components)?;
        dict.set_item("foster_sum", self.inner.foster_sum)?;
        dict.set_item("expected_foster", self.inner.expected_foster)?;
        dict.set_item("foster_rel_err", self.inner.foster_rel_err)?;
        dict.set_item("mean_link_curvature", self.inner.mean_link_curvature)?;
        dict.set_item("std_link_curvature", self.inner.std_link_curvature)?;
        dict.set_item("min_link_curvature", self.inner.min_link_curvature)?;
        dict.set_item("max_link_curvature", self.inner.max_link_curvature)?;
        dict.set_item("mean_vertex_curvature", self.inner.mean_vertex_curvature)?;
        dict.set_item("min_vertex_curvature", self.inner.min_vertex_curvature)?;
        dict.set_item("max_vertex_curvature", self.inner.max_vertex_curvature)?;
        dict.set_item("used_exact", self.inner.used_exact)?;
        dict.set_item("amg_total_iterations", self.inner.amg_total_iterations)?;
        dict.set_item(
            "amg_max_relative_residual",
            self.inner.amg_max_relative_residual,
        )?;
        dict.set_item("amg_all_converged", self.inner.amg_all_converged)?;
        dict.set_item("suggested_geometry", self.suggested_geometry())?;
        dict.set_item("suggested_manifold", self.suggested_manifold())?;
        Ok(dict.unbind())
    }

    /// Return `num_edges`.
    fn __len__(&self) -> usize {
        self.inner.num_edges
    }

    /// Return a concise string representation of the `ResistanceCurvatureFlowResult`.
    fn __repr__(&self) -> String {
        format!(
            "ResistanceCurvatureFlowResult(num_nodes={}, num_edges={}, mean_link_curvature={:.4}, suggested_geometry='{}', foster_rel_err={:.3e})",
            self.inner.num_nodes,
            self.inner.num_edges,
            self.inner.mean_link_curvature,
            self.suggested_geometry(),
            self.inner.foster_rel_err,
        )
    }
}

/// Run Devriendt-Lambiotte resistance curvature flow on `graph` and classify target geometry.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (graph, *, steps = StrictUsize(2), eta = StrictF64(0.2), kappa_max = StrictF64(2.0), min_edge_length = StrictF64(1e-6), jl_dim = StrictUsize(64), seed = StrictU64(42), weights_are_lengths = true, exact = None, exact_max_nodes = StrictUsize(64), eig_eps = StrictF64(1e-12), rel_tol = StrictF64(1e-8), max_iterations = StrictUsize(500), hyperbolic_threshold = StrictF64(-0.05), spherical_threshold = StrictF64(0.2), torus_std_threshold = StrictF64(0.05)),
    text_signature = "(graph, *, steps=2, eta=0.2, kappa_max=2.0, min_edge_length=1e-6, jl_dim=64, seed=42, weights_are_lengths=True, exact=None, exact_max_nodes=64, eig_eps=1e-12, rel_tol=1e-8, max_iterations=500, hyperbolic_threshold=-0.05, spherical_threshold=0.2, torus_std_threshold=0.05)"
)]
fn resistance_curvature_flow(
    py: Python<'_>,
    graph: &PyCsrGraph,
    steps: StrictUsize,
    eta: StrictF64,
    kappa_max: StrictF64,
    min_edge_length: StrictF64,
    jl_dim: StrictUsize,
    seed: StrictU64,
    weights_are_lengths: bool,
    exact: Option<bool>,
    exact_max_nodes: StrictUsize,
    eig_eps: StrictF64,
    rel_tol: StrictF64,
    max_iterations: StrictUsize,
    hyperbolic_threshold: StrictF64,
    spherical_threshold: StrictF64,
    torus_std_threshold: StrictF64,
) -> PyResult<PyResistanceCurvatureFlowResult> {
    let cfg = ResistanceCurvatureConfig {
        steps: steps.0,
        eta: eta.0,
        kappa_max: kappa_max.0,
        min_edge_length: min_edge_length.0,
        resistance: EffectiveResistanceConfig {
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
        },
        hyperbolic_threshold: hyperbolic_threshold.0,
        spherical_threshold: spherical_threshold.0,
        torus_std_threshold: torus_std_threshold.0,
    };
    let g = graph.inner();
    let res = py
        .detach(|| linalg_resistance_curvature_flow(&*g, &cfg))
        .map_err(core_error_to_py)?;
    Ok(PyResistanceCurvatureFlowResult { inner: res })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResistanceCurvatureFlowResult>()?;
    m.add_function(wrap_pyfunction!(resistance_curvature_flow, m)?)?;
    Ok(())
}
