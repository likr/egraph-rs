//! Graph UMAP (`GuMapSolver`) layout solver bindings.

use egraph::layout as egraph_layout;
use egraph_layout::GuMapSolver;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, solver_status_str};
use crate::neighborhood::{PyDogVaradhanLandmarks, PyKnnGraph, PySparseProbabilities};
use crate::solver_common::{DualIterativeSolver, SolverAction, impl_dual_solver};

use super::gumap_builders::{
    build_gumap_from_fuzzy_set, build_gumap_from_graph_dog, build_gumap_from_knn,
    build_gumap_from_landmarks,
};
use super::layout_dispatch::{
    execute_any_layout_action, impl_dual_manifold_solver, impl_dual_nd_solver,
};

impl_dual_solver!(GuMapInner, GuMapSolver);
impl_dual_nd_solver!(GuMapInner, GuMapSolver);
impl_dual_manifold_solver!(GuMapInner, GuMapSolver);

/// Cooperative Graph UMAP (`GuMap`) layout solver with `dist_sq.powf(b)` gradient evaluation.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d` in `"float64"` and `"float32"`
/// precision while releasing the Python GIL during stepping.
#[pyclass(name = "GuMapSolver", module = "egraph._core")]
pub struct PyGuMapSolver {
    inner: GuMapInner,
}

impl PyGuMapSolver {
    pub(super) fn from_inner(inner: GuMapInner) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyGuMapSolver {
    /// Construct a `GuMapSolver` from a `CsrGraph` using 1+2-hop priority CSR and regularized DoG Varadhan distances.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, max_2hop_per_node = StrictUsize(32), *, degree = None, num_landmarks = StrictUsize(64), degree_normalize = true, p_exp = StrictF64(0.5), eps_psd = StrictF64(1e-4), t_coarse = StrictF64(48.0), t1 = None, t2 = None, alpha = None, beta_floor = None, min_degree = StrictUsize(2), jl_dim = StrictUsize(64), iterations = StrictUsize(200), negative_sample_rate = StrictUsize(5), learning_rate = StrictF64(1.0), min_dist = StrictF64(0.1), spread = StrictF64(1.0), a = None, b = None, sigma_iters = StrictUsize(64), sigma_tolerance = StrictF64(1e-5), symmetric_negative = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, max_2hop_per_node=32, *, degree=None, num_landmarks=64, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0, t1=None, t2=None, alpha=None, beta_floor=None, min_degree=2, jl_dim=64, iterations=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, symmetric_negative=True, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        max_2hop_per_node: StrictUsize,
        degree: Option<StrictUsize>,
        num_landmarks: StrictUsize,
        degree_normalize: bool,
        p_exp: StrictF64,
        eps_psd: StrictF64,
        t_coarse: StrictF64,
        t1: Option<StrictF64>,
        t2: Option<StrictF64>,
        alpha: Option<StrictF64>,
        beta_floor: Option<StrictF64>,
        min_degree: StrictUsize,
        jl_dim: StrictUsize,
        iterations: StrictUsize,
        negative_sample_rate: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        symmetric_negative: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_gumap_from_graph_dog(
            py,
            graph,
            max_2hop_per_node,
            degree,
            num_landmarks,
            degree_normalize,
            p_exp,
            eps_psd,
            t_coarse,
            t1,
            t2,
            alpha,
            beta_floor,
            min_degree,
            jl_dim,
            iterations,
            negative_sample_rate,
            learning_rate,
            min_dist,
            spread,
            a,
            b,
            sigma_iters,
            sigma_tolerance,
            symmetric_negative,
            seed,
            dtype,
        )
    }

    /// Construct a `GuMapSolver` from a precomputed fuzzy simplicial set `SparseProbabilities`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (probs, *, iterations = StrictUsize(200), negative_sample_rate = StrictUsize(5), learning_rate = StrictF64(1.0), min_dist = StrictF64(0.1), spread = StrictF64(1.0), a = None, b = None, sigma_iters = StrictUsize(64), sigma_tolerance = StrictF64(1e-5), symmetric_negative = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(probs, *, iterations=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, symmetric_negative=True, seed=42, dtype='float64')"
    )]
    fn from_fuzzy_set(
        py: Python<'_>,
        probs: &PySparseProbabilities,
        iterations: StrictUsize,
        negative_sample_rate: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        symmetric_negative: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_gumap_from_fuzzy_set(
            py,
            probs,
            iterations,
            negative_sample_rate,
            learning_rate,
            min_dist,
            spread,
            a,
            b,
            sigma_iters,
            sigma_tolerance,
            symmetric_negative,
            seed,
            dtype,
        )
    }

    /// Construct a `GuMapSolver` from a validated `KnnGraph` via `compute_fuzzy_simplicial_set`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (knn, *, iterations = StrictUsize(200), negative_sample_rate = StrictUsize(5), learning_rate = StrictF64(1.0), min_dist = StrictF64(0.1), spread = StrictF64(1.0), a = None, b = None, sigma_iters = StrictUsize(64), sigma_tolerance = StrictF64(1e-5), symmetric_negative = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(knn, *, iterations=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, symmetric_negative=True, seed=42, dtype='float64')"
    )]
    fn from_knn(
        py: Python<'_>,
        knn: &PyKnnGraph,
        iterations: StrictUsize,
        negative_sample_rate: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        symmetric_negative: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_gumap_from_knn(
            py,
            knn,
            iterations,
            negative_sample_rate,
            learning_rate,
            min_dist,
            spread,
            a,
            b,
            sigma_iters,
            sigma_tolerance,
            symmetric_negative,
            seed,
            dtype,
        )
    }

    /// Construct a `GuMapSolver` from a `CsrGraph` using 1+2-hop priority CSR and regularized DoG Varadhan distances.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, max_2hop_per_node = StrictUsize(32), *, degree = None, num_landmarks = StrictUsize(64), degree_normalize = true, p_exp = StrictF64(0.5), eps_psd = StrictF64(1e-4), t_coarse = StrictF64(48.0), t1 = None, t2 = None, alpha = None, beta_floor = None, min_degree = StrictUsize(2), jl_dim = StrictUsize(64), iterations = StrictUsize(200), negative_sample_rate = StrictUsize(5), learning_rate = StrictF64(1.0), min_dist = StrictF64(0.1), spread = StrictF64(1.0), a = None, b = None, sigma_iters = StrictUsize(64), sigma_tolerance = StrictF64(1e-5), symmetric_negative = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, max_2hop_per_node=32, *, degree=None, num_landmarks=64, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0, t1=None, t2=None, alpha=None, beta_floor=None, min_degree=2, jl_dim=64, iterations=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, symmetric_negative=True, seed=42, dtype='float64')"
    )]
    fn from_graph_dog(
        py: Python<'_>,
        graph: &PyCsrGraph,
        max_2hop_per_node: StrictUsize,
        degree: Option<StrictUsize>,
        num_landmarks: StrictUsize,
        degree_normalize: bool,
        p_exp: StrictF64,
        eps_psd: StrictF64,
        t_coarse: StrictF64,
        t1: Option<StrictF64>,
        t2: Option<StrictF64>,
        alpha: Option<StrictF64>,
        beta_floor: Option<StrictF64>,
        min_degree: StrictUsize,
        jl_dim: StrictUsize,
        iterations: StrictUsize,
        negative_sample_rate: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        symmetric_negative: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_gumap_from_graph_dog(
            py,
            graph,
            max_2hop_per_node,
            degree,
            num_landmarks,
            degree_normalize,
            p_exp,
            eps_psd,
            t_coarse,
            t1,
            t2,
            alpha,
            beta_floor,
            min_degree,
            jl_dim,
            iterations,
            negative_sample_rate,
            learning_rate,
            min_dist,
            spread,
            a,
            b,
            sigma_iters,
            sigma_tolerance,
            symmetric_negative,
            seed,
            dtype,
        )
    }

    /// Construct a `GuMapSolver` from precomputed `DogVaradhanLandmarks` and 1+2-hop priority CSR.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, landmarks, max_2hop_per_node = StrictUsize(32), *, iterations = StrictUsize(200), negative_sample_rate = StrictUsize(5), learning_rate = StrictF64(1.0), min_dist = StrictF64(0.1), spread = StrictF64(1.0), a = None, b = None, sigma_iters = StrictUsize(64), sigma_tolerance = StrictF64(1e-5), symmetric_negative = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, landmarks, max_2hop_per_node=32, *, iterations=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, symmetric_negative=True, seed=42, dtype='float64')"
    )]
    fn from_landmarks(
        py: Python<'_>,
        graph: &PyCsrGraph,
        landmarks: &PyDogVaradhanLandmarks,
        max_2hop_per_node: StrictUsize,
        iterations: StrictUsize,
        negative_sample_rate: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        symmetric_negative: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_gumap_from_landmarks(
            py,
            graph,
            landmarks,
            max_2hop_per_node,
            iterations,
            negative_sample_rate,
            learning_rate,
            min_dist,
            spread,
            a,
            b,
            sigma_iters,
            sigma_tolerance,
            symmetric_negative,
            seed,
            dtype,
        )
    }

    /// Execute a single optimization epoch in place on `layout` (`Layout2d | LayoutNd | ManifoldLayout2d`).
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_any_layout_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` optimization epochs in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyAny>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_any_layout_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run optimization to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_any_layout_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of epochs executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Scheduled total number of epochs (`n_epochs`).
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Number of stored directed positive probability edges.
    #[getter]
    fn num_edges(&self) -> usize {
        match &self.inner {
            GuMapInner::F64(s) => s.num_edges(),
            GuMapInner::F32(s) => s.num_edges(),
        }
    }

    /// Expected number of vertices in `Layout2d`.
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            GuMapInner::F64(s) => s.num_nodes(),
            GuMapInner::F32(s) => s.num_nodes(),
        }
    }

    /// Resolved low-dimensional curve parameter `a`.
    #[getter]
    fn a(&self) -> f64 {
        self.ab().0
    }

    /// Resolved low-dimensional curve parameter `b`.
    #[getter]
    fn b(&self) -> f64 {
        self.ab().1
    }

    /// Resolved low-dimensional curve parameters `(a, b)`.
    #[getter]
    fn ab(&self) -> (f64, f64) {
        match &self.inner {
            GuMapInner::F64(s) => s.ab(),
            GuMapInner::F32(s) => s.ab(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `GuMapSolver`.
    fn __repr__(&self) -> String {
        format!(
            "GuMapSolver(num_nodes={}, num_edges={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.num_edges(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
