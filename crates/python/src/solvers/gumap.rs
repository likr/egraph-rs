//! Graph UMAP (`GuMapSolver`) layout solver bindings across `Layout2d | Layout3d | LayoutNd | ManifoldLayout2d`.

use egraph::core::ManifoldKind;
use egraph::layout::GuMapSolver;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, solver_status_str};
use crate::neighborhood::{PyDogVaradhanLandmarks, PyKnnGraph, PySparseProbabilities};
use crate::solver_common::format_optional_usize;

use super::gumap_builders::{
    build_gumap_from_fuzzy_set, build_gumap_from_graph_dog, build_gumap_from_knn,
    build_gumap_from_landmarks,
};
use super::layout_dispatch::{
    DualTargetSolver, SolverAction, SolverTargetKind, execute_target_action,
    impl_riemannian_target_solver, match_riemannian_target,
};

impl_riemannian_target_solver!(GuMapInner, GuMapSolver);

/// Cooperative Graph UMAP (`GuMap`) layout solver supporting `Layout2d`, `Layout3d`, `LayoutNd`, and `ManifoldLayout2d`.
#[pyclass(name = "GuMapSolver", module = "egraph._core")]
pub struct PyGuMapSolver {
    inner: GuMapInner,
    locked_manifold: Option<ManifoldKind>,
}

impl PyGuMapSolver {
    pub(super) fn from_inner(inner: GuMapInner) -> Self {
        Self {
            inner,
            locked_manifold: None,
        }
    }
}

#[pymethods]
impl PyGuMapSolver {
    /// Construct a `GuMapSolver` from a `CsrGraph` using 1+2-hop priority CSR and regularized DoG Varadhan distances.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, max_2hop_per_node = StrictUsize(32), *, target = SolverTargetKind::Layout2d, degree = None, num_landmarks = StrictUsize(64), degree_normalize = true, p_exp = StrictF64(0.5), eps_psd = StrictF64(1e-4), t_coarse = StrictF64(48.0), t1 = None, t2 = None, alpha = None, beta_floor = None, min_degree = StrictUsize(2), jl_dim = StrictUsize(64), iterations = StrictUsize(200), negative_sample_rate = StrictUsize(5), learning_rate = StrictF64(1.0), min_dist = StrictF64(0.1), spread = StrictF64(1.0), a = None, b = None, sigma_iters = StrictUsize(64), sigma_tolerance = StrictF64(1e-5), symmetric_negative = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, max_2hop_per_node=32, *, target='layout2d', degree=None, num_landmarks=64, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0, t1=None, t2=None, alpha=None, beta_floor=None, min_degree=2, jl_dim=64, iterations=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, symmetric_negative=True, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        max_2hop_per_node: StrictUsize,
        target: SolverTargetKind,
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
            target,
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
        signature = (probs, *, target = SolverTargetKind::Layout2d, iterations = StrictUsize(200), negative_sample_rate = StrictUsize(5), learning_rate = StrictF64(1.0), min_dist = StrictF64(0.1), spread = StrictF64(1.0), a = None, b = None, sigma_iters = StrictUsize(64), sigma_tolerance = StrictF64(1e-5), symmetric_negative = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(probs, *, target='layout2d', iterations=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, symmetric_negative=True, seed=42, dtype='float64')"
    )]
    fn from_fuzzy_set(
        py: Python<'_>,
        probs: &PySparseProbabilities,
        target: SolverTargetKind,
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
            target,
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
        signature = (knn, *, target = SolverTargetKind::Layout2d, iterations = StrictUsize(200), negative_sample_rate = StrictUsize(5), learning_rate = StrictF64(1.0), min_dist = StrictF64(0.1), spread = StrictF64(1.0), a = None, b = None, sigma_iters = StrictUsize(64), sigma_tolerance = StrictF64(1e-5), symmetric_negative = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(knn, *, target='layout2d', iterations=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, symmetric_negative=True, seed=42, dtype='float64')"
    )]
    fn from_knn(
        py: Python<'_>,
        knn: &PyKnnGraph,
        target: SolverTargetKind,
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
            target,
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
        signature = (graph, max_2hop_per_node = StrictUsize(32), *, target = SolverTargetKind::Layout2d, degree = None, num_landmarks = StrictUsize(64), degree_normalize = true, p_exp = StrictF64(0.5), eps_psd = StrictF64(1e-4), t_coarse = StrictF64(48.0), t1 = None, t2 = None, alpha = None, beta_floor = None, min_degree = StrictUsize(2), jl_dim = StrictUsize(64), iterations = StrictUsize(200), negative_sample_rate = StrictUsize(5), learning_rate = StrictF64(1.0), min_dist = StrictF64(0.1), spread = StrictF64(1.0), a = None, b = None, sigma_iters = StrictUsize(64), sigma_tolerance = StrictF64(1e-5), symmetric_negative = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, max_2hop_per_node=32, *, target='layout2d', degree=None, num_landmarks=64, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0, t1=None, t2=None, alpha=None, beta_floor=None, min_degree=2, jl_dim=64, iterations=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, symmetric_negative=True, seed=42, dtype='float64')"
    )]
    fn from_graph_dog(
        py: Python<'_>,
        graph: &PyCsrGraph,
        max_2hop_per_node: StrictUsize,
        target: SolverTargetKind,
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
            target,
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
        signature = (graph, landmarks, max_2hop_per_node = StrictUsize(32), *, target = SolverTargetKind::Layout2d, iterations = StrictUsize(200), negative_sample_rate = StrictUsize(5), learning_rate = StrictF64(1.0), min_dist = StrictF64(0.1), spread = StrictF64(1.0), a = None, b = None, sigma_iters = StrictUsize(64), sigma_tolerance = StrictF64(1e-5), symmetric_negative = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, landmarks, max_2hop_per_node=32, *, target='layout2d', iterations=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, symmetric_negative=True, seed=42, dtype='float64')"
    )]
    fn from_landmarks(
        py: Python<'_>,
        graph: &PyCsrGraph,
        landmarks: &PyDogVaradhanLandmarks,
        max_2hop_per_node: StrictUsize,
        target: SolverTargetKind,
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
            target,
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

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_target_action(
            &mut self.inner,
            &mut self.locked_manifold,
            py,
            layout,
            SolverAction::Step,
        )
    }

    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyAny>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_target_action(
            &mut self.inner,
            &mut self.locked_manifold,
            py,
            layout,
            SolverAction::RunSteps(steps.0),
        )
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_target_action(
            &mut self.inner,
            &mut self.locked_manifold,
            py,
            layout,
            SolverAction::Run,
        )
    }

    #[getter]
    fn target(&self) -> &'static str {
        self.inner.target_kind().as_str()
    }

    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    #[getter]
    fn total_iterations(&self) -> Option<usize> {
        self.inner.solver_total_iterations()
    }

    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    #[getter]
    fn num_edges(&self) -> usize {
        match_riemannian_target!(GuMapInner, &self.inner, s => s.num_edges())
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        match_riemannian_target!(GuMapInner, &self.inner, s => s.num_nodes())
    }

    #[getter]
    fn a(&self) -> f64 {
        self.ab().0
    }

    #[getter]
    fn b(&self) -> f64 {
        self.ab().1
    }

    #[getter]
    fn ab(&self) -> (f64, f64) {
        match_riemannian_target!(GuMapInner, &self.inner, s => s.ab())
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "GuMapSolver(num_nodes={}, target='{}', num_edges={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.target(),
            self.num_edges(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}
