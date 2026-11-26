//! Thin Python binding facade for `IbUmapSolver`.

use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, solver_status_str};
use crate::layout_buffers::PyLayout2d;
use crate::neighborhood::{PyDogVaradhanLandmarks, PyKnnGraph, PySparseProbabilities};
use crate::solver_common::{DualIterativeSolver, SolverAction, execute_solver_action};

use super::ib_umap_builders::{
    build_ib_umap_from_fuzzy_set, build_ib_umap_from_knn, build_ib_umap_from_landmarks,
    build_ib_umap_new,
};
use super::t_fdp::FieldLayoutInner;

/// Cooperative interpolation-based batch Graph UMAP (`ibUMAP`) layout solver.
///
/// Supports `"ibfft"` (default), `"barnes_hut"`, and `"exact"` repulsion backends in both
/// `"float64"` and `"float32"` precision, releasing the Python GIL during stepping.
#[pyclass(name = "IbUmapSolver", module = "egraph._core")]
pub struct PyIbUmapSolver {
    inner: FieldLayoutInner,
}

#[pymethods]
impl PyIbUmapSolver {
    /// Construct an `IbUmapSolver` from a `CsrGraph` using 1+2-hop pairs and regularized DoG Varadhan landmarks.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            max_2hop_per_node = StrictUsize(30),
            degree = None,
            num_landmarks = StrictUsize(64),
            min_degree = StrictUsize(2),
            jl_dim = StrictUsize(64),
            degree_normalize = true,
            p_exp = StrictF64(0.5),
            eps_psd = StrictF64(1e-4),
            t_coarse = StrictF64(48.0),
            t1 = None,
            t2 = None,
            alpha = None,
            beta_floor = None,
            backend = "ibfft",
            n_epochs = StrictUsize(200),
            learning_rate = StrictF64(1.0),
            min_dist = StrictF64(0.1),
            spread = StrictF64(1.0),
            a = None,
            b = None,
            c_k = StrictF64(4.0),
            tau_scale = StrictF64(4.0),
            hub_damping = true,
            theta = StrictF64(0.5),
            ibfft_order = StrictUsize(1),
            ibfft_intervals = StrictUsize(25),
            ibfft_min_grid_dim = StrictUsize(64),
            ibfft_max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, max_2hop_per_node=30, degree=None, num_landmarks=64, min_degree=2, jl_dim=64, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0, t1=None, t2=None, alpha=None, beta_floor=None, backend='ibfft', n_epochs=200, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, c_k=4.0, tau_scale=4.0, hub_damping=True, theta=0.5, ibfft_order=1, ibfft_intervals=25, ibfft_min_grid_dim=64, ibfft_max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        max_2hop_per_node: StrictUsize,
        degree: Option<StrictUsize>,
        num_landmarks: StrictUsize,
        min_degree: StrictUsize,
        jl_dim: StrictUsize,
        degree_normalize: bool,
        p_exp: StrictF64,
        eps_psd: StrictF64,
        t_coarse: StrictF64,
        t1: Option<StrictF64>,
        t2: Option<StrictF64>,
        alpha: Option<StrictF64>,
        beta_floor: Option<StrictF64>,
        backend: &str,
        n_epochs: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        c_k: StrictF64,
        tau_scale: StrictF64,
        hub_damping: bool,
        theta: StrictF64,
        ibfft_order: StrictUsize,
        ibfft_intervals: StrictUsize,
        ibfft_min_grid_dim: StrictUsize,
        ibfft_max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let inner = build_ib_umap_new(
            py,
            graph,
            max_2hop_per_node,
            degree,
            num_landmarks,
            min_degree,
            jl_dim,
            degree_normalize,
            p_exp,
            eps_psd,
            t_coarse,
            t1,
            t2,
            alpha,
            beta_floor,
            backend,
            n_epochs,
            learning_rate,
            min_dist,
            spread,
            a,
            b,
            c_k,
            tau_scale,
            hub_damping,
            theta,
            ibfft_order,
            ibfft_intervals,
            ibfft_min_grid_dim,
            ibfft_max_grid_dim,
            seed,
            dtype,
        )?;
        Ok(Self { inner })
    }

    /// Construct an `IbUmapSolver` from precomputed `DogVaradhanLandmarks` and 1+2-hop pairs.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            landmarks,
            max_2hop_per_node = StrictUsize(32),
            *,
            backend = "ibfft",
            n_epochs = StrictUsize(200),
            learning_rate = StrictF64(1.0),
            min_dist = StrictF64(0.1),
            spread = StrictF64(1.0),
            a = None,
            b = None,
            c_k = StrictF64(4.0),
            tau_scale = StrictF64(4.0),
            hub_damping = true,
            theta = StrictF64(0.5),
            ibfft_order = StrictUsize(1),
            ibfft_intervals = StrictUsize(25),
            ibfft_min_grid_dim = StrictUsize(64),
            ibfft_max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, landmarks, max_2hop_per_node=32, *, backend='ibfft', n_epochs=200, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, c_k=4.0, tau_scale=4.0, hub_damping=True, theta=0.5, ibfft_order=1, ibfft_intervals=25, ibfft_min_grid_dim=64, ibfft_max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_landmarks(
        py: Python<'_>,
        graph: &PyCsrGraph,
        landmarks: &PyDogVaradhanLandmarks,
        max_2hop_per_node: StrictUsize,
        backend: &str,
        n_epochs: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        c_k: StrictF64,
        tau_scale: StrictF64,
        hub_damping: bool,
        theta: StrictF64,
        ibfft_order: StrictUsize,
        ibfft_intervals: StrictUsize,
        ibfft_min_grid_dim: StrictUsize,
        ibfft_max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let inner = build_ib_umap_from_landmarks(
            py,
            graph,
            landmarks,
            max_2hop_per_node,
            backend,
            n_epochs,
            learning_rate,
            min_dist,
            spread,
            a,
            b,
            c_k,
            tau_scale,
            hub_damping,
            theta,
            ibfft_order,
            ibfft_intervals,
            ibfft_min_grid_dim,
            ibfft_max_grid_dim,
            seed,
            dtype,
        )?;
        Ok(Self { inner })
    }

    /// Construct an `IbUmapSolver` from a validated `KnnGraph` via `compute_fuzzy_simplicial_set`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            knn,
            *,
            sigma_iters = StrictUsize(64),
            sigma_tol = StrictF64(1e-5),
            backend = "ibfft",
            n_epochs = StrictUsize(200),
            learning_rate = StrictF64(1.0),
            min_dist = StrictF64(0.1),
            spread = StrictF64(1.0),
            a = None,
            b = None,
            c_k = StrictF64(4.0),
            tau_scale = StrictF64(4.0),
            hub_damping = true,
            theta = StrictF64(0.5),
            ibfft_order = StrictUsize(1),
            ibfft_intervals = StrictUsize(25),
            ibfft_min_grid_dim = StrictUsize(64),
            ibfft_max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(knn, *, sigma_iters=64, sigma_tol=1e-5, backend='ibfft', n_epochs=200, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, c_k=4.0, tau_scale=4.0, hub_damping=True, theta=0.5, ibfft_order=1, ibfft_intervals=25, ibfft_min_grid_dim=64, ibfft_max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_knn(
        py: Python<'_>,
        knn: &PyKnnGraph,
        sigma_iters: StrictUsize,
        sigma_tol: StrictF64,
        backend: &str,
        n_epochs: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        c_k: StrictF64,
        tau_scale: StrictF64,
        hub_damping: bool,
        theta: StrictF64,
        ibfft_order: StrictUsize,
        ibfft_intervals: StrictUsize,
        ibfft_min_grid_dim: StrictUsize,
        ibfft_max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let inner = build_ib_umap_from_knn(
            py,
            knn,
            sigma_iters,
            sigma_tol,
            backend,
            n_epochs,
            learning_rate,
            min_dist,
            spread,
            a,
            b,
            c_k,
            tau_scale,
            hub_damping,
            theta,
            ibfft_order,
            ibfft_intervals,
            ibfft_min_grid_dim,
            ibfft_max_grid_dim,
            seed,
            dtype,
        )?;
        Ok(Self { inner })
    }

    /// Construct an `IbUmapSolver` directly from a precomputed fuzzy simplicial set `SparseProbabilities`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            probs,
            *,
            degrees = None,
            backend = "ibfft",
            n_epochs = StrictUsize(200),
            learning_rate = StrictF64(1.0),
            min_dist = StrictF64(0.1),
            spread = StrictF64(1.0),
            a = None,
            b = None,
            c_k = StrictF64(4.0),
            tau_scale = StrictF64(4.0),
            hub_damping = true,
            theta = StrictF64(0.5),
            ibfft_order = StrictUsize(1),
            ibfft_intervals = StrictUsize(25),
            ibfft_min_grid_dim = StrictUsize(64),
            ibfft_max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(probs, *, degrees=None, backend='ibfft', n_epochs=200, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, c_k=4.0, tau_scale=4.0, hub_damping=True, theta=0.5, ibfft_order=1, ibfft_intervals=25, ibfft_min_grid_dim=64, ibfft_max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_fuzzy_set(
        py: Python<'_>,
        probs: &PySparseProbabilities,
        degrees: Option<&Bound<'_, PyAny>>,
        backend: &str,
        n_epochs: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        c_k: StrictF64,
        tau_scale: StrictF64,
        hub_damping: bool,
        theta: StrictF64,
        ibfft_order: StrictUsize,
        ibfft_intervals: StrictUsize,
        ibfft_min_grid_dim: StrictUsize,
        ibfft_max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let inner = build_ib_umap_from_fuzzy_set(
            py,
            probs,
            degrees,
            backend,
            n_epochs,
            learning_rate,
            min_dist,
            spread,
            a,
            b,
            c_k,
            tau_scale,
            hub_damping,
            theta,
            ibfft_order,
            ibfft_intervals,
            ibfft_min_grid_dim,
            ibfft_max_grid_dim,
            seed,
            dtype,
        )?;
        Ok(Self { inner })
    }

    /// Execute a single optimization epoch in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` optimization epochs in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run optimization to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
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

    /// Total scheduled epochs (`n_epochs`).
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Expected number of vertices in `Layout2d`.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Current repulsion schedule multiplier (`1.0` for `ibUMAP`).
    #[getter]
    fn current_rho(&self) -> f64 {
        self.inner.current_rho()
    }

    /// Number of attractive edges in the `IbUmapSolver`.
    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Cumulative number of 2D FFT transforms executed across all steps.
    #[getter]
    fn total_fft_2d_calls(&self) -> u64 {
        self.inner.total_fft_2d_calls()
    }

    /// Fitted or explicit attractive kernel `a` parameter, or `None` if unavailable.
    #[getter]
    fn a(&self) -> Option<f64> {
        self.inner.ab().map(|(a, _)| a)
    }

    /// Fitted or explicit attractive kernel `b` parameter, or `None` if unavailable.
    #[getter]
    fn b(&self) -> Option<f64> {
        self.inner.ab().map(|(_, b)| b)
    }

    /// Fitted or explicit `(a, b)` kernel parameters, or `None` if unavailable.
    #[getter]
    fn ab(&self) -> Option<(f64, f64)> {
        self.inner.ab()
    }

    /// Return a concise string representation of the `IbUmapSolver`.
    fn __repr__(&self) -> String {
        format!(
            "IbUmapSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
