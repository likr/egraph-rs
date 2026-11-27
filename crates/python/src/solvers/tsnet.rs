//! Thin Python binding facade for `TsNetSolver`.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, solver_status_str};
use crate::neighborhood::{PyKnnGraph, PySparseProbabilities};
use crate::solver_common::{DualIterativeSolver, SolverAction};

use super::layout_dispatch::execute_any_layout_action;
use super::tsnet_builders::{
    build_from_graph_partial_bfs, build_from_graph_row_normalized, build_from_knn,
    build_from_probabilities, build_from_probabilities_with_stages, build_pure_2hop,
};
use super::tsnet_config::{TsNetInner, TsNetStageOpts};

macro_rules! tsnet_stage_opts {
    ($si:expr, $ex:expr, $lc1:expr, $lc2:expr, $lc3:expr, $lr3:expr, $lr:expr, $lrd:expr, $jp:expr, $mom:expr, $eps:expr, $th:expr, $iv:expr, $ord:expr, $ming:expr, $maxg:expr, $seed:expr) => {
        TsNetStageOpts {
            stage_iterations: $si,
            exaggeration: $ex.0,
            lambda_c_stage1: $lc1.0,
            lambda_c_stage2: $lc2.0,
            lambda_c_stage3: $lc3.0,
            lambda_r_stage3: $lr3.0,
            learning_rate: $lr.map(|v| v.0),
            learning_rate_divisor: $lrd.0,
            jacobi_precondition: $jp,
            momentum: $mom.0,
            epsilon_r: $eps.0,
            theta: $th.0,
            intervals: $iv.0,
            interpolation_order: $ord.0,
            min_grid_dim: $ming.0,
            max_grid_dim: $maxg.0,
            seed: $seed.0,
        }
    };
}

/// Cooperative unified tsNET solver supporting `"exact"`, `"barnes_hut"`, `"fft"`,
/// `"linear_fft"`, and `"auto"` backends across `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`.
#[pyclass(name = "TsNetSolver", module = "egraph._core")]
pub struct PyTsNetSolver {
    inner: TsNetInner,
}

impl PyTsNetSolver {
    pub(super) fn from_inner(inner: TsNetInner) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTsNetSolver {
    /// Construct a `TsNetSolver` directly from a `CsrGraph` using the Pure 1+2-Hop preset.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, backend = "fft", *, perplexity_c = StrictF64(0.9), min_perplexity = StrictF64(5.0), max_perplexity = StrictF64(1200.0), power = StrictF64(2.0), sigma_iters = StrictUsize(50), sigma_tolerance = StrictF64(1e-5), row_normalize = false, stage_iterations = None, exaggeration = StrictF64(4.0), lambda_c_stage1 = StrictF64(0.1), lambda_c_stage2 = StrictF64(0.01), lambda_c_stage3 = StrictF64(0.0), lambda_r_stage3 = StrictF64(0.0), learning_rate = None, learning_rate_divisor = StrictF64(3.0), momentum = StrictF64(0.8), epsilon_r = StrictF64(0.05), theta = StrictF64(0.5), intervals = StrictUsize(25), interpolation_order = StrictUsize(3), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), jacobi_precondition = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, backend='fft', *, perplexity_c=0.9, min_perplexity=5.0, max_perplexity=1200.0, power=2.0, sigma_iters=50, sigma_tolerance=1e-5, row_normalize=False, stage_iterations=None, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, learning_rate_divisor=3.0, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=3, min_grid_dim=64, max_grid_dim=512, jacobi_precondition=True, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        backend: &str,
        perplexity_c: StrictF64,
        min_perplexity: StrictF64,
        max_perplexity: StrictF64,
        power: StrictF64,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        row_normalize: bool,
        stage_iterations: Option<&Bound<'_, PyAny>>,
        exaggeration: StrictF64,
        lambda_c_stage1: StrictF64,
        lambda_c_stage2: StrictF64,
        lambda_c_stage3: StrictF64,
        lambda_r_stage3: StrictF64,
        learning_rate: Option<StrictF64>,
        learning_rate_divisor: StrictF64,
        momentum: StrictF64,
        epsilon_r: StrictF64,
        theta: StrictF64,
        intervals: StrictUsize,
        interpolation_order: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        jacobi_precondition: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let opts = tsnet_stage_opts! {
            stage_iterations,
            exaggeration, lambda_c_stage1, lambda_c_stage2, lambda_c_stage3, lambda_r_stage3,
            learning_rate, learning_rate_divisor, jacobi_precondition, momentum, epsilon_r, theta,
            intervals, interpolation_order, min_grid_dim, max_grid_dim, seed
        };
        build_pure_2hop(
            py,
            graph,
            backend,
            perplexity_c,
            min_perplexity,
            max_perplexity,
            power,
            sigma_iters,
            sigma_tolerance,
            row_normalize,
            opts,
            dtype,
        )
    }

    /// Construct a `TsNetSolver` directly from a `CsrGraph` using the Pure 1+2-Hop preset.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, backend = "fft", *, perplexity_c = StrictF64(0.9), min_perplexity = StrictF64(5.0), max_perplexity = StrictF64(1200.0), power = StrictF64(2.0), sigma_iters = StrictUsize(50), sigma_tolerance = StrictF64(1e-5), row_normalize = false, stage_iterations = None, exaggeration = StrictF64(4.0), lambda_c_stage1 = StrictF64(0.1), lambda_c_stage2 = StrictF64(0.01), lambda_c_stage3 = StrictF64(0.0), lambda_r_stage3 = StrictF64(0.0), learning_rate = None, learning_rate_divisor = StrictF64(3.0), momentum = StrictF64(0.8), epsilon_r = StrictF64(0.05), theta = StrictF64(0.5), intervals = StrictUsize(25), interpolation_order = StrictUsize(3), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), jacobi_precondition = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, backend='fft', *, perplexity_c=0.9, min_perplexity=5.0, max_perplexity=1200.0, power=2.0, sigma_iters=50, sigma_tolerance=1e-5, row_normalize=False, stage_iterations=None, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, learning_rate_divisor=3.0, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=3, min_grid_dim=64, max_grid_dim=512, jacobi_precondition=True, seed=42, dtype='float64')"
    )]
    fn pure_2hop(
        py: Python<'_>,
        graph: &PyCsrGraph,
        backend: &str,
        perplexity_c: StrictF64,
        min_perplexity: StrictF64,
        max_perplexity: StrictF64,
        power: StrictF64,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        row_normalize: bool,
        stage_iterations: Option<&Bound<'_, PyAny>>,
        exaggeration: StrictF64,
        lambda_c_stage1: StrictF64,
        lambda_c_stage2: StrictF64,
        lambda_c_stage3: StrictF64,
        lambda_r_stage3: StrictF64,
        learning_rate: Option<StrictF64>,
        learning_rate_divisor: StrictF64,
        momentum: StrictF64,
        epsilon_r: StrictF64,
        theta: StrictF64,
        intervals: StrictUsize,
        interpolation_order: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        jacobi_precondition: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let opts = tsnet_stage_opts! {
            stage_iterations,
            exaggeration, lambda_c_stage1, lambda_c_stage2, lambda_c_stage3, lambda_r_stage3,
            learning_rate, learning_rate_divisor, jacobi_precondition, momentum, epsilon_r, theta,
            intervals, interpolation_order, min_grid_dim, max_grid_dim, seed
        };
        build_pure_2hop(
            py,
            graph,
            backend,
            perplexity_c,
            min_perplexity,
            max_perplexity,
            power,
            sigma_iters,
            sigma_tolerance,
            row_normalize,
            opts,
            dtype,
        )
    }

    /// Construct a `TsNetSolver` from a precomputed `KnnGraph` and perplexities.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (knn, perplexities = None, uniform_perplexity = StrictF64(15.0), *, backend = "fft", power = StrictF64(2.0), sigma_iters = StrictUsize(50), sigma_tolerance = StrictF64(1e-5), row_normalize = false, stage_iterations = None, exaggeration = StrictF64(4.0), lambda_c_stage1 = StrictF64(0.1), lambda_c_stage2 = StrictF64(0.01), lambda_c_stage3 = StrictF64(0.0), lambda_r_stage3 = StrictF64(0.0), learning_rate = None, learning_rate_divisor = StrictF64(3.0), momentum = StrictF64(0.8), epsilon_r = StrictF64(0.05), theta = StrictF64(0.5), intervals = StrictUsize(25), interpolation_order = StrictUsize(3), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), jacobi_precondition = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(knn, perplexities=None, uniform_perplexity=15.0, *, backend='fft', power=2.0, sigma_iters=50, sigma_tolerance=1e-5, row_normalize=False, stage_iterations=None, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, learning_rate_divisor=3.0, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=3, min_grid_dim=64, max_grid_dim=512, jacobi_precondition=True, seed=42, dtype='float64')"
    )]
    fn from_knn(
        py: Python<'_>,
        knn: &PyKnnGraph,
        perplexities: Option<&Bound<'_, PyAny>>,
        uniform_perplexity: StrictF64,
        backend: &str,
        power: StrictF64,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        row_normalize: bool,
        stage_iterations: Option<&Bound<'_, PyAny>>,
        exaggeration: StrictF64,
        lambda_c_stage1: StrictF64,
        lambda_c_stage2: StrictF64,
        lambda_c_stage3: StrictF64,
        lambda_r_stage3: StrictF64,
        learning_rate: Option<StrictF64>,
        learning_rate_divisor: StrictF64,
        momentum: StrictF64,
        epsilon_r: StrictF64,
        theta: StrictF64,
        intervals: StrictUsize,
        interpolation_order: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        jacobi_precondition: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let opts = tsnet_stage_opts! {
            stage_iterations,
            exaggeration, lambda_c_stage1, lambda_c_stage2, lambda_c_stage3, lambda_r_stage3,
            learning_rate, learning_rate_divisor, jacobi_precondition, momentum, epsilon_r, theta,
            intervals, interpolation_order, min_grid_dim, max_grid_dim, seed
        };
        build_from_knn(
            py,
            knn,
            perplexities,
            uniform_perplexity,
            backend,
            power,
            sigma_iters,
            sigma_tolerance,
            row_normalize,
            opts,
            dtype,
        )
    }

    /// Construct a `TsNetSolver` directly from precomputed symmetric joint `SparseProbabilities`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (probs, backend = "fft", *, stage_iterations = None, exaggeration = StrictF64(4.0), lambda_c_stage1 = StrictF64(0.1), lambda_c_stage2 = StrictF64(0.01), lambda_c_stage3 = StrictF64(0.0), lambda_r_stage3 = StrictF64(0.0), learning_rate = None, learning_rate_divisor = StrictF64(3.0), momentum = StrictF64(0.8), epsilon_r = StrictF64(0.05), theta = StrictF64(0.5), intervals = StrictUsize(25), interpolation_order = StrictUsize(3), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), jacobi_precondition = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(probs, backend='fft', *, stage_iterations=None, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, learning_rate_divisor=3.0, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=3, min_grid_dim=64, max_grid_dim=512, jacobi_precondition=True, seed=42, dtype='float64')"
    )]
    fn from_probabilities(
        py: Python<'_>,
        probs: &PySparseProbabilities,
        backend: &str,
        stage_iterations: Option<&Bound<'_, PyAny>>,
        exaggeration: StrictF64,
        lambda_c_stage1: StrictF64,
        lambda_c_stage2: StrictF64,
        lambda_c_stage3: StrictF64,
        lambda_r_stage3: StrictF64,
        learning_rate: Option<StrictF64>,
        learning_rate_divisor: StrictF64,
        momentum: StrictF64,
        epsilon_r: StrictF64,
        theta: StrictF64,
        intervals: StrictUsize,
        interpolation_order: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        jacobi_precondition: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let opts = tsnet_stage_opts! {
            stage_iterations,
            exaggeration, lambda_c_stage1, lambda_c_stage2, lambda_c_stage3, lambda_r_stage3,
            learning_rate, learning_rate_divisor, jacobi_precondition, momentum, epsilon_r, theta,
            intervals, interpolation_order, min_grid_dim, max_grid_dim, seed
        };
        build_from_probabilities(py, probs, backend, opts, dtype)
    }

    /// Construct a `TsNetSolver` from `SparseProbabilities` with an explicit list of `TsNetStage` instances.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (probs, stages, backend = "fft", *, learning_rate = None, learning_rate_divisor = StrictF64(3.0), momentum = StrictF64(0.8), epsilon_r = StrictF64(0.05), theta = StrictF64(0.5), intervals = StrictUsize(25), interpolation_order = StrictUsize(3), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), jacobi_precondition = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(probs, stages, backend='fft', *, learning_rate=None, learning_rate_divisor=3.0, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=3, min_grid_dim=64, max_grid_dim=512, jacobi_precondition=True, seed=42, dtype='float64')"
    )]
    fn from_probabilities_with_stages(
        py: Python<'_>,
        probs: &PySparseProbabilities,
        stages: &Bound<'_, PyAny>,
        backend: &str,
        learning_rate: Option<StrictF64>,
        learning_rate_divisor: StrictF64,
        momentum: StrictF64,
        epsilon_r: StrictF64,
        theta: StrictF64,
        intervals: StrictUsize,
        interpolation_order: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        jacobi_precondition: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_from_probabilities_with_stages(
            py,
            probs,
            stages,
            backend,
            learning_rate,
            learning_rate_divisor,
            jacobi_precondition,
            momentum,
            epsilon_r,
            theta,
            intervals,
            interpolation_order,
            min_grid_dim,
            max_grid_dim,
            seed,
            dtype,
        )
    }

    /// Construct a `TsNetSolver` from a `CsrGraph` using partial-BFS k-NN and uniform perplexity.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, k = StrictUsize(15), uniform_perplexity = StrictF64(15.0), *, backend = "fft", power = StrictF64(2.0), sigma_iters = StrictUsize(50), sigma_tolerance = StrictF64(1e-5), row_normalize = false, stage_iterations = None, exaggeration = StrictF64(4.0), lambda_c_stage1 = StrictF64(0.1), lambda_c_stage2 = StrictF64(0.01), lambda_c_stage3 = StrictF64(0.0), lambda_r_stage3 = StrictF64(0.0), learning_rate = None, learning_rate_divisor = StrictF64(3.0), momentum = StrictF64(0.8), epsilon_r = StrictF64(0.05), theta = StrictF64(0.5), intervals = StrictUsize(25), interpolation_order = StrictUsize(3), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), jacobi_precondition = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, k=15, uniform_perplexity=15.0, *, backend='fft', power=2.0, sigma_iters=50, sigma_tolerance=1e-5, row_normalize=False, stage_iterations=None, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, learning_rate_divisor=3.0, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=3, min_grid_dim=64, max_grid_dim=512, jacobi_precondition=True, seed=42, dtype='float64')"
    )]
    fn from_graph_partial_bfs(
        py: Python<'_>,
        graph: &PyCsrGraph,
        k: StrictUsize,
        uniform_perplexity: StrictF64,
        backend: &str,
        power: StrictF64,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        row_normalize: bool,
        stage_iterations: Option<&Bound<'_, PyAny>>,
        exaggeration: StrictF64,
        lambda_c_stage1: StrictF64,
        lambda_c_stage2: StrictF64,
        lambda_c_stage3: StrictF64,
        lambda_r_stage3: StrictF64,
        learning_rate: Option<StrictF64>,
        learning_rate_divisor: StrictF64,
        momentum: StrictF64,
        epsilon_r: StrictF64,
        theta: StrictF64,
        intervals: StrictUsize,
        interpolation_order: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        jacobi_precondition: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let opts = tsnet_stage_opts! {
            stage_iterations,
            exaggeration, lambda_c_stage1, lambda_c_stage2, lambda_c_stage3, lambda_r_stage3,
            learning_rate, learning_rate_divisor, jacobi_precondition, momentum, epsilon_r, theta,
            intervals, interpolation_order, min_grid_dim, max_grid_dim, seed
        };
        build_from_graph_partial_bfs(
            py,
            graph,
            k,
            uniform_perplexity,
            backend,
            power,
            sigma_iters,
            sigma_tolerance,
            row_normalize,
            opts,
            dtype,
        )
    }

    /// Construct a `TsNetSolver` from a `CsrGraph` using direct row-normalized `A1 + w2 * A2` affinity.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, w2 = StrictF64(0.5), max_2hop_per_node = None, *, degree_beta = StrictF64(0.0), mode = "two_step_random_walk", max_bridge_degree = None, backend = "fft", stage_iterations = None, exaggeration = StrictF64(4.0), lambda_c_stage1 = StrictF64(0.1), lambda_c_stage2 = StrictF64(0.01), lambda_c_stage3 = StrictF64(0.0), lambda_r_stage3 = StrictF64(0.0), learning_rate = None, learning_rate_divisor = StrictF64(3.0), momentum = StrictF64(0.8), epsilon_r = StrictF64(0.05), theta = StrictF64(0.5), intervals = StrictUsize(25), interpolation_order = StrictUsize(3), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), jacobi_precondition = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, w2=0.5, max_2hop_per_node=None, *, degree_beta=0.0, mode='two_step_random_walk', max_bridge_degree=None, backend='fft', stage_iterations=None, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, learning_rate_divisor=3.0, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=3, min_grid_dim=64, max_grid_dim=512, jacobi_precondition=True, seed=42, dtype='float64')"
    )]
    fn from_graph_row_normalized(
        py: Python<'_>,
        graph: &PyCsrGraph,
        w2: StrictF64,
        max_2hop_per_node: Option<StrictUsize>,
        degree_beta: StrictF64,
        mode: &str,
        max_bridge_degree: Option<StrictUsize>,
        backend: &str,
        stage_iterations: Option<&Bound<'_, PyAny>>,
        exaggeration: StrictF64,
        lambda_c_stage1: StrictF64,
        lambda_c_stage2: StrictF64,
        lambda_c_stage3: StrictF64,
        lambda_r_stage3: StrictF64,
        learning_rate: Option<StrictF64>,
        learning_rate_divisor: StrictF64,
        momentum: StrictF64,
        epsilon_r: StrictF64,
        theta: StrictF64,
        intervals: StrictUsize,
        interpolation_order: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        jacobi_precondition: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let opts = tsnet_stage_opts! {
            stage_iterations,
            exaggeration, lambda_c_stage1, lambda_c_stage2, lambda_c_stage3, lambda_r_stage3,
            learning_rate, learning_rate_divisor, jacobi_precondition, momentum, epsilon_r, theta,
            intervals, interpolation_order, min_grid_dim, max_grid_dim, seed
        };
        build_from_graph_row_normalized(
            py,
            graph,
            w2,
            max_2hop_per_node,
            degree_beta,
            mode,
            max_bridge_degree,
            backend,
            opts,
            dtype,
        )
    }

    /// Execute one optimization step in place on `layout` (`Layout2d | LayoutNd | ManifoldLayout2d`).
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_any_layout_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` optimization iterations in place on `layout`.
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

    /// Number of iterations executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Total scheduled iterations across all stages.
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Concrete repulsion backend resolved at construction (expanding `"auto"`).
    #[getter]
    fn resolved_backend(&self) -> &'static str {
        self.inner.resolved_backend()
    }

    /// Cumulative number of 2D FFT transforms executed across all steps.
    #[getter]
    fn total_fft_2d_calls(&self) -> u64 {
        self.inner.total_fft_2d_calls()
    }

    /// Return telemetry counters from the most recent FFT field evaluation, or `None`.
    fn last_fft_stats<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.inner.last_fft_stats_dict(py)
    }

    /// Expected number of vertices in the layout.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Active or final optimization stage index.
    #[getter]
    fn stage_index(&self) -> usize {
        self.inner.stage_index()
    }

    /// Stability-capped learning rate of the current stage.
    #[getter]
    fn current_stage_lr(&self) -> f64 {
        self.inner.current_stage_lr()
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "TsNetSolver(num_nodes={}, backend='{}', status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.resolved_backend(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
