//! Thin Python binding facade for `TsNetSolver`.

use egraph::core::ManifoldKind;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, solver_status_str};
use crate::neighborhood::{PyKnnGraph, PySparseProbabilities};
use crate::solver_common::SolverAction;

use super::fit_field_stats::PyFitFieldStats;
use super::layout_dispatch::{DualTargetSolver, execute_target_action, validate_target_space};
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

/// Cooperative unified tsNET solver supporting Policy 1A targets (`"layout2d"`, `"layout3d"`,
/// `"layout_nd"`, `"manifold2d"`) and Option 1 repulsion backends.
#[pyclass(name = "TsNetSolver", module = "egraph._core")]
pub struct PyTsNetSolver {
    inner: TsNetInner,
    locked_manifold: Option<ManifoldKind>,
}

impl PyTsNetSolver {
    pub(super) fn from_inner(inner: TsNetInner) -> Self {
        Self {
            inner,
            locked_manifold: None,
        }
    }
}

#[pymethods]
impl PyTsNetSolver {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, backend = "auto", *, target = "layout2d", perplexity_c = StrictF64(0.9), min_perplexity = StrictF64(5.0), max_perplexity = StrictF64(1200.0), power = StrictF64(2.0), sigma_iters = StrictUsize(50), sigma_tolerance = StrictF64(1e-5), row_normalize = false, stage_iterations = None, exaggeration = StrictF64(4.0), lambda_c_stage1 = StrictF64(0.1), lambda_c_stage2 = StrictF64(0.01), lambda_c_stage3 = StrictF64(0.0), lambda_r_stage3 = StrictF64(0.0), learning_rate = None, learning_rate_divisor = StrictF64(3.0), momentum = StrictF64(0.8), epsilon_r = StrictF64(0.05), theta = StrictF64(0.5), intervals = StrictUsize(25), interpolation_order = StrictUsize(3), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), jacobi_precondition = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, backend='auto', *, target='layout2d', perplexity_c=0.9, min_perplexity=5.0, max_perplexity=1200.0, power=2.0, sigma_iters=50, sigma_tolerance=1e-5, row_normalize=False, stage_iterations=None, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, learning_rate_divisor=3.0, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=3, min_grid_dim=64, max_grid_dim=512, jacobi_precondition=True, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        backend: &str,
        target: &str,
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
        let opts = tsnet_stage_opts!(
            stage_iterations,
            exaggeration,
            lambda_c_stage1,
            lambda_c_stage2,
            lambda_c_stage3,
            lambda_r_stage3,
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
            seed
        );
        build_pure_2hop(
            py,
            graph,
            target,
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

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, backend = "auto", *, target = "layout2d", perplexity_c = StrictF64(0.9), min_perplexity = StrictF64(5.0), max_perplexity = StrictF64(1200.0), power = StrictF64(2.0), sigma_iters = StrictUsize(50), sigma_tolerance = StrictF64(1e-5), row_normalize = false, stage_iterations = None, exaggeration = StrictF64(4.0), lambda_c_stage1 = StrictF64(0.1), lambda_c_stage2 = StrictF64(0.01), lambda_c_stage3 = StrictF64(0.0), lambda_r_stage3 = StrictF64(0.0), learning_rate = None, learning_rate_divisor = StrictF64(3.0), momentum = StrictF64(0.8), epsilon_r = StrictF64(0.05), theta = StrictF64(0.5), intervals = StrictUsize(25), interpolation_order = StrictUsize(3), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), jacobi_precondition = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, backend='auto', *, target='layout2d', perplexity_c=0.9, min_perplexity=5.0, max_perplexity=1200.0, power=2.0, sigma_iters=50, sigma_tolerance=1e-5, row_normalize=False, stage_iterations=None, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, learning_rate_divisor=3.0, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=3, min_grid_dim=64, max_grid_dim=512, jacobi_precondition=True, seed=42, dtype='float64')"
    )]
    fn pure_2hop(
        py: Python<'_>,
        graph: &PyCsrGraph,
        backend: &str,
        target: &str,
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
        let opts = tsnet_stage_opts!(
            stage_iterations,
            exaggeration,
            lambda_c_stage1,
            lambda_c_stage2,
            lambda_c_stage3,
            lambda_r_stage3,
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
            seed
        );
        build_pure_2hop(
            py,
            graph,
            target,
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

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (knn, perplexities = None, uniform_perplexity = StrictF64(15.0), *, target = "layout2d", backend = "auto", power = StrictF64(2.0), sigma_iters = StrictUsize(50), sigma_tolerance = StrictF64(1e-5), row_normalize = false, stage_iterations = None, exaggeration = StrictF64(4.0), lambda_c_stage1 = StrictF64(0.1), lambda_c_stage2 = StrictF64(0.01), lambda_c_stage3 = StrictF64(0.0), lambda_r_stage3 = StrictF64(0.0), learning_rate = None, learning_rate_divisor = StrictF64(3.0), momentum = StrictF64(0.8), epsilon_r = StrictF64(0.05), theta = StrictF64(0.5), intervals = StrictUsize(25), interpolation_order = StrictUsize(3), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), jacobi_precondition = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(knn, perplexities=None, uniform_perplexity=15.0, *, target='layout2d', backend='auto', power=2.0, sigma_iters=50, sigma_tolerance=1e-5, row_normalize=False, stage_iterations=None, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, learning_rate_divisor=3.0, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=3, min_grid_dim=64, max_grid_dim=512, jacobi_precondition=True, seed=42, dtype='float64')"
    )]
    fn from_knn(
        py: Python<'_>,
        knn: &PyKnnGraph,
        perplexities: Option<&Bound<'_, PyAny>>,
        uniform_perplexity: StrictF64,
        target: &str,
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
        let opts = tsnet_stage_opts!(
            stage_iterations,
            exaggeration,
            lambda_c_stage1,
            lambda_c_stage2,
            lambda_c_stage3,
            lambda_r_stage3,
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
            seed
        );
        build_from_knn(
            py,
            knn,
            perplexities,
            uniform_perplexity,
            target,
            backend,
            power,
            sigma_iters,
            sigma_tolerance,
            row_normalize,
            opts,
            dtype,
        )
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (probs, backend = "auto", *, target = "layout2d", stage_iterations = None, exaggeration = StrictF64(4.0), lambda_c_stage1 = StrictF64(0.1), lambda_c_stage2 = StrictF64(0.01), lambda_c_stage3 = StrictF64(0.0), lambda_r_stage3 = StrictF64(0.0), learning_rate = None, learning_rate_divisor = StrictF64(3.0), momentum = StrictF64(0.8), epsilon_r = StrictF64(0.05), theta = StrictF64(0.5), intervals = StrictUsize(25), interpolation_order = StrictUsize(3), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), jacobi_precondition = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(probs, backend='auto', *, target='layout2d', stage_iterations=None, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, learning_rate_divisor=3.0, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=3, min_grid_dim=64, max_grid_dim=512, jacobi_precondition=True, seed=42, dtype='float64')"
    )]
    fn from_probabilities(
        py: Python<'_>,
        probs: &PySparseProbabilities,
        backend: &str,
        target: &str,
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
        let opts = tsnet_stage_opts!(
            stage_iterations,
            exaggeration,
            lambda_c_stage1,
            lambda_c_stage2,
            lambda_c_stage3,
            lambda_r_stage3,
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
            seed
        );
        build_from_probabilities(py, probs, target, backend, opts, dtype)
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (probs, stages, *, target = "layout2d", backend = "auto", learning_rate = None, learning_rate_divisor = StrictF64(3.0), jacobi_precondition = true, momentum = StrictF64(0.8), epsilon_r = StrictF64(0.05), theta = StrictF64(0.5), intervals = StrictUsize(25), interpolation_order = StrictUsize(3), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(probs, stages, *, target='layout2d', backend='auto', learning_rate=None, learning_rate_divisor=3.0, jacobi_precondition=True, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=3, min_grid_dim=64, max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_probabilities_with_stages(
        py: Python<'_>,
        probs: &PySparseProbabilities,
        stages: &Bound<'_, PyAny>,
        target: &str,
        backend: &str,
        learning_rate: Option<StrictF64>,
        learning_rate_divisor: StrictF64,
        jacobi_precondition: bool,
        momentum: StrictF64,
        epsilon_r: StrictF64,
        theta: StrictF64,
        intervals: StrictUsize,
        interpolation_order: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_from_probabilities_with_stages(
            py,
            probs,
            stages,
            target,
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

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, k = StrictUsize(15), uniform_perplexity = StrictF64(15.0), target = "layout2d", backend = "auto", power = StrictF64(2.0), sigma_iters = StrictUsize(50), sigma_tolerance = StrictF64(1e-5), row_normalize = false, stage_iterations = None, exaggeration = StrictF64(4.0), lambda_c_stage1 = StrictF64(0.1), lambda_c_stage2 = StrictF64(0.01), lambda_c_stage3 = StrictF64(0.0), lambda_r_stage3 = StrictF64(0.0), learning_rate = None, learning_rate_divisor = StrictF64(3.0), momentum = StrictF64(0.8), epsilon_r = StrictF64(0.05), theta = StrictF64(0.5), intervals = StrictUsize(25), interpolation_order = StrictUsize(3), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), jacobi_precondition = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, k=15, uniform_perplexity=15.0, target='layout2d', backend='auto', power=2.0, sigma_iters=50, sigma_tolerance=1e-5, row_normalize=False, stage_iterations=None, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, learning_rate_divisor=3.0, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=3, min_grid_dim=64, max_grid_dim=512, jacobi_precondition=True, seed=42, dtype='float64')"
    )]
    fn from_graph_partial_bfs(
        py: Python<'_>,
        graph: &PyCsrGraph,
        k: StrictUsize,
        uniform_perplexity: StrictF64,
        target: &str,
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
        let opts = tsnet_stage_opts!(
            stage_iterations,
            exaggeration,
            lambda_c_stage1,
            lambda_c_stage2,
            lambda_c_stage3,
            lambda_r_stage3,
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
            seed
        );
        build_from_graph_partial_bfs(
            py,
            graph,
            k,
            uniform_perplexity,
            target,
            backend,
            power,
            sigma_iters,
            sigma_tolerance,
            row_normalize,
            opts,
            dtype,
        )
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, w2 = StrictF64(0.35), max_2hop_per_node = Some(StrictUsize(32)), degree_beta = StrictF64(0.0), mode = "degree_ordered_hub_suppressed", max_bridge_degree = None, target = "layout2d", backend = "auto", stage_iterations = None, exaggeration = StrictF64(4.0), lambda_c_stage1 = StrictF64(0.1), lambda_c_stage2 = StrictF64(0.01), lambda_c_stage3 = StrictF64(0.0), lambda_r_stage3 = StrictF64(0.0), learning_rate = None, learning_rate_divisor = StrictF64(3.0), momentum = StrictF64(0.8), epsilon_r = StrictF64(0.05), theta = StrictF64(0.5), intervals = StrictUsize(25), interpolation_order = StrictUsize(3), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), jacobi_precondition = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, w2=0.35, max_2hop_per_node=32, degree_beta=0.0, mode='degree_ordered_hub_suppressed', max_bridge_degree=None, target='layout2d', backend='auto', stage_iterations=None, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, learning_rate_divisor=3.0, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=3, min_grid_dim=64, max_grid_dim=512, jacobi_precondition=True, seed=42, dtype='float64')"
    )]
    fn from_graph_row_normalized(
        py: Python<'_>,
        graph: &PyCsrGraph,
        w2: StrictF64,
        max_2hop_per_node: Option<StrictUsize>,
        degree_beta: StrictF64,
        mode: &str,
        max_bridge_degree: Option<StrictUsize>,
        target: &str,
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
        let opts = tsnet_stage_opts!(
            stage_iterations,
            exaggeration,
            lambda_c_stage1,
            lambda_c_stage2,
            lambda_c_stage3,
            lambda_r_stage3,
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
            seed
        );
        build_from_graph_row_normalized(
            py,
            graph,
            w2,
            max_2hop_per_node,
            degree_beta,
            mode,
            max_bridge_degree,
            target,
            backend,
            opts,
            dtype,
        )
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn validate_target(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<()> {
        validate_target_space(&self.inner, &mut self.locked_manifold, py, layout)
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
    fn resolved_backend(&self) -> &'static str {
        self.inner.resolved_backend()
    }
    #[getter]
    fn total_fft_2d_calls(&self) -> u64 {
        self.inner.total_fft_2d_calls()
    }
    fn last_fft_stats(&self) -> Option<PyFitFieldStats> {
        self.inner.last_fft_stats()
    }
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }
    #[getter]
    fn stage_index(&self) -> usize {
        self.inner.stage_index()
    }
    #[getter]
    fn current_stage_lr(&self) -> f64 {
        self.inner.current_stage_lr()
    }
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "TsNetSolver(num_nodes={}, target='{}', backend='{}', status='{}', iteration={}/{:?}, dtype='{}')",
            self.num_nodes(),
            self.target(),
            self.resolved_backend(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
