//! Thin Python binding facade for `IbUmapSolver` (`Layout2d` only).

use egraph::core::IterativeSolver;
use egraph::layout::IbUmapSolver;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::core_error_to_py;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, solver_status_str};
use crate::layout_buffers::PyLayout2d;
use crate::neighborhood::{PyDogVaradhanLandmarks, PyKnnGraph, PySparseProbabilities};
use crate::solver_common::{
    DualIterativeSolver, SolverAction, execute_solver_action, extract_solver_layout,
    impl_dual_solver,
};

use super::fit_field_stats::PyFieldFftStats;
use super::ib_umap_builders::{
    build_ib_umap_from_fuzzy_set, build_ib_umap_from_knn, build_ib_umap_from_landmarks,
    build_ib_umap_new,
};

impl_dual_solver!(IbUmapInner, IbUmapSolver);

/// Cooperative interpolation-based batch Graph UMAP (`ibUMAP`) layout solver (`Layout2d`).
#[pyclass(name = "IbUmapSolver", module = "egraph._core")]
pub struct PyIbUmapSolver {
    inner: IbUmapInner,
}

#[pymethods]
impl PyIbUmapSolver {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, target = "layout2d", max_2hop_per_node = StrictUsize(30), degree = None, num_landmarks = StrictUsize(64), min_degree = StrictUsize(2), jl_dim = StrictUsize(64), degree_normalize = true, p_exp = StrictF64(0.5), eps_psd = StrictF64(1e-4), t_coarse = StrictF64(48.0), t1 = None, t2 = None, alpha = None, beta_floor = None, backend = "ib_fft", iterations = StrictUsize(200), learning_rate = StrictF64(1.0), min_dist = StrictF64(0.1), spread = StrictF64(1.0), a = None, b = None, c_k = StrictF64(4.0), tau_scale = StrictF64(4.0), hub_damping = true, theta = StrictF64(0.5), interpolation_order = StrictUsize(1), intervals = StrictUsize(25), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, target='layout2d', max_2hop_per_node=30, degree=None, num_landmarks=64, min_degree=2, jl_dim=64, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0, t1=None, t2=None, alpha=None, beta_floor=None, backend='ib_fft', iterations=200, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, c_k=4.0, tau_scale=4.0, hub_damping=True, theta=0.5, interpolation_order=1, intervals=25, min_grid_dim=64, max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: &str,
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
        iterations: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        c_k: StrictF64,
        tau_scale: StrictF64,
        hub_damping: bool,
        theta: StrictF64,
        interpolation_order: StrictUsize,
        intervals: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: build_ib_umap_new(
                py,
                graph,
                target,
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
                iterations,
                learning_rate,
                min_dist,
                spread,
                a,
                b,
                c_k,
                tau_scale,
                hub_damping,
                theta,
                interpolation_order,
                intervals,
                min_grid_dim,
                max_grid_dim,
                seed,
                dtype,
            )?,
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, landmarks, max_2hop_per_node = StrictUsize(32), *, target = "layout2d", backend = "ib_fft", iterations = StrictUsize(200), learning_rate = StrictF64(1.0), min_dist = StrictF64(0.1), spread = StrictF64(1.0), a = None, b = None, c_k = StrictF64(4.0), tau_scale = StrictF64(4.0), hub_damping = true, theta = StrictF64(0.5), interpolation_order = StrictUsize(1), intervals = StrictUsize(25), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, landmarks, max_2hop_per_node=32, *, target='layout2d', backend='ib_fft', iterations=200, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, c_k=4.0, tau_scale=4.0, hub_damping=True, theta=0.5, interpolation_order=1, intervals=25, min_grid_dim=64, max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_landmarks(
        py: Python<'_>,
        graph: &PyCsrGraph,
        landmarks: &PyDogVaradhanLandmarks,
        max_2hop_per_node: StrictUsize,
        target: &str,
        backend: &str,
        iterations: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        c_k: StrictF64,
        tau_scale: StrictF64,
        hub_damping: bool,
        theta: StrictF64,
        interpolation_order: StrictUsize,
        intervals: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: build_ib_umap_from_landmarks(
                py,
                graph,
                landmarks,
                max_2hop_per_node,
                target,
                backend,
                iterations,
                learning_rate,
                min_dist,
                spread,
                a,
                b,
                c_k,
                tau_scale,
                hub_damping,
                theta,
                interpolation_order,
                intervals,
                min_grid_dim,
                max_grid_dim,
                seed,
                dtype,
            )?,
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (knn, *, target = "layout2d", sigma_iters = StrictUsize(64), sigma_tolerance = StrictF64(1e-5), backend = "ib_fft", iterations = StrictUsize(200), learning_rate = StrictF64(1.0), min_dist = StrictF64(0.1), spread = StrictF64(1.0), a = None, b = None, c_k = StrictF64(4.0), tau_scale = StrictF64(4.0), hub_damping = true, theta = StrictF64(0.5), interpolation_order = StrictUsize(1), intervals = StrictUsize(25), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(knn, *, target='layout2d', sigma_iters=64, sigma_tolerance=1e-5, backend='ib_fft', iterations=200, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, c_k=4.0, tau_scale=4.0, hub_damping=True, theta=0.5, interpolation_order=1, intervals=25, min_grid_dim=64, max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_knn(
        py: Python<'_>,
        knn: &PyKnnGraph,
        target: &str,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        backend: &str,
        iterations: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        c_k: StrictF64,
        tau_scale: StrictF64,
        hub_damping: bool,
        theta: StrictF64,
        interpolation_order: StrictUsize,
        intervals: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: build_ib_umap_from_knn(
                py,
                knn,
                sigma_iters,
                sigma_tolerance,
                target,
                backend,
                iterations,
                learning_rate,
                min_dist,
                spread,
                a,
                b,
                c_k,
                tau_scale,
                hub_damping,
                theta,
                interpolation_order,
                intervals,
                min_grid_dim,
                max_grid_dim,
                seed,
                dtype,
            )?,
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (probs, *, target = "layout2d", degrees = None, backend = "ib_fft", iterations = StrictUsize(200), learning_rate = StrictF64(1.0), min_dist = StrictF64(0.1), spread = StrictF64(1.0), a = None, b = None, c_k = StrictF64(4.0), tau_scale = StrictF64(4.0), hub_damping = true, theta = StrictF64(0.5), interpolation_order = StrictUsize(1), intervals = StrictUsize(25), min_grid_dim = StrictUsize(64), max_grid_dim = StrictUsize(512), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(probs, *, target='layout2d', degrees=None, backend='ib_fft', iterations=200, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, c_k=4.0, tau_scale=4.0, hub_damping=True, theta=0.5, interpolation_order=1, intervals=25, min_grid_dim=64, max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_fuzzy_set(
        py: Python<'_>,
        probs: &PySparseProbabilities,
        target: &str,
        degrees: Option<&Bound<'_, PyAny>>,
        backend: &str,
        iterations: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        c_k: StrictF64,
        tau_scale: StrictF64,
        hub_damping: bool,
        theta: StrictF64,
        interpolation_order: StrictUsize,
        intervals: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: build_ib_umap_from_fuzzy_set(
                py,
                probs,
                degrees,
                target,
                backend,
                iterations,
                learning_rate,
                min_dist,
                spread,
                a,
                b,
                c_k,
                tau_scale,
                hub_damping,
                theta,
                interpolation_order,
                intervals,
                min_grid_dim,
                max_grid_dim,
                seed,
                dtype,
            )?,
        })
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn validate_target(&self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<()> {
        let soa = extract_solver_layout(layout, self.inner.dtype())?;
        py.detach(|| match (&self.inner, &soa) {
            (IbUmapInner::F64(s), crate::solver_common::SolverLayout::F64(l)) => {
                s.validate_target(l)
            }
            (IbUmapInner::F32(s), crate::solver_common::SolverLayout::F32(l)) => {
                s.validate_target(l)
            }
            _ => unreachable!(),
        })
        .map_err(core_error_to_py)
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    fn last_fft_stats(&self) -> Option<PyFieldFftStats> {
        match &self.inner {
            IbUmapInner::F64(s) => s.last_fft_stats().map(PyFieldFftStats::from_core),
            IbUmapInner::F32(s) => s.last_fft_stats().map(PyFieldFftStats::from_core),
        }
    }

    #[getter]
    fn target(&self) -> &'static str {
        "layout2d"
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
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            IbUmapInner::F64(s) => s.num_nodes(),
            IbUmapInner::F32(s) => s.num_nodes(),
        }
    }
    #[getter]
    fn current_rho(&self) -> f64 {
        1.0
    }
    #[getter]
    fn num_edges(&self) -> usize {
        match &self.inner {
            IbUmapInner::F64(s) => s.num_edges(),
            IbUmapInner::F32(s) => s.num_edges(),
        }
    }
    #[getter]
    fn total_fft_2d_calls(&self) -> u64 {
        match &self.inner {
            IbUmapInner::F64(s) => s.total_fft_2d_calls(),
            IbUmapInner::F32(s) => s.total_fft_2d_calls(),
        }
    }
    #[getter]
    fn a(&self) -> f64 {
        match &self.inner {
            IbUmapInner::F64(s) => s.a(),
            IbUmapInner::F32(s) => s.a(),
        }
    }
    #[getter]
    fn b(&self) -> f64 {
        match &self.inner {
            IbUmapInner::F64(s) => s.b(),
            IbUmapInner::F32(s) => s.b(),
        }
    }
    #[getter]
    fn ab(&self) -> (f64, f64) {
        match &self.inner {
            IbUmapInner::F64(s) => s.ab(),
            IbUmapInner::F32(s) => s.ab(),
        }
    }
    fn __repr__(&self) -> String {
        format!(
            "IbUmapSolver(num_nodes={}, status='{}', iteration={}/{:?}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
