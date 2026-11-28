//! Cooperative Kamada-Kawai (1989) spring layout solver bindings.

use egraph::core::ManifoldKind;
use egraph::layout::{KamadaKawaiConfig, KamadaKawaiSolver};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::core_error_to_py;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictUsize, solver_status_str};
use crate::solver_common::format_optional_usize;

use super::distance_extract::extract_distance_matrix_2d;
use super::layout_dispatch::{
    DualTargetSolver, SolverAction, SolverTargetKind, dispatch_riemannian_target,
    execute_target_action, impl_riemannian_target_solver, match_riemannian_target,
};

impl_riemannian_target_solver!(KamadaKawaiInner, KamadaKawaiSolver);

/// Cooperative Kamada-Kawai (1989) spring layout solver with Hessian singularity fallback.
#[pyclass(name = "KamadaKawaiSolver", module = "egraph._core")]
pub struct PyKamadaKawaiSolver {
    inner: KamadaKawaiInner,
    locked_manifold: Option<ManifoldKind>,
}

#[pymethods]
impl PyKamadaKawaiSolver {
    #[new]
    #[pyo3(
        signature = (graph, *, target = SolverTargetKind::Layout2d, eps = StrictF64(1e-2), max_inner_iters = StrictUsize(50), max_iterations = None, dtype = ScalarDtype::F64),
        text_signature = "(graph, *, target='layout2d', eps=1e-2, max_inner_iters=50, max_iterations=None, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        eps: StrictF64,
        max_inner_iters: StrictUsize,
        max_iterations: Option<StrictUsize>,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        Self::from_graph(
            py,
            graph,
            target,
            eps,
            max_inner_iters,
            max_iterations,
            dtype,
        )
    }

    #[staticmethod]
    #[pyo3(
        signature = (graph, *, target = SolverTargetKind::Layout2d, eps = StrictF64(1e-2), max_inner_iters = StrictUsize(50), max_iterations = None, dtype = ScalarDtype::F64),
        text_signature = "(graph, *, target='layout2d', eps=1e-2, max_inner_iters=50, max_iterations=None, dtype='float64')"
    )]
    fn from_graph(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        eps: StrictF64,
        max_inner_iters: StrictUsize,
        max_iterations: Option<StrictUsize>,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = KamadaKawaiConfig::new()
            .with_eps(eps.0)
            .with_max_inner_iters(max_inner_iters.0)
            .with_max_iterations(max_iterations.map(|v| v.0));
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| {
                dispatch_riemannian_target!(
                    KamadaKawaiInner,
                    target,
                    dtype,
                    KamadaKawaiSolver::from_graph(&*g, cfg)
                )
            })
            .map_err(core_error_to_py)?;
        Ok(Self {
            inner,
            locked_manifold: None,
        })
    }

    #[staticmethod]
    #[pyo3(
        signature = (distances, *, target = SolverTargetKind::Layout2d, eps = StrictF64(1e-2), max_inner_iters = StrictUsize(50), max_iterations = None, dtype = ScalarDtype::F64),
        text_signature = "(distances, *, target='layout2d', eps=1e-2, max_inner_iters=50, max_iterations=None, dtype='float64')"
    )]
    fn from_distance_matrix(
        py: Python<'_>,
        distances: &Bound<'_, PyAny>,
        target: SolverTargetKind,
        eps: StrictF64,
        max_inner_iters: StrictUsize,
        max_iterations: Option<StrictUsize>,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = KamadaKawaiConfig::new()
            .with_eps(eps.0)
            .with_max_inner_iters(max_inner_iters.0)
            .with_max_iterations(max_iterations.map(|v| v.0));
        cfg.validate().map_err(core_error_to_py)?;
        let (dists_flat, n) = extract_distance_matrix_2d(distances)?;
        let inner = py
            .detach(|| {
                dispatch_riemannian_target!(
                    KamadaKawaiInner,
                    target,
                    dtype,
                    KamadaKawaiSolver::from_distance_matrix(n, &dists_flat, cfg)
                )
            })
            .map_err(core_error_to_py)?;
        Ok(Self {
            inner,
            locked_manifold: None,
        })
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
    fn num_nodes(&self) -> usize {
        match_riemannian_target!(KamadaKawaiInner, &self.inner, s => s.num_nodes())
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "KamadaKawaiSolver(num_nodes={}, target='{}', status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.target(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}
