//! Deterministic negative-sampling `LargeVis` (`Tang et al., 2016`) layout solver bindings.

use egraph::core::ManifoldKind;
use egraph::layout::{LargeVisConfig, LargeVisSolver};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::core_error_to_py;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, solver_status_str};
use crate::solver_common::format_optional_usize;

use super::layout_dispatch::{
    DualTargetSolver, SolverAction, SolverTargetKind, dispatch_riemannian_target,
    execute_target_action, impl_riemannian_target_solver, match_riemannian_target,
};

impl_riemannian_target_solver!(LargeVisInner, LargeVisSolver);

/// Cooperative Tang et al. (2016) negative-sampling `LargeVis` graph embedding solver.
#[pyclass(name = "LargeVisSolver", module = "egraph._core")]
pub struct PyLargeVisSolver {
    inner: LargeVisInner,
    locked_manifold: Option<ManifoldKind>,
}

#[pymethods]
impl PyLargeVisSolver {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            target = SolverTargetKind::Layout2d,
            iterations = StrictUsize(200),
            negative_samples = StrictUsize(5),
            gamma = StrictF64(7.0),
            initial_lr = StrictF64(1.0),
            min_lr = StrictF64(1e-4),
            degree_exponent = StrictF64(0.75),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, target='layout2d', iterations=200, negative_samples=5, gamma=7.0, initial_lr=1.0, min_lr=1e-4, degree_exponent=0.75, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        iterations: StrictUsize,
        negative_samples: StrictUsize,
        gamma: StrictF64,
        initial_lr: StrictF64,
        min_lr: StrictF64,
        degree_exponent: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = LargeVisConfig::new()
            .with_iterations(iterations.0)
            .with_negative_samples(negative_samples.0)
            .with_gamma(gamma.0)
            .with_initial_lr(initial_lr.0)
            .with_min_lr(min_lr.0)
            .with_degree_exponent(degree_exponent.0)
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| {
                dispatch_riemannian_target!(
                    LargeVisInner,
                    target,
                    dtype,
                    LargeVisSolver::new(&*g, cfg)
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
        match_riemannian_target!(LargeVisInner, &self.inner, s => s.num_nodes())
    }

    #[getter]
    fn num_springs(&self) -> usize {
        match_riemannian_target!(LargeVisInner, &self.inner, s => s.num_springs())
    }

    #[getter]
    fn current_lr(&self) -> f64 {
        match_riemannian_target!(LargeVisInner, &self.inner, s => s.current_lr())
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "LargeVisSolver(num_nodes={}, target='{}', springs={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.target(),
            self.num_springs(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}
