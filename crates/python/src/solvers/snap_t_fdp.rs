//! Compact-memory edge-incident negative sampling `SNAP-tFDP` layout solver bindings.

use egraph::core::ManifoldKind;
use egraph::layout::{SnapTFdpConfig, SnapTFdpSolver};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::core_error_to_py;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, solver_status_str};
use crate::solver_common::format_optional_usize;

use super::layout_dispatch::{
    DualTargetSolver, SolverAction, SolverTargetKind, dispatch_riemannian_target,
    execute_target_action, impl_riemannian_target_solver, match_riemannian_target,
};

impl_riemannian_target_solver!(SnapTFdpInner, SnapTFdpSolver);

/// Cooperative compact-memory edge-incident negative sampling `SNAP-tFDP` solver.
#[pyclass(name = "SnapTFdpSolver", module = "egraph._core")]
pub struct PySnapTFdpSolver {
    inner: SnapTFdpInner,
    locked_manifold: Option<ManifoldKind>,
}

#[pymethods]
impl PySnapTFdpSolver {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            target = SolverTargetKind::Layout2d,
            iterations = StrictUsize(200),
            negatives_per_edge = StrictUsize(3),
            learning_rate = StrictF64(1.0),
            alpha = StrictF64(1.0),
            repulsion_scale = StrictF64(1.0),
            rho_start = StrictF64(1.0),
            rho_end = StrictF64(0.1),
            rho_power = StrictF64(1.0),
            degree_beta = StrictF64(0.0),
            bounded_attraction = true,
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, target='layout2d', iterations=200, negatives_per_edge=3, learning_rate=1.0, alpha=1.0, repulsion_scale=1.0, rho_start=1.0, rho_end=0.1, rho_power=1.0, degree_beta=0.0, bounded_attraction=True, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        iterations: StrictUsize,
        negatives_per_edge: StrictUsize,
        learning_rate: StrictF64,
        alpha: StrictF64,
        repulsion_scale: StrictF64,
        rho_start: StrictF64,
        rho_end: StrictF64,
        rho_power: StrictF64,
        degree_beta: StrictF64,
        bounded_attraction: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = SnapTFdpConfig::new()
            .with_iterations(iterations.0)
            .with_negatives_per_edge(negatives_per_edge.0)
            .with_learning_rate(learning_rate.0)
            .with_alpha(alpha.0)
            .with_repulsion_scale(repulsion_scale.0)
            .with_rho_start(rho_start.0)
            .with_rho_end(rho_end.0)
            .with_rho_power(rho_power.0)
            .with_degree_beta(degree_beta.0)
            .with_bounded_attraction(bounded_attraction)
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| {
                dispatch_riemannian_target!(
                    SnapTFdpInner,
                    target,
                    dtype,
                    SnapTFdpSolver::new(&*g, cfg)
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
        match_riemannian_target!(SnapTFdpInner, &self.inner, s => s.num_nodes())
    }

    #[getter]
    fn num_springs(&self) -> usize {
        match_riemannian_target!(SnapTFdpInner, &self.inner, s => s.num_springs())
    }

    #[getter]
    fn current_eta(&self) -> f64 {
        match_riemannian_target!(SnapTFdpInner, &self.inner, s => s.current_eta())
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "SnapTFdpSolver(num_nodes={}, target='{}', springs={}, status='{}', iteration={}/{}, dtype='{}')",
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
