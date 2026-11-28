//! Doubly-stochastic mini-batch diagonal/tangent-Hessian preconditioned stress SGD (`Sgd2Solver`) bindings.

use egraph::core::{CoreError, ManifoldKind};
use egraph::layout::{Sgd2Config, Sgd2Solver};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::{core_error_to_py, unsupported_operation_err};
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, solver_status_str};
use crate::manifold::manifold_kind_str;
use crate::solver_common::format_optional_usize;

use super::layout_dispatch::{
    DualTargetSolver, SolverAction, SolverTargetKind, TargetLayout, dispatch_riemannian_target,
    execute_target_action, extract_target_layout, impl_riemannian_target_solver,
    match_riemannian_target, write_back_target_layout,
};
use super::sgd_core::build_schedule_config;

impl_riemannian_target_solver!(Sgd2Inner, Sgd2Solver);

/// Doubly-stochastic mini-batch tangent-Hessian preconditioned stress SGD solver
/// (distinct from Ahmed et al. 2021 multi-criteria `(SGD)^2`).
#[pyclass(name = "Sgd2Solver", module = "egraph._core")]
pub struct PySgd2Solver {
    inner: Sgd2Inner,
    locked_manifold: Option<ManifoldKind>,
}

#[pymethods]
impl PySgd2Solver {
    /// Construct a doubly-stochastic mini-batch diagonal/tangent-Hessian preconditioned stress SGD solver
    /// (distinct from Ahmed et al. 2021 multi-criteria `(SGD)^2`).
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            target = SolverTargetKind::Layout2d,
            iterations = StrictUsize(60),
            eps = StrictF64(0.01),
            step_scale = StrictF64(1.0),
            batch_size = StrictUsize(4096),
            damping = StrictF64(1e-4),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, target='layout2d', iterations=60, eps=0.01, step_scale=1.0, batch_size=4096, damping=1e-4, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        iterations: StrictUsize,
        eps: StrictF64,
        step_scale: StrictF64,
        batch_size: StrictUsize,
        damping: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = Sgd2Config::new()
            .with_iterations(iterations.0)
            .with_eps(eps.0)
            .with_step_scale(step_scale.0)
            .with_batch_size(batch_size.0)
            .with_damping(damping.0)
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| {
                dispatch_riemannian_target!(Sgd2Inner, target, dtype, Sgd2Solver::new(&*g, cfg))
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

    #[pyo3(
        signature = (*, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "squared", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42)),
        text_signature = "($self, *, iterations=60, eps=0.01, schedule='squared', step_scale=1.0, max_mu=None, seed=42)"
    )]
    fn reset_schedule(
        &mut self,
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
    ) -> PyResult<()> {
        let sched = build_schedule_config(
            iterations.0,
            eps.0,
            schedule,
            step_scale.0,
            max_mu.map(|v| v.0),
        )?;
        self.locked_manifold = None;
        match_riemannian_target!(Sgd2Inner, &mut self.inner, s => s.reset_schedule(sched, seed.0))
            .map_err(core_error_to_py)
    }

    #[pyo3(signature = (layout, start_iter, count), text_signature = "($self, layout, start_iter, count)")]
    fn run_schedule_window(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyAny>,
        start_iter: StrictUsize,
        count: StrictUsize,
    ) -> PyResult<&'static str> {
        let mut extracted =
            extract_target_layout(layout, self.inner.target_kind(), self.inner.dtype())?;
        if let (Some(prev), Some(curr)) = (self.locked_manifold, extracted.manifold_kind())
            && prev != curr
        {
            return Err(unsupported_operation_err(format!(
                "manifold solver is locked to '{}', got '{}'",
                manifold_kind_str(prev),
                manifold_kind_str(curr)
            )));
        }
        let kind_opt = extracted.manifold_kind();
        let status = py
            .detach(|| match (&mut self.inner, &mut extracted) {
                (Sgd2Inner::L2F64(s), TargetLayout::L2F64(l)) => {
                    s.run_schedule_window(l, start_iter.0, count.0)
                }
                (Sgd2Inner::L2F32(s), TargetLayout::L2F32(l)) => {
                    s.run_schedule_window(l, start_iter.0, count.0)
                }
                (Sgd2Inner::L3F64(s), TargetLayout::L3F64(l)) => {
                    s.run_schedule_window(l, start_iter.0, count.0)
                }
                (Sgd2Inner::L3F32(s), TargetLayout::L3F32(l)) => {
                    s.run_schedule_window(l, start_iter.0, count.0)
                }
                (Sgd2Inner::NdF64(s), TargetLayout::NdF64(l)) => {
                    s.run_schedule_window(l, start_iter.0, count.0)
                }
                (Sgd2Inner::NdF32(s), TargetLayout::NdF32(l)) => {
                    s.run_schedule_window(l, start_iter.0, count.0)
                }
                (Sgd2Inner::M2F64(s), TargetLayout::M2F64(l)) => {
                    s.run_schedule_window(l, start_iter.0, count.0)
                }
                (Sgd2Inner::M2F32(s), TargetLayout::M2F32(l)) => {
                    s.run_schedule_window(l, start_iter.0, count.0)
                }
                _ => Err(CoreError::invalid_input(
                    "internal solver target/dtype mismatch",
                )),
            })
            .map_err(core_error_to_py)?;
        if self.locked_manifold.is_none()
            && let Some(curr) = kind_opt
        {
            self.locked_manifold = Some(curr);
        }
        write_back_target_layout(layout, extracted)?;
        Ok(solver_status_str(status))
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
        match_riemannian_target!(Sgd2Inner, &self.inner, s => s.num_nodes())
    }

    #[getter]
    fn num_springs(&self) -> usize {
        match_riemannian_target!(Sgd2Inner, &self.inner, s => s.num_springs())
    }

    #[getter]
    fn current_eta(&self) -> f64 {
        match_riemannian_target!(Sgd2Inner, &self.inner, s => s.current_eta())
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "Sgd2Solver(num_nodes={}, target='{}', springs={}, status='{}', iteration={}/{}, dtype='{}')",
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
