//! Cooperative multilevel Yifan Hu (2005) layout solver bindings (`Layout2d | Layout3d | LayoutNd`).

use egraph::core::{
    CoreError, CoreResult, IterativeSolver, Layout2d, Layout3d, LayoutNd, SolverStatus,
};
use egraph::layout::force::Exact as ForceExact;
use egraph::layout::{ForceBackend, YifanHuConfig, YifanHuSolver};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::core_error_to_py;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, solver_status_str};
use crate::solver_common::format_optional_usize;

use super::force_classic::{ResolvedForceBackend, resolve_force_backend};
use super::layout_dispatch::{
    DualTargetSolver, SolverAction, SolverTargetKind, TargetLayout, execute_target_action,
};

enum YifanHuInner {
    L2F64(YifanHuSolver<Layout2d<f64>, ForceBackend>),
    L2F32(YifanHuSolver<Layout2d<f32>, ForceBackend>),
    L3F64(YifanHuSolver<Layout3d<f64>, ForceBackend>),
    L3F32(YifanHuSolver<Layout3d<f32>, ForceBackend>),
    NdF64(YifanHuSolver<LayoutNd<f64>, ForceExact>),
    NdF32(YifanHuSolver<LayoutNd<f32>, ForceExact>),
}

impl DualTargetSolver for YifanHuInner {
    fn target_kind(&self) -> SolverTargetKind {
        match self {
            Self::L2F64(_) | Self::L2F32(_) => SolverTargetKind::Layout2d,
            Self::L3F64(_) | Self::L3F32(_) => SolverTargetKind::Layout3d,
            Self::NdF64(_) | Self::NdF32(_) => SolverTargetKind::LayoutNd,
        }
    }
    fn dtype(&self) -> ScalarDtype {
        match self {
            Self::L2F64(_) | Self::L3F64(_) | Self::NdF64(_) => ScalarDtype::F64,
            Self::L2F32(_) | Self::L3F32(_) | Self::NdF32(_) => ScalarDtype::F32,
        }
    }
    fn solver_status(&self) -> SolverStatus {
        match self {
            Self::L2F64(s) => IterativeSolver::status(s),
            Self::L2F32(s) => IterativeSolver::status(s),
            Self::L3F64(s) => IterativeSolver::status(s),
            Self::L3F32(s) => IterativeSolver::status(s),
            Self::NdF64(s) => IterativeSolver::status(s),
            Self::NdF32(s) => IterativeSolver::status(s),
        }
    }
    fn solver_iteration(&self) -> usize {
        match self {
            Self::L2F64(s) => IterativeSolver::iteration(s),
            Self::L2F32(s) => IterativeSolver::iteration(s),
            Self::L3F64(s) => IterativeSolver::iteration(s),
            Self::L3F32(s) => IterativeSolver::iteration(s),
            Self::NdF64(s) => IterativeSolver::iteration(s),
            Self::NdF32(s) => IterativeSolver::iteration(s),
        }
    }
    fn solver_total_iterations(&self) -> Option<usize> {
        match self {
            Self::L2F64(s) => IterativeSolver::total_iterations(s),
            Self::L2F32(s) => IterativeSolver::total_iterations(s),
            Self::L3F64(s) => IterativeSolver::total_iterations(s),
            Self::L3F32(s) => IterativeSolver::total_iterations(s),
            Self::NdF64(s) => IterativeSolver::total_iterations(s),
            Self::NdF32(s) => IterativeSolver::total_iterations(s),
        }
    }
    fn validate_raw(&self, layout: &TargetLayout) -> CoreResult<()> {
        match (self, layout) {
            (Self::L2F64(s), TargetLayout::L2F64(l)) => IterativeSolver::validate_target(s, l),
            (Self::L2F32(s), TargetLayout::L2F32(l)) => IterativeSolver::validate_target(s, l),
            (Self::L3F64(s), TargetLayout::L3F64(l)) => IterativeSolver::validate_target(s, l),
            (Self::L3F32(s), TargetLayout::L3F32(l)) => IterativeSolver::validate_target(s, l),
            (Self::NdF64(s), TargetLayout::NdF64(l)) => IterativeSolver::validate_target(s, l),
            (Self::NdF32(s), TargetLayout::NdF32(l)) => IterativeSolver::validate_target(s, l),
            _ => Err(CoreError::invalid_input(
                "internal solver target/dtype mismatch",
            )),
        }
    }
    fn step_raw(&mut self, layout: &mut TargetLayout) -> CoreResult<SolverStatus> {
        match (self, layout) {
            (Self::L2F64(s), TargetLayout::L2F64(l)) => IterativeSolver::step(s, l),
            (Self::L2F32(s), TargetLayout::L2F32(l)) => IterativeSolver::step(s, l),
            (Self::L3F64(s), TargetLayout::L3F64(l)) => IterativeSolver::step(s, l),
            (Self::L3F32(s), TargetLayout::L3F32(l)) => IterativeSolver::step(s, l),
            (Self::NdF64(s), TargetLayout::NdF64(l)) => IterativeSolver::step(s, l),
            (Self::NdF32(s), TargetLayout::NdF32(l)) => IterativeSolver::step(s, l),
            _ => Err(CoreError::invalid_input(
                "internal solver target/dtype mismatch",
            )),
        }
    }
    fn run_steps_raw(
        &mut self,
        layout: &mut TargetLayout,
        steps: usize,
    ) -> CoreResult<SolverStatus> {
        match (self, layout) {
            (Self::L2F64(s), TargetLayout::L2F64(l)) => IterativeSolver::run_steps(s, l, steps),
            (Self::L2F32(s), TargetLayout::L2F32(l)) => IterativeSolver::run_steps(s, l, steps),
            (Self::L3F64(s), TargetLayout::L3F64(l)) => IterativeSolver::run_steps(s, l, steps),
            (Self::L3F32(s), TargetLayout::L3F32(l)) => IterativeSolver::run_steps(s, l, steps),
            (Self::NdF64(s), TargetLayout::NdF64(l)) => IterativeSolver::run_steps(s, l, steps),
            (Self::NdF32(s), TargetLayout::NdF32(l)) => IterativeSolver::run_steps(s, l, steps),
            _ => Err(CoreError::invalid_input(
                "internal solver target/dtype mismatch",
            )),
        }
    }
    fn run_raw(&mut self, layout: &mut TargetLayout) -> CoreResult<SolverStatus> {
        match (self, layout) {
            (Self::L2F64(s), TargetLayout::L2F64(l)) => IterativeSolver::run(s, l),
            (Self::L2F32(s), TargetLayout::L2F32(l)) => IterativeSolver::run(s, l),
            (Self::L3F64(s), TargetLayout::L3F64(l)) => IterativeSolver::run(s, l),
            (Self::L3F32(s), TargetLayout::L3F32(l)) => IterativeSolver::run(s, l),
            (Self::NdF64(s), TargetLayout::NdF64(l)) => IterativeSolver::run(s, l),
            (Self::NdF32(s), TargetLayout::NdF32(l)) => IterativeSolver::run(s, l),
            _ => Err(CoreError::invalid_input(
                "internal solver target/dtype mismatch",
            )),
        }
    }
}

impl YifanHuInner {
    fn config(&self) -> &YifanHuConfig {
        match self {
            Self::L2F64(s) => s.config(),
            Self::L2F32(s) => s.config(),
            Self::L3F64(s) => s.config(),
            Self::L3F32(s) => s.config(),
            Self::NdF64(s) => s.config(),
            Self::NdF32(s) => s.config(),
        }
    }

    fn theta(&self) -> f64 {
        match self {
            Self::L2F64(s) => match s.backend() {
                ForceBackend::BarnesHut(bh) => bh.theta(),
                _ => 0.0,
            },
            Self::L2F32(s) => match s.backend() {
                ForceBackend::BarnesHut(bh) => bh.theta(),
                _ => 0.0,
            },
            Self::L3F64(s) => match s.backend() {
                ForceBackend::BarnesHut(bh) => bh.theta(),
                _ => 0.0,
            },
            Self::L3F32(s) => match s.backend() {
                ForceBackend::BarnesHut(bh) => bh.theta(),
                _ => 0.0,
            },
            Self::NdF64(_) | Self::NdF32(_) => 0.0,
        }
    }
}

/// Cooperative multilevel Yifan Hu (2005) layout solver.
#[pyclass(name = "YifanHuSolver", module = "egraph._core")]
pub struct PyYifanHuSolver {
    inner: YifanHuInner,
}

#[pymethods]
impl PyYifanHuSolver {
    /// Construct a `YifanHuSolver` from a `CsrGraph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            target = SolverTargetKind::Layout2d,
            backend = "auto",
            max_iterations_per_level = StrictUsize(200),
            step_size = StrictF64(1.0),
            step_ratio = StrictF64(0.9),
            relative_strength = StrictF64(0.2),
            optimal_distance = None,
            theta = StrictF64(1.2),
            min_coarsen_size = StrictUsize(30),
            min_coarsen_ratio = StrictF64(0.75),
            convergence_tolerance = StrictF64(1e-4),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, target='layout2d', backend='auto', max_iterations_per_level=200, step_size=1.0, step_ratio=0.9, relative_strength=0.2, optimal_distance=None, theta=1.2, min_coarsen_size=30, min_coarsen_ratio=0.75, convergence_tolerance=1e-4, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        backend: &str,
        max_iterations_per_level: StrictUsize,
        step_size: StrictF64,
        step_ratio: StrictF64,
        relative_strength: StrictF64,
        optimal_distance: Option<StrictF64>,
        theta: StrictF64,
        min_coarsen_size: StrictUsize,
        min_coarsen_ratio: StrictF64,
        convergence_tolerance: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let resolved_backend = resolve_force_backend(target, backend, None, theta.0)?;
        let cfg = YifanHuConfig::new()
            .with_max_iterations_per_level(max_iterations_per_level.0)
            .with_step_size(step_size.0)
            .with_step_ratio(step_ratio.0)
            .with_relative_strength(relative_strength.0)
            .with_optimal_distance(optimal_distance.map(|v| v.0))
            .with_min_coarsen_size(min_coarsen_size.0)
            .with_min_coarsen_ratio(min_coarsen_ratio.0)
            .with_convergence_tolerance(convergence_tolerance.0)
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match (resolved_backend, dtype) {
                (ResolvedForceBackend::Euclid23(b), ScalarDtype::F64) => match target {
                    SolverTargetKind::Layout2d => {
                        YifanHuSolver::<Layout2d<f64>, ForceBackend>::with_backend(&*g, cfg, b)
                            .map(YifanHuInner::L2F64)
                    }
                    SolverTargetKind::Layout3d => {
                        YifanHuSolver::<Layout3d<f64>, ForceBackend>::with_backend(&*g, cfg, b)
                            .map(YifanHuInner::L3F64)
                    }
                    _ => unreachable!(),
                },
                (ResolvedForceBackend::Euclid23(b), ScalarDtype::F32) => match target {
                    SolverTargetKind::Layout2d => {
                        YifanHuSolver::<Layout2d<f32>, ForceBackend>::with_backend(&*g, cfg, b)
                            .map(YifanHuInner::L2F32)
                    }
                    SolverTargetKind::Layout3d => {
                        YifanHuSolver::<Layout3d<f32>, ForceBackend>::with_backend(&*g, cfg, b)
                            .map(YifanHuInner::L3F32)
                    }
                    _ => unreachable!(),
                },
                (ResolvedForceBackend::NdExact(b), ScalarDtype::F64) => {
                    YifanHuSolver::<LayoutNd<f64>, ForceExact>::with_backend(&*g, cfg, b)
                        .map(YifanHuInner::NdF64)
                }
                (ResolvedForceBackend::NdExact(b), ScalarDtype::F32) => {
                    YifanHuSolver::<LayoutNd<f32>, ForceExact>::with_backend(&*g, cfg, b)
                        .map(YifanHuInner::NdF32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_target_action(&mut self.inner, &mut None, py, layout, SolverAction::Step)
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
            &mut None,
            py,
            layout,
            SolverAction::RunSteps(steps.0),
        )
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_target_action(&mut self.inner, &mut None, py, layout, SolverAction::Run)
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
        match &self.inner {
            YifanHuInner::L2F64(s) => s.num_nodes(),
            YifanHuInner::L2F32(s) => s.num_nodes(),
            YifanHuInner::L3F64(s) => s.num_nodes(),
            YifanHuInner::L3F32(s) => s.num_nodes(),
            YifanHuInner::NdF64(s) => s.num_nodes(),
            YifanHuInner::NdF32(s) => s.num_nodes(),
        }
    }

    #[getter]
    fn num_levels(&self) -> usize {
        match &self.inner {
            YifanHuInner::L2F64(s) => s.num_levels(),
            YifanHuInner::L2F32(s) => s.num_levels(),
            YifanHuInner::L3F64(s) => s.num_levels(),
            YifanHuInner::L3F32(s) => s.num_levels(),
            YifanHuInner::NdF64(s) => s.num_levels(),
            YifanHuInner::NdF32(s) => s.num_levels(),
        }
    }

    #[getter]
    fn current_level(&self) -> usize {
        match &self.inner {
            YifanHuInner::L2F64(s) => s.current_level(),
            YifanHuInner::L2F32(s) => s.current_level(),
            YifanHuInner::L3F64(s) => s.current_level(),
            YifanHuInner::L3F32(s) => s.current_level(),
            YifanHuInner::NdF64(s) => s.current_level(),
            YifanHuInner::NdF32(s) => s.current_level(),
        }
    }

    #[getter]
    fn max_iterations_per_level(&self) -> usize {
        self.inner.config().max_iterations_per_level
    }

    #[getter]
    fn step_size(&self) -> f64 {
        self.inner.config().step_size
    }

    #[getter]
    fn step_ratio(&self) -> f64 {
        self.inner.config().step_ratio
    }

    #[getter]
    fn relative_strength(&self) -> f64 {
        self.inner.config().relative_strength
    }

    #[getter]
    fn optimal_distance(&self) -> Option<f64> {
        self.inner.config().optimal_distance
    }

    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta()
    }

    #[getter]
    fn min_coarsen_size(&self) -> usize {
        self.inner.config().min_coarsen_size
    }

    #[getter]
    fn min_coarsen_ratio(&self) -> f64 {
        self.inner.config().min_coarsen_ratio
    }

    #[getter]
    fn convergence_tolerance(&self) -> f64 {
        self.inner.config().convergence_tolerance
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.config().seed
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "YifanHuSolver(num_nodes={}, target='{}', levels={}, current_level={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.target(),
            self.num_levels(),
            self.current_level(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}
