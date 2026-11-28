//! Cooperative Noack (2007/2009) LinLog layout solver bindings (`Layout2d | Layout3d | LayoutNd`).

use egraph::core::{
    CoreError, CoreResult, IterativeSolver, Layout2d, Layout3d, LayoutNd, SolverStatus,
};
use egraph::layout::force::{BarnesHut as ForceBarnesHut, Exact as ForceExact};
use egraph::layout::{ForceBackend, LinLogConfig, LinLogRepulsion, LinLogSolver};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::{core_error_to_py, invalid_input_err, unsupported_operation_err};
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, normalize_enum_key, solver_status_str,
};
use crate::solver_common::format_optional_usize;

use super::layout_dispatch::{
    DualTargetSolver, SolverAction, SolverTargetKind, TargetLayout, execute_target_action,
    extract_target_layout,
};

pub(super) fn parse_linlog_repulsion(raw: &str) -> Result<LinLogRepulsion, String> {
    match normalize_enum_key(raw).as_str() {
        "edgerepulsion" | "edge" => Ok(LinLogRepulsion::EdgeRepulsion),
        "noderepulsion" | "node" => Ok(LinLogRepulsion::NodeRepulsion),
        _ => Err(format!(
            "unknown repulsion_model {raw:?}; expected 'edge_repulsion' or 'node_repulsion'"
        )),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ForceBackendChoice {
    Auto,
    Exact,
    BarnesHut,
}

pub(super) enum ResolvedForceBackend {
    Euclid23(ForceBackend),
    NdExact(ForceExact),
}

pub(super) fn resolve_force_backend(
    target: SolverTargetKind,
    backend: &str,
    use_barnes_hut: Option<bool>,
    theta: f64,
) -> PyResult<ResolvedForceBackend> {
    ForceBarnesHut::new(theta)
        .validate()
        .map_err(core_error_to_py)?;
    let parsed_backend = match normalize_enum_key(backend).as_str() {
        "auto" => ForceBackendChoice::Auto,
        "exact" => ForceBackendChoice::Exact,
        "barneshut" | "bh" => ForceBackendChoice::BarnesHut,
        _ => {
            return Err(unsupported_operation_err(format!(
                "unknown force backend {backend:?}; expected 'auto', 'exact', or 'barnes_hut'"
            )));
        }
    };
    let effective = match (parsed_backend, use_barnes_hut) {
        (ForceBackendChoice::Auto, None) => match target {
            SolverTargetKind::Layout2d | SolverTargetKind::Layout3d => {
                ForceBackendChoice::BarnesHut
            }
            SolverTargetKind::LayoutNd => ForceBackendChoice::Exact,
            SolverTargetKind::Manifold2d => ForceBackendChoice::Exact,
        },
        (ForceBackendChoice::Auto, Some(true)) => ForceBackendChoice::BarnesHut,
        (ForceBackendChoice::Auto, Some(false)) => ForceBackendChoice::Exact,
        (ForceBackendChoice::Exact, Some(true)) => {
            return Err(invalid_input_err(
                "conflicting options: backend='exact' cannot be combined with use_barnes_hut=True",
            ));
        }
        (ForceBackendChoice::BarnesHut, Some(false)) => {
            return Err(invalid_input_err(
                "conflicting options: backend='barnes_hut' cannot be combined with use_barnes_hut=False",
            ));
        }
        (b, _) => b,
    };
    match target {
        SolverTargetKind::Layout2d | SolverTargetKind::Layout3d => {
            Ok(ResolvedForceBackend::Euclid23(match effective {
                ForceBackendChoice::Exact => ForceBackend::exact(),
                ForceBackendChoice::BarnesHut | ForceBackendChoice::Auto => {
                    ForceBackend::barnes_hut(theta)
                }
            }))
        }
        SolverTargetKind::LayoutNd => match effective {
            ForceBackendChoice::Exact | ForceBackendChoice::Auto => {
                Ok(ResolvedForceBackend::NdExact(ForceExact::new()))
            }
            ForceBackendChoice::BarnesHut => Err(unsupported_operation_err(
                "Barnes-Hut force repulsion only supports target='layout2d' or target='layout3d'; use backend='exact' (or 'auto') for target='layout_nd'",
            )),
        },
        SolverTargetKind::Manifold2d => Err(unsupported_operation_err(
            "Euclidean force solver does not support target='manifold2d'; use FruchtermanReingoldSolver or target='layout2d'/'layout3d'/'layout_nd'",
        )),
    }
}

pub(super) enum LinLogInner {
    L2F64(LinLogSolver<Layout2d<f64>, ForceBackend>),
    L2F32(LinLogSolver<Layout2d<f32>, ForceBackend>),
    L3F64(LinLogSolver<Layout3d<f64>, ForceBackend>),
    L3F32(LinLogSolver<Layout3d<f32>, ForceBackend>),
    NdF64(LinLogSolver<LayoutNd<f64>, ForceExact>),
    NdF32(LinLogSolver<LayoutNd<f32>, ForceExact>),
}

impl DualTargetSolver for LinLogInner {
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

/// Cooperative true Noack (2007/2009) LinLog / `(a, r)`-PolyLog layout solver.
#[pyclass(name = "LinLogSolver", module = "egraph._core")]
pub struct PyLinLogSolver {
    inner: LinLogInner,
}

#[pymethods]
impl PyLinLogSolver {
    /// Construct a `LinLogSolver` from a `CsrGraph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            target = SolverTargetKind::Layout2d,
            backend = "auto",
            iterations = StrictUsize(200),
            repulsion_model = "edge_repulsion",
            attraction_exponent = StrictF64(0.0),
            repulsion_exponent = StrictF64(-1.0),
            theta = StrictF64(1.2),
            use_barnes_hut = None,
            gravity = StrictF64(0.01),
            edge_weight_influence = StrictF64(1.0),
            max_displacement = None,
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, target='layout2d', backend='auto', iterations=200, repulsion_model='edge_repulsion', attraction_exponent=0.0, repulsion_exponent=-1.0, theta=1.2, use_barnes_hut=None, gravity=0.01, edge_weight_influence=1.0, max_displacement=None, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        backend: &str,
        iterations: StrictUsize,
        repulsion_model: &str,
        attraction_exponent: StrictF64,
        repulsion_exponent: StrictF64,
        theta: StrictF64,
        use_barnes_hut: Option<bool>,
        gravity: StrictF64,
        edge_weight_influence: StrictF64,
        max_displacement: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let model = parse_linlog_repulsion(repulsion_model).map_err(invalid_input_err)?;
        let resolved_backend = resolve_force_backend(target, backend, use_barnes_hut, theta.0)?;
        let cfg = LinLogConfig::new()
            .with_iterations(iterations.0)
            .with_repulsion_model(model)
            .with_attraction_exponent(attraction_exponent.0)
            .with_repulsion_exponent(repulsion_exponent.0)
            .with_gravity(gravity.0)
            .with_edge_weight_influence(edge_weight_influence.0)
            .with_max_displacement(max_displacement.map(|v| v.0))
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match (resolved_backend, dtype) {
                (ResolvedForceBackend::Euclid23(b), ScalarDtype::F64) => match target {
                    SolverTargetKind::Layout2d => {
                        LinLogSolver::<Layout2d<f64>, ForceBackend>::with_backend(&*g, cfg, b)
                            .map(LinLogInner::L2F64)
                    }
                    SolverTargetKind::Layout3d => {
                        LinLogSolver::<Layout3d<f64>, ForceBackend>::with_backend(&*g, cfg, b)
                            .map(LinLogInner::L3F64)
                    }
                    _ => unreachable!(),
                },
                (ResolvedForceBackend::Euclid23(b), ScalarDtype::F32) => match target {
                    SolverTargetKind::Layout2d => {
                        LinLogSolver::<Layout2d<f32>, ForceBackend>::with_backend(&*g, cfg, b)
                            .map(LinLogInner::L2F32)
                    }
                    SolverTargetKind::Layout3d => {
                        LinLogSolver::<Layout3d<f32>, ForceBackend>::with_backend(&*g, cfg, b)
                            .map(LinLogInner::L3F32)
                    }
                    _ => unreachable!(),
                },
                (ResolvedForceBackend::NdExact(b), ScalarDtype::F64) => {
                    LinLogSolver::<LayoutNd<f64>, ForceExact>::with_backend(&*g, cfg, b)
                        .map(LinLogInner::NdF64)
                }
                (ResolvedForceBackend::NdExact(b), ScalarDtype::F32) => {
                    LinLogSolver::<LayoutNd<f32>, ForceExact>::with_backend(&*g, cfg, b)
                        .map(LinLogInner::NdF32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one LinLog iteration in place on `layout`.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_target_action(&mut self.inner, &mut None, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` LinLog iterations in place on `layout` while releasing the GIL.
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

    /// Run LinLog optimization to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_target_action(&mut self.inner, &mut None, py, layout, SolverAction::Run)
    }

    /// Evaluate the exact Noack (2009) `(a, r)`-LinLog energy of `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn energy(&self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<f64> {
        let extracted =
            extract_target_layout(layout, self.inner.target_kind(), self.inner.dtype())?;
        py.detach(|| match (&self.inner, &extracted) {
            (LinLogInner::L2F64(s), TargetLayout::L2F64(l)) => s.energy(l),
            (LinLogInner::L2F32(s), TargetLayout::L2F32(l)) => s.energy(l),
            (LinLogInner::L3F64(s), TargetLayout::L3F64(l)) => s.energy(l),
            (LinLogInner::L3F32(s), TargetLayout::L3F32(l)) => s.energy(l),
            (LinLogInner::NdF64(s), TargetLayout::NdF64(l)) => s.energy(l),
            (LinLogInner::NdF32(s), TargetLayout::NdF32(l)) => s.energy(l),
            _ => Err(CoreError::invalid_input(
                "internal solver target/dtype mismatch",
            )),
        })
        .map_err(core_error_to_py)
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
            LinLogInner::L2F64(s) => s.num_nodes(),
            LinLogInner::L2F32(s) => s.num_nodes(),
            LinLogInner::L3F64(s) => s.num_nodes(),
            LinLogInner::L3F32(s) => s.num_nodes(),
            LinLogInner::NdF64(s) => s.num_nodes(),
            LinLogInner::NdF32(s) => s.num_nodes(),
        }
    }

    #[getter]
    fn max_displacement(&self) -> f64 {
        match &self.inner {
            LinLogInner::L2F64(s) => s.max_displacement(),
            LinLogInner::L2F32(s) => s.max_displacement(),
            LinLogInner::L3F64(s) => s.max_displacement(),
            LinLogInner::L3F32(s) => s.max_displacement(),
            LinLogInner::NdF64(s) => s.max_displacement(),
            LinLogInner::NdF32(s) => s.max_displacement(),
        }
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "LinLogSolver(num_nodes={}, target='{}', status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.target(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}
