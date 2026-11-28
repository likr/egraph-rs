//! Cooperative Fruchterman-Reingold (1991) force-directed layout solver bindings.

use egraph::core::{
    CoreError, CoreResult, IterativeSolver, Layout2d, Layout3d, LayoutNd, ManifoldKind,
    ManifoldLayout2d, SolverStatus,
};
use egraph::layout::force::{BarnesHut as ForceBarnesHut, Exact as ForceExact};
use egraph::layout::{FrBackend, FruchtermanReingoldConfig, FruchtermanReingoldSolver};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::{core_error_to_py, unsupported_operation_err};
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, normalize_enum_key, solver_status_str,
};
use crate::solver_common::format_optional_usize;

use super::layout_dispatch::{
    DualTargetSolver, SolverAction, SolverTargetKind, TargetLayout, execute_target_action,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum FrBackendChoice {
    Auto,
    Exact,
    BarnesHut,
    Grid,
}

pub(super) fn parse_fr_backend_choice(raw: &str) -> Result<FrBackendChoice, String> {
    match normalize_enum_key(raw).as_str() {
        "auto" => Ok(FrBackendChoice::Auto),
        "exact" => Ok(FrBackendChoice::Exact),
        "barneshut" | "bh" => Ok(FrBackendChoice::BarnesHut),
        "grid" => Ok(FrBackendChoice::Grid),
        _ => Err(format!(
            "unknown Fruchterman-Reingold backend {raw:?}; expected 'auto', 'exact', 'barnes_hut', or 'grid'"
        )),
    }
}

#[cfg(test)]
pub(super) fn parse_fr_backend(raw: &str) -> Result<FrBackend, String> {
    match parse_fr_backend_choice(raw)? {
        FrBackendChoice::Auto | FrBackendChoice::Exact => Ok(FrBackend::exact()),
        FrBackendChoice::BarnesHut => Ok(FrBackend::barnes_hut(1.2)),
        FrBackendChoice::Grid => Ok(FrBackend::grid()),
    }
}

enum ResolvedFrBackend {
    Euclid23(FrBackend),
    ExactOnly(ForceExact),
}

fn validate_and_resolve_fr_backend(
    target: SolverTargetKind,
    raw_backend: &str,
    theta: f64,
) -> PyResult<ResolvedFrBackend> {
    ForceBarnesHut::new(theta)
        .validate()
        .map_err(core_error_to_py)?;
    let choice = parse_fr_backend_choice(raw_backend).map_err(unsupported_operation_err)?;
    match target {
        SolverTargetKind::Layout2d | SolverTargetKind::Layout3d => {
            let b = match choice {
                FrBackendChoice::Auto | FrBackendChoice::Exact => FrBackend::exact(),
                FrBackendChoice::BarnesHut => FrBackend::barnes_hut(theta),
                FrBackendChoice::Grid => FrBackend::grid(),
            };
            Ok(ResolvedFrBackend::Euclid23(b))
        }
        SolverTargetKind::LayoutNd => match choice {
            FrBackendChoice::Auto | FrBackendChoice::Exact => {
                Ok(ResolvedFrBackend::ExactOnly(ForceExact::new()))
            }
            FrBackendChoice::BarnesHut | FrBackendChoice::Grid => Err(unsupported_operation_err(
                "FruchtermanReingoldSolver with target='layout_nd' only supports backend='exact' (or 'auto'); use target='layout2d' or target='layout3d' for Barnes-Hut or Grid repulsion",
            )),
        },
        SolverTargetKind::Manifold2d => match choice {
            FrBackendChoice::Auto | FrBackendChoice::Exact => {
                Ok(ResolvedFrBackend::ExactOnly(ForceExact::new()))
            }
            FrBackendChoice::BarnesHut | FrBackendChoice::Grid => Err(unsupported_operation_err(
                "FruchtermanReingoldSolver with target='manifold2d' only supports backend='exact' (or 'auto') geodesic repulsion; use target='layout2d' or target='layout3d' for Barnes-Hut or Grid repulsion",
            )),
        },
    }
}

pub(super) enum FrInner {
    L2F64(FruchtermanReingoldSolver<Layout2d<f64>, FrBackend>),
    L2F32(FruchtermanReingoldSolver<Layout2d<f32>, FrBackend>),
    L3F64(FruchtermanReingoldSolver<Layout3d<f64>, FrBackend>),
    L3F32(FruchtermanReingoldSolver<Layout3d<f32>, FrBackend>),
    NdF64(FruchtermanReingoldSolver<LayoutNd<f64>, ForceExact>),
    NdF32(FruchtermanReingoldSolver<LayoutNd<f32>, ForceExact>),
    M2F64(FruchtermanReingoldSolver<ManifoldLayout2d<f64>, ForceExact>),
    M2F32(FruchtermanReingoldSolver<ManifoldLayout2d<f32>, ForceExact>),
}

impl DualTargetSolver for FrInner {
    fn target_kind(&self) -> SolverTargetKind {
        match self {
            Self::L2F64(_) | Self::L2F32(_) => SolverTargetKind::Layout2d,
            Self::L3F64(_) | Self::L3F32(_) => SolverTargetKind::Layout3d,
            Self::NdF64(_) | Self::NdF32(_) => SolverTargetKind::LayoutNd,
            Self::M2F64(_) | Self::M2F32(_) => SolverTargetKind::Manifold2d,
        }
    }
    fn dtype(&self) -> ScalarDtype {
        match self {
            Self::L2F64(_) | Self::L3F64(_) | Self::NdF64(_) | Self::M2F64(_) => ScalarDtype::F64,
            Self::L2F32(_) | Self::L3F32(_) | Self::NdF32(_) | Self::M2F32(_) => ScalarDtype::F32,
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
            Self::M2F64(s) => IterativeSolver::status(s),
            Self::M2F32(s) => IterativeSolver::status(s),
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
            Self::M2F64(s) => IterativeSolver::iteration(s),
            Self::M2F32(s) => IterativeSolver::iteration(s),
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
            Self::M2F64(s) => IterativeSolver::total_iterations(s),
            Self::M2F32(s) => IterativeSolver::total_iterations(s),
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
            (Self::M2F64(s), TargetLayout::M2F64(l)) => IterativeSolver::validate_target(s, l),
            (Self::M2F32(s), TargetLayout::M2F32(l)) => IterativeSolver::validate_target(s, l),
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
            (Self::M2F64(s), TargetLayout::M2F64(l)) => IterativeSolver::step(s, l),
            (Self::M2F32(s), TargetLayout::M2F32(l)) => IterativeSolver::step(s, l),
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
            (Self::M2F64(s), TargetLayout::M2F64(l)) => IterativeSolver::run_steps(s, l, steps),
            (Self::M2F32(s), TargetLayout::M2F32(l)) => IterativeSolver::run_steps(s, l, steps),
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
            (Self::M2F64(s), TargetLayout::M2F64(l)) => IterativeSolver::run(s, l),
            (Self::M2F32(s), TargetLayout::M2F32(l)) => IterativeSolver::run(s, l),
            _ => Err(CoreError::invalid_input(
                "internal solver target/dtype mismatch",
            )),
        }
    }
}

/// Cooperative Fruchterman-Reingold (1991) force-directed layout solver.
#[pyclass(name = "FruchtermanReingoldSolver", module = "egraph._core")]
pub struct PyFruchtermanReingoldSolver {
    inner: FrInner,
    locked_manifold: Option<ManifoldKind>,
}

#[pymethods]
impl PyFruchtermanReingoldSolver {
    /// Construct a `FruchtermanReingoldSolver` from a `CsrGraph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            target = SolverTargetKind::Layout2d,
            backend = "exact",
            iterations = StrictUsize(200),
            k = None,
            area = None,
            initial_temperature = None,
            gravity = StrictF64(0.0),
            theta = StrictF64(1.2),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, target='layout2d', backend='exact', iterations=200, k=None, area=None, initial_temperature=None, gravity=0.0, theta=1.2, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        backend: &str,
        iterations: StrictUsize,
        k: Option<StrictF64>,
        area: Option<StrictF64>,
        initial_temperature: Option<StrictF64>,
        gravity: StrictF64,
        theta: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let resolved_backend = validate_and_resolve_fr_backend(target, backend, theta.0)?;
        let cfg = FruchtermanReingoldConfig::new()
            .with_iterations(iterations.0)
            .with_k(k.map(|v| v.0))
            .with_area(area.map(|v| v.0))
            .with_initial_temperature(initial_temperature.map(|v| v.0))
            .with_gravity(gravity.0)
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match (resolved_backend, dtype) {
                (ResolvedFrBackend::Euclid23(b), ScalarDtype::F64) => match target {
                    SolverTargetKind::Layout2d => FruchtermanReingoldSolver::<
                        Layout2d<f64>,
                        FrBackend,
                    >::with_backend(&*g, cfg, b)
                    .map(FrInner::L2F64),
                    SolverTargetKind::Layout3d => FruchtermanReingoldSolver::<
                        Layout3d<f64>,
                        FrBackend,
                    >::with_backend(&*g, cfg, b)
                    .map(FrInner::L3F64),
                    _ => unreachable!(),
                },
                (ResolvedFrBackend::Euclid23(b), ScalarDtype::F32) => match target {
                    SolverTargetKind::Layout2d => FruchtermanReingoldSolver::<
                        Layout2d<f32>,
                        FrBackend,
                    >::with_backend(&*g, cfg, b)
                    .map(FrInner::L2F32),
                    SolverTargetKind::Layout3d => FruchtermanReingoldSolver::<
                        Layout3d<f32>,
                        FrBackend,
                    >::with_backend(&*g, cfg, b)
                    .map(FrInner::L3F32),
                    _ => unreachable!(),
                },
                (ResolvedFrBackend::ExactOnly(b), ScalarDtype::F64) => match target {
                    SolverTargetKind::LayoutNd => FruchtermanReingoldSolver::<
                        LayoutNd<f64>,
                        ForceExact,
                    >::with_backend(&*g, cfg, b)
                    .map(FrInner::NdF64),
                    SolverTargetKind::Manifold2d => FruchtermanReingoldSolver::<
                        ManifoldLayout2d<f64>,
                        ForceExact,
                    >::with_backend(&*g, cfg, b)
                    .map(FrInner::M2F64),
                    _ => unreachable!(),
                },
                (ResolvedFrBackend::ExactOnly(b), ScalarDtype::F32) => match target {
                    SolverTargetKind::LayoutNd => FruchtermanReingoldSolver::<
                        LayoutNd<f32>,
                        ForceExact,
                    >::with_backend(&*g, cfg, b)
                    .map(FrInner::NdF32),
                    SolverTargetKind::Manifold2d => FruchtermanReingoldSolver::<
                        ManifoldLayout2d<f32>,
                        ForceExact,
                    >::with_backend(&*g, cfg, b)
                    .map(FrInner::M2F32),
                    _ => unreachable!(),
                },
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
        match &self.inner {
            FrInner::L2F64(s) => s.num_nodes(),
            FrInner::L2F32(s) => s.num_nodes(),
            FrInner::L3F64(s) => s.num_nodes(),
            FrInner::L3F32(s) => s.num_nodes(),
            FrInner::NdF64(s) => s.num_nodes(),
            FrInner::NdF32(s) => s.num_nodes(),
            FrInner::M2F64(s) => s.num_nodes(),
            FrInner::M2F32(s) => s.num_nodes(),
        }
    }

    #[getter]
    fn optimal_distance(&self) -> f64 {
        match &self.inner {
            FrInner::L2F64(s) => s.optimal_distance(),
            FrInner::L2F32(s) => s.optimal_distance(),
            FrInner::L3F64(s) => s.optimal_distance(),
            FrInner::L3F32(s) => s.optimal_distance(),
            FrInner::NdF64(s) => s.optimal_distance(),
            FrInner::NdF32(s) => s.optimal_distance(),
            FrInner::M2F64(s) => s.optimal_distance(),
            FrInner::M2F32(s) => s.optimal_distance(),
        }
    }

    #[getter]
    fn current_temperature(&self) -> f64 {
        match &self.inner {
            FrInner::L2F64(s) => s.current_temperature(),
            FrInner::L2F32(s) => s.current_temperature(),
            FrInner::L3F64(s) => s.current_temperature(),
            FrInner::L3F32(s) => s.current_temperature(),
            FrInner::NdF64(s) => s.current_temperature(),
            FrInner::NdF32(s) => s.current_temperature(),
            FrInner::M2F64(s) => s.current_temperature(),
            FrInner::M2F32(s) => s.current_temperature(),
        }
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "FruchtermanReingoldSolver(num_nodes={}, target='{}', status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.target(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}
