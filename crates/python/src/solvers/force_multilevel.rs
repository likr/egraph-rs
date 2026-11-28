//! Cooperative ForceAtlas2 layout solver bindings (`Layout2d | Layout3d | LayoutNd`).

use egraph::core::{
    CoreError, CoreResult, IterativeSolver, Layout2d, Layout3d, LayoutNd, SolverStatus,
};
use egraph::layout::force::Exact as ForceExact;
use egraph::layout::{ForceAtlas2Config, ForceAtlas2Solver, ForceBackend};
use numpy::PyArray1;
use pyo3::prelude::*;

use crate::array_extract::extract_f64_1d;
use crate::csr_graph::PyCsrGraph;
use crate::errors::core_error_to_py;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, solver_status_str};
use crate::solver_common::format_optional_usize;

use super::force_classic::{ResolvedForceBackend, resolve_force_backend};
use super::layout_dispatch::{
    DualTargetSolver, SolverAction, SolverTargetKind, TargetLayout, execute_target_action,
};

enum ForceAtlas2Inner {
    L2F64(ForceAtlas2Solver<Layout2d<f64>, ForceBackend>),
    L2F32(ForceAtlas2Solver<Layout2d<f32>, ForceBackend>),
    L3F64(ForceAtlas2Solver<Layout3d<f64>, ForceBackend>),
    L3F32(ForceAtlas2Solver<Layout3d<f32>, ForceBackend>),
    NdF64(ForceAtlas2Solver<LayoutNd<f64>, ForceExact>),
    NdF32(ForceAtlas2Solver<LayoutNd<f32>, ForceExact>),
}

impl DualTargetSolver for ForceAtlas2Inner {
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

impl ForceAtlas2Inner {
    fn config(&self) -> &ForceAtlas2Config {
        match self {
            Self::L2F64(s) => s.config(),
            Self::L2F32(s) => s.config(),
            Self::L3F64(s) => s.config(),
            Self::L3F32(s) => s.config(),
            Self::NdF64(s) => s.config(),
            Self::NdF32(s) => s.config(),
        }
    }

    fn backend_theta(&self) -> f64 {
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

    fn uses_barnes_hut(&self) -> bool {
        match self {
            Self::L2F64(s) => matches!(s.backend(), ForceBackend::BarnesHut(_)),
            Self::L2F32(s) => matches!(s.backend(), ForceBackend::BarnesHut(_)),
            Self::L3F64(s) => matches!(s.backend(), ForceBackend::BarnesHut(_)),
            Self::L3F32(s) => matches!(s.backend(), ForceBackend::BarnesHut(_)),
            Self::NdF64(_) | Self::NdF32(_) => false,
        }
    }
}

/// Cooperative ForceAtlas2 force-directed layout solver.
#[pyclass(name = "ForceAtlas2Solver", module = "egraph._core")]
pub struct PyForceAtlas2Solver {
    inner: ForceAtlas2Inner,
}

#[pymethods]
impl PyForceAtlas2Solver {
    /// Construct a `ForceAtlas2Solver` from a `CsrGraph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            target = SolverTargetKind::Layout2d,
            backend = "auto",
            iterations = StrictUsize(200),
            theta = StrictF64(1.2),
            use_barnes_hut = None,
            linlog_mode = false,
            strong_gravity = false,
            gravity = StrictF64(1.0),
            scaling_ratio = StrictF64(2.0),
            edge_weight_influence = StrictF64(1.0),
            outbound_attraction_distribution = false,
            jitter_tolerance = StrictF64(1.0),
            max_displacement = None,
            prevent_overlap = false,
            default_node_radius = StrictF64(1.0),
            node_radii = None,
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, target='layout2d', backend='auto', iterations=200, theta=1.2, use_barnes_hut=None, linlog_mode=False, strong_gravity=False, gravity=1.0, scaling_ratio=2.0, edge_weight_influence=1.0, outbound_attraction_distribution=False, jitter_tolerance=1.0, max_displacement=None, prevent_overlap=False, default_node_radius=1.0, node_radii=None, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        backend: &str,
        iterations: StrictUsize,
        theta: StrictF64,
        use_barnes_hut: Option<bool>,
        linlog_mode: bool,
        strong_gravity: bool,
        gravity: StrictF64,
        scaling_ratio: StrictF64,
        edge_weight_influence: StrictF64,
        outbound_attraction_distribution: bool,
        jitter_tolerance: StrictF64,
        max_displacement: Option<StrictF64>,
        prevent_overlap: bool,
        default_node_radius: StrictF64,
        node_radii: Option<&Bound<'_, PyAny>>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let resolved_backend = resolve_force_backend(target, backend, use_barnes_hut, theta.0)?;
        let radii_vec = match node_radii {
            Some(obj) if !obj.is_none() => Some(extract_f64_1d(obj, "node_radii")?),
            _ => None,
        };
        let cfg = ForceAtlas2Config::new()
            .with_iterations(iterations.0)
            .with_linlog_mode(linlog_mode)
            .with_strong_gravity(strong_gravity)
            .with_gravity(gravity.0)
            .with_scaling_ratio(scaling_ratio.0)
            .with_edge_weight_influence(edge_weight_influence.0)
            .with_outbound_attraction_distribution(outbound_attraction_distribution)
            .with_jitter_tolerance(jitter_tolerance.0)
            .with_max_displacement(max_displacement.map(|v| v.0))
            .with_prevent_overlap(prevent_overlap)
            .with_default_node_radius(default_node_radius.0)
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match (resolved_backend, dtype, radii_vec.as_deref()) {
                (ResolvedForceBackend::Euclid23(b), ScalarDtype::F64, Some(r)) => {
                    match target {
                        SolverTargetKind::Layout2d => ForceAtlas2Solver::<
                            Layout2d<f64>,
                            ForceBackend,
                        >::with_node_radii_and_backend(
                            &*g, cfg, r, b
                        )
                        .map(ForceAtlas2Inner::L2F64),
                        SolverTargetKind::Layout3d => ForceAtlas2Solver::<
                            Layout3d<f64>,
                            ForceBackend,
                        >::with_node_radii_and_backend(
                            &*g, cfg, r, b
                        )
                        .map(ForceAtlas2Inner::L3F64),
                        _ => unreachable!(),
                    }
                }
                (ResolvedForceBackend::Euclid23(b), ScalarDtype::F32, Some(r)) => {
                    match target {
                        SolverTargetKind::Layout2d => ForceAtlas2Solver::<
                            Layout2d<f32>,
                            ForceBackend,
                        >::with_node_radii_and_backend(
                            &*g, cfg, r, b
                        )
                        .map(ForceAtlas2Inner::L2F32),
                        SolverTargetKind::Layout3d => ForceAtlas2Solver::<
                            Layout3d<f32>,
                            ForceBackend,
                        >::with_node_radii_and_backend(
                            &*g, cfg, r, b
                        )
                        .map(ForceAtlas2Inner::L3F32),
                        _ => unreachable!(),
                    }
                }
                (ResolvedForceBackend::NdExact(b), ScalarDtype::F64, Some(r)) => {
                    ForceAtlas2Solver::<LayoutNd<f64>, ForceExact>::with_node_radii_and_backend(
                        &*g, cfg, r, b,
                    )
                    .map(ForceAtlas2Inner::NdF64)
                }
                (ResolvedForceBackend::NdExact(b), ScalarDtype::F32, Some(r)) => {
                    ForceAtlas2Solver::<LayoutNd<f32>, ForceExact>::with_node_radii_and_backend(
                        &*g, cfg, r, b,
                    )
                    .map(ForceAtlas2Inner::NdF32)
                }
                (ResolvedForceBackend::Euclid23(b), ScalarDtype::F64, None) => match target {
                    SolverTargetKind::Layout2d => {
                        ForceAtlas2Solver::<Layout2d<f64>, ForceBackend>::with_backend(&*g, cfg, b)
                            .map(ForceAtlas2Inner::L2F64)
                    }
                    SolverTargetKind::Layout3d => {
                        ForceAtlas2Solver::<Layout3d<f64>, ForceBackend>::with_backend(&*g, cfg, b)
                            .map(ForceAtlas2Inner::L3F64)
                    }
                    _ => unreachable!(),
                },
                (ResolvedForceBackend::Euclid23(b), ScalarDtype::F32, None) => match target {
                    SolverTargetKind::Layout2d => {
                        ForceAtlas2Solver::<Layout2d<f32>, ForceBackend>::with_backend(&*g, cfg, b)
                            .map(ForceAtlas2Inner::L2F32)
                    }
                    SolverTargetKind::Layout3d => {
                        ForceAtlas2Solver::<Layout3d<f32>, ForceBackend>::with_backend(&*g, cfg, b)
                            .map(ForceAtlas2Inner::L3F32)
                    }
                    _ => unreachable!(),
                },
                (ResolvedForceBackend::NdExact(b), ScalarDtype::F64, None) => {
                    ForceAtlas2Solver::<LayoutNd<f64>, ForceExact>::with_backend(&*g, cfg, b)
                        .map(ForceAtlas2Inner::NdF64)
                }
                (ResolvedForceBackend::NdExact(b), ScalarDtype::F32, None) => {
                    ForceAtlas2Solver::<LayoutNd<f32>, ForceExact>::with_backend(&*g, cfg, b)
                        .map(ForceAtlas2Inner::NdF32)
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
            ForceAtlas2Inner::L2F64(s) => s.num_nodes(),
            ForceAtlas2Inner::L2F32(s) => s.num_nodes(),
            ForceAtlas2Inner::L3F64(s) => s.num_nodes(),
            ForceAtlas2Inner::L3F32(s) => s.num_nodes(),
            ForceAtlas2Inner::NdF64(s) => s.num_nodes(),
            ForceAtlas2Inner::NdF32(s) => s.num_nodes(),
        }
    }

    #[getter]
    fn prevent_overlap(&self) -> bool {
        self.inner.config().prevent_overlap
    }

    #[getter]
    fn default_node_radius(&self) -> f64 {
        self.inner.config().default_node_radius
    }

    #[getter]
    fn node_radii<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        match &self.inner {
            ForceAtlas2Inner::L2F64(s) => PyArray1::from_slice(py, s.node_radii()),
            ForceAtlas2Inner::L2F32(s) => PyArray1::from_slice(py, s.node_radii()),
            ForceAtlas2Inner::L3F64(s) => PyArray1::from_slice(py, s.node_radii()),
            ForceAtlas2Inner::L3F32(s) => PyArray1::from_slice(py, s.node_radii()),
            ForceAtlas2Inner::NdF64(s) => PyArray1::from_slice(py, s.node_radii()),
            ForceAtlas2Inner::NdF32(s) => PyArray1::from_slice(py, s.node_radii()),
        }
    }

    #[getter]
    fn theta(&self) -> f64 {
        self.inner.backend_theta()
    }

    #[getter]
    fn use_barnes_hut(&self) -> bool {
        self.inner.uses_barnes_hut()
    }

    #[getter]
    fn linlog_mode(&self) -> bool {
        self.inner.config().linlog_mode
    }

    #[getter]
    fn strong_gravity(&self) -> bool {
        self.inner.config().strong_gravity
    }

    #[getter]
    fn gravity(&self) -> f64 {
        self.inner.config().gravity
    }

    #[getter]
    fn scaling_ratio(&self) -> f64 {
        self.inner.config().scaling_ratio
    }

    #[getter]
    fn edge_weight_influence(&self) -> f64 {
        self.inner.config().edge_weight_influence
    }

    #[getter]
    fn outbound_attraction_distribution(&self) -> bool {
        self.inner.config().outbound_attraction_distribution
    }

    #[getter]
    fn jitter_tolerance(&self) -> f64 {
        self.inner.config().jitter_tolerance
    }

    #[getter]
    fn max_displacement(&self) -> Option<f64> {
        self.inner.config().max_displacement
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
            "ForceAtlas2Solver(num_nodes={}, target='{}', status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.target(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}
