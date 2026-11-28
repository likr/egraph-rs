//! Multilevel and entropy-based stress solver (`MaxentStressSolver`) bindings.

use egraph::core::{
    CoreError, CoreResult, IterativeSolver, Layout2d, Layout3d, LayoutNd, SolverStatus,
};
use egraph::layout::stress::{BarnesHut as MaxentBarnesHut, Exact as MaxentExact};
use egraph::layout::{MaxentStressConfig, MaxentStressSolver};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::{core_error_to_py, invalid_input_err, unsupported_operation_err};
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, normalize_enum_key, solver_status_str,
};
use crate::solver_common::format_optional_usize;

use super::layout_dispatch::{
    DualTargetSolver, SolverAction, SolverTargetKind, TargetLayout, execute_target_action,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum MaxentBackendChoice {
    Auto,
    Exact,
    BarnesHut,
}

fn resolve_maxent_backend(
    target: SolverTargetKind,
    backend: &str,
    use_barnes_hut: Option<bool>,
    theta: f64,
) -> PyResult<MaxentBackendChoice> {
    MaxentBarnesHut::new(theta)
        .validate()
        .map_err(core_error_to_py)?;
    let parsed = match normalize_enum_key(backend).as_str() {
        "auto" => MaxentBackendChoice::Auto,
        "exact" => MaxentBackendChoice::Exact,
        "barneshut" | "bh" => MaxentBackendChoice::BarnesHut,
        _ => {
            return Err(unsupported_operation_err(format!(
                "unknown MaxentStressSolver backend {backend:?}; expected 'auto', 'exact', or 'barnes_hut'"
            )));
        }
    };
    let effective = match (parsed, use_barnes_hut) {
        (MaxentBackendChoice::Auto, None) => match target {
            SolverTargetKind::Layout2d | SolverTargetKind::Layout3d => {
                MaxentBackendChoice::BarnesHut
            }
            SolverTargetKind::LayoutNd => MaxentBackendChoice::Exact,
            SolverTargetKind::Manifold2d => MaxentBackendChoice::Exact,
        },
        (MaxentBackendChoice::Auto, Some(true)) => MaxentBackendChoice::BarnesHut,
        (MaxentBackendChoice::Auto, Some(false)) => MaxentBackendChoice::Exact,
        (MaxentBackendChoice::Exact, Some(true)) => {
            return Err(invalid_input_err(
                "conflicting options: backend='exact' cannot be combined with use_barnes_hut=True",
            ));
        }
        (MaxentBackendChoice::BarnesHut, Some(false)) => {
            return Err(invalid_input_err(
                "conflicting options: backend='barnes_hut' cannot be combined with use_barnes_hut=False",
            ));
        }
        (b, _) => b,
    };
    match (target, effective) {
        (SolverTargetKind::Manifold2d, _) => Err(unsupported_operation_err(
            "MaxentStressSolver supports target='layout2d', 'layout3d', or 'layout_nd' (not 'manifold2d')",
        )),
        (SolverTargetKind::LayoutNd, MaxentBackendChoice::BarnesHut) => {
            Err(unsupported_operation_err(
                "MaxentStressSolver Barnes-Hut repulsion only supports target='layout2d' or target='layout3d'; use backend='exact' (or 'auto') for target='layout_nd'",
            ))
        }
        (_, choice) => Ok(choice),
    }
}

enum MaxentStressInner {
    L2BhF64(MaxentStressSolver<Layout2d<f64>, MaxentBarnesHut>),
    L2BhF32(MaxentStressSolver<Layout2d<f32>, MaxentBarnesHut>),
    L2ExF64(MaxentStressSolver<Layout2d<f64>, MaxentExact>),
    L2ExF32(MaxentStressSolver<Layout2d<f32>, MaxentExact>),
    L3BhF64(MaxentStressSolver<Layout3d<f64>, MaxentBarnesHut>),
    L3BhF32(MaxentStressSolver<Layout3d<f32>, MaxentBarnesHut>),
    L3ExF64(MaxentStressSolver<Layout3d<f64>, MaxentExact>),
    L3ExF32(MaxentStressSolver<Layout3d<f32>, MaxentExact>),
    NdExF64(MaxentStressSolver<LayoutNd<f64>, MaxentExact>),
    NdExF32(MaxentStressSolver<LayoutNd<f32>, MaxentExact>),
}

macro_rules! match_maxent {
    ($self:expr, $s:ident => $body:expr) => {
        match $self {
            MaxentStressInner::L2BhF64($s) => $body,
            MaxentStressInner::L2BhF32($s) => $body,
            MaxentStressInner::L2ExF64($s) => $body,
            MaxentStressInner::L2ExF32($s) => $body,
            MaxentStressInner::L3BhF64($s) => $body,
            MaxentStressInner::L3BhF32($s) => $body,
            MaxentStressInner::L3ExF64($s) => $body,
            MaxentStressInner::L3ExF32($s) => $body,
            MaxentStressInner::NdExF64($s) => $body,
            MaxentStressInner::NdExF32($s) => $body,
        }
    };
}

macro_rules! match_maxent_target {
    ($self:expr, $layout:expr, ($s:ident, $l:ident) => $body:expr) => {
        match ($self, $layout) {
            (MaxentStressInner::L2BhF64($s), TargetLayout::L2F64($l)) => $body,
            (MaxentStressInner::L2BhF32($s), TargetLayout::L2F32($l)) => $body,
            (MaxentStressInner::L2ExF64($s), TargetLayout::L2F64($l)) => $body,
            (MaxentStressInner::L2ExF32($s), TargetLayout::L2F32($l)) => $body,
            (MaxentStressInner::L3BhF64($s), TargetLayout::L3F64($l)) => $body,
            (MaxentStressInner::L3BhF32($s), TargetLayout::L3F32($l)) => $body,
            (MaxentStressInner::L3ExF64($s), TargetLayout::L3F64($l)) => $body,
            (MaxentStressInner::L3ExF32($s), TargetLayout::L3F32($l)) => $body,
            (MaxentStressInner::NdExF64($s), TargetLayout::NdF64($l)) => $body,
            (MaxentStressInner::NdExF32($s), TargetLayout::NdF32($l)) => $body,
            _ => Err(CoreError::invalid_input(
                "internal solver target/dtype mismatch",
            )),
        }
    };
}

impl DualTargetSolver for MaxentStressInner {
    fn target_kind(&self) -> SolverTargetKind {
        match self {
            Self::L2BhF64(_) | Self::L2BhF32(_) | Self::L2ExF64(_) | Self::L2ExF32(_) => {
                SolverTargetKind::Layout2d
            }
            Self::L3BhF64(_) | Self::L3BhF32(_) | Self::L3ExF64(_) | Self::L3ExF32(_) => {
                SolverTargetKind::Layout3d
            }
            Self::NdExF64(_) | Self::NdExF32(_) => SolverTargetKind::LayoutNd,
        }
    }
    fn dtype(&self) -> ScalarDtype {
        match self {
            Self::L2BhF64(_)
            | Self::L2ExF64(_)
            | Self::L3BhF64(_)
            | Self::L3ExF64(_)
            | Self::NdExF64(_) => ScalarDtype::F64,
            _ => ScalarDtype::F32,
        }
    }
    fn solver_status(&self) -> SolverStatus {
        match_maxent!(self, s => IterativeSolver::status(s))
    }
    fn solver_iteration(&self) -> usize {
        match_maxent!(self, s => IterativeSolver::iteration(s))
    }
    fn solver_total_iterations(&self) -> Option<usize> {
        match_maxent!(self, s => IterativeSolver::total_iterations(s))
    }
    fn validate_raw(&self, layout: &TargetLayout) -> CoreResult<()> {
        match_maxent_target!(self, layout, (s, l) => IterativeSolver::validate_target(s, l))
    }
    fn step_raw(&mut self, layout: &mut TargetLayout) -> CoreResult<SolverStatus> {
        match_maxent_target!(self, layout, (s, l) => IterativeSolver::step(s, l))
    }
    fn run_steps_raw(
        &mut self,
        layout: &mut TargetLayout,
        steps: usize,
    ) -> CoreResult<SolverStatus> {
        match_maxent_target!(self, layout, (s, l) => IterativeSolver::run_steps(s, l, steps))
    }
    fn run_raw(&mut self, layout: &mut TargetLayout) -> CoreResult<SolverStatus> {
        match_maxent_target!(self, layout, (s, l) => IterativeSolver::run(s, l))
    }
}

/// Cooperative Gansner et al. (2013) Maxent-Stress layout solver (`Layout2d | Layout3d | LayoutNd`).
#[pyclass(name = "MaxentStressSolver", module = "egraph._core")]
pub struct PyMaxentStressSolver {
    inner: MaxentStressInner,
}

#[pymethods]
impl PyMaxentStressSolver {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            target = SolverTargetKind::Layout2d,
            backend = "auto",
            q = None,
            alpha_init = StrictF64(1.0),
            alpha_decay = StrictF64(0.3),
            alpha_min = StrictF64(0.008),
            stages = StrictUsize(5),
            steps_per_stage = StrictUsize(50),
            tol = StrictF64(1e-3),
            cg_tol = StrictF64(0.1),
            cg_max_iters = StrictUsize(10),
            theta = StrictF64(1.2),
            use_barnes_hut = None,
            eps_sq = StrictF64(1e-6),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, target='layout2d', backend='auto', q=None, alpha_init=1.0, alpha_decay=0.3, alpha_min=0.008, stages=5, steps_per_stage=50, tol=1e-3, cg_tol=0.1, cg_max_iters=10, theta=1.2, use_barnes_hut=None, eps_sq=1e-6, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        backend: &str,
        q: Option<StrictF64>,
        alpha_init: StrictF64,
        alpha_decay: StrictF64,
        alpha_min: StrictF64,
        stages: StrictUsize,
        steps_per_stage: StrictUsize,
        tol: StrictF64,
        cg_tol: StrictF64,
        cg_max_iters: StrictUsize,
        theta: StrictF64,
        use_barnes_hut: Option<bool>,
        eps_sq: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let choice = resolve_maxent_backend(target, backend, use_barnes_hut, theta.0)?;
        let cfg = MaxentStressConfig::new()
            .with_q(q.map(|v| v.0))
            .with_alpha_init(alpha_init.0)
            .with_alpha_decay(alpha_decay.0)
            .with_alpha_min(alpha_min.0)
            .with_stages(stages.0)
            .with_steps_per_stage(steps_per_stage.0)
            .with_tol(tol.0)
            .with_cg_tol(cg_tol.0)
            .with_cg_max_iters(cg_max_iters.0)
            .with_eps_sq(eps_sq.0)
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match (target, choice, dtype) {
                (SolverTargetKind::Layout2d, MaxentBackendChoice::BarnesHut, ScalarDtype::F64) => {
                    MaxentStressSolver::<Layout2d<f64>, MaxentBarnesHut>::new_with_backend(
                        &*g,
                        cfg,
                        MaxentBarnesHut::new(theta.0),
                    )
                    .map(MaxentStressInner::L2BhF64)
                }
                (SolverTargetKind::Layout2d, MaxentBackendChoice::BarnesHut, ScalarDtype::F32) => {
                    MaxentStressSolver::<Layout2d<f32>, MaxentBarnesHut>::new_with_backend(
                        &*g,
                        cfg,
                        MaxentBarnesHut::new(theta.0),
                    )
                    .map(MaxentStressInner::L2BhF32)
                }
                (SolverTargetKind::Layout2d, MaxentBackendChoice::Exact, ScalarDtype::F64) => {
                    MaxentStressSolver::<Layout2d<f64>, MaxentExact>::new_with_backend(
                        &*g,
                        cfg,
                        MaxentExact::new(),
                    )
                    .map(MaxentStressInner::L2ExF64)
                }
                (SolverTargetKind::Layout2d, MaxentBackendChoice::Exact, ScalarDtype::F32) => {
                    MaxentStressSolver::<Layout2d<f32>, MaxentExact>::new_with_backend(
                        &*g,
                        cfg,
                        MaxentExact::new(),
                    )
                    .map(MaxentStressInner::L2ExF32)
                }
                (SolverTargetKind::Layout3d, MaxentBackendChoice::BarnesHut, ScalarDtype::F64) => {
                    MaxentStressSolver::<Layout3d<f64>, MaxentBarnesHut>::new_with_backend(
                        &*g,
                        cfg,
                        MaxentBarnesHut::new(theta.0),
                    )
                    .map(MaxentStressInner::L3BhF64)
                }
                (SolverTargetKind::Layout3d, MaxentBackendChoice::BarnesHut, ScalarDtype::F32) => {
                    MaxentStressSolver::<Layout3d<f32>, MaxentBarnesHut>::new_with_backend(
                        &*g,
                        cfg,
                        MaxentBarnesHut::new(theta.0),
                    )
                    .map(MaxentStressInner::L3BhF32)
                }
                (SolverTargetKind::Layout3d, MaxentBackendChoice::Exact, ScalarDtype::F64) => {
                    MaxentStressSolver::<Layout3d<f64>, MaxentExact>::new_with_backend(
                        &*g,
                        cfg,
                        MaxentExact::new(),
                    )
                    .map(MaxentStressInner::L3ExF64)
                }
                (SolverTargetKind::Layout3d, MaxentBackendChoice::Exact, ScalarDtype::F32) => {
                    MaxentStressSolver::<Layout3d<f32>, MaxentExact>::new_with_backend(
                        &*g,
                        cfg,
                        MaxentExact::new(),
                    )
                    .map(MaxentStressInner::L3ExF32)
                }
                (SolverTargetKind::LayoutNd, _, ScalarDtype::F64) => MaxentStressSolver::<
                    LayoutNd<f64>,
                    MaxentExact,
                >::new_with_backend(
                    &*g, cfg, MaxentExact::new()
                )
                .map(MaxentStressInner::NdExF64),
                (SolverTargetKind::LayoutNd, _, ScalarDtype::F32) => MaxentStressSolver::<
                    LayoutNd<f32>,
                    MaxentExact,
                >::new_with_backend(
                    &*g, cfg, MaxentExact::new()
                )
                .map(MaxentStressInner::NdExF32),
                _ => unreachable!(),
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
    fn current_alpha(&self) -> f64 {
        match_maxent!(&self.inner, s => s.current_alpha())
    }

    #[getter]
    fn effective_q(&self) -> f64 {
        match_maxent!(&self.inner, s => s.effective_q())
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
        match_maxent!(&self.inner, s => s.num_nodes())
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "MaxentStressSolver(num_nodes={}, target='{}', q={:.3}, alpha={:.4}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.target(),
            self.effective_q(),
            self.current_alpha(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}
