//! 3-tier `PaCMAP-Graph` (`NB` / `MN` / `FP`) layout solver bindings.

use egraph::core::ManifoldKind;
use egraph::layout::{PacmapGraphConfig, PacmapGraphSolver};
use pyo3::prelude::*;

use crate::array_extract::extract_f64_2d;
use crate::csr_graph::PyCsrGraph;
use crate::errors::{core_error_to_py, dimension_mismatch_err};
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, solver_status_str};
use crate::neighborhood::PyDogVaradhanLandmarks;
use crate::solver_common::format_optional_usize;

use super::layout_dispatch::{
    DualTargetSolver, SolverAction, SolverTargetKind, dispatch_riemannian_target,
    execute_target_action, impl_riemannian_target_solver, match_riemannian_target,
};

impl_riemannian_target_solver!(PacmapGraphInner, PacmapGraphSolver);

/// Cooperative 3-tier `PaCMAP-Graph` layout solver (`NB` 1+2-hop, `MN` mid-near, `FP` further).
#[pyclass(name = "PacmapGraphSolver", module = "egraph._core")]
pub struct PyPacmapGraphSolver {
    inner: PacmapGraphInner,
    locked_manifold: Option<ManifoldKind>,
}

impl PyPacmapGraphSolver {
    pub(super) fn from_inner(inner: PacmapGraphInner) -> Self {
        Self {
            inner,
            locked_manifold: None,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_pacmap_config(
    n_neighbors: StrictUsize,
    mn_ratio: StrictF64,
    fp_ratio: StrictF64,
    phase_iterations: (StrictUsize, StrictUsize, StrictUsize),
    learning_rate: StrictF64,
    keep_all_1hop: bool,
    max_1hop_per_node: Option<StrictUsize>,
    degree_weight_beta: StrictF64,
    num_landmarks: StrictUsize,
    seed: StrictU64,
) -> PyResult<PacmapGraphConfig> {
    let cfg = PacmapGraphConfig::new()
        .with_n_neighbors(n_neighbors.0)
        .with_mn_ratio(mn_ratio.0)
        .with_fp_ratio(fp_ratio.0)
        .with_phase_iterations([
            phase_iterations.0.0,
            phase_iterations.1.0,
            phase_iterations.2.0,
        ])
        .with_learning_rate(learning_rate.0)
        .with_keep_all_1hop(keep_all_1hop)
        .with_max_1hop_per_node(max_1hop_per_node.map(|v| v.0))
        .with_degree_weight_beta(degree_weight_beta.0)
        .with_num_landmarks(num_landmarks.0)
        .with_seed(seed.0);
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

#[pymethods]
impl PyPacmapGraphSolver {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            target = SolverTargetKind::Layout2d,
            n_neighbors = StrictUsize(10),
            mn_ratio = StrictF64(0.5),
            fp_ratio = StrictF64(2.0),
            phase_iterations = (StrictUsize(100), StrictUsize(100), StrictUsize(250)),
            learning_rate = StrictF64(1.0),
            keep_all_1hop = false,
            max_1hop_per_node = None,
            degree_weight_beta = StrictF64(0.0),
            num_landmarks = StrictUsize(16),
            landmarks = None,
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, target='layout2d', n_neighbors=10, mn_ratio=0.5, fp_ratio=2.0, phase_iterations=(100, 100, 250), learning_rate=1.0, keep_all_1hop=False, max_1hop_per_node=None, degree_weight_beta=0.0, num_landmarks=16, landmarks=None, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        n_neighbors: StrictUsize,
        mn_ratio: StrictF64,
        fp_ratio: StrictF64,
        phase_iterations: (StrictUsize, StrictUsize, StrictUsize),
        learning_rate: StrictF64,
        keep_all_1hop: bool,
        max_1hop_per_node: Option<StrictUsize>,
        degree_weight_beta: StrictF64,
        num_landmarks: StrictUsize,
        landmarks: Option<&Bound<'_, PyAny>>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_pacmap_config(
            n_neighbors,
            mn_ratio,
            fp_ratio,
            phase_iterations,
            learning_rate,
            keep_all_1hop,
            max_1hop_per_node,
            degree_weight_beta,
            num_landmarks,
            seed,
        )?;
        let g = graph.inner();
        let inner = match landmarks {
            Some(obj) if !obj.is_none() => {
                if let Ok(lm_ref) = obj.extract::<PyRef<'_, PyDogVaradhanLandmarks>>() {
                    let lm = lm_ref.inner();
                    py.detach(|| {
                        dispatch_riemannian_target!(
                            PacmapGraphInner,
                            target,
                            dtype,
                            PacmapGraphSolver::with_landmarks(&*g, lm, cfg)
                        )
                    })
                    .map_err(core_error_to_py)?
                } else {
                    let (flat, nrows, ncols) = extract_f64_2d(obj, "landmarks")?;
                    if nrows != g.num_nodes() {
                        return Err(dimension_mismatch_err(format!(
                            "landmarks row count ({nrows}) must match graph.num_nodes ({})",
                            g.num_nodes()
                        )));
                    }
                    py.detach(|| {
                        dispatch_riemannian_target!(
                            PacmapGraphInner,
                            target,
                            dtype,
                            PacmapGraphSolver::with_landmark_distances(&*g, &flat, ncols, cfg)
                        )
                    })
                    .map_err(core_error_to_py)?
                }
            }
            _ => py
                .detach(|| {
                    dispatch_riemannian_target!(
                        PacmapGraphInner,
                        target,
                        dtype,
                        PacmapGraphSolver::new(&*g, cfg)
                    )
                })
                .map_err(core_error_to_py)?,
        };
        Ok(Self::from_inner(inner))
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            landmarks,
            *,
            target = SolverTargetKind::Layout2d,
            n_neighbors = StrictUsize(10),
            mn_ratio = StrictF64(0.5),
            fp_ratio = StrictF64(2.0),
            phase_iterations = (StrictUsize(100), StrictUsize(100), StrictUsize(250)),
            learning_rate = StrictF64(1.0),
            keep_all_1hop = false,
            max_1hop_per_node = None,
            degree_weight_beta = StrictF64(0.0),
            num_landmarks = StrictUsize(16),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, landmarks, *, target='layout2d', n_neighbors=10, mn_ratio=0.5, fp_ratio=2.0, phase_iterations=(100, 100, 250), learning_rate=1.0, keep_all_1hop=False, max_1hop_per_node=None, degree_weight_beta=0.0, num_landmarks=16, seed=42, dtype='float64')"
    )]
    fn with_landmarks(
        py: Python<'_>,
        graph: &PyCsrGraph,
        landmarks: &Bound<'_, PyAny>,
        target: SolverTargetKind,
        n_neighbors: StrictUsize,
        mn_ratio: StrictF64,
        fp_ratio: StrictF64,
        phase_iterations: (StrictUsize, StrictUsize, StrictUsize),
        learning_rate: StrictF64,
        keep_all_1hop: bool,
        max_1hop_per_node: Option<StrictUsize>,
        degree_weight_beta: StrictF64,
        num_landmarks: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        Self::new(
            py,
            graph,
            target,
            n_neighbors,
            mn_ratio,
            fp_ratio,
            phase_iterations,
            learning_rate,
            keep_all_1hop,
            max_1hop_per_node,
            degree_weight_beta,
            num_landmarks,
            Some(landmarks),
            seed,
            dtype,
        )
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
        match_riemannian_target!(PacmapGraphInner, &self.inner, s => s.num_nodes())
    }

    #[getter]
    fn num_springs(&self) -> usize {
        match_riemannian_target!(PacmapGraphInner, &self.inner, s => s.num_springs())
    }

    #[getter]
    fn num_hinges(&self) -> usize {
        match_riemannian_target!(PacmapGraphInner, &self.inner, s => s.num_hinges())
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "PacmapGraphSolver(num_nodes={}, target='{}', springs={}, hinges={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.target(),
            self.num_springs(),
            self.num_hinges(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}
