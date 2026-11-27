//! Multilevel and entropy-based stress solvers (`MaxentStressSolver`, `GripSolver`) bindings.

use egraph::layout as egraph_layout;
use egraph_layout::{GripConfig, GripSolver, MaxentStressConfig, MaxentStressSolver};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, solver_status_str,
};
use crate::layout_buffers::PyLayout2d;
use crate::solver_common::{
    DualIterativeSolver, SolverAction, execute_solver_action, impl_dual_solver,
};

use super::layout_dispatch::{execute_euclid_action, impl_dual_nd_solver};

impl_dual_solver!(MaxentStressInner, MaxentStressSolver);
impl_dual_nd_solver!(MaxentStressInner, MaxentStressSolver);
impl_dual_solver!(GripInner, GripSolver);

/// Cooperative Gansner et al. (2013) Maxent-Stress layout solver.
///
/// Combines AMG-PCG graph-Laplacian stress majorization with generalized `q`-entropy
/// long-range repulsion (Barnes-Hut quadtree or exact pairwise summation), supporting
/// both `"float64"` and `"float32"` `Layout2d` while releasing the Python GIL.
#[pyclass(name = "MaxentStressSolver", module = "egraph._core")]
pub struct PyMaxentStressSolver {
    inner: MaxentStressInner,
}

#[pymethods]
impl PyMaxentStressSolver {
    /// Construct a `MaxentStressSolver` from a `CsrGraph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
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
            use_barnes_hut = true,
            eps_sq = StrictF64(1e-6),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, q=None, alpha_init=1.0, alpha_decay=0.3, alpha_min=0.008, stages=5, steps_per_stage=50, tol=1e-3, cg_tol=0.1, cg_max_iters=10, theta=1.2, use_barnes_hut=True, eps_sq=1e-6, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
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
        use_barnes_hut: bool,
        eps_sq: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = MaxentStressConfig {
            q: q.map(|v| v.0),
            alpha_init: alpha_init.0,
            alpha_decay: alpha_decay.0,
            alpha_min: alpha_min.0,
            stages: stages.0,
            steps_per_stage: steps_per_stage.0,
            tol: tol.0,
            cg_tol: cg_tol.0,
            cg_max_iters: cg_max_iters.0,
            theta: theta.0,
            use_barnes_hut,
            eps_sq: eps_sq.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    MaxentStressSolver::<f64>::new(&*g, cfg).map(MaxentStressInner::F64)
                }
                ScalarDtype::F32 => {
                    MaxentStressSolver::<f32>::new(&*g, cfg).map(MaxentStressInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one Maxent-Stress majorization step in place on `layout` (`Layout2d | LayoutNd`).
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_euclid_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` Maxent-Stress majorization steps in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyAny>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_euclid_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run Maxent-Stress optimization to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_euclid_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Current annealing repulsion weight `alpha`.
    #[getter]
    fn current_alpha(&self) -> f64 {
        match &self.inner {
            MaxentStressInner::F64(s) => s.current_alpha(),
            MaxentStressInner::F32(s) => s.current_alpha(),
        }
    }

    /// Effective repulsion exponent `q` (explicit `q`, or auto-selected `0.0` / `0.8`).
    #[getter]
    fn effective_q(&self) -> f64 {
        match &self.inner {
            MaxentStressInner::F64(s) => s.effective_q(),
            MaxentStressInner::F32(s) => s.effective_q(),
        }
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of iterations executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured total iterations (`stages * steps_per_stage`).
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Expected number of vertices in `Layout2d`.
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            MaxentStressInner::F64(s) => s.num_nodes(),
            MaxentStressInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `MaxentStressSolver`.
    fn __repr__(&self) -> String {
        format!(
            "MaxentStressSolver(num_nodes={}, q={:.3}, alpha={:.4}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.effective_q(),
            self.current_alpha(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

/// Cooperative Gajer & Kobourov (2002) GRIP multilevel graph drawing solver.
///
/// Uses Maximal Independent Set (MIS) filtration with local Kamada-Kawai refinement on
/// coarse levels and Fruchterman-Reingold forces on the finest level, supporting both
/// `"float64"` and `"float32"` `Layout2d` while releasing the Python GIL.
#[pyclass(name = "GripSolver", module = "egraph._core")]
pub struct PyGripSolver {
    inner: GripInner,
}

#[pymethods]
impl PyGripSolver {
    /// Construct a `GripSolver` from a `CsrGraph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            rounds_min = StrictUsize(5),
            rounds_max = StrictUsize(30),
            local_neighborhood_size = StrictUsize(20),
            repulsion_scale = StrictF64(0.05),
            edge_length = StrictF64(1.0),
            temp_r = StrictF64(0.15),
            temp_s = StrictF64(3.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, rounds_min=5, rounds_max=30, local_neighborhood_size=20, repulsion_scale=0.05, edge_length=1.0, temp_r=0.15, temp_s=3.0, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        rounds_min: StrictUsize,
        rounds_max: StrictUsize,
        local_neighborhood_size: StrictUsize,
        repulsion_scale: StrictF64,
        edge_length: StrictF64,
        temp_r: StrictF64,
        temp_s: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = GripConfig {
            rounds_min: rounds_min.0,
            rounds_max: rounds_max.0,
            local_neighborhood_size: local_neighborhood_size.0,
            repulsion_scale: repulsion_scale.0,
            edge_length: edge_length.0,
            temp_r: temp_r.0,
            temp_s: temp_s.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => GripSolver::<f64>::new(&*g, cfg).map(GripInner::F64),
                ScalarDtype::F32 => GripSolver::<f32>::new(&*g, cfg).map(GripInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one GRIP refinement round in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` GRIP refinement rounds in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run GRIP multilevel optimization to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Maximum number of MIS filtration levels across connected components.
    #[getter]
    fn num_levels(&self) -> usize {
        match &self.inner {
            GripInner::F64(s) => s.num_levels(),
            GripInner::F32(s) => s.num_levels(),
        }
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of rounds executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Scheduled total refinement rounds across MIS levels.
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Expected number of vertices in `Layout2d`.
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            GripInner::F64(s) => s.num_nodes(),
            GripInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `GripSolver`.
    fn __repr__(&self) -> String {
        format!(
            "GripSolver(num_nodes={}, num_levels={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.num_levels(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
