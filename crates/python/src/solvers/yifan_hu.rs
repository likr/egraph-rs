//! Multilevel Yifan Hu (`YifanHuSolver`) force-directed layout solver bindings.

use egraph::layout as egraph_layout;
use egraph_layout::{YifanHuConfig, YifanHuSolver};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, solver_status_str,
};
use crate::solver_common::{DualIterativeSolver, SolverAction, impl_dual_solver};

use super::layout_dispatch::{execute_euclid_action, impl_dual_nd_solver};

impl_dual_solver!(YifanHuInner, YifanHuSolver);
impl_dual_nd_solver!(YifanHuInner, YifanHuSolver);

/// Cooperative multilevel Yifan Hu (`EC + MIVS`) layout solver with Barnes-Hut repulsion.
///
/// Supports `Layout2d` and `LayoutNd` in `"float64"` and `"float32"` precision while releasing
/// the Python GIL during stepping.
#[pyclass(name = "YifanHuSolver", module = "egraph._core")]
pub struct PyYifanHuSolver {
    inner: YifanHuInner,
}

#[pymethods]
impl PyYifanHuSolver {
    /// Construct a `YifanHuSolver` and build its `EC + MIVS` coarsening hierarchy from `graph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
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
        text_signature = "(graph, *, max_iterations_per_level=200, step_size=1.0, step_ratio=0.9, relative_strength=0.2, optimal_distance=None, theta=1.2, min_coarsen_size=30, min_coarsen_ratio=0.75, convergence_tolerance=1e-4, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
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
        let cfg = YifanHuConfig {
            max_iterations_per_level: max_iterations_per_level.0,
            step_size: step_size.0,
            step_ratio: step_ratio.0,
            relative_strength: relative_strength.0,
            optimal_distance: optimal_distance.map(|v| v.0),
            theta: theta.0,
            min_coarsen_size: min_coarsen_size.0,
            min_coarsen_ratio: min_coarsen_ratio.0,
            convergence_tolerance: convergence_tolerance.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => YifanHuSolver::<f64>::new(&*g, cfg).map(YifanHuInner::F64),
                ScalarDtype::F32 => YifanHuSolver::<f32>::new(&*g, cfg).map(YifanHuInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one Yifan Hu iteration in place on `layout` (`Layout2d | LayoutNd`).
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_euclid_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` Yifan Hu iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyAny>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_euclid_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run multilevel Yifan Hu to completion or convergence in place on `layout`.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_euclid_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Total iterations executed across all multilevel scales so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Maximum total iterations (`num_levels * max_iterations_per_level`).
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Expected number of vertices in the layout.
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            YifanHuInner::F64(s) => s.num_nodes(),
            YifanHuInner::F32(s) => s.num_nodes(),
        }
    }

    /// Number of levels in the `EC + MIVS` coarsening hierarchy (`>= 1`).
    #[getter]
    fn num_levels(&self) -> usize {
        match &self.inner {
            YifanHuInner::F64(s) => s.num_levels(),
            YifanHuInner::F32(s) => s.num_levels(),
        }
    }

    /// Currently active coarsening level (`num_levels - 1` coarsest down to `0` finest).
    #[getter]
    fn current_level(&self) -> usize {
        match &self.inner {
            YifanHuInner::F64(s) => s.current_level(),
            YifanHuInner::F32(s) => s.current_level(),
        }
    }

    /// Maximum iterations per coarsening level.
    #[getter]
    fn max_iterations_per_level(&self) -> usize {
        match &self.inner {
            YifanHuInner::F64(s) => s.config().max_iterations_per_level,
            YifanHuInner::F32(s) => s.config().max_iterations_per_level,
        }
    }

    /// Initial step size.
    #[getter]
    fn step_size(&self) -> f64 {
        match &self.inner {
            YifanHuInner::F64(s) => s.config().step_size,
            YifanHuInner::F32(s) => s.config().step_size,
        }
    }

    /// Step cooling ratio.
    #[getter]
    fn step_ratio(&self) -> f64 {
        match &self.inner {
            YifanHuInner::F64(s) => s.config().step_ratio,
            YifanHuInner::F32(s) => s.config().step_ratio,
        }
    }

    /// Relative repulsive strength `C`.
    #[getter]
    fn relative_strength(&self) -> f64 {
        match &self.inner {
            YifanHuInner::F64(s) => s.config().relative_strength,
            YifanHuInner::F32(s) => s.config().relative_strength,
        }
    }

    /// Configured optional optimal distance `K`.
    #[getter]
    fn optimal_distance(&self) -> Option<f64> {
        match &self.inner {
            YifanHuInner::F64(s) => s.config().optimal_distance,
            YifanHuInner::F32(s) => s.config().optimal_distance,
        }
    }

    /// Barnes-Hut opening angle threshold `theta`.
    #[getter]
    fn theta(&self) -> f64 {
        match &self.inner {
            YifanHuInner::F64(s) => s.config().theta,
            YifanHuInner::F32(s) => s.config().theta,
        }
    }

    /// Minimum coarsening level node count.
    #[getter]
    fn min_coarsen_size(&self) -> usize {
        match &self.inner {
            YifanHuInner::F64(s) => s.config().min_coarsen_size,
            YifanHuInner::F32(s) => s.config().min_coarsen_size,
        }
    }

    /// Maximum allowed coarsening ratio between levels.
    #[getter]
    fn min_coarsen_ratio(&self) -> f64 {
        match &self.inner {
            YifanHuInner::F64(s) => s.config().min_coarsen_ratio,
            YifanHuInner::F32(s) => s.config().min_coarsen_ratio,
        }
    }

    /// Relative displacement convergence tolerance.
    #[getter]
    fn convergence_tolerance(&self) -> f64 {
        match &self.inner {
            YifanHuInner::F64(s) => s.config().convergence_tolerance,
            YifanHuInner::F32(s) => s.config().convergence_tolerance,
        }
    }

    /// Deterministic PRNG seed.
    #[getter]
    fn seed(&self) -> u64 {
        match &self.inner {
            YifanHuInner::F64(s) => s.config().seed,
            YifanHuInner::F32(s) => s.config().seed,
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "YifanHuSolver(num_nodes={}, levels={}, current_level={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.num_levels(),
            self.current_level(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
