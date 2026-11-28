//! Cooperative Gajer & Kobourov (2002) GRIP multilevel graph drawing solver (`GripSolver`).

use egraph::layout::{GripConfig, GripSolver};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::core_error_to_py;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, solver_status_str};
use crate::impl_dual_solver;
use crate::layout_buffers::PyLayout2d;
use crate::solver_common::{
    DualIterativeSolver, SolverAction as DualSolverAction, execute_solver_action,
    format_optional_usize,
};

impl_dual_solver!(GripInner, GripSolver);

/// Cooperative Gajer & Kobourov (2002) GRIP multilevel graph drawing solver (`Layout2d`).
#[pyclass(name = "GripSolver", module = "egraph._core")]
pub struct PyGripSolver {
    inner: GripInner,
}

#[pymethods]
impl PyGripSolver {
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
        let cfg = GripConfig::new()
            .with_rounds_min(rounds_min.0)
            .with_rounds_max(rounds_max.0)
            .with_local_neighborhood_size(local_neighborhood_size.0)
            .with_repulsion_scale(repulsion_scale.0)
            .with_edge_length(edge_length.0)
            .with_temp_r(temp_r.0)
            .with_temp_s(temp_s.0)
            .with_seed(seed.0);
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

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, DualSolverAction::Step)
    }

    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(
            &mut self.inner,
            py,
            layout,
            DualSolverAction::RunSteps(steps.0),
        )
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, DualSolverAction::Run)
    }

    #[getter]
    fn num_levels(&self) -> usize {
        match &self.inner {
            GripInner::F64(s) => s.num_levels(),
            GripInner::F32(s) => s.num_levels(),
        }
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
            GripInner::F64(s) => s.num_nodes(),
            GripInner::F32(s) => s.num_nodes(),
        }
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "GripSolver(num_nodes={}, num_levels={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.num_levels(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}
