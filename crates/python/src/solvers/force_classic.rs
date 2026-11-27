//! Classic force-directed layout solver (`LinLogSolver`, `FruchtermanReingoldSolver`) bindings.

use egraph::layout as egraph_layout;
use egraph_layout::{
    FrBackend, FruchtermanReingoldConfig, FruchtermanReingoldSolver, LinLogConfig, LinLogRepulsion,
    LinLogSolver,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, normalize_enum_key,
    solver_status_str,
};
use crate::layout_buffers::{PyLayout2d, dispatch_layout2d};
use crate::solver_common::{DualIterativeSolver, SolverAction, impl_dual_solver};

use super::layout_dispatch::{
    execute_any_layout_action, execute_euclid_action, impl_dual_manifold_solver,
    impl_dual_nd_solver,
};

pub(super) fn parse_linlog_repulsion(raw: &str) -> Result<LinLogRepulsion, String> {
    match normalize_enum_key(raw).as_str() {
        "edgerepulsion" | "edge" => Ok(LinLogRepulsion::EdgeRepulsion),
        "noderepulsion" | "node" => Ok(LinLogRepulsion::NodeRepulsion),
        _ => Err(format!(
            "unknown LinLog repulsion_model {raw:?}; expected 'edge_repulsion' or 'node_repulsion'"
        )),
    }
}

pub(super) fn parse_fr_backend(raw: &str) -> Result<FrBackend, String> {
    match normalize_enum_key(raw).as_str() {
        "exact" => Ok(FrBackend::Exact),
        "barneshut" | "bh" => Ok(FrBackend::BarnesHut),
        "grid" => Ok(FrBackend::Grid),
        _ => Err(format!(
            "unknown Fruchterman-Reingold backend {raw:?}; expected 'exact', 'barnes_hut', or 'grid'"
        )),
    }
}

impl_dual_solver!(LinLogInner, LinLogSolver);
impl_dual_nd_solver!(LinLogInner, LinLogSolver);
impl_dual_solver!(FrInner, FruchtermanReingoldSolver);
impl_dual_nd_solver!(FrInner, FruchtermanReingoldSolver);
impl_dual_manifold_solver!(FrInner, FruchtermanReingoldSolver);

/// Cooperative Noack (2007/2009) LinLog and generalized `(a, r)`-PolyLog layout solver.
///
/// Supports edge-repulsion and node-repulsion models, optional Barnes-Hut quadtree
/// acceleration, and both `"float64"` and `"float32"` `Layout2d` while releasing the GIL.
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
            iterations = StrictUsize(200),
            repulsion_model = "edge_repulsion",
            attraction_exponent = StrictF64(0.0),
            repulsion_exponent = StrictF64(-1.0),
            theta = StrictF64(1.2),
            use_barnes_hut = true,
            gravity = StrictF64(0.01),
            edge_weight_influence = StrictF64(1.0),
            max_displacement = None,
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, iterations=200, repulsion_model='edge_repulsion', attraction_exponent=0.0, repulsion_exponent=-1.0, theta=1.2, use_barnes_hut=True, gravity=0.01, edge_weight_influence=1.0, max_displacement=None, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        iterations: StrictUsize,
        repulsion_model: &str,
        attraction_exponent: StrictF64,
        repulsion_exponent: StrictF64,
        theta: StrictF64,
        use_barnes_hut: bool,
        gravity: StrictF64,
        edge_weight_influence: StrictF64,
        max_displacement: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let model = parse_linlog_repulsion(repulsion_model).map_err(PyValueError::new_err)?;
        let cfg = LinLogConfig {
            iterations: iterations.0,
            repulsion_model: model,
            attraction_exponent: attraction_exponent.0,
            repulsion_exponent: repulsion_exponent.0,
            theta: theta.0,
            use_barnes_hut,
            gravity: gravity.0,
            edge_weight_influence: edge_weight_influence.0,
            max_displacement: max_displacement.map(|v| v.0),
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => LinLogSolver::<f64>::new(&*g, cfg).map(LinLogInner::F64),
                ScalarDtype::F32 => LinLogSolver::<f32>::new(&*g, cfg).map(LinLogInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one LinLog iteration in place on `layout` (`Layout2d | LayoutNd`).
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_euclid_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` LinLog iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyAny>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_euclid_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run LinLog optimization to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_euclid_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Evaluate the exact Noack (2009) `(a, r)`-LinLog energy of `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn energy(&self, py: Python<'_>, layout: &PyLayout2d) -> PyResult<f64> {
        dispatch_layout2d!(py, layout, |soa| match &self.inner {
            LinLogInner::F64(s) => s.energy(&soa),
            LinLogInner::F32(s) => s.energy(&soa),
        })
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

    /// Configured total iterations.
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
            LinLogInner::F64(s) => s.num_nodes(),
            LinLogInner::F32(s) => s.num_nodes(),
        }
    }

    /// Effective per-step vertex displacement cap.
    #[getter]
    fn max_displacement(&self) -> f64 {
        match &self.inner {
            LinLogInner::F64(s) => s.max_displacement(),
            LinLogInner::F32(s) => s.max_displacement(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `LinLogSolver`.
    fn __repr__(&self) -> String {
        format!(
            "LinLogSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

/// Cooperative Fruchterman-Reingold (1991) force-directed layout solver.
///
/// Supports `"exact"`, `"barnes_hut"`, and `"grid"` spatial repulsion backends in both
/// `"float64"` and `"float32"` precision while releasing the Python GIL.
#[pyclass(name = "FruchtermanReingoldSolver", module = "egraph._core")]
pub struct PyFruchtermanReingoldSolver {
    inner: FrInner,
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
            iterations = StrictUsize(200),
            backend = "exact",
            k = None,
            area = None,
            initial_temperature = None,
            gravity = StrictF64(0.0),
            theta = StrictF64(1.2),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, iterations=200, backend='exact', k=None, area=None, initial_temperature=None, gravity=0.0, theta=1.2, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        iterations: StrictUsize,
        backend: &str,
        k: Option<StrictF64>,
        area: Option<StrictF64>,
        initial_temperature: Option<StrictF64>,
        gravity: StrictF64,
        theta: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let parsed_backend = parse_fr_backend(backend).map_err(PyValueError::new_err)?;
        let cfg = FruchtermanReingoldConfig {
            iterations: iterations.0,
            backend: parsed_backend,
            k: k.map(|v| v.0),
            area: area.map(|v| v.0),
            initial_temperature: initial_temperature.map(|v| v.0),
            gravity: gravity.0,
            theta: theta.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    FruchtermanReingoldSolver::<f64>::new(&*g, cfg).map(FrInner::F64)
                }
                ScalarDtype::F32 => {
                    FruchtermanReingoldSolver::<f32>::new(&*g, cfg).map(FrInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one Fruchterman-Reingold iteration in place on `layout` (`Layout2d | LayoutNd | ManifoldLayout2d`).
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_any_layout_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` Fruchterman-Reingold iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyAny>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_any_layout_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run Fruchterman-Reingold to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_any_layout_action(&mut self.inner, py, layout, SolverAction::Run)
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

    /// Configured total iterations.
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
            FrInner::F64(s) => s.num_nodes(),
            FrInner::F32(s) => s.num_nodes(),
        }
    }

    /// Effective optimal pairwise distance `k`.
    #[getter]
    fn optimal_distance(&self) -> f64 {
        match &self.inner {
            FrInner::F64(s) => s.optimal_distance(),
            FrInner::F32(s) => s.optimal_distance(),
        }
    }

    /// Temperature for the upcoming (or final) iteration.
    #[getter]
    fn current_temperature(&self) -> f64 {
        match &self.inner {
            FrInner::F64(s) => s.current_temperature(),
            FrInner::F32(s) => s.current_temperature(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `FruchtermanReingoldSolver`.
    fn __repr__(&self) -> String {
        format!(
            "FruchtermanReingoldSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
