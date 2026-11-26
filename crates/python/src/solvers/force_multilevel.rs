//! Continuous force-directed placement (`ForceAtlas2Solver`, `YifanHuSolver`) bindings.

use egraph::layout as egraph_layout;
use egraph_layout::{ForceAtlas2Config, ForceAtlas2Solver, YifanHuConfig, YifanHuSolver};
use numpy::PyArray1;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_optional_weights_1d,
    solver_status_str,
};
use crate::layout_buffers::PyLayout2d;
use crate::solver_common::{
    DualIterativeSolver, SolverAction, execute_solver_action, impl_dual_solver,
};

impl_dual_solver!(ForceAtlas2Inner, ForceAtlas2Solver);
impl_dual_solver!(YifanHuInner, YifanHuSolver);

/// Cooperative ForceAtlas2 continuous force-directed layout solver.
///
/// Supports Barnes-Hut quadtree or exact repulsion, `linlog_mode` (Jacomy et al. 2014
/// ForceAtlas2 logarithmic attraction `w * ln(1 + d) / d * (p_v - p_u)`, which decays to
/// zero as `d -> 0` and is distinct from Noack 2007 true LinLog `LinLogSolver`), gravity
/// options, and both `"float64"` and `"float32"` `Layout2d` while releasing the Python GIL.
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
            iterations = StrictUsize(200),
            theta = StrictF64(1.2),
            use_barnes_hut = true,
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
        text_signature = "(graph, *, iterations=200, theta=1.2, use_barnes_hut=True, linlog_mode=False, strong_gravity=False, gravity=1.0, scaling_ratio=2.0, edge_weight_influence=1.0, outbound_attraction_distribution=False, jitter_tolerance=1.0, max_displacement=None, prevent_overlap=False, default_node_radius=1.0, node_radii=None, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        iterations: StrictUsize,
        theta: StrictF64,
        use_barnes_hut: bool,
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
        let radii_opt = extract_optional_weights_1d(node_radii, "node_radii")?;
        let cfg = ForceAtlas2Config {
            iterations: iterations.0,
            theta: theta.0,
            use_barnes_hut,
            linlog_mode,
            strong_gravity,
            gravity: gravity.0,
            scaling_ratio: scaling_ratio.0,
            edge_weight_influence: edge_weight_influence.0,
            outbound_attraction_distribution,
            jitter_tolerance: jitter_tolerance.0,
            max_displacement: max_displacement.map(|v| v.0),
            prevent_overlap,
            default_node_radius: default_node_radius.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match (dtype, radii_opt) {
                (ScalarDtype::F64, Some(radii)) => {
                    ForceAtlas2Solver::<f64>::with_node_radii(g, cfg, radii)
                        .map(ForceAtlas2Inner::F64)
                }
                (ScalarDtype::F32, Some(radii)) => {
                    ForceAtlas2Solver::<f32>::with_node_radii(g, cfg, radii)
                        .map(ForceAtlas2Inner::F32)
                }
                (ScalarDtype::F64, None) => {
                    ForceAtlas2Solver::<f64>::new(g, cfg).map(ForceAtlas2Inner::F64)
                }
                (ScalarDtype::F32, None) => {
                    ForceAtlas2Solver::<f32>::new(g, cfg).map(ForceAtlas2Inner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one ForceAtlas2 iteration in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` ForceAtlas2 iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run ForceAtlas2 to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
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
            ForceAtlas2Inner::F64(s) => s.num_nodes(),
            ForceAtlas2Inner::F32(s) => s.num_nodes(),
        }
    }

    /// Whether Jacomy (2014) border-to-border overlap prevention is enabled.
    #[getter]
    fn prevent_overlap(&self) -> bool {
        match &self.inner {
            ForceAtlas2Inner::F64(s) => s.config().prevent_overlap,
            ForceAtlas2Inner::F32(s) => s.config().prevent_overlap,
        }
    }

    /// Configured default per-node radius used when `node_radii` was omitted.
    #[getter]
    fn default_node_radius(&self) -> f64 {
        match &self.inner {
            ForceAtlas2Inner::F64(s) => s.config().default_node_radius,
            ForceAtlas2Inner::F32(s) => s.config().default_node_radius,
        }
    }

    /// Effective per-node collision radii as a `float64` NumPy array of length `N`.
    #[getter]
    fn node_radii<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        match &self.inner {
            ForceAtlas2Inner::F64(s) => PyArray1::from_slice(py, s.node_radii()),
            ForceAtlas2Inner::F32(s) => PyArray1::from_slice(py, s.node_radii()),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `ForceAtlas2Solver`.
    fn __repr__(&self) -> String {
        format!(
            "ForceAtlas2Solver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

/// Cooperative multilevel Yifan Hu (`EC + MIVS`) layout solver with Barnes-Hut repulsion.
///
/// Supports both `"float64"` and `"float32"` `Layout2d` and releases the Python GIL during stepping.
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
                ScalarDtype::F64 => YifanHuSolver::<f64>::new(g, cfg).map(YifanHuInner::F64),
                ScalarDtype::F32 => YifanHuSolver::<f32>::new(g, cfg).map(YifanHuInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one Yifan Hu iteration in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` Yifan Hu iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run multilevel Yifan Hu to completion or convergence in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
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

    /// Expected number of vertices in `Layout2d`.
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

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `YifanHuSolver`.
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
