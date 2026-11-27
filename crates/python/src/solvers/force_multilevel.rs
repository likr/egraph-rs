//! ForceAtlas2 continuous force-directed placement (`ForceAtlas2Solver`) bindings.

use egraph::layout as egraph_layout;
use egraph_layout::{ForceAtlas2Config, ForceAtlas2Solver};
use numpy::PyArray1;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_optional_weights_1d,
    solver_status_str,
};
use crate::solver_common::{DualIterativeSolver, SolverAction, impl_dual_solver};

use super::layout_dispatch::{execute_euclid_action, impl_dual_nd_solver};

impl_dual_solver!(ForceAtlas2Inner, ForceAtlas2Solver);
impl_dual_nd_solver!(ForceAtlas2Inner, ForceAtlas2Solver);

/// Cooperative ForceAtlas2 continuous force-directed layout solver.
///
/// Supports Barnes-Hut or exact repulsion, `linlog_mode` (Jacomy et al. 2014 ForceAtlas2
/// logarithmic attraction; compare Noack's LinLog energy model), gravity options, and both
/// `Layout2d` and `LayoutNd` in `"float64"` and `"float32"` precision while releasing the GIL.
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
                    ForceAtlas2Solver::<f64>::with_node_radii(&*g, cfg, radii)
                        .map(ForceAtlas2Inner::F64)
                }
                (ScalarDtype::F32, Some(radii)) => {
                    ForceAtlas2Solver::<f32>::with_node_radii(&*g, cfg, radii)
                        .map(ForceAtlas2Inner::F32)
                }
                (ScalarDtype::F64, None) => {
                    ForceAtlas2Solver::<f64>::new(&*g, cfg).map(ForceAtlas2Inner::F64)
                }
                (ScalarDtype::F32, None) => {
                    ForceAtlas2Solver::<f32>::new(&*g, cfg).map(ForceAtlas2Inner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one ForceAtlas2 iteration in place on `layout` (`Layout2d | LayoutNd`).
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_euclid_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` ForceAtlas2 iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyAny>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_euclid_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run ForceAtlas2 to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_euclid_action(&mut self.inner, py, layout, SolverAction::Run)
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

    /// Expected number of vertices in the layout.
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

    /// Configured Barnes-Hut opening angle threshold `theta`.
    #[getter]
    fn theta(&self) -> f64 {
        match &self.inner {
            ForceAtlas2Inner::F64(s) => s.config().theta,
            ForceAtlas2Inner::F32(s) => s.config().theta,
        }
    }

    /// Whether Barnes-Hut spatial repulsion is enabled.
    #[getter]
    fn use_barnes_hut(&self) -> bool {
        match &self.inner {
            ForceAtlas2Inner::F64(s) => s.config().use_barnes_hut,
            ForceAtlas2Inner::F32(s) => s.config().use_barnes_hut,
        }
    }

    /// Whether Jacomy ForceAtlas2 logarithmic attraction mode is enabled.
    #[getter]
    fn linlog_mode(&self) -> bool {
        match &self.inner {
            ForceAtlas2Inner::F64(s) => s.config().linlog_mode,
            ForceAtlas2Inner::F32(s) => s.config().linlog_mode,
        }
    }

    /// Whether strong linear gravity is enabled.
    #[getter]
    fn strong_gravity(&self) -> bool {
        match &self.inner {
            ForceAtlas2Inner::F64(s) => s.config().strong_gravity,
            ForceAtlas2Inner::F32(s) => s.config().strong_gravity,
        }
    }

    /// Gravity strength parameter.
    #[getter]
    fn gravity(&self) -> f64 {
        match &self.inner {
            ForceAtlas2Inner::F64(s) => s.config().gravity,
            ForceAtlas2Inner::F32(s) => s.config().gravity,
        }
    }

    /// Repulsion scaling ratio parameter.
    #[getter]
    fn scaling_ratio(&self) -> f64 {
        match &self.inner {
            ForceAtlas2Inner::F64(s) => s.config().scaling_ratio,
            ForceAtlas2Inner::F32(s) => s.config().scaling_ratio,
        }
    }

    /// Edge weight influence exponent.
    #[getter]
    fn edge_weight_influence(&self) -> f64 {
        match &self.inner {
            ForceAtlas2Inner::F64(s) => s.config().edge_weight_influence,
            ForceAtlas2Inner::F32(s) => s.config().edge_weight_influence,
        }
    }

    /// Whether outbound attraction distribution (hub dissuasion) is enabled.
    #[getter]
    fn outbound_attraction_distribution(&self) -> bool {
        match &self.inner {
            ForceAtlas2Inner::F64(s) => s.config().outbound_attraction_distribution,
            ForceAtlas2Inner::F32(s) => s.config().outbound_attraction_distribution,
        }
    }

    /// Adaptive jitter tolerance parameter.
    #[getter]
    fn jitter_tolerance(&self) -> f64 {
        match &self.inner {
            ForceAtlas2Inner::F64(s) => s.config().jitter_tolerance,
            ForceAtlas2Inner::F32(s) => s.config().jitter_tolerance,
        }
    }

    /// Optional per-step maximum displacement cap.
    #[getter]
    fn max_displacement(&self) -> Option<f64> {
        match &self.inner {
            ForceAtlas2Inner::F64(s) => s.config().max_displacement,
            ForceAtlas2Inner::F32(s) => s.config().max_displacement,
        }
    }

    /// Deterministic PRNG seed.
    #[getter]
    fn seed(&self) -> u64 {
        match &self.inner {
            ForceAtlas2Inner::F64(s) => s.config().seed,
            ForceAtlas2Inner::F32(s) => s.config().seed,
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

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
