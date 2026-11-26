//! Force-directed baseline solver (`GemSolver`, `SelLayoutSolver`) bindings.

use egraph::layout as egraph_layout;
use egraph_layout::{
    GemConfig, GemSolver, SelConfig, SelLayoutBackend, SelLayoutSolver, SelWeightMapping,
};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, normalize_enum_key,
    solver_status_str,
};
use crate::layout_buffers::PyLayout2d;
use crate::solver_common::{
    DualIterativeSolver, SolverAction, execute_solver_action, impl_dual_solver,
};

pub(super) fn parse_sel_weight_mapping(raw: &str) -> Result<SelWeightMapping, String> {
    match normalize_enum_key(raw).as_str() {
        "sigmoid" => Ok(SelWeightMapping::Sigmoid),
        "linear" => Ok(SelWeightMapping::Linear),
        "tanh" => Ok(SelWeightMapping::Tanh),
        "customlogistic" | "logistic" => Ok(SelWeightMapping::CustomLogistic),
        "binary" => Ok(SelWeightMapping::Binary),
        _ => Err(format!(
            "unknown SEL weight mapping {raw:?}; expected 'sigmoid', 'linear', 'tanh', 'custom_logistic', or 'binary'"
        )),
    }
}

pub(super) fn parse_sel_layout_backend(raw: &str) -> Result<SelLayoutBackend, String> {
    match normalize_enum_key(raw).as_str() {
        "fr" | "fruchtermanreingold" => Ok(SelLayoutBackend::Fr),
        "tfdp" => Ok(SelLayoutBackend::TFdp),
        _ => Err(format!(
            "unknown SEL layout_backend {raw:?}; expected 'fr' or 't_fdp'"
        )),
    }
}

impl_dual_solver!(GemInner, GemSolver);
impl_dual_solver!(SelLayoutInner, SelLayoutSolver);

/// Cooperative Frick, Ludwig & Mehldau (1995) GEM force-directed graph layout solver.
///
/// Updates vertex positions with per-node local temperature, skew-gauge oscillation and
/// rotation detection, and barycenter gravity across both `"float64"` and `"float32"`
/// `Layout2d` while releasing the Python GIL.
#[pyclass(name = "GemSolver", module = "egraph._core")]
pub struct PyGemSolver {
    inner: GemInner,
}

#[pymethods]
impl PyGemSolver {
    /// Construct a `GemSolver` from a `CsrGraph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            edge_length = StrictF64(128.0),
            gravity = StrictF64(1.0 / 16.0),
            t_init = StrictF64(128.0),
            t_max = StrictF64(256.0),
            t_min = StrictF64(3.0),
            max_rounds = None,
            oscillation_angle = StrictF64(std::f64::consts::PI),
            rotation_angle = StrictF64(std::f64::consts::PI / 3.0),
            oscillation_sensitivity = StrictF64(1.0 / 3.0),
            rotation_sensitivity = None,
            perturbation = StrictF64(32.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, edge_length=128.0, gravity=0.0625, t_init=128.0, t_max=256.0, t_min=3.0, max_rounds=None, oscillation_angle=3.141592653589793, rotation_angle=1.0471975511965976, oscillation_sensitivity=0.3333333333333333, rotation_sensitivity=None, perturbation=32.0, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        edge_length: StrictF64,
        gravity: StrictF64,
        t_init: StrictF64,
        t_max: StrictF64,
        t_min: StrictF64,
        max_rounds: Option<StrictUsize>,
        oscillation_angle: StrictF64,
        rotation_angle: StrictF64,
        oscillation_sensitivity: StrictF64,
        rotation_sensitivity: Option<StrictF64>,
        perturbation: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = GemConfig {
            edge_length: edge_length.0,
            gravity: gravity.0,
            t_init: t_init.0,
            t_max: t_max.0,
            t_min: t_min.0,
            max_iterations: max_rounds.map(|v| v.0),
            oscillation_angle: oscillation_angle.0,
            rotation_angle: rotation_angle.0,
            oscillation_sensitivity: oscillation_sensitivity.0,
            rotation_sensitivity: rotation_sensitivity.map(|v| v.0),
            perturbation: perturbation.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => GemSolver::<f64>::new(g, cfg).map(GemInner::F64),
                ScalarDtype::F32 => GemSolver::<f32>::new(g, cfg).map(GemInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one GEM round in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` GEM rounds in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run GEM optimization to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
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

    /// Maximum number of GEM rounds (`max_rounds` or `4 * max(1, N)`).
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
            GemInner::F64(s) => s.num_nodes(),
            GemInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `GemSolver`.
    fn __repr__(&self) -> String {
        format!(
            "GemSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

/// Cooperative Xue et al. (2025) Structural Entropy Layout (`SEL`) solver.
///
/// Builds a `k`-dimensional structural-entropy coding tree, detects the elbow threshold
/// `se_target`, modulates edge weights (`"sigmoid"`, `"linear"`, `"tanh"`,
/// `"custom_logistic"`, or `"binary"`), and optimizes via `"fr"` or `"t_fdp"` while
/// releasing the Python GIL.
#[pyclass(name = "SelLayoutSolver", module = "egraph._core")]
pub struct PySelLayoutSolver {
    inner: SelLayoutInner,
}

#[pymethods]
impl PySelLayoutSolver {
    /// Construct a `SelLayoutSolver` from a `CsrGraph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            tree_depth = StrictUsize(5),
            pq_max_pops = StrictUsize(10),
            mapping = "sigmoid",
            layout_backend = "fr",
            iterations = StrictUsize(200),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, tree_depth=5, pq_max_pops=10, mapping='sigmoid', layout_backend='fr', iterations=200, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        tree_depth: StrictUsize,
        pq_max_pops: StrictUsize,
        mapping: &str,
        layout_backend: &str,
        iterations: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let parsed_mapping = parse_sel_weight_mapping(mapping).map_err(PyValueError::new_err)?;
        let parsed_backend =
            parse_sel_layout_backend(layout_backend).map_err(PyValueError::new_err)?;
        let cfg = SelConfig {
            tree_depth: tree_depth.0,
            pq_max_pops: pq_max_pops.0,
            mapping: parsed_mapping,
            layout_backend: parsed_backend,
            iterations: iterations.0,
            seed: seed.0,
            ..SelConfig::default()
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => SelLayoutSolver::<f64>::new(g, cfg).map(SelLayoutInner::F64),
                ScalarDtype::F32 => SelLayoutSolver::<f32>::new(g, cfg).map(SelLayoutInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one backend layout step in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` backend layout steps in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run SEL backend optimization to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Return a `float64` NumPy copy of SEL-modulated edge weights aligned with canonical undirected edges `(u < v)`.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn modulated_weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let slice = match &self.inner {
            SelLayoutInner::F64(s) => s.modulated_weights(),
            SelLayoutInner::F32(s) => s.modulated_weights(),
        };
        PyArray1::from_slice(py, slice)
    }

    /// Elbow-selected structural entropy threshold `SE_target`.
    #[getter]
    fn se_target(&self) -> f64 {
        match &self.inner {
            SelLayoutInner::F64(s) => s.se_target(),
            SelLayoutInner::F32(s) => s.se_target(),
        }
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of backend iterations executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured total backend iterations.
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
            SelLayoutInner::F64(s) => s.num_nodes(),
            SelLayoutInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `SelLayoutSolver`.
    fn __repr__(&self) -> String {
        format!(
            "SelLayoutSolver(num_nodes={}, se_target={:.4}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.se_target(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
