//! Force-directed baseline solver (`GemSolver`, `SelLayoutSolver`) bindings.

use egraph::layout as egraph_layout;
use egraph_layout::{
    FruchtermanReingoldConfig, GemConfig, GemSolver, SelConfig, SelLayoutBackend, SelLayoutSolver,
    SelWeightMapping, TFdpConfig,
};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, normalize_enum_key,
    solver_status_str,
};
use crate::layout_buffers::PyLayout2d;
use crate::solver_common::{
    DualIterativeSolver, SolverAction, execute_solver_action, impl_dual_solver,
};

use super::force_classic::parse_fr_backend;
use super::t_fdp::build_t_fdp_config;

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
            max_iterations = None,
            oscillation_angle = StrictF64(std::f64::consts::PI),
            rotation_angle = StrictF64(std::f64::consts::PI / 3.0),
            oscillation_sensitivity = StrictF64(1.0 / 3.0),
            rotation_sensitivity = None,
            perturbation = StrictF64(32.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, edge_length=128.0, gravity=0.0625, t_init=128.0, t_max=256.0, t_min=3.0, max_iterations=None, oscillation_angle=3.141592653589793, rotation_angle=1.0471975511965976, oscillation_sensitivity=0.3333333333333333, rotation_sensitivity=None, perturbation=32.0, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        edge_length: StrictF64,
        gravity: StrictF64,
        t_init: StrictF64,
        t_max: StrictF64,
        t_min: StrictF64,
        max_iterations: Option<StrictUsize>,
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
            max_iterations: max_iterations.map(|v| v.0),
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
                ScalarDtype::F64 => GemSolver::<f64>::new(&*g, cfg).map(GemInner::F64),
                ScalarDtype::F32 => GemSolver::<f32>::new(&*g, cfg).map(GemInner::F32),
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

    /// Maximum number of GEM iterations (`max_iterations` or `min(4 * N, 400)`).
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

fn parse_sel_fr_config(dict: &Bound<'_, PyDict>) -> PyResult<FruchtermanReingoldConfig> {
    let mut cfg = FruchtermanReingoldConfig::default();
    for (k_any, v_any) in dict.iter() {
        let key: String = k_any.extract()?;
        match key.as_str() {
            "iterations" => cfg.iterations = v_any.extract::<StrictUsize>()?.0,
            "backend" => {
                let s: String = v_any.extract()?;
                cfg.backend = parse_fr_backend(&s).map_err(PyValueError::new_err)?;
            }
            "k" => cfg.k = v_any.extract::<Option<StrictF64>>()?.map(|x| x.0),
            "area" => cfg.area = v_any.extract::<Option<StrictF64>>()?.map(|x| x.0),
            "initial_temperature" => {
                cfg.initial_temperature = v_any.extract::<Option<StrictF64>>()?.map(|x| x.0);
            }
            "gravity" => cfg.gravity = v_any.extract::<StrictF64>()?.0,
            "theta" => cfg.theta = v_any.extract::<StrictF64>()?.0,
            "seed" => cfg.seed = v_any.extract::<StrictU64>()?.0,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown fr_config key {key:?}"
                )));
            }
        }
    }
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

fn parse_sel_t_fdp_config(dict: &Bound<'_, PyDict>) -> PyResult<TFdpConfig> {
    let def = TFdpConfig::default();
    let mut backend = "ibfft".to_string();
    let mut iterations = def.iterations;
    let mut learning_rate = def.learning_rate;
    let mut momentum = def.momentum;
    let mut theta = def.theta;
    let mut alpha = def.alpha;
    let mut gamma = def.gamma;
    let mut rho_start = def.rho.start_weight;
    let mut rho_end = def.rho.end_weight;
    let mut rho_power = def.rho.power;
    let mut interpolation_order = def.ibfft.order.order();
    let mut intervals = def.ibfft.intervals;
    let mut min_grid_dim = def.ibfft.min_grid_dim;
    let mut max_grid_dim = def.ibfft.max_grid_dim;
    let mut seed = def.seed;
    for (k_any, v_any) in dict.iter() {
        let key: String = k_any.extract()?;
        match key.as_str() {
            "backend" => backend = v_any.extract()?,
            "iterations" => iterations = v_any.extract::<StrictUsize>()?.0,
            "learning_rate" => learning_rate = v_any.extract::<StrictF64>()?.0,
            "momentum" => momentum = v_any.extract::<StrictF64>()?.0,
            "theta" => theta = v_any.extract::<StrictF64>()?.0,
            "alpha" => alpha = v_any.extract::<StrictF64>()?.0,
            "gamma" => gamma = v_any.extract::<StrictF64>()?.0,
            "rho_start" => rho_start = v_any.extract::<StrictF64>()?.0,
            "rho_end" => rho_end = v_any.extract::<StrictF64>()?.0,
            "rho_power" => rho_power = v_any.extract::<StrictF64>()?.0,
            "interpolation_order" => interpolation_order = v_any.extract::<StrictUsize>()?.0,
            "intervals" => intervals = v_any.extract::<StrictUsize>()?.0,
            "min_grid_dim" => min_grid_dim = v_any.extract::<StrictUsize>()?.0,
            "max_grid_dim" => max_grid_dim = v_any.extract::<StrictUsize>()?.0,
            "seed" => seed = v_any.extract::<StrictU64>()?.0,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown t_fdp_config key {key:?}"
                )));
            }
        }
    }
    build_t_fdp_config(
        &backend,
        iterations,
        learning_rate,
        momentum,
        theta,
        alpha,
        gamma,
        rho_start,
        rho_end,
        rho_power,
        interpolation_order,
        intervals,
        min_grid_dim,
        max_grid_dim,
        seed,
    )
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
            fr_config = None,
            t_fdp_config = None,
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, tree_depth=5, pq_max_pops=10, mapping='sigmoid', layout_backend='fr', iterations=200, seed=42, fr_config=None, t_fdp_config=None, dtype='float64')"
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
        fr_config: Option<&Bound<'_, PyDict>>,
        t_fdp_config: Option<&Bound<'_, PyDict>>,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let parsed_mapping = parse_sel_weight_mapping(mapping).map_err(PyValueError::new_err)?;
        let parsed_backend =
            parse_sel_layout_backend(layout_backend).map_err(PyValueError::new_err)?;
        let parsed_fr = fr_config.map(parse_sel_fr_config).transpose()?;
        let parsed_t_fdp = t_fdp_config.map(parse_sel_t_fdp_config).transpose()?;
        let cfg = SelConfig {
            tree_depth: tree_depth.0,
            pq_max_pops: pq_max_pops.0,
            mapping: parsed_mapping,
            layout_backend: parsed_backend,
            iterations: iterations.0,
            seed: seed.0,
            fr_config: parsed_fr,
            t_fdp_config: parsed_t_fdp,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => SelLayoutSolver::<f64>::new(&*g, cfg).map(SelLayoutInner::F64),
                ScalarDtype::F32 => SelLayoutSolver::<f32>::new(&*g, cfg).map(SelLayoutInner::F32),
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
