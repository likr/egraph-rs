//! Cooperative GEM and Structural Entropy Layout (`SEL`) solver bindings (`Layout2d`).

use egraph::layout::{
    FrBackend, FruchtermanReingoldConfig, GemConfig, GemSolver, RhoSchedule, SelConfig,
    SelLayoutBackend, SelLayoutSolver, SelWeightMapping, TFdpBackend2d, TFdpConfig,
};
use egraph::spatial::{IbFftConfig, InterpolationOrder};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::csr_graph::PyCsrGraph;
use crate::errors::{core_error_to_py, invalid_input_err, unsupported_operation_err};
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, normalize_enum_key, solver_status_str,
};
use crate::impl_dual_solver;
use crate::layout_buffers::PyLayout2d;
use crate::solver_common::{
    DualIterativeSolver, SolverAction, execute_solver_action, format_optional_usize,
};

use super::force_fr::{FrBackendChoice, parse_fr_backend_choice};
use super::t_fdp::{TFdpBackendChoice, build_t_fdp_backend_2d};

pub(super) fn parse_sel_weight_mapping(raw: &str) -> Result<SelWeightMapping, String> {
    match normalize_enum_key(raw).as_str() {
        "sigmoid" => Ok(SelWeightMapping::Sigmoid),
        "linear" => Ok(SelWeightMapping::Linear),
        "tanh" => Ok(SelWeightMapping::Tanh),
        "customlogistic" | "logistic" => Ok(SelWeightMapping::CustomLogistic),
        "binary" => Ok(SelWeightMapping::Binary),
        _ => Err(format!(
            "unknown SEL mapping {raw:?}; expected 'sigmoid', 'linear', 'tanh', 'custom_logistic', or 'binary'"
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

/// Cooperative Frick et al. (1995) GEM layout solver (`Layout2d`).
#[pyclass(name = "GemSolver", module = "egraph._core")]
pub struct PyGemSolver {
    inner: GemInner,
}

#[pymethods]
impl PyGemSolver {
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
        let cfg = GemConfig::new()
            .with_edge_length(edge_length.0)
            .with_gravity(gravity.0)
            .with_t_init(t_init.0)
            .with_t_max(t_max.0)
            .with_t_min(t_min.0)
            .with_max_iterations(max_iterations.map(|v| v.0))
            .with_oscillation_angle(oscillation_angle.0)
            .with_rotation_angle(rotation_angle.0)
            .with_oscillation_sensitivity(oscillation_sensitivity.0)
            .with_rotation_sensitivity(rotation_sensitivity.map(|v| v.0))
            .with_perturbation(perturbation.0)
            .with_seed(seed.0);
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

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
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
            GemInner::F64(s) => s.num_nodes(),
            GemInner::F32(s) => s.num_nodes(),
        }
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "GemSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}

impl_dual_solver!(SelLayoutInner, SelLayoutSolver);

/// Cooperative Xue et al. (2025) Structural Entropy Layout (`SEL`) solver (`Layout2d`).
#[pyclass(name = "SelLayoutSolver", module = "egraph._core")]
pub struct PySelLayoutSolver {
    inner: SelLayoutInner,
}

fn parse_sel_fr_config(
    dict: &Bound<'_, PyDict>,
) -> PyResult<(FruchtermanReingoldConfig, FrBackend)> {
    let mut cfg = FruchtermanReingoldConfig::default();
    let mut backend_str = "exact".to_string();
    let mut theta = 1.2_f64;
    for (k_any, v_any) in dict.iter() {
        let key: String = k_any.extract()?;
        match key.as_str() {
            "iterations" => cfg = cfg.with_iterations(v_any.extract::<StrictUsize>()?.0),
            "backend" => backend_str = v_any.extract()?,
            "k" => cfg = cfg.with_k(v_any.extract::<Option<StrictF64>>()?.map(|x| x.0)),
            "area" => cfg = cfg.with_area(v_any.extract::<Option<StrictF64>>()?.map(|x| x.0)),
            "initial_temperature" => {
                cfg = cfg
                    .with_initial_temperature(v_any.extract::<Option<StrictF64>>()?.map(|x| x.0));
            }
            "gravity" => cfg = cfg.with_gravity(v_any.extract::<StrictF64>()?.0),
            "theta" => theta = v_any.extract::<StrictF64>()?.0,
            "seed" => cfg = cfg.with_seed(v_any.extract::<StrictU64>()?.0),
            _ => {
                return Err(invalid_input_err(format!("unknown fr_config key {key:?}")));
            }
        }
    }
    cfg.validate().map_err(core_error_to_py)?;
    let fr_backend =
        match parse_fr_backend_choice(&backend_str).map_err(unsupported_operation_err)? {
            FrBackendChoice::Auto | FrBackendChoice::Exact => FrBackend::exact(),
            FrBackendChoice::BarnesHut => FrBackend::barnes_hut(theta),
            FrBackendChoice::Grid => FrBackend::grid(),
        };
    fr_backend.validate().map_err(core_error_to_py)?;
    Ok((cfg, fr_backend))
}

fn parse_sel_t_fdp_config(dict: &Bound<'_, PyDict>) -> PyResult<(TFdpConfig, TFdpBackend2d)> {
    let def = TFdpConfig::default();
    let ibfft_def = IbFftConfig::default();
    let mut backend = "ibfft".to_string();
    let mut iterations = def.iterations;
    let mut learning_rate = def.learning_rate;
    let mut momentum = def.momentum;
    let mut theta = 1.2_f64;
    let mut alpha = def.alpha;
    let mut gamma = def.gamma;
    let mut rho_start = def.rho.start_weight;
    let mut rho_end = def.rho.end_weight;
    let mut rho_power = def.rho.power;
    let mut interpolation_order = ibfft_def.order.order();
    let mut intervals = ibfft_def.intervals;
    let mut min_grid_dim = ibfft_def.min_grid_dim;
    let mut max_grid_dim = ibfft_def.max_grid_dim;
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
                return Err(invalid_input_err(format!(
                    "unknown t_fdp_config key {key:?}"
                )));
            }
        }
    }
    let choice = TFdpBackendChoice::parse(&backend)?;
    let order = InterpolationOrder::from_order(interpolation_order).map_err(core_error_to_py)?;
    let ibfft = IbFftConfig::default()
        .with_order(order)
        .with_intervals(intervals)
        .with_min_grid_dim(min_grid_dim)
        .with_max_grid_dim(max_grid_dim);
    let resolved_backend = build_t_fdp_backend_2d(choice, theta, ibfft)?;
    let rho = RhoSchedule::new()
        .with_start_weight(rho_start)
        .with_end_weight(rho_end)
        .with_power(rho_power);
    let cfg = TFdpConfig::new()
        .with_iterations(iterations)
        .with_learning_rate(learning_rate)
        .with_momentum(momentum)
        .with_alpha(alpha)
        .with_gamma(gamma)
        .with_rho(rho)
        .with_seed(seed);
    cfg.validate().map_err(core_error_to_py)?;
    Ok((cfg, resolved_backend))
}

#[pymethods]
impl PySelLayoutSolver {
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
        let parsed_mapping = parse_sel_weight_mapping(mapping).map_err(invalid_input_err)?;
        let parsed_backend =
            parse_sel_layout_backend(layout_backend).map_err(unsupported_operation_err)?;
        let mut cfg = SelConfig::new()
            .with_tree_depth(tree_depth.0)
            .with_pq_max_pops(pq_max_pops.0)
            .with_mapping(parsed_mapping)
            .with_layout_backend(parsed_backend)
            .with_iterations(iterations.0)
            .with_seed(seed.0);
        if let Some((fr_cfg, fr_b)) = fr_config.map(parse_sel_fr_config).transpose()? {
            cfg = cfg.with_fr_config(Some(fr_cfg)).with_fr_backend(fr_b);
        }
        if let Some((tfdp_cfg, tfdp_b)) = t_fdp_config.map(parse_sel_t_fdp_config).transpose()? {
            cfg = cfg
                .with_t_fdp_config(Some(tfdp_cfg))
                .with_t_fdp_backend(tfdp_b);
        }
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

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    #[pyo3(signature = (), text_signature = "($self)")]
    fn modulated_weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let slice = match &self.inner {
            SelLayoutInner::F64(s) => s.modulated_weights(),
            SelLayoutInner::F32(s) => s.modulated_weights(),
        };
        PyArray1::from_slice(py, slice)
    }

    #[getter]
    fn se_target(&self) -> f64 {
        match &self.inner {
            SelLayoutInner::F64(s) => s.se_target(),
            SelLayoutInner::F32(s) => s.se_target(),
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
            SelLayoutInner::F64(s) => s.num_nodes(),
            SelLayoutInner::F32(s) => s.num_nodes(),
        }
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "SelLayoutSolver(num_nodes={}, se_target={:.4}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.se_target(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}
