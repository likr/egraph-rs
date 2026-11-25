//! M4 v6 cooperative field-layout, compact SGD, continuous FDP, PRISM overlap removal,
//! separation constraint, layering, and FDEB edge bundling bindings backed by `egraph-layout`.

use egraph::{
    core as egraph_core, layout as egraph_layout, neighborhood as egraph_neighborhood,
    spatial as egraph_spatial,
};
use egraph_core::{CoreError, CoreResult, IterativeSolver, SolverStatus};
use egraph_layout::{
    ConstraintProjector, FdebConfig, FdebSolver, FieldLayoutSolver, FieldRepulsionBackend,
    ForceAtlas2Config, ForceAtlas2Solver, LayeringMethod, PacmapGraphConfig, PrismConfig,
    PrismOverlapRemoval, RhoSchedule, SeparationConstraint, Sgd2Config, SgdSolver, SnapTFdpConfig,
    TFdpConfig, YifanHuConfig, YifanHuSolver, assign_layers as layout_assign_layers,
    cycle_edges as layout_cycle_edges,
    generate_layered_constraints as layout_generate_layered_constraints,
    project_clustered_rectangle_no_overlap_2d as layout_project_clustered_rectangle_no_overlap_2d,
    project_rectangle_no_overlap_2d as layout_project_rectangle_no_overlap_2d,
    remove_cycles as layout_remove_cycles,
};
use egraph_neighborhood::{DogVaradhanConfig, KnnGraph};
use egraph_spatial::{IbFftConfig, InterpolationOrder};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use super::v6_m2::{
    PyDogVaradhanLandmarks, PyKnnGraph, PySparseProbabilities, extract_f64_1d, extract_f64_2d,
    extract_usize_1d,
};
use super::v6_m3::{
    DualIterativeSolver, PySgdSolver, SgdInner, SolverAction, SolverLayout, execute_solver_action,
    extract_solver_layout, impl_dual_solver, normalize_enum_key, solver_status_str,
    vec_f64_to_f32_checked, write_back_solver_layout,
};
use super::{
    Layout2dInner, PyCsrGraph, PyLayout2d, ScalarDtype, StrictF64, StrictU64, StrictUsize,
    core_error_to_py, extract_optional_weights_1d, extract_u32_1d, is_non_sequence_container,
};

fn parse_field_backend(raw: &str) -> Result<FieldRepulsionBackend, String> {
    match normalize_enum_key(raw).as_str() {
        "ibfft" | "fft" => Ok(FieldRepulsionBackend::IbFft),
        "barneshut" | "bh" => Ok(FieldRepulsionBackend::BarnesHut),
        "exact" => Ok(FieldRepulsionBackend::Exact),
        _ => Err(format!(
            "unknown field repulsion backend {raw:?}; expected 'ibfft', 'barnes_hut', or 'exact'"
        )),
    }
}

fn parse_layering_method(raw: &str) -> Result<LayeringMethod, String> {
    match normalize_enum_key(raw).as_str() {
        "longestpath" | "lp" => Ok(LayeringMethod::LongestPath),
        "networksimplex" | "ns" | "simplex" => Ok(LayeringMethod::NetworkSimplex),
        _ => Err(format!(
            "unknown layering method {raw:?}; expected 'longest_path' or 'network_simplex'"
        )),
    }
}

impl_dual_solver!(FieldLayoutInner, FieldLayoutSolver);
impl_dual_solver!(ForceAtlas2Inner, ForceAtlas2Solver);
impl_dual_solver!(YifanHuInner, YifanHuSolver);
impl_dual_solver!(PrismInner, PrismOverlapRemoval);

// ---------------------------------------------------------------------------
// 1. TFdpSolver & IbUmapSolver (`FieldLayoutSolver`)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_ibfft_config(
    ibfft_order: usize,
    ibfft_intervals: usize,
    ibfft_min_grid_dim: usize,
    ibfft_max_grid_dim: usize,
) -> PyResult<IbFftConfig> {
    let order = InterpolationOrder::from_order(ibfft_order).map_err(core_error_to_py)?;
    Ok(IbFftConfig {
        order,
        intervals: ibfft_intervals,
        min_grid_dim: ibfft_min_grid_dim,
        max_grid_dim: ibfft_max_grid_dim,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_t_fdp_config(
    backend: &str,
    total_iterations: usize,
    learning_rate: f64,
    momentum: f64,
    theta: f64,
    alpha: f64,
    gamma: f64,
    rho_start: f64,
    rho_end: f64,
    rho_power: f64,
    ibfft_order: usize,
    ibfft_intervals: usize,
    ibfft_min_grid_dim: usize,
    ibfft_max_grid_dim: usize,
    seed: u64,
) -> PyResult<TFdpConfig> {
    let parsed_backend = parse_field_backend(backend).map_err(PyValueError::new_err)?;
    let ibfft = build_ibfft_config(
        ibfft_order,
        ibfft_intervals,
        ibfft_min_grid_dim,
        ibfft_max_grid_dim,
    )?;
    let cfg = TFdpConfig {
        backend: parsed_backend,
        total_iterations,
        learning_rate,
        momentum,
        theta,
        alpha,
        gamma,
        rho: RhoSchedule {
            start_weight: rho_start,
            end_weight: rho_end,
            power: rho_power,
        },
        ibfft,
        seed,
    };
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

#[allow(clippy::too_many_arguments)]
fn build_ib_umap_config(
    backend: &str,
    n_epochs: usize,
    learning_rate: f64,
    min_dist: f64,
    spread: f64,
    a: Option<f64>,
    b: Option<f64>,
    c_k: f64,
    tau_scale: f64,
    hub_damping: bool,
    theta: f64,
    ibfft_order: usize,
    ibfft_intervals: usize,
    ibfft_min_grid_dim: usize,
    ibfft_max_grid_dim: usize,
    seed: u64,
) -> PyResult<egraph_layout::IbUmapConfig> {
    let parsed_backend = parse_field_backend(backend).map_err(PyValueError::new_err)?;
    let ibfft = build_ibfft_config(
        ibfft_order,
        ibfft_intervals,
        ibfft_min_grid_dim,
        ibfft_max_grid_dim,
    )?;
    let cfg = egraph_layout::IbUmapConfig {
        backend: parsed_backend,
        n_epochs,
        learning_rate,
        min_dist,
        spread,
        a,
        b,
        c_k,
        tau_scale,
        hub_damping,
        theta,
        ibfft,
        seed,
    };
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

/// Cooperative `t-FDP` bounded Student-t force-directed placement solver.
///
/// Supports `"ibfft"` (default), `"barnes_hut"`, and `"exact"` repulsion backends in both
/// `"float64"` and `"float32"` precision, releasing the Python GIL during stepping.
#[pyclass(name = "TFdpSolver", module = "egraph._core")]
pub struct PyTFdpSolver {
    inner: FieldLayoutInner,
}

#[pymethods]
impl PyTFdpSolver {
    /// Construct a `TFdpSolver` directly from a `CsrGraph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            backend = "ibfft",
            total_iterations = StrictUsize(200),
            learning_rate = StrictF64(1.0),
            momentum = StrictF64(0.8),
            theta = StrictF64(0.5),
            alpha = StrictF64(1.0),
            gamma = StrictF64(1.0),
            rho_start = StrictF64(1.0),
            rho_end = StrictF64(0.1),
            rho_power = StrictF64(1.0),
            ibfft_order = StrictUsize(1),
            ibfft_intervals = StrictUsize(25),
            ibfft_min_grid_dim = StrictUsize(64),
            ibfft_max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, backend='ibfft', total_iterations=200, learning_rate=1.0, momentum=0.8, theta=0.5, alpha=1.0, gamma=1.0, rho_start=1.0, rho_end=0.1, rho_power=1.0, ibfft_order=1, ibfft_intervals=25, ibfft_min_grid_dim=64, ibfft_max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        backend: &str,
        total_iterations: StrictUsize,
        learning_rate: StrictF64,
        momentum: StrictF64,
        theta: StrictF64,
        alpha: StrictF64,
        gamma: StrictF64,
        rho_start: StrictF64,
        rho_end: StrictF64,
        rho_power: StrictF64,
        ibfft_order: StrictUsize,
        ibfft_intervals: StrictUsize,
        ibfft_min_grid_dim: StrictUsize,
        ibfft_max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_t_fdp_config(
            backend,
            total_iterations.0,
            learning_rate.0,
            momentum.0,
            theta.0,
            alpha.0,
            gamma.0,
            rho_start.0,
            rho_end.0,
            rho_power.0,
            ibfft_order.0,
            ibfft_intervals.0,
            ibfft_min_grid_dim.0,
            ibfft_max_grid_dim.0,
            seed.0,
        )?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    FieldLayoutSolver::<f64>::t_fdp(g, cfg).map(FieldLayoutInner::F64)
                }
                ScalarDtype::F32 => {
                    FieldLayoutSolver::<f32>::t_fdp(g, cfg).map(FieldLayoutInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `TFdpSolver` from either a `KnnGraph` or CSR `(indptr, indices, weights=None)`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            knn_or_indptr,
            indices = None,
            weights = None,
            *,
            backend = "ibfft",
            total_iterations = StrictUsize(200),
            learning_rate = StrictF64(1.0),
            momentum = StrictF64(0.8),
            theta = StrictF64(0.5),
            alpha = StrictF64(1.0),
            gamma = StrictF64(1.0),
            rho_start = StrictF64(1.0),
            rho_end = StrictF64(0.1),
            rho_power = StrictF64(1.0),
            ibfft_order = StrictUsize(1),
            ibfft_intervals = StrictUsize(25),
            ibfft_min_grid_dim = StrictUsize(64),
            ibfft_max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(knn_or_indptr, indices=None, weights=None, *, backend='ibfft', total_iterations=200, learning_rate=1.0, momentum=0.8, theta=0.5, alpha=1.0, gamma=1.0, rho_start=1.0, rho_end=0.1, rho_power=1.0, ibfft_order=1, ibfft_intervals=25, ibfft_min_grid_dim=64, ibfft_max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_knn(
        py: Python<'_>,
        knn_or_indptr: &Bound<'_, PyAny>,
        indices: Option<&Bound<'_, PyAny>>,
        weights: Option<&Bound<'_, PyAny>>,
        backend: &str,
        total_iterations: StrictUsize,
        learning_rate: StrictF64,
        momentum: StrictF64,
        theta: StrictF64,
        alpha: StrictF64,
        gamma: StrictF64,
        rho_start: StrictF64,
        rho_end: StrictF64,
        rho_power: StrictF64,
        ibfft_order: StrictUsize,
        ibfft_intervals: StrictUsize,
        ibfft_min_grid_dim: StrictUsize,
        ibfft_max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_t_fdp_config(
            backend,
            total_iterations.0,
            learning_rate.0,
            momentum.0,
            theta.0,
            alpha.0,
            gamma.0,
            rho_start.0,
            rho_end.0,
            rho_power.0,
            ibfft_order.0,
            ibfft_intervals.0,
            ibfft_min_grid_dim.0,
            ibfft_max_grid_dim.0,
            seed.0,
        )?;

        if let Ok(knn_obj) = knn_or_indptr.cast::<PyKnnGraph>() {
            if indices.is_some_and(|o| !o.is_none()) || weights.is_some_and(|o| !o.is_none()) {
                return Err(PyTypeError::new_err(
                    "TFdpSolver.from_knn(knn, ...) does not accept indices or weights when given a KnnGraph",
                ));
            }
            let knn_borrowed = knn_obj.borrow();
            let knn_ref = knn_borrowed.inner();
            let inner = py
                .detach(|| match dtype {
                    ScalarDtype::F64 => FieldLayoutSolver::<f64>::t_fdp_from_knn(knn_ref, cfg)
                        .map(FieldLayoutInner::F64),
                    ScalarDtype::F32 => FieldLayoutSolver::<f32>::t_fdp_from_knn(knn_ref, cfg)
                        .map(FieldLayoutInner::F32),
                })
                .map_err(core_error_to_py)?;
            return Ok(Self { inner });
        }

        let Some(indices_obj) = indices.filter(|o| !o.is_none()) else {
            return Err(PyTypeError::new_err(
                "TFdpSolver.from_knn expects either a KnnGraph or (indptr, indices, weights=None)",
            ));
        };
        let indptr_vec = extract_usize_1d(knn_or_indptr, "indptr")?;
        if indptr_vec.is_empty() {
            return Err(PyValueError::new_err(
                "indptr must have length num_nodes + 1 (>= 1)",
            ));
        }
        let num_nodes = indptr_vec.len() - 1;
        let indices_vec = extract_u32_1d(indices_obj, "indices")?;
        let distances_vec = match extract_optional_weights_1d(weights, "weights")? {
            Some(w) => w,
            None => vec![0.0_f64; indices_vec.len()],
        };

        let inner = py
            .detach(|| {
                let knn =
                    KnnGraph::new_validated(num_nodes, indptr_vec, indices_vec, distances_vec)?;
                match dtype {
                    ScalarDtype::F64 => FieldLayoutSolver::<f64>::t_fdp_from_knn(&knn, cfg)
                        .map(FieldLayoutInner::F64),
                    ScalarDtype::F32 => FieldLayoutSolver::<f32>::t_fdp_from_knn(&knn, cfg)
                        .map(FieldLayoutInner::F32),
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute a single optimization step in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` optimization iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run optimization to completion in place on `layout` while releasing the GIL.
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

    /// Total scheduled iterations.
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
            FieldLayoutInner::F64(s) => s.num_nodes(),
            FieldLayoutInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Current repulsion schedule multiplier `rho(t)`.
    #[getter]
    fn current_rho(&self) -> f64 {
        match &self.inner {
            FieldLayoutInner::F64(s) => s.current_rho(),
            FieldLayoutInner::F32(s) => s.current_rho(),
        }
    }

    /// Number of attractive edges in the `TFdpSolver`.
    #[getter]
    fn num_edges(&self) -> usize {
        match &self.inner {
            FieldLayoutInner::F64(s) => s.num_edges(),
            FieldLayoutInner::F32(s) => s.num_edges(),
        }
    }

    /// Cumulative number of 2D FFT transforms executed across all steps.
    #[getter]
    fn total_fft_2d_calls(&self) -> u64 {
        match &self.inner {
            FieldLayoutInner::F64(s) => s.total_fft_2d_calls(),
            FieldLayoutInner::F32(s) => s.total_fft_2d_calls(),
        }
    }

    /// Return a concise string representation of the `TFdpSolver`.
    fn __repr__(&self) -> String {
        format!(
            "TFdpSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

/// Cooperative interpolation-based batch Graph UMAP (`ibUMAP`) layout solver.
///
/// Supports `"ibfft"` (default), `"barnes_hut"`, and `"exact"` repulsion backends in both
/// `"float64"` and `"float32"` precision, releasing the Python GIL during stepping.
#[pyclass(name = "IbUmapSolver", module = "egraph._core")]
pub struct PyIbUmapSolver {
    inner: FieldLayoutInner,
}

#[pymethods]
impl PyIbUmapSolver {
    /// Construct an `IbUmapSolver` from a `CsrGraph` using 1+2-hop pairs and regularized DoG Varadhan landmarks.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            max_2hop_per_node = StrictUsize(30),
            degree = None,
            num_landmarks = StrictUsize(64),
            min_degree = StrictUsize(2),
            jl_dim = StrictUsize(64),
            degree_normalize = true,
            p_exp = StrictF64(0.5),
            eps_psd = StrictF64(1e-4),
            t_coarse = StrictF64(48.0),
            t1 = None,
            t2 = None,
            alpha = None,
            beta_floor = None,
            backend = "ibfft",
            n_epochs = StrictUsize(200),
            learning_rate = StrictF64(1.0),
            min_dist = StrictF64(0.1),
            spread = StrictF64(1.0),
            a = None,
            b = None,
            c_k = StrictF64(4.0),
            tau_scale = StrictF64(4.0),
            hub_damping = true,
            theta = StrictF64(0.5),
            ibfft_order = StrictUsize(1),
            ibfft_intervals = StrictUsize(25),
            ibfft_min_grid_dim = StrictUsize(64),
            ibfft_max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, max_2hop_per_node=30, degree=None, num_landmarks=64, min_degree=2, jl_dim=64, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0, t1=None, t2=None, alpha=None, beta_floor=None, backend='ibfft', n_epochs=200, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, c_k=4.0, tau_scale=4.0, hub_damping=True, theta=0.5, ibfft_order=1, ibfft_intervals=25, ibfft_min_grid_dim=64, ibfft_max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        max_2hop_per_node: StrictUsize,
        degree: Option<StrictUsize>,
        num_landmarks: StrictUsize,
        min_degree: StrictUsize,
        jl_dim: StrictUsize,
        degree_normalize: bool,
        p_exp: StrictF64,
        eps_psd: StrictF64,
        t_coarse: StrictF64,
        t1: Option<StrictF64>,
        t2: Option<StrictF64>,
        alpha: Option<StrictF64>,
        beta_floor: Option<StrictF64>,
        backend: &str,
        n_epochs: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        c_k: StrictF64,
        tau_scale: StrictF64,
        hub_damping: bool,
        theta: StrictF64,
        ibfft_order: StrictUsize,
        ibfft_intervals: StrictUsize,
        ibfft_min_grid_dim: StrictUsize,
        ibfft_max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_ib_umap_config(
            backend,
            n_epochs.0,
            learning_rate.0,
            min_dist.0,
            spread.0,
            a.map(|v| v.0),
            b.map(|v| v.0),
            c_k.0,
            tau_scale.0,
            hub_damping,
            theta.0,
            ibfft_order.0,
            ibfft_intervals.0,
            ibfft_min_grid_dim.0,
            ibfft_max_grid_dim.0,
            seed.0,
        )?;
        let dog_cfg = DogVaradhanConfig {
            t1: t1.map(|v| v.0),
            t2: t2.map(|v| v.0),
            alpha: alpha.map(|v| v.0),
            beta_floor: beta_floor.map(|v| v.0),
            degree: degree.map(|v| v.0),
            n_landmarks: num_landmarks.0,
            min_degree: min_degree.0,
            jl_dim: jl_dim.0,
            eps_psd: eps_psd.0,
            degree_normalize,
            p_exp: p_exp.0,
            t_coarse: t_coarse.0,
            seed: seed.0,
            ..DogVaradhanConfig::default()
        };
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    FieldLayoutSolver::<f64>::ib_umap(g, max_2hop_per_node.0, &dog_cfg, cfg)
                        .map(FieldLayoutInner::F64)
                }
                ScalarDtype::F32 => {
                    FieldLayoutSolver::<f32>::ib_umap(g, max_2hop_per_node.0, &dog_cfg, cfg)
                        .map(FieldLayoutInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct an `IbUmapSolver` from precomputed `DogVaradhanLandmarks` and 1+2-hop pairs.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            landmarks,
            max_2hop_per_node = StrictUsize(32),
            *,
            backend = "ibfft",
            n_epochs = StrictUsize(200),
            learning_rate = StrictF64(1.0),
            min_dist = StrictF64(0.1),
            spread = StrictF64(1.0),
            a = None,
            b = None,
            c_k = StrictF64(4.0),
            tau_scale = StrictF64(4.0),
            hub_damping = true,
            theta = StrictF64(0.5),
            ibfft_order = StrictUsize(1),
            ibfft_intervals = StrictUsize(25),
            ibfft_min_grid_dim = StrictUsize(64),
            ibfft_max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, landmarks, max_2hop_per_node=32, *, backend='ibfft', n_epochs=200, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, c_k=4.0, tau_scale=4.0, hub_damping=True, theta=0.5, ibfft_order=1, ibfft_intervals=25, ibfft_min_grid_dim=64, ibfft_max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_landmarks(
        py: Python<'_>,
        graph: &PyCsrGraph,
        landmarks: &PyDogVaradhanLandmarks,
        max_2hop_per_node: StrictUsize,
        backend: &str,
        n_epochs: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        c_k: StrictF64,
        tau_scale: StrictF64,
        hub_damping: bool,
        theta: StrictF64,
        ibfft_order: StrictUsize,
        ibfft_intervals: StrictUsize,
        ibfft_min_grid_dim: StrictUsize,
        ibfft_max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_ib_umap_config(
            backend,
            n_epochs.0,
            learning_rate.0,
            min_dist.0,
            spread.0,
            a.map(|v| v.0),
            b.map(|v| v.0),
            c_k.0,
            tau_scale.0,
            hub_damping,
            theta.0,
            ibfft_order.0,
            ibfft_intervals.0,
            ibfft_min_grid_dim.0,
            ibfft_max_grid_dim.0,
            seed.0,
        )?;
        let g = graph.inner();
        let lm = landmarks.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => FieldLayoutSolver::<f64>::ib_umap_from_landmarks(
                    g,
                    lm,
                    max_2hop_per_node.0,
                    cfg,
                )
                .map(FieldLayoutInner::F64),
                ScalarDtype::F32 => FieldLayoutSolver::<f32>::ib_umap_from_landmarks(
                    g,
                    lm,
                    max_2hop_per_node.0,
                    cfg,
                )
                .map(FieldLayoutInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct an `IbUmapSolver` from a validated `KnnGraph` via `compute_fuzzy_simplicial_set`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            knn,
            *,
            sigma_iters = StrictUsize(64),
            sigma_tol = StrictF64(1e-5),
            backend = "ibfft",
            n_epochs = StrictUsize(200),
            learning_rate = StrictF64(1.0),
            min_dist = StrictF64(0.1),
            spread = StrictF64(1.0),
            a = None,
            b = None,
            c_k = StrictF64(4.0),
            tau_scale = StrictF64(4.0),
            hub_damping = true,
            theta = StrictF64(0.5),
            ibfft_order = StrictUsize(1),
            ibfft_intervals = StrictUsize(25),
            ibfft_min_grid_dim = StrictUsize(64),
            ibfft_max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(knn, *, sigma_iters=64, sigma_tol=1e-5, backend='ibfft', n_epochs=200, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, c_k=4.0, tau_scale=4.0, hub_damping=True, theta=0.5, ibfft_order=1, ibfft_intervals=25, ibfft_min_grid_dim=64, ibfft_max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_knn(
        py: Python<'_>,
        knn: &PyKnnGraph,
        sigma_iters: StrictUsize,
        sigma_tol: StrictF64,
        backend: &str,
        n_epochs: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        c_k: StrictF64,
        tau_scale: StrictF64,
        hub_damping: bool,
        theta: StrictF64,
        ibfft_order: StrictUsize,
        ibfft_intervals: StrictUsize,
        ibfft_min_grid_dim: StrictUsize,
        ibfft_max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_ib_umap_config(
            backend,
            n_epochs.0,
            learning_rate.0,
            min_dist.0,
            spread.0,
            a.map(|v| v.0),
            b.map(|v| v.0),
            c_k.0,
            tau_scale.0,
            hub_damping,
            theta.0,
            ibfft_order.0,
            ibfft_intervals.0,
            ibfft_min_grid_dim.0,
            ibfft_max_grid_dim.0,
            seed.0,
        )?;
        let knn_ref = knn.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => FieldLayoutSolver::<f64>::ib_umap_from_knn(
                    knn_ref,
                    sigma_iters.0,
                    sigma_tol.0,
                    cfg,
                )
                .map(FieldLayoutInner::F64),
                ScalarDtype::F32 => FieldLayoutSolver::<f32>::ib_umap_from_knn(
                    knn_ref,
                    sigma_iters.0,
                    sigma_tol.0,
                    cfg,
                )
                .map(FieldLayoutInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct an `IbUmapSolver` directly from a precomputed fuzzy simplicial set `SparseProbabilities`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            probs,
            *,
            degrees = None,
            backend = "ibfft",
            n_epochs = StrictUsize(200),
            learning_rate = StrictF64(1.0),
            min_dist = StrictF64(0.1),
            spread = StrictF64(1.0),
            a = None,
            b = None,
            c_k = StrictF64(4.0),
            tau_scale = StrictF64(4.0),
            hub_damping = true,
            theta = StrictF64(0.5),
            ibfft_order = StrictUsize(1),
            ibfft_intervals = StrictUsize(25),
            ibfft_min_grid_dim = StrictUsize(64),
            ibfft_max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(probs, *, degrees=None, backend='ibfft', n_epochs=200, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, c_k=4.0, tau_scale=4.0, hub_damping=True, theta=0.5, ibfft_order=1, ibfft_intervals=25, ibfft_min_grid_dim=64, ibfft_max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_fuzzy_set(
        py: Python<'_>,
        probs: &PySparseProbabilities,
        degrees: Option<&Bound<'_, PyAny>>,
        backend: &str,
        n_epochs: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        c_k: StrictF64,
        tau_scale: StrictF64,
        hub_damping: bool,
        theta: StrictF64,
        ibfft_order: StrictUsize,
        ibfft_intervals: StrictUsize,
        ibfft_min_grid_dim: StrictUsize,
        ibfft_max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_ib_umap_config(
            backend,
            n_epochs.0,
            learning_rate.0,
            min_dist.0,
            spread.0,
            a.map(|v| v.0),
            b.map(|v| v.0),
            c_k.0,
            tau_scale.0,
            hub_damping,
            theta.0,
            ibfft_order.0,
            ibfft_intervals.0,
            ibfft_min_grid_dim.0,
            ibfft_max_grid_dim.0,
            seed.0,
        )?;
        let degs_opt = match degrees {
            Some(obj) if !obj.is_none() => Some(extract_u32_1d(obj, "degrees")?),
            _ => None,
        };
        let probs_ref = probs.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => FieldLayoutSolver::<f64>::ib_umap_from_fuzzy_set_ref(
                    probs_ref,
                    degs_opt.as_deref(),
                    cfg,
                )
                .map(FieldLayoutInner::F64),
                ScalarDtype::F32 => FieldLayoutSolver::<f32>::ib_umap_from_fuzzy_set_ref(
                    probs_ref,
                    degs_opt.as_deref(),
                    cfg,
                )
                .map(FieldLayoutInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute a single optimization epoch in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` optimization epochs in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run optimization to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of epochs executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Total scheduled epochs (`n_epochs`).
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
            FieldLayoutInner::F64(s) => s.num_nodes(),
            FieldLayoutInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Current repulsion schedule multiplier (`1.0` for `ibUMAP`).
    #[getter]
    fn current_rho(&self) -> f64 {
        match &self.inner {
            FieldLayoutInner::F64(s) => s.current_rho(),
            FieldLayoutInner::F32(s) => s.current_rho(),
        }
    }

    /// Number of attractive edges in the `IbUmapSolver`.
    #[getter]
    fn num_edges(&self) -> usize {
        match &self.inner {
            FieldLayoutInner::F64(s) => s.num_edges(),
            FieldLayoutInner::F32(s) => s.num_edges(),
        }
    }

    /// Cumulative number of 2D FFT transforms executed across all steps.
    #[getter]
    fn total_fft_2d_calls(&self) -> u64 {
        match &self.inner {
            FieldLayoutInner::F64(s) => s.total_fft_2d_calls(),
            FieldLayoutInner::F32(s) => s.total_fft_2d_calls(),
        }
    }

    /// Fitted or explicit attractive kernel `a` parameter, or `None` if unavailable.
    #[getter]
    fn a(&self) -> Option<f64> {
        match &self.inner {
            FieldLayoutInner::F64(s) => s.ab().map(|(a, _)| a),
            FieldLayoutInner::F32(s) => s.ab().map(|(a, _)| a),
        }
    }

    /// Fitted or explicit attractive kernel `b` parameter, or `None` if unavailable.
    #[getter]
    fn b(&self) -> Option<f64> {
        match &self.inner {
            FieldLayoutInner::F64(s) => s.ab().map(|(_, b)| b),
            FieldLayoutInner::F32(s) => s.ab().map(|(_, b)| b),
        }
    }

    /// Fitted or explicit `(a, b)` kernel parameters, or `None` if unavailable.
    #[getter]
    fn ab(&self) -> Option<(f64, f64)> {
        match &self.inner {
            FieldLayoutInner::F64(s) => s.ab(),
            FieldLayoutInner::F32(s) => s.ab(),
        }
    }

    /// Return a concise string representation of the `IbUmapSolver`.
    fn __repr__(&self) -> String {
        format!(
            "IbUmapSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

// ---------------------------------------------------------------------------
// 2. SgdSolver M4 constructors (`snap_t_fdp`, `pacmap_graph`, `sgd2`)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(super) fn build_snap_t_fdp_solver(
    py: Python<'_>,
    graph: &PyCsrGraph,
    iterations: StrictUsize,
    negatives_per_edge: StrictUsize,
    learning_rate: StrictF64,
    alpha: StrictF64,
    repulsion_scale: StrictF64,
    rho_start: StrictF64,
    rho_end: StrictF64,
    rho_power: StrictF64,
    degree_beta: StrictF64,
    bounded_attraction: bool,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let cfg = SnapTFdpConfig {
        iterations: iterations.0,
        negatives_per_edge: negatives_per_edge.0,
        learning_rate: learning_rate.0,
        alpha: alpha.0,
        repulsion_scale: repulsion_scale.0,
        rho_start: rho_start.0,
        rho_end: rho_end.0,
        rho_power: rho_power.0,
        degree_beta: degree_beta.0,
        bounded_attraction,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => SgdSolver::<f64>::snap_t_fdp(g, cfg).map(SgdInner::F64),
            ScalarDtype::F32 => SgdSolver::<f32>::snap_t_fdp(g, cfg).map(SgdInner::F32),
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_pacmap_graph_solver(
    py: Python<'_>,
    graph: &PyCsrGraph,
    n_neighbors: StrictUsize,
    mn_ratio: StrictF64,
    fp_ratio: StrictF64,
    phase1_iters: StrictUsize,
    phase2_iters: StrictUsize,
    phase3_iters: StrictUsize,
    learning_rate: StrictF64,
    keep_all_1hop: bool,
    max_1hop_per_node: Option<StrictUsize>,
    degree_weight_beta: StrictF64,
    num_landmarks: StrictUsize,
    landmarks: Option<&Bound<'_, PyAny>>,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let cfg = PacmapGraphConfig {
        n_neighbors: n_neighbors.0,
        mn_ratio: mn_ratio.0,
        fp_ratio: fp_ratio.0,
        phase1_iters: phase1_iters.0,
        phase2_iters: phase2_iters.0,
        phase3_iters: phase3_iters.0,
        learning_rate: learning_rate.0,
        seed: seed.0,
        keep_all_1hop,
        max_1hop_per_node: max_1hop_per_node.map(|v| v.0),
        degree_weight_beta: degree_weight_beta.0,
        num_landmarks: num_landmarks.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let inner = match landmarks {
        None => py
            .detach(|| match dtype {
                ScalarDtype::F64 => SgdSolver::<f64>::pacmap_graph(g, cfg).map(SgdInner::F64),
                ScalarDtype::F32 => SgdSolver::<f32>::pacmap_graph(g, cfg).map(SgdInner::F32),
            })
            .map_err(core_error_to_py)?,
        Some(obj) if obj.is_none() => py
            .detach(|| match dtype {
                ScalarDtype::F64 => SgdSolver::<f64>::pacmap_graph(g, cfg).map(SgdInner::F64),
                ScalarDtype::F32 => SgdSolver::<f32>::pacmap_graph(g, cfg).map(SgdInner::F32),
            })
            .map_err(core_error_to_py)?,
        Some(obj) => {
            if let Ok(lm_ref) = obj.extract::<PyRef<'_, PyDogVaradhanLandmarks>>() {
                let lm = lm_ref.inner();
                py.detach(|| match dtype {
                    ScalarDtype::F64 => {
                        SgdSolver::<f64>::pacmap_graph_with_landmarks(g, lm, cfg).map(SgdInner::F64)
                    }
                    ScalarDtype::F32 => {
                        SgdSolver::<f32>::pacmap_graph_with_landmarks(g, lm, cfg).map(SgdInner::F32)
                    }
                })
                .map_err(core_error_to_py)?
            } else {
                let (flat, nrows, ncols) = extract_f64_2d(obj, "landmarks")?;
                if nrows != g.num_nodes() {
                    return Err(PyValueError::new_err(format!(
                        "landmarks row count ({nrows}) must match graph.num_nodes ({})",
                        g.num_nodes()
                    )));
                }
                py.detach(|| match dtype {
                    ScalarDtype::F64 => {
                        SgdSolver::<f64>::pacmap_graph_with_landmark_distances(g, &flat, ncols, cfg)
                            .map(SgdInner::F64)
                    }
                    ScalarDtype::F32 => {
                        SgdSolver::<f32>::pacmap_graph_with_landmark_distances(g, &flat, ncols, cfg)
                            .map(SgdInner::F32)
                    }
                })
                .map_err(core_error_to_py)?
            }
        }
    };
    Ok(PySgdSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_sgd2_solver(
    py: Python<'_>,
    graph: &PyCsrGraph,
    t_max: StrictUsize,
    eps: StrictF64,
    step_scale: StrictF64,
    batch_size: StrictUsize,
    damping: StrictF64,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let cfg = Sgd2Config {
        t_max: t_max.0,
        eps: eps.0,
        step_scale: step_scale.0,
        batch_size: batch_size.0,
        damping: damping.0,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => SgdSolver::<f64>::sgd2(g, cfg).map(SgdInner::F64),
            ScalarDtype::F32 => SgdSolver::<f32>::sgd2(g, cfg).map(SgdInner::F32),
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}

// ---------------------------------------------------------------------------
// 3. ForceAtlas2Solver & YifanHuSolver (`ContinuousFdp`)
// ---------------------------------------------------------------------------

/// Cooperative ForceAtlas2 continuous force-directed layout solver.
///
/// Supports Barnes-Hut quadtree or exact repulsion, LinLog attraction, gravity options,
/// and both `"float64"` and `"float32"` `Layout2d` while releasing the Python GIL.
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
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, iterations=200, theta=1.2, use_barnes_hut=True, linlog_mode=False, strong_gravity=False, gravity=1.0, scaling_ratio=2.0, edge_weight_influence=1.0, outbound_attraction_distribution=False, jitter_tolerance=1.0, max_displacement=None, seed=42, dtype='float64')"
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
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
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
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    ForceAtlas2Solver::<f64>::new(g, cfg).map(ForceAtlas2Inner::F64)
                }
                ScalarDtype::F32 => {
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

// ---------------------------------------------------------------------------
// 4. PrismOverlapRemoval
// ---------------------------------------------------------------------------

/// Cooperative Gansner-Hu (2010) PRISM proximity-stress node overlap removal solver.
///
/// Uses `Delaunay2d` proximity triangulation and sparse CSR Symmetric SSOR-PCG stress majorization
/// in both `"float64"` and `"float32"` precision while releasing the Python GIL.
#[pyclass(name = "PrismOverlapRemoval", module = "egraph._core")]
pub struct PyPrismOverlapRemoval {
    inner: PrismInner,
}

#[pymethods]
impl PyPrismOverlapRemoval {
    /// Create a `PrismOverlapRemoval` solver from per-node `half_widths` and `half_heights`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            half_widths,
            half_heights,
            *,
            max_iterations = StrictUsize(50),
            max_scale_factor = StrictF64(1.5),
            cg_max_iters = StrictUsize(80),
            cg_tolerance = StrictF64(1e-3),
            padding = StrictF64(0.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(half_widths, half_heights, *, max_iterations=50, max_scale_factor=1.5, cg_max_iters=80, cg_tolerance=1e-3, padding=0.0, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        half_widths: &Bound<'_, PyAny>,
        half_heights: &Bound<'_, PyAny>,
        max_iterations: StrictUsize,
        max_scale_factor: StrictF64,
        cg_max_iters: StrictUsize,
        cg_tolerance: StrictF64,
        padding: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = PrismConfig {
            max_iterations: max_iterations.0,
            max_scale_factor: max_scale_factor.0,
            cg_max_iters: cg_max_iters.0,
            cg_tolerance: cg_tolerance.0,
            padding: padding.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
        let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    PrismOverlapRemoval::<f64>::new(hw_f64, hh_f64, cfg).map(PrismInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
                let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
                py.detach(|| {
                    PrismOverlapRemoval::<f32>::new(hw_f32, hh_f32, cfg).map(PrismInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Create a `PrismOverlapRemoval` solver for square/circular nodes with equal half-widths and half-heights.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            radii,
            *,
            max_iterations = StrictUsize(50),
            max_scale_factor = StrictF64(1.5),
            cg_max_iters = StrictUsize(80),
            cg_tolerance = StrictF64(1e-3),
            padding = StrictF64(0.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(radii, *, max_iterations=50, max_scale_factor=1.5, cg_max_iters=80, cg_tolerance=1e-3, padding=0.0, seed=42, dtype='float64')"
    )]
    fn from_radii(
        py: Python<'_>,
        radii: &Bound<'_, PyAny>,
        max_iterations: StrictUsize,
        max_scale_factor: StrictF64,
        cg_max_iters: StrictUsize,
        cg_tolerance: StrictF64,
        padding: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = PrismConfig {
            max_iterations: max_iterations.0,
            max_scale_factor: max_scale_factor.0,
            cg_max_iters: cg_max_iters.0,
            cg_tolerance: cg_tolerance.0,
            padding: padding.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let r_f64 = extract_f64_1d(radii, "radii")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| PrismOverlapRemoval::<f64>::from_radii(r_f64, cfg).map(PrismInner::F64))
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let r_f32 = vec_f64_to_f32_checked(&r_f64, "radii")?;
                py.detach(|| {
                    PrismOverlapRemoval::<f32>::from_radii(r_f32, cfg).map(PrismInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Execute one PRISM overlap removal round in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` PRISM overlap removal rounds in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run PRISM overlap removal until zero overlaps remain or `max_iterations` is reached.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Number of overlapping rectangle pairs remaining after the most recent step.
    #[getter]
    fn last_overlap_count(&self) -> usize {
        match &self.inner {
            PrismInner::F64(s) => s.last_overlap_count(),
            PrismInner::F32(s) => s.last_overlap_count(),
        }
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of PRISM rounds executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured maximum number of PRISM rounds.
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
            PrismInner::F64(s) => s.num_nodes(),
            PrismInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `PrismOverlapRemoval` solver.
    fn __repr__(&self) -> String {
        format!(
            "PrismOverlapRemoval(num_nodes={}, last_overlap_count={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.last_overlap_count(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

// ---------------------------------------------------------------------------
// 5. SeparationConstraints (`SeparationConstraint`, projection & overlap helpers)
// ---------------------------------------------------------------------------

/// One-dimensional separation constraint `coord[left] + gap <= coord[right]` for v6 layouts.
#[pyclass(name = "SeparationConstraint", module = "egraph._core", from_py_object)]
#[derive(Clone, Debug)]
pub struct PySeparationConstraint {
    inner: SeparationConstraint<f64>,
}

impl PySeparationConstraint {
    pub(crate) fn from_inner(inner: SeparationConstraint<f64>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySeparationConstraint {
    /// Construct a separation constraint enforcing `coord[left] + gap <= coord[right]`.
    #[new]
    #[pyo3(signature = (left, right, gap), text_signature = "(left, right, gap)")]
    fn new(left: StrictUsize, right: StrictUsize, gap: StrictF64) -> PyResult<Self> {
        let left_u32 = u32::try_from(left.0)
            .map_err(|_| PyValueError::new_err(format!("left index {} exceeds u32", left.0)))?;
        let right_u32 = u32::try_from(right.0)
            .map_err(|_| PyValueError::new_err(format!("right index {} exceeds u32", right.0)))?;
        if !gap.0.is_finite() {
            return Err(PyValueError::new_err(
                "SeparationConstraint gap must be finite",
            ));
        }
        Ok(Self {
            inner: SeparationConstraint::new(left_u32, right_u32, gap.0),
        })
    }

    /// Left (preceding) variable index.
    #[getter]
    fn left(&self) -> u32 {
        self.inner.left
    }

    /// Right (following) variable index.
    #[getter]
    fn right(&self) -> u32 {
        self.inner.right
    }

    /// Minimum required separation distance `gap`.
    #[getter]
    fn gap(&self) -> f64 {
        self.inner.gap
    }

    /// Return a concise string representation of the `SeparationConstraint`.
    fn __repr__(&self) -> String {
        format!(
            "SeparationConstraint(left={}, right={}, gap={})",
            self.inner.left, self.inner.right, self.inner.gap
        )
    }
}

fn extract_separation_constraints(
    constraints: &Bound<'_, PyAny>,
) -> PyResult<Vec<SeparationConstraint<f64>>> {
    if is_non_sequence_container(constraints) {
        return Err(PyTypeError::new_err(
            "constraints must be a sequence or iterable of SeparationConstraint",
        ));
    }
    let iter = constraints.try_iter().map_err(|_| {
        PyTypeError::new_err("constraints must be a sequence or iterable of SeparationConstraint")
    })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item = item_res?;
        let c = item.extract::<PySeparationConstraint>().map_err(|_| {
            PyTypeError::new_err("constraints entries must be SeparationConstraint instances")
        })?;
        out.push(c.inner);
    }
    Ok(out)
}

fn constraints_to_f32(
    constraints: &[SeparationConstraint<f64>],
) -> PyResult<Vec<SeparationConstraint<f32>>> {
    let mut out = Vec::with_capacity(constraints.len());
    for c in constraints {
        if !c.gap.is_finite() {
            return Err(PyValueError::new_err(
                "non-finite value in SeparationConstraint.gap",
            ));
        }
        let g32 = c.gap as f32;
        if !g32.is_finite() {
            return Err(PyValueError::new_err(format!(
                "SeparationConstraint.gap {} overflows float32",
                c.gap
            )));
        }
        out.push(SeparationConstraint::new(c.left, c.right, g32));
    }
    Ok(out)
}

/// Project 1D coordinates `coords` onto `constraints` while minimizing quadratic displacement.
///
/// If `coords` is a writable `float64` or `float32` NumPy 1D array, updates it in place and
/// returns the projected NumPy array with matching dtype. Otherwise returns a new `float64`
/// NumPy array. Releases the Python GIL during projection.
pub(crate) fn v6_project_1d_impl<'py>(
    py: Python<'py>,
    coords: &Bound<'py, PyAny>,
    constraints: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let c64 = extract_separation_constraints(constraints)?;

    if let Ok(untyped) = coords.cast::<PyUntypedArray>() {
        if untyped.ndim() != 1 {
            return Err(PyValueError::new_err(format!(
                "coords must be a 1D array, got {}D",
                untyped.ndim()
            )));
        }
        if let Ok(arr) = coords.cast::<PyArray1<f64>>() {
            let mut vec: Vec<f64> = {
                let ro = arr
                    .try_readonly()
                    .map_err(|e| PyValueError::new_err(format!("coords borrow error: {e}")))?;
                ro.as_array().iter().copied().collect()
            };
            let n = vec.len();
            py.detach(|| ConstraintProjector::<f64>::new(n).project_1d(&mut vec, &c64))
                .map_err(core_error_to_py)?;
            if let Ok(mut rw) = arr.try_readwrite() {
                for (dst, &src) in rw.as_array_mut().iter_mut().zip(vec.iter()) {
                    *dst = src;
                }
            }
            return Ok(PyArray1::from_vec(py, vec).into_any());
        }
        if let Ok(arr) = coords.cast::<PyArray1<f32>>() {
            let c32 = constraints_to_f32(&c64)?;
            let mut vec: Vec<f32> = {
                let ro = arr
                    .try_readonly()
                    .map_err(|e| PyValueError::new_err(format!("coords borrow error: {e}")))?;
                ro.as_array().iter().copied().collect()
            };
            let n = vec.len();
            py.detach(|| ConstraintProjector::<f32>::new(n).project_1d(&mut vec, &c32))
                .map_err(core_error_to_py)?;
            if let Ok(mut rw) = arr.try_readwrite() {
                for (dst, &src) in rw.as_array_mut().iter_mut().zip(vec.iter()) {
                    *dst = src;
                }
            }
            return Ok(PyArray1::from_vec(py, vec).into_any());
        }
    }

    let mut vec = extract_f64_1d(coords, "coords")?;
    let n = vec.len();
    py.detach(|| ConstraintProjector::<f64>::new(n).project_1d(&mut vec, &c64))
        .map_err(core_error_to_py)?;
    Ok(PyArray1::from_vec(py, vec).into_any())
}

/// Project one axis (`0` for X, `1` for Y) of a `Layout2d` onto `constraints` in place.
///
/// Releases the Python GIL during projection.
#[pyfunction]
#[pyo3(
    signature = (layout, axis, constraints),
    text_signature = "(layout, axis, constraints)"
)]
fn project_layout_axis(
    py: Python<'_>,
    layout: &Bound<'_, PyLayout2d>,
    axis: StrictUsize,
    constraints: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let c64 = extract_separation_constraints(constraints)?;
    let target_dtype = match layout.try_borrow()?.inner() {
        Layout2dInner::F64(_) => ScalarDtype::F64,
        Layout2dInner::F32(_) => ScalarDtype::F32,
    };
    let mut soa = extract_solver_layout(layout, target_dtype)?;
    let axis_val = axis.0;
    match &mut soa {
        SolverLayout::F64(l64) => {
            let n = l64.len();
            py.detach(|| {
                ConstraintProjector::<f64>::new(n).project_layout_axis(l64, axis_val, &c64)
            })
            .map_err(core_error_to_py)?;
        }
        SolverLayout::F32(l32) => {
            let c32 = constraints_to_f32(&c64)?;
            let n = l32.len();
            py.detach(|| {
                ConstraintProjector::<f32>::new(n).project_layout_axis(l32, axis_val, &c32)
            })
            .map_err(core_error_to_py)?;
        }
    }
    write_back_solver_layout(layout, &soa)
}

/// Generate hierarchical separation constraints for directed edges spanning `layers[u] < layers[v]`.
///
/// Releases the Python GIL during constraint generation.
pub(crate) fn v6_generate_layered_constraints_impl(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layers: &Bound<'_, PyAny>,
    min_layer_distance: StrictF64,
) -> PyResult<Vec<PySeparationConstraint>> {
    if !min_layer_distance.0.is_finite() {
        return Err(PyValueError::new_err("min_layer_distance must be finite"));
    }
    let layers_vec = extract_usize_1d(layers, "layers")?;
    let g = graph.inner();
    let dist = min_layer_distance.0;
    let raw = py
        .detach(|| layout_generate_layered_constraints(g, &layers_vec, dist))
        .map_err(core_error_to_py)?;
    Ok(raw
        .into_iter()
        .map(PySeparationConstraint::from_inner)
        .collect())
}

/// Project 2D rectangle positions in `layout` in place to remove all axis-aligned overlaps.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, half_widths, half_heights),
    text_signature = "(layout, half_widths, half_heights)"
)]
fn project_rectangle_no_overlap_2d(
    py: Python<'_>,
    layout: &Bound<'_, PyLayout2d>,
    half_widths: &Bound<'_, PyAny>,
    half_heights: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
    let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
    let target_dtype = match layout.try_borrow()?.inner() {
        Layout2dInner::F64(_) => ScalarDtype::F64,
        Layout2dInner::F32(_) => ScalarDtype::F32,
    };
    let mut soa = extract_solver_layout(layout, target_dtype)?;
    match &mut soa {
        SolverLayout::F64(l64) => py
            .detach(|| layout_project_rectangle_no_overlap_2d(l64, &hw_f64, &hh_f64))
            .map_err(core_error_to_py)?,
        SolverLayout::F32(l32) => {
            let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
            let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
            py.detach(|| layout_project_rectangle_no_overlap_2d(l32, &hw_f32, &hh_f32))
                .map_err(core_error_to_py)?;
        }
    }
    write_back_solver_layout(layout, &soa)
}

/// Remove overlaps between axis-aligned cluster bounding boxes in `layout` in place while
/// preserving intra-cluster relative coordinates.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, cluster_ids, half_widths, half_heights),
    text_signature = "(layout, cluster_ids, half_widths, half_heights)"
)]
fn project_clustered_rectangle_no_overlap_2d(
    py: Python<'_>,
    layout: &Bound<'_, PyLayout2d>,
    cluster_ids: &Bound<'_, PyAny>,
    half_widths: &Bound<'_, PyAny>,
    half_heights: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let cids = extract_u32_1d(cluster_ids, "cluster_ids")?;
    let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
    let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
    let target_dtype = match layout.try_borrow()?.inner() {
        Layout2dInner::F64(_) => ScalarDtype::F64,
        Layout2dInner::F32(_) => ScalarDtype::F32,
    };
    let mut soa = extract_solver_layout(layout, target_dtype)?;
    match &mut soa {
        SolverLayout::F64(l64) => py
            .detach(|| {
                layout_project_clustered_rectangle_no_overlap_2d(l64, &cids, &hw_f64, &hh_f64)
            })
            .map_err(core_error_to_py)?,
        SolverLayout::F32(l32) => {
            let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
            let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
            py.detach(|| {
                layout_project_clustered_rectangle_no_overlap_2d(l32, &cids, &hw_f32, &hh_f32)
            })
            .map_err(core_error_to_py)?;
        }
    }
    write_back_solver_layout(layout, &soa)
}

// ---------------------------------------------------------------------------
// 6. Layering (`cycle_edges`, `remove_cycles`, `assign_layers`)
// ---------------------------------------------------------------------------

/// Identify back-edges `(u, v)` whose reversal eliminates all directed cycles in a `CsrGraph`.
///
/// Returns a `uint32` NumPy array of shape `(M, 2)` and releases the Python GIL.
pub(crate) fn v6_cycle_edges_impl<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let g = graph.inner();
    let pairs = py
        .detach(|| layout_cycle_edges(g))
        .map_err(core_error_to_py)?;
    let m = pairs.len();
    let mut flat = Vec::with_capacity(m * 2);
    for (u, v) in pairs {
        flat.push(u);
        flat.push(v);
    }
    let arr =
        Array2::from_shape_vec((m, 2), flat).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Return a directed acyclic `CsrGraph` by reversing cycle-closing back-edges.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph), text_signature = "(graph)")]
fn remove_cycles(py: Python<'_>, graph: &PyCsrGraph) -> PyResult<PyCsrGraph> {
    let g = graph.inner();
    let dag = py
        .detach(|| layout_remove_cycles(g))
        .map_err(core_error_to_py)?;
    Ok(PyCsrGraph::from_csr_graph(dag))
}

/// Assign hierarchical layer indices (`0, 1, ...`) to all vertices of a directed `CsrGraph`.
///
/// Supports `method="longest_path"` (default) and `method="network_simplex"`.
/// Returns a `uint32` NumPy array of length `graph.num_nodes` and releases the GIL.
#[pyfunction]
#[pyo3(
    signature = (graph, *, method = "longest_path"),
    text_signature = "(graph, *, method='longest_path')"
)]
fn assign_layers<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    method: &str,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let parsed_method = parse_layering_method(method).map_err(PyValueError::new_err)?;
    let g = graph.inner();
    let layers = py
        .detach(|| layout_assign_layers(g, parsed_method))
        .map_err(core_error_to_py)?;
    let layers_u32: Vec<u32> = layers
        .into_iter()
        .map(|l| {
            u32::try_from(l).map_err(|_| CoreError::invalid_input("layer index exceeds u32::MAX"))
        })
        .collect::<CoreResult<Vec<u32>>>()
        .map_err(core_error_to_py)?;
    Ok(PyArray1::from_vec(py, layers_u32))
}

// ---------------------------------------------------------------------------
// 7. FdebSolver
// ---------------------------------------------------------------------------

enum FdebInner {
    F64(FdebSolver<f64>),
    F32(FdebSolver<f32>),
}

impl FdebInner {
    fn dtype(&self) -> ScalarDtype {
        match self {
            Self::F64(_) => ScalarDtype::F64,
            Self::F32(_) => ScalarDtype::F32,
        }
    }

    fn status(&self) -> SolverStatus {
        match self {
            Self::F64(s) => s.status(),
            Self::F32(s) => s.status(),
        }
    }

    fn iteration(&self) -> usize {
        match self {
            Self::F64(s) => s.iteration(),
            Self::F32(s) => s.iteration(),
        }
    }

    fn total_iterations(&self) -> usize {
        match self {
            Self::F64(s) => s.total_iterations(),
            Self::F32(s) => s.total_iterations(),
        }
    }

    fn num_edges(&self) -> usize {
        match self {
            Self::F64(s) => s.num_edges(),
            Self::F32(s) => s.num_edges(),
        }
    }
}

/// Cooperative Force-Directed Edge Bundling (`FDEB`) solver with flat polyline outputs.
///
/// Takes `CsrGraph` and `Layout2d` by reference, supports `"float64"` and `"float32"`,
/// and releases the Python GIL during `step()`, `run_steps()`, and `run()`.
#[pyclass(name = "FdebSolver", module = "egraph._core")]
pub struct PyFdebSolver {
    inner: FdebInner,
}

#[pymethods]
impl PyFdebSolver {
    /// Construct an `FdebSolver` from `graph` and `layout` without taking ownership of `layout`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            layout,
            *,
            cycles = StrictUsize(6),
            s0 = StrictF64(0.1),
            i0 = StrictUsize(90),
            s_step = StrictF64(0.5),
            i_step = StrictF64(2.0 / 3.0),
            spring_k = StrictF64(0.1),
            minimum_edge_compatibility = StrictF64(0.6),
            dtype = None
        ),
        text_signature = "(graph, layout, *, cycles=6, s0=0.1, i0=90, s_step=0.5, i_step=2.0/3.0, spring_k=0.1, minimum_edge_compatibility=0.6, dtype=None)"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        layout: &Bound<'_, PyLayout2d>,
        cycles: StrictUsize,
        s0: StrictF64,
        i0: StrictUsize,
        s_step: StrictF64,
        i_step: StrictF64,
        spring_k: StrictF64,
        minimum_edge_compatibility: StrictF64,
        dtype: Option<ScalarDtype>,
    ) -> PyResult<Self> {
        let cfg = FdebConfig {
            cycles: cycles.0,
            s0: s0.0,
            i0: i0.0,
            s_step: s_step.0,
            i_step: i_step.0,
            spring_k: spring_k.0,
            minimum_edge_compatibility: minimum_edge_compatibility.0,
        };
        cfg.validate().map_err(core_error_to_py)?;

        let resolved_dtype = match dtype {
            Some(dt) => dt,
            None => match layout.try_borrow()?.inner() {
                Layout2dInner::F64(_) => ScalarDtype::F64,
                Layout2dInner::F32(_) => ScalarDtype::F32,
            },
        };
        let soa = extract_solver_layout(layout, resolved_dtype)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match soa {
                SolverLayout::F64(l64) => FdebSolver::<f64>::new(g, &l64, cfg).map(FdebInner::F64),
                SolverLayout::F32(l32) => FdebSolver::<f32>::new(g, &l32, cfg).map(FdebInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute a single FDEB force iteration while releasing the GIL.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn step(&mut self, py: Python<'_>) -> PyResult<&'static str> {
        let status = py
            .detach(|| match &mut self.inner {
                FdebInner::F64(s) => s.step(),
                FdebInner::F32(s) => s.step(),
            })
            .map_err(core_error_to_py)?;
        Ok(solver_status_str(status))
    }

    /// Execute up to `steps` FDEB force iterations while releasing the GIL.
    #[pyo3(signature = (steps), text_signature = "($self, steps)")]
    fn run_steps(&mut self, py: Python<'_>, steps: StrictUsize) -> PyResult<&'static str> {
        let n_steps = steps.0;
        let status = py
            .detach(|| match &mut self.inner {
                FdebInner::F64(s) => s.run_steps(n_steps),
                FdebInner::F32(s) => s.run_steps(n_steps),
            })
            .map_err(core_error_to_py)?;
        Ok(solver_status_str(status))
    }

    /// Run all remaining FDEB subdivision cycles to completion while releasing the GIL.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn run(&mut self, py: Python<'_>) -> PyResult<&'static str> {
        let status = py
            .detach(|| match &mut self.inner {
                FdebInner::F64(s) => s.run(),
                FdebInner::F32(s) => s.run(),
            })
            .map_err(core_error_to_py)?;
        Ok(solver_status_str(status))
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.status())
    }

    /// Number of iterations executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Total scheduled iterations across all subdivision cycles.
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    /// Number of edges being bundled.
    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Scalar coordinate precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a `uint32` NumPy copy of CSR-style polyline offsets of length `num_edges + 1`.
    fn edge_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        match &self.inner {
            FdebInner::F64(s) => PyArray1::from_slice(py, s.edge_offsets()),
            FdebInner::F32(s) => PyArray1::from_slice(py, s.edge_offsets()),
        }
    }

    /// Return a `(K, 2)` NumPy array of all edge polyline points matching `self.dtype`.
    fn points<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            FdebInner::F64(s) => {
                let k = s.points_x().len();
                let interleaved = s.to_interleaved_points();
                let arr = Array2::from_shape_vec((k, 2), interleaved)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyArray2::from_owned_array(py, arr).into_any())
            }
            FdebInner::F32(s) => {
                let k = s.points_x().len();
                let interleaved = s.to_interleaved_points();
                let arr = Array2::from_shape_vec((k, 2), interleaved)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyArray2::from_owned_array(py, arr).into_any())
            }
        }
    }

    /// Return a concise string representation of the `FdebSolver`.
    fn __repr__(&self) -> String {
        format!(
            "FdebSolver(num_edges={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_edges(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

/// Register M4 `egraph-layout` solver, constraint, layering, and FDEB bindings on `egraph._core`.
pub(super) fn register_m4(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTFdpSolver>()?;
    m.add_class::<PyIbUmapSolver>()?;
    m.add_class::<PyForceAtlas2Solver>()?;
    m.add_class::<PyYifanHuSolver>()?;
    m.add_class::<PyPrismOverlapRemoval>()?;
    m.add_class::<PySeparationConstraint>()?;
    m.add_class::<PyFdebSolver>()?;
    m.add_function(wrap_pyfunction!(project_layout_axis, m)?)?;
    m.add_function(wrap_pyfunction!(project_rectangle_no_overlap_2d, m)?)?;
    m.add_function(wrap_pyfunction!(
        project_clustered_rectangle_no_overlap_2d,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(remove_cycles, m)?)?;
    m.add_function(wrap_pyfunction!(assign_layers, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        FieldRepulsionBackend, LayeringMethod, parse_field_backend, parse_layering_method,
    };

    #[test]
    fn test_pure_rust_m4_enum_parsers() {
        assert_eq!(
            parse_field_backend("ibfft").unwrap(),
            FieldRepulsionBackend::IbFft
        );
        assert_eq!(
            parse_field_backend("IB_FFT").unwrap(),
            FieldRepulsionBackend::IbFft
        );
        assert_eq!(
            parse_field_backend("fft").unwrap(),
            FieldRepulsionBackend::IbFft
        );
        assert_eq!(
            parse_field_backend("barnes_hut").unwrap(),
            FieldRepulsionBackend::BarnesHut
        );
        assert_eq!(
            parse_field_backend("bh").unwrap(),
            FieldRepulsionBackend::BarnesHut
        );
        assert_eq!(
            parse_field_backend("EXACT").unwrap(),
            FieldRepulsionBackend::Exact
        );
        assert!(parse_field_backend("quadtree").is_err());

        assert_eq!(
            parse_layering_method("longest_path").unwrap(),
            LayeringMethod::LongestPath
        );
        assert_eq!(
            parse_layering_method("LP").unwrap(),
            LayeringMethod::LongestPath
        );
        assert_eq!(
            parse_layering_method("network_simplex").unwrap(),
            LayeringMethod::NetworkSimplex
        );
        assert_eq!(
            parse_layering_method("Network-Simplex").unwrap(),
            LayeringMethod::NetworkSimplex
        );
        assert_eq!(
            parse_layering_method("ns").unwrap(),
            LayeringMethod::NetworkSimplex
        );
        assert!(parse_layering_method("coffman_graham").is_err());
    }
}
