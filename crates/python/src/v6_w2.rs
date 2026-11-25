//! Wave 2 (W2-1..W2-4) Python bindings for HDE, Ortmann Sparse Stress, LargeVis,
//! DRGraph, Graph-CNE, Maxent-Stress, GRIP, GEM, SEL, PFS', and SFC layout solvers.

use egraph::{core as egraph_core, layout as egraph_layout, linalg as egraph_linalg};
use egraph_core::{CoreError, CoreResult, IterativeSolver, SolverStatus};
use egraph_layout::{
    DrGraphConfig, GemConfig, GemSolver, GraphCneConfig, GraphCneObjective, GripConfig, GripSolver,
    LargeVisConfig, MaxentStressConfig, MaxentStressSolver, OrtmannPivotStrategy,
    OrtmannSparseStressConfig, PfsPrimeConfig, PfsPrimeOverlapRemoval, SelConfig, SelLayoutBackend,
    SelLayoutSolver, SelWeightMapping, SfcConfig, SfcCurveKind, SfcLayoutSolver, SgdSolver,
};
use egraph_linalg::{
    HdeConfig, hde as linalg_hde, hde_nd as linalg_hde_nd,
    hde_nd_with_pivots as linalg_hde_nd_with_pivots, hde_with_pivots as linalg_hde_with_pivots,
};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::v6_m2::extract_f64_1d;
use super::v6_m3::{
    DualIterativeSolver, PySgdSolver, SgdInner, SolverAction, SolverLayout, execute_solver_action,
    impl_dual_solver, normalize_enum_key, solver_status_str, vec_f64_to_f32_checked,
};
use super::{
    PyCsrGraph, PyLayout2d, PyLayoutNd, ScalarDtype, StrictF64, StrictU64, StrictUsize,
    core_error_to_py, extract_u32_1d,
};

// ---------------------------------------------------------------------------
// Enum parsers (pure Rust returning Result<_, String> for libpython-free tests)
// ---------------------------------------------------------------------------

fn parse_ortmann_pivot_strategy(raw: &str) -> Result<OrtmannPivotStrategy, String> {
    match normalize_enum_key(raw).as_str() {
        "kmeanssp" | "kmeans" => Ok(OrtmannPivotStrategy::KMeansSp),
        "maxminrandomsp" | "maxminrandom" => Ok(OrtmannPivotStrategy::MaxMinRandomSp),
        "maxminsp" | "maxmin" => Ok(OrtmannPivotStrategy::MaxMinSp),
        _ => Err(format!(
            "unknown Ortmann pivot_strategy {raw:?}; expected 'kmeans_sp', 'maxmin_random_sp', or 'maxmin_sp'"
        )),
    }
}

fn parse_graph_cne_objective(raw: &str) -> Result<GraphCneObjective, String> {
    match normalize_enum_key(raw).as_str() {
        "infonce" => Ok(GraphCneObjective::InfoNce),
        "negtsne" => Ok(GraphCneObjective::NegTSne),
        _ => Err(format!(
            "unknown Graph-CNE objective {raw:?}; expected 'infonce' or 'neg_tsne'"
        )),
    }
}

fn parse_sel_weight_mapping(raw: &str) -> Result<SelWeightMapping, String> {
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

fn parse_sel_layout_backend(raw: &str) -> Result<SelLayoutBackend, String> {
    match normalize_enum_key(raw).as_str() {
        "fr" | "fruchtermanreingold" => Ok(SelLayoutBackend::Fr),
        "tfdp" => Ok(SelLayoutBackend::TFdp),
        _ => Err(format!(
            "unknown SEL layout_backend {raw:?}; expected 'fr' or 't_fdp'"
        )),
    }
}

fn parse_sfc_curve_kind(raw: &str) -> Result<SfcCurveKind, String> {
    match normalize_enum_key(raw).as_str() {
        "hilbert" => Ok(SfcCurveKind::Hilbert),
        "morton" | "zorder" => Ok(SfcCurveKind::Morton),
        _ => Err(format!(
            "unknown SFC curve {raw:?}; expected 'hilbert' or 'morton'"
        )),
    }
}

// ---------------------------------------------------------------------------
// W2-1a: High-Dimensional Embedding (`hde`, `hde_nd`)
// ---------------------------------------------------------------------------

/// Compute a 2D `Layout2d` via Harel-Koren High-Dimensional Embedding (`HDE`, 2004).
///
/// Embeds the graph into `dimensions` shortest-path pivot coordinates using MaxMin
/// pivot selection (or explicit `pivots`) and projects via iterative Jacobi/PCA to 2D.
/// Supports `"float64"` (default) and `"float32"` while releasing the Python GIL.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (
        graph,
        *,
        dimensions = StrictUsize(50),
        seed = StrictU64(42),
        tol = StrictF64(1e-3),
        max_sweeps = StrictUsize(100),
        normalize = true,
        pivots = None,
        dtype = ScalarDtype::F64
    ),
    text_signature = "(graph, *, dimensions=50, seed=42, tol=1e-3, max_sweeps=100, normalize=True, pivots=None, dtype='float64')"
)]
fn hde(
    py: Python<'_>,
    graph: &PyCsrGraph,
    dimensions: StrictUsize,
    seed: StrictU64,
    tol: StrictF64,
    max_sweeps: StrictUsize,
    normalize: bool,
    pivots: Option<&Bound<'_, PyAny>>,
    dtype: ScalarDtype,
) -> PyResult<PyLayout2d> {
    let pivots_vec = match pivots {
        Some(obj) if !obj.is_none() => Some(extract_u32_1d(obj, "pivots")?),
        _ => None,
    };
    let cfg = HdeConfig {
        dimensions: dimensions.0,
        output_dims: 2,
        seed: seed.0,
        tol: tol.0,
        max_sweeps: max_sweeps.0,
        normalize,
    };
    let g = graph.inner();
    py.detach(|| match (dtype, pivots_vec.as_deref()) {
        (ScalarDtype::F64, None) => {
            linalg_hde::<f64, f64>(g, &cfg).map(|l| PyLayout2d::from_f64_layout(&l))
        }
        (ScalarDtype::F32, None) => {
            linalg_hde::<f64, f32>(g, &cfg).map(|l| PyLayout2d::from_f32_layout(&l))
        }
        (ScalarDtype::F64, Some(p)) => {
            linalg_hde_with_pivots::<f64, f64>(g, p, &cfg).map(|l| PyLayout2d::from_f64_layout(&l))
        }
        (ScalarDtype::F32, Some(p)) => {
            linalg_hde_with_pivots::<f64, f32>(g, p, &cfg).map(|l| PyLayout2d::from_f32_layout(&l))
        }
    })
    .map_err(core_error_to_py)
}

/// Compute an `output_dims`-dimensional `LayoutNd` via Harel-Koren High-Dimensional Embedding (`HDE`, 2004).
///
/// Embeds the graph into `dimensions` shortest-path pivot coordinates using MaxMin
/// pivot selection (or explicit `pivots`) and projects via iterative Jacobi/PCA to `output_dims`.
/// Supports `"float64"` (default) and `"float32"` while releasing the Python GIL.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (
        graph,
        *,
        dimensions = StrictUsize(50),
        output_dims = StrictUsize(2),
        seed = StrictU64(42),
        tol = StrictF64(1e-3),
        max_sweeps = StrictUsize(100),
        normalize = true,
        pivots = None,
        dtype = ScalarDtype::F64
    ),
    text_signature = "(graph, *, dimensions=50, output_dims=2, seed=42, tol=1e-3, max_sweeps=100, normalize=True, pivots=None, dtype='float64')"
)]
fn hde_nd(
    py: Python<'_>,
    graph: &PyCsrGraph,
    dimensions: StrictUsize,
    output_dims: StrictUsize,
    seed: StrictU64,
    tol: StrictF64,
    max_sweeps: StrictUsize,
    normalize: bool,
    pivots: Option<&Bound<'_, PyAny>>,
    dtype: ScalarDtype,
) -> PyResult<PyLayoutNd> {
    let pivots_vec = match pivots {
        Some(obj) if !obj.is_none() => Some(extract_u32_1d(obj, "pivots")?),
        _ => None,
    };
    let cfg = HdeConfig {
        dimensions: dimensions.0,
        output_dims: output_dims.0,
        seed: seed.0,
        tol: tol.0,
        max_sweeps: max_sweeps.0,
        normalize,
    };
    let g = graph.inner();
    py.detach(|| match (dtype, pivots_vec.as_deref()) {
        (ScalarDtype::F64, None) => linalg_hde_nd::<f64, f64>(g, &cfg).map(PyLayoutNd::from_f64_nd),
        (ScalarDtype::F32, None) => linalg_hde_nd::<f64, f32>(g, &cfg).map(PyLayoutNd::from_f32_nd),
        (ScalarDtype::F64, Some(p)) => {
            linalg_hde_nd_with_pivots::<f64, f64>(g, p, &cfg).map(PyLayoutNd::from_f64_nd)
        }
        (ScalarDtype::F32, Some(p)) => {
            linalg_hde_nd_with_pivots::<f64, f32>(g, p, &cfg).map(PyLayoutNd::from_f32_nd)
        }
    })
    .map_err(core_error_to_py)
}

// ---------------------------------------------------------------------------
// W2-1b & W2-3: SgdSolver constructors (exposed via PySgdSolver in v6_m3.rs)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(super) fn build_ortmann_sparse_stress_solver(
    py: Python<'_>,
    graph: &PyCsrGraph,
    num_pivots: StrictUsize,
    pivot_strategy: &str,
    iterations: StrictUsize,
    eta_max: Option<StrictF64>,
    eta_min: Option<StrictF64>,
    eps: StrictF64,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let strategy = parse_ortmann_pivot_strategy(pivot_strategy).map_err(PyValueError::new_err)?;
    let cfg = OrtmannSparseStressConfig {
        num_pivots: num_pivots.0,
        pivot_strategy: strategy,
        iterations: iterations.0,
        eta_max: eta_max.map(|v| v.0),
        eta_min: eta_min.map(|v| v.0),
        eps: eps.0,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => SgdSolver::<f64>::ortmann_sparse_stress(g, cfg).map(SgdInner::F64),
            ScalarDtype::F32 => SgdSolver::<f32>::ortmann_sparse_stress(g, cfg).map(SgdInner::F32),
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_largevis_solver(
    py: Python<'_>,
    graph: &PyCsrGraph,
    iterations: StrictUsize,
    negative_samples: StrictUsize,
    gamma: StrictF64,
    initial_lr: StrictF64,
    min_lr: StrictF64,
    degree_exponent: StrictF64,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let cfg = LargeVisConfig {
        iterations: iterations.0,
        negative_samples: negative_samples.0,
        gamma: gamma.0,
        initial_lr: initial_lr.0,
        min_lr: min_lr.0,
        degree_exponent: degree_exponent.0,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => SgdSolver::<f64>::largevis(g, cfg).map(SgdInner::F64),
            ScalarDtype::F32 => SgdSolver::<f32>::largevis(g, cfg).map(SgdInner::F32),
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_drgraph_solver(
    py: Python<'_>,
    graph: &PyCsrGraph,
    k_order: StrictUsize,
    negative_samples: StrictUsize,
    b: StrictF64,
    gamma_coarse: StrictF64,
    gamma_fine: StrictF64,
    coarsen_stop_ratio: StrictF64,
    iterations: StrictUsize,
    initial_lr: StrictF64,
    min_lr: StrictF64,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let cfg = DrGraphConfig {
        k_order: k_order.0,
        negative_samples: negative_samples.0,
        b: b.0,
        gamma_coarse: gamma_coarse.0,
        gamma_fine: gamma_fine.0,
        coarsen_stop_ratio: coarsen_stop_ratio.0,
        iterations: iterations.0,
        initial_lr: initial_lr.0,
        min_lr: min_lr.0,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => SgdSolver::<f64>::drgraph(g, cfg).map(SgdInner::F64),
            ScalarDtype::F32 => SgdSolver::<f32>::drgraph(g, cfg).map(SgdInner::F32),
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_graph_cne_solver(
    py: Python<'_>,
    graph: &PyCsrGraph,
    objective: &str,
    negative_samples: StrictUsize,
    iterations: StrictUsize,
    initial_lr: StrictF64,
    min_lr: StrictF64,
    repulsion_weight: StrictF64,
    row_normalize: bool,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let parsed_obj = parse_graph_cne_objective(objective).map_err(PyValueError::new_err)?;
    let cfg = GraphCneConfig {
        objective: parsed_obj,
        negative_samples: negative_samples.0,
        iterations: iterations.0,
        initial_lr: initial_lr.0,
        min_lr: min_lr.0,
        repulsion_weight: repulsion_weight.0,
        row_normalize,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => SgdSolver::<f64>::graph_cne(g, cfg).map(SgdInner::F64),
            ScalarDtype::F32 => SgdSolver::<f32>::graph_cne(g, cfg).map(SgdInner::F32),
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}

// ---------------------------------------------------------------------------
// W2-2: MaxentStressSolver
// ---------------------------------------------------------------------------

impl_dual_solver!(MaxentStressInner, MaxentStressSolver);

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
                    MaxentStressSolver::<f64>::new(g, cfg).map(MaxentStressInner::F64)
                }
                ScalarDtype::F32 => {
                    MaxentStressSolver::<f32>::new(g, cfg).map(MaxentStressInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one Maxent-Stress majorization step in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` Maxent-Stress majorization steps in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run Maxent-Stress optimization to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
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

// ---------------------------------------------------------------------------
// W2-4a: GripSolver
// ---------------------------------------------------------------------------

impl_dual_solver!(GripInner, GripSolver);

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
                ScalarDtype::F64 => GripSolver::<f64>::new(g, cfg).map(GripInner::F64),
                ScalarDtype::F32 => GripSolver::<f32>::new(g, cfg).map(GripInner::F32),
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

// ---------------------------------------------------------------------------
// W2-4b: GemSolver
// ---------------------------------------------------------------------------

impl_dual_solver!(GemInner, GemSolver);

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
            max_rounds: max_rounds.map(|v| v.0),
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

// ---------------------------------------------------------------------------
// W2-4c: SelLayoutSolver
// ---------------------------------------------------------------------------

impl_dual_solver!(SelLayoutInner, SelLayoutSolver);

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

// ---------------------------------------------------------------------------
// W2-4d: PfsPrimeOverlapRemoval
// ---------------------------------------------------------------------------

impl_dual_solver!(PfsPrimeInner, PfsPrimeOverlapRemoval);

/// Cooperative Nachmanson et al. / Chen et al. (2020) `PFS'` dual-axis scan-line node overlap removal solver.
///
/// Removes all axis-aligned rectangle overlaps in one deterministic pass (`total_iterations == 1`)
/// while strictly preserving orthogonal ordering (`oo_nni == 0.0`). Supports `"float64"` and
/// `"float32"` `Layout2d` while releasing the Python GIL.
#[pyclass(name = "PfsPrimeOverlapRemoval", module = "egraph._core")]
pub struct PyPfsPrimeOverlapRemoval {
    inner: PfsPrimeInner,
}

#[pymethods]
impl PyPfsPrimeOverlapRemoval {
    /// Create a `PfsPrimeOverlapRemoval` solver from per-node `half_widths` and `half_heights`.
    #[new]
    #[pyo3(
        signature = (half_widths, half_heights, *, padding = StrictF64(0.0), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(half_widths, half_heights, *, padding=0.0, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        half_widths: &Bound<'_, PyAny>,
        half_heights: &Bound<'_, PyAny>,
        padding: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = PfsPrimeConfig {
            padding: padding.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
        let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    PfsPrimeOverlapRemoval::<f64>::new(hw_f64, hh_f64, cfg).map(PfsPrimeInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
                let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
                py.detach(|| {
                    PfsPrimeOverlapRemoval::<f32>::new(hw_f32, hh_f32, cfg).map(PfsPrimeInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Create a `PfsPrimeOverlapRemoval` solver for square/circular nodes with equal half-widths and half-heights.
    #[staticmethod]
    #[pyo3(
        signature = (radii, *, padding = StrictF64(0.0), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(radii, *, padding=0.0, seed=42, dtype='float64')"
    )]
    fn from_radii(
        py: Python<'_>,
        radii: &Bound<'_, PyAny>,
        padding: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = PfsPrimeConfig {
            padding: padding.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let r_f64 = extract_f64_1d(radii, "radii")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    PfsPrimeOverlapRemoval::<f64>::from_radii(r_f64, cfg).map(PfsPrimeInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let r_f32 = vec_f64_to_f32_checked(&r_f64, "radii")?;
                py.detach(|| {
                    PfsPrimeOverlapRemoval::<f32>::from_radii(r_f32, cfg).map(PfsPrimeInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Execute the single-pass `PFS'` overlap removal step in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` `PFS'` steps in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run `PFS'` overlap removal to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Number of overlapping rectangle pairs remaining after the most recent step.
    #[getter]
    fn last_overlap_count(&self) -> usize {
        match &self.inner {
            PfsPrimeInner::F64(s) => s.last_overlap_count(),
            PfsPrimeInner::F32(s) => s.last_overlap_count(),
        }
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of steps executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Total number of `PFS'` steps (`1`).
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
            PfsPrimeInner::F64(s) => s.num_nodes(),
            PfsPrimeInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `PfsPrimeOverlapRemoval` solver.
    fn __repr__(&self) -> String {
        format!(
            "PfsPrimeOverlapRemoval(num_nodes={}, last_overlap_count={}, status='{}', iteration={}/{}, dtype='{}')",
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
// W2-4e: SfcLayoutSolver
// ---------------------------------------------------------------------------

impl_dual_solver!(SfcLayoutInner, SfcLayoutSolver);

/// Cooperative community-ordered Space-Filling Curve (`SFC`) layout solver.
///
/// Precomputes a hierarchical Louvain community vertex ordering and maps ranks injectively
/// onto a `2^k x 2^k` Hilbert or Morton (`"hilbert"` / `"morton"`) grid inside
/// `[-scale / 2, scale / 2]^2`, guaranteeing zero vertex co-location in both `"float64"`
/// and `"float32"` (`total_iterations == 1`) while releasing the Python GIL.
///
/// Note (`muelder2008_space_filling` `MISSING_PDF`): Muelder & Ma (2008) is marked
/// `MISSING_PDF` in the project literature archive, so exact clustering, ordering, and
/// post-refinement parameters from that paper are not claimed; this solver provides a
/// canonical Louvain community-ordered Hilbert/Morton space-filling curve baseline.
#[pyclass(name = "SfcLayoutSolver", module = "egraph._core")]
pub struct PySfcLayoutSolver {
    inner: SfcLayoutInner,
}

#[pymethods]
impl PySfcLayoutSolver {
    /// Construct an `SfcLayoutSolver` from a `CsrGraph`.
    #[new]
    #[pyo3(
        signature = (graph, *, curve = "hilbert", scale = StrictF64(100.0), resolution_order = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, curve='hilbert', scale=100.0, resolution_order=None, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        curve: &str,
        scale: StrictF64,
        resolution_order: Option<StrictUsize>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let parsed_curve = parse_sfc_curve_kind(curve).map_err(PyValueError::new_err)?;
        let res_order = match resolution_order {
            Some(ord) => Some(u32::try_from(ord.0).map_err(|_| {
                PyValueError::new_err(format!(
                    "resolution_order must be in [1, 15], got {}",
                    ord.0
                ))
            })?),
            None => None,
        };
        let cfg = SfcConfig {
            curve: parsed_curve,
            scale: scale.0,
            resolution_order: res_order,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => SfcLayoutSolver::<f64>::new(g, cfg).map(SfcLayoutInner::F64),
                ScalarDtype::F32 => SfcLayoutSolver::<f32>::new(g, cfg).map(SfcLayoutInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Write the precomputed space-filling curve coordinates into `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` space-filling curve steps in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run SFC coordinate placement to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Return a `uint32` NumPy copy of the community-ordered vertex permutation along the curve.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn vertex_order<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let slice = match &self.inner {
            SfcLayoutInner::F64(s) => s.vertex_order(),
            SfcLayoutInner::F32(s) => s.vertex_order(),
        };
        PyArray1::from_slice(py, slice)
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of steps executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Total number of SFC placement steps (`1`).
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
            SfcLayoutInner::F64(s) => s.num_nodes(),
            SfcLayoutInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `SfcLayoutSolver`.
    fn __repr__(&self) -> String {
        format!(
            "SfcLayoutSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

/// Register Wave 2 classes and functions onto the Python extension module.
pub(super) fn register_w2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyMaxentStressSolver>()?;
    m.add_class::<PyGripSolver>()?;
    m.add_class::<PyGemSolver>()?;
    m.add_class::<PySelLayoutSolver>()?;
    m.add_class::<PyPfsPrimeOverlapRemoval>()?;
    m.add_class::<PySfcLayoutSolver>()?;

    // Functions
    m.add_function(wrap_pyfunction!(hde, m)?)?;
    m.add_function(wrap_pyfunction!(hde_nd, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        GraphCneObjective, OrtmannPivotStrategy, SelLayoutBackend, SelWeightMapping, SfcCurveKind,
        parse_graph_cne_objective, parse_ortmann_pivot_strategy, parse_sel_layout_backend,
        parse_sel_weight_mapping, parse_sfc_curve_kind,
    };

    #[test]
    fn test_w2_enum_parsers() {
        assert_eq!(
            parse_ortmann_pivot_strategy("kmeans_sp").unwrap(),
            OrtmannPivotStrategy::KMeansSp
        );
        assert_eq!(
            parse_ortmann_pivot_strategy("kmeans").unwrap(),
            OrtmannPivotStrategy::KMeansSp
        );
        assert_eq!(
            parse_ortmann_pivot_strategy("maxmin-random-sp").unwrap(),
            OrtmannPivotStrategy::MaxMinRandomSp
        );
        assert_eq!(
            parse_ortmann_pivot_strategy("MaxMin_SP").unwrap(),
            OrtmannPivotStrategy::MaxMinSp
        );
        assert!(parse_ortmann_pivot_strategy("random").is_err());

        assert_eq!(
            parse_graph_cne_objective("infonce").unwrap(),
            GraphCneObjective::InfoNce
        );
        assert_eq!(
            parse_graph_cne_objective("InfoNCE").unwrap(),
            GraphCneObjective::InfoNce
        );
        assert_eq!(
            parse_graph_cne_objective("neg_tsne").unwrap(),
            GraphCneObjective::NegTSne
        );
        assert_eq!(
            parse_graph_cne_objective("neg-tsne").unwrap(),
            GraphCneObjective::NegTSne
        );
        assert!(parse_graph_cne_objective("umap").is_err());

        assert_eq!(
            parse_sel_weight_mapping("sigmoid").unwrap(),
            SelWeightMapping::Sigmoid
        );
        assert_eq!(
            parse_sel_weight_mapping("linear").unwrap(),
            SelWeightMapping::Linear
        );
        assert_eq!(
            parse_sel_weight_mapping("TANH").unwrap(),
            SelWeightMapping::Tanh
        );
        assert_eq!(
            parse_sel_weight_mapping("custom_logistic").unwrap(),
            SelWeightMapping::CustomLogistic
        );
        assert_eq!(
            parse_sel_weight_mapping("logistic").unwrap(),
            SelWeightMapping::CustomLogistic
        );
        assert_eq!(
            parse_sel_weight_mapping("binary").unwrap(),
            SelWeightMapping::Binary
        );
        assert!(parse_sel_weight_mapping("relu").is_err());

        assert_eq!(
            parse_sel_layout_backend("fr").unwrap(),
            SelLayoutBackend::Fr
        );
        assert_eq!(
            parse_sel_layout_backend("fruchterman_reingold").unwrap(),
            SelLayoutBackend::Fr
        );
        assert_eq!(
            parse_sel_layout_backend("t_fdp").unwrap(),
            SelLayoutBackend::TFdp
        );
        assert_eq!(
            parse_sel_layout_backend("t-fdp").unwrap(),
            SelLayoutBackend::TFdp
        );
        assert!(parse_sel_layout_backend("fa2").is_err());

        assert_eq!(
            parse_sfc_curve_kind("hilbert").unwrap(),
            SfcCurveKind::Hilbert
        );
        assert_eq!(
            parse_sfc_curve_kind("Morton").unwrap(),
            SfcCurveKind::Morton
        );
        assert_eq!(
            parse_sfc_curve_kind("z_order").unwrap(),
            SfcCurveKind::Morton
        );
        assert!(parse_sfc_curve_kind("peano").is_err());
    }
}
