//! Wave 1 (W1-1..W1-6) Python bindings for Leiden community detection, LinLog/FR layout solvers,
//! Quadrilateral Simmelian Backbone extraction & layout, Cluster-refine & Group-in-a-Box,
//! GTree & FORBID node overlap removal, and `eval-baselines` cluster/separation metrics.

use egraph::{clustering as egraph_clustering, layout as egraph_layout, metrics as egraph_metrics};
use egraph_clustering::{LeidenConfig, LeidenObjective, leiden as clu_leiden};
use egraph_layout::{
    BackboneConfig, BackboneLayoutMode, BackboneResult, BackboneSolver, ClusterRefineConfig,
    ClusterRefineMethod, ClusterRefineSolver, ForbidConfig, ForbidOverlapRemoval, ForbidVariant,
    FrBackend, FruchtermanReingoldConfig, FruchtermanReingoldSolver, GTreeConfig,
    GTreeOverlapRemoval, GibBoxLayoutMode, GibIntraLayoutMode, GroupInABoxConfig,
    GroupInABoxSolver, LinLogConfig, LinLogRepulsion, LinLogSolver,
    extract_quadrilateral_backbone as layout_extract_quadrilateral_backbone,
    refine_cluster_layout as layout_refine_cluster_layout,
    squarified_treemap as layout_squarified_treemap,
};
use egraph_metrics::{NorMetrics, PartitionAgreement};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::array_extract::extract_f64_1d;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_optional_weights_1d,
    extract_u32_1d, normalize_enum_key, pairs_to_u32_array2, solver_status_str,
};
use crate::layout_buffers::{Layout2dInner, PyLayout2d, dispatch_layout2d, layout_f32_to_f64};
use crate::solver_common::{
    DualIterativeSolver, SolverAction, SolverLayout, execute_solver_action, extract_solver_layout,
    impl_dual_solver, vec_f64_to_f32_checked, write_back_solver_layout,
};

// ---------------------------------------------------------------------------
// Enum parsers (pure Rust returning Result<_, String> for libpython-free tests)
// ---------------------------------------------------------------------------

fn parse_leiden_objective(raw: &str) -> Result<LeidenObjective, String> {
    match normalize_enum_key(raw).as_str() {
        "modularity" | "mod" => Ok(LeidenObjective::Modularity),
        "cpm" | "constantpottsmodel" => Ok(LeidenObjective::Cpm),
        _ => Err(format!(
            "unknown Leiden objective {raw:?}; expected 'modularity' or 'cpm'"
        )),
    }
}

fn parse_linlog_repulsion(raw: &str) -> Result<LinLogRepulsion, String> {
    match normalize_enum_key(raw).as_str() {
        "edgerepulsion" | "edge" => Ok(LinLogRepulsion::EdgeRepulsion),
        "noderepulsion" | "node" => Ok(LinLogRepulsion::NodeRepulsion),
        _ => Err(format!(
            "unknown LinLog repulsion_model {raw:?}; expected 'edge_repulsion' or 'node_repulsion'"
        )),
    }
}

fn parse_fr_backend(raw: &str) -> Result<FrBackend, String> {
    match normalize_enum_key(raw).as_str() {
        "exact" => Ok(FrBackend::Exact),
        "barneshut" | "bh" => Ok(FrBackend::BarnesHut),
        "grid" => Ok(FrBackend::Grid),
        _ => Err(format!(
            "unknown Fruchterman-Reingold backend {raw:?}; expected 'exact', 'barnes_hut', or 'grid'"
        )),
    }
}

fn parse_backbone_layout_mode(raw: &str) -> Result<BackboneLayoutMode, String> {
    match normalize_enum_key(raw).as_str() {
        "stress" | "stressmajorization" | "sm" => Ok(BackboneLayoutMode::StressMajorization),
        "sgd" => Ok(BackboneLayoutMode::Sgd),
        _ => Err(format!(
            "unknown backbone layout_mode {raw:?}; expected 'stress' or 'sgd'"
        )),
    }
}

fn parse_cluster_refine_method(raw: &str) -> Result<ClusterRefineMethod, String> {
    match normalize_enum_key(raw).as_str() {
        "kmeans" | "km" => Ok(ClusterRefineMethod::Kmeans),
        "hac" | "upgma" => Ok(ClusterRefineMethod::Hac),
        _ => Err(format!(
            "unknown cluster refine method {raw:?}; expected 'kmeans' or 'hac'"
        )),
    }
}

fn parse_gib_box_mode(raw: &str) -> Result<GibBoxLayoutMode, String> {
    match normalize_enum_key(raw).as_str() {
        "treemap" | "squarifiedtreemap" | "tr" | "trgib" => Ok(GibBoxLayoutMode::SquarifiedTreemap),
        "forcedirected" | "fd" | "fdgib" => Ok(GibBoxLayoutMode::ForceDirected),
        _ => Err(format!(
            "unknown Group-in-a-Box box_mode {raw:?}; expected 'treemap' or 'force_directed'"
        )),
    }
}

fn parse_gib_intra_mode(raw: &str) -> Result<GibIntraLayoutMode, String> {
    match normalize_enum_key(raw).as_str() {
        "fr" | "fruchtermanreingold" => Ok(GibIntraLayoutMode::Fr),
        "sgdfull" | "sgd" => Ok(GibIntraLayoutMode::SgdFull),
        "yifanhu" | "yf" => Ok(GibIntraLayoutMode::YifanHu),
        _ => Err(format!(
            "unknown Group-in-a-Box intra_mode {raw:?}; expected 'fr', 'sgd_full', or 'yifan_hu'"
        )),
    }
}

fn parse_forbid_variant(raw: &str) -> Result<ForbidVariant, String> {
    match normalize_enum_key(raw).as_str() {
        "forbid" => Ok(ForbidVariant::Forbid),
        "forbidprime" | "forbid'" | "forbidp" => Ok(ForbidVariant::ForbidPrime),
        "sparseforbid" | "sparse" => Ok(ForbidVariant::SparseForbid),
        _ => Err(format!(
            "unknown FORBID variant {raw:?}; expected 'forbid', 'forbid_prime', or 'sparse_forbid'"
        )),
    }
}

// ---------------------------------------------------------------------------
// W1-1: Leiden community detection
// ---------------------------------------------------------------------------

/// Detect communities on an undirected `CsrGraph` using the Leiden algorithm (Traag et al., 2019).
///
/// Supports `"modularity"` and `"cpm"` objectives while guaranteeing internally connected
/// communities. Returns a dict with `"labels"`, `"num_communities"`, `"quality"`,
/// `"modularity"`, and `"passes"`. Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (
        graph,
        *,
        objective = "modularity",
        resolution = StrictF64(1.0),
        randomness = StrictF64(0.01),
        max_passes = StrictUsize(100),
        min_quality_gain = StrictF64(1e-7),
        seed = StrictU64(42)
    ),
    text_signature = "(graph, *, objective='modularity', resolution=1.0, randomness=0.01, max_passes=100, min_quality_gain=1e-7, seed=42)"
)]
fn leiden_communities(
    py: Python<'_>,
    graph: &PyCsrGraph,
    objective: &str,
    resolution: StrictF64,
    randomness: StrictF64,
    max_passes: StrictUsize,
    min_quality_gain: StrictF64,
    seed: StrictU64,
) -> PyResult<Py<PyDict>> {
    let parsed_objective = parse_leiden_objective(objective).map_err(PyValueError::new_err)?;
    let cfg = LeidenConfig {
        objective: parsed_objective,
        resolution: resolution.0,
        randomness: randomness.0,
        max_passes: max_passes.0,
        min_quality_gain: min_quality_gain.0,
        seed: seed.0,
    };
    let g = graph.inner();
    let res = py
        .detach(|| clu_leiden(g, &cfg))
        .map_err(core_error_to_py)?;
    let dict = PyDict::new(py);
    dict.set_item("labels", PyArray1::from_vec(py, res.labels))?;
    dict.set_item("num_communities", res.num_communities)?;
    dict.set_item("quality", res.quality)?;
    dict.set_item("modularity", res.modularity)?;
    dict.set_item("passes", res.passes)?;
    Ok(dict.unbind())
}

// ---------------------------------------------------------------------------
// W1-2: LinLogSolver & FruchtermanReingoldSolver
// ---------------------------------------------------------------------------

impl_dual_solver!(LinLogInner, LinLogSolver);
impl_dual_solver!(FrInner, FruchtermanReingoldSolver);

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
                ScalarDtype::F64 => LinLogSolver::<f64>::new(g, cfg).map(LinLogInner::F64),
                ScalarDtype::F32 => LinLogSolver::<f32>::new(g, cfg).map(LinLogInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one LinLog iteration in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` LinLog iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run LinLog optimization to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
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
                ScalarDtype::F64 => FruchtermanReingoldSolver::<f64>::new(g, cfg).map(FrInner::F64),
                ScalarDtype::F32 => FruchtermanReingoldSolver::<f32>::new(g, cfg).map(FrInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one Fruchterman-Reingold iteration in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` Fruchterman-Reingold iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run Fruchterman-Reingold to completion in place on `layout` while releasing the GIL.
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

// ---------------------------------------------------------------------------
// W1-3: Quadrilateral Simmelian Backbone extraction & solver
// ---------------------------------------------------------------------------

impl_dual_solver!(BackboneInner, BackboneSolver);

fn backbone_result_to_py_dict(py: Python<'_>, res: BackboneResult) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("umst_edges", pairs_to_u32_array2(py, res.umst_edges)?)?;
    dict.set_item(
        "backbone_edges",
        pairs_to_u32_array2(py, res.backbone_edges)?,
    )?;
    dict.set_item(
        "simmelian_weights",
        PyArray1::from_vec(py, res.simmelian_weights),
    )?;
    dict.set_item("edge_weights", PyArray1::from_vec(py, res.edge_weights))?;
    dict.set_item("backbone_csr", PyCsrGraph::from_csr_graph(res.backbone_csr))?;
    Ok(dict.unbind())
}

/// Extract the Quadrilateral Simmelian Backbone (Nocaj, Ortmann, & Brandes, 2015) from `graph`.
///
/// Returns a dict with `"umst_edges"`, `"backbone_edges"`, `"simmelian_weights"`,
/// `"edge_weights"`, and `"backbone_csr"`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (
        graph,
        *,
        retain_ratio = StrictF64(0.2),
        umst_weight_scale = StrictF64(1.0),
        non_backbone_weight_scale = StrictF64(0.0)
    ),
    text_signature = "(graph, *, retain_ratio=0.2, umst_weight_scale=1.0, non_backbone_weight_scale=0.0)"
)]
fn extract_quadrilateral_backbone(
    py: Python<'_>,
    graph: &PyCsrGraph,
    retain_ratio: StrictF64,
    umst_weight_scale: StrictF64,
    non_backbone_weight_scale: StrictF64,
) -> PyResult<Py<PyDict>> {
    let cfg = BackboneConfig {
        retain_ratio: retain_ratio.0,
        umst_weight_scale: umst_weight_scale.0,
        non_backbone_weight_scale: non_backbone_weight_scale.0,
        ..BackboneConfig::default()
    };
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let res = py
        .detach(|| layout_extract_quadrilateral_backbone(g, &cfg))
        .map_err(core_error_to_py)?;
    backbone_result_to_py_dict(py, res)
}

/// Cooperative Quadrilateral Simmelian Backbone layout solver (Nocaj et al., 2015).
///
/// Extracts the Simmelian backbone and optimizes node coordinates using `"stress"` or `"sgd"`
/// in either `"float64"` or `"float32"` precision while releasing the Python GIL.
#[pyclass(name = "BackboneSolver", module = "egraph._core")]
pub struct PyBackboneSolver {
    inner: BackboneInner,
}

#[pymethods]
impl PyBackboneSolver {
    /// Construct a `BackboneSolver` by extracting the quadrilateral backbone from `graph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            retain_ratio = StrictF64(0.2),
            umst_weight_scale = StrictF64(1.0),
            non_backbone_weight_scale = StrictF64(0.0),
            layout_mode = "stress",
            iterations = StrictUsize(100),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, retain_ratio=0.2, umst_weight_scale=1.0, non_backbone_weight_scale=0.0, layout_mode='stress', iterations=100, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        retain_ratio: StrictF64,
        umst_weight_scale: StrictF64,
        non_backbone_weight_scale: StrictF64,
        layout_mode: &str,
        iterations: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let mode = parse_backbone_layout_mode(layout_mode).map_err(PyValueError::new_err)?;
        let cfg = BackboneConfig {
            retain_ratio: retain_ratio.0,
            umst_weight_scale: umst_weight_scale.0,
            non_backbone_weight_scale: non_backbone_weight_scale.0,
            layout_mode: mode,
            iterations: iterations.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => BackboneSolver::<f64>::new(g, cfg).map(BackboneInner::F64),
                ScalarDtype::F32 => BackboneSolver::<f32>::new(g, cfg).map(BackboneInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one backbone layout iteration in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` backbone layout iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run backbone layout optimization to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Return the extracted quadrilateral Simmelian backbone dictionary.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn backbone_result(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let res = match &self.inner {
            BackboneInner::F64(s) => s.backbone_result().clone(),
            BackboneInner::F32(s) => s.backbone_result().clone(),
        };
        backbone_result_to_py_dict(py, res)
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

    /// Configured maximum number of layout iterations.
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
            BackboneInner::F64(s) => s.num_nodes(),
            BackboneInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `BackboneSolver`.
    fn __repr__(&self) -> String {
        format!(
            "BackboneSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

// ---------------------------------------------------------------------------
// W1-4: ClusterRefineSolver, refine_cluster_layout, squarified_treemap, GroupInABoxSolver
// ---------------------------------------------------------------------------

impl_dual_solver!(ClusterRefineInner, ClusterRefineSolver);
impl_dual_solver!(GroupInABoxInner, GroupInABoxSolver);

/// Cooperative cluster-faithful layout refiner (`Cluster-kmeans` and `Cluster-HAC`, Cai et al., 2024).
///
/// Supports both `"float64"` and `"float32"` `Layout2d` while releasing the Python GIL.
#[pyclass(name = "ClusterRefineSolver", module = "egraph._core")]
pub struct PyClusterRefineSolver {
    inner: ClusterRefineInner,
}

#[pymethods]
impl PyClusterRefineSolver {
    /// Construct a `ClusterRefineSolver` from per-node `cluster_labels`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            cluster_labels,
            *,
            method = "kmeans",
            max_rounds = StrictUsize(10),
            kmeans_n_init = StrictUsize(5),
            kmeans_max_iters = StrictUsize(100),
            shift_step = StrictF64(1.0),
            margin = StrictF64(0.1),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(cluster_labels, *, method='kmeans', max_rounds=10, kmeans_n_init=5, kmeans_max_iters=100, shift_step=1.0, margin=0.1, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        cluster_labels: &Bound<'_, PyAny>,
        method: &str,
        max_rounds: StrictUsize,
        kmeans_n_init: StrictUsize,
        kmeans_max_iters: StrictUsize,
        shift_step: StrictF64,
        margin: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let parsed_method = parse_cluster_refine_method(method).map_err(PyValueError::new_err)?;
        let labels = extract_u32_1d(cluster_labels, "cluster_labels")?;
        let cfg = ClusterRefineConfig {
            method: parsed_method,
            max_rounds: max_rounds.0,
            kmeans_n_init: kmeans_n_init.0,
            kmeans_max_iters: kmeans_max_iters.0,
            shift_step: shift_step.0,
            margin: margin.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    ClusterRefineSolver::<f64>::new(labels, cfg).map(ClusterRefineInner::F64)
                }
                ScalarDtype::F32 => {
                    ClusterRefineSolver::<f32>::new(labels, cfg).map(ClusterRefineInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `ClusterRefineSolver` using `graph` edges for mean edge length scaling in HAC mode.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            cluster_labels,
            *,
            method = "kmeans",
            max_rounds = StrictUsize(10),
            kmeans_n_init = StrictUsize(5),
            kmeans_max_iters = StrictUsize(100),
            shift_step = StrictF64(1.0),
            margin = StrictF64(0.1),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, cluster_labels, *, method='kmeans', max_rounds=10, kmeans_n_init=5, kmeans_max_iters=100, shift_step=1.0, margin=0.1, seed=42, dtype='float64')"
    )]
    fn from_graph(
        py: Python<'_>,
        graph: &PyCsrGraph,
        cluster_labels: &Bound<'_, PyAny>,
        method: &str,
        max_rounds: StrictUsize,
        kmeans_n_init: StrictUsize,
        kmeans_max_iters: StrictUsize,
        shift_step: StrictF64,
        margin: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let parsed_method = parse_cluster_refine_method(method).map_err(PyValueError::new_err)?;
        let labels = extract_u32_1d(cluster_labels, "cluster_labels")?;
        let cfg = ClusterRefineConfig {
            method: parsed_method,
            max_rounds: max_rounds.0,
            kmeans_n_init: kmeans_n_init.0,
            kmeans_max_iters: kmeans_max_iters.0,
            shift_step: shift_step.0,
            margin: margin.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => ClusterRefineSolver::<f64>::from_graph(g, labels, cfg)
                    .map(ClusterRefineInner::F64),
                ScalarDtype::F32 => ClusterRefineSolver::<f32>::from_graph(g, labels, cfg)
                    .map(ClusterRefineInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one cluster refinement round in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` cluster refinement rounds in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run cluster refinement until convergence or `max_rounds` in place on `layout`.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Number of misplaced vertices shifted in the most recent refinement round.
    #[getter]
    fn last_moved_count(&self) -> usize {
        match &self.inner {
            ClusterRefineInner::F64(s) => s.last_moved_count(),
            ClusterRefineInner::F32(s) => s.last_moved_count(),
        }
    }

    /// Number of distinct ground-truth clusters.
    #[getter]
    fn num_clusters(&self) -> usize {
        match &self.inner {
            ClusterRefineInner::F64(s) => s.num_clusters(),
            ClusterRefineInner::F32(s) => s.num_clusters(),
        }
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of refinement rounds executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured maximum number of refinement rounds.
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
            ClusterRefineInner::F64(s) => s.num_nodes(),
            ClusterRefineInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `ClusterRefineSolver`.
    fn __repr__(&self) -> String {
        format!(
            "ClusterRefineSolver(num_nodes={}, num_clusters={}, last_moved_count={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.num_clusters(),
            self.last_moved_count(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

/// Refine `layout` in place to improve geometric cluster faithfulness and return total vertex shifts.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (
        layout,
        cluster_labels,
        *,
        method = "kmeans",
        max_rounds = StrictUsize(10),
        kmeans_n_init = StrictUsize(5),
        kmeans_max_iters = StrictUsize(100),
        shift_step = StrictF64(1.0),
        margin = StrictF64(0.1),
        seed = StrictU64(42)
    ),
    text_signature = "(layout, cluster_labels, *, method='kmeans', max_rounds=10, kmeans_n_init=5, kmeans_max_iters=100, shift_step=1.0, margin=0.1, seed=42)"
)]
fn refine_cluster_layout(
    py: Python<'_>,
    layout: &Bound<'_, PyLayout2d>,
    cluster_labels: &Bound<'_, PyAny>,
    method: &str,
    max_rounds: StrictUsize,
    kmeans_n_init: StrictUsize,
    kmeans_max_iters: StrictUsize,
    shift_step: StrictF64,
    margin: StrictF64,
    seed: StrictU64,
) -> PyResult<usize> {
    let parsed_method = parse_cluster_refine_method(method).map_err(PyValueError::new_err)?;
    let labels = extract_u32_1d(cluster_labels, "cluster_labels")?;
    let cfg = ClusterRefineConfig {
        method: parsed_method,
        max_rounds: max_rounds.0,
        kmeans_n_init: kmeans_n_init.0,
        kmeans_max_iters: kmeans_max_iters.0,
        shift_step: shift_step.0,
        margin: margin.0,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    let target_dtype = match layout.try_borrow()?.inner() {
        Layout2dInner::F64(_) => ScalarDtype::F64,
        Layout2dInner::F32(_) => ScalarDtype::F32,
    };
    let mut soa = extract_solver_layout(layout, target_dtype)?;
    let total_moved = match &mut soa {
        SolverLayout::F64(l64) => py
            .detach(|| layout_refine_cluster_layout(l64, &labels, &cfg))
            .map_err(core_error_to_py)?,
        SolverLayout::F32(l32) => py
            .detach(|| layout_refine_cluster_layout(l32, &labels, &cfg))
            .map_err(core_error_to_py)?,
    };
    write_back_solver_layout(layout, &soa)?;
    Ok(total_moved)
}

/// Partition rectangle `[x, x + width] x [y, y + height]` into a squarified treemap (Bruls et al., 2000).
///
/// Returns a `(K, 4)` `float64` NumPy array with columns `[x_min, x_max, y_min, y_max]`.
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (weights, *, width = StrictF64(100.0), height = StrictF64(100.0), x = StrictF64(0.0), y = StrictF64(0.0)),
    text_signature = "(weights, *, width=100.0, height=100.0, x=0.0, y=0.0)"
)]
fn squarified_treemap<'py>(
    py: Python<'py>,
    weights: &Bound<'py, PyAny>,
    width: StrictF64,
    height: StrictF64,
    x: StrictF64,
    y: StrictF64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let w_vec = extract_f64_1d(weights, "weights")?;
    let rects = py
        .detach(|| layout_squarified_treemap(&w_vec, x.0, y.0, width.0, height.0))
        .map_err(core_error_to_py)?;
    let k = rects.len();
    let mut flat = Vec::with_capacity(k * 4);
    for rect in rects {
        flat.extend_from_slice(&rect);
    }
    let arr =
        Array2::from_shape_vec((k, 4), flat).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Cooperative Group-in-a-Box (`TR-GIB` / `FD-GIB`, Rodrigues et al., 2011) clustered graph layout solver.
///
/// Partitions the canvas into cluster boxes and runs intra-cluster layouts cooperatively across
/// both `"float64"` and `"float32"` `Layout2d` while releasing the Python GIL.
#[pyclass(name = "GroupInABoxSolver", module = "egraph._core")]
pub struct PyGroupInABoxSolver {
    inner: GroupInABoxInner,
}

#[pymethods]
impl PyGroupInABoxSolver {
    /// Construct a `GroupInABoxSolver` from `graph` and per-node `cluster_labels`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            cluster_labels,
            *,
            box_mode = "treemap",
            intra_mode = "fr",
            total_width = StrictF64(100.0),
            total_height = StrictF64(100.0),
            box_padding = StrictF64(0.05),
            intra_iterations = StrictUsize(100),
            fd_iterations = StrictUsize(100),
            cluster_weights = None,
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, cluster_labels, *, box_mode='treemap', intra_mode='fr', total_width=100.0, total_height=100.0, box_padding=0.05, intra_iterations=100, fd_iterations=100, cluster_weights=None, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        cluster_labels: &Bound<'_, PyAny>,
        box_mode: &str,
        intra_mode: &str,
        total_width: StrictF64,
        total_height: StrictF64,
        box_padding: StrictF64,
        intra_iterations: StrictUsize,
        fd_iterations: StrictUsize,
        cluster_weights: Option<&Bound<'_, PyAny>>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let parsed_box = parse_gib_box_mode(box_mode).map_err(PyValueError::new_err)?;
        let parsed_intra = parse_gib_intra_mode(intra_mode).map_err(PyValueError::new_err)?;
        let labels = extract_u32_1d(cluster_labels, "cluster_labels")?;
        let cw_opt = extract_optional_weights_1d(cluster_weights, "cluster_weights")?;
        let cfg = GroupInABoxConfig {
            box_mode: parsed_box,
            intra_mode: parsed_intra,
            total_width: total_width.0,
            total_height: total_height.0,
            box_padding: box_padding.0,
            intra_iterations: intra_iterations.0,
            fd_iterations: fd_iterations.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match (dtype, cw_opt) {
                (ScalarDtype::F64, Some(cw)) => {
                    GroupInABoxSolver::<f64>::with_cluster_weights(g, labels, cw, cfg)
                        .map(GroupInABoxInner::F64)
                }
                (ScalarDtype::F32, Some(cw)) => {
                    GroupInABoxSolver::<f32>::with_cluster_weights(g, labels, cw, cfg)
                        .map(GroupInABoxInner::F32)
                }
                (ScalarDtype::F64, None) => {
                    GroupInABoxSolver::<f64>::new(g, labels, cfg).map(GroupInABoxInner::F64)
                }
                (ScalarDtype::F32, None) => {
                    GroupInABoxSolver::<f32>::new(g, labels, cfg).map(GroupInABoxInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one Group-in-a-Box intra-cluster iteration in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` Group-in-a-Box iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run Group-in-a-Box intra-cluster optimization to completion in place on `layout`.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Return the computed cluster bounding boxes as a list of dictionaries.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn boxes(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let slice = match &self.inner {
            GroupInABoxInner::F64(s) => s.boxes(),
            GroupInABoxInner::F32(s) => s.boxes(),
        };
        let mut out = Vec::with_capacity(slice.len());
        for b in slice {
            let dict = PyDict::new(py);
            dict.set_item("cluster_id", b.cluster_id)?;
            dict.set_item("x_min", b.x_min)?;
            dict.set_item("x_max", b.x_max)?;
            dict.set_item("y_min", b.y_min)?;
            dict.set_item("y_max", b.y_max)?;
            dict.set_item("node_count", b.node_count)?;
            dict.set_item("weight", b.weight)?;
            out.push(dict.unbind());
        }
        Ok(out)
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of intra-cluster iterations executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured maximum number of intra-cluster iterations.
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
            GroupInABoxInner::F64(s) => s.num_nodes(),
            GroupInABoxInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `GroupInABoxSolver`.
    fn __repr__(&self) -> String {
        format!(
            "GroupInABoxSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

// ---------------------------------------------------------------------------
// W1-5: GTreeOverlapRemoval & ForbidOverlapRemoval
// ---------------------------------------------------------------------------

impl_dual_solver!(GTreeInner, GTreeOverlapRemoval);
impl_dual_solver!(ForbidInner, ForbidOverlapRemoval);

/// Cooperative Nachmanson et al. (2016) GTree proximity-MST node overlap removal solver.
///
/// Supports both `"float64"` and `"float32"` `Layout2d` while releasing the Python GIL.
#[pyclass(name = "GTreeOverlapRemoval", module = "egraph._core")]
pub struct PyGTreeOverlapRemoval {
    inner: GTreeInner,
}

#[pymethods]
impl PyGTreeOverlapRemoval {
    /// Create a `GTreeOverlapRemoval` solver from per-node `half_widths` and `half_heights`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            half_widths,
            half_heights,
            *,
            max_iterations = StrictUsize(20),
            padding = StrictF64(0.0),
            compact_t_max = None,
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(half_widths, half_heights, *, max_iterations=20, padding=0.0, compact_t_max=None, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        half_widths: &Bound<'_, PyAny>,
        half_heights: &Bound<'_, PyAny>,
        max_iterations: StrictUsize,
        padding: StrictF64,
        compact_t_max: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = GTreeConfig {
            max_iterations: max_iterations.0,
            padding: padding.0,
            compact_t_max: compact_t_max.map(|v| v.0),
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
        let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    GTreeOverlapRemoval::<f64>::new(hw_f64, hh_f64, cfg).map(GTreeInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
                let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
                py.detach(|| {
                    GTreeOverlapRemoval::<f32>::new(hw_f32, hh_f32, cfg).map(GTreeInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Create a `GTreeOverlapRemoval` solver for square/circular nodes with equal half-widths and half-heights.
    #[staticmethod]
    #[pyo3(
        signature = (
            radii,
            *,
            max_iterations = StrictUsize(20),
            padding = StrictF64(0.0),
            compact_t_max = None,
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(radii, *, max_iterations=20, padding=0.0, compact_t_max=None, seed=42, dtype='float64')"
    )]
    fn from_radii(
        py: Python<'_>,
        radii: &Bound<'_, PyAny>,
        max_iterations: StrictUsize,
        padding: StrictF64,
        compact_t_max: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = GTreeConfig {
            max_iterations: max_iterations.0,
            padding: padding.0,
            compact_t_max: compact_t_max.map(|v| v.0),
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let r_f64 = extract_f64_1d(radii, "radii")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| GTreeOverlapRemoval::<f64>::from_radii(r_f64, cfg).map(GTreeInner::F64))
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let r_f32 = vec_f64_to_f32_checked(&r_f64, "radii")?;
                py.detach(|| {
                    GTreeOverlapRemoval::<f32>::from_radii(r_f32, cfg).map(GTreeInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Execute one GTree overlap removal round in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` GTree overlap removal rounds in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run GTree overlap removal to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Number of overlapping rectangle pairs remaining after the most recent step.
    #[getter]
    fn last_overlap_count(&self) -> usize {
        match &self.inner {
            GTreeInner::F64(s) => s.last_overlap_count(),
            GTreeInner::F32(s) => s.last_overlap_count(),
        }
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of GTree rounds executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured maximum number of GTree rounds.
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
            GTreeInner::F64(s) => s.num_nodes(),
            GTreeInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `GTreeOverlapRemoval` solver.
    fn __repr__(&self) -> String {
        format!(
            "GTreeOverlapRemoval(num_nodes={}, last_overlap_count={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.last_overlap_count(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

/// Cooperative Giovannangeli et al. (2022) FORBID stochastic gradient descent overlap removal solver.
///
/// Supports `"forbid"`, `"forbid_prime"`, and `"sparse_forbid"` variants across both
/// `"float64"` and `"float32"` `Layout2d` while releasing the Python GIL.
#[pyclass(name = "ForbidOverlapRemoval", module = "egraph._core")]
pub struct PyForbidOverlapRemoval {
    inner: ForbidInner,
}

#[pymethods]
impl PyForbidOverlapRemoval {
    /// Create a `ForbidOverlapRemoval` solver from per-node `half_widths` and `half_heights`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            half_widths,
            half_heights,
            *,
            variant = "forbid",
            max_scale_iters = StrictUsize(20),
            sgd_iterations = StrictUsize(30),
            sparse_k = StrictUsize(30),
            overlap_weight_k = StrictF64(1.0),
            scale_step = StrictF64(0.1),
            padding = StrictF64(0.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(half_widths, half_heights, *, variant='forbid', max_scale_iters=20, sgd_iterations=30, sparse_k=30, overlap_weight_k=1.0, scale_step=0.1, padding=0.0, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        half_widths: &Bound<'_, PyAny>,
        half_heights: &Bound<'_, PyAny>,
        variant: &str,
        max_scale_iters: StrictUsize,
        sgd_iterations: StrictUsize,
        sparse_k: StrictUsize,
        overlap_weight_k: StrictF64,
        scale_step: StrictF64,
        padding: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let parsed_variant = parse_forbid_variant(variant).map_err(PyValueError::new_err)?;
        let cfg = ForbidConfig {
            variant: parsed_variant,
            max_scale_iters: max_scale_iters.0,
            sgd_iterations: sgd_iterations.0,
            sparse_k: sparse_k.0,
            overlap_weight_k: overlap_weight_k.0,
            scale_step: scale_step.0,
            padding: padding.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
        let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    ForbidOverlapRemoval::<f64>::new(hw_f64, hh_f64, cfg).map(ForbidInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
                let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
                py.detach(|| {
                    ForbidOverlapRemoval::<f32>::new(hw_f32, hh_f32, cfg).map(ForbidInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Create a `ForbidOverlapRemoval` solver for square/circular nodes with equal half-widths and half-heights.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            radii,
            *,
            variant = "forbid",
            max_scale_iters = StrictUsize(20),
            sgd_iterations = StrictUsize(30),
            sparse_k = StrictUsize(30),
            overlap_weight_k = StrictF64(1.0),
            scale_step = StrictF64(0.1),
            padding = StrictF64(0.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(radii, *, variant='forbid', max_scale_iters=20, sgd_iterations=30, sparse_k=30, overlap_weight_k=1.0, scale_step=0.1, padding=0.0, seed=42, dtype='float64')"
    )]
    fn from_radii(
        py: Python<'_>,
        radii: &Bound<'_, PyAny>,
        variant: &str,
        max_scale_iters: StrictUsize,
        sgd_iterations: StrictUsize,
        sparse_k: StrictUsize,
        overlap_weight_k: StrictF64,
        scale_step: StrictF64,
        padding: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let parsed_variant = parse_forbid_variant(variant).map_err(PyValueError::new_err)?;
        let cfg = ForbidConfig {
            variant: parsed_variant,
            max_scale_iters: max_scale_iters.0,
            sgd_iterations: sgd_iterations.0,
            sparse_k: sparse_k.0,
            overlap_weight_k: overlap_weight_k.0,
            scale_step: scale_step.0,
            padding: padding.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let r_f64 = extract_f64_1d(radii, "radii")?;
        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    ForbidOverlapRemoval::<f64>::from_radii(r_f64, cfg).map(ForbidInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let r_f32 = vec_f64_to_f32_checked(&r_f64, "radii")?;
                py.detach(|| {
                    ForbidOverlapRemoval::<f32>::from_radii(r_f32, cfg).map(ForbidInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Execute one FORBID binary-search scaling iteration in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` FORBID scaling iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run FORBID overlap removal to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Number of overlapping rectangle pairs remaining after the most recent step.
    #[getter]
    fn last_overlap_count(&self) -> usize {
        match &self.inner {
            ForbidInner::F64(s) => s.last_overlap_count(),
            ForbidInner::F32(s) => s.last_overlap_count(),
        }
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of FORBID scaling iterations executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured maximum number of FORBID scaling iterations.
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
            ForbidInner::F64(s) => s.num_nodes(),
            ForbidInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `ForbidOverlapRemoval` solver.
    fn __repr__(&self) -> String {
        format!(
            "ForbidOverlapRemoval(num_nodes={}, last_overlap_count={}, status='{}', iteration={}/{}, dtype='{}')",
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
// W1-6: Cluster separation, CVI, Sprawlter-NN, and NOR baseline metrics
// ---------------------------------------------------------------------------

fn partition_agreement_to_py_dict(py: Python<'_>, pa: PartitionAgreement) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("ari", pa.ari)?;
    dict.set_item("ami", pa.ami)?;
    dict.set_item("homogeneity", pa.homogeneity)?;
    dict.set_item("completeness", pa.completeness)?;
    dict.set_item("v_measure", pa.v_measure)?;
    dict.set_item("fmi", pa.fmi)?;
    Ok(dict.unbind())
}

fn nor_metrics_to_py_dict(py: Python<'_>, nor: NorMetrics) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("oo_nni", nor.oo_nni)?;
    dict.set_item("sp_ch_a", nor.sp_ch_a)?;
    dict.set_item("gs_bb_iar", nor.gs_bb_iar)?;
    dict.set_item("nm_dm_imse", nor.nm_dm_imse)?;
    dict.set_item("el_rsd", nor.el_rsd)?;
    Ok(dict.unbind())
}

/// Compute partition agreement metrics (`ari`, `ami`, `homogeneity`, `completeness`, `v_measure`, `fmi`).
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (labels_true, labels_pred),
    text_signature = "(labels_true, labels_pred)"
)]
fn partition_agreement(
    py: Python<'_>,
    labels_true: &Bound<'_, PyAny>,
    labels_pred: &Bound<'_, PyAny>,
) -> PyResult<Py<PyDict>> {
    let lt = extract_u32_1d(labels_true, "labels_true")?;
    let lp = extract_u32_1d(labels_pred, "labels_pred")?;
    let pa = py
        .detach(|| egraph_metrics::partition_agreement(&lt, &lp))
        .map_err(core_error_to_py)?;
    partition_agreement_to_py_dict(py, pa)
}

/// Compute K-Means++ Cluster Quality (`CQ-kmeans`) partition agreement on `layout`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, community_labels, *, n_init = StrictUsize(5), max_iters = StrictUsize(100), seed = StrictU64(42)),
    text_signature = "(layout, community_labels, *, n_init=5, max_iters=100, seed=42)"
)]
fn cq_kmeans(
    py: Python<'_>,
    layout: &PyLayout2d,
    community_labels: &Bound<'_, PyAny>,
    n_init: StrictUsize,
    max_iters: StrictUsize,
    seed: StrictU64,
) -> PyResult<Py<PyDict>> {
    let labels = extract_u32_1d(community_labels, "community_labels")?;
    let ni = n_init.0;
    let mi = max_iters.0;
    let s = seed.0;
    let pa = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::cq_kmeans(&soa, &labels, ni, mi, s)
    })?;
    partition_agreement_to_py_dict(py, pa)
}

/// Compute average-linkage HAC Cluster Quality (`CQ-HAC`) partition agreement on `layout`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, community_labels),
    text_signature = "(layout, community_labels)"
)]
fn cq_hac(
    py: Python<'_>,
    layout: &PyLayout2d,
    community_labels: &Bound<'_, PyAny>,
) -> PyResult<Py<PyDict>> {
    let labels = extract_u32_1d(community_labels, "community_labels")?;
    let pa = dispatch_layout2d!(py, layout, |soa| egraph_metrics::cq_hac(&soa, &labels))?;
    partition_agreement_to_py_dict(py, pa)
}

/// Compute Distance Consistency (`DSC`) in `[0.0, 1.0]` for `layout` and discrete `labels`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn distance_consistency(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::distance_consistency(&soa, &lbl)
    })
}

/// Compute Distribution Consistency (`DC`) in `[0.0, 1.0]` on a `grid_size x grid_size` grid.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, grid_size = StrictUsize(20)),
    text_signature = "(layout, labels, grid_size=20)"
)]
fn distribution_consistency(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    grid_size: StrictUsize,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let gs = grid_size.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::distribution_consistency(&soa, &lbl, gs)
    })
}

/// Compute Class Density Measure and 2D Histogram Density Measure `(cdm, hdm)` in `[0.0, 1.0]`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, *, grid_size = StrictUsize(64), bandwidth = None),
    text_signature = "(layout, labels, *, grid_size=64, bandwidth=None)"
)]
fn class_density_measure(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    grid_size: StrictUsize,
    bandwidth: Option<StrictF64>,
) -> PyResult<(f64, f64)> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let gs = grid_size.0;
    let bw = bandwidth.map(|v| v.0);
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::class_density_measure(&soa, &lbl, gs, bw)
    })
}

/// Compute SepMe $k$-NN Graph DIR CPT (`sepme_knng`) in `[0.0, 1.0]`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, k = StrictUsize(2)),
    text_signature = "(layout, labels, k=2)"
)]
fn sepme_knng(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    k: StrictUsize,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let kv = k.0;
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::sepme_knng(&soa, &lbl, kv))
}

/// Compute SepMe Relative Neighborhood Graph DIR CPT (`sepme_rng`) in `[0.0, 1.0]`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn sepme_rng(py: Python<'_>, layout: &PyLayout2d, labels: &Bound<'_, PyAny>) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::sepme_rng(&soa, &lbl))
}

/// Compute SepMe Gabriel Graph DIR CPT (`sepme_gab`) in `[0.0, 1.0]`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn sepme_gab(py: Python<'_>, layout: &PyLayout2d, labels: &Bound<'_, PyAny>) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::sepme_gab(&soa, &lbl))
}

/// Compute SepMe $\gamma$-Observable Neighbor Graph DIR CPT (`sepme_gong`) in `[0.0, 1.0]`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, gamma = StrictF64(0.35)),
    text_signature = "(layout, labels, gamma=0.35)"
)]
fn sepme_gong(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    gamma: StrictF64,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let g = gamma.0;
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::sepme_gong(&soa, &lbl, g))
}

/// Compute Neighborhood Hit (`M_NH`) in `[0.0, 1.0]` over each node's `k` nearest neighbors.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, k = StrictUsize(7)),
    text_signature = "(layout, labels, k=7)"
)]
fn neighborhood_hit(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    k: StrictUsize,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let kv = k.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::neighborhood_hit(&soa, &lbl, kv)
    })
}

/// Compute mean Silhouette coefficient in `[-1.0, 1.0]` for `layout` and `labels`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn silhouette_score(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::silhouette_score(&soa, &lbl)
    })
}

/// Compute Calinski-Harabasz Index (`>= 0.0`) for `layout` and `labels`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn calinski_harabasz_score(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::calinski_harabasz_score(&soa, &lbl)
    })
}

/// Compute Davies-Bouldin Index (`>= 0.0`, lower is better) for `layout` and `labels`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, labels), text_signature = "(layout, labels)")]
fn davies_bouldin_score(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::davies_bouldin_score(&soa, &lbl)
    })
}

/// Compute Pairwise Distance-Overlap Error (`PDOE`) in `[0.0, 1.0]` using `num_bins` histogram bins.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, labels, num_bins = StrictUsize(100)),
    text_signature = "(layout, labels, num_bins=100)"
)]
fn pairwise_distance_overlap_error(
    py: Python<'_>,
    layout: &PyLayout2d,
    labels: &Bound<'_, PyAny>,
    num_bins: StrictUsize,
) -> PyResult<f64> {
    let lbl = extract_u32_1d(labels, "labels")?;
    let bins = num_bins.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::pairwise_distance_overlap_error(&soa, &lbl, bins)
    })
}

/// Compute Sprawlter-NN composite overlap clutter measure (`>= 0.0`).
///
/// When `half_heights` is `None`, treats `half_widths` as uniform per-node radii (`sprawlter_nn_radii`).
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, half_widths, half_heights = None),
    text_signature = "(layout, half_widths, half_heights=None)"
)]
fn sprawlter_nn(
    py: Python<'_>,
    layout: &PyLayout2d,
    half_widths: &Bound<'_, PyAny>,
    half_heights: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
    let hh_opt = match half_heights {
        Some(obj) if !obj.is_none() => Some(extract_f64_1d(obj, "half_heights")?),
        _ => None,
    };
    match layout.inner() {
        Layout2dInner::F64(buf) => {
            let soa = buf.to_layout2d();
            py.detach(|| match hh_opt.as_deref() {
                Some(hh) => egraph_metrics::sprawlter_nn(&soa, &hw_f64, hh),
                None => egraph_metrics::sprawlter_nn_radii(&soa, &hw_f64),
            })
            .map_err(core_error_to_py)
        }
        Layout2dInner::F32(buf) => {
            let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
            let hh_f32_opt = match hh_opt {
                Some(hh) => Some(vec_f64_to_f32_checked(&hh, "half_heights")?),
                None => None,
            };
            let soa = buf.to_layout2d();
            py.detach(|| match hh_f32_opt.as_deref() {
                Some(hh) => egraph_metrics::sprawlter_nn(&soa, &hw_f32, hh),
                None => egraph_metrics::sprawlter_nn_radii(&soa, &hw_f32),
            })
            .map_err(core_error_to_py)
        }
    }
}

/// Compute Chen et al. (2020) Node Overlap Removal (NOR) 5-metric evaluation summary.
///
/// Returns a dict with keys `"oo_nni"`, `"sp_ch_a"`, `"gs_bb_iar"`, `"nm_dm_imse"`, and
/// `"el_rsd"` (`el_rsd` is `0.0` when `graph` is `None`). Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (orig_layout, adjusted_layout, half_widths, half_heights, *, graph = None),
    text_signature = "(orig_layout, adjusted_layout, half_widths, half_heights, *, graph=None)"
)]
fn nor_metrics(
    py: Python<'_>,
    orig_layout: &PyLayout2d,
    adjusted_layout: &PyLayout2d,
    half_widths: &Bound<'_, PyAny>,
    half_heights: &Bound<'_, PyAny>,
    graph: Option<&PyCsrGraph>,
) -> PyResult<Py<PyDict>> {
    let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
    let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
    let g_opt = graph.map(PyCsrGraph::inner);
    let nor = match (orig_layout.inner(), adjusted_layout.inner()) {
        (Layout2dInner::F64(orig), Layout2dInner::F64(adj)) => {
            let o = orig.to_layout2d();
            let a = adj.to_layout2d();
            py.detach(|| match g_opt {
                Some(g) => egraph_metrics::nor_metrics(g, &o, &a, &hw_f64, &hh_f64),
                None => egraph_metrics::nor_metrics_without_graph(&o, &a, &hw_f64, &hh_f64),
            })
            .map_err(core_error_to_py)?
        }
        (Layout2dInner::F32(orig), Layout2dInner::F32(adj)) => {
            let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
            let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
            let o = orig.to_layout2d();
            let a = adj.to_layout2d();
            py.detach(|| match g_opt {
                Some(g) => egraph_metrics::nor_metrics(g, &o, &a, &hw_f32, &hh_f32),
                None => egraph_metrics::nor_metrics_without_graph(&o, &a, &hw_f32, &hh_f32),
            })
            .map_err(core_error_to_py)?
        }
        (Layout2dInner::F64(orig), Layout2dInner::F32(adj)) => {
            let o = orig.to_layout2d();
            let a = layout_f32_to_f64(&adj.to_layout2d());
            py.detach(|| match g_opt {
                Some(g) => egraph_metrics::nor_metrics(g, &o, &a, &hw_f64, &hh_f64),
                None => egraph_metrics::nor_metrics_without_graph(&o, &a, &hw_f64, &hh_f64),
            })
            .map_err(core_error_to_py)?
        }
        (Layout2dInner::F32(orig), Layout2dInner::F64(adj)) => {
            let o = layout_f32_to_f64(&orig.to_layout2d());
            let a = adj.to_layout2d();
            py.detach(|| match g_opt {
                Some(g) => egraph_metrics::nor_metrics(g, &o, &a, &hw_f64, &hh_f64),
                None => egraph_metrics::nor_metrics_without_graph(&o, &a, &hw_f64, &hh_f64),
            })
            .map_err(core_error_to_py)?
        }
    };
    nor_metrics_to_py_dict(py, nor)
}

/// Register Wave 1 classes and functions onto the Python extension module.
pub(super) fn register_w1(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyLinLogSolver>()?;
    m.add_class::<PyFruchtermanReingoldSolver>()?;
    m.add_class::<PyBackboneSolver>()?;
    m.add_class::<PyClusterRefineSolver>()?;
    m.add_class::<PyGroupInABoxSolver>()?;
    m.add_class::<PyGTreeOverlapRemoval>()?;
    m.add_class::<PyForbidOverlapRemoval>()?;

    // Functions
    m.add_function(wrap_pyfunction!(leiden_communities, m)?)?;
    m.add_function(wrap_pyfunction!(extract_quadrilateral_backbone, m)?)?;
    m.add_function(wrap_pyfunction!(refine_cluster_layout, m)?)?;
    m.add_function(wrap_pyfunction!(squarified_treemap, m)?)?;
    m.add_function(wrap_pyfunction!(partition_agreement, m)?)?;
    m.add_function(wrap_pyfunction!(cq_kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(cq_hac, m)?)?;
    m.add_function(wrap_pyfunction!(distance_consistency, m)?)?;
    m.add_function(wrap_pyfunction!(distribution_consistency, m)?)?;
    m.add_function(wrap_pyfunction!(class_density_measure, m)?)?;
    m.add_function(wrap_pyfunction!(sepme_knng, m)?)?;
    m.add_function(wrap_pyfunction!(sepme_rng, m)?)?;
    m.add_function(wrap_pyfunction!(sepme_gab, m)?)?;
    m.add_function(wrap_pyfunction!(sepme_gong, m)?)?;
    m.add_function(wrap_pyfunction!(neighborhood_hit, m)?)?;
    m.add_function(wrap_pyfunction!(silhouette_score, m)?)?;
    m.add_function(wrap_pyfunction!(calinski_harabasz_score, m)?)?;
    m.add_function(wrap_pyfunction!(davies_bouldin_score, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_distance_overlap_error, m)?)?;
    m.add_function(wrap_pyfunction!(sprawlter_nn, m)?)?;
    m.add_function(wrap_pyfunction!(nor_metrics, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        BackboneLayoutMode, ClusterRefineMethod, ForbidVariant, FrBackend, GibBoxLayoutMode,
        GibIntraLayoutMode, LeidenObjective, LinLogRepulsion, parse_backbone_layout_mode,
        parse_cluster_refine_method, parse_forbid_variant, parse_fr_backend, parse_gib_box_mode,
        parse_gib_intra_mode, parse_leiden_objective, parse_linlog_repulsion,
    };

    #[test]
    fn test_w1_enum_parsers() {
        assert_eq!(
            parse_leiden_objective("modularity").unwrap(),
            LeidenObjective::Modularity
        );
        assert_eq!(
            parse_leiden_objective("Mod").unwrap(),
            LeidenObjective::Modularity
        );
        assert_eq!(parse_leiden_objective("CPM").unwrap(), LeidenObjective::Cpm);
        assert!(parse_leiden_objective("louvain").is_err());

        assert_eq!(
            parse_linlog_repulsion("edge_repulsion").unwrap(),
            LinLogRepulsion::EdgeRepulsion
        );
        assert_eq!(
            parse_linlog_repulsion("node").unwrap(),
            LinLogRepulsion::NodeRepulsion
        );
        assert!(parse_linlog_repulsion("invalid").is_err());

        assert_eq!(parse_fr_backend("exact").unwrap(), FrBackend::Exact);
        assert_eq!(
            parse_fr_backend("barnes-hut").unwrap(),
            FrBackend::BarnesHut
        );
        assert_eq!(parse_fr_backend("bh").unwrap(), FrBackend::BarnesHut);
        assert_eq!(parse_fr_backend("GRID").unwrap(), FrBackend::Grid);
        assert!(parse_fr_backend("fft").is_err());

        assert_eq!(
            parse_backbone_layout_mode("stress").unwrap(),
            BackboneLayoutMode::StressMajorization
        );
        assert_eq!(
            parse_backbone_layout_mode("stress_majorization").unwrap(),
            BackboneLayoutMode::StressMajorization
        );
        assert_eq!(
            parse_backbone_layout_mode("sgd").unwrap(),
            BackboneLayoutMode::Sgd
        );
        assert!(parse_backbone_layout_mode("fr").is_err());

        assert_eq!(
            parse_cluster_refine_method("k-means").unwrap(),
            ClusterRefineMethod::Kmeans
        );
        assert_eq!(
            parse_cluster_refine_method("HAC").unwrap(),
            ClusterRefineMethod::Hac
        );
        assert!(parse_cluster_refine_method("spectral").is_err());

        assert_eq!(
            parse_gib_box_mode("treemap").unwrap(),
            GibBoxLayoutMode::SquarifiedTreemap
        );
        assert_eq!(
            parse_gib_box_mode("tr-gib").unwrap(),
            GibBoxLayoutMode::SquarifiedTreemap
        );
        assert_eq!(
            parse_gib_box_mode("force_directed").unwrap(),
            GibBoxLayoutMode::ForceDirected
        );
        assert_eq!(
            parse_gib_box_mode("fd_gib").unwrap(),
            GibBoxLayoutMode::ForceDirected
        );
        assert!(parse_gib_box_mode("circle").is_err());

        assert_eq!(parse_gib_intra_mode("fr").unwrap(), GibIntraLayoutMode::Fr);
        assert_eq!(
            parse_gib_intra_mode("sgd_full").unwrap(),
            GibIntraLayoutMode::SgdFull
        );
        assert_eq!(
            parse_gib_intra_mode("yifan_hu").unwrap(),
            GibIntraLayoutMode::YifanHu
        );
        assert!(parse_gib_intra_mode("tsnet").is_err());

        assert_eq!(
            parse_forbid_variant("forbid").unwrap(),
            ForbidVariant::Forbid
        );
        assert_eq!(
            parse_forbid_variant("forbid_prime").unwrap(),
            ForbidVariant::ForbidPrime
        );
        assert_eq!(
            parse_forbid_variant("forbid'").unwrap(),
            ForbidVariant::ForbidPrime
        );
        assert_eq!(
            parse_forbid_variant("sparse_forbid").unwrap(),
            ForbidVariant::SparseForbid
        );
        assert!(parse_forbid_variant("prism").is_err());
    }
}
