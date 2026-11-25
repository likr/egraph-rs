//! M3 v6 WebAssembly bindings for `egraph-clustering` and `egraph-layout`.
//!
//! Exposes pure-Rust graph clustering (`KMeans++`, `Spectral`, `Louvain`, `LabelPropagation`,
//! coarsening/projection), non-Euclidean `ManifoldLayout2d` (`f32`), and all 6 cooperative
//! `f32` layout solvers (`TsNetSolver`, `SgdSolver`, `GuMapSolver`, `SpatialIntruderRefiner`,
//! `StressMajorizationSolver`, `KamadaKawaiSolver`) mutating `Layout2d` buffers in place
//! without `serde`, `getrandom`, or threading dependencies.

use egraph::clustering::{
    KMeansConfig, KMeansResult, LabelPropagationConfig, LouvainConfig, LouvainResult,
    SpectralClusteringConfig, coarsen_graph, kmeans_plus_plus, label_propagation, louvain,
    project_clustered_layout, spectral_clustering,
};
use egraph::core::{
    CoreError, CsrGraph, IterativeSolver, Layout2d, ManifoldKind, ManifoldLayout2d, SolverStatus,
    poincare_distance, poincare_mobius_add, spherical_distance, torus_distance, wrap_torus_point,
};
use egraph::layout::{
    BarrierSymmetry, GuMapConfig, GuMapSolver, KamadaKawaiConfig, KamadaKawaiSolver,
    SgdScheduleConfig, SgdScheduleKind, SgdSolver, SparseBarrierConfig, SparseBarrierStats,
    SpatialIntruderRefiner, SpatialRefinerConfig, SpatialRefinerRoundStats, SpatialRefinerStats,
    StreamingNegativeConfig, StressMajorizationConfig, StressMajorizationSolver, TsNetBackend,
    TsNetConfig, TsNetSolver,
};
use egraph::neighborhood::{DogVaradhanConfig, DogVaradhanStats, SymmetrizationMode};
use egraph::spatial::{FitFieldStats, InterpolationOrder};
use wasm_bindgen::prelude::*;

use crate::v6::{
    JsCsrGraph, JsLayout2d, core_err_to_js, ffi_err_to_js, parse_non_negative_usize, parse_seed_u64,
};
use crate::v6_m2::{
    JsDogVaradhanLandmarks, JsKnnGraph, JsSparseProbabilities, JsSpectralMomentStats,
    parse_opt_usize_param, parse_optional_usize,
};

const DEFAULT_KMEANS_K: usize = 2;
const DEFAULT_KMEANS_MAX_ITERS: usize = 100;
const DEFAULT_KMEANS_TOL: f64 = 1e-4;
const DEFAULT_KMEANS_N_INIT: usize = 1;

const DEFAULT_SPECTRAL_K: usize = 2;
const DEFAULT_SPECTRAL_MAX_EIGEN_ITERS: usize = 300;
const DEFAULT_SPECTRAL_EIGEN_TOL: f64 = 1e-4;

const DEFAULT_LOUVAIN_MAX_PASSES: usize = 100;
const DEFAULT_LOUVAIN_RESOLUTION: f64 = 1.0;
const DEFAULT_LOUVAIN_MIN_GAIN: f64 = 1e-7;

const DEFAULT_LPA_MAX_ITERS: usize = 100;
const DEFAULT_SPREAD_RADIUS: f64 = 0.1;

const DEFAULT_UNIFORM_PERPLEXITY: f64 = 30.0;
const DEFAULT_RANDOM_PAIRS_PER_NODE: usize = 30;
const DEFAULT_OMEGA_MIN_DIST: f64 = 1e-3;
const DEFAULT_GUMAP_MAX_2HOP: usize = 32;

pub(crate) fn normalize_enum_token(raw: &str) -> String {
    raw.chars()
        .filter(|ch| !matches!(ch, '_' | '-' | ' '))
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

pub(crate) fn solver_status_str(status: SolverStatus) -> &'static str {
    match status {
        SolverStatus::Initialized => "initialized",
        SolverStatus::Running => "running",
        SolverStatus::Converged => "converged",
        SolverStatus::Completed => "completed",
    }
}

fn parse_manifold_kind(raw: &str) -> Result<ManifoldKind, JsError> {
    match normalize_enum_token(raw).as_str() {
        "hyperbolic" | "hyperbolicpoincare" | "poincare" => Ok(ManifoldKind::HyperbolicPoincare),
        "spherical" | "sphere" => Ok(ManifoldKind::Spherical),
        "torus" | "torus2d" => Ok(ManifoldKind::Torus2d),
        _ => Err(JsError::new(&format!(
            "unknown manifold kind {raw:?}; expected 'hyperbolic', 'spherical', or 'torus'"
        ))),
    }
}

fn manifold_kind_str(kind: ManifoldKind) -> &'static str {
    match kind {
        ManifoldKind::HyperbolicPoincare => "hyperbolic",
        ManifoldKind::Spherical => "spherical",
        ManifoldKind::Torus2d => "torus",
    }
}

fn parse_tsnet_backend(raw: &str) -> Result<TsNetBackend, JsError> {
    match normalize_enum_token(raw).as_str() {
        "exact" => Ok(TsNetBackend::Exact),
        "barneshut" | "bh" => Ok(TsNetBackend::BarnesHut),
        "fft" | "fftinterpolation" | "fit" => Ok(TsNetBackend::FftInterpolation),
        "linearfft" | "l" => Ok(TsNetBackend::LinearFft),
        _ => Err(JsError::new(&format!(
            "unknown TsNetBackend {raw:?}; expected 'exact', 'barnes_hut', 'fft', or 'linear_fft'"
        ))),
    }
}

fn parse_sgd_schedule_kind(raw: &str) -> Result<SgdScheduleKind, JsError> {
    match normalize_enum_token(raw).as_str() {
        "exponential" => Ok(SgdScheduleKind::Exponential),
        "linear" => Ok(SgdScheduleKind::Linear),
        "quadratic" => Ok(SgdScheduleKind::Quadratic),
        "reciprocal" => Ok(SgdScheduleKind::Reciprocal),
        "constant" => Ok(SgdScheduleKind::Constant),
        "squared" => Ok(SgdScheduleKind::Squared),
        _ => Err(JsError::new(&format!(
            "unknown SgdScheduleKind {raw:?}; expected 'exponential', 'linear', 'quadratic', 'reciprocal', 'constant', or 'squared'"
        ))),
    }
}

fn parse_barrier_symmetry(raw: &str) -> Result<BarrierSymmetry, JsError> {
    match normalize_enum_token(raw).as_str() {
        "symmax" => Ok(BarrierSymmetry::SymMax),
        "symavg" => Ok(BarrierSymmetry::SymAvg),
        _ => Err(JsError::new(&format!(
            "unknown BarrierSymmetry {raw:?}; expected 'sym_max' or 'sym_avg'"
        ))),
    }
}

pub(crate) fn parse_ibfft_order(val: &JsValue) -> Result<InterpolationOrder, JsError> {
    if let Some(num) = val.as_f64() {
        let ord = parse_non_negative_usize(num, "ibfftOrder")?;
        return InterpolationOrder::from_order(ord).map_err(core_err_to_js);
    }
    if let Some(raw) = val.as_string() {
        return match normalize_enum_token(&raw).as_str() {
            "bilinear" | "1" => Ok(InterpolationOrder::Bilinear),
            "cubic" | "3" => Ok(InterpolationOrder::Cubic),
            _ => Err(JsError::new(&format!(
                "unknown ibfftOrder {raw:?}; expected 'bilinear' (1) or 'cubic' (3)"
            ))),
        };
    }
    Err(JsError::new(
        "ibfftOrder must be 'bilinear', 'cubic', 1, or 3",
    ))
}

pub(crate) fn is_plain_js_object(val: &JsValue) -> bool {
    val.is_object()
        && !js_sys::Array::is_array(val)
        && !js_sys::ArrayBuffer::is_view(val)
        && !val.is_function()
}

pub(crate) fn get_prop(obj: &JsValue, key: &str) -> Result<JsValue, JsError> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .map_err(|err| JsError::new(&format!("failed to read options.{key}: {err:?}")))
}

pub(crate) fn opt_usize_prop(obj: &JsValue, key: &str, default: usize) -> Result<usize, JsError> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(default);
    }
    let num = val
        .as_f64()
        .ok_or_else(|| JsError::new(&format!("options.{key} must be an integer number")))?;
    parse_non_negative_usize(num, key)
}

pub(crate) fn opt_maybe_usize_prop(obj: &JsValue, key: &str) -> Result<Option<usize>, JsError> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    let num = val
        .as_f64()
        .ok_or_else(|| JsError::new(&format!("options.{key} must be an integer number")))?;
    parse_non_negative_usize(num, key).map(Some)
}

pub(crate) fn opt_f64_prop(obj: &JsValue, key: &str, default: f64) -> Result<f64, JsError> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(default);
    }
    val.as_f64()
        .ok_or_else(|| JsError::new(&format!("options.{key} must be a number")))
}

pub(crate) fn opt_maybe_f64_prop(obj: &JsValue, key: &str) -> Result<Option<f64>, JsError> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    val.as_f64()
        .map(Some)
        .ok_or_else(|| JsError::new(&format!("options.{key} must be a number")))
}

pub(crate) fn opt_bool_prop(obj: &JsValue, key: &str, default: bool) -> Result<bool, JsError> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(default);
    }
    val.as_bool()
        .ok_or_else(|| JsError::new(&format!("options.{key} must be a boolean")))
}

pub(crate) fn opt_string_prop(obj: &JsValue, key: &str) -> Result<Option<String>, JsError> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    val.as_string()
        .map(Some)
        .ok_or_else(|| JsError::new(&format!("options.{key} must be a string")))
}

pub(crate) fn opt_seed_prop(obj: &JsValue) -> Result<u64, JsError> {
    let val = get_prop(obj, "seed")?;
    if val.is_undefined() || val.is_null() {
        return parse_seed_u64(None);
    }
    let num = val
        .as_f64()
        .ok_or_else(|| JsError::new("options.seed must be a non-negative safe integer"))?;
    parse_seed_u64(Some(num))
}

pub(crate) fn reject_extra_args_with_options(has_extra: bool) -> Result<(), JsError> {
    if has_extra {
        return Err(JsError::new(
            "unexpected trailing positional argument when passing an options object",
        ));
    }
    Ok(())
}

fn validate_layout_for_solver(
    layout: &JsLayout2d,
    expected_nodes: usize,
    ctx: &'static str,
) -> Result<(), JsError> {
    if layout.len() != expected_nodes {
        return Err(core_err_to_js(CoreError::dimension_mismatch(
            expected_nodes,
            layout.len(),
            ctx,
        )));
    }
    if !layout.inner().is_all_finite() {
        return Err(core_err_to_js(CoreError::non_finite(ctx)));
    }
    Ok(())
}

pub(crate) fn step_layout_in_place<S: IterativeSolver<f32>>(
    solver: &mut S,
    layout: &mut JsLayout2d,
) -> Result<String, JsError> {
    let mut soa = layout.inner().to_layout2d();
    let status = solver.step(&mut soa).map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

pub(crate) fn run_steps_layout_in_place<S: IterativeSolver<f32>>(
    expected_nodes: usize,
    solver: &mut S,
    layout: &mut JsLayout2d,
    steps: f64,
    ctx: &'static str,
) -> Result<String, JsError> {
    let n_steps = parse_non_negative_usize(steps, "steps")?;
    validate_layout_for_solver(layout, expected_nodes, ctx)?;
    let mut soa = layout.inner().to_layout2d();
    let status = solver
        .run_steps(&mut soa, n_steps)
        .map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

pub(crate) fn run_layout_in_place<S: IterativeSolver<f32>>(
    expected_nodes: usize,
    solver: &mut S,
    layout: &mut JsLayout2d,
    ctx: &'static str,
) -> Result<String, JsError> {
    validate_layout_for_solver(layout, expected_nodes, ctx)?;
    let mut soa = layout.inner().to_layout2d();
    let status = solver.run(&mut soa).map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

// ============================================================================
// 1. Clustering & Coarsening APIs
// ============================================================================

/// Output of `kmeansPlusPlus`.
#[wasm_bindgen(js_name = KMeansResult)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsKMeansResult {
    inner: KMeansResult,
    k: usize,
    dims: usize,
}

#[wasm_bindgen(js_class = KMeansResult)]
impl JsKMeansResult {
    /// Returns canonical cluster labels in `0..k` (`Uint32Array`).
    pub fn labels(&self) -> Vec<u32> {
        self.inner.labels.clone()
    }

    /// Returns flat row-major `(k, dims)` cluster centroids (`Float64Array`).
    pub fn centroids(&self) -> Vec<f64> {
        self.inner.centroids.clone()
    }

    /// Returns the total within-cluster sum of squared Euclidean distances.
    #[wasm_bindgen(getter)]
    pub fn inertia(&self) -> f64 {
        self.inner.inertia
    }

    /// Returns the number of Lloyd iterations executed in the winning trial.
    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> usize {
        self.inner.iterations
    }

    /// Returns `true` if the centroid movement converged within `tol`.
    #[wasm_bindgen(getter)]
    pub fn converged(&self) -> bool {
        self.inner.converged
    }

    /// Returns the number of clusters `k`.
    #[wasm_bindgen(getter)]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the feature dimensionality `dims`.
    #[wasm_bindgen(getter)]
    pub fn dims(&self) -> usize {
        self.dims
    }
}

/// Output of `louvainCommunities`.
#[wasm_bindgen(js_name = LouvainResult)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsLouvainResult {
    inner: LouvainResult,
}

#[wasm_bindgen(js_class = LouvainResult)]
impl JsLouvainResult {
    /// Returns canonical community labels in `0..numCommunities` (`Uint32Array`).
    pub fn labels(&self) -> Vec<u32> {
        self.inner.labels.clone()
    }

    /// Returns the number of distinct non-empty communities detected.
    #[wasm_bindgen(getter, js_name = numCommunities)]
    pub fn num_communities(&self) -> usize {
        self.inner.num_communities
    }

    /// Returns the resolution-weighted modularity $Q_\gamma$.
    #[wasm_bindgen(getter)]
    pub fn modularity(&self) -> f64 {
        self.inner.modularity
    }

    /// Returns the number of multi-level aggregation passes committed.
    #[wasm_bindgen(getter)]
    pub fn passes(&self) -> usize {
        self.inner.passes
    }
}

/// Output of `coarsenGraph`.
#[wasm_bindgen(js_name = CoarsenedGraphResult)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsCoarsenedGraphResult {
    graph: CsrGraph<f32>,
    node_to_coarse: Vec<u32>,
    coarse_sizes: Vec<u32>,
    num_clusters: usize,
}

#[wasm_bindgen(js_class = CoarsenedGraphResult)]
impl JsCoarsenedGraphResult {
    /// Returns a cloned handle to the weighted undirected coarsened `CsrGraph`.
    pub fn graph(&self) -> JsCsrGraph {
        JsCsrGraph::from_inner(self.graph.clone())
    }

    /// Returns the compacted `0..numClusters` cluster ID for each original vertex (`Uint32Array`).
    #[wasm_bindgen(js_name = nodeToCoarse)]
    pub fn node_to_coarse(&self) -> Vec<u32> {
        self.node_to_coarse.clone()
    }

    /// Returns the number of original vertices assigned to each coarse node (`Uint32Array`).
    #[wasm_bindgen(js_name = coarseSizes)]
    pub fn coarse_sizes(&self) -> Vec<u32> {
        self.coarse_sizes.clone()
    }

    /// Returns the number of distinct compacted clusters.
    #[wasm_bindgen(getter, js_name = numClusters)]
    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }
}

/// Clusters flat row-major `(numNodes, dims)` points with deterministic K-Means++.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = kmeansPlusPlus)]
pub fn js_kmeans_plus_plus(
    points: &[f64],
    num_nodes: f64,
    dims: f64,
    k: Option<f64>,
    max_iters: Option<f64>,
    tol: Option<f64>,
    n_init: Option<f64>,
    seed: Option<f64>,
) -> Result<JsKMeansResult, JsError> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let d = parse_non_negative_usize(dims, "dims")?;
    let k_val = parse_optional_usize(k, DEFAULT_KMEANS_K, "k")?;
    let max_iters_val = parse_optional_usize(max_iters, DEFAULT_KMEANS_MAX_ITERS, "maxIters")?;
    let n_init_val = parse_optional_usize(n_init, DEFAULT_KMEANS_N_INIT, "nInit")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let cfg = KMeansConfig {
        k: k_val,
        max_iters: max_iters_val,
        tol: tol.unwrap_or(DEFAULT_KMEANS_TOL),
        n_init: n_init_val,
        seed: seed_u64,
    };
    let inner = kmeans_plus_plus(points, n, d, &cfg).map_err(core_err_to_js)?;
    Ok(JsKMeansResult {
        inner,
        k: k_val,
        dims: d,
    })
}

/// Performs Ng-Jordan-Weiss spectral clustering on an undirected `CsrGraph`.
#[wasm_bindgen(js_name = spectralClustering)]
pub fn js_spectral_clustering(
    graph: &JsCsrGraph,
    k: Option<f64>,
    max_eigen_iters: Option<f64>,
    eigen_tol: Option<f64>,
    kmeans_max_iters: Option<f64>,
    kmeans_tol: Option<f64>,
    seed: Option<f64>,
) -> Result<Vec<u32>, JsError> {
    let cfg = SpectralClusteringConfig {
        k: parse_optional_usize(k, DEFAULT_SPECTRAL_K, "k")?,
        max_eigen_iters: parse_optional_usize(
            max_eigen_iters,
            DEFAULT_SPECTRAL_MAX_EIGEN_ITERS,
            "maxEigenIters",
        )?,
        eigen_tol: eigen_tol.unwrap_or(DEFAULT_SPECTRAL_EIGEN_TOL),
        kmeans_max_iters: parse_optional_usize(
            kmeans_max_iters,
            DEFAULT_KMEANS_MAX_ITERS,
            "kmeansMaxIters",
        )?,
        kmeans_tol: kmeans_tol.unwrap_or(DEFAULT_KMEANS_TOL),
        seed: parse_seed_u64(seed)?,
    };
    spectral_clustering(graph.inner(), &cfg).map_err(core_err_to_js)
}

/// Detects communities on an undirected `CsrGraph` via multi-level Louvain modularity optimization.
#[wasm_bindgen(js_name = louvainCommunities)]
pub fn js_louvain_communities(
    graph: &JsCsrGraph,
    max_passes: Option<f64>,
    resolution: Option<f64>,
    min_modularity_gain: Option<f64>,
    seed: Option<f64>,
) -> Result<JsLouvainResult, JsError> {
    let cfg = LouvainConfig {
        max_passes: parse_optional_usize(max_passes, DEFAULT_LOUVAIN_MAX_PASSES, "maxPasses")?,
        resolution: resolution.unwrap_or(DEFAULT_LOUVAIN_RESOLUTION),
        min_modularity_gain: min_modularity_gain.unwrap_or(DEFAULT_LOUVAIN_MIN_GAIN),
        seed: parse_seed_u64(seed)?,
    };
    let inner = louvain(graph.inner(), &cfg).map_err(core_err_to_js)?;
    Ok(JsLouvainResult { inner })
}

/// Detects communities on an undirected `CsrGraph` via asynchronous Label Propagation.
#[wasm_bindgen(js_name = labelPropagation)]
pub fn js_label_propagation(
    graph: &JsCsrGraph,
    max_iters: Option<f64>,
    seed: Option<f64>,
) -> Result<Vec<u32>, JsError> {
    let cfg = LabelPropagationConfig {
        max_iters: parse_optional_usize(max_iters, DEFAULT_LPA_MAX_ITERS, "maxIters")?,
        seed: parse_seed_u64(seed)?,
    };
    label_propagation(graph.inner(), &cfg).map_err(core_err_to_js)
}

/// Coarsens an undirected `CsrGraph` by contracting vertices that share a cluster label.
#[wasm_bindgen(js_name = coarsenGraph)]
pub fn js_coarsen_graph(
    graph: &JsCsrGraph,
    labels: &[u32],
) -> Result<JsCoarsenedGraphResult, JsError> {
    let coarse = coarsen_graph(graph.inner(), labels).map_err(core_err_to_js)?;
    let weights_f32: Option<Vec<f32>> = coarse
        .graph
        .weights()
        .map(|ws| {
            ws.iter()
                .map(|&w| {
                    let wf = w as f32;
                    if !wf.is_finite() {
                        return Err(JsError::new("coarsened edge weight overflowed f32"));
                    }
                    Ok(wf)
                })
                .collect::<Result<Vec<f32>, JsError>>()
        })
        .transpose()?;
    let graph_f32 = CsrGraph::<f32>::from_csr(
        coarse.graph.indptr(),
        coarse.graph.indices(),
        weights_f32.as_deref(),
        coarse.graph.is_directed(),
        false,
    )
    .map_err(core_err_to_js)?;
    Ok(JsCoarsenedGraphResult {
        graph: graph_f32,
        node_to_coarse: coarse.node_to_coarse,
        coarse_sizes: coarse.coarse_sizes,
        num_clusters: coarse.num_clusters,
    })
}

/// Projects a coarse 2D `Layout2d` (`f32`) back onto fine vertices via `nodeToCoarse`.
#[wasm_bindgen(js_name = projectClusteredLayout)]
pub fn js_project_clustered_layout(
    coarse_layout: &JsLayout2d,
    node_to_coarse: &[u32],
    spread_radius: Option<f64>,
    seed: Option<f64>,
) -> Result<JsLayout2d, JsError> {
    let radius = spread_radius.unwrap_or(DEFAULT_SPREAD_RADIUS);
    if !radius.is_finite() {
        return Err(core_err_to_js(CoreError::non_finite(
            "projectClusteredLayout::spreadRadius",
        )));
    }
    if radius < 0.0 {
        return Err(core_err_to_js(CoreError::invalid_input(
            "spreadRadius must be non-negative",
        )));
    }
    let radius_f32 = radius as f32;
    if !radius_f32.is_finite() {
        return Err(core_err_to_js(CoreError::non_finite(
            "projectClusteredLayout::spreadRadius",
        )));
    }
    let seed_u64 = parse_seed_u64(seed)?;
    let coarse_soa = coarse_layout.inner().to_layout2d();
    let fine = project_clustered_layout(&coarse_soa, node_to_coarse, radius_f32, seed_u64)
        .map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&fine))
}

// ============================================================================
// 2. Non-Euclidean ManifoldLayout2d (f32)
// ============================================================================

/// Non-Euclidean 2D node layout (`f32`) constrained to `"hyperbolic"`, `"spherical"`, or `"torus"`.
#[wasm_bindgen(js_name = ManifoldLayout2d)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsManifoldLayout2d {
    inner: ManifoldLayout2d<f32>,
}

impl JsManifoldLayout2d {
    pub(crate) fn inner(&self) -> &ManifoldLayout2d<f32> {
        &self.inner
    }

    pub(crate) fn inner_mut(&mut self) -> &mut ManifoldLayout2d<f32> {
        &mut self.inner
    }
}

#[wasm_bindgen(js_class = ManifoldLayout2d)]
impl JsManifoldLayout2d {
    /// Creates a zero-initialized `ManifoldLayout2d` for `n` vertices on geometry `kind`.
    pub fn zeros(kind: &str, n: f64) -> Result<JsManifoldLayout2d, JsError> {
        let m_kind = parse_manifold_kind(kind)?;
        let count = parse_non_negative_usize(n, "n")?;
        Ok(Self {
            inner: ManifoldLayout2d::<f32>::zeros(m_kind, count),
        })
    }

    /// Creates a zero-initialized Poincaré disk `ManifoldLayout2d` for `n` vertices.
    #[wasm_bindgen(js_name = hyperbolicPoincare)]
    pub fn hyperbolic_poincare(n: f64) -> Result<JsManifoldLayout2d, JsError> {
        let count = parse_non_negative_usize(n, "n")?;
        Ok(Self {
            inner: ManifoldLayout2d::<f32>::zeros(ManifoldKind::HyperbolicPoincare, count),
        })
    }

    /// Creates a zero-initialized spherical `ManifoldLayout2d` for `n` vertices.
    pub fn spherical(n: f64) -> Result<JsManifoldLayout2d, JsError> {
        let count = parse_non_negative_usize(n, "n")?;
        Ok(Self {
            inner: ManifoldLayout2d::<f32>::zeros(ManifoldKind::Spherical, count),
        })
    }

    /// Creates a zero-initialized flat unit torus `ManifoldLayout2d` for `n` vertices.
    #[wasm_bindgen(js_name = torus2d)]
    pub fn torus_2d(n: f64) -> Result<JsManifoldLayout2d, JsError> {
        let count = parse_non_negative_usize(n, "n")?;
        Ok(Self {
            inner: ManifoldLayout2d::<f32>::zeros(ManifoldKind::Torus2d, count),
        })
    }

    /// Creates a `ManifoldLayout2d` from a Euclidean `Layout2d`, projecting points onto `kind`.
    #[wasm_bindgen(js_name = fromLayout)]
    pub fn from_layout(kind: &str, layout: &JsLayout2d) -> Result<JsManifoldLayout2d, JsError> {
        let m_kind = parse_manifold_kind(kind)?;
        let soa = layout.inner().to_layout2d();
        let inner = ManifoldLayout2d::<f32>::from_layout(m_kind, soa).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Creates a `ManifoldLayout2d` from flat interleaved `[x0, y0, ...]` coordinates, projecting onto `kind`.
    #[wasm_bindgen(js_name = fromCoords)]
    pub fn from_coords(kind: &str, coords: &[f32]) -> Result<JsManifoldLayout2d, JsError> {
        let m_kind = parse_manifold_kind(kind)?;
        let soa = Layout2d::<f32>::from_interleaved(coords).map_err(core_err_to_js)?;
        let inner = ManifoldLayout2d::<f32>::from_layout(m_kind, soa).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Creates a deterministically seeded safe random initialization on `kind`.
    #[wasm_bindgen(js_name = randomInit)]
    pub fn random_init(
        kind: &str,
        n: f64,
        scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsManifoldLayout2d, JsError> {
        let m_kind = parse_manifold_kind(kind)?;
        let count = parse_non_negative_usize(n, "n")?;
        let scale_val = scale.unwrap_or(0.25);
        let seed_u64 = parse_seed_u64(seed)?;
        let inner = ManifoldLayout2d::<f32>::random_init(m_kind, count, scale_val, seed_u64)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Creates a `ManifoldLayout2d` from a Euclidean `Layout2d` scaled safely inside `maxRadius` (default `0.7`).
    #[wasm_bindgen(js_name = fromLayoutScaled)]
    pub fn from_layout_scaled(
        kind: &str,
        layout: &JsLayout2d,
        max_radius: Option<f64>,
    ) -> Result<JsManifoldLayout2d, JsError> {
        let m_kind = parse_manifold_kind(kind)?;
        let mr = max_radius.unwrap_or(0.7);
        let soa = layout.inner().to_layout2d();
        let inner = ManifoldLayout2d::<f32>::from_layout_scaled(m_kind, &soa, mr)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns the canonical manifold kind string: `"hyperbolic"`, `"spherical"`, or `"torus"`.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        manifold_kind_str(self.inner.kind()).to_owned()
    }

    /// Alias for `kind` matching `ManifoldLayout2d.manifold`.
    #[wasm_bindgen(getter)]
    pub fn manifold(&self) -> String {
        manifold_kind_str(self.inner.kind()).to_owned()
    }

    /// Returns the number of vertices in the manifold layout.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns the number of vertices in the manifold layout.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the manifold layout contains zero vertices.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns a flat interleaved `[x0, y0, ...]` `Float32Array` copy of all coordinates.
    pub fn coords(&self) -> Vec<f32> {
        self.inner.as_layout().to_interleaved()
    }

    /// Converts this manifold layout to a `Layout2d` copy.
    #[wasm_bindgen(js_name = toLayout)]
    pub fn to_layout(&self) -> JsLayout2d {
        JsLayout2d::from_layout2d(self.inner.as_layout())
    }

    /// Computes the exact geodesic distance between vertices `i` and `j`.
    #[wasm_bindgen(js_name = geodesicDistance)]
    pub fn geodesic_distance(&self, i: f64, j: f64) -> Result<f32, JsError> {
        let ii = parse_non_negative_usize(i, "i")?;
        let jj = parse_non_negative_usize(j, "j")?;
        self.inner.geodesic_distance(ii, jj).map_err(core_err_to_js)
    }

    /// Computes the Riemannian logarithmic map `[vx, vy]` at vertex `i` pointing toward `j`.
    #[wasm_bindgen(js_name = tangentDelta)]
    pub fn tangent_delta(&self, i: f64, j: f64) -> Result<Vec<f32>, JsError> {
        let ii = parse_non_negative_usize(i, "i")?;
        let jj = parse_non_negative_usize(j, "j")?;
        let (vx, vy) = self.inner.tangent_delta(ii, jj).map_err(core_err_to_js)?;
        Ok(vec![vx, vy])
    }

    /// Advances vertex `i` along the geodesic defined by tangent vector `(vx, vy)`.
    #[wasm_bindgen(js_name = applyTangentStep)]
    pub fn apply_tangent_step(&mut self, i: f64, vx: f32, vy: f32) -> Result<(), JsError> {
        let idx = parse_non_negative_usize(i, "i")?;
        self.inner
            .apply_tangent_step(idx, vx, vy)
            .map_err(core_err_to_js)
    }

    /// Re-projects all coordinates onto the manifold's canonical domain after checking finiteness.
    #[wasm_bindgen(js_name = projectAll)]
    pub fn project_all(&mut self) -> Result<(), JsError> {
        self.inner.project_all().map_err(core_err_to_js)
    }

    /// Returns the first coordinate (`x` or `lon`) of vertex `index`.
    pub fn x(&self, index: f64) -> Result<f32, JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        if idx >= self.inner.len() {
            return Err(core_err_to_js(CoreError::index_out_of_bounds(
                idx,
                self.inner.len(),
                "ManifoldLayout2d::x",
            )));
        }
        Ok(self.inner.x()[idx])
    }

    /// Returns the second coordinate (`y` or `colat`) of vertex `index`.
    pub fn y(&self, index: f64) -> Result<f32, JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        if idx >= self.inner.len() {
            return Err(core_err_to_js(CoreError::index_out_of_bounds(
                idx,
                self.inner.len(),
                "ManifoldLayout2d::y",
            )));
        }
        Ok(self.inner.y()[idx])
    }

    /// Sets vertex `index`'s coordinates `(x, y)` and projects onto the manifold domain.
    #[wasm_bindgen(js_name = setPoint)]
    pub fn set_point(&mut self, index: f64, x: f32, y: f32) -> Result<(), JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner.set_point(idx, x, y).map_err(core_err_to_js)
    }

    /// Returns `true` if every coordinate in the manifold layout is finite.
    #[wasm_bindgen(js_name = isAllFinite)]
    pub fn is_all_finite(&self) -> bool {
        self.inner.as_layout().is_all_finite()
    }

    /// Computes the closed-form Möbius-equivariant Einstein midpoint `[cx, cy]` on the Poincaré disk.
    #[wasm_bindgen(js_name = poincareCentroid)]
    pub fn poincare_centroid(&self) -> Result<Vec<f64>, JsError> {
        let (cx, cy) = self.inner.poincare_centroid().map_err(core_err_to_js)?;
        Ok(vec![f64::from(cx), f64::from(cy)])
    }

    /// Isometrically recenters all Poincaré disk points at `(cx, cy)` via Möbius automorphism.
    #[wasm_bindgen(js_name = recenterPoincare)]
    pub fn recenter_poincare(&mut self, cx: f64, cy: f64) -> Result<(), JsError> {
        if !cx.is_finite() || !cy.is_finite() {
            return Err(core_err_to_js(CoreError::non_finite(
                "ManifoldLayout2d::recenterPoincare",
            )));
        }
        let cx32 = cx as f32;
        let cy32 = cy as f32;
        if !cx32.is_finite() || !cy32.is_finite() {
            return Err(core_err_to_js(CoreError::invalid_input(
                "recenterPoincare coordinate overflows float32",
            )));
        }
        self.inner
            .recenter_poincare(cx32, cy32)
            .map_err(core_err_to_js)
    }

    /// Isometrically recenters the Poincaré disk layout so vertex `index` moves to `(0, 0)`.
    #[wasm_bindgen(js_name = recenterPoincareAt)]
    pub fn recenter_poincare_at(&mut self, index: f64) -> Result<(), JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner.recenter_poincare_at(idx).map_err(core_err_to_js)
    }

    /// Isometrically recenters the Poincaré disk layout so its Einstein midpoint lies at `(0, 0)`.
    #[wasm_bindgen(js_name = centerPoincareCentroid)]
    pub fn center_poincare_centroid(&mut self) -> Result<(), JsError> {
        self.inner
            .center_poincare_centroid()
            .map_err(core_err_to_js)
    }
}

/// Computes the Poincaré disk hyperbolic geodesic distance between `(x1, y1)` and `(x2, y2)`.
#[wasm_bindgen(js_name = poincareDistance)]
pub fn js_poincare_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    poincare_distance(x1, y1, x2, y2)
}

/// Computes Poincaré disk Möbius addition `(x1, y1) (+) (x2, y2)` inside the open unit disk, returning `[x, y]`.
#[wasm_bindgen(js_name = poincareMobiusAdd)]
pub fn js_poincare_mobius_add(x1: f64, y1: f64, x2: f64, y2: f64) -> Vec<f64> {
    let (rx, ry) = poincare_mobius_add(x1, y1, x2, y2);
    vec![rx, ry]
}

/// Computes the great-circle geodesic distance on the unit sphere in `[0, pi]`.
#[wasm_bindgen(js_name = sphericalDistance)]
pub fn js_spherical_distance(lon1: f64, colat1: f64, lon2: f64, colat2: f64) -> f64 {
    spherical_distance(lon1, colat1, lon2, colat2)
}

/// Computes the geodesic distance on the flat unit torus `[0, 1)^2`.
#[wasm_bindgen(js_name = torusDistance)]
pub fn js_torus_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    torus_distance(x1, y1, x2, y2)
}

/// Wraps a 2D coordinate pair onto the flat unit torus `[0, 1)^2`, returning `[x, y]`.
#[wasm_bindgen(js_name = wrapTorusPoint)]
pub fn js_wrap_torus_point(x: f64, y: f64) -> Vec<f64> {
    let (wx, wy) = wrap_torus_point(x, y);
    vec![wx, wy]
}

// ============================================================================
// 3. TsNetSolver (f32)
// ============================================================================

/// FFT field telemetry counters returned by `TsNetSolver.lastFftStats()`.
#[wasm_bindgen(js_name = FitFieldStats)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsFitFieldStats {
    inner: FitFieldStats,
}

#[wasm_bindgen(js_class = FitFieldStats)]
impl JsFitFieldStats {
    /// Number of forward 2D FFTs executed on particle charge grids.
    #[wasm_bindgen(getter, js_name = particleForwardFfts)]
    pub fn particle_forward_ffts(&self) -> f64 {
        self.inner.particle_forward_ffts as f64
    }

    /// Number of forward 2D FFTs executed on kernel grids.
    #[wasm_bindgen(getter, js_name = kernelForwardFfts)]
    pub fn kernel_forward_ffts(&self) -> f64 {
        self.inner.kernel_forward_ffts as f64
    }

    /// Number of inverse 2D FFTs executed.
    #[wasm_bindgen(getter, js_name = inverseFfts)]
    pub fn inverse_ffts(&self) -> f64 {
        self.inner.inverse_ffts as f64
    }

    /// Total `fft_2d` calls (`particleForwardFfts + kernelForwardFfts + inverseFfts`).
    #[wasm_bindgen(getter, js_name = totalFft2dCalls)]
    pub fn total_fft_2d_calls(&self) -> f64 {
        self.inner.total_fft_2d_calls as f64
    }
}

#[allow(clippy::too_many_arguments)]
fn parse_tsnet_config(
    backend_or_options: &JsValue,
    iterations_stage1: Option<f64>,
    iterations_stage2: Option<f64>,
    iterations_stage3: Option<f64>,
    learning_rate: Option<f64>,
    perplexity_c: Option<f64>,
    min_perplexity: Option<f64>,
    max_perplexity: Option<f64>,
    exaggeration: Option<f64>,
    lambda_c_stage1: Option<f64>,
    lambda_c_stage2: Option<f64>,
    lambda_c_stage3: Option<f64>,
    lambda_r_stage3: Option<f64>,
    momentum: Option<f64>,
    epsilon_r: Option<f64>,
    theta: Option<f64>,
    power: Option<f64>,
    sigma_iters: Option<f64>,
    sigma_tolerance: Option<f64>,
    row_normalize: Option<bool>,
    seed: Option<f64>,
    intervals: Option<f64>,
    interpolation_order: &JsValue,
    min_grid_dim: Option<f64>,
    max_grid_dim: Option<f64>,
) -> Result<TsNetConfig, JsError> {
    let mut cfg = TsNetConfig::pure_2hop();
    if is_plain_js_object(backend_or_options) {
        let has_extra = iterations_stage1.is_some()
            || iterations_stage2.is_some()
            || iterations_stage3.is_some()
            || learning_rate.is_some()
            || perplexity_c.is_some()
            || min_perplexity.is_some()
            || max_perplexity.is_some()
            || exaggeration.is_some()
            || lambda_c_stage1.is_some()
            || lambda_c_stage2.is_some()
            || lambda_c_stage3.is_some()
            || lambda_r_stage3.is_some()
            || momentum.is_some()
            || epsilon_r.is_some()
            || theta.is_some()
            || power.is_some()
            || sigma_iters.is_some()
            || sigma_tolerance.is_some()
            || row_normalize.is_some()
            || seed.is_some()
            || intervals.is_some()
            || (!interpolation_order.is_undefined() && !interpolation_order.is_null())
            || min_grid_dim.is_some()
            || max_grid_dim.is_some();
        reject_extra_args_with_options(has_extra)?;

        if let Some(b_str) = opt_string_prop(backend_or_options, "backend")? {
            cfg.backend = parse_tsnet_backend(&b_str)?;
        }
        cfg.perplexity_c = opt_f64_prop(backend_or_options, "perplexityC", cfg.perplexity_c)?;
        cfg.min_perplexity = opt_f64_prop(backend_or_options, "minPerplexity", cfg.min_perplexity)?;
        cfg.max_perplexity = opt_f64_prop(backend_or_options, "maxPerplexity", cfg.max_perplexity)?;
        cfg.power = opt_f64_prop(backend_or_options, "power", cfg.power)?;
        cfg.sigma_iters = opt_usize_prop(backend_or_options, "sigmaIters", cfg.sigma_iters)?;
        cfg.sigma_tolerance =
            opt_f64_prop(backend_or_options, "sigmaTolerance", cfg.sigma_tolerance)?;
        if opt_bool_prop(backend_or_options, "rowNormalize", false)? {
            cfg.symmetrization = SymmetrizationMode::RowNormalized;
        }
        cfg.jacobi_precondition = opt_bool_prop(
            backend_or_options,
            "jacobiPrecondition",
            cfg.jacobi_precondition,
        )?;
        cfg.iterations_stage1 = opt_usize_prop(
            backend_or_options,
            "iterationsStage1",
            cfg.iterations_stage1,
        )?;
        cfg.iterations_stage2 = opt_usize_prop(
            backend_or_options,
            "iterationsStage2",
            cfg.iterations_stage2,
        )?;
        cfg.iterations_stage3 = opt_usize_prop(
            backend_or_options,
            "iterationsStage3",
            cfg.iterations_stage3,
        )?;
        cfg.exaggeration = opt_f64_prop(backend_or_options, "exaggeration", cfg.exaggeration)?;
        cfg.lambda_c_stage1 =
            opt_f64_prop(backend_or_options, "lambdaCStage1", cfg.lambda_c_stage1)?;
        cfg.lambda_c_stage2 =
            opt_f64_prop(backend_or_options, "lambdaCStage2", cfg.lambda_c_stage2)?;
        cfg.lambda_c_stage3 =
            opt_f64_prop(backend_or_options, "lambdaCStage3", cfg.lambda_c_stage3)?;
        cfg.lambda_r_stage3 =
            opt_f64_prop(backend_or_options, "lambdaRStage3", cfg.lambda_r_stage3)?;
        cfg.learning_rate =
            opt_maybe_f64_prop(backend_or_options, "learningRate")?.or(cfg.learning_rate);
        cfg.momentum = opt_f64_prop(backend_or_options, "momentum", cfg.momentum)?;
        cfg.epsilon_r = opt_f64_prop(backend_or_options, "epsilonR", cfg.epsilon_r)?;
        cfg.theta = opt_f64_prop(backend_or_options, "theta", cfg.theta)?;

        let ord_primary = get_prop(backend_or_options, "interpolationOrder")?;
        let ord_alt = get_prop(backend_or_options, "ibfftOrder")?;
        if !ord_primary.is_undefined() && !ord_primary.is_null() {
            cfg.ibfft.order = parse_ibfft_order(&ord_primary)?;
        } else if !ord_alt.is_undefined() && !ord_alt.is_null() {
            cfg.ibfft.order = parse_ibfft_order(&ord_alt)?;
        }
        let int_primary = opt_maybe_usize_prop(backend_or_options, "intervals")?;
        let int_alt = opt_maybe_usize_prop(backend_or_options, "ibfftIntervals")?;
        cfg.ibfft.intervals = int_primary.or(int_alt).unwrap_or(cfg.ibfft.intervals);

        let min_g_primary = opt_maybe_usize_prop(backend_or_options, "minGridDim")?;
        let min_g_alt = opt_maybe_usize_prop(backend_or_options, "ibfftMinGridDim")?;
        cfg.ibfft.min_grid_dim = min_g_primary
            .or(min_g_alt)
            .unwrap_or(cfg.ibfft.min_grid_dim);

        let max_g_primary = opt_maybe_usize_prop(backend_or_options, "maxGridDim")?;
        let max_g_alt = opt_maybe_usize_prop(backend_or_options, "ibfftMaxGridDim")?;
        cfg.ibfft.max_grid_dim = max_g_primary
            .or(max_g_alt)
            .unwrap_or(cfg.ibfft.max_grid_dim);

        cfg.seed = opt_seed_prop(backend_or_options)?;
        return Ok(cfg);
    }

    if !backend_or_options.is_undefined() && !backend_or_options.is_null() {
        let b_str = backend_or_options.as_string().ok_or_else(|| {
            JsError::new(
                "backendOrOptions must be a backend string, options object, null, or undefined",
            )
        })?;
        cfg.backend = parse_tsnet_backend(&b_str)?;
    }

    cfg.iterations_stage1 =
        parse_optional_usize(iterations_stage1, cfg.iterations_stage1, "iterationsStage1")?;
    cfg.iterations_stage2 =
        parse_optional_usize(iterations_stage2, cfg.iterations_stage2, "iterationsStage2")?;
    cfg.iterations_stage3 =
        parse_optional_usize(iterations_stage3, cfg.iterations_stage3, "iterationsStage3")?;
    cfg.learning_rate = learning_rate.or(cfg.learning_rate);
    cfg.perplexity_c = perplexity_c.unwrap_or(cfg.perplexity_c);
    cfg.min_perplexity = min_perplexity.unwrap_or(cfg.min_perplexity);
    cfg.max_perplexity = max_perplexity.unwrap_or(cfg.max_perplexity);
    cfg.exaggeration = exaggeration.unwrap_or(cfg.exaggeration);
    cfg.lambda_c_stage1 = lambda_c_stage1.unwrap_or(cfg.lambda_c_stage1);
    cfg.lambda_c_stage2 = lambda_c_stage2.unwrap_or(cfg.lambda_c_stage2);
    cfg.lambda_c_stage3 = lambda_c_stage3.unwrap_or(cfg.lambda_c_stage3);
    cfg.lambda_r_stage3 = lambda_r_stage3.unwrap_or(cfg.lambda_r_stage3);
    cfg.momentum = momentum.unwrap_or(cfg.momentum);
    cfg.epsilon_r = epsilon_r.unwrap_or(cfg.epsilon_r);
    cfg.theta = theta.unwrap_or(cfg.theta);
    cfg.power = power.unwrap_or(cfg.power);
    cfg.sigma_iters = parse_optional_usize(sigma_iters, cfg.sigma_iters, "sigmaIters")?;
    cfg.sigma_tolerance = sigma_tolerance.unwrap_or(cfg.sigma_tolerance);
    if row_normalize.unwrap_or(false) {
        cfg.symmetrization = SymmetrizationMode::RowNormalized;
    }
    cfg.seed = parse_seed_u64(seed)?;
    cfg.ibfft.intervals = parse_optional_usize(intervals, cfg.ibfft.intervals, "intervals")?;
    if !interpolation_order.is_undefined() && !interpolation_order.is_null() {
        cfg.ibfft.order = parse_ibfft_order(interpolation_order)?;
    }
    cfg.ibfft.min_grid_dim =
        parse_optional_usize(min_grid_dim, cfg.ibfft.min_grid_dim, "minGridDim")?;
    cfg.ibfft.max_grid_dim =
        parse_optional_usize(max_grid_dim, cfg.ibfft.max_grid_dim, "maxGridDim")?;
    Ok(cfg)
}

/// Cooperative unified tsNET / BH-tsNET / FIt-tsNET / L-tsNET solver (`f32`).
#[wasm_bindgen(js_name = TsNetSolver)]
#[derive(Debug, Clone)]
pub struct JsTsNetSolver {
    inner: TsNetSolver<f32>,
}

#[wasm_bindgen(js_class = TsNetSolver)]
impl JsTsNetSolver {
    /// Constructs a `TsNetSolver` using the Pure 1+2-Hop node-wise adaptive perplexity preset.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = pure2Hop)]
    pub fn pure_2hop(
        graph: &JsCsrGraph,
        backend_or_options: &JsValue,
        iterations_stage1: Option<f64>,
        iterations_stage2: Option<f64>,
        iterations_stage3: Option<f64>,
        learning_rate: Option<f64>,
        perplexity_c: Option<f64>,
        min_perplexity: Option<f64>,
        max_perplexity: Option<f64>,
        exaggeration: Option<f64>,
        lambda_c_stage1: Option<f64>,
        lambda_c_stage2: Option<f64>,
        lambda_c_stage3: Option<f64>,
        lambda_r_stage3: Option<f64>,
        momentum: Option<f64>,
        epsilon_r: Option<f64>,
        theta: Option<f64>,
        power: Option<f64>,
        sigma_iters: Option<f64>,
        sigma_tolerance: Option<f64>,
        row_normalize: Option<bool>,
        seed: Option<f64>,
        intervals: Option<f64>,
        interpolation_order: &JsValue,
        min_grid_dim: Option<f64>,
        max_grid_dim: Option<f64>,
    ) -> Result<JsTsNetSolver, JsError> {
        let cfg = parse_tsnet_config(
            backend_or_options,
            iterations_stage1,
            iterations_stage2,
            iterations_stage3,
            learning_rate,
            perplexity_c,
            min_perplexity,
            max_perplexity,
            exaggeration,
            lambda_c_stage1,
            lambda_c_stage2,
            lambda_c_stage3,
            lambda_r_stage3,
            momentum,
            epsilon_r,
            theta,
            power,
            sigma_iters,
            sigma_tolerance,
            row_normalize,
            seed,
            intervals,
            interpolation_order,
            min_grid_dim,
            max_grid_dim,
        )?;
        let inner =
            TsNetSolver::<f32>::from_graph_pure_2hop(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `TsNetSolver` from a `KnnGraph` and optional per-node `perplexities`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromKnn)]
    pub fn from_knn(
        knn: &JsKnnGraph,
        perplexities: Option<Vec<f64>>,
        uniform_perplexity: Option<f64>,
        backend_or_options: &JsValue,
        iterations_stage1: Option<f64>,
        iterations_stage2: Option<f64>,
        iterations_stage3: Option<f64>,
        learning_rate: Option<f64>,
        exaggeration: Option<f64>,
        lambda_c_stage1: Option<f64>,
        lambda_c_stage2: Option<f64>,
        lambda_c_stage3: Option<f64>,
        lambda_r_stage3: Option<f64>,
        momentum: Option<f64>,
        epsilon_r: Option<f64>,
        theta: Option<f64>,
        power: Option<f64>,
        sigma_iters: Option<f64>,
        sigma_tolerance: Option<f64>,
        row_normalize: Option<bool>,
        seed: Option<f64>,
        intervals: Option<f64>,
        interpolation_order: &JsValue,
        min_grid_dim: Option<f64>,
        max_grid_dim: Option<f64>,
    ) -> Result<JsTsNetSolver, JsError> {
        let cfg = parse_tsnet_config(
            backend_or_options,
            iterations_stage1,
            iterations_stage2,
            iterations_stage3,
            learning_rate,
            None,
            None,
            None,
            exaggeration,
            lambda_c_stage1,
            lambda_c_stage2,
            lambda_c_stage3,
            lambda_r_stage3,
            momentum,
            epsilon_r,
            theta,
            power,
            sigma_iters,
            sigma_tolerance,
            row_normalize,
            seed,
            intervals,
            interpolation_order,
            min_grid_dim,
            max_grid_dim,
        )?;
        let uni = uniform_perplexity.unwrap_or(DEFAULT_UNIFORM_PERPLEXITY);
        let inner = TsNetSolver::<f32>::from_knn(knn.inner(), perplexities.as_deref(), uni, cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `TsNetSolver` directly from precomputed symmetric `SparseProbabilities`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromProbabilities)]
    pub fn from_probabilities(
        probs: &JsSparseProbabilities,
        backend_or_options: &JsValue,
        iterations_stage1: Option<f64>,
        iterations_stage2: Option<f64>,
        iterations_stage3: Option<f64>,
        learning_rate: Option<f64>,
        exaggeration: Option<f64>,
        lambda_c_stage1: Option<f64>,
        lambda_c_stage2: Option<f64>,
        lambda_c_stage3: Option<f64>,
        lambda_r_stage3: Option<f64>,
        momentum: Option<f64>,
        epsilon_r: Option<f64>,
        theta: Option<f64>,
        seed: Option<f64>,
        intervals: Option<f64>,
        interpolation_order: &JsValue,
        min_grid_dim: Option<f64>,
        max_grid_dim: Option<f64>,
    ) -> Result<JsTsNetSolver, JsError> {
        let cfg = parse_tsnet_config(
            backend_or_options,
            iterations_stage1,
            iterations_stage2,
            iterations_stage3,
            learning_rate,
            None,
            None,
            None,
            exaggeration,
            lambda_c_stage1,
            lambda_c_stage2,
            lambda_c_stage3,
            lambda_r_stage3,
            momentum,
            epsilon_r,
            theta,
            None,
            None,
            None,
            None,
            seed,
            intervals,
            interpolation_order,
            min_grid_dim,
            max_grid_dim,
        )?;
        let inner = TsNetSolver::<f32>::from_probabilities(probs.inner().clone(), cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `TsNetSolver` from a `CsrGraph` using Partial BFS $k$-NN and a uniform perplexity.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromGraphPartialBfs)]
    pub fn from_graph_partial_bfs(
        graph: &JsCsrGraph,
        k: Option<f64>,
        uniform_perplexity: Option<f64>,
        backend_or_options: &JsValue,
        iterations_stage1: Option<f64>,
        iterations_stage2: Option<f64>,
        iterations_stage3: Option<f64>,
        learning_rate: Option<f64>,
        exaggeration: Option<f64>,
        lambda_c_stage1: Option<f64>,
        lambda_c_stage2: Option<f64>,
        lambda_c_stage3: Option<f64>,
        lambda_r_stage3: Option<f64>,
        momentum: Option<f64>,
        epsilon_r: Option<f64>,
        theta: Option<f64>,
        power: Option<f64>,
        sigma_iters: Option<f64>,
        sigma_tolerance: Option<f64>,
        row_normalize: Option<bool>,
        seed: Option<f64>,
        intervals: Option<f64>,
        interpolation_order: &JsValue,
        min_grid_dim: Option<f64>,
        max_grid_dim: Option<f64>,
    ) -> Result<JsTsNetSolver, JsError> {
        let k_val = parse_optional_usize(k, 30, "k")?;
        let uni = uniform_perplexity.unwrap_or(DEFAULT_UNIFORM_PERPLEXITY);
        let cfg = parse_tsnet_config(
            backend_or_options,
            iterations_stage1,
            iterations_stage2,
            iterations_stage3,
            learning_rate,
            None,
            None,
            None,
            exaggeration,
            lambda_c_stage1,
            lambda_c_stage2,
            lambda_c_stage3,
            lambda_r_stage3,
            momentum,
            epsilon_r,
            theta,
            power,
            sigma_iters,
            sigma_tolerance,
            row_normalize,
            seed,
            intervals,
            interpolation_order,
            min_grid_dim,
            max_grid_dim,
        )?;
        let inner = TsNetSolver::<f32>::from_graph_partial_bfs(graph.inner(), k_val, uni, cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `TsNetSolver` from a `CsrGraph` using direct row-normalized `A1 + w2 * A2` affinity.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromGraphRowNormalized)]
    pub fn from_graph_row_normalized(
        graph: &JsCsrGraph,
        w2_or_options: &JsValue,
        max_2hop_per_node: Option<f64>,
        backend_or_options: &JsValue,
        iterations_stage1: Option<f64>,
        iterations_stage2: Option<f64>,
        iterations_stage3: Option<f64>,
        learning_rate: Option<f64>,
        exaggeration: Option<f64>,
        lambda_c_stage1: Option<f64>,
        lambda_c_stage2: Option<f64>,
        lambda_c_stage3: Option<f64>,
        lambda_r_stage3: Option<f64>,
        momentum: Option<f64>,
        epsilon_r: Option<f64>,
        theta: Option<f64>,
        seed: Option<f64>,
        intervals: Option<f64>,
        interpolation_order: &JsValue,
        min_grid_dim: Option<f64>,
        max_grid_dim: Option<f64>,
        degree_beta: Option<f64>,
    ) -> Result<JsTsNetSolver, JsError> {
        let (w2, max_2hop, beta_val, cfg) = if is_plain_js_object(w2_or_options) {
            let has_extra = max_2hop_per_node.is_some()
                || (!backend_or_options.is_undefined() && !backend_or_options.is_null())
                || iterations_stage1.is_some()
                || iterations_stage2.is_some()
                || iterations_stage3.is_some()
                || learning_rate.is_some()
                || exaggeration.is_some()
                || lambda_c_stage1.is_some()
                || lambda_c_stage2.is_some()
                || lambda_c_stage3.is_some()
                || lambda_r_stage3.is_some()
                || momentum.is_some()
                || epsilon_r.is_some()
                || theta.is_some()
                || seed.is_some()
                || intervals.is_some()
                || (!interpolation_order.is_undefined() && !interpolation_order.is_null())
                || min_grid_dim.is_some()
                || max_grid_dim.is_some()
                || degree_beta.is_some();
            reject_extra_args_with_options(has_extra)?;
            let w2_val = opt_f64_prop(w2_or_options, "w2", 0.5)?;
            let m2 = opt_maybe_usize_prop(w2_or_options, "max2hopPerNode")?;
            let beta = opt_f64_prop(w2_or_options, "degreeBeta", 0.0)?;
            let cfg = parse_tsnet_config(
                w2_or_options,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                &JsValue::UNDEFINED,
                None,
                None,
            )?;
            (w2_val, m2, beta, cfg)
        } else {
            let w2_val = if w2_or_options.is_undefined() || w2_or_options.is_null() {
                0.5
            } else {
                w2_or_options.as_f64().ok_or_else(|| {
                    JsError::new("w2OrOptions must be a finite non-negative number, options object, null, or undefined")
                })?
            };
            let m2 = parse_opt_usize_param(max_2hop_per_node, "max2hopPerNode")?;
            let beta = degree_beta.unwrap_or(0.0);
            let cfg = parse_tsnet_config(
                backend_or_options,
                iterations_stage1,
                iterations_stage2,
                iterations_stage3,
                learning_rate,
                None,
                None,
                None,
                exaggeration,
                lambda_c_stage1,
                lambda_c_stage2,
                lambda_c_stage3,
                lambda_r_stage3,
                momentum,
                epsilon_r,
                theta,
                None,
                None,
                None,
                None,
                seed,
                intervals,
                interpolation_order,
                min_grid_dim,
                max_grid_dim,
            )?;
            (w2_val, m2, beta, cfg)
        };
        let inner = TsNetSolver::<f32>::from_graph_row_normalized_with_degree_decay(
            graph.inner(),
            w2,
            max_2hop,
            beta_val,
            cfg,
        )
        .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns telemetry counters from the most recent FFT field evaluation, or `undefined`.
    #[wasm_bindgen(js_name = lastFftStats)]
    pub fn last_fft_stats(&self) -> Option<JsFitFieldStats> {
        self.inner
            .last_fft_stats()
            .map(|inner| JsFitFieldStats { inner })
    }

    /// Executes one optimization iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "TsNetSolver::run_steps",
        )
    }

    /// Runs optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "TsNetSolver::run",
        )
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of iterations executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the total scheduled iteration count across all stages.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Returns `true` when the solver has converged or completed.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    /// Returns the expected vertex count.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the active (or final) 0-based stage index.
    #[wasm_bindgen(getter, js_name = stageIndex)]
    pub fn stage_index(&self) -> usize {
        self.inner.stage_index()
    }

    /// Returns the cumulative number of `fft_2d` calls executed across all steps.
    #[wasm_bindgen(getter, js_name = totalFft2dCalls)]
    pub fn total_fft_2d_calls(&self) -> f64 {
        self.inner.total_fft_2d_calls() as f64
    }

    /// Returns the stability-capped learning rate of the current stage.
    #[wasm_bindgen(getter, js_name = currentStageLr)]
    pub fn current_stage_lr(&self) -> f64 {
        self.inner.current_stage_lr()
    }
}

// ============================================================================
// 4. SgdSolver & SparseBarrierStats (f32)
// ============================================================================

/// Diagnostics emitted by `SgdSolver.sparseBarrierFromLandmarks` and `SgdSolver.sparseBarrierDog`.
#[wasm_bindgen(js_name = SparseBarrierStats)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsSparseBarrierStats {
    inner: SparseBarrierStats,
}

#[wasm_bindgen(js_class = SparseBarrierStats)]
impl JsSparseBarrierStats {
    /// Returns the count of Tier 1 1-hop pairs.
    #[wasm_bindgen(getter, js_name = nPairs1hop)]
    pub fn n_pairs_1hop(&self) -> usize {
        self.inner.n_pairs_1hop
    }

    /// Returns the count of Tier 2 2-hop pairs.
    #[wasm_bindgen(getter, js_name = nPairs2hop)]
    pub fn n_pairs_2hop(&self) -> usize {
        self.inner.n_pairs_2hop
    }

    /// Returns the count of Tier 3 non-2-hop barrier pairs.
    #[wasm_bindgen(getter, js_name = nPairsNon2)]
    pub fn n_pairs_non2(&self) -> usize {
        self.inner.n_pairs_non2
    }

    /// Returns the count of one-sided hinge pairs.
    #[wasm_bindgen(getter, js_name = nPairsHinge)]
    pub fn n_pairs_hinge(&self) -> usize {
        self.inner.n_pairs_hinge
    }

    /// Returns the count of two-sided spring pairs.
    #[wasm_bindgen(getter, js_name = nPairsTwoSided)]
    pub fn n_pairs_two_sided(&self) -> usize {
        self.inner.n_pairs_two_sided
    }

    /// Returns the adaptive or explicit perplexity value.
    #[wasm_bindgen(getter, js_name = perplexityVal)]
    pub fn perplexity_val(&self) -> f64 {
        self.inner.perplexity_val
    }

    /// Returns the per-node negative sample budget used.
    #[wasm_bindgen(getter, js_name = perNodeNegUsed)]
    pub fn per_node_neg_used(&self) -> usize {
        self.inner.per_node_neg_used
    }

    /// Returns `"adaptive_2hop"` or `"explicit"`.
    #[wasm_bindgen(getter, js_name = perplexitySource)]
    pub fn perplexity_source(&self) -> String {
        self.inner.perplexity_source.to_owned()
    }

    /// Returns the applied relative distance floor.
    #[wasm_bindgen(getter, js_name = dFloor)]
    pub fn d_floor(&self) -> f64 {
        self.inner.d_floor
    }

    /// Returns the effective non-2-hop repulsion multiplier $\gamma_{\mathrm{eff}}$.
    #[wasm_bindgen(getter, js_name = gammaEff)]
    pub fn gamma_eff(&self) -> f64 {
        self.inner.gamma_eff
    }
}

fn parse_sgd_schedule_and_seed(
    t_max_or_options: &JsValue,
    eps: Option<f64>,
    kind: Option<String>,
    step_scale: Option<f64>,
    seed: Option<f64>,
) -> Result<(SgdScheduleConfig, u64), JsError> {
    let mut cfg = SgdScheduleConfig::default();
    if is_plain_js_object(t_max_or_options) {
        let has_extra = eps.is_some() || kind.is_some() || step_scale.is_some() || seed.is_some();
        reject_extra_args_with_options(has_extra)?;
        cfg.t_max = opt_usize_prop(t_max_or_options, "tMax", cfg.t_max)?;
        cfg.eps = opt_f64_prop(t_max_or_options, "eps", cfg.eps)?;
        if let Some(k_str) = opt_string_prop(t_max_or_options, "kind")? {
            cfg.kind = parse_sgd_schedule_kind(&k_str)?;
        }
        cfg.step_scale = opt_f64_prop(t_max_or_options, "stepScale", cfg.step_scale)?;
        let seed_u64 = opt_seed_prop(t_max_or_options)?;
        return Ok((cfg, seed_u64));
    }

    if !t_max_or_options.is_undefined() && !t_max_or_options.is_null() {
        let t_val = t_max_or_options.as_f64().ok_or_else(|| {
            JsError::new(
                "tMaxOrOptions must be a non-negative integer, options object, null, or undefined",
            )
        })?;
        cfg.t_max = parse_non_negative_usize(t_val, "tMax")?;
    }
    cfg.eps = eps.unwrap_or(cfg.eps);
    if let Some(k_str) = kind {
        cfg.kind = parse_sgd_schedule_kind(&k_str)?;
    }
    cfg.step_scale = step_scale.unwrap_or(cfg.step_scale);
    let seed_u64 = parse_seed_u64(seed)?;
    Ok((cfg, seed_u64))
}

#[allow(clippy::too_many_arguments)]
fn parse_sparse_barrier_config(
    max_2hop_or_options: &JsValue,
    hinge_repulsion: Option<bool>,
    delta_rel: Option<f64>,
    barrier_sym: Option<String>,
    omega_1: Option<f64>,
    omega_2: Option<f64>,
    gamma_non2: Option<f64>,
    ht_repulsion: Option<bool>,
    min_ratio: Option<f64>,
    eps_dist: Option<f64>,
    neg_per_node: Option<f64>,
    num_pivots: Option<f64>,
    core_pivots: Option<bool>,
    t_max: Option<f64>,
    eps: Option<f64>,
    kind: Option<String>,
    step_scale: Option<f64>,
    seed: Option<f64>,
    has_dog_extra: bool,
) -> Result<SparseBarrierConfig, JsError> {
    let mut cfg = SparseBarrierConfig::default();
    if is_plain_js_object(max_2hop_or_options) {
        let has_extra = has_dog_extra
            || hinge_repulsion.is_some()
            || delta_rel.is_some()
            || barrier_sym.is_some()
            || omega_1.is_some()
            || omega_2.is_some()
            || gamma_non2.is_some()
            || ht_repulsion.is_some()
            || min_ratio.is_some()
            || eps_dist.is_some()
            || neg_per_node.is_some()
            || num_pivots.is_some()
            || core_pivots.is_some()
            || t_max.is_some()
            || eps.is_some()
            || kind.is_some()
            || step_scale.is_some()
            || seed.is_some();
        reject_extra_args_with_options(has_extra)?;

        cfg.max_2hop_per_node =
            opt_usize_prop(max_2hop_or_options, "max2hopPerNode", cfg.max_2hop_per_node)?;
        cfg.hinge_repulsion =
            opt_bool_prop(max_2hop_or_options, "hingeRepulsion", cfg.hinge_repulsion)?;
        cfg.delta_rel = opt_f64_prop(max_2hop_or_options, "deltaRel", cfg.delta_rel)?;
        if let Some(sym_str) = opt_string_prop(max_2hop_or_options, "barrierSym")? {
            cfg.barrier_sym = parse_barrier_symmetry(&sym_str)?;
        }
        cfg.omega_1 = opt_f64_prop(max_2hop_or_options, "omega1", cfg.omega_1)?;
        cfg.omega_2 = opt_f64_prop(max_2hop_or_options, "omega2", cfg.omega_2)?;
        cfg.gamma_non2 = opt_f64_prop(max_2hop_or_options, "gammaNon2", cfg.gamma_non2)?;
        cfg.ht_repulsion = opt_bool_prop(max_2hop_or_options, "htRepulsion", cfg.ht_repulsion)?;
        cfg.min_ratio = opt_f64_prop(max_2hop_or_options, "minRatio", cfg.min_ratio)?;
        cfg.eps_dist = opt_f64_prop(max_2hop_or_options, "epsDist", cfg.eps_dist)?;
        cfg.neg_per_node = opt_maybe_usize_prop(max_2hop_or_options, "negPerNode")?;
        cfg.num_pivots = opt_usize_prop(max_2hop_or_options, "numPivots", cfg.num_pivots)?;
        cfg.core_pivots = opt_bool_prop(max_2hop_or_options, "corePivots", cfg.core_pivots)?;
        cfg.degree_weight_beta = opt_f64_prop(
            max_2hop_or_options,
            "degreeWeightBeta",
            cfg.degree_weight_beta,
        )?;
        cfg.schedule.t_max = opt_usize_prop(max_2hop_or_options, "tMax", cfg.schedule.t_max)?;
        cfg.schedule.eps = opt_f64_prop(max_2hop_or_options, "eps", cfg.schedule.eps)?;
        if let Some(k_str) = opt_string_prop(max_2hop_or_options, "kind")? {
            cfg.schedule.kind = parse_sgd_schedule_kind(&k_str)?;
        }
        cfg.schedule.step_scale =
            opt_f64_prop(max_2hop_or_options, "stepScale", cfg.schedule.step_scale)?;
        cfg.seed = opt_seed_prop(max_2hop_or_options)?;
        return Ok(cfg);
    }

    if !max_2hop_or_options.is_undefined() && !max_2hop_or_options.is_null() {
        let v = max_2hop_or_options.as_f64().ok_or_else(|| {
            JsError::new("max2hopOrOptions must be a non-negative integer, options object, null, or undefined")
        })?;
        cfg.max_2hop_per_node = parse_non_negative_usize(v, "max2hopPerNode")?;
    }
    cfg.hinge_repulsion = hinge_repulsion.unwrap_or(cfg.hinge_repulsion);
    cfg.delta_rel = delta_rel.unwrap_or(cfg.delta_rel);
    if let Some(sym_str) = barrier_sym {
        cfg.barrier_sym = parse_barrier_symmetry(&sym_str)?;
    }
    cfg.omega_1 = omega_1.unwrap_or(cfg.omega_1);
    cfg.omega_2 = omega_2.unwrap_or(cfg.omega_2);
    cfg.gamma_non2 = gamma_non2.unwrap_or(cfg.gamma_non2);
    cfg.ht_repulsion = ht_repulsion.unwrap_or(cfg.ht_repulsion);
    cfg.min_ratio = min_ratio.unwrap_or(cfg.min_ratio);
    cfg.eps_dist = eps_dist.unwrap_or(cfg.eps_dist);
    cfg.neg_per_node = parse_opt_usize_param(neg_per_node, "negPerNode")?;
    cfg.num_pivots = parse_optional_usize(num_pivots, cfg.num_pivots, "numPivots")?;
    cfg.core_pivots = core_pivots.unwrap_or(cfg.core_pivots);
    cfg.schedule.t_max = parse_optional_usize(t_max, cfg.schedule.t_max, "tMax")?;
    cfg.schedule.eps = eps.unwrap_or(cfg.schedule.eps);
    if let Some(k_str) = kind {
        cfg.schedule.kind = parse_sgd_schedule_kind(&k_str)?;
    }
    cfg.schedule.step_scale = step_scale.unwrap_or(cfg.schedule.step_scale);
    cfg.seed = parse_seed_u64(seed)?;
    Ok(cfg)
}

/// Regularized DoG Varadhan landmark diagnostics returned by `SgdSolver.dogStats()`.
#[wasm_bindgen(js_name = DogStats)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsDogStats {
    inner: DogVaradhanStats,
}

#[wasm_bindgen(js_class = DogStats)]
impl JsDogStats {
    #[wasm_bindgen(getter, js_name = momentStats)]
    pub fn moment_stats(&self) -> JsSpectralMomentStats {
        JsSpectralMomentStats::from_inner(self.inner.moment_stats)
    }

    #[wasm_bindgen(getter)]
    pub fn t1(&self) -> f64 {
        self.inner.t1
    }

    #[wasm_bindgen(getter)]
    pub fn t2(&self) -> f64 {
        self.inner.t2
    }

    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    #[wasm_bindgen(getter)]
    pub fn degree(&self) -> usize {
        self.inner.degree
    }

    #[wasm_bindgen(getter, js_name = deltaPsd)]
    pub fn delta_psd(&self) -> f64 {
        self.inner.delta_psd
    }

    #[wasm_bindgen(getter, js_name = hMin)]
    pub fn h_min(&self) -> f64 {
        self.inner.h_min
    }

    #[wasm_bindgen(getter, js_name = kMin)]
    pub fn k_min(&self) -> f64 {
        self.inner.k_min
    }

    #[wasm_bindgen(getter, js_name = kMed)]
    pub fn k_med(&self) -> f64 {
        self.inner.k_med
    }

    #[wasm_bindgen(getter, js_name = betaAuto)]
    pub fn beta_auto(&self) -> f64 {
        self.inner.beta_auto
    }

    #[wasm_bindgen(getter, js_name = betaEff)]
    pub fn beta_eff(&self) -> f64 {
        self.inner.beta_eff
    }

    #[wasm_bindgen(getter, js_name = numLandmarks)]
    pub fn num_landmarks(&self) -> usize {
        self.inner.n_landmarks
    }

    #[wasm_bindgen(getter, js_name = tCoarse)]
    pub fn t_coarse(&self) -> f64 {
        self.inner.t_coarse
    }

    #[wasm_bindgen(getter, js_name = usedCoarseScaffold)]
    pub fn used_coarse_scaffold(&self) -> bool {
        self.inner.used_coarse_scaffold
    }
}

fn default_inv_sq_weights_f32(dists: &[f32]) -> Vec<f32> {
    dists
        .iter()
        .map(|&d| {
            let df = f64::from(d);
            if df.is_finite() && df > 0.0 {
                (1.0 / (df * df)) as f32
            } else {
                0.0
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_dog_config_from_options_or_args(
    maybe_options: &JsValue,
    seed: u64,
    degree: Option<f64>,
    num_landmarks: Option<f64>,
    degree_normalize: Option<bool>,
    p_exp: Option<f64>,
    eps_psd: Option<f64>,
    t_coarse: Option<f64>,
    t1: Option<f64>,
    t2: Option<f64>,
    alpha: Option<f64>,
    beta_floor: Option<f64>,
    min_degree: Option<f64>,
    jl_dim: Option<f64>,
) -> Result<DogVaradhanConfig, JsError> {
    let mut cfg = DogVaradhanConfig {
        seed,
        ..DogVaradhanConfig::default()
    };
    if is_plain_js_object(maybe_options) {
        let deg_primary = opt_maybe_usize_prop(maybe_options, "degree")?;
        let deg_alt = opt_maybe_usize_prop(maybe_options, "dogDegree")?;
        cfg.degree = deg_primary.or(deg_alt);

        let lm_primary = opt_maybe_usize_prop(maybe_options, "numLandmarks")?;
        let lm_alt = opt_maybe_usize_prop(maybe_options, "dogNumLandmarks")?;
        cfg.n_landmarks = lm_primary.or(lm_alt).unwrap_or(cfg.n_landmarks);

        cfg.min_degree = opt_usize_prop(maybe_options, "minDegree", cfg.min_degree)?;
        cfg.jl_dim = opt_usize_prop(maybe_options, "jlDim", cfg.jl_dim)?;

        let dn_val = get_prop(maybe_options, "degreeNormalize")?;
        let dn_alt = get_prop(maybe_options, "dogDegreeNormalize")?;
        if !dn_val.is_undefined() && !dn_val.is_null() {
            cfg.degree_normalize = dn_val
                .as_bool()
                .ok_or_else(|| JsError::new("options.degreeNormalize must be a boolean"))?;
        } else if !dn_alt.is_undefined() && !dn_alt.is_null() {
            cfg.degree_normalize = dn_alt
                .as_bool()
                .ok_or_else(|| JsError::new("options.dogDegreeNormalize must be a boolean"))?;
        }

        let p_primary = opt_maybe_f64_prop(maybe_options, "pExp")?;
        let p_alt = opt_maybe_f64_prop(maybe_options, "dogPExp")?;
        cfg.p_exp = p_primary.or(p_alt).unwrap_or(cfg.p_exp);

        let psd_primary = opt_maybe_f64_prop(maybe_options, "epsPsd")?;
        let psd_alt = opt_maybe_f64_prop(maybe_options, "dogEpsPsd")?;
        cfg.eps_psd = psd_primary.or(psd_alt).unwrap_or(cfg.eps_psd);

        let tc_primary = opt_maybe_f64_prop(maybe_options, "tCoarse")?;
        let tc_alt = opt_maybe_f64_prop(maybe_options, "dogTCoarse")?;
        cfg.t_coarse = tc_primary.or(tc_alt).unwrap_or(cfg.t_coarse);

        cfg.t1 = opt_maybe_f64_prop(maybe_options, "t1")?
            .or(opt_maybe_f64_prop(maybe_options, "dogT1")?);
        cfg.t2 = opt_maybe_f64_prop(maybe_options, "t2")?
            .or(opt_maybe_f64_prop(maybe_options, "dogT2")?);
        cfg.alpha = opt_maybe_f64_prop(maybe_options, "alpha")?
            .or(opt_maybe_f64_prop(maybe_options, "dogAlpha")?);
        cfg.beta_floor = opt_maybe_f64_prop(maybe_options, "betaFloor")?
            .or(opt_maybe_f64_prop(maybe_options, "dogBetaFloor")?);
        return Ok(cfg);
    }

    cfg.degree = parse_opt_usize_param(degree, "degree")?;
    cfg.n_landmarks = parse_optional_usize(num_landmarks, cfg.n_landmarks, "numLandmarks")?;
    cfg.min_degree = parse_optional_usize(min_degree, cfg.min_degree, "minDegree")?;
    cfg.jl_dim = parse_optional_usize(jl_dim, cfg.jl_dim, "jlDim")?;
    cfg.degree_normalize = degree_normalize.unwrap_or(cfg.degree_normalize);
    cfg.p_exp = p_exp.unwrap_or(cfg.p_exp);
    cfg.eps_psd = eps_psd.unwrap_or(cfg.eps_psd);
    cfg.t_coarse = t_coarse.unwrap_or(cfg.t_coarse);
    cfg.t1 = t1;
    cfg.t2 = t2;
    cfg.alpha = alpha;
    cfg.beta_floor = beta_floor;
    Ok(cfg)
}

/// Cooperative SoA `u32` SGD solver supporting 3-tier sparse barrier, streaming negatives,
/// Ortmann Sparse Stress, LargeVis, DRGraph, Graph-CNE, SNAP-tFDP, PaCMAP-Graph,
/// diagonal-preconditioned mini-batch stress SGD (`sgd2`), Euclidean `Layout2d` (`f32`),
/// and non-Euclidean `ManifoldLayout2d` (`f32`).
#[wasm_bindgen(js_name = SgdSolver)]
#[derive(Debug, Clone)]
pub struct JsSgdSolver {
    inner: SgdSolver<f32>,
    barrier_stats: Option<SparseBarrierStats>,
    dog_stats: Option<DogVaradhanStats>,
}

impl JsSgdSolver {
    pub(crate) fn from_inner(inner: SgdSolver<f32>) -> Self {
        Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
        }
    }
}

#[wasm_bindgen(js_class = SgdSolver)]
impl JsSgdSolver {
    /// Constructs an `SgdSolver` from explicit flat SoA two-sided spring arrays.
    ///
    /// When `dji` is omitted, defaults to `dij`. When `wij`/`wji` are omitted, defaults to $1 / d^2$.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromPairs)]
    pub fn from_pairs(
        num_nodes: f64,
        src: &[u32],
        tgt: &[u32],
        dij: &[f32],
        dji: Option<Vec<f32>>,
        wij: Option<Vec<f32>>,
        wji: Option<Vec<f32>>,
        t_max_or_options: &JsValue,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsSgdSolver, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let (sched, seed_u64) =
            parse_sgd_schedule_and_seed(t_max_or_options, eps, kind, step_scale, seed)?;
        let dji_vec = dji.unwrap_or_else(|| dij.to_vec());
        let wij_was_none = wij.is_none();
        let wij_vec = wij.unwrap_or_else(|| default_inv_sq_weights_f32(dij));
        let wji_vec = wji.unwrap_or_else(|| {
            if wij_was_none {
                default_inv_sq_weights_f32(&dji_vec)
            } else {
                wij_vec.clone()
            }
        });
        let inner = SgdSolver::<f32>::from_pairs(
            n,
            src.to_vec(),
            tgt.to_vec(),
            dij.to_vec(),
            dji_vec,
            wij_vec,
            wji_vec,
            sched,
            seed_u64,
        )
        .map_err(core_err_to_js)?;
        Ok(Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
        })
    }

    /// Constructs an `SgdSolver` combining two-sided spring pairs and one-sided hinge pairs.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromPairsWithHinges)]
    pub fn from_pairs_with_hinges(
        num_nodes: f64,
        spring_src: &[u32],
        spring_tgt: &[u32],
        spring_dij: &[f32],
        spring_dji: &[f32],
        spring_wij: &[f32],
        spring_wji: &[f32],
        hinge_src: &[u32],
        hinge_tgt: &[u32],
        hinge_dij: &[f32],
        hinge_dji: &[f32],
        hinge_wij: &[f32],
        hinge_wji: &[f32],
        t_max_or_options: &JsValue,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsSgdSolver, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let (sched, seed_u64) =
            parse_sgd_schedule_and_seed(t_max_or_options, eps, kind, step_scale, seed)?;
        let inner = SgdSolver::<f32>::from_pairs_with_hinges(
            n,
            spring_src.to_vec(),
            spring_tgt.to_vec(),
            spring_dij.to_vec(),
            spring_dji.to_vec(),
            spring_wij.to_vec(),
            spring_wji.to_vec(),
            hinge_src.to_vec(),
            hinge_tgt.to_vec(),
            hinge_dij.to_vec(),
            hinge_dji.to_vec(),
            hinge_wij.to_vec(),
            hinge_wji.to_vec(),
            sched,
            seed_u64,
        )
        .map_err(core_err_to_js)?;
        Ok(Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
        })
    }

    /// Constructs a full all-pairs shortest-path SGD solver from `graph`.
    pub fn full(
        graph: &JsCsrGraph,
        t_max_or_options: &JsValue,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsSgdSolver, JsError> {
        let (sched, seed_u64) =
            parse_sgd_schedule_and_seed(t_max_or_options, eps, kind, step_scale, seed)?;
        let inner =
            SgdSolver::<f32>::full(graph.inner(), sched, seed_u64).map_err(core_err_to_js)?;
        Ok(Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
        })
    }

    /// Constructs a Track B 3-tier Sparse Barrier SGD solver from precomputed `DogVaradhanLandmarks`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = sparseBarrierFromLandmarks)]
    pub fn sparse_barrier_from_landmarks(
        graph: &JsCsrGraph,
        landmarks: &JsDogVaradhanLandmarks,
        pivots: Option<Vec<u32>>,
        max_2hop_or_options: &JsValue,
        hinge_repulsion: Option<bool>,
        delta_rel: Option<f64>,
        barrier_sym: Option<String>,
        omega_1: Option<f64>,
        omega_2: Option<f64>,
        gamma_non2: Option<f64>,
        ht_repulsion: Option<bool>,
        min_ratio: Option<f64>,
        eps_dist: Option<f64>,
        neg_per_node: Option<f64>,
        num_pivots: Option<f64>,
        core_pivots: Option<bool>,
        t_max: Option<f64>,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsSgdSolver, JsError> {
        let cfg = parse_sparse_barrier_config(
            max_2hop_or_options,
            hinge_repulsion,
            delta_rel,
            barrier_sym,
            omega_1,
            omega_2,
            gamma_non2,
            ht_repulsion,
            min_ratio,
            eps_dist,
            neg_per_node,
            num_pivots,
            core_pivots,
            t_max,
            eps,
            kind,
            step_scale,
            seed,
            false,
        )?;
        let (inner, stats) = SgdSolver::<f32>::sparse_barrier_from_landmarks(
            graph.inner(),
            landmarks.inner(),
            pivots.as_deref(),
            &cfg,
        )
        .map_err(core_err_to_js)?;
        Ok(Self {
            inner,
            barrier_stats: Some(stats),
            dog_stats: None,
        })
    }

    /// Computes regularized Chebyshev DoG Varadhan landmarks and constructs a 3-tier Sparse Barrier SGD solver.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = sparseBarrierDog)]
    pub fn sparse_barrier_dog(
        graph: &JsCsrGraph,
        max_2hop_or_options: &JsValue,
        hinge_repulsion: Option<bool>,
        delta_rel: Option<f64>,
        barrier_sym: Option<String>,
        omega_1: Option<f64>,
        omega_2: Option<f64>,
        gamma_non2: Option<f64>,
        ht_repulsion: Option<bool>,
        min_ratio: Option<f64>,
        eps_dist: Option<f64>,
        neg_per_node: Option<f64>,
        num_pivots: Option<f64>,
        core_pivots: Option<bool>,
        t_max: Option<f64>,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
        degree: Option<f64>,
        num_landmarks: Option<f64>,
        degree_normalize: Option<bool>,
        p_exp: Option<f64>,
        eps_psd: Option<f64>,
        t_coarse: Option<f64>,
        t1: Option<f64>,
        t2: Option<f64>,
        alpha: Option<f64>,
        beta_floor: Option<f64>,
        min_degree: Option<f64>,
        jl_dim: Option<f64>,
    ) -> Result<JsSgdSolver, JsError> {
        let has_dog_extra = degree.is_some()
            || num_landmarks.is_some()
            || degree_normalize.is_some()
            || p_exp.is_some()
            || eps_psd.is_some()
            || t_coarse.is_some()
            || t1.is_some()
            || t2.is_some()
            || alpha.is_some()
            || beta_floor.is_some()
            || min_degree.is_some()
            || jl_dim.is_some();
        let barrier_cfg = parse_sparse_barrier_config(
            max_2hop_or_options,
            hinge_repulsion,
            delta_rel,
            barrier_sym,
            omega_1,
            omega_2,
            gamma_non2,
            ht_repulsion,
            min_ratio,
            eps_dist,
            neg_per_node,
            num_pivots,
            core_pivots,
            t_max,
            eps,
            kind,
            step_scale,
            seed,
            has_dog_extra,
        )?;
        let dog_cfg = parse_dog_config_from_options_or_args(
            max_2hop_or_options,
            barrier_cfg.seed,
            degree,
            num_landmarks,
            degree_normalize,
            p_exp,
            eps_psd,
            t_coarse,
            t1,
            t2,
            alpha,
            beta_floor,
            min_degree,
            jl_dim,
        )?;
        let (inner, stats, dog_stats) =
            SgdSolver::<f32>::sparse_barrier_dog(graph.inner(), &dog_cfg, &barrier_cfg)
                .map_err(core_err_to_js)?;
        Ok(Self {
            inner,
            barrier_stats: Some(stats),
            dog_stats: Some(dog_stats),
        })
    }

    /// Constructs a random-pair sparse SGD solver combining 1-hop edges with sampled random pairs.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = randomPair)]
    pub fn random_pair(
        graph: &JsCsrGraph,
        pairs_per_node_or_options: &JsValue,
        t_max_or_options: &JsValue,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsSgdSolver, JsError> {
        let (pairs_per_node, sched, seed_u64) = if is_plain_js_object(pairs_per_node_or_options) {
            let has_extra = (!t_max_or_options.is_undefined() && !t_max_or_options.is_null())
                || eps.is_some()
                || kind.is_some()
                || step_scale.is_some()
                || seed.is_some();
            reject_extra_args_with_options(has_extra)?;
            let ppn = opt_usize_prop(
                pairs_per_node_or_options,
                "pairsPerNode",
                DEFAULT_RANDOM_PAIRS_PER_NODE,
            )?;
            let (s, sd) =
                parse_sgd_schedule_and_seed(pairs_per_node_or_options, None, None, None, None)?;
            (ppn, s, sd)
        } else {
            let ppn = if pairs_per_node_or_options.is_undefined()
                || pairs_per_node_or_options.is_null()
            {
                DEFAULT_RANDOM_PAIRS_PER_NODE
            } else {
                let v = pairs_per_node_or_options.as_f64().ok_or_else(|| {
                    JsError::new("pairsPerNodeOrOptions must be a non-negative integer, options object, null, or undefined")
                })?;
                parse_non_negative_usize(v, "pairsPerNode")?
            };
            let (s, sd) =
                parse_sgd_schedule_and_seed(t_max_or_options, eps, kind, step_scale, seed)?;
            (ppn, s, sd)
        };
        let inner = SgdSolver::<f32>::random_pair(graph.inner(), pairs_per_node, sched, seed_u64)
            .map_err(core_err_to_js)?;
        Ok(Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
        })
    }

    /// Constructs an Omega SGD solver from a flat row-major `(numNodes, dims)` spectral embedding.
    #[allow(clippy::too_many_arguments)]
    pub fn omega(
        graph: &JsCsrGraph,
        embedding: &[f64],
        dims: f64,
        k_or_options: &JsValue,
        min_dist: Option<f64>,
        t_max_or_options: &JsValue,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsSgdSolver, JsError> {
        let dims_val = parse_non_negative_usize(dims, "dims")?;
        let (k_val, md_val, sched, seed_u64) = if is_plain_js_object(k_or_options) {
            let has_extra = min_dist.is_some()
                || (!t_max_or_options.is_undefined() && !t_max_or_options.is_null())
                || eps.is_some()
                || kind.is_some()
                || step_scale.is_some()
                || seed.is_some();
            reject_extra_args_with_options(has_extra)?;
            let k_v = opt_usize_prop(k_or_options, "k", DEFAULT_RANDOM_PAIRS_PER_NODE)?;
            let md_v = opt_f64_prop(k_or_options, "minDist", DEFAULT_OMEGA_MIN_DIST)?;
            let (s, sd) = parse_sgd_schedule_and_seed(k_or_options, None, None, None, None)?;
            (k_v, md_v, s, sd)
        } else {
            let k_v = if k_or_options.is_undefined() || k_or_options.is_null() {
                DEFAULT_RANDOM_PAIRS_PER_NODE
            } else {
                let v = k_or_options.as_f64().ok_or_else(|| {
                    JsError::new(
                        "kOrOptions must be a non-negative integer, options object, null, or undefined",
                    )
                })?;
                parse_non_negative_usize(v, "k")?
            };
            let md_v = min_dist.unwrap_or(DEFAULT_OMEGA_MIN_DIST);
            let (s, sd) =
                parse_sgd_schedule_and_seed(t_max_or_options, eps, kind, step_scale, seed)?;
            (k_v, md_v, s, sd)
        };
        let inner = SgdSolver::<f32>::omega(
            graph.inner(),
            embedding,
            dims_val,
            k_val,
            md_val,
            sched,
            seed_u64,
        )
        .map_err(core_err_to_js)?;
        Ok(Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
        })
    }

    /// Constructs a minimal-RAM SGD solver storing 1-hop springs and streaming non-edge negatives.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = streamingNegatives)]
    pub fn streaming_negatives(
        graph: &JsCsrGraph,
        negatives_or_options: &JsValue,
        target_distance: Option<f64>,
        weight: Option<f64>,
        hinge_only: Option<bool>,
        t_max: Option<f64>,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsSgdSolver, JsError> {
        let mut neg_cfg = StreamingNegativeConfig::default();
        let mut sched = SgdScheduleConfig::default();
        let seed_u64 = if is_plain_js_object(negatives_or_options) {
            let has_extra = target_distance.is_some()
                || weight.is_some()
                || hinge_only.is_some()
                || t_max.is_some()
                || eps.is_some()
                || kind.is_some()
                || step_scale.is_some()
                || seed.is_some();
            reject_extra_args_with_options(has_extra)?;
            neg_cfg.negatives_per_node = opt_usize_prop(
                negatives_or_options,
                "negativesPerNode",
                neg_cfg.negatives_per_node,
            )?;
            neg_cfg.target_distance = opt_f64_prop(
                negatives_or_options,
                "targetDistance",
                neg_cfg.target_distance,
            )?;
            neg_cfg.weight = opt_f64_prop(negatives_or_options, "weight", neg_cfg.weight)?;
            neg_cfg.hinge_only =
                opt_bool_prop(negatives_or_options, "hingeOnly", neg_cfg.hinge_only)?;
            let (s, sd) =
                parse_sgd_schedule_and_seed(negatives_or_options, None, None, None, None)?;
            sched = s;
            sd
        } else {
            if !negatives_or_options.is_undefined() && !negatives_or_options.is_null() {
                let v = negatives_or_options.as_f64().ok_or_else(|| {
                    JsError::new("negativesOrOptions must be a non-negative integer, options object, null, or undefined")
                })?;
                neg_cfg.negatives_per_node = parse_non_negative_usize(v, "negativesPerNode")?;
            }
            neg_cfg.target_distance = target_distance.unwrap_or(neg_cfg.target_distance);
            neg_cfg.weight = weight.unwrap_or(neg_cfg.weight);
            neg_cfg.hinge_only = hinge_only.unwrap_or(neg_cfg.hinge_only);
            sched.t_max = parse_optional_usize(t_max, sched.t_max, "tMax")?;
            sched.eps = eps.unwrap_or(sched.eps);
            if let Some(k_str) = kind {
                sched.kind = parse_sgd_schedule_kind(&k_str)?;
            }
            sched.step_scale = step_scale.unwrap_or(sched.step_scale);
            parse_seed_u64(seed)?
        };
        let inner = SgdSolver::<f32>::streaming_negatives(graph.inner(), neg_cfg, sched, seed_u64)
            .map_err(core_err_to_js)?;
        Ok(Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
        })
    }

    /// Constructs a compact-memory edge-incident negative sampling `SNAP-tFDP` solver.
    #[wasm_bindgen(js_name = snapTFdp)]
    pub fn snap_t_fdp(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsError> {
        crate::v6_m4::sgd_snap_t_fdp(graph, options)
    }

    /// Constructs a 3-tier `PaCMAP-Graph` (`NB` / `MN` / `FP`) Adam layout solver.
    #[wasm_bindgen(js_name = pacmapGraph)]
    pub fn pacmap_graph(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsError> {
        crate::v6_m4::sgd_pacmap_graph(graph, options)
    }

    /// Constructs a 3-tier `PaCMAP-Graph` Adam layout solver using precomputed `DogVaradhanLandmarks`.
    #[wasm_bindgen(js_name = pacmapGraphWithLandmarks)]
    pub fn pacmap_graph_with_landmarks(
        graph: &JsCsrGraph,
        landmarks: &crate::v6_m2::JsDogVaradhanLandmarks,
        options: &JsValue,
    ) -> Result<JsSgdSolver, JsError> {
        crate::v6_m4::sgd_pacmap_graph_with_landmarks(graph, landmarks, options)
    }

    /// Constructs a 3-tier `PaCMAP-Graph` Adam layout solver using a flat row-major
    /// `(num_nodes, num_landmarks)` landmark distance table.
    #[wasm_bindgen(js_name = pacmapGraphWithLandmarkDistances)]
    pub fn pacmap_graph_with_landmark_distances(
        graph: &JsCsrGraph,
        landmark_distances: &[f64],
        num_landmarks: f64,
        options: &JsValue,
    ) -> Result<JsSgdSolver, JsError> {
        crate::v6_m4::sgd_pacmap_graph_with_landmark_distances(
            graph,
            landmark_distances,
            num_landmarks,
            options,
        )
    }

    /// Constructs a 3-tier `PaCMAP-Graph` solver coupled with a coarse Louvain area-reserved
    /// backbone initialization and `num_landmarks`-BFS Mid-Near (`MN`) landmarks.
    #[wasm_bindgen(js_name = pacmapGraphWithBackbone)]
    pub fn pacmap_graph_with_backbone(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<crate::v6_m5::JsPacmapBackboneResult, JsError> {
        crate::v6_m5::sgd_pacmap_graph_with_backbone(graph, options)
    }

    /// Constructs a doubly-stochastic mini-batch diagonal-preconditioned stress SGD solver
    /// (`SgdScheduleKind::Squared`, distinct from Ahmed et al. 2021 multi-criteria `(SGD)^2`).
    pub fn sgd2(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsError> {
        crate::v6_m4::sgd_sgd2(graph, options)
    }

    /// Constructs an Ortmann et al. (2016) Sparse Stress SGD solver.
    #[wasm_bindgen(js_name = ortmannSparseStress)]
    pub fn ortmann_sparse_stress(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsSgdSolver, JsError> {
        crate::v6_w2::sgd_ortmann_sparse_stress(graph, options)
    }

    /// Constructs a single-threaded deterministic `LargeVis` (Tang et al., 2016) solver.
    pub fn largevis(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsError> {
        crate::v6_w2::sgd_largevis(graph, options)
    }

    /// Constructs a multilevel negative-sampling `DRGraph` (Zhu et al., 2020) solver.
    pub fn drgraph(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsError> {
        crate::v6_w2::sgd_drgraph(graph, options)
    }

    /// Constructs a `Graph-CNE` (Bohm et al., 2025) contrastive neighbor embedding solver.
    #[wasm_bindgen(js_name = graphCne)]
    pub fn graph_cne(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsError> {
        crate::v6_w2::sgd_graph_cne(graph, options)
    }

    /// Executes one Euclidean SGD iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` Euclidean SGD iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "SgdSolver::run_steps",
        )
    }

    /// Runs Euclidean SGD optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "SgdSolver::run",
        )
    }

    /// Executes one non-Euclidean SGD iteration on a `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsError> {
        let status = self
            .inner
            .step_manifold(layout.inner_mut())
            .map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Executes up to `steps` non-Euclidean SGD iterations on a `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = runStepsManifold)]
    pub fn run_steps_manifold(
        &mut self,
        layout: &mut JsManifoldLayout2d,
        steps: f64,
    ) -> Result<String, JsError> {
        let n_steps = parse_non_negative_usize(steps, "steps")?;
        if layout.len() != self.inner.num_nodes() {
            return Err(core_err_to_js(CoreError::dimension_mismatch(
                self.inner.num_nodes(),
                layout.len(),
                "SgdSolver::run_steps_manifold",
            )));
        }
        layout
            .inner()
            .as_layout()
            .validate_finite()
            .map_err(core_err_to_js)?;
        let status = self
            .inner
            .run_steps_manifold(layout.inner_mut(), n_steps)
            .map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Runs non-Euclidean SGD optimization to completion on a `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsError> {
        if layout.len() != self.inner.num_nodes() {
            return Err(core_err_to_js(CoreError::dimension_mismatch(
                self.inner.num_nodes(),
                layout.len(),
                "SgdSolver::run_manifold",
            )));
        }
        layout
            .inner()
            .as_layout()
            .validate_finite()
            .map_err(core_err_to_js)?;
        let status = self
            .inner
            .run_manifold(layout.inner_mut())
            .map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Returns `SparseBarrierStats` when constructed via `sparseBarrierFromLandmarks` or
    /// `sparseBarrierDog`, or `undefined` otherwise.
    pub fn stats(&self) -> Option<JsSparseBarrierStats> {
        self.barrier_stats
            .map(|inner| JsSparseBarrierStats { inner })
    }

    /// Returns regularized DoG Varadhan landmark diagnostics `DogStats` when constructed via
    /// `sparseBarrierDog`, or `undefined` otherwise.
    #[wasm_bindgen(js_name = dogStats)]
    pub fn dog_stats(&self) -> Option<JsDogStats> {
        self.dog_stats.map(|inner| JsDogStats { inner })
    }

    /// Returns the number of two-sided spring pairs stored.
    #[wasm_bindgen(getter, js_name = numSprings)]
    pub fn num_springs(&self) -> usize {
        self.inner.num_springs()
    }

    /// Returns the number of one-sided hinge repulsion pairs stored.
    #[wasm_bindgen(getter, js_name = numHinges)]
    pub fn num_hinges(&self) -> usize {
        self.inner.num_hinges()
    }

    /// Returns the learning rate $\eta$ for the upcoming iteration (`0.0` once finished).
    #[wasm_bindgen(getter, js_name = currentEta)]
    pub fn current_eta(&self) -> f64 {
        self.inner.current_eta()
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of iterations executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the scheduled maximum iteration count (`tMax`).
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Returns `true` when the solver has converged or completed.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    /// Returns the expected vertex count.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }
}

// ============================================================================
// 5. GuMapSolver (f32)
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn parse_gumap_config(
    n_epochs_or_options: &JsValue,
    negative_sample_rate: Option<f64>,
    learning_rate: Option<f64>,
    min_dist: Option<f64>,
    spread: Option<f64>,
    a: Option<f64>,
    b: Option<f64>,
    sigma_iters: Option<f64>,
    sigma_tolerance: Option<f64>,
    seed: Option<f64>,
    has_extra_args: bool,
) -> Result<GuMapConfig, JsError> {
    let mut cfg = GuMapConfig::default();
    if is_plain_js_object(n_epochs_or_options) {
        let has_extra = has_extra_args
            || negative_sample_rate.is_some()
            || learning_rate.is_some()
            || min_dist.is_some()
            || spread.is_some()
            || a.is_some()
            || b.is_some()
            || sigma_iters.is_some()
            || sigma_tolerance.is_some()
            || seed.is_some();
        reject_extra_args_with_options(has_extra)?;

        cfg.n_epochs = opt_usize_prop(n_epochs_or_options, "nEpochs", cfg.n_epochs)?;
        cfg.negative_sample_rate = opt_usize_prop(
            n_epochs_or_options,
            "negativeSampleRate",
            cfg.negative_sample_rate,
        )?;
        cfg.learning_rate = opt_f64_prop(n_epochs_or_options, "learningRate", cfg.learning_rate)?;
        cfg.min_dist = opt_f64_prop(n_epochs_or_options, "minDist", cfg.min_dist)?;
        cfg.spread = opt_f64_prop(n_epochs_or_options, "spread", cfg.spread)?;
        cfg.a = opt_maybe_f64_prop(n_epochs_or_options, "a")?;
        cfg.b = opt_maybe_f64_prop(n_epochs_or_options, "b")?;
        cfg.sigma_iters = opt_usize_prop(n_epochs_or_options, "sigmaIters", cfg.sigma_iters)?;
        cfg.sigma_tolerance =
            opt_f64_prop(n_epochs_or_options, "sigmaTolerance", cfg.sigma_tolerance)?;
        cfg.symmetric_negative = opt_bool_prop(
            n_epochs_or_options,
            "symmetricNegative",
            cfg.symmetric_negative,
        )?;
        cfg.seed = opt_seed_prop(n_epochs_or_options)?;
        return Ok(cfg);
    }

    if !n_epochs_or_options.is_undefined() && !n_epochs_or_options.is_null() {
        let v = n_epochs_or_options.as_f64().ok_or_else(|| {
            JsError::new("nEpochsOrOptions must be a non-negative integer, options object, null, or undefined")
        })?;
        cfg.n_epochs = parse_non_negative_usize(v, "nEpochs")?;
    }
    cfg.negative_sample_rate = parse_optional_usize(
        negative_sample_rate,
        cfg.negative_sample_rate,
        "negativeSampleRate",
    )?;
    cfg.learning_rate = learning_rate.unwrap_or(cfg.learning_rate);
    cfg.min_dist = min_dist.unwrap_or(cfg.min_dist);
    cfg.spread = spread.unwrap_or(cfg.spread);
    cfg.a = a;
    cfg.b = b;
    cfg.sigma_iters = parse_optional_usize(sigma_iters, cfg.sigma_iters, "sigmaIters")?;
    cfg.sigma_tolerance = sigma_tolerance.unwrap_or(cfg.sigma_tolerance);
    cfg.seed = parse_seed_u64(seed)?;
    Ok(cfg)
}

/// Cooperative Graph UMAP (`GuMap`) layout solver (`f32`).
#[wasm_bindgen(js_name = GuMapSolver)]
#[derive(Debug, Clone)]
pub struct JsGuMapSolver {
    inner: GuMapSolver<f32>,
}

#[wasm_bindgen(js_class = GuMapSolver)]
impl JsGuMapSolver {
    /// Constructs a `GuMapSolver` directly from a fuzzy simplicial set `SparseProbabilities`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromFuzzySet)]
    pub fn from_fuzzy_set(
        probs: &JsSparseProbabilities,
        n_epochs_or_options: &JsValue,
        negative_sample_rate: Option<f64>,
        learning_rate: Option<f64>,
        min_dist: Option<f64>,
        spread: Option<f64>,
        a: Option<f64>,
        b: Option<f64>,
        seed: Option<f64>,
        sigma_iters: Option<f64>,
        sigma_tolerance: Option<f64>,
    ) -> Result<JsGuMapSolver, JsError> {
        let cfg = parse_gumap_config(
            n_epochs_or_options,
            negative_sample_rate,
            learning_rate,
            min_dist,
            spread,
            a,
            b,
            sigma_iters,
            sigma_tolerance,
            seed,
            false,
        )?;
        let inner =
            GuMapSolver::<f32>::from_fuzzy_set_ref(probs.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `GuMapSolver` from a `KnnGraph` via `computeFuzzySimplicialSet`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromKnn)]
    pub fn from_knn(
        knn: &JsKnnGraph,
        n_epochs_or_options: &JsValue,
        negative_sample_rate: Option<f64>,
        learning_rate: Option<f64>,
        min_dist: Option<f64>,
        spread: Option<f64>,
        a: Option<f64>,
        b: Option<f64>,
        sigma_iters: Option<f64>,
        sigma_tolerance: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsGuMapSolver, JsError> {
        let cfg = parse_gumap_config(
            n_epochs_or_options,
            negative_sample_rate,
            learning_rate,
            min_dist,
            spread,
            a,
            b,
            sigma_iters,
            sigma_tolerance,
            seed,
            false,
        )?;
        let inner = GuMapSolver::<f32>::from_knn(knn.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `GuMapSolver` from a `CsrGraph` using 1+2-hop pairs and DoG Varadhan landmarks.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromGraphDog)]
    pub fn from_graph_dog(
        graph: &JsCsrGraph,
        max_2hop_or_options: &JsValue,
        n_epochs_or_options: &JsValue,
        negative_sample_rate: Option<f64>,
        learning_rate: Option<f64>,
        min_dist: Option<f64>,
        spread: Option<f64>,
        a: Option<f64>,
        b: Option<f64>,
        sigma_iters: Option<f64>,
        sigma_tolerance: Option<f64>,
        seed: Option<f64>,
        degree: Option<f64>,
        num_landmarks: Option<f64>,
        degree_normalize: Option<bool>,
        p_exp: Option<f64>,
        eps_psd: Option<f64>,
        t_coarse: Option<f64>,
        t1: Option<f64>,
        t2: Option<f64>,
        alpha: Option<f64>,
        beta_floor: Option<f64>,
        min_degree: Option<f64>,
        jl_dim: Option<f64>,
    ) -> Result<JsGuMapSolver, JsError> {
        let has_dog_extra = degree.is_some()
            || num_landmarks.is_some()
            || degree_normalize.is_some()
            || p_exp.is_some()
            || eps_psd.is_some()
            || t_coarse.is_some()
            || t1.is_some()
            || t2.is_some()
            || alpha.is_some()
            || beta_floor.is_some()
            || min_degree.is_some()
            || jl_dim.is_some();

        let (max_2hop, gumap_cfg, dog_cfg) = if is_plain_js_object(max_2hop_or_options) {
            let has_extra = has_dog_extra
                || (!n_epochs_or_options.is_undefined() && !n_epochs_or_options.is_null());
            let m2 = opt_usize_prop(
                max_2hop_or_options,
                "max2hopPerNode",
                DEFAULT_GUMAP_MAX_2HOP,
            )?;
            let g_cfg = parse_gumap_config(
                max_2hop_or_options,
                negative_sample_rate,
                learning_rate,
                min_dist,
                spread,
                a,
                b,
                sigma_iters,
                sigma_tolerance,
                seed,
                has_extra,
            )?;
            let d_cfg = parse_dog_config_from_options_or_args(
                max_2hop_or_options,
                g_cfg.seed,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )?;
            (m2, g_cfg, d_cfg)
        } else {
            let m2 = if max_2hop_or_options.is_undefined() || max_2hop_or_options.is_null() {
                DEFAULT_GUMAP_MAX_2HOP
            } else {
                let v = max_2hop_or_options.as_f64().ok_or_else(|| {
                    JsError::new("max2hopOrOptions must be a non-negative integer, options object, null, or undefined")
                })?;
                parse_non_negative_usize(v, "max2hopPerNode")?
            };
            let g_cfg = parse_gumap_config(
                n_epochs_or_options,
                negative_sample_rate,
                learning_rate,
                min_dist,
                spread,
                a,
                b,
                sigma_iters,
                sigma_tolerance,
                seed,
                has_dog_extra && is_plain_js_object(n_epochs_or_options),
            )?;
            let d_cfg = parse_dog_config_from_options_or_args(
                n_epochs_or_options,
                g_cfg.seed,
                degree,
                num_landmarks,
                degree_normalize,
                p_exp,
                eps_psd,
                t_coarse,
                t1,
                t2,
                alpha,
                beta_floor,
                min_degree,
                jl_dim,
            )?;
            (m2, g_cfg, d_cfg)
        };

        let inner =
            GuMapSolver::<f32>::from_graph_dog(graph.inner(), max_2hop, &dog_cfg, gumap_cfg)
                .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `GuMapSolver` from precomputed `DogVaradhanLandmarks` and 1+2-hop priority CSR.
    #[wasm_bindgen(js_name = fromLandmarks)]
    pub fn from_landmarks(
        graph: &JsCsrGraph,
        landmarks: &JsDogVaradhanLandmarks,
        max_2hop_or_options: &JsValue,
        options: &JsValue,
    ) -> Result<JsGuMapSolver, JsError> {
        let (max_2hop, gumap_cfg) = if is_plain_js_object(max_2hop_or_options) {
            reject_extra_args_with_options(!options.is_undefined() && !options.is_null())?;
            let m2 = opt_usize_prop(
                max_2hop_or_options,
                "max2hopPerNode",
                DEFAULT_GUMAP_MAX_2HOP,
            )?;
            let g_cfg = parse_gumap_config(
                max_2hop_or_options,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                false,
            )?;
            (m2, g_cfg)
        } else {
            let m2 = if max_2hop_or_options.is_undefined() || max_2hop_or_options.is_null() {
                DEFAULT_GUMAP_MAX_2HOP
            } else {
                let v = max_2hop_or_options.as_f64().ok_or_else(|| {
                    JsError::new("max2hopOrOptions must be a non-negative integer, options object, null, or undefined")
                })?;
                parse_non_negative_usize(v, "max2hopPerNode")?
            };
            if !options.is_undefined() && !options.is_null() && !is_plain_js_object(options) {
                return Err(JsError::new(
                    "GuMapSolver.fromLandmarks: options must be a plain object, null, or undefined",
                ));
            }
            let g_cfg = parse_gumap_config(
                options, None, None, None, None, None, None, None, None, None, false,
            )?;
            (m2, g_cfg)
        };
        let inner = GuMapSolver::<f32>::from_landmarks(
            graph.inner(),
            landmarks.inner(),
            max_2hop,
            gumap_cfg,
        )
        .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one epoch on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` epochs on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "GuMapSolver::run_steps",
        )
    }

    /// Runs GuMap optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "GuMapSolver::run",
        )
    }

    /// Returns the number of directed positive attractive edges stored.
    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Returns the resolved UMAP curve parameter `a`.
    #[wasm_bindgen(getter)]
    pub fn a(&self) -> f64 {
        self.inner.ab().0
    }

    /// Returns the resolved UMAP curve parameter `b`.
    #[wasm_bindgen(getter)]
    pub fn b(&self) -> f64 {
        self.inner.ab().1
    }

    /// Returns the resolved UMAP curve parameters `[a, b]`.
    #[wasm_bindgen(getter)]
    pub fn ab(&self) -> Vec<f64> {
        let (a, b) = self.inner.ab();
        vec![a, b]
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of epochs executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the total scheduled epoch count (`nEpochs`).
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Returns `true` when the solver has converged or completed.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    /// Returns the expected vertex count.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }
}

// ============================================================================
// 6. SpatialIntruderRefiner & SpatialRefinerStats (f32)
// ============================================================================

/// Per-round diagnostics emitted by `SpatialIntruderRefiner`.
#[wasm_bindgen(js_name = SpatialRefinerRoundStats)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsSpatialRefinerRoundStats {
    inner: SpatialRefinerRoundStats,
}

#[wasm_bindgen(js_class = SpatialRefinerRoundStats)]
impl JsSpatialRefinerRoundStats {
    /// Returns the 1-indexed refinement round number.
    #[wasm_bindgen(getter)]
    pub fn round(&self) -> usize {
        self.inner.round
    }

    /// Returns the count of Tier 1 1-hop anchor pairs.
    #[wasm_bindgen(getter, js_name = nAnchors)]
    pub fn n_anchors(&self) -> usize {
        self.inner.n_anchors
    }

    /// Returns the count of active Tier 2 false-negative pairs.
    #[wasm_bindgen(getter, js_name = nFn)]
    pub fn n_fn(&self) -> usize {
        self.inner.n_fn
    }

    /// Returns the count of active Tier 3 spatial false-positive intruder pairs.
    #[wasm_bindgen(getter, js_name = nFp)]
    pub fn n_fp(&self) -> usize {
        self.inner.n_fp
    }

    /// Returns the count of active short-range separation hinge pairs.
    #[wasm_bindgen(getter, js_name = nSep)]
    pub fn n_sep(&self) -> usize {
        self.inner.n_sep
    }

    /// Returns the total active pair count passed to the inner SGD pass.
    #[wasm_bindgen(getter, js_name = nPairsTotal)]
    pub fn n_pairs_total(&self) -> usize {
        self.inner.n_pairs_total
    }

    /// Returns `true` when zero intruders/stragglers were found and SGD was skipped.
    #[wasm_bindgen(getter)]
    pub fn skipped(&self) -> bool {
        self.inner.skipped
    }
}

/// Cumulative diagnostics returned by `SpatialIntruderRefiner.stats()`.
#[wasm_bindgen(js_name = SpatialRefinerStats)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsSpatialRefinerStats {
    inner: SpatialRefinerStats,
}

#[wasm_bindgen(js_class = SpatialRefinerStats)]
impl JsSpatialRefinerStats {
    /// Returns the number of active refinement rounds executed.
    #[wasm_bindgen(getter, js_name = roundsExecuted)]
    pub fn rounds_executed(&self) -> usize {
        self.inner.rounds_executed
    }

    /// Returns cumulative Tier 1 anchor pairs across executed rounds.
    #[wasm_bindgen(getter, js_name = nAnchorsTotal)]
    pub fn n_anchors_total(&self) -> usize {
        self.inner.n_anchors_total
    }

    /// Returns cumulative Tier 2 false-negative pairs across executed rounds.
    #[wasm_bindgen(getter, js_name = nFnTotal)]
    pub fn n_fn_total(&self) -> usize {
        self.inner.n_fn_total
    }

    /// Returns cumulative Tier 3 false-positive intruder pairs across executed rounds.
    #[wasm_bindgen(getter, js_name = nFpTotal)]
    pub fn n_fp_total(&self) -> usize {
        self.inner.n_fp_total
    }

    /// Returns cumulative short-range separation hinge pairs across executed rounds.
    #[wasm_bindgen(getter, js_name = nSepTotal)]
    pub fn n_sep_total(&self) -> usize {
        self.inner.n_sep_total
    }

    /// Returns cumulative total active pairs across executed rounds.
    #[wasm_bindgen(getter, js_name = nPairsTotal)]
    pub fn n_pairs_total(&self) -> usize {
        self.inner.n_pairs_total
    }

    /// Returns the initial median 1-hop edge scale before refinement.
    #[wasm_bindgen(getter, js_name = initialScale)]
    pub fn initial_scale(&self) -> f64 {
        self.inner.initial_scale
    }

    /// Returns the final median 1-hop edge scale after refinement.
    #[wasm_bindgen(getter, js_name = finalScale)]
    pub fn final_scale(&self) -> f64 {
        self.inner.final_scale
    }

    /// Returns the relative distance floor applied during refinement.
    #[wasm_bindgen(getter, js_name = dFloor)]
    pub fn d_floor(&self) -> f64 {
        self.inner.d_floor
    }

    /// Returns the ratio of final layout std to initial layout std.
    #[wasm_bindgen(getter, js_name = scaleRatio)]
    pub fn scale_ratio(&self) -> f64 {
        self.inner.scale_ratio
    }

    /// Returns `true` when the hub-safe 2-hop fallback path was used.
    #[wasm_bindgen(getter, js_name = usedHubFallback)]
    pub fn used_hub_fallback(&self) -> bool {
        self.inner.used_hub_fallback
    }

    /// Returns `true` when Tier 3 FPs used one-sided hinge repulsion.
    #[wasm_bindgen(getter, js_name = hingeFp)]
    pub fn hinge_fp(&self) -> bool {
        self.inner.hinge_fp
    }

    /// Returns per-round diagnostic entries.
    #[wasm_bindgen(js_name = roundStats)]
    pub fn round_stats(&self) -> Vec<JsSpatialRefinerRoundStats> {
        self.inner
            .round_stats
            .iter()
            .copied()
            .map(|inner| JsSpatialRefinerRoundStats { inner })
            .collect()
    }
}

/// Cooperative $O(N \log N)$ `StaticKdTree` active-set spatial intruder refiner (`f32`).
#[wasm_bindgen(js_name = SpatialIntruderRefiner)]
#[derive(Debug, Clone)]
pub struct JsSpatialIntruderRefiner {
    inner: SpatialIntruderRefiner<f32>,
}

#[wasm_bindgen(js_class = SpatialIntruderRefiner)]
impl JsSpatialIntruderRefiner {
    /// Precomputes canonical 1-hop topology and hub-safe 2-hop structures for `graph`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &JsCsrGraph,
        rounds_or_options: &JsValue,
        k_cap: Option<f64>,
        margin_scale: Option<f64>,
        w_anchor: Option<f64>,
        w_fn: Option<f64>,
        w_fp: Option<f64>,
        fn_shrink: Option<f64>,
        round_decay: Option<f64>,
        t_max: Option<f64>,
        eps: Option<f64>,
        hinge_fp: Option<bool>,
        seed: Option<f64>,
        w_anchor2: Option<f64>,
        k_cap_decay: Option<f64>,
        k_cap_min: Option<f64>,
        min_sep_ratio: Option<f64>,
        w_sep: Option<f64>,
    ) -> Result<JsSpatialIntruderRefiner, JsError> {
        let mut cfg = SpatialRefinerConfig::default();
        if is_plain_js_object(rounds_or_options) {
            let has_extra = k_cap.is_some()
                || margin_scale.is_some()
                || w_anchor.is_some()
                || w_fn.is_some()
                || w_fp.is_some()
                || fn_shrink.is_some()
                || round_decay.is_some()
                || t_max.is_some()
                || eps.is_some()
                || hinge_fp.is_some()
                || seed.is_some()
                || w_anchor2.is_some()
                || k_cap_decay.is_some()
                || k_cap_min.is_some()
                || min_sep_ratio.is_some()
                || w_sep.is_some();
            reject_extra_args_with_options(has_extra)?;
            cfg.rounds = opt_usize_prop(rounds_or_options, "rounds", cfg.rounds)?;
            cfg.k_cap = opt_usize_prop(rounds_or_options, "kCap", cfg.k_cap)?;
            cfg.margin_scale = opt_f64_prop(rounds_or_options, "marginScale", cfg.margin_scale)?;
            cfg.w_anchor = opt_f64_prop(rounds_or_options, "wAnchor", cfg.w_anchor)?;
            cfg.w_anchor2 = opt_f64_prop(rounds_or_options, "wAnchor2", cfg.w_anchor2)?;
            cfg.w_fn = opt_f64_prop(rounds_or_options, "wFn", cfg.w_fn)?;
            cfg.w_fp = opt_f64_prop(rounds_or_options, "wFp", cfg.w_fp)?;
            cfg.fn_shrink = opt_f64_prop(rounds_or_options, "fnShrink", cfg.fn_shrink)?;
            cfg.round_decay = opt_f64_prop(rounds_or_options, "roundDecay", cfg.round_decay)?;
            cfg.k_cap_decay = opt_f64_prop(rounds_or_options, "kCapDecay", cfg.k_cap_decay)?;
            cfg.k_cap_min = opt_usize_prop(rounds_or_options, "kCapMin", cfg.k_cap_min)?;
            cfg.t_max = opt_usize_prop(rounds_or_options, "tMax", cfg.t_max)?;
            cfg.eps = opt_f64_prop(rounds_or_options, "eps", cfg.eps)?;
            cfg.hinge_fp = opt_bool_prop(rounds_or_options, "hingeFp", cfg.hinge_fp)?;
            cfg.min_sep_ratio = opt_f64_prop(rounds_or_options, "minSepRatio", cfg.min_sep_ratio)?;
            cfg.w_sep = opt_f64_prop(rounds_or_options, "wSep", cfg.w_sep)?;
            cfg.seed = opt_seed_prop(rounds_or_options)?;
        } else {
            if !rounds_or_options.is_undefined() && !rounds_or_options.is_null() {
                let v = rounds_or_options.as_f64().ok_or_else(|| {
                    JsError::new("roundsOrOptions must be a non-negative integer, options object, null, or undefined")
                })?;
                cfg.rounds = parse_non_negative_usize(v, "rounds")?;
            }
            cfg.k_cap = parse_optional_usize(k_cap, cfg.k_cap, "kCap")?;
            cfg.margin_scale = margin_scale.unwrap_or(cfg.margin_scale);
            cfg.w_anchor = w_anchor.unwrap_or(cfg.w_anchor);
            cfg.w_anchor2 = w_anchor2.unwrap_or(cfg.w_anchor2);
            cfg.w_fn = w_fn.unwrap_or(cfg.w_fn);
            cfg.w_fp = w_fp.unwrap_or(cfg.w_fp);
            cfg.fn_shrink = fn_shrink.unwrap_or(cfg.fn_shrink);
            cfg.round_decay = round_decay.unwrap_or(cfg.round_decay);
            cfg.k_cap_decay = k_cap_decay.unwrap_or(cfg.k_cap_decay);
            cfg.k_cap_min = parse_optional_usize(k_cap_min, cfg.k_cap_min, "kCapMin")?;
            cfg.t_max = parse_optional_usize(t_max, cfg.t_max, "tMax")?;
            cfg.eps = eps.unwrap_or(cfg.eps);
            cfg.hinge_fp = hinge_fp.unwrap_or(cfg.hinge_fp);
            cfg.min_sep_ratio = min_sep_ratio.unwrap_or(cfg.min_sep_ratio);
            cfg.w_sep = w_sep.unwrap_or(cfg.w_sep);
            cfg.seed = parse_seed_u64(seed)?;
        }

        let inner =
            SpatialIntruderRefiner::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one active-set refinement round on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` refinement rounds on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "SpatialIntruderRefiner::run_steps",
        )
    }

    /// Runs refinement rounds to completion or early convergence on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "SpatialIntruderRefiner::run",
        )
    }

    /// Returns cumulative and per-round diagnostic statistics.
    pub fn stats(&self) -> JsSpatialRefinerStats {
        JsSpatialRefinerStats {
            inner: self.inner.stats().clone(),
        }
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of refinement rounds completed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the maximum scheduled refinement rounds.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Returns `true` when refinement has converged or completed.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    /// Returns the expected vertex count.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the current dynamic active-set neighborhood cap `kCap`.
    #[wasm_bindgen(getter, js_name = currentKCap)]
    pub fn current_k_cap(&self) -> usize {
        self.inner.current_k_cap()
    }

    /// Updates the active-set neighborhood cap `kCap` for subsequent refinement rounds.
    #[wasm_bindgen(js_name = setKCap)]
    pub fn set_k_cap(&mut self, k_cap: f64) -> Result<(), JsError> {
        let cap = parse_non_negative_usize(k_cap, "kCap")?;
        self.inner.set_k_cap(cap).map_err(core_err_to_js)
    }
}

// ============================================================================
// 7. StressMajorizationSolver (f32)
// ============================================================================

fn parse_stress_majorization_config(
    epsilon_or_options: &JsValue,
    max_iterations: Option<f64>,
    pcg_tolerance: Option<f64>,
    pcg_max_iterations: Option<f64>,
    min_norm: Option<f64>,
) -> Result<StressMajorizationConfig, JsError> {
    let mut cfg = StressMajorizationConfig::default();
    if is_plain_js_object(epsilon_or_options) {
        let has_extra = max_iterations.is_some()
            || pcg_tolerance.is_some()
            || pcg_max_iterations.is_some()
            || min_norm.is_some();
        reject_extra_args_with_options(has_extra)?;
        cfg.epsilon = opt_f64_prop(epsilon_or_options, "epsilon", cfg.epsilon)?;
        cfg.max_iterations =
            opt_usize_prop(epsilon_or_options, "maxIterations", cfg.max_iterations)?;
        cfg.pcg_tolerance = opt_f64_prop(epsilon_or_options, "pcgTolerance", cfg.pcg_tolerance)?;
        cfg.pcg_max_iterations = opt_maybe_usize_prop(epsilon_or_options, "pcgMaxIterations")?;
        cfg.min_norm = opt_f64_prop(epsilon_or_options, "minNorm", cfg.min_norm)?;
        return Ok(cfg);
    }

    if !epsilon_or_options.is_undefined() && !epsilon_or_options.is_null() {
        cfg.epsilon = epsilon_or_options.as_f64().ok_or_else(|| {
            JsError::new("epsilonOrOptions must be a number, options object, null, or undefined")
        })?;
    }
    cfg.max_iterations = parse_optional_usize(max_iterations, cfg.max_iterations, "maxIterations")?;
    cfg.pcg_tolerance = pcg_tolerance.unwrap_or(cfg.pcg_tolerance);
    cfg.pcg_max_iterations = parse_opt_usize_param(pcg_max_iterations, "pcgMaxIterations")?;
    cfg.min_norm = min_norm.unwrap_or(cfg.min_norm);
    Ok(cfg)
}

/// Cooperative Gansner-Koren-North (2004) Stress Majorization layout solver (`f32`).
#[wasm_bindgen(js_name = StressMajorizationSolver)]
#[derive(Debug, Clone)]
pub struct JsStressMajorizationSolver {
    inner: StressMajorizationSolver<f32>,
}

#[wasm_bindgen(js_class = StressMajorizationSolver)]
impl JsStressMajorizationSolver {
    /// Constructs a `StressMajorizationSolver` by computing APSP distances from `graph`.
    #[wasm_bindgen(js_name = fromGraph)]
    pub fn from_graph(
        graph: &JsCsrGraph,
        epsilon_or_options: &JsValue,
        max_iterations: Option<f64>,
        pcg_tolerance: Option<f64>,
        pcg_max_iterations: Option<f64>,
        min_norm: Option<f64>,
    ) -> Result<JsStressMajorizationSolver, JsError> {
        let cfg = parse_stress_majorization_config(
            epsilon_or_options,
            max_iterations,
            pcg_tolerance,
            pcg_max_iterations,
            min_norm,
        )?;
        let inner = StressMajorizationSolver::<f32>::from_graph(graph.inner(), cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `StressMajorizationSolver` from a flat row-major `(numNodes, numNodes)` distance matrix.
    #[wasm_bindgen(js_name = fromDistanceMatrix)]
    pub fn from_distance_matrix(
        num_nodes: f64,
        dists_flat: &[f64],
        epsilon_or_options: &JsValue,
        max_iterations: Option<f64>,
        pcg_tolerance: Option<f64>,
        pcg_max_iterations: Option<f64>,
        min_norm: Option<f64>,
    ) -> Result<JsStressMajorizationSolver, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let cfg = parse_stress_majorization_config(
            epsilon_or_options,
            max_iterations,
            pcg_tolerance,
            pcg_max_iterations,
            min_norm,
        )?;
        let inner = StressMajorizationSolver::<f32>::from_distance_matrix(n, dists_flat, cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one majorization step on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` majorization steps on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "StressMajorizationSolver::run_steps",
        )
    }

    /// Runs majorization to completion or relative stress convergence on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "StressMajorizationSolver::run",
        )
    }

    /// Evaluates weighted stress $\sum_{i < j} w_{ij} (\|x_i - x_j\| - d_{ij})^2$ for `layout`.
    #[wasm_bindgen(js_name = computeStress)]
    pub fn compute_stress(&self, layout: &JsLayout2d) -> Result<f64, JsError> {
        let soa = layout.inner().to_layout2d();
        self.inner.compute_stress(&soa).map_err(core_err_to_js)
    }

    /// Returns the stress recorded after the most recent step, or `undefined`.
    #[wasm_bindgen(getter, js_name = lastStress)]
    pub fn last_stress(&self) -> Option<f64> {
        self.inner.last_stress()
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of iterations executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the scheduled maximum iteration count.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Returns `true` when the solver has converged or completed.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    /// Returns the expected vertex count.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }
}

// ============================================================================
// 8. KamadaKawaiSolver (f32)
// ============================================================================

fn parse_kamada_kawai_config(
    eps_or_options: &JsValue,
    max_inner_iters: Option<f64>,
    max_outer_iters: Option<f64>,
) -> Result<KamadaKawaiConfig, JsError> {
    let mut cfg = KamadaKawaiConfig::default();
    if is_plain_js_object(eps_or_options) {
        let has_extra = max_inner_iters.is_some() || max_outer_iters.is_some();
        reject_extra_args_with_options(has_extra)?;
        cfg.eps = opt_f64_prop(eps_or_options, "eps", cfg.eps)?;
        cfg.max_inner_iters = opt_usize_prop(eps_or_options, "maxInnerIters", cfg.max_inner_iters)?;
        cfg.max_outer_iters = opt_maybe_usize_prop(eps_or_options, "maxOuterIters")?;
        return Ok(cfg);
    }

    if !eps_or_options.is_undefined() && !eps_or_options.is_null() {
        cfg.eps = eps_or_options.as_f64().ok_or_else(|| {
            JsError::new("epsOrOptions must be a number, options object, null, or undefined")
        })?;
    }
    cfg.max_inner_iters =
        parse_optional_usize(max_inner_iters, cfg.max_inner_iters, "maxInnerIters")?;
    cfg.max_outer_iters = parse_opt_usize_param(max_outer_iters, "maxOuterIters")?;
    Ok(cfg)
}

/// Cooperative Kamada-Kawai (1989) spring layout solver (`f32`).
#[wasm_bindgen(js_name = KamadaKawaiSolver)]
#[derive(Debug, Clone)]
pub struct JsKamadaKawaiSolver {
    inner: KamadaKawaiSolver<f32>,
}

#[wasm_bindgen(js_class = KamadaKawaiSolver)]
impl JsKamadaKawaiSolver {
    /// Constructs a `KamadaKawaiSolver` by computing APSP distances from `graph`.
    #[wasm_bindgen(js_name = fromGraph)]
    pub fn from_graph(
        graph: &JsCsrGraph,
        eps_or_options: &JsValue,
        max_inner_iters: Option<f64>,
        max_outer_iters: Option<f64>,
    ) -> Result<JsKamadaKawaiSolver, JsError> {
        let cfg = parse_kamada_kawai_config(eps_or_options, max_inner_iters, max_outer_iters)?;
        let inner =
            KamadaKawaiSolver::<f32>::from_graph(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `KamadaKawaiSolver` from a flat row-major `(numNodes, numNodes)` distance matrix.
    #[wasm_bindgen(js_name = fromDistanceMatrix)]
    pub fn from_distance_matrix(
        num_nodes: f64,
        dists_flat: &[f64],
        eps_or_options: &JsValue,
        max_inner_iters: Option<f64>,
        max_outer_iters: Option<f64>,
    ) -> Result<JsKamadaKawaiSolver, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let cfg = parse_kamada_kawai_config(eps_or_options, max_inner_iters, max_outer_iters)?;
        let inner = KamadaKawaiSolver::<f32>::from_distance_matrix(n, dists_flat, cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one outer node-selection Newton-Raphson step on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` outer steps on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "KamadaKawaiSolver::run_steps",
        )
    }

    /// Runs Kamada-Kawai optimization to convergence or `maxOuterIters` on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "KamadaKawaiSolver::run",
        )
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of outer steps executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the outer step cap.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Returns `true` when the solver has converged or completed.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    /// Returns the expected vertex count.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }
}
