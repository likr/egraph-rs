//! WebAssembly bindings for `egraph-clustering` community detection, coarsening, and layout projection.

use egraph::clustering::{
    KMeansConfig, KMeansResult, LouvainConfig, LouvainResult, coarsen_graph, kmeans_plus_plus,
    louvain, project_clustered_layout,
};
use egraph::core::{CoreError, CsrGraph};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    DEFAULT_SEED, core_err_to_js, csr_f64_to_f32, invalid_input_err, parse_non_negative_usize,
    set_f64_prop, set_js_prop,
};
use crate::js_options::{check_options_arg, opt_f64_prop, opt_seed_prop, opt_usize_prop};
use crate::layout_buffers::JsLayout2d;

const DEFAULT_KMEANS_K: usize = 2;
const DEFAULT_KMEANS_MAX_ITERS: usize = 100;
const DEFAULT_KMEANS_TOL: f64 = 1e-4;
const DEFAULT_KMEANS_N_INIT: usize = 1;

const DEFAULT_LOUVAIN_MAX_PASSES: usize = 100;
const DEFAULT_LOUVAIN_RESOLUTION: f64 = 1.0;
const DEFAULT_LOUVAIN_MIN_GAIN: f64 = 1e-7;

const DEFAULT_SPREAD_RADIUS: f64 = 0.1;

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

    /// Returns the linear-memory byte offset of `labels` for zero-copy `labelsView()`.
    #[wasm_bindgen(js_name = labelsPtr)]
    pub fn labels_ptr(&self) -> usize {
        self.inner.labels.as_ptr() as usize
    }

    /// Returns the element count of `labels`.
    #[wasm_bindgen(js_name = labelsLen)]
    pub fn labels_len(&self) -> usize {
        self.inner.labels.len()
    }

    /// Returns flat row-major `(k, dims)` cluster centroids (`Float64Array`).
    pub fn centroids(&self) -> Vec<f64> {
        self.inner.centroids.clone()
    }

    /// Returns the linear-memory byte offset of `centroids` for zero-copy `centroidsView()`.
    #[wasm_bindgen(js_name = centroidsPtr)]
    pub fn centroids_ptr(&self) -> usize {
        self.inner.centroids.as_ptr() as usize
    }

    /// Returns the scalar element count of `centroids`.
    #[wasm_bindgen(js_name = centroidsLen)]
    pub fn centroids_len(&self) -> usize {
        self.inner.centroids.len()
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

    /// Converts this result into a plain JavaScript object without requiring `serde`.
    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsValue> {
        let obj = js_sys::Object::new();
        set_js_prop(
            &obj,
            "labels",
            &js_sys::Uint32Array::from(self.inner.labels.as_slice()).into(),
        )?;
        set_js_prop(
            &obj,
            "centroids",
            &js_sys::Float64Array::from(self.inner.centroids.as_slice()).into(),
        )?;
        set_f64_prop(&obj, "inertia", self.inner.inertia)?;
        set_f64_prop(&obj, "iterations", self.inner.iterations as f64)?;
        set_js_prop(&obj, "converged", &JsValue::from_bool(self.inner.converged))?;
        set_f64_prop(&obj, "k", self.k as f64)?;
        set_f64_prop(&obj, "dims", self.dims as f64)?;
        Ok(obj)
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

    /// Returns the linear-memory byte offset of `labels` for zero-copy `labelsView()`.
    #[wasm_bindgen(js_name = labelsPtr)]
    pub fn labels_ptr(&self) -> usize {
        self.inner.labels.as_ptr() as usize
    }

    /// Returns the element count of `labels`.
    #[wasm_bindgen(js_name = labelsLen)]
    pub fn labels_len(&self) -> usize {
        self.inner.labels.len()
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

    /// Converts this result into a plain JavaScript object without requiring `serde`.
    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsValue> {
        let obj = js_sys::Object::new();
        set_js_prop(
            &obj,
            "labels",
            &js_sys::Uint32Array::from(self.inner.labels.as_slice()).into(),
        )?;
        set_f64_prop(&obj, "numCommunities", self.inner.num_communities as f64)?;
        set_f64_prop(&obj, "modularity", self.inner.modularity)?;
        set_f64_prop(&obj, "passes", self.inner.passes as f64)?;
        Ok(obj)
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
#[wasm_bindgen(js_name = kmeansPlusPlus)]
pub fn js_kmeans_plus_plus(
    points: &[f64],
    num_nodes: f64,
    dims: f64,
    options: &JsValue,
) -> Result<JsKMeansResult, JsValue> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let d = parse_non_negative_usize(dims, "dims")?;
    let mut cfg = KMeansConfig::default();
    cfg.k = DEFAULT_KMEANS_K;
    cfg.max_iterations = DEFAULT_KMEANS_MAX_ITERS;
    cfg.tol = DEFAULT_KMEANS_TOL;
    cfg.n_init = DEFAULT_KMEANS_N_INIT;
    cfg.seed = DEFAULT_SEED;
    if check_options_arg(options, "kmeansPlusPlus")? {
        cfg.k = opt_usize_prop(options, "k", cfg.k)?;
        cfg.max_iterations = opt_usize_prop(options, "maxIterations", cfg.max_iterations)?;
        cfg.tol = opt_f64_prop(options, "tol", cfg.tol)?;
        cfg.n_init = opt_usize_prop(options, "nInit", cfg.n_init)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    let k_val = cfg.k;
    let inner = kmeans_plus_plus(points, n, d, &cfg).map_err(core_err_to_js)?;
    Ok(JsKMeansResult {
        inner,
        k: k_val,
        dims: d,
    })
}

/// Detects communities on an undirected `CsrGraph` via multi-level Louvain modularity optimization.
#[wasm_bindgen(js_name = louvainCommunities)]
pub fn js_louvain_communities(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsLouvainResult, JsValue> {
    let mut cfg = LouvainConfig::default();
    cfg.max_passes = DEFAULT_LOUVAIN_MAX_PASSES;
    cfg.resolution = DEFAULT_LOUVAIN_RESOLUTION;
    cfg.min_modularity_gain = DEFAULT_LOUVAIN_MIN_GAIN;
    cfg.seed = DEFAULT_SEED;
    if check_options_arg(options, "louvainCommunities")? {
        cfg.max_passes = opt_usize_prop(options, "maxPasses", cfg.max_passes)?;
        cfg.resolution = opt_f64_prop(options, "resolution", cfg.resolution)?;
        cfg.min_modularity_gain =
            opt_f64_prop(options, "minModularityGain", cfg.min_modularity_gain)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    let inner = louvain(graph.inner(), &cfg).map_err(core_err_to_js)?;
    Ok(JsLouvainResult { inner })
}

/// Coarsens an undirected `CsrGraph` by contracting vertices that share a cluster label.
#[wasm_bindgen(js_name = coarsenGraph)]
pub fn js_coarsen_graph(
    graph: &JsCsrGraph,
    labels: &[u32],
) -> Result<JsCoarsenedGraphResult, JsValue> {
    let coarse = coarsen_graph(graph.inner(), labels).map_err(core_err_to_js)?;
    let graph_f32 = csr_f64_to_f32(&coarse.graph)?;
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
    options: &JsValue,
) -> Result<JsLayout2d, JsValue> {
    let mut radius = DEFAULT_SPREAD_RADIUS;
    let mut seed_u64 = DEFAULT_SEED;
    if check_options_arg(options, "projectClusteredLayout")? {
        radius = opt_f64_prop(options, "spreadRadius", radius)?;
        seed_u64 = opt_seed_prop(options)?;
    }
    if !radius.is_finite() {
        return Err(core_err_to_js(CoreError::non_finite(
            "projectClusteredLayout::spreadRadius",
        )));
    }
    if radius < 0.0 {
        return Err(invalid_input_err("spreadRadius must be non-negative"));
    }
    let radius_f32 = radius as f32;
    if !radius_f32.is_finite() {
        return Err(core_err_to_js(CoreError::non_finite(
            "projectClusteredLayout::spreadRadius",
        )));
    }
    let coarse_soa = coarse_layout.inner().to_layout2d();
    let fine = project_clustered_layout(&coarse_soa, node_to_coarse, radius_f32, seed_u64)
        .map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&fine))
}
