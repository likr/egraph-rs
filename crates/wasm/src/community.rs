//! WebAssembly bindings for `egraph-clustering` community detection, coarsening, and layout projection.

use egraph::clustering::{
    KMeansConfig, KMeansResult, LabelPropagationConfig, LouvainConfig, LouvainResult,
    SpectralClusteringConfig, coarsen_graph, kmeans_plus_plus, label_propagation, louvain,
    project_clustered_layout, spectral_clustering,
};
use egraph::core::{CoreError, CsrGraph};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, csr_f64_to_f32, parse_non_negative_usize, parse_optional_usize, parse_seed_u64,
};
use crate::layout_buffers::JsLayout2d;

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
        max_iterations: max_iters_val,
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
        max_eigen_iterations: parse_optional_usize(
            max_eigen_iters,
            DEFAULT_SPECTRAL_MAX_EIGEN_ITERS,
            "maxEigenIters",
        )?,
        eigen_tol: eigen_tol.unwrap_or(DEFAULT_SPECTRAL_EIGEN_TOL),
        kmeans_max_iterations: parse_optional_usize(
            kmeans_max_iters,
            DEFAULT_KMEANS_MAX_ITERS,
            "kmeansMaxIters",
        )?,
        kmeans_tol: kmeans_tol.unwrap_or(DEFAULT_KMEANS_TOL),
        seed: parse_seed_u64(seed)?,
    };
    spectral_clustering(graph.inner(), &cfg)
        .map(|res| res.labels)
        .map_err(core_err_to_js)
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
        max_iterations: parse_optional_usize(max_iters, DEFAULT_LPA_MAX_ITERS, "maxIters")?,
        seed: parse_seed_u64(seed)?,
    };
    label_propagation(graph.inner(), &cfg)
        .map(|res| res.labels)
        .map_err(core_err_to_js)
}

/// Coarsens an undirected `CsrGraph` by contracting vertices that share a cluster label.
#[wasm_bindgen(js_name = coarsenGraph)]
pub fn js_coarsen_graph(
    graph: &JsCsrGraph,
    labels: &[u32],
) -> Result<JsCoarsenedGraphResult, JsError> {
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
