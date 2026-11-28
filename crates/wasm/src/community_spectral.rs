//! WebAssembly bindings for Label Propagation and Ng-Jordan-Weiss Spectral Clustering.

use egraph::clustering::{
    LabelPropagationConfig, LabelPropagationResult, SpectralClusteringConfig,
    SpectralClusteringResult, label_propagation, spectral_clustering,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    DEFAULT_SEED, core_err_to_js, parse_optional_usize, parse_seed_u64, set_f64_prop, set_js_prop,
};
use crate::js_options::{check_options_arg, opt_f64_prop, opt_seed_prop, opt_usize_prop};

const DEFAULT_KMEANS_MAX_ITERS: usize = 100;
const DEFAULT_KMEANS_TOL: f64 = 1e-4;
const DEFAULT_SPECTRAL_K: usize = 2;
const DEFAULT_SPECTRAL_MAX_EIGEN_ITERS: usize = 300;
const DEFAULT_SPECTRAL_EIGEN_TOL: f64 = 1e-4;
const DEFAULT_LPA_MAX_ITERS: usize = 100;

/// Output of `labelPropagation`.
#[wasm_bindgen(js_name = LabelPropagationResult)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsLabelPropagationResult {
    inner: LabelPropagationResult,
}

#[wasm_bindgen(js_class = LabelPropagationResult)]
impl JsLabelPropagationResult {
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

    /// Returns the number of asynchronous propagation sweeps executed.
    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> usize {
        self.inner.iterations
    }

    /// Returns `true` when propagation reached a fixed point.
    #[wasm_bindgen(getter)]
    pub fn converged(&self) -> bool {
        self.inner.converged
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
        set_f64_prop(&obj, "iterations", self.inner.iterations as f64)?;
        set_js_prop(&obj, "converged", &JsValue::from_bool(self.inner.converged))?;
        Ok(obj)
    }
}

/// Output of `spectralClustering`.
#[wasm_bindgen(js_name = SpectralClusteringResult)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsSpectralClusteringResult {
    inner: SpectralClusteringResult,
}

#[wasm_bindgen(js_class = SpectralClusteringResult)]
impl JsSpectralClusteringResult {
    /// Returns canonical cluster labels in `0..numClusters` (`Uint32Array`).
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

    /// Returns the number of distinct non-empty clusters detected.
    #[wasm_bindgen(getter, js_name = numClusters)]
    pub fn num_clusters(&self) -> usize {
        self.inner.num_clusters
    }

    /// Returns the number of Lanczos thick-restart iterations executed.
    #[wasm_bindgen(getter, js_name = eigenIterations)]
    pub fn eigen_iterations(&self) -> usize {
        self.inner.eigen_iterations
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
        set_f64_prop(&obj, "numClusters", self.inner.num_clusters as f64)?;
        set_f64_prop(&obj, "eigenIterations", self.inner.eigen_iterations as f64)?;
        Ok(obj)
    }
}

/// Performs Ng-Jordan-Weiss spectral clustering on an undirected `CsrGraph`.
#[wasm_bindgen(js_name = spectralClustering)]
pub fn js_spectral_clustering(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsSpectralClusteringResult, JsValue> {
    let mut cfg = SpectralClusteringConfig::default();
    cfg.k = DEFAULT_SPECTRAL_K;
    cfg.max_eigen_iterations = DEFAULT_SPECTRAL_MAX_EIGEN_ITERS;
    cfg.eigen_tol = DEFAULT_SPECTRAL_EIGEN_TOL;
    cfg.kmeans_max_iterations = DEFAULT_KMEANS_MAX_ITERS;
    cfg.kmeans_tol = DEFAULT_KMEANS_TOL;
    cfg.seed = DEFAULT_SEED;
    if check_options_arg(options, "spectralClustering")? {
        cfg.k = opt_usize_prop(options, "k", cfg.k)?;
        cfg.max_eigen_iterations =
            opt_usize_prop(options, "maxEigenIterations", cfg.max_eigen_iterations)?;
        cfg.eigen_tol = opt_f64_prop(options, "eigenTol", cfg.eigen_tol)?;
        cfg.kmeans_max_iterations =
            opt_usize_prop(options, "kmeansMaxIterations", cfg.kmeans_max_iterations)?;
        cfg.kmeans_tol = opt_f64_prop(options, "kmeansTol", cfg.kmeans_tol)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    let inner = spectral_clustering(graph.inner(), &cfg).map_err(core_err_to_js)?;
    Ok(JsSpectralClusteringResult { inner })
}

/// Detects communities on an undirected `CsrGraph` via asynchronous Label Propagation.
#[wasm_bindgen(js_name = labelPropagation)]
pub fn js_label_propagation(
    graph: &JsCsrGraph,
    max_iterations: Option<f64>,
    seed: &JsValue,
) -> Result<JsLabelPropagationResult, JsValue> {
    let mut cfg = LabelPropagationConfig::default();
    cfg.max_iterations =
        parse_optional_usize(max_iterations, DEFAULT_LPA_MAX_ITERS, "maxIterations")?;
    cfg.seed = parse_seed_u64(seed)?;
    let inner = label_propagation(graph.inner(), &cfg).map_err(core_err_to_js)?;
    Ok(JsLabelPropagationResult { inner })
}
