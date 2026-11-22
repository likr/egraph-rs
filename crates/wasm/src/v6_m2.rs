//! M2 v6 WebAssembly bindings for `egraph-linalg` and `egraph-neighborhood`.
//!
//! Exposes pure-Rust spectral initialization, eigensolvers, Chebyshev heat/DoG diffusion,
//! 1+2-hop and hybrid adaptive $k$-NN graphs, perplexity / fuzzy simplicial probabilities,
//! and regularized DoG Varadhan landmark tables without `serde` or threading dependencies.

use egraph::core::{CsrGraph, Layout2d, LayoutNd};
use egraph::linalg::{
    ChebyshevWorkspace, EigenResult, PivotMdsConfig, SpectralMomentStats,
    adaptive_chebyshev_degree, build_dog_terms, classical_mds_2d,
    compute_chebyshev_bandpass_coefficients, koren_spectral_init, normalize_layout_2d, pivot_mds,
    project_psd_spectral_filter, rd_mds, representative_mds, select_maxmin_sp_pivots,
    smallest_sym_laplacian_eigenpairs, spectral_moment_stats, symmetric_normalized_laplacian,
};
use egraph::neighborhood::{
    DEFAULT_MAX_PERPLEXITY, DEFAULT_MEDIAN_MIN_PERPLEXITY, DEFAULT_MEDIAN_PERPLEXITY_C,
    DEFAULT_NODEWISE_MIN_PERPLEXITY, DEFAULT_NODEWISE_PERPLEXITY_C, DogVaradhanConfig,
    DogVaradhanLandmarks, DogVaradhanStats, HybridKnnConfig, KnnGraph, KnnStats, PerplexityConfig,
    SparseProbabilities, SymmetrizationMode, apply_relative_distance_floor,
    build_gumap_12hop_priority_csr, build_hybrid_adaptive_knn, build_pure_2hop_knn,
    build_rerank_knn, compute_2hop_neighborhood_sizes, compute_diffusion_embedding,
    compute_fuzzy_simplicial_set, compute_horvitz_thompson_gamma, compute_knn_probabilities,
    compute_median_2hop_perplexity, compute_nodewise_2hop_perplexities,
    compute_regularized_dog_varadhan_landmarks, compute_resistance_embedding,
    extract_1hop_2hop_csr, find_ab_params, graph_row_normalized_probabilities, partial_bfs_knn,
    symmetrize_row_normalized,
};
use wasm_bindgen::prelude::*;

use crate::v6::{
    JsCsrGraph, JsLayout2d, JsLayoutNd, core_err_to_js, ffi_err_to_js, parse_non_negative_usize,
    parse_seed_u64,
};

const DEFAULT_NUM_PIVOTS: usize = 50;
const DEFAULT_MIN_DEGREE: usize = 2;
const DEFAULT_TOL: f64 = 1e-4;
const DEFAULT_MAX_ITERS: usize = 300;
const DEFAULT_N_MODES: usize = 16;
const DEFAULT_LAMBDA_MAX: f64 = 2.0;
const DEFAULT_JL_DIM: usize = 64;
const DEFAULT_DOG_DEGREE: usize = 25;
const DEFAULT_EPS_PSD: f64 = 1e-4;
const DEFAULT_K_FACTOR: f64 = 3.0;
const DEFAULT_MARGIN_GAP: f64 = 0.1;
const DEFAULT_COARSE_DIM: usize = 10;
const DEFAULT_RERANK_FACTOR: usize = 16;
const DEFAULT_PARTIAL_BFS_K: usize = 30;
const DEFAULT_POWER: f64 = 2.0;
const DEFAULT_PERPLEXITY_SIGMA_ITERS: usize = 50;
const DEFAULT_FUZZY_SIGMA_ITERS: usize = 64;
const DEFAULT_SIGMA_TOLERANCE: f64 = 1e-5;
const DEFAULT_MIN_DIST: f64 = 0.1;
const DEFAULT_SPREAD: f64 = 1.0;
const DEFAULT_VARADHAN_LANDMARKS: usize = 64;
const DEFAULT_P_EXP: f64 = 0.5;
const DEFAULT_T_COARSE: f64 = 48.0;

pub(crate) fn parse_optional_usize(
    value: Option<f64>,
    default: usize,
    name: &str,
) -> Result<usize, JsError> {
    match value {
        None => Ok(default),
        Some(v) => parse_non_negative_usize(v, name),
    }
}

pub(crate) fn parse_opt_usize_param(
    value: Option<f64>,
    name: &str,
) -> Result<Option<usize>, JsError> {
    value.map(|v| parse_non_negative_usize(v, name)).transpose()
}

fn parse_optional_target_std(value: &JsValue) -> Result<Option<f64>, JsError> {
    if value.is_undefined() {
        return Ok(Some(1.0));
    }
    if value.is_null() {
        return Ok(None);
    }
    if let Some(num) = value.as_f64() {
        return Ok(Some(num));
    }
    Err(JsError::new(
        "targetStd must be a positive finite number, null, or undefined",
    ))
}

fn indptr_usize_to_u32(indptr: &[usize], context: &str) -> Result<Vec<u32>, JsError> {
    let mut out = Vec::with_capacity(indptr.len());
    for &offset in indptr {
        let converted = u32::try_from(offset).map_err(|_| {
            JsError::new(&format!(
                "{context}: indptr offset {offset} exceeds u32::MAX"
            ))
        })?;
        out.push(converted);
    }
    Ok(out)
}

fn unweighted_f64_csr_to_f32(graph: &CsrGraph<f64>) -> Result<CsrGraph<f32>, JsError> {
    CsrGraph::<f32>::from_csr(
        graph.indptr(),
        graph.indices(),
        None,
        graph.is_directed(),
        false,
    )
    .map_err(core_err_to_js)
}

/// Variable-length Compressed Sparse Row (CSR) $k$-nearest-neighbor graph in `f64`.
#[wasm_bindgen(js_name = KnnGraph)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsKnnGraph {
    inner: KnnGraph,
}

impl JsKnnGraph {
    /// Returns a shared reference to the underlying `egraph_neighborhood::KnnGraph`.
    pub fn inner(&self) -> &KnnGraph {
        &self.inner
    }
}

#[wasm_bindgen(js_class = KnnGraph)]
impl JsKnnGraph {
    /// Constructs a validated `KnnGraph` from CSR `numNodes`, `indptr`, `indices`, and `distances`.
    #[wasm_bindgen(js_name = fromCsr)]
    pub fn from_csr(
        num_nodes: f64,
        indptr: &[u32],
        indices: &[u32],
        distances: &[f64],
    ) -> Result<JsKnnGraph, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let indptr_usize: Vec<usize> = indptr.iter().map(|&v| v as usize).collect();
        let inner = KnnGraph::new_validated(n, indptr_usize, indices.to_vec(), distances.to_vec())
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns the number of vertices in the $k$-NN graph.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the total number of stored neighbor entries.
    #[wasm_bindgen(getter)]
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Returns a copy of the CSR `indptr` row-offset array (`length === numNodes + 1`).
    pub fn indptr(&self) -> Result<Vec<u32>, JsError> {
        indptr_usize_to_u32(self.inner.indptr(), "KnnGraph.indptr")
    }

    /// Returns a copy of the flat `Uint32Array` neighbor indices (`length === nnz`).
    pub fn indices(&self) -> Vec<u32> {
        self.inner.indices().to_vec()
    }

    /// Returns a copy of the flat `Float64Array` distances (`length === nnz`).
    pub fn distances(&self) -> Vec<f64> {
        self.inner.distances().to_vec()
    }
}

/// Canonical sorted symmetric CSR sparse joint probability matrix $P$.
#[wasm_bindgen(js_name = SparseProbabilities)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsSparseProbabilities {
    inner: SparseProbabilities,
}

impl JsSparseProbabilities {
    /// Returns a shared reference to the underlying `egraph_neighborhood::SparseProbabilities`.
    pub fn inner(&self) -> &SparseProbabilities {
        &self.inner
    }
}

#[wasm_bindgen(js_class = SparseProbabilities)]
impl JsSparseProbabilities {
    /// Returns the number of vertices in the probability matrix.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the total number of stored probability entries.
    #[wasm_bindgen(getter)]
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Returns a copy of the CSR `indptr` row-offset array (`length === numNodes + 1`).
    pub fn indptr(&self) -> Result<Vec<u32>, JsError> {
        indptr_usize_to_u32(&self.inner.indptr, "SparseProbabilities.indptr")
    }

    /// Returns a copy of the flat `Uint32Array` neighbor column indices (`length === nnz`).
    pub fn indices(&self) -> Vec<u32> {
        self.inner.indices.clone()
    }

    /// Returns a copy of the flat `Float64Array` symmetric probability values (`length === nnz`).
    pub fn values(&self) -> Vec<f64> {
        self.inner.values.clone()
    }
}

/// Precomputed regularized DoG Varadhan landmark tables and R1/R2 regularization diagnostics.
#[wasm_bindgen(js_name = DogVaradhanLandmarks)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsDogVaradhanLandmarks {
    inner: DogVaradhanLandmarks,
    stats: DogVaradhanStats,
}

impl JsDogVaradhanLandmarks {
    /// Returns a shared reference to the underlying `egraph_neighborhood::DogVaradhanLandmarks`.
    pub fn inner(&self) -> &DogVaradhanLandmarks {
        &self.inner
    }
}

#[wasm_bindgen(js_class = DogVaradhanLandmarks)]
impl JsDogVaradhanLandmarks {
    /// Returns the number of vertices in the graph.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the number of landmark columns.
    #[wasm_bindgen(getter, js_name = numLandmarks)]
    pub fn num_landmarks(&self) -> usize {
        self.inner.num_landmarks()
    }

    /// Returns a copy of the selected landmark vertex IDs (`Uint32Array`).
    pub fn landmarks(&self) -> Vec<u32> {
        self.inner.landmarks().to_vec()
    }

    /// Returns a flat row-major `(numNodes, numLandmarks)` `Float64Array` Varadhan distance table.
    pub fn table(&self) -> Vec<f64> {
        self.inner.d_dog().to_vec()
    }

    /// Returns a flat row-major `(numNodes, numLandmarks)` coarse scaffold table, or `undefined`.
    #[wasm_bindgen(js_name = coarseTable)]
    pub fn coarse_table(&self) -> Option<Vec<f64>> {
        self.inner.d_coarse().map(|slice| slice.to_vec())
    }

    /// Returns the effective short-range diffusion time $t_1$.
    #[wasm_bindgen(getter)]
    pub fn t1(&self) -> f64 {
        self.stats.t1
    }

    /// Returns the effective long-range diffusion time $t_2$.
    #[wasm_bindgen(getter)]
    pub fn t2(&self) -> f64 {
        self.stats.t2
    }

    /// Returns the effective DoG attenuation $\alpha$.
    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f64 {
        self.stats.alpha
    }

    /// Returns the R2 PSD projection shift $\delta_{\mathrm{psd}}$.
    #[wasm_bindgen(getter, js_name = deltaPsd)]
    pub fn delta_psd(&self) -> f64 {
        self.stats.delta_psd
    }

    /// Returns the effective rank-1 positivity floor `betaEff`.
    #[wasm_bindgen(getter, js_name = betaEff)]
    pub fn beta_eff(&self) -> f64 {
        self.stats.beta_eff
    }

    /// Returns the parameter-free automatic positivity floor `betaAuto`.
    #[wasm_bindgen(getter, js_name = betaAuto)]
    pub fn beta_auto(&self) -> f64 {
        self.stats.beta_auto
    }

    /// Returns `true` if a coarse diffusive scaffold table was computed.
    #[wasm_bindgen(getter, js_name = usedCoarseScaffold)]
    pub fn used_coarse_scaffold(&self) -> bool {
        self.stats.used_coarse_scaffold
    }

    /// Evaluates landmark ALT lower-bound distances for pairs `(u[i], v[i])`.
    #[wasm_bindgen(js_name = altDistances)]
    pub fn alt_distances(&self, u: &[u32], v: &[u32]) -> Result<Vec<f64>, JsError> {
        self.inner.alt_pair_distances(u, v).map_err(core_err_to_js)
    }
}

/// Smallest nontrivial eigenpairs of the symmetric normalized Laplacian $L_{\mathrm{sym}}$.
#[wasm_bindgen(js_name = EigenResult)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsEigenResult {
    inner: EigenResult,
}

#[wasm_bindgen(js_class = EigenResult)]
impl JsEigenResult {
    /// Returns the number of vertices in the graph.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes
    }

    /// Returns the number of returned nontrivial eigenmodes.
    #[wasm_bindgen(getter, js_name = numModes)]
    pub fn num_modes(&self) -> usize {
        self.inner.num_modes
    }

    /// Returns `true` when the dense Jacobi eigensolver fallback path was used.
    #[wasm_bindgen(getter, js_name = usedDenseFallback)]
    pub fn used_dense_fallback(&self) -> bool {
        self.inner.used_dense_fallback
    }

    /// Returns the ascending nontrivial eigenvalues (`length === numModes`).
    pub fn lambdas(&self) -> Vec<f64> {
        self.inner.lambdas.clone()
    }

    /// Returns the flat row-major `(numNodes, numModes)` orthonormal eigenvectors.
    pub fn vectors(&self) -> Vec<f64> {
        self.inner.vectors.clone()
    }

    /// Returns the $D^{-1/2}$ inverse square-root degree vector (`length === numNodes`).
    #[wasm_bindgen(js_name = invSqrtDeg)]
    pub fn inv_sqrt_deg(&self) -> Vec<f64> {
        self.inner.inv_sqrt_deg.clone()
    }
}

/// Exact $O(|E|)$ spectral variance and degree distribution summary statistics of $L_{\mathrm{sym}}$.
#[wasm_bindgen(js_name = SpectralMomentStats)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsSpectralMomentStats {
    inner: SpectralMomentStats,
}

#[wasm_bindgen(js_class = SpectralMomentStats)]
impl JsSpectralMomentStats {
    pub(crate) fn from_inner(inner: SpectralMomentStats) -> Self {
        Self { inner }
    }

    /// Returns the exact spectral variance $\sigma_\lambda^2$.
    #[wasm_bindgen(getter, js_name = sigmaLambdaSq)]
    pub fn sigma_lambda_sq(&self) -> f64 {
        self.inner.sigma_lambda_sq
    }

    /// Returns the clamped spectral standard deviation $\sigma_\lambda$.
    #[wasm_bindgen(getter, js_name = sigmaLambda)]
    pub fn sigma_lambda(&self) -> f64 {
        self.inner.sigma_lambda
    }

    /// Returns the degree coefficient of variation ($\mathrm{std}(d)/\mathrm{mean}(d)$).
    #[wasm_bindgen(getter, js_name = degCv)]
    pub fn deg_cv(&self) -> f64 {
        self.inner.deg_cv
    }

    /// Returns the mean unweighted simple degree across all vertices.
    #[wasm_bindgen(getter, js_name = meanDegree)]
    pub fn mean_degree(&self) -> f64 {
        self.inner.mean_degree
    }

    /// Returns the fraction of vertices with degree equal to 1.
    #[wasm_bindgen(getter, js_name = leafFraction)]
    pub fn leaf_fraction(&self) -> f64 {
        self.inner.leaf_fraction
    }
}

/// Disjoint unweighted 1-hop and pure 2-hop CSR graphs returned by `extract1hop2hopCsr`.
#[wasm_bindgen(js_name = TwoHopCsr)]
#[derive(Debug, Clone)]
pub struct JsTwoHopCsr {
    hop1: CsrGraph<f32>,
    hop2: CsrGraph<f32>,
}

#[wasm_bindgen(js_class = TwoHopCsr)]
impl JsTwoHopCsr {
    /// Returns a cloned handle to the canonical unweighted 1-hop `CsrGraph`.
    pub fn hop1(&self) -> JsCsrGraph {
        JsCsrGraph::from_inner(self.hop1.clone())
    }

    /// Returns a cloned handle to the disjoint pure 2-hop `CsrGraph`.
    pub fn hop2(&self) -> JsCsrGraph {
        JsCsrGraph::from_inner(self.hop2.clone())
    }
}

/// Pure 1+2-hop CSR `KnnGraph` and summary degree statistics returned by `buildPure2hopKnn`.
#[wasm_bindgen(js_name = TwoHopKnnResult)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsTwoHopKnnResult {
    knn: KnnGraph,
    k2_deg: Vec<u32>,
    stats: KnnStats,
}

#[wasm_bindgen(js_class = TwoHopKnnResult)]
impl JsTwoHopKnnResult {
    /// Returns a cloned `KnnGraph` containing exact 1-hop (`1.0`) and 2-hop (`2.0`) neighbors.
    pub fn knn(&self) -> JsKnnGraph {
        JsKnnGraph {
            inner: self.knn.clone(),
        }
    }

    /// Returns per-node 2-hop neighborhood sizes $|B_2(u) \setminus \{u\}|$ (`Uint32Array`).
    #[wasm_bindgen(js_name = k2Deg)]
    pub fn k2_deg(&self) -> Vec<u32> {
        self.k2_deg.clone()
    }

    /// Returns the mean 1-hop degree across all vertices.
    #[wasm_bindgen(getter, js_name = avgDeg1)]
    pub fn avg_deg1(&self) -> f64 {
        self.stats.avg_deg1
    }

    /// Returns the mean 1+2-hop neighborhood size across all vertices.
    #[wasm_bindgen(getter, js_name = avgDeg2)]
    pub fn avg_deg2(&self) -> f64 {
        self.stats.avg_deg2
    }

    /// Returns the total number of stored neighbor entries.
    #[wasm_bindgen(getter)]
    pub fn nnz(&self) -> usize {
        self.stats.nnz
    }
}

/// Hybrid 1+2-hop + 2-stage reranked `KnnGraph` and diagnostics returned by `buildHybridAdaptiveKnn`.
#[wasm_bindgen(js_name = HybridKnnResult)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsHybridKnnResult {
    knn: KnnGraph,
    stats: KnnStats,
}

#[wasm_bindgen(js_class = HybridKnnResult)]
impl JsHybridKnnResult {
    /// Returns a cloned hybrid adaptive `KnnGraph`.
    pub fn knn(&self) -> JsKnnGraph {
        JsKnnGraph {
            inner: self.knn.clone(),
        }
    }

    /// Returns the mean 1-hop degree across all vertices.
    #[wasm_bindgen(getter, js_name = avgDeg1)]
    pub fn avg_deg1(&self) -> f64 {
        self.stats.avg_deg1
    }

    /// Returns the mean 1+2-hop neighborhood size across all vertices.
    #[wasm_bindgen(getter, js_name = avgDeg2)]
    pub fn avg_deg2(&self) -> f64 {
        self.stats.avg_deg2
    }

    /// Returns the mean output row degree in the constructed `KnnGraph`.
    #[wasm_bindgen(getter, js_name = avgRowK)]
    pub fn avg_row_k(&self) -> f64 {
        self.stats.avg_row_k
    }

    /// Returns the total number of stored neighbor entries.
    #[wasm_bindgen(getter)]
    pub fn nnz(&self) -> usize {
        self.stats.nnz
    }

    /// Returns the number of rows augmented with non-2-hop residual candidates.
    #[wasm_bindgen(getter, js_name = nFilledResidual)]
    pub fn n_filled_residual(&self) -> usize {
        self.stats.n_filled_residual
    }

    /// Returns the number of rows composed solely of 1-hop and 2-hop neighbors.
    #[wasm_bindgen(getter, js_name = nPure2hopRows)]
    pub fn n_pure_2hop_rows(&self) -> usize {
        self.stats.n_pure_2hop_rows
    }
}

/// 1. Selects well-spaced shortest-path pivots via Core-First MaxMin BFS.
#[wasm_bindgen(js_name = selectMaxminSpPivots)]
pub fn js_select_maxmin_sp_pivots(
    graph: &JsCsrGraph,
    num_pivots: Option<f64>,
    seed: Option<f64>,
    min_degree: Option<f64>,
) -> Result<Vec<u32>, JsError> {
    let pivots = parse_optional_usize(num_pivots, DEFAULT_NUM_PIVOTS, "numPivots")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let min_deg = parse_optional_usize(min_degree, DEFAULT_MIN_DEGREE, "minDegree")?;
    select_maxmin_sp_pivots(graph.inner(), pivots, seed_u64, min_deg).map_err(core_err_to_js)
}

/// 2. Computes a 2D `Layout2d` (`f32`) via Core-First MaxMin PivotMDS.
///
/// When `targetStd` is omitted or `undefined`, coordinates are normalized to `targetStd = 1.0`.
/// Pass `null` to disable isotropic standard-deviation normalization.
#[wasm_bindgen(js_name = pivotMdsInit)]
pub fn js_pivot_mds_init(
    graph: &JsCsrGraph,
    num_pivots: Option<f64>,
    seed: Option<f64>,
    min_degree: Option<f64>,
    target_std: &JsValue,
) -> Result<JsLayout2d, JsError> {
    let pivots = parse_optional_usize(num_pivots, DEFAULT_NUM_PIVOTS, "numPivots")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let min_deg = parse_optional_usize(min_degree, DEFAULT_MIN_DEGREE, "minDegree")?;
    let t_std = parse_optional_target_std(target_std)?;
    let cfg = PivotMdsConfig {
        num_pivots: pivots,
        min_degree: min_deg,
        seed: seed_u64,
        target_std: t_std,
    };
    let layout: Layout2d<f32> = pivot_mds(graph.inner(), &cfg).map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&layout))
}

/// 3. Computes Koren's degree-normalized 2D spectral initialization `Layout2d` (`f32`).
///
/// When `targetStd` is omitted or `undefined`, coordinates are normalized to `targetStd = 1.0`.
/// Pass `null` to disable isotropic standard-deviation normalization.
#[wasm_bindgen(js_name = korenSpectralInit)]
pub fn js_koren_spectral_init(
    graph: &JsCsrGraph,
    target_std: &JsValue,
    tol: Option<f64>,
    max_iters: Option<f64>,
    seed: Option<f64>,
) -> Result<JsLayout2d, JsError> {
    let t_std = parse_optional_target_std(target_std)?;
    let tol_val = tol.unwrap_or(DEFAULT_TOL);
    if !tol_val.is_finite() || tol_val <= 0.0 {
        return Err(JsError::new("tol must be finite and strictly positive"));
    }
    let max_iters_val = parse_optional_usize(max_iters, DEFAULT_MAX_ITERS, "maxIters")?;
    if max_iters_val == 0 {
        return Err(JsError::new("maxIters must be >= 1"));
    }
    let seed_u64 = parse_seed_u64(seed)?;
    let layout: Layout2d<f32> =
        koren_spectral_init(graph.inner(), t_std, tol_val, max_iters_val, seed_u64)
            .map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&layout))
}

/// 4. Computes the smallest `nModes` nontrivial eigenpairs of $L_{\mathrm{sym}}$.
#[wasm_bindgen(js_name = smallestLaplacianEigenpairs)]
pub fn js_smallest_laplacian_eigenpairs(
    graph: &JsCsrGraph,
    n_modes: Option<f64>,
    tol: Option<f64>,
    max_iters: Option<f64>,
    seed: Option<f64>,
) -> Result<JsEigenResult, JsError> {
    let modes = parse_optional_usize(n_modes, DEFAULT_N_MODES, "nModes")?;
    let tol_val = tol.unwrap_or(DEFAULT_TOL);
    let max_iters_val = parse_optional_usize(max_iters, DEFAULT_MAX_ITERS, "maxIters")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let inner =
        smallest_sym_laplacian_eigenpairs(graph.inner(), modes, tol_val, max_iters_val, seed_u64)
            .map_err(core_err_to_js)?;
    Ok(JsEigenResult { inner })
}

/// 5. Computes exact $O(|E|)$ spectral variance and degree summary statistics of $L_{\mathrm{sym}}$.
#[wasm_bindgen(js_name = spectralMomentStats)]
pub fn js_spectral_moment_stats(graph: &JsCsrGraph) -> JsSpectralMomentStats {
    JsSpectralMomentStats {
        inner: spectral_moment_stats(graph.inner()),
    }
}

/// 6. Applies Chebyshev heat diffusion $\exp(-t L_{\mathrm{sym}})$ to a row-major `(numNodes, numCols)` buffer.
#[wasm_bindgen(js_name = chebyshevHeatSpmv)]
pub fn js_chebyshev_heat_spmv(
    graph: &JsCsrGraph,
    vectors: &[f64],
    num_cols: f64,
    t: f64,
    degree: Option<f64>,
    lambda_max: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    let ncols = parse_non_negative_usize(num_cols, "numCols")?;
    let lmax = lambda_max.unwrap_or(DEFAULT_LAMBDA_MAX);
    let deg = match degree {
        Some(d) => parse_non_negative_usize(d, "degree")?,
        None => adaptive_chebyshev_degree(t),
    };
    if vectors.iter().any(|v| !v.is_finite()) {
        return Err(JsError::new("vectors must contain only finite values"));
    }
    let coeffs =
        compute_chebyshev_bandpass_coefficients(&[(1.0, t)], lmax, deg).map_err(core_err_to_js)?;
    let n = graph.inner().num_nodes();
    let total = n
        .checked_mul(ncols)
        .ok_or_else(|| JsError::new("chebyshevHeatSpmv matrix size overflow"))?;
    let mut out = vec![0.0_f64; total];
    let (l_sym, _) = symmetric_normalized_laplacian(graph.inner());
    let mut ws = ChebyshevWorkspace::with_capacity(total);
    ws.apply_batch_with_lambda_max(&l_sym, lmax, &coeffs, vectors, &mut out, ncols)
        .map_err(core_err_to_js)?;
    Ok(out)
}

/// 7. Computes a deterministic R1/R2 regularized DoG JL spectral embedding `(numNodes, jlDim)`.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = chebyshevDogJlEmbedding)]
pub fn js_chebyshev_dog_jl_embedding(
    graph: &JsCsrGraph,
    jl_dim: Option<f64>,
    degree: Option<f64>,
    seed: Option<f64>,
    degree_normalize: Option<bool>,
    eps_psd: Option<f64>,
    t1: Option<f64>,
    t2: Option<f64>,
    alpha: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    let jl_dim_val = parse_optional_usize(jl_dim, DEFAULT_JL_DIM, "jlDim")?;
    let deg = parse_optional_usize(degree, DEFAULT_DOG_DEGREE, "degree")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let deg_norm = degree_normalize.unwrap_or(true);
    let eps_psd_val = eps_psd.unwrap_or(DEFAULT_EPS_PSD);

    let stats = spectral_moment_stats(graph.inner());
    let s = stats.sigma_lambda;
    let t1_eff = t1.unwrap_or(1.0 / (1.0 + s));
    let t2_eff = t2.unwrap_or(4.0 / (1.0 + s));
    let alpha_eff = alpha.unwrap_or((1.0 - s).max(0.0));

    let base_terms = build_dog_terms(t1_eff, t2_eff, alpha_eff, 0.0).map_err(core_err_to_js)?;
    let (terms_psd, _delta_psd, _h_min) =
        project_psd_spectral_filter(&base_terms, DEFAULT_LAMBDA_MAX, deg, eps_psd_val)
            .map_err(core_err_to_js)?;

    let (l_sym, inv_sqrt_deg) = symmetric_normalized_laplacian(graph.inner());
    let mut ws = ChebyshevWorkspace::new();
    ws.compute_kernel_jl_embedding(
        &l_sym,
        &terms_psd,
        DEFAULT_LAMBDA_MAX,
        deg,
        jl_dim_val,
        seed_u64,
        deg_norm.then_some(inv_sqrt_deg.as_slice()),
    )
    .map_err(core_err_to_js)
}

/// 8. Extracts canonical unweighted 1-hop and disjoint pure 2-hop CSR graphs.
#[wasm_bindgen(js_name = extract1hop2hopCsr)]
pub fn js_extract_1hop_2hop_csr(graph: &JsCsrGraph) -> Result<JsTwoHopCsr, JsError> {
    let (adj1, adj2) = extract_1hop_2hop_csr(graph.inner()).map_err(core_err_to_js)?;
    let hop1 = unweighted_f64_csr_to_f32(&adj1)?;
    let hop2 = unweighted_f64_csr_to_f32(&adj2)?;
    Ok(JsTwoHopCsr { hop1, hop2 })
}

/// 9. Builds a pure 1+2-hop CSR `KnnGraph` with `{1.0, 2.0}` distances and degree statistics.
#[wasm_bindgen(js_name = buildPure2hopKnn)]
pub fn js_build_pure_2hop_knn(graph: &JsCsrGraph) -> Result<JsTwoHopKnnResult, JsError> {
    let (knn, k2_deg, stats) = build_pure_2hop_knn(graph.inner()).map_err(core_err_to_js)?;
    Ok(JsTwoHopKnnResult { knn, k2_deg, stats })
}

/// 10. Computes exact distinct 1-hop + 2-hop neighborhood sizes $|B_2(u) \setminus \{u\}|$.
#[wasm_bindgen(js_name = compute2hopNeighborhoodSizes)]
pub fn js_compute_2hop_neighborhood_sizes(graph: &JsCsrGraph) -> Vec<u32> {
    compute_2hop_neighborhood_sizes(graph.inner())
}

/// 11. Computes per-node 2-hop adaptive perplexities.
#[wasm_bindgen(js_name = computeNodewise2hopPerplexities)]
pub fn js_compute_nodewise_2hop_perplexities(
    graph: &JsCsrGraph,
    c: Option<f64>,
    min_perplexity: Option<f64>,
    max_perplexity: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    compute_nodewise_2hop_perplexities(
        graph.inner(),
        c.unwrap_or(DEFAULT_NODEWISE_PERPLEXITY_C),
        min_perplexity.unwrap_or(DEFAULT_NODEWISE_MIN_PERPLEXITY),
        max_perplexity.unwrap_or(DEFAULT_MAX_PERPLEXITY),
    )
    .map_err(core_err_to_js)
}

/// 12. Computes scalar median 2-hop adaptive perplexity.
#[wasm_bindgen(js_name = computeMedian2hopPerplexity)]
pub fn js_compute_median_2hop_perplexity(
    graph: &JsCsrGraph,
    c: Option<f64>,
    min_perplexity: Option<f64>,
    max_perplexity: Option<f64>,
) -> Result<f64, JsError> {
    compute_median_2hop_perplexity(
        graph.inner(),
        c.unwrap_or(DEFAULT_MEDIAN_PERPLEXITY_C),
        min_perplexity.unwrap_or(DEFAULT_MEDIAN_MIN_PERPLEXITY),
        max_perplexity.unwrap_or(DEFAULT_MAX_PERPLEXITY),
    )
    .map_err(core_err_to_js)
}

/// 13. Builds a hybrid 1+2-hop + 2-stage coarse `StaticKdTree` + full-$D$ reranked `KnnGraph`.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = buildHybridAdaptiveKnn)]
pub fn js_build_hybrid_adaptive_knn(
    graph: &JsCsrGraph,
    coords: &[f64],
    dim: f64,
    perplexities: &[f64],
    k_factor: Option<f64>,
    margin_gap: Option<f64>,
    coarse_dim: Option<f64>,
    rerank_factor: Option<f64>,
    k: Option<f64>,
    max_row_k: Option<f64>,
) -> Result<JsHybridKnnResult, JsError> {
    let dim_val = parse_non_negative_usize(dim, "dim")?;
    let cfg = HybridKnnConfig {
        k_factor: k_factor.unwrap_or(DEFAULT_K_FACTOR),
        k_override: parse_opt_usize_param(k, "k")?,
        margin_gap: margin_gap.unwrap_or(DEFAULT_MARGIN_GAP),
        coarse_dim: parse_optional_usize(coarse_dim, DEFAULT_COARSE_DIM, "coarseDim")?,
        rerank_factor: parse_optional_usize(rerank_factor, DEFAULT_RERANK_FACTOR, "rerankFactor")?,
        max_row_k: parse_opt_usize_param(max_row_k, "maxRowK")?,
    };
    let (knn, stats) =
        build_hybrid_adaptive_knn(graph.inner(), coords, dim_val, perplexities, &cfg)
            .map_err(core_err_to_js)?;
    Ok(JsHybridKnnResult { knn, stats })
}

/// 14. Computes a $k$-NN graph via Partial BFS with deterministic Fisher-Yates tie-breaking.
#[wasm_bindgen(js_name = partialBfsKnn)]
pub fn js_partial_bfs_knn(
    graph: &JsCsrGraph,
    k: Option<f64>,
    seed: Option<f64>,
) -> Result<JsKnnGraph, JsError> {
    let k_val = parse_optional_usize(k, DEFAULT_PARTIAL_BFS_K, "k")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let inner = partial_bfs_knn(graph.inner(), k_val, seed_u64).map_err(core_err_to_js)?;
    Ok(JsKnnGraph { inner })
}

/// 15. Computes sparse symmetric joint probabilities from a `KnnGraph` and per-node perplexities.
#[wasm_bindgen(js_name = computeKnnProbabilities)]
pub fn js_compute_knn_probabilities(
    knn: &JsKnnGraph,
    perplexities: &[f64],
    power: Option<f64>,
    sigma_iters: Option<f64>,
    sigma_tolerance: Option<f64>,
    row_normalize: Option<bool>,
) -> Result<JsSparseProbabilities, JsError> {
    let cfg = PerplexityConfig {
        power: power.unwrap_or(DEFAULT_POWER),
        sigma_iters: parse_optional_usize(
            sigma_iters,
            DEFAULT_PERPLEXITY_SIGMA_ITERS,
            "sigmaIters",
        )?,
        sigma_tolerance: sigma_tolerance.unwrap_or(DEFAULT_SIGMA_TOLERANCE),
        symmetrization: if row_normalize.unwrap_or(false) {
            SymmetrizationMode::RowNormalized
        } else {
            SymmetrizationMode::StandardTsne
        },
    };
    let inner =
        compute_knn_probabilities(knn.inner(), perplexities, &cfg).map_err(core_err_to_js)?;
    Ok(JsSparseProbabilities { inner })
}

/// 16. Computes the UMAP / GuMap fuzzy simplicial set probabilities from a `KnnGraph`.
#[wasm_bindgen(js_name = computeFuzzySimplicialSet)]
pub fn js_compute_fuzzy_simplicial_set(
    knn: &JsKnnGraph,
    sigma_iters: Option<f64>,
    sigma_tolerance: Option<f64>,
) -> Result<JsSparseProbabilities, JsError> {
    let iters = parse_optional_usize(sigma_iters, DEFAULT_FUZZY_SIGMA_ITERS, "sigmaIters")?;
    let tol = sigma_tolerance.unwrap_or(DEFAULT_SIGMA_TOLERANCE);
    let inner = compute_fuzzy_simplicial_set(knn.inner(), iters, tol).map_err(core_err_to_js)?;
    Ok(JsSparseProbabilities { inner })
}

/// 17. Fits low-dimensional UMAP parameters `[a, b]` from `minDist` and `spread`.
#[wasm_bindgen(js_name = findAbParams)]
pub fn js_find_ab_params(min_dist: Option<f64>, spread: Option<f64>) -> Result<Vec<f64>, JsError> {
    let (a, b) = find_ab_params(
        min_dist.unwrap_or(DEFAULT_MIN_DIST),
        spread.unwrap_or(DEFAULT_SPREAD),
    )
    .map_err(core_err_to_js)?;
    Ok(vec![a, b])
}

/// 18. Computes R3 parameter-free Horvitz-Thompson repulsion scaling $\gamma_{\mathrm{eff}}$.
#[wasm_bindgen(js_name = computeHorvitzThompsonGamma)]
pub fn js_compute_horvitz_thompson_gamma(
    gamma_non2: f64,
    num_nodes: f64,
    neg_per_node: f64,
) -> Result<f64, JsError> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let neg = parse_non_negative_usize(neg_per_node, "negPerNode")?;
    compute_horvitz_thompson_gamma(gamma_non2, n, neg).map_err(core_err_to_js)
}

/// 19. Computes regularized DoG Varadhan landmark tables and optional coarse heat scaffold.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = computeRegularizedDogVaradhanLandmarks)]
pub fn js_compute_regularized_dog_varadhan_landmarks(
    graph: &JsCsrGraph,
    degree: Option<f64>,
    num_landmarks: Option<f64>,
    seed: Option<f64>,
    degree_normalize: Option<bool>,
    p_exp: Option<f64>,
    eps_psd: Option<f64>,
    t_coarse: Option<f64>,
    t1: Option<f64>,
    t2: Option<f64>,
    alpha: Option<f64>,
    beta_floor: Option<f64>,
) -> Result<JsDogVaradhanLandmarks, JsError> {
    let deg = parse_opt_usize_param(degree, "degree")?;
    let n_landmarks =
        parse_optional_usize(num_landmarks, DEFAULT_VARADHAN_LANDMARKS, "numLandmarks")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let cfg = DogVaradhanConfig {
        t1,
        t2,
        alpha,
        beta_floor,
        degree: deg,
        n_landmarks,
        degree_normalize: degree_normalize.unwrap_or(true),
        p_exp: p_exp.unwrap_or(DEFAULT_P_EXP),
        eps_psd: eps_psd.unwrap_or(DEFAULT_EPS_PSD),
        t_coarse: t_coarse.unwrap_or(DEFAULT_T_COARSE),
        seed: seed_u64,
        ..DogVaradhanConfig::default()
    };
    let (inner, stats) =
        compute_regularized_dog_varadhan_landmarks(graph.inner(), &cfg).map_err(core_err_to_js)?;
    Ok(JsDogVaradhanLandmarks { inner, stats })
}

/// Spectral embedding output returned by `computeResistanceEmbedding` and `computeDiffusionEmbedding`.
#[wasm_bindgen(js_name = SpectralEmbeddingResult)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsSpectralEmbeddingResult {
    coords: Vec<f64>,
    lambdas: Vec<f64>,
    num_nodes: usize,
    dim: usize,
    t_used: Option<f64>,
}

#[wasm_bindgen(js_class = SpectralEmbeddingResult)]
impl JsSpectralEmbeddingResult {
    /// Returns the flat row-major `(numNodes, dim)` `Float64Array` embedding coordinates.
    pub fn coords(&self) -> Vec<f64> {
        self.coords.clone()
    }

    /// Returns the kept nontrivial Laplacian eigenvalues (`Float64Array` of length `dim`).
    pub fn lambdas(&self) -> Vec<f64> {
        self.lambdas.clone()
    }

    /// Returns the number of vertices.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the effective embedding dimension (`lambdas().length`).
    #[wasm_bindgen(getter)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the effective diffusion time $t_{\mathrm{used}}$ (for `computeDiffusionEmbedding`), or `undefined`.
    #[wasm_bindgen(getter, js_name = tUsed)]
    pub fn t_used(&self) -> Option<f64> {
        self.t_used
    }
}

/// 20. Computes a 2D `Layout2d` (`f32`) using zero-guarded Classical MDS.
#[wasm_bindgen(js_name = classicalMds2d)]
pub fn js_classical_mds_2d(graph: &JsCsrGraph) -> Result<JsLayout2d, JsError> {
    let layout: Layout2d<f32> = classical_mds_2d(graph.inner()).map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&layout))
}

/// 21. Computes a `LayoutNd` (`f32`) zero-guarded resistance-distance spectral embedding (`RdMds`).
#[wasm_bindgen(js_name = rdMds)]
pub fn js_rd_mds(
    graph: &JsCsrGraph,
    dims: Option<f64>,
    eig_eps: Option<f64>,
    tol: Option<f64>,
    max_iters: Option<f64>,
    seed: Option<f64>,
) -> Result<JsLayoutNd, JsError> {
    let d = parse_optional_usize(dims, 2, "dims")?;
    let eps = eig_eps.unwrap_or(1e-10);
    let tol_val = tol.unwrap_or(1e-6);
    let m_iters = parse_optional_usize(max_iters, 1000, "maxIters")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let layout: LayoutNd<f32> =
        rd_mds(graph.inner(), d, eps, tol_val, m_iters, seed_u64).map_err(core_err_to_js)?;
    Ok(JsLayoutNd::from_inner(layout))
}

/// 22. Sanitizes non-finite entries, centers `layout` in place, and scales isotropically to `targetStd` (default `1.0`).
#[wasm_bindgen(js_name = normalizeLayout2d)]
pub fn js_normalize_layout_2d(
    layout: &mut JsLayout2d,
    target_std: Option<f64>,
    seed: Option<f64>,
) -> Result<(), JsError> {
    let t_std = target_std.unwrap_or(1.0);
    let seed_u64 = parse_seed_u64(seed)?;
    let mut soa = layout.inner().to_layout2d();
    normalize_layout_2d(&mut soa, t_std, seed_u64).map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(())
}

/// 23. Computes a `dim`-dimensional Representative / Landmark MDS `LayoutNd` (`f32`) from a flat
/// row-major `(numNodes, numReps)` distance matrix.
#[wasm_bindgen(js_name = representativeMds)]
pub fn js_representative_mds(
    distances_to_reps: &[f64],
    num_nodes: f64,
    num_reps: f64,
    dims: f64,
) -> Result<JsLayoutNd, JsError> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let r = parse_non_negative_usize(num_reps, "numReps")?;
    let d = parse_non_negative_usize(dims, "dims")?;
    let layout: LayoutNd<f32> =
        representative_mds(distances_to_reps, n, r, d).map_err(core_err_to_js)?;
    Ok(JsLayoutNd::from_inner(layout))
}

/// 24. Computes effective-resistance spectral embedding $D^{-1/2} U \Lambda^{-1/2}$.
#[wasm_bindgen(js_name = computeResistanceEmbedding)]
pub fn js_compute_resistance_embedding(
    graph: &JsCsrGraph,
    dim: Option<f64>,
    tol: Option<f64>,
    max_iters: Option<f64>,
    seed: Option<f64>,
    eig_eps: Option<f64>,
) -> Result<JsSpectralEmbeddingResult, JsError> {
    let d = parse_optional_usize(dim, DEFAULT_N_MODES, "dim")?;
    let tol_val = tol.unwrap_or(DEFAULT_TOL);
    let m_iters = parse_optional_usize(max_iters, DEFAULT_MAX_ITERS, "maxIters")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let eps = eig_eps.unwrap_or(1e-10);
    let (coords, lambdas) =
        compute_resistance_embedding(graph.inner(), d, tol_val, m_iters, seed_u64, eps)
            .map_err(core_err_to_js)?;
    let eff_dim = lambdas.len();
    Ok(JsSpectralEmbeddingResult {
        coords,
        lambdas,
        num_nodes: graph.inner().num_nodes(),
        dim: eff_dim,
        t_used: None,
    })
}

/// 25. Computes heat-diffusion spectral embedding $D^{-1/2} U \exp(-0.5 t \Lambda)$.
#[wasm_bindgen(js_name = computeDiffusionEmbedding)]
pub fn js_compute_diffusion_embedding(
    graph: &JsCsrGraph,
    dim: Option<f64>,
    p: Option<f64>,
    t: Option<f64>,
    tol: Option<f64>,
    max_iters: Option<f64>,
    seed: Option<f64>,
) -> Result<JsSpectralEmbeddingResult, JsError> {
    let d = parse_optional_usize(dim, DEFAULT_N_MODES, "dim")?;
    let p_val = p.unwrap_or(0.05);
    let tol_val = tol.unwrap_or(DEFAULT_TOL);
    let m_iters = parse_optional_usize(max_iters, DEFAULT_MAX_ITERS, "maxIters")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let (coords, lambdas, t_used) =
        compute_diffusion_embedding(graph.inner(), d, p_val, t, tol_val, m_iters, seed_u64)
            .map_err(core_err_to_js)?;
    let eff_dim = lambdas.len();
    Ok(JsSpectralEmbeddingResult {
        coords,
        lambdas,
        num_nodes: graph.inner().num_nodes(),
        dim: eff_dim,
        t_used: Some(t_used),
    })
}

/// 26. Constructs a 2-stage coarse `StaticKdTree` + full-`dim` reranked `KnnGraph` from row-major
/// `(numNodes, dim)` coordinates.
#[wasm_bindgen(js_name = buildRerankKnn)]
pub fn js_build_rerank_knn(
    coords: &[f64],
    num_nodes: f64,
    dim: f64,
    k: f64,
    coarse_dim: Option<f64>,
    rerank_factor: Option<f64>,
) -> Result<JsKnnGraph, JsError> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let d = parse_non_negative_usize(dim, "dim")?;
    let k_val = parse_non_negative_usize(k, "k")?;
    let c_dim = parse_optional_usize(coarse_dim, DEFAULT_COARSE_DIM, "coarseDim")?;
    let r_factor = parse_optional_usize(rerank_factor, DEFAULT_RERANK_FACTOR, "rerankFactor")?;
    let inner = build_rerank_knn(coords, n, d, k_val, c_dim, r_factor).map_err(core_err_to_js)?;
    Ok(JsKnnGraph { inner })
}

/// 27. Computes row-normalized symmetrized graph probabilities from a `CsrGraph`.
#[wasm_bindgen(js_name = graphRowNormalizedProbabilities)]
pub fn js_graph_row_normalized_probabilities(
    graph: &JsCsrGraph,
) -> Result<JsSparseProbabilities, JsError> {
    let inner = graph_row_normalized_probabilities(graph.inner()).map_err(core_err_to_js)?;
    Ok(JsSparseProbabilities { inner })
}

/// 28. Computes row-normalized symmetrized joint probabilities from arbitrary CSR directed affinities.
#[wasm_bindgen(js_name = symmetrizeRowNormalized)]
pub fn js_symmetrize_row_normalized(
    num_nodes: f64,
    indptr: &[u32],
    indices: &[u32],
    weights: &[f64],
) -> Result<JsSparseProbabilities, JsError> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let indptr_usize: Vec<usize> = indptr.iter().map(|&v| v as usize).collect();
    let inner =
        symmetrize_row_normalized(n, &indptr_usize, indices, weights).map_err(core_err_to_js)?;
    Ok(JsSparseProbabilities { inner })
}

/// 29. Builds a sorted symmetric CSR `KnnGraph` from 1-hop and 2-hop pair distances enforcing
/// strict 1-hop priority ($d_1 < \mathrm{tier2Margin} \cdot d_2$ and $d_1 < d_2$).
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = buildGumap12hopPriorityCsr)]
pub fn js_build_gumap_12hop_priority_csr(
    num_nodes: f64,
    u1: &[u32],
    v1: &[u32],
    d1: &[f64],
    u2: &[u32],
    v2: &[u32],
    d2: &[f64],
    tier2_margin: Option<f64>,
) -> Result<JsKnnGraph, JsError> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let margin = tier2_margin.unwrap_or(1.1);
    let inner = build_gumap_12hop_priority_csr(n, u1, v1, d1, u2, v2, d2, margin)
        .map_err(core_err_to_js)?;
    Ok(JsKnnGraph { inner })
}

/// 30. Clamps pair distances `d1`, `d2`, and `d3` in place using `max(epsDist, minRatio * median_pos)`,
/// returning the applied distance floor.
#[wasm_bindgen(js_name = applyRelativeDistanceFloor)]
pub fn js_apply_relative_distance_floor(
    d1: &mut [f64],
    d2: &mut [f64],
    d3: &mut [f64],
    eps_dist: Option<f64>,
    min_ratio: Option<f64>,
) -> Result<f64, JsError> {
    apply_relative_distance_floor(
        d1,
        d2,
        d3,
        eps_dist.unwrap_or(1e-3),
        min_ratio.unwrap_or(0.1),
    )
    .map_err(core_err_to_js)
}
