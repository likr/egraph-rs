//! M2 v6 WebAssembly bindings for `egraph-linalg` and `egraph-neighborhood`.
//!
//! Exposes pure-Rust spectral initialization, eigensolvers, Chebyshev heat/DoG diffusion,
//! 1+2-hop and hybrid adaptive $k$-NN graphs, perplexity / fuzzy simplicial probabilities,
//! and regularized DoG Varadhan landmark tables without `serde` or threading dependencies.

use egraph::core::{CoreError, CsrGraph, Layout2d, LayoutNd};
use egraph::linalg::{
    AmgPcgBatchResult, AmgPcgConfig, AmgPcgSolveResult, AmgPcgSolver, AmgPcgWorkspace,
    AnisotropicLaplacianConfig, ChebyshevWorkspace, CsrMatrix,
    DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR, DEFAULT_SPECTRUM_LANCZOS_STEPS,
    DEFAULT_SPECTRUM_POWER_ITERS, DEFAULT_SPECTRUM_SAFETY_FACTOR, EffectiveResistanceConfig,
    EffectiveResistanceResult, EigenResult, LaplacianKind, PivotMdsConfig,
    ResistanceCurvatureConfig, ResistanceCurvatureFlowResult, ResistanceCurvatureStep,
    SpectralMomentStats, adaptive_chebyshev_degree_scaled, anisotropic_laplacian, build_dog_terms,
    classical_mds_2d, compute_chebyshev_bandpass_coefficients, compute_effective_resistances,
    koren_spectral_init, normalize_layout_2d, pivot_mds, project_psd_spectral_filter, rd_mds,
    representative_mds, resistance_curvature_flow, robust_normalize_layout_2d,
    select_maxmin_sp_pivots, smallest_sym_laplacian_eigenpairs, spectral_moment_stats,
    symmetric_normalized_laplacian,
};
use egraph::neighborhood::{
    DEFAULT_MAX_PERPLEXITY, DEFAULT_MEDIAN_MIN_PERPLEXITY, DEFAULT_MEDIAN_PERPLEXITY_C,
    DEFAULT_NODEWISE_MIN_PERPLEXITY, DEFAULT_NODEWISE_PERPLEXITY_C, DogVaradhanConfig,
    DogVaradhanLandmarks, DogVaradhanStats, HybridKnnConfig, KnnGraph, KnnStats, PerplexityConfig,
    RhghLandmarkStats, Sparse12HopPairs, SparseProbabilities, SymmetrizationMode,
    apply_relative_distance_floor, build_capped_12hop_knn, build_gumap_12hop_priority_csr,
    build_hybrid_adaptive_knn, build_pure_2hop_knn, build_rerank_knn,
    compute_2hop_neighborhood_sizes, compute_diffusion_embedding, compute_fuzzy_simplicial_set,
    compute_horvitz_thompson_gamma, compute_knn_probabilities, compute_median_2hop_perplexity,
    compute_nodewise_2hop_perplexities, compute_regularized_dog_varadhan_landmarks,
    compute_regularized_dog_varadhan_landmarks_from_matrix, compute_resistance_embedding,
    extract_1hop_2hop_csr, extract_sparse_1hop_2hop_pairs, find_ab_params,
    graph_12hop_row_normalized_probabilities_with_degree_decay, graph_row_normalized_probabilities,
    partial_bfs_knn, symmetrize_row_normalized,
};
use wasm_bindgen::prelude::*;

use crate::v6::{
    JsCsrGraph, JsLayout2d, JsLayoutNd, core_err_to_js, ffi_err_to_js, parse_non_negative_usize,
    parse_seed_u64,
};
use crate::v6_m3::{
    get_prop, is_plain_js_object, normalize_enum_token, opt_bool_prop, opt_f64_prop, opt_seed_prop,
    opt_usize_prop, solver_status_str,
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
const DEFAULT_SPARSE_MAX_2HOP: usize = 32;

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

/// Extracted canonical undirected 1-hop and hub-safe capped 2-hop vertex pair arrays (`u < v`).
#[wasm_bindgen(js_name = Sparse12HopPairs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsSparse12HopPairs {
    inner: Sparse12HopPairs,
}

#[wasm_bindgen(js_class = Sparse12HopPairs)]
impl JsSparse12HopPairs {
    /// Returns the number of unique undirected 1-hop pairs (`u1 < v1`).
    #[wasm_bindgen(getter, js_name = num1hop)]
    pub fn num_1hop(&self) -> usize {
        self.inner.u1.len()
    }

    /// Returns the number of unique undirected 2-hop pairs (`u2 < v2`).
    #[wasm_bindgen(getter, js_name = num2hop)]
    pub fn num_2hop(&self) -> usize {
        self.inner.u2.len()
    }

    /// Returns a copy of 1-hop source vertex indices (`Uint32Array`).
    #[wasm_bindgen(getter)]
    pub fn u1(&self) -> Vec<u32> {
        self.inner.u1.clone()
    }

    /// Returns a copy of 1-hop target vertex indices (`Uint32Array`).
    #[wasm_bindgen(getter)]
    pub fn v1(&self) -> Vec<u32> {
        self.inner.v1.clone()
    }

    /// Returns a copy of 2-hop source vertex indices (`Uint32Array`).
    #[wasm_bindgen(getter)]
    pub fn u2(&self) -> Vec<u32> {
        self.inner.u2.clone()
    }

    /// Returns a copy of 2-hop target vertex indices (`Uint32Array`).
    #[wasm_bindgen(getter)]
    pub fn v2(&self) -> Vec<u32> {
        self.inner.v2.clone()
    }
}

/// Double-precision square Compressed Sparse Row matrix (`CsrMatrix`) with spectral bounds
/// and matrix-vector operations.
#[wasm_bindgen(js_name = CsrMatrix)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsCsrMatrix {
    inner: CsrMatrix,
}

impl JsCsrMatrix {
    /// Returns a shared reference to the underlying `egraph_linalg::CsrMatrix`.
    pub fn inner(&self) -> &CsrMatrix {
        &self.inner
    }
}

#[wasm_bindgen(js_class = CsrMatrix)]
impl JsCsrMatrix {
    /// Constructs a validated `CsrMatrix` from CSR `numNodes`, `indptr`, `indices`, and `values`.
    #[wasm_bindgen(js_name = fromCsr)]
    pub fn from_csr(
        num_nodes: f64,
        indptr: &[u32],
        indices: &[u32],
        values: &[f64],
    ) -> Result<JsCsrMatrix, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let inner = CsrMatrix::from_csr(n, indptr.to_vec(), indices.to_vec(), values.to_vec())
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns the matrix dimension `numNodes`.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the number of stored non-zero entries.
    #[wasm_bindgen(getter)]
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Returns a copy of CSR `indptr` (`Uint32Array`, length `numNodes + 1`).
    pub fn indptr(&self) -> Vec<u32> {
        self.inner.indptr().to_vec()
    }

    /// Returns a copy of sorted CSR column `indices` (`Uint32Array`, length `nnz`).
    pub fn indices(&self) -> Vec<u32> {
        self.inner.indices().to_vec()
    }

    /// Returns a copy of CSR `values` (`Float64Array`, length `nnz`).
    pub fn values(&self) -> Vec<f64> {
        self.inner.values().to_vec()
    }

    /// Returns a copy of the main diagonal entries (`Float64Array`, length `numNodes`).
    pub fn diagonal(&self) -> Vec<f64> {
        self.inner.diagonal().to_vec()
    }

    /// Returns the matrix trace `sum(diagonal)`.
    pub fn trace(&self) -> f64 {
        self.inner.trace()
    }

    /// Returns the symmetric Gershgorin upper spectral bound `max_i (|A_ii| + sum_{j!=i} |A_ij|)`.
    #[wasm_bindgen(js_name = gershgorinSpectralBound)]
    pub fn gershgorin_spectral_bound(&self) -> f64 {
        self.inner.gershgorin_spectral_bound()
    }

    /// Estimates `lambdaMax` via deterministic Rademacher power iteration clamped by Gershgorin.
    #[wasm_bindgen(js_name = estimateLambdaMax)]
    pub fn estimate_lambda_max(
        &self,
        num_iters: Option<f64>,
        safety_factor: Option<f64>,
        seed: Option<f64>,
    ) -> Result<f64, JsError> {
        let iters = parse_optional_usize(num_iters, DEFAULT_SPECTRUM_POWER_ITERS, "numIters")?;
        let sf = safety_factor.unwrap_or(DEFAULT_SPECTRUM_SAFETY_FACTOR);
        let s = parse_seed_u64(seed)?;
        self.inner
            .estimate_lambda_max(iters, sf, s)
            .map_err(core_err_to_js)
    }

    /// Estimates `lambdaMax` via deterministic Lanczos tridiagonalization clamped by Gershgorin.
    #[wasm_bindgen(js_name = estimateLambdaMaxLanczos)]
    pub fn estimate_lambda_max_lanczos(
        &self,
        num_steps: Option<f64>,
        safety_factor: Option<f64>,
        seed: Option<f64>,
    ) -> Result<f64, JsError> {
        let steps = parse_optional_usize(num_steps, DEFAULT_SPECTRUM_LANCZOS_STEPS, "numSteps")?;
        let sf = safety_factor.unwrap_or(DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR);
        let s = parse_seed_u64(seed)?;
        self.inner
            .estimate_lambda_max_lanczos(steps, sf, s)
            .map_err(core_err_to_js)
    }

    /// Computes sparse matrix-vector product `y = A * x`.
    pub fn spmv(&self, x: &[f64]) -> Result<Vec<f64>, JsError> {
        if x.iter().any(|v| !v.is_finite()) {
            return Err(JsError::new("x must contain only finite values"));
        }
        let mut out = vec![0.0_f64; self.inner.num_nodes()];
        self.inner.spmv(x, &mut out).map_err(core_err_to_js)?;
        Ok(out)
    }

    /// Returns `scale * A + shift * I`.
    #[wasm_bindgen(js_name = scaleAndShift)]
    pub fn scale_and_shift(&self, scale: f64, shift: f64) -> Result<JsCsrMatrix, JsError> {
        let inner = self
            .inner
            .scale_and_shift(scale, shift)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns `true` if `|A_ij - A_ji| <= tol` for all entries.
    #[wasm_bindgen(js_name = isSymmetric)]
    pub fn is_symmetric(&self, tol: Option<f64>) -> bool {
        self.inner.is_symmetric(tol.unwrap_or(1e-10))
    }

    /// Converts to a flat row-major `(numNodes, numNodes)` `Float64Array`.
    #[wasm_bindgen(js_name = toDense)]
    pub fn to_dense(&self) -> Result<Vec<f64>, JsError> {
        self.inner.to_dense().map_err(core_err_to_js)
    }
}

/// Output of `anisotropicLaplacian(graph, ...)`.
#[wasm_bindgen(js_name = AnisotropicLaplacianResult)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsAnisotropicLaplacianResult {
    matrix: CsrMatrix,
    nullspace: Vec<f64>,
    outer_scale: Vec<f64>,
}

#[wasm_bindgen(js_class = AnisotropicLaplacianResult)]
impl JsAnisotropicLaplacianResult {
    /// Returns the symmetric PSD `CsrMatrix` $S (D^{(\alpha)} - A^{(\alpha)}) S$.
    pub fn matrix(&self) -> JsCsrMatrix {
        JsCsrMatrix {
            inner: self.matrix.clone(),
        }
    }

    /// Returns the near-nullspace vector (`Float64Array`).
    pub fn nullspace(&self) -> Vec<f64> {
        self.nullspace.clone()
    }

    /// Returns the outer symmetric scaling factors (`Float64Array`).
    #[wasm_bindgen(js_name = outerScale)]
    pub fn outer_scale(&self) -> Vec<f64> {
        self.outer_scale.clone()
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

    pub(crate) fn from_rhgh_parts(inner: DogVaradhanLandmarks, rhgh: &RhghLandmarkStats) -> Self {
        let stats = DogVaradhanStats {
            moment_stats: SpectralMomentStats {
                sigma_lambda_sq: 0.0,
                sigma_lambda: 0.0,
                deg_cv: rhgh.deg_cv,
                mean_degree: rhgh.mean_degree,
                leaf_fraction: rhgh.leaf_fraction,
            },
            t1: rhgh.gamma,
            t2: rhgh.gamma,
            alpha: rhgh.p_exp,
            degree: 0,
            delta_psd: 0.0,
            h_min: 0.0,
            k_min: 0.0,
            k_med: 0.0,
            beta_auto: 0.0,
            beta_eff: 0.0,
            n_landmarks: rhgh.n_landmarks,
            t_coarse: 0.0,
            used_coarse_scaffold: false,
        };
        Self { inner, stats }
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

    /// Returns the Chebyshev polynomial degree used for the DoG filter.
    #[wasm_bindgen(getter)]
    pub fn degree(&self) -> usize {
        self.stats.degree
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

    /// Returns the number of stored 1-hop directed neighbor entries.
    #[wasm_bindgen(getter, js_name = n1hop)]
    pub fn n_1hop(&self) -> usize {
        self.stats.n_1hop
    }

    /// Returns the number of stored 2-hop directed neighbor entries.
    #[wasm_bindgen(getter, js_name = n2hop)]
    pub fn n_2hop(&self) -> usize {
        self.stats.n_2hop
    }

    /// Returns the number of vertices whose 2-hop expansion was truncated by `max2hopPerNode`.
    #[wasm_bindgen(getter, js_name = nTruncated2hop)]
    pub fn n_truncated_2hop(&self) -> usize {
        self.stats.n_truncated_2hop
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

/// 5b. Constructs a symmetric PSD Coifman-Lafon anisotropic graph Laplacian `L_aniso`.
#[wasm_bindgen(js_name = anisotropicLaplacian)]
pub fn js_anisotropic_laplacian(
    graph: &JsCsrGraph,
    alpha: Option<f64>,
    beta: Option<f64>,
    degree_reg: Option<f64>,
    degree_cap: Option<f64>,
    reweighted_degree_norm: Option<bool>,
) -> Result<JsAnisotropicLaplacianResult, JsError> {
    let cfg = AnisotropicLaplacianConfig {
        alpha: alpha.unwrap_or(0.5),
        beta: beta.unwrap_or(0.0),
        degree_reg: degree_reg.unwrap_or(0.0),
        degree_cap,
        reweighted_degree_norm: reweighted_degree_norm.unwrap_or(true),
    };
    let res = anisotropic_laplacian(graph.inner(), cfg).map_err(core_err_to_js)?;
    Ok(JsAnisotropicLaplacianResult {
        matrix: res.matrix,
        nullspace: res.nullspace,
        outer_scale: res.outer_scale,
    })
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
        None => adaptive_chebyshev_degree_scaled(t, lmax),
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

/// 6b. Applies Chebyshev heat diffusion $\exp(-t A)$ to a row-major `(numNodes, numCols)` buffer
/// for an arbitrary symmetric PSD `CsrMatrix` with dynamic `[0, lambdaMax]` scaling.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = chebyshevHeatSpmvMatrix)]
pub fn js_chebyshev_heat_spmv_matrix(
    matrix: &JsCsrMatrix,
    vectors: &[f64],
    num_cols: f64,
    t: f64,
    degree: Option<f64>,
    lambda_max: Option<f64>,
    lanczos_steps: Option<f64>,
    safety_factor: Option<f64>,
    seed: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    let ncols = parse_non_negative_usize(num_cols, "numCols")?;
    let deg = parse_opt_usize_param(degree, "degree")?;
    let steps = parse_optional_usize(
        lanczos_steps,
        DEFAULT_SPECTRUM_LANCZOS_STEPS,
        "lanczosSteps",
    )?;
    let sf = safety_factor.unwrap_or(DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR);
    let seed_u64 = parse_seed_u64(seed)?;
    if vectors.iter().any(|v| !v.is_finite()) {
        return Err(JsError::new("vectors must contain only finite values"));
    }
    let n = matrix.inner().num_nodes();
    let total = n
        .checked_mul(ncols)
        .ok_or_else(|| JsError::new("chebyshevHeatSpmvMatrix size overflow"))?;
    let mut out = vec![0.0_f64; total];
    let mut ws = ChebyshevWorkspace::with_capacity(total);
    ws.apply_heat_dynamic(
        matrix.inner(),
        vectors,
        &mut out,
        ncols,
        t,
        deg,
        lambda_max,
        steps,
        sf,
        seed_u64,
    )
    .map_err(core_err_to_js)?;
    Ok(out)
}

/// 7b. Computes a deterministic R1/R2 regularized DoG JL spectral embedding `(numNodes, jlDim)`
/// on an arbitrary symmetric PSD `CsrMatrix` with dynamic `[0, lambdaMax]` scaling.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = chebyshevDogJlEmbeddingMatrix)]
pub fn js_chebyshev_dog_jl_embedding_matrix(
    matrix: &JsCsrMatrix,
    jl_dim: Option<f64>,
    degree: Option<f64>,
    seed: Option<f64>,
    eps_psd: Option<f64>,
    t1: Option<f64>,
    t2: Option<f64>,
    alpha: Option<f64>,
    lambda_max: Option<f64>,
    lanczos_steps: Option<f64>,
    safety_factor: Option<f64>,
    scale_vec: Option<Vec<f64>>,
) -> Result<Vec<f64>, JsError> {
    let jl_dim_val = parse_optional_usize(jl_dim, DEFAULT_JL_DIM, "jlDim")?;
    let deg = parse_opt_usize_param(degree, "degree")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let eps_psd_val = eps_psd.unwrap_or(DEFAULT_EPS_PSD);
    let steps = parse_optional_usize(
        lanczos_steps,
        DEFAULT_SPECTRUM_LANCZOS_STEPS,
        "lanczosSteps",
    )?;
    let sf = safety_factor.unwrap_or(DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR);

    let mat = matrix.inner();
    let n = mat.num_nodes();
    let s = if n > 0 {
        let n_f = n as f64;
        let mu1 = mat.trace() / n_f;
        let frob_sq: f64 = mat.values().iter().map(|&v| v * v).sum();
        (frob_sq / n_f - mu1 * mu1).max(1e-6).sqrt()
    } else {
        1e-3
    };
    let t1_eff = t1.unwrap_or(1.0 / (1.0 + s));
    let t2_eff = t2.unwrap_or(4.0 / (1.0 + s));
    let alpha_eff = alpha.unwrap_or((1.0 - s).max(0.0));
    let base_terms = build_dog_terms(t1_eff, t2_eff, alpha_eff, 0.0).map_err(core_err_to_js)?;
    let mut ws = ChebyshevWorkspace::with_capacity(n.saturating_mul(jl_dim_val));
    let (out, _) = ws
        .compute_dog_jl_dynamic(
            mat,
            &base_terms,
            lambda_max,
            deg,
            jl_dim_val,
            eps_psd_val,
            steps,
            sf,
            seed_u64,
            scale_vec.as_deref(),
        )
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

/// 8b. Extracts canonical undirected 1-hop and hub-safe capped 2-hop vertex pair arrays `Sparse12HopPairs`.
#[wasm_bindgen(js_name = extractSparse1hop2hopPairs)]
pub fn js_extract_sparse_1hop_2hop_pairs(
    graph: &JsCsrGraph,
    max_2hop_per_node: Option<f64>,
    seed: Option<f64>,
) -> Result<JsSparse12HopPairs, JsError> {
    let cap = parse_optional_usize(max_2hop_per_node, DEFAULT_SPARSE_MAX_2HOP, "max2hopPerNode")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let inner =
        extract_sparse_1hop_2hop_pairs(graph.inner(), cap, seed_u64).map_err(core_err_to_js)?;
    Ok(JsSparse12HopPairs { inner })
}

/// 9. Builds a pure 1+2-hop CSR `KnnGraph` with `{1.0, 2.0}` distances and degree statistics.
#[wasm_bindgen(js_name = buildPure2hopKnn)]
pub fn js_build_pure_2hop_knn(graph: &JsCsrGraph) -> Result<JsTwoHopKnnResult, JsError> {
    let (knn, k2_deg, stats) = build_pure_2hop_knn(graph.inner()).map_err(core_err_to_js)?;
    Ok(JsTwoHopKnnResult { knn, k2_deg, stats })
}

/// 9b. Builds a hub-safe capped 1+2-hop CSR `KnnGraph` with `{1.0, 2.0}` distances and degree statistics.
#[wasm_bindgen(js_name = buildCapped12hopKnn)]
pub fn js_build_capped_12hop_knn(
    graph: &JsCsrGraph,
    max_2hop_per_node: Option<f64>,
    seed: Option<f64>,
) -> Result<JsTwoHopKnnResult, JsError> {
    let cap = parse_optional_usize(max_2hop_per_node, DEFAULT_SPARSE_MAX_2HOP, "max2hopPerNode")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let (knn, k2_deg, stats) =
        build_capped_12hop_knn(graph.inner(), cap, seed_u64).map_err(core_err_to_js)?;
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

/// 19b. Computes regularized DoG Varadhan landmark tables from a custom symmetric PSD `CsrMatrix`
/// with dynamic `[0, lambdaMax]` spectrum scaling.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = computeRegularizedDogVaradhanLandmarksFromMatrix)]
pub fn js_compute_regularized_dog_varadhan_landmarks_from_matrix(
    graph: &JsCsrGraph,
    matrix: &JsCsrMatrix,
    lambda_max: Option<f64>,
    degree: Option<f64>,
    num_landmarks: Option<f64>,
    min_degree: Option<f64>,
    jl_dim: Option<f64>,
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
    let min_deg = parse_optional_usize(min_degree, DEFAULT_MIN_DEGREE, "minDegree")?;
    let jl_d = parse_optional_usize(jl_dim, DEFAULT_JL_DIM, "jlDim")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let cfg = DogVaradhanConfig {
        t1,
        t2,
        alpha,
        beta_floor,
        degree: deg,
        n_landmarks,
        min_degree: min_deg,
        jl_dim: jl_d,
        degree_normalize: degree_normalize.unwrap_or(true),
        p_exp: p_exp.unwrap_or(DEFAULT_P_EXP),
        eps_psd: eps_psd.unwrap_or(DEFAULT_EPS_PSD),
        t_coarse: t_coarse.unwrap_or(DEFAULT_T_COARSE),
        seed: seed_u64,
        ..DogVaradhanConfig::default()
    };
    let (inner, stats) = compute_regularized_dog_varadhan_landmarks_from_matrix(
        graph.inner(),
        matrix.inner(),
        lambda_max,
        &cfg,
    )
    .map_err(core_err_to_js)?;
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

/// 22b. Median-centers and Gaussian-calibrated radial quantile-scales `layout` in place so core spread matches `targetStd` (default `1.0`).
#[wasm_bindgen(js_name = robustNormalizeLayout2d)]
pub fn js_robust_normalize_layout_2d(
    layout: &mut JsLayout2d,
    target_std: Option<f64>,
    quantile: Option<f64>,
    seed: Option<f64>,
) -> Result<(), JsError> {
    let t_std = target_std.unwrap_or(1.0);
    let q_val = quantile.unwrap_or(0.95);
    let seed_u64 = parse_seed_u64(seed)?;
    let mut soa = layout.inner().to_layout2d();
    robust_normalize_layout_2d(&mut soa, t_std, q_val, seed_u64).map_err(core_err_to_js)?;
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

/// 27b. Computes row-normalized symmetrized 1+2-hop probabilities with optional capping and `(d_i d_j)^(-degreeBeta)` decay.
#[wasm_bindgen(js_name = graph12hopRowNormalizedProbabilities)]
pub fn js_graph_12hop_row_normalized_probabilities(
    graph: &JsCsrGraph,
    w2: Option<f64>,
    max_2hop_per_node: Option<f64>,
    degree_beta: Option<f64>,
    seed: Option<f64>,
) -> Result<JsSparseProbabilities, JsError> {
    let w2_val = w2.unwrap_or(0.5);
    let cap_opt = parse_opt_usize_param(max_2hop_per_node, "max2hopPerNode")?;
    let beta_val = degree_beta.unwrap_or(0.0);
    let seed_u64 = parse_seed_u64(seed)?;
    let inner = graph_12hop_row_normalized_probabilities_with_degree_decay(
        graph.inner(),
        w2_val,
        cap_opt,
        beta_val,
        seed_u64,
    )
    .map_err(core_err_to_js)?;
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

fn parse_amg_pcg_config_js(options: &JsValue) -> Result<(AmgPcgConfig, Option<Vec<f64>>), JsError> {
    let mut cfg = AmgPcgConfig::default();
    let mut near_nullspace: Option<Vec<f64>> = None;
    if !options.is_undefined() && !options.is_null() {
        if !is_plain_js_object(options) {
            return Err(JsError::new(
                "AmgPcgSolver: options must be a plain object, null, or undefined",
            ));
        }
        cfg.rel_tol = opt_f64_prop(options, "relTol", cfg.rel_tol)?;
        cfg.abs_tol = opt_f64_prop(options, "absTol", cfg.abs_tol)?;
        cfg.max_iterations = opt_usize_prop(options, "maxIterations", cfg.max_iterations)?;
        cfg.theta = opt_f64_prop(options, "theta", cfg.theta)?;
        cfg.max_levels = opt_usize_prop(options, "maxLevels", cfg.max_levels)?;
        cfg.max_coarse_size = opt_usize_prop(options, "maxCoarseSize", cfg.max_coarse_size)?;
        cfg.pre_sweeps = opt_usize_prop(options, "preSweeps", cfg.pre_sweeps)?;
        cfg.post_sweeps = opt_usize_prop(options, "postSweeps", cfg.post_sweeps)?;
        cfg.max_agg_size = opt_usize_prop(options, "maxAggSize", cfg.max_agg_size)?;
        cfg.diag_eps = opt_f64_prop(options, "diagEps", cfg.diag_eps)?;
        cfg.eig_eps = opt_f64_prop(options, "eigEps", cfg.eig_eps)?;
        cfg.sym_tol = opt_f64_prop(options, "symTol", cfg.sym_tol)?;
        cfg.seed = opt_seed_prop(options)?;
        cfg.randomize_ties = opt_bool_prop(options, "randomizeTies", cfg.randomize_ties)?;

        let ns_val = get_prop(options, "nearNullspace")?;
        if !ns_val.is_undefined() && !ns_val.is_null() {
            if let Some(arr) = ns_val.dyn_ref::<js_sys::Float64Array>() {
                near_nullspace = Some(arr.to_vec());
            } else if let Some(arr) = ns_val.dyn_ref::<js_sys::Float32Array>() {
                near_nullspace = Some(arr.to_vec().into_iter().map(f64::from).collect());
            } else if !ns_val.is_string() && ns_val.is_object() {
                let iter = js_sys::try_iter(&ns_val)
                    .map_err(|err| {
                        JsError::new(&format!("failed to iterate nearNullspace: {err:?}"))
                    })?
                    .ok_or_else(|| {
                        JsError::new("options.nearNullspace must be a Float64Array or number array")
                    })?;
                let mut vec = Vec::new();
                for item_res in iter {
                    let item = item_res.map_err(|err| {
                        JsError::new(&format!("failed to read nearNullspace: {err:?}"))
                    })?;
                    let num = item.as_f64().ok_or_else(|| {
                        JsError::new("options.nearNullspace entries must be numbers")
                    })?;
                    vec.push(num);
                }
                near_nullspace = Some(vec);
            } else {
                return Err(JsError::new(
                    "options.nearNullspace must be a Float64Array or number array",
                ));
            }
        }
    }
    cfg.validate().map_err(core_err_to_js)?;
    Ok((cfg, near_nullspace))
}

fn parse_amg_laplacian_kind_js(raw: &str) -> Result<LaplacianKind, JsError> {
    match normalize_enum_token(raw).as_str() {
        "combinatorial" | "unnormalized" | "standard" => Ok(LaplacianKind::Combinatorial),
        "symmetricnormalized" | "symnorm" | "normalized" | "sym" => {
            Ok(LaplacianKind::SymmetricNormalized)
        }
        "randomwalk" | "rw" => Ok(LaplacianKind::RandomWalk),
        _ => Err(JsError::new(&format!(
            "unknown Laplacian kind {raw:?}; expected 'combinatorial' or 'symmetric_normalized'"
        ))),
    }
}

/// Single-RHS solution and diagnostics returned by `AmgPcgSolver.solve()`.
#[wasm_bindgen(js_name = AmgPcgSolveResult)]
#[derive(Debug, Clone)]
pub struct JsAmgPcgSolveResult {
    x_vec: Vec<f64>,
    res: AmgPcgSolveResult,
}

#[wasm_bindgen(js_class = AmgPcgSolveResult)]
impl JsAmgPcgSolveResult {
    /// Returns a copy of the solution vector (`Float64Array`).
    pub fn x(&self) -> Vec<f64> {
        self.x_vec.clone()
    }

    /// Number of PCG iterations executed.
    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> usize {
        self.res.iterations
    }

    /// True Euclidean residual norm `||b - A x||_2`.
    #[wasm_bindgen(getter, js_name = residualNorm)]
    pub fn residual_norm(&self) -> f64 {
        self.res.residual_norm
    }

    /// Relative residual norm `||b - A x||_2 / ||b||_2`.
    #[wasm_bindgen(getter, js_name = relativeResidual)]
    pub fn relative_residual(&self) -> f64 {
        self.res.relative_residual
    }

    /// `true` if the stopping tolerance was met.
    #[wasm_bindgen(getter)]
    pub fn converged(&self) -> bool {
        self.res.converged
    }

    /// Final solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.res.status).to_owned()
    }
}

/// Batched-RHS solution and diagnostics returned by `AmgPcgSolver.solveBatch()`.
#[wasm_bindgen(js_name = AmgPcgBatchResult)]
#[derive(Debug, Clone)]
pub struct JsAmgPcgBatchResult {
    num_nodes: usize,
    ncols: usize,
    x_flat: Vec<f64>,
    res: AmgPcgBatchResult,
}

#[wasm_bindgen(js_class = AmgPcgBatchResult)]
impl JsAmgPcgBatchResult {
    /// Returns a flat row-major `(numNodes, ncols)` `Float64Array` copy of all solution columns.
    pub fn x(&self) -> Vec<f64> {
        self.x_flat.clone()
    }

    /// Number of rows (`numNodes`).
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Number of RHS columns (`ncols`).
    #[wasm_bindgen(getter)]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Sum of PCG iterations across all columns.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.res.total_iterations
    }

    /// Maximum PCG iterations executed across all columns.
    #[wasm_bindgen(getter, js_name = maxIterationsUsed)]
    pub fn max_iterations_used(&self) -> usize {
        self.res.max_iterations_used
    }

    /// Minimum PCG iterations executed across all columns.
    #[wasm_bindgen(getter, js_name = minIterationsUsed)]
    pub fn min_iterations_used(&self) -> usize {
        self.res.min_iterations_used
    }

    /// Maximum true residual norm across all columns.
    #[wasm_bindgen(getter, js_name = maxResidualNorm)]
    pub fn max_residual_norm(&self) -> f64 {
        self.res.max_residual_norm
    }

    /// Maximum relative residual norm across all columns.
    #[wasm_bindgen(getter, js_name = maxRelativeResidual)]
    pub fn max_relative_residual(&self) -> f64 {
        self.res.max_relative_residual
    }

    /// `true` if every column converged within tolerance.
    #[wasm_bindgen(getter, js_name = allConverged)]
    pub fn all_converged(&self) -> bool {
        self.res.all_converged
    }

    /// Returns per-column iteration counts (`Uint32Array`).
    #[wasm_bindgen(js_name = iterationsPerCol)]
    pub fn iterations_per_col(&self) -> Vec<u32> {
        self.res
            .iterations_per_col
            .iter()
            .map(|&v| v as u32)
            .collect()
    }

    /// Returns per-column true residual norms (`Float64Array`).
    #[wasm_bindgen(js_name = residualNorms)]
    pub fn residual_norms(&self) -> Vec<f64> {
        self.res.residual_norms.clone()
    }

    /// Returns per-column relative residual norms (`Float64Array`).
    #[wasm_bindgen(js_name = relativeResiduals)]
    pub fn relative_residuals(&self) -> Vec<f64> {
        self.res.relative_residuals.clone()
    }

    /// Returns per-column convergence flags (`Uint8Array` with `1` for converged, `0` otherwise).
    #[wasm_bindgen(js_name = convergedPerCol)]
    pub fn converged_per_col(&self) -> Vec<u8> {
        self.res
            .converged_per_col
            .iter()
            .map(|&v| u8::from(v))
            .collect()
    }
}

/// Hardened single-threaded Algebraic Multigrid Preconditioned Conjugate Gradient (`AmgPcgSolver`)
/// supporting symmetric CSR matrices, regularized/shifted graph Laplacians, batched RHS, and
/// cooperative `step()` execution.
#[wasm_bindgen(js_name = AmgPcgSolver)]
#[derive(Debug, Clone)]
pub struct JsAmgPcgSolver {
    inner: AmgPcgSolver,
    ws: AmgPcgWorkspace,
}

#[wasm_bindgen(js_class = AmgPcgSolver)]
impl JsAmgPcgSolver {
    /// Constructs an `AmgPcgSolver` directly from raw symmetric CSR buffers.
    #[wasm_bindgen(js_name = fromCsr)]
    pub fn from_csr(
        num_nodes: f64,
        indptr: &[u32],
        indices: &[u32],
        values: &[f64],
        options: &JsValue,
    ) -> Result<JsAmgPcgSolver, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let (cfg, ns) = parse_amg_pcg_config_js(options)?;
        let mat = CsrMatrix::from_csr(n, indptr.to_vec(), indices.to_vec(), values.to_vec())
            .map_err(core_err_to_js)?;
        let inner =
            AmgPcgSolver::with_near_nullspace(&mat, ns.as_deref(), cfg).map_err(core_err_to_js)?;
        let ws = inner.new_workspace();
        Ok(Self { inner, ws })
    }

    /// Constructs an `AmgPcgSolver` directly from an existing `CsrMatrix`.
    #[wasm_bindgen(js_name = fromCsrMatrix)]
    pub fn from_csr_matrix(
        matrix: &JsCsrMatrix,
        options: &JsValue,
    ) -> Result<JsAmgPcgSolver, JsError> {
        let (cfg, ns) = parse_amg_pcg_config_js(options)?;
        let inner = AmgPcgSolver::with_near_nullspace(matrix.inner(), ns.as_deref(), cfg)
            .map_err(core_err_to_js)?;
        let ws = inner.new_workspace();
        Ok(Self { inner, ws })
    }

    /// Constructs an `AmgPcgSolver` for `I + gamma * L_sym` with exact `D^{1/2} 1` near-nullspace.
    #[wasm_bindgen(js_name = fromGraphRegularizedLaplacian)]
    pub fn from_graph_regularized_laplacian(
        graph: &JsCsrGraph,
        gamma: Option<f64>,
        options: &JsValue,
    ) -> Result<JsAmgPcgSolver, JsError> {
        let (cfg, _) = parse_amg_pcg_config_js(options)?;
        let inner =
            AmgPcgSolver::from_regularized_sym_laplacian(graph.inner(), gamma.unwrap_or(1.0), cfg)
                .map_err(core_err_to_js)?;
        let ws = inner.new_workspace();
        Ok(Self { inner, ws })
    }

    /// Constructs an `AmgPcgSolver` for a shifted graph Laplacian (`"combinatorial"` or `"symmetric_normalized"`).
    #[wasm_bindgen(js_name = fromGraphLaplacian)]
    pub fn from_graph_laplacian(
        graph: &JsCsrGraph,
        kind: Option<String>,
        shift: Option<f64>,
        options: &JsValue,
    ) -> Result<JsAmgPcgSolver, JsError> {
        let lap_kind = match kind {
            Some(ref k) => parse_amg_laplacian_kind_js(k)?,
            None => LaplacianKind::Combinatorial,
        };
        let (cfg, _) = parse_amg_pcg_config_js(options)?;
        let inner =
            AmgPcgSolver::from_laplacian(graph.inner(), lap_kind, shift.unwrap_or(0.0), cfg)
                .map_err(core_err_to_js)?;
        let ws = inner.new_workspace();
        Ok(Self { inner, ws })
    }

    /// Solves `A x = b` synchronously with optional initial guess `x0`.
    pub fn solve(
        &mut self,
        b: &[f64],
        x0: Option<Vec<f64>>,
    ) -> Result<JsAmgPcgSolveResult, JsError> {
        self.inner
            .init_solve(b, x0.as_deref(), &mut self.ws)
            .map_err(core_err_to_js)?;
        self.inner.run(&mut self.ws).map_err(core_err_to_js)?;
        Ok(JsAmgPcgSolveResult {
            x_vec: self.ws.x().to_vec(),
            res: self.ws.result(),
        })
    }

    /// Solves `A X = B` for flat row-major `(numNodes, ncols)` RHS matrix `bFlat`.
    #[wasm_bindgen(js_name = solveBatch)]
    pub fn solve_batch(
        &mut self,
        b_flat: &[f64],
        ncols: f64,
        x0_flat: Option<Vec<f64>>,
    ) -> Result<JsAmgPcgBatchResult, JsError> {
        let nc = parse_non_negative_usize(ncols, "ncols")?;
        let n = self.inner.num_nodes();
        let expected = n
            .checked_mul(nc)
            .ok_or_else(|| JsError::new("AmgPcgSolver.solveBatch size overflow"))?;
        let mut x_out = match x0_flat {
            Some(v) => {
                if v.len() != expected {
                    return Err(core_err_to_js(CoreError::dimension_mismatch(
                        expected,
                        v.len(),
                        "AmgPcgSolver.solveBatch::x0Flat",
                    )));
                }
                v
            }
            None => vec![0.0_f64; expected],
        };
        let res = self
            .inner
            .solve_batch(b_flat, &mut x_out, nc, &mut self.ws)
            .map_err(core_err_to_js)?;
        Ok(JsAmgPcgBatchResult {
            num_nodes: n,
            ncols: nc,
            x_flat: x_out,
            res,
        })
    }

    /// Initializes a cooperative PCG solve for `b` and optional initial guess `x0`.
    #[wasm_bindgen(js_name = startSolve)]
    pub fn start_solve(&mut self, b: &[f64], x0: Option<Vec<f64>>) -> Result<String, JsError> {
        let status = self
            .inner
            .init_solve(b, x0.as_deref(), &mut self.ws)
            .map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Executes a single cooperative PCG iteration and returns the updated status string.
    pub fn step(&mut self) -> Result<String, JsError> {
        let status = self.inner.step(&mut self.ws).map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Executes up to `steps` cooperative PCG iterations and returns the updated status string.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, steps: f64) -> Result<String, JsError> {
        let n_steps = parse_non_negative_usize(steps, "steps")?;
        let status = self
            .inner
            .run_steps(&mut self.ws, n_steps)
            .map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Runs the active cooperative solve to completion and returns the final status string.
    pub fn run(&mut self) -> Result<String, JsError> {
        let status = self.inner.run(&mut self.ws).map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Returns a copy of the current solution vector (`Float64Array`).
    pub fn solution(&self) -> Vec<f64> {
        self.ws.x().to_vec()
    }

    /// Returns the matrix dimension `numNodes`.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the number of AMG hierarchy levels.
    #[wasm_bindgen(getter, js_name = numLevels)]
    pub fn num_levels(&self) -> usize {
        self.inner.num_levels()
    }

    /// Returns the coarsest-level matrix dimension.
    #[wasm_bindgen(getter, js_name = coarsestSize)]
    pub fn coarsest_size(&self) -> usize {
        self.inner.coarsest_size()
    }

    /// Returns the AMG operator complexity.
    #[wasm_bindgen(getter, js_name = operatorComplexity)]
    pub fn operator_complexity(&self) -> f64 {
        self.inner.operator_complexity()
    }

    /// Returns the maximum CSR nonzero count across all AMG hierarchy levels.
    #[wasm_bindgen(getter, js_name = maxLevelNnz)]
    pub fn max_level_nnz(&self) -> usize {
        self.inner.max_level_nnz()
    }

    /// Returns the current solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.ws.status()).to_owned()
    }

    /// Returns the number of PCG iterations executed in the active solve.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.ws.iteration()
    }

    /// Returns the maximum scheduled iterations for the active solve.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.ws.total_iterations()
    }

    /// Returns `true` when the active solve has converged or completed.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.ws.is_finished()
    }

    /// Returns the most recent Euclidean residual norm.
    #[wasm_bindgen(getter, js_name = residualNorm)]
    pub fn residual_norm(&self) -> f64 {
        self.ws.residual_norm()
    }

    /// Returns the most recent relative Euclidean residual norm.
    #[wasm_bindgen(getter, js_name = relativeResidual)]
    pub fn relative_residual(&self) -> f64 {
        self.ws.relative_residual()
    }

    /// Returns `true` if the active solve converged within tolerance.
    #[wasm_bindgen(getter)]
    pub fn converged(&self) -> bool {
        self.ws.converged()
    }
}

fn opt_maybe_bool_prop(obj: &JsValue, key: &str) -> Result<Option<bool>, JsError> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    val.as_bool()
        .map(Some)
        .ok_or_else(|| JsError::new(&format!("options.{key} must be a boolean")))
}

fn parse_effective_resistance_options(
    options: &JsValue,
) -> Result<EffectiveResistanceConfig, JsError> {
    let mut cfg = EffectiveResistanceConfig::default();
    if !options.is_undefined() && !options.is_null() {
        if !is_plain_js_object(options) {
            return Err(JsError::new(
                "computeEffectiveResistances: options must be a plain object, null, or undefined",
            ));
        }
        cfg.jl_dim = opt_usize_prop(options, "jlDim", cfg.jl_dim)?;
        cfg.seed = opt_seed_prop(options)?;
        cfg.weights_are_lengths =
            opt_bool_prop(options, "weightsAreLengths", cfg.weights_are_lengths)?;
        cfg.exact = opt_maybe_bool_prop(options, "exact")?;
        cfg.exact_max_nodes = opt_usize_prop(options, "exactMaxNodes", cfg.exact_max_nodes)?;
        cfg.eig_eps = opt_f64_prop(options, "eigEps", cfg.eig_eps)?;
        cfg.amg_config.rel_tol = opt_f64_prop(options, "relTol", cfg.amg_config.rel_tol)?;
        cfg.amg_config.max_iterations =
            opt_usize_prop(options, "maxIterations", cfg.amg_config.max_iterations)?;
        cfg.amg_config.eig_eps = cfg.eig_eps;
        cfg.amg_config.seed = cfg.seed;
    }
    cfg.validate().map_err(core_err_to_js)?;
    Ok(cfg)
}

fn parse_resistance_curvature_options(
    options: &JsValue,
) -> Result<ResistanceCurvatureConfig, JsError> {
    let mut cfg = ResistanceCurvatureConfig::default();
    if !options.is_undefined() && !options.is_null() {
        if !is_plain_js_object(options) {
            return Err(JsError::new(
                "resistanceCurvatureFlow: options must be a plain object, null, or undefined",
            ));
        }
        cfg.steps = opt_usize_prop(options, "steps", cfg.steps)?;
        cfg.eta = opt_f64_prop(options, "eta", cfg.eta)?;
        cfg.kappa_max = opt_f64_prop(options, "kappaMax", cfg.kappa_max)?;
        cfg.min_edge_length = opt_f64_prop(options, "minEdgeLength", cfg.min_edge_length)?;
        cfg.hyperbolic_threshold =
            opt_f64_prop(options, "hyperbolicThreshold", cfg.hyperbolic_threshold)?;
        cfg.spherical_threshold =
            opt_f64_prop(options, "sphericalThreshold", cfg.spherical_threshold)?;
        cfg.torus_std_threshold =
            opt_f64_prop(options, "torusStdThreshold", cfg.torus_std_threshold)?;
        cfg.resistance = parse_effective_resistance_options(options)?;
    }
    cfg.validate().map_err(core_err_to_js)?;
    Ok(cfg)
}

/// Effective edge resistances and Foster's theorem diagnostics for canonical simple undirected edges.
#[wasm_bindgen(js_name = EffectiveResistanceResult)]
#[derive(Debug, Clone)]
pub struct JsEffectiveResistanceResult {
    inner: EffectiveResistanceResult,
}

#[wasm_bindgen(js_class = EffectiveResistanceResult)]
impl JsEffectiveResistanceResult {
    /// Returns the number of vertices in the graph.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes
    }

    /// Returns the number of canonical undirected simple edges (`u < v`).
    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges
    }

    /// Returns the number of connected components (including isolated vertices).
    #[wasm_bindgen(getter, js_name = numComponents)]
    pub fn num_components(&self) -> usize {
        self.inner.num_components
    }

    /// Returns the observed Foster sum `sum_e w_e * R_e`.
    #[wasm_bindgen(getter, js_name = fosterSum)]
    pub fn foster_sum(&self) -> f64 {
        self.inner.foster_sum
    }

    /// Returns the exact theoretical Foster sum `numNodes - numComponents`.
    #[wasm_bindgen(getter, js_name = expectedFoster)]
    pub fn expected_foster(&self) -> f64 {
        self.inner.expected_foster
    }

    /// Returns the relative Foster error.
    #[wasm_bindgen(getter, js_name = fosterRelErr)]
    pub fn foster_rel_err(&self) -> f64 {
        self.inner.foster_rel_err
    }

    /// Returns `true` if the exact dense pseudoinverse solver path was used.
    #[wasm_bindgen(getter, js_name = usedExact)]
    pub fn used_exact(&self) -> bool {
        self.inner.used_exact
    }

    /// Returns total AMG-PCG iterations executed across all JL RHS columns.
    #[wasm_bindgen(getter, js_name = amgTotalIterations)]
    pub fn amg_total_iterations(&self) -> usize {
        self.inner.amg_total_iterations
    }

    /// Returns maximum relative PCG residual across all JL RHS columns.
    #[wasm_bindgen(getter, js_name = amgMaxRelativeResidual)]
    pub fn amg_max_relative_residual(&self) -> f64 {
        self.inner.amg_max_relative_residual
    }

    /// Returns `true` if every JL PCG column converged within tolerance.
    #[wasm_bindgen(getter, js_name = amgAllConverged)]
    pub fn amg_all_converged(&self) -> bool {
        self.inner.amg_all_converged
    }

    /// Returns canonical edge source endpoints `u < v` (`Uint32Array`).
    #[wasm_bindgen(js_name = edgeU)]
    pub fn edge_u(&self) -> Vec<u32> {
        self.inner.edge_u.clone()
    }

    /// Returns canonical edge target endpoints `v > u` (`Uint32Array`).
    #[wasm_bindgen(js_name = edgeV)]
    pub fn edge_v(&self) -> Vec<u32> {
        self.inner.edge_v.clone()
    }

    /// Returns canonical edge lengths `l_e` (`Float64Array`).
    #[wasm_bindgen(js_name = edgeLengths)]
    pub fn edge_lengths(&self) -> Vec<f64> {
        self.inner.edge_lengths.clone()
    }

    /// Returns canonical edge conductances `w_e = 1 / l_e` (`Float64Array`).
    pub fn conductances(&self) -> Vec<f64> {
        self.inner.conductances.clone()
    }

    /// Returns effective resistances `R_e` (`Float64Array`).
    #[wasm_bindgen(js_name = effectiveResistances)]
    pub fn effective_resistances(&self) -> Vec<f64> {
        self.inner.effective_resistances.clone()
    }
}

/// Per-iteration summary recorded during `resistanceCurvatureFlow()`.
#[wasm_bindgen(js_name = ResistanceCurvatureStep)]
#[derive(Debug, Clone, Copy)]
pub struct JsResistanceCurvatureStep {
    inner: ResistanceCurvatureStep,
}

#[wasm_bindgen(js_class = ResistanceCurvatureStep)]
impl JsResistanceCurvatureStep {
    /// Flow step index (`0..steps`, plus final evaluation step `steps`).
    #[wasm_bindgen(getter)]
    pub fn step(&self) -> usize {
        self.inner.step
    }

    /// Observed Foster sum `sum_e w_e * R_e`.
    #[wasm_bindgen(getter, js_name = fosterSum)]
    pub fn foster_sum(&self) -> f64 {
        self.inner.foster_sum
    }

    /// Exact theoretical Foster sum `numNodes - numComponents`.
    #[wasm_bindgen(getter, js_name = expectedFoster)]
    pub fn expected_foster(&self) -> f64 {
        self.inner.expected_foster
    }

    /// Relative Foster error.
    #[wasm_bindgen(getter, js_name = fosterRelErr)]
    pub fn foster_rel_err(&self) -> f64 {
        self.inner.foster_rel_err
    }

    /// Mean link resistance curvature.
    #[wasm_bindgen(getter, js_name = meanLinkCurvature)]
    pub fn mean_link_curvature(&self) -> f64 {
        self.inner.mean_link_curvature
    }

    /// Population standard deviation of link resistance curvatures.
    #[wasm_bindgen(getter, js_name = stdLinkCurvature)]
    pub fn std_link_curvature(&self) -> f64 {
        self.inner.std_link_curvature
    }

    /// Minimum link resistance curvature.
    #[wasm_bindgen(getter, js_name = minLinkCurvature)]
    pub fn min_link_curvature(&self) -> f64 {
        self.inner.min_link_curvature
    }

    /// Maximum link resistance curvature.
    #[wasm_bindgen(getter, js_name = maxLinkCurvature)]
    pub fn max_link_curvature(&self) -> f64 {
        self.inner.max_link_curvature
    }

    /// Mean vertex resistance curvature.
    #[wasm_bindgen(getter, js_name = meanVertexCurvature)]
    pub fn mean_vertex_curvature(&self) -> f64 {
        self.inner.mean_vertex_curvature
    }

    /// Minimum vertex resistance curvature.
    #[wasm_bindgen(getter, js_name = minVertexCurvature)]
    pub fn min_vertex_curvature(&self) -> f64 {
        self.inner.min_vertex_curvature
    }

    /// Maximum vertex resistance curvature.
    #[wasm_bindgen(getter, js_name = maxVertexCurvature)]
    pub fn max_vertex_curvature(&self) -> f64 {
        self.inner.max_vertex_curvature
    }

    /// Whether the exact dense pseudoinverse solver was used.
    #[wasm_bindgen(getter, js_name = usedExact)]
    pub fn used_exact(&self) -> bool {
        self.inner.used_exact
    }

    /// Total AMG-PCG iterations executed in this step.
    #[wasm_bindgen(getter, js_name = amgTotalIterations)]
    pub fn amg_total_iterations(&self) -> usize {
        self.inner.amg_total_iterations
    }

    /// Maximum relative PCG residual in this step.
    #[wasm_bindgen(getter, js_name = amgMaxRelativeResidual)]
    pub fn amg_max_relative_residual(&self) -> f64 {
        self.inner.amg_max_relative_residual
    }
}

/// Devriendt-Lambiotte resistance curvature flow result with reweighted graphs and geometry classification.
#[wasm_bindgen(js_name = ResistanceCurvatureFlowResult)]
#[derive(Debug, Clone)]
pub struct JsResistanceCurvatureFlowResult {
    inner: ResistanceCurvatureFlowResult,
}

#[wasm_bindgen(js_class = ResistanceCurvatureFlowResult)]
impl JsResistanceCurvatureFlowResult {
    /// Returns the number of vertices in the graph.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes
    }

    /// Returns the number of canonical undirected simple edges (`u < v`).
    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges
    }

    /// Returns the number of connected components (including isolated vertices).
    #[wasm_bindgen(getter, js_name = numComponents)]
    pub fn num_components(&self) -> usize {
        self.inner.num_components
    }

    /// Returns the final Foster sum `sum_e w_e * R_e`.
    #[wasm_bindgen(getter, js_name = fosterSum)]
    pub fn foster_sum(&self) -> f64 {
        self.inner.foster_sum
    }

    /// Returns the exact theoretical Foster sum `numNodes - numComponents`.
    #[wasm_bindgen(getter, js_name = expectedFoster)]
    pub fn expected_foster(&self) -> f64 {
        self.inner.expected_foster
    }

    /// Returns the final relative Foster error.
    #[wasm_bindgen(getter, js_name = fosterRelErr)]
    pub fn foster_rel_err(&self) -> f64 {
        self.inner.foster_rel_err
    }

    /// Returns mean final link resistance curvature.
    #[wasm_bindgen(getter, js_name = meanLinkCurvature)]
    pub fn mean_link_curvature(&self) -> f64 {
        self.inner.mean_link_curvature
    }

    /// Returns population standard deviation of final link resistance curvatures.
    #[wasm_bindgen(getter, js_name = stdLinkCurvature)]
    pub fn std_link_curvature(&self) -> f64 {
        self.inner.std_link_curvature
    }

    /// Returns minimum final link resistance curvature.
    #[wasm_bindgen(getter, js_name = minLinkCurvature)]
    pub fn min_link_curvature(&self) -> f64 {
        self.inner.min_link_curvature
    }

    /// Returns maximum final link resistance curvature.
    #[wasm_bindgen(getter, js_name = maxLinkCurvature)]
    pub fn max_link_curvature(&self) -> f64 {
        self.inner.max_link_curvature
    }

    /// Returns mean final vertex resistance curvature.
    #[wasm_bindgen(getter, js_name = meanVertexCurvature)]
    pub fn mean_vertex_curvature(&self) -> f64 {
        self.inner.mean_vertex_curvature
    }

    /// Returns minimum final vertex resistance curvature.
    #[wasm_bindgen(getter, js_name = minVertexCurvature)]
    pub fn min_vertex_curvature(&self) -> f64 {
        self.inner.min_vertex_curvature
    }

    /// Returns maximum final vertex resistance curvature.
    #[wasm_bindgen(getter, js_name = maxVertexCurvature)]
    pub fn max_vertex_curvature(&self) -> f64 {
        self.inner.max_vertex_curvature
    }

    /// Returns `true` if the final evaluation used the exact dense pseudoinverse solver.
    #[wasm_bindgen(getter, js_name = usedExact)]
    pub fn used_exact(&self) -> bool {
        self.inner.used_exact
    }

    /// Returns total AMG-PCG iterations executed in the final evaluation.
    #[wasm_bindgen(getter, js_name = amgTotalIterations)]
    pub fn amg_total_iterations(&self) -> usize {
        self.inner.amg_total_iterations
    }

    /// Returns maximum relative PCG residual in the final evaluation.
    #[wasm_bindgen(getter, js_name = amgMaxRelativeResidual)]
    pub fn amg_max_relative_residual(&self) -> f64 {
        self.inner.amg_max_relative_residual
    }

    /// Returns `true` if every JL PCG column converged in the final evaluation.
    #[wasm_bindgen(getter, js_name = amgAllConverged)]
    pub fn amg_all_converged(&self) -> bool {
        self.inner.amg_all_converged
    }

    /// Returns the suggested 2D geometry (`"hyperbolic"`, `"spherical"`, `"torus"`, or `"euclidean"`).
    #[wasm_bindgen(getter, js_name = suggestedGeometry)]
    pub fn suggested_geometry(&self) -> String {
        self.inner.suggested_geometry.as_str().to_owned()
    }

    /// Returns the suggested `ManifoldKind` (`"hyperbolic"`, `"spherical"`, `"torus"`), or `undefined`.
    #[wasm_bindgen(getter, js_name = suggestedManifold)]
    pub fn suggested_manifold(&self) -> Option<String> {
        self.inner
            .suggested_manifold_kind()
            .map(|_| self.inner.suggested_geometry.as_str().to_owned())
    }

    /// Returns canonical edge source endpoints `u < v` (`Uint32Array`).
    #[wasm_bindgen(js_name = edgeU)]
    pub fn edge_u(&self) -> Vec<u32> {
        self.inner.edge_u.clone()
    }

    /// Returns canonical edge target endpoints `v > u` (`Uint32Array`).
    #[wasm_bindgen(js_name = edgeV)]
    pub fn edge_v(&self) -> Vec<u32> {
        self.inner.edge_v.clone()
    }

    /// Returns final normalized edge lengths `l_e` (`Float64Array`).
    #[wasm_bindgen(js_name = edgeLengths)]
    pub fn edge_lengths(&self) -> Vec<f64> {
        self.inner.edge_lengths.clone()
    }

    /// Returns final edge conductances `w_e = 1 / l_e` (`Float64Array`).
    pub fn conductances(&self) -> Vec<f64> {
        self.inner.conductances.clone()
    }

    /// Returns final effective resistances `R_e` (`Float64Array`).
    #[wasm_bindgen(js_name = effectiveResistances)]
    pub fn effective_resistances(&self) -> Vec<f64> {
        self.inner.effective_resistances.clone()
    }

    /// Returns final vertex resistance curvatures `p_i` (`Float64Array`).
    #[wasm_bindgen(js_name = vertexCurvatures)]
    pub fn vertex_curvatures(&self) -> Vec<f64> {
        self.inner.vertex_curvatures.clone()
    }

    /// Returns final link resistance curvatures `kappa_res(u, v)` (`Float64Array`).
    #[wasm_bindgen(js_name = linkCurvatures)]
    pub fn link_curvatures(&self) -> Vec<f64> {
        self.inner.link_curvatures.clone()
    }

    /// Returns per-step diagnostics across `0..=steps`.
    #[wasm_bindgen(js_name = stepHistory)]
    pub fn step_history(&self) -> Vec<JsResistanceCurvatureStep> {
        self.inner
            .step_history
            .iter()
            .copied()
            .map(|inner| JsResistanceCurvatureStep { inner })
            .collect()
    }

    /// Builds a weighted undirected `CsrGraph` (`f32`) storing final edge lengths `l_e`.
    #[wasm_bindgen(js_name = lengthGraph)]
    pub fn length_graph(&self) -> Result<JsCsrGraph, JsError> {
        self.inner
            .to_length_graph_f32()
            .map(JsCsrGraph::from_inner)
            .map_err(core_err_to_js)
    }

    /// Builds a weighted undirected `CsrGraph` (`f32`) storing final conductances `w_e = 1 / l_e`.
    #[wasm_bindgen(js_name = conductanceGraph)]
    pub fn conductance_graph(&self) -> Result<JsCsrGraph, JsError> {
        self.inner
            .to_conductance_graph_f32()
            .map(JsCsrGraph::from_inner)
            .map_err(core_err_to_js)
    }
}

/// Computes effective edge resistances and Foster's theorem diagnostics via exact pseudoinverse
/// (for small graphs) or Spielman-Srivastava JL projection solved with `AmgPcgSolver`.
#[wasm_bindgen(js_name = computeEffectiveResistances)]
pub fn js_compute_effective_resistances(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsEffectiveResistanceResult, JsError> {
    let cfg = parse_effective_resistance_options(options)?;
    let inner = compute_effective_resistances(graph.inner(), &cfg).map_err(core_err_to_js)?;
    Ok(JsEffectiveResistanceResult { inner })
}

/// Runs Devriendt-Lambiotte resistance curvature flow on `graph` and classifies target geometry.
#[wasm_bindgen(js_name = resistanceCurvatureFlow)]
pub fn js_resistance_curvature_flow(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsResistanceCurvatureFlowResult, JsError> {
    let cfg = parse_resistance_curvature_options(options)?;
    let inner = resistance_curvature_flow(graph.inner(), &cfg).map_err(core_err_to_js)?;
    Ok(JsResistanceCurvatureFlowResult { inner })
}
