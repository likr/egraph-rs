//! WebAssembly bindings for CSR sparse matrices, anisotropic Laplacians,
//! eigensolver results, and $O(|E|)$ spectral moment summary statistics.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, parse_non_negative_usize, parse_optional_usize, parse_seed_u64,
};
use egraph::linalg::{
    AnisotropicLaplacianConfig, CsrMatrix, DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR,
    DEFAULT_SPECTRUM_LANCZOS_STEPS, DEFAULT_SPECTRUM_POWER_ITERS, DEFAULT_SPECTRUM_SAFETY_FACTOR,
    EigenResult, SpectralMomentStats, anisotropic_laplacian, spectral_moment_stats,
};
use wasm_bindgen::prelude::*;

/// Double-precision square Compressed Sparse Row matrix (`CsrMatrix`) with spectral bounds
/// and matrix-vector operations.
#[wasm_bindgen(js_name = CsrMatrix)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsCsrMatrix {
    inner: CsrMatrix,
}

impl JsCsrMatrix {
    /// Wraps an existing `egraph_linalg::CsrMatrix` in a `JsCsrMatrix`.
    pub(crate) fn from_inner(inner: CsrMatrix) -> Self {
        Self { inner }
    }

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

/// Smallest nontrivial eigenpairs of the symmetric normalized Laplacian $L_{\mathrm{sym}}$.
#[wasm_bindgen(js_name = EigenResult)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsEigenResult {
    inner: EigenResult,
}

impl JsEigenResult {
    pub(crate) fn from_inner(inner: EigenResult) -> Self {
        Self { inner }
    }
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

/// Computes exact $O(|E|)$ spectral variance and degree summary statistics of $L_{\mathrm{sym}}$.
#[wasm_bindgen(js_name = spectralMomentStats)]
pub fn js_spectral_moment_stats(graph: &JsCsrGraph) -> JsSpectralMomentStats {
    JsSpectralMomentStats {
        inner: spectral_moment_stats(graph.inner()),
    }
}

/// Constructs a symmetric PSD Coifman-Lafon anisotropic graph Laplacian `L_aniso`.
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
