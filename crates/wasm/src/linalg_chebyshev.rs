//! WebAssembly bindings for $O(|E|)$ Chebyshev heat diffusion and DoG Johnson-Lindenstrauss embeddings.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, parse_non_negative_usize, parse_opt_usize_param, parse_optional_usize,
    parse_seed_u64,
};
use crate::linalg_matrix::JsCsrMatrix;
use egraph::linalg::{
    ChebyshevWorkspace, DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR, DEFAULT_SPECTRUM_LANCZOS_STEPS,
    adaptive_chebyshev_degree_scaled, build_dog_terms, compute_chebyshev_bandpass_coefficients,
    project_psd_spectral_filter, spectral_moment_stats, symmetric_normalized_laplacian,
};
use wasm_bindgen::prelude::*;

const DEFAULT_LAMBDA_MAX: f64 = 2.0;
const DEFAULT_JL_DIM: usize = 64;
const DEFAULT_DOG_DEGREE: usize = 25;
const DEFAULT_EPS_PSD: f64 = 1e-4;

/// Applies Chebyshev heat diffusion $\exp(-t L_{\mathrm{sym}})$ to a row-major `(numNodes, numCols)` buffer.
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

/// Applies Chebyshev heat diffusion $\exp(-t A)$ to a row-major `(numNodes, numCols)` buffer
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

/// Computes a deterministic R1/R2 regularized DoG JL spectral embedding `(numNodes, jlDim)`
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

/// Computes a deterministic R1/R2 regularized DoG JL spectral embedding `(numNodes, jlDim)`.
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
