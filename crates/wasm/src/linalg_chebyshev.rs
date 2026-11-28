//! WebAssembly bindings for $O(|E|)$ Chebyshev heat diffusion and DoG Johnson-Lindenstrauss embeddings.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    DEFAULT_SEED, core_err_to_js, invalid_input_err, non_finite_err, parse_non_negative_usize,
    parse_optional_f64_sequence,
};
use crate::js_options::{
    check_options_arg, get_prop, opt_bool_prop, opt_f64_prop, opt_maybe_f64_prop,
    opt_maybe_usize_prop, opt_seed_prop, opt_usize_prop,
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
    options: &JsValue,
) -> Result<Vec<f64>, JsValue> {
    let ncols = parse_non_negative_usize(num_cols, "numCols")?;
    let (lmax, deg_opt) = if check_options_arg(options, "chebyshevHeatSpmv")? {
        (
            opt_f64_prop(options, "lambdaMax", DEFAULT_LAMBDA_MAX)?,
            opt_maybe_usize_prop(options, "degree")?,
        )
    } else {
        (DEFAULT_LAMBDA_MAX, None)
    };
    let deg = deg_opt.unwrap_or_else(|| adaptive_chebyshev_degree_scaled(t, lmax));
    if vectors.iter().any(|v| !v.is_finite()) {
        return Err(non_finite_err("vectors must contain only finite values"));
    }
    let coeffs =
        compute_chebyshev_bandpass_coefficients(&[(1.0, t)], lmax, deg).map_err(core_err_to_js)?;
    let n = graph.inner().num_nodes();
    let total = n
        .checked_mul(ncols)
        .ok_or_else(|| invalid_input_err("chebyshevHeatSpmv matrix size overflow"))?;
    let mut out = vec![0.0_f64; total];
    let (l_sym, _) = symmetric_normalized_laplacian(graph.inner());
    let mut ws = ChebyshevWorkspace::with_capacity(total);
    ws.apply_batch_with_lambda_max(&l_sym, lmax, &coeffs, vectors, &mut out, ncols)
        .map_err(core_err_to_js)?;
    Ok(out)
}

/// Applies Chebyshev heat diffusion $\exp(-t A)$ to a row-major `(numNodes, numCols)` buffer
/// for an arbitrary symmetric PSD `CsrMatrix` with dynamic `[0, lambdaMax]` scaling.
#[wasm_bindgen(js_name = chebyshevHeatSpmvMatrix)]
pub fn js_chebyshev_heat_spmv_matrix(
    matrix: &JsCsrMatrix,
    vectors: &[f64],
    num_cols: f64,
    t: f64,
    options: &JsValue,
) -> Result<Vec<f64>, JsValue> {
    let ncols = parse_non_negative_usize(num_cols, "numCols")?;
    let (deg, lambda_max, steps, sf, seed_u64) =
        if check_options_arg(options, "chebyshevHeatSpmvMatrix")? {
            (
                opt_maybe_usize_prop(options, "degree")?,
                opt_maybe_f64_prop(options, "lambdaMax")?,
                opt_usize_prop(options, "lanczosSteps", DEFAULT_SPECTRUM_LANCZOS_STEPS)?,
                opt_f64_prop(
                    options,
                    "safetyFactor",
                    DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR,
                )?,
                opt_seed_prop(options)?,
            )
        } else {
            (
                None,
                None,
                DEFAULT_SPECTRUM_LANCZOS_STEPS,
                DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR,
                DEFAULT_SEED,
            )
        };
    if vectors.iter().any(|v| !v.is_finite()) {
        return Err(non_finite_err("vectors must contain only finite values"));
    }
    let n = matrix.inner().num_nodes();
    let total = n
        .checked_mul(ncols)
        .ok_or_else(|| invalid_input_err("chebyshevHeatSpmvMatrix size overflow"))?;
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
#[wasm_bindgen(js_name = chebyshevDogJlEmbeddingMatrix)]
pub fn js_chebyshev_dog_jl_embedding_matrix(
    matrix: &JsCsrMatrix,
    options: &JsValue,
) -> Result<Vec<f64>, JsValue> {
    let (jl_dim_val, deg, seed_u64, eps_psd_val, t1, t2, alpha, lambda_max, steps, sf, scale_vec) =
        if check_options_arg(options, "chebyshevDogJlEmbeddingMatrix")? {
            let sv_val = get_prop(options, "scaleVec")?;
            (
                opt_usize_prop(options, "jlDim", DEFAULT_JL_DIM)?,
                opt_maybe_usize_prop(options, "degree")?,
                opt_seed_prop(options)?,
                opt_f64_prop(options, "epsPsd", DEFAULT_EPS_PSD)?,
                opt_maybe_f64_prop(options, "t1")?,
                opt_maybe_f64_prop(options, "t2")?,
                opt_maybe_f64_prop(options, "alpha")?,
                opt_maybe_f64_prop(options, "lambdaMax")?,
                opt_usize_prop(options, "lanczosSteps", DEFAULT_SPECTRUM_LANCZOS_STEPS)?,
                opt_f64_prop(
                    options,
                    "safetyFactor",
                    DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR,
                )?,
                parse_optional_f64_sequence(&sv_val, "scaleVec")?,
            )
        } else {
            (
                DEFAULT_JL_DIM,
                None,
                DEFAULT_SEED,
                DEFAULT_EPS_PSD,
                None,
                None,
                None,
                None,
                DEFAULT_SPECTRUM_LANCZOS_STEPS,
                DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR,
                None,
            )
        };

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
#[wasm_bindgen(js_name = chebyshevDogJlEmbedding)]
pub fn js_chebyshev_dog_jl_embedding(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<Vec<f64>, JsValue> {
    let (jl_dim_val, deg, seed_u64, deg_norm, eps_psd_val, t1, t2, alpha) =
        if check_options_arg(options, "chebyshevDogJlEmbedding")? {
            (
                opt_usize_prop(options, "jlDim", DEFAULT_JL_DIM)?,
                opt_usize_prop(options, "degree", DEFAULT_DOG_DEGREE)?,
                opt_seed_prop(options)?,
                opt_bool_prop(options, "degreeNormalize", true)?,
                opt_f64_prop(options, "epsPsd", DEFAULT_EPS_PSD)?,
                opt_maybe_f64_prop(options, "t1")?,
                opt_maybe_f64_prop(options, "t2")?,
                opt_maybe_f64_prop(options, "alpha")?,
            )
        } else {
            (
                DEFAULT_JL_DIM,
                DEFAULT_DOG_DEGREE,
                DEFAULT_SEED,
                true,
                DEFAULT_EPS_PSD,
                None,
                None,
                None,
            )
        };

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
