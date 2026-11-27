//! WebAssembly bindings for `ChebyshevDogKernel` and Chebyshev spectral filter helpers.

use egraph::linalg::{
    ChebyshevDogConfig, ChebyshevDogKernel, ChebyshevWorkspace, adaptive_chebyshev_degree,
    adaptive_chebyshev_degree_scaled, build_dog_terms, build_regularized_dog_terms,
    compute_chebyshev_bandpass_coefficients, project_psd_spectral_filter,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, parse_f64_sequence, parse_non_negative_usize, set_f64_prop, set_js_prop,
};
use crate::js_options::{
    check_options_arg, opt_f64_prop, opt_maybe_f64_prop, opt_maybe_usize_prop, opt_seed_prop,
    opt_usize_prop,
};
use crate::linalg_matrix::{JsCsrMatrix, JsSpectralMomentStats};

fn parse_chebyshev_dog_config(options: &JsValue, ctx: &str) -> Result<ChebyshevDogConfig, JsError> {
    let mut cfg = ChebyshevDogConfig::default();
    if check_options_arg(options, ctx)? {
        cfg.t1 = opt_maybe_f64_prop(options, "t1")?;
        cfg.t2 = opt_maybe_f64_prop(options, "t2")?;
        cfg.alpha = opt_maybe_f64_prop(options, "alpha")?;
        cfg.beta_floor = opt_maybe_f64_prop(options, "betaFloor")?;
        cfg.degree = opt_maybe_usize_prop(options, "degree")?;
        cfg.n_landmarks = opt_usize_prop(options, "numLandmarks", cfg.n_landmarks)?;
        cfg.min_degree = opt_usize_prop(options, "minDegree", cfg.min_degree)?;
        cfg.jl_dim = opt_usize_prop(options, "jlDim", cfg.jl_dim)?;
        cfg.eps_psd = opt_f64_prop(options, "epsPsd", cfg.eps_psd)?;
        cfg.lambda_max = opt_f64_prop(options, "lambdaMax", cfg.lambda_max)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

fn parse_filter_terms(terms: &JsValue, ctx: &str) -> Result<Vec<(f64, f64)>, JsError> {
    if terms.is_null() || terms.is_undefined() || !terms.is_object() {
        return Err(JsError::new(&format!(
            "{ctx}: terms must be an array of [weight, t] pairs or flat interleaved numeric array"
        )));
    }
    if let Some(arr) = terms.dyn_ref::<js_sys::Array>() {
        if arr.length() > 0 && js_sys::Array::is_array(&arr.get(0)) {
            let mut out = Vec::with_capacity(arr.length() as usize);
            for idx in 0..arr.length() {
                let pair = parse_f64_sequence(&arr.get(idx), "terms[i]")?;
                if pair.len() != 2 {
                    return Err(JsError::new(&format!(
                        "{ctx}: each term must have length 2 [weight, t], got {}",
                        pair.len()
                    )));
                }
                out.push((pair[0], pair[1]));
            }
            return Ok(out);
        }
    }
    let flat = parse_f64_sequence(terms, "terms")?;
    if !flat.len().is_multiple_of(2) {
        return Err(JsError::new(&format!(
            "{ctx}: flat terms array length must be even, got {}",
            flat.len()
        )));
    }
    Ok(flat.chunks_exact(2).map(|c| (c[0], c[1])).collect())
}

fn terms_to_js_array(terms: &[(f64, f64)]) -> js_sys::Array {
    let arr = js_sys::Array::new_with_length(terms.len() as u32);
    for (idx, &(w, t)) in terms.iter().enumerate() {
        let pair = js_sys::Array::new_with_length(2);
        pair.set(0, JsValue::from_f64(w));
        pair.set(1, JsValue::from_f64(t));
        arr.set(idx as u32, pair.into());
    }
    arr
}

/// Pure $O(\mathrm{deg} \cdot (k_{\mathrm{lm}} + d_{\mathrm{JL}}) \cdot |E|)$ regularized DoG
/// diffusion kernel backed by Chebyshev polynomials and deterministic JL diagonal probes.
#[wasm_bindgen(js_name = ChebyshevDogKernel)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsChebyshevDogKernel {
    inner: ChebyshevDogKernel,
}

#[wasm_bindgen(js_class = ChebyshevDogKernel)]
impl JsChebyshevDogKernel {
    /// Builds a `ChebyshevDogKernel` from `graph` using Core-First MaxMin landmarks.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsChebyshevDogKernel, JsError> {
        let cfg = parse_chebyshev_dog_config(options, "ChebyshevDogKernel")?;
        let inner = ChebyshevDogKernel::build(graph.inner(), &cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Builds a `ChebyshevDogKernel` using explicitly provided `landmarks`.
    #[wasm_bindgen(js_name = withLandmarks)]
    pub fn with_landmarks(
        graph: &JsCsrGraph,
        landmarks: &[u32],
        options: &JsValue,
    ) -> Result<JsChebyshevDogKernel, JsError> {
        let cfg = parse_chebyshev_dog_config(options, "ChebyshevDogKernel.withLandmarks")?;
        let inner = ChebyshevDogKernel::build_with_landmarks(graph.inner(), &cfg, landmarks)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    #[wasm_bindgen(getter, js_name = numLandmarks)]
    pub fn num_landmarks(&self) -> usize {
        self.inner.num_landmarks()
    }

    #[wasm_bindgen(getter, js_name = jlDim)]
    pub fn jl_dim(&self) -> usize {
        self.inner.jl_dim()
    }

    pub fn landmarks(&self) -> Vec<u32> {
        self.inner.landmarks().to_vec()
    }

    #[wasm_bindgen(js_name = landmarkKernelMatrix)]
    pub fn landmark_kernel_matrix(&self) -> Vec<f64> {
        self.inner.landmark_kernel_matrix().to_vec()
    }

    #[wasm_bindgen(js_name = diagEstimate)]
    pub fn diag_estimate(&self) -> Vec<f64> {
        self.inner.diag_estimate().to_vec()
    }

    #[wasm_bindgen(js_name = jlEmbedding)]
    pub fn jl_embedding(&self) -> Vec<f64> {
        self.inner.jl_embedding().to_vec()
    }

    #[wasm_bindgen(js_name = momentStats)]
    pub fn moment_stats(&self) -> JsSpectralMomentStats {
        JsSpectralMomentStats::from_inner(*self.inner.moment_stats())
    }

    #[wasm_bindgen(js_name = termsPsd)]
    pub fn terms_psd(&self) -> js_sys::Array {
        terms_to_js_array(self.inner.terms_psd())
    }

    #[wasm_bindgen(getter)]
    pub fn t1(&self) -> f64 {
        self.inner.t1()
    }

    #[wasm_bindgen(getter)]
    pub fn t2(&self) -> f64 {
        self.inner.t2()
    }

    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha()
    }

    #[wasm_bindgen(getter)]
    pub fn degree(&self) -> usize {
        self.inner.degree()
    }

    #[wasm_bindgen(getter, js_name = deltaPsd)]
    pub fn delta_psd(&self) -> f64 {
        self.inner.delta_psd()
    }

    #[wasm_bindgen(getter, js_name = hMin)]
    pub fn h_min(&self) -> f64 {
        self.inner.h_min()
    }

    #[wasm_bindgen(getter, js_name = betaAuto)]
    pub fn beta_auto(&self) -> f64 {
        self.inner.beta_auto()
    }

    #[wasm_bindgen(getter, js_name = betaEff)]
    pub fn beta_eff(&self) -> f64 {
        self.inner.beta_eff()
    }

    /// Applies the regularized DoG operator to row-major `(numNodes, ncols)` matrix `v`.
    #[wasm_bindgen(js_name = applyBatch)]
    pub fn apply_batch(
        &self,
        l_sym: &JsCsrMatrix,
        v: &[f64],
        ncols: f64,
    ) -> Result<Vec<f64>, JsError> {
        let nc = parse_non_negative_usize(ncols, "ncols")?;
        let n = self.inner.num_nodes();
        let total = n
            .checked_mul(nc)
            .ok_or_else(|| JsError::new("ChebyshevDogKernel.applyBatch size overflow"))?;
        let mut out = vec![0.0_f64; total];
        let mut ws = ChebyshevWorkspace::with_capacity(total);
        self.inner
            .apply_batch(l_sym.inner(), v, &mut out, nc, &mut ws)
            .map_err(core_err_to_js)?;
        Ok(out)
    }

    /// Converts the landmark kernel block into a `(numNodes, numLandmarks)` Normalized Varadhan table.
    #[wasm_bindgen(js_name = varadhanLandmarkTable)]
    pub fn varadhan_landmark_table(
        &self,
        p_exp: Option<f64>,
        degree_normalize: Option<bool>,
        eps_sim: Option<f64>,
    ) -> Result<Vec<f64>, JsError> {
        self.inner
            .varadhan_landmark_table(
                p_exp.unwrap_or(0.5),
                degree_normalize.unwrap_or(true),
                eps_sim.unwrap_or(1e-12),
            )
            .map_err(core_err_to_js)
    }
}

/// Builds 2-term Difference-of-Gaussians filter terms `[[1 - betaFloor, t1], [-alpha * (1 - betaFloor), t2]]`.
#[wasm_bindgen(js_name = buildDogTerms)]
pub fn js_build_dog_terms(
    t1: f64,
    t2: f64,
    alpha: f64,
    beta_floor: Option<f64>,
) -> Result<js_sys::Array, JsError> {
    let terms =
        build_dog_terms(t1, t2, alpha, beta_floor.unwrap_or(0.0)).map_err(core_err_to_js)?;
    Ok(terms_to_js_array(&terms))
}

/// Builds parameter-free R1 regularized DoG terms from $\sigma_\lambda$.
#[wasm_bindgen(js_name = buildRegularizedDogTerms)]
pub fn js_build_regularized_dog_terms(
    sigma_lambda: f64,
    beta_floor: Option<f64>,
) -> Result<js_sys::Array, JsError> {
    let terms = build_regularized_dog_terms(sigma_lambda, beta_floor.unwrap_or(0.0))
        .map_err(core_err_to_js)?;
    Ok(terms_to_js_array(&terms))
}

/// Returns adaptive Chebyshev polynomial degree for diffusion time `t`.
#[wasm_bindgen(js_name = adaptiveChebyshevDegree)]
pub fn js_adaptive_chebyshev_degree(t: f64) -> usize {
    adaptive_chebyshev_degree(t)
}

/// Returns adaptive Chebyshev polynomial degree scaled for spectral upper bound `lambdaMax`.
#[wasm_bindgen(js_name = adaptiveChebyshevDegreeScaled)]
pub fn js_adaptive_chebyshev_degree_scaled(t: f64, lambda_max: Option<f64>) -> usize {
    adaptive_chebyshev_degree_scaled(t, lambda_max.unwrap_or(2.0))
}

/// Computes Chebyshev expansion coefficients for $\sum_m w_m \exp(-t_m \lambda)$ over `[0, lambdaMax]`.
#[wasm_bindgen(js_name = computeChebyshevBandpassCoefficients)]
pub fn js_compute_chebyshev_bandpass_coefficients(
    terms: &JsValue,
    lambda_max: Option<f64>,
    degree: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    let parsed = parse_filter_terms(terms, "computeChebyshevBandpassCoefficients")?;
    let lmax = lambda_max.unwrap_or(2.0);
    let deg = degree
        .map(|d| parse_non_negative_usize(d, "degree"))
        .transpose()?
        .unwrap_or(25);
    compute_chebyshev_bandpass_coefficients(&parsed, lmax, deg).map_err(core_err_to_js)
}

/// Projects spectral filter `terms` onto the PSD cone, returning `{ termsPsd, deltaPsd, hMin }`.
#[wasm_bindgen(js_name = projectPsdSpectralFilter)]
pub fn js_project_psd_spectral_filter(
    terms: &JsValue,
    lambda_max: Option<f64>,
    degree: Option<f64>,
    eps_psd: Option<f64>,
) -> Result<JsValue, JsError> {
    let parsed = parse_filter_terms(terms, "projectPsdSpectralFilter")?;
    let lmax = lambda_max.unwrap_or(2.0);
    let deg = degree
        .map(|d| parse_non_negative_usize(d, "degree"))
        .transpose()?
        .unwrap_or(25);
    let eps = eps_psd.unwrap_or(1e-4);
    let (terms_psd, delta_psd, h_min) =
        project_psd_spectral_filter(&parsed, lmax, deg, eps).map_err(core_err_to_js)?;
    let obj = js_sys::Object::new();
    set_js_prop(&obj, "termsPsd", &terms_to_js_array(&terms_psd).into())?;
    set_f64_prop(&obj, "deltaPsd", delta_psd)?;
    set_f64_prop(&obj, "hMin", h_min)?;
    Ok(obj.into())
}
