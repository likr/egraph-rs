//! WebAssembly bindings for regularized DoG Varadhan landmark tables.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, parse_opt_usize_param, parse_optional_usize, parse_seed_u64,
};
use crate::linalg_matrix::JsCsrMatrix;
use egraph::linalg::SpectralMomentStats;
use egraph::neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, DogVaradhanStats, RhghLandmarkStats,
    compute_regularized_dog_varadhan_landmarks,
    compute_regularized_dog_varadhan_landmarks_from_matrix,
};
use wasm_bindgen::prelude::*;

const DEFAULT_MIN_DEGREE: usize = 2;
const DEFAULT_JL_DIM: usize = 64;
const DEFAULT_EPS_PSD: f64 = 1e-4;
const DEFAULT_VARADHAN_LANDMARKS: usize = 64;
const DEFAULT_P_EXP: f64 = 0.5;
const DEFAULT_T_COARSE: f64 = 48.0;

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

/// Computes regularized DoG Varadhan landmark tables and optional coarse heat scaffold.
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

/// Computes regularized DoG Varadhan landmark tables from a custom symmetric PSD `CsrMatrix`
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
