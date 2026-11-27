//! WebAssembly bindings for spectral graph layouts, MDS variants,
//! Laplacian eigensolvers, and 2D layout normalization utilities.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, ffi_err_to_js, parse_non_negative_usize, parse_optional_usize, parse_seed_u64,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::linalg_matrix::JsEigenResult;
use egraph::core::{Layout2d, LayoutNd};
use egraph::linalg::{
    EigenResult, PivotMdsConfig, RdMds, classical_mds_2d, koren_spectral_init, normalize_layout_2d,
    pivot_mds, representative_mds, robust_normalize_layout_2d, select_maxmin_sp_pivots,
    smallest_sym_laplacian_eigenpairs, symmetric_eigen_jacobi,
};
use egraph::neighborhood::{compute_diffusion_embedding, compute_resistance_embedding};
use wasm_bindgen::prelude::*;

const DEFAULT_NUM_PIVOTS: usize = 50;
const DEFAULT_MIN_DEGREE: usize = 2;
const DEFAULT_TOL: f64 = 1e-4;
const DEFAULT_MAX_ITERS: usize = 300;
const DEFAULT_N_MODES: usize = 16;

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

/// Selects well-spaced shortest-path pivots via Core-First MaxMin BFS.
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

/// Computes a 2D `Layout2d` (`f32`) via Core-First MaxMin PivotMDS.
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

/// Computes Koren's degree-normalized 2D spectral initialization `Layout2d` (`f32`).
///
/// When `targetStd` is omitted or `undefined`, coordinates are normalized to `targetStd = 1.0`.
/// Pass `null` to disable isotropic standard-deviation normalization.
#[wasm_bindgen(js_name = korenSpectralInit)]
pub fn js_koren_spectral_init(
    graph: &JsCsrGraph,
    target_std: &JsValue,
    tol: Option<f64>,
    max_iterations: Option<f64>,
    seed: Option<f64>,
) -> Result<JsLayout2d, JsError> {
    let t_std = parse_optional_target_std(target_std)?;
    let tol_val = tol.unwrap_or(DEFAULT_TOL);
    if !tol_val.is_finite() || tol_val <= 0.0 {
        return Err(JsError::new("tol must be finite and strictly positive"));
    }
    let max_iters_val = parse_optional_usize(max_iterations, DEFAULT_MAX_ITERS, "maxIterations")?;
    if max_iters_val == 0 {
        return Err(JsError::new("maxIterations must be >= 1"));
    }
    let seed_u64 = parse_seed_u64(seed)?;
    let layout: Layout2d<f32> =
        koren_spectral_init(graph.inner(), t_std, tol_val, max_iters_val, seed_u64)
            .map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&layout))
}

/// Computes the smallest `nModes` nontrivial eigenpairs of $L_{\mathrm{sym}}$.
#[wasm_bindgen(js_name = smallestLaplacianEigenpairs)]
pub fn js_smallest_laplacian_eigenpairs(
    graph: &JsCsrGraph,
    n_modes: Option<f64>,
    tol: Option<f64>,
    max_iterations: Option<f64>,
    seed: Option<f64>,
) -> Result<JsEigenResult, JsError> {
    let modes = parse_optional_usize(n_modes, DEFAULT_N_MODES, "nModes")?;
    let tol_val = tol.unwrap_or(DEFAULT_TOL);
    let max_iters_val = parse_optional_usize(max_iterations, DEFAULT_MAX_ITERS, "maxIterations")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let inner =
        smallest_sym_laplacian_eigenpairs(graph.inner(), modes, tol_val, max_iters_val, seed_u64)
            .map_err(core_err_to_js)?;
    Ok(JsEigenResult::from_inner(inner))
}

/// Computes ascending symmetric eigenpairs `(lambdas, vectors)` of an `n x n` row-major matrix
/// via dense cyclic Jacobi rotations in `f64`, optionally truncated to the smallest `nModes`.
#[wasm_bindgen(js_name = symmetricEigenJacobi)]
pub fn js_symmetric_eigen_jacobi(
    a_row_major: &[f64],
    n: f64,
    n_modes: Option<f64>,
    max_sweeps: Option<f64>,
    tol: Option<f64>,
) -> Result<JsEigenResult, JsError> {
    let n_val = parse_non_negative_usize(n, "n")?;
    let requested_modes = match n_modes {
        None => None,
        Some(v) => {
            let m = parse_non_negative_usize(v, "nModes")?;
            if m == 0 {
                return Err(JsError::new("nModes must be >= 1"));
            }
            Some(m)
        }
    };
    let sweeps = parse_optional_usize(max_sweeps, 120, "maxSweeps")?;
    let tol_val = tol.unwrap_or(1e-12);
    let (mut lambdas, all_vectors) =
        symmetric_eigen_jacobi(a_row_major, n_val, sweeps, tol_val).map_err(core_err_to_js)?;
    let k = requested_modes.map_or(n_val, |m| m.min(n_val));
    lambdas.truncate(k);
    let vectors = if k == n_val {
        all_vectors
    } else {
        let mut sliced = Vec::with_capacity(n_val * k);
        for row in 0..n_val {
            let start = row * n_val;
            sliced.extend_from_slice(&all_vectors[start..start + k]);
        }
        sliced
    };
    Ok(JsEigenResult::from_inner(EigenResult {
        lambdas,
        vectors,
        inv_sqrt_deg: Vec::new(),
        num_nodes: n_val,
        num_modes: k,
        used_dense_fallback: true,
        iterations: 0,
    }))
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

/// Computes a 2D `Layout2d` (`f32`) using zero-guarded Classical MDS.
#[wasm_bindgen(js_name = classicalMds2d)]
pub fn js_classical_mds_2d(graph: &JsCsrGraph) -> Result<JsLayout2d, JsError> {
    let layout: Layout2d<f32> = classical_mds_2d(graph.inner()).map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&layout))
}

/// Computes a `LayoutNd` (`f32`) zero-guarded resistance-distance spectral embedding (`RdMds`).
#[wasm_bindgen(js_name = rdMds)]
pub fn js_rd_mds(
    graph: &JsCsrGraph,
    dims: Option<f64>,
    eig_eps: Option<f64>,
    tol: Option<f64>,
    max_iterations: Option<f64>,
    seed: Option<f64>,
    degree_normalize: Option<bool>,
) -> Result<JsLayoutNd, JsError> {
    let d = parse_optional_usize(dims, 2, "dims")?;
    let eps = eig_eps.unwrap_or(1e-10);
    let tol_val = tol.unwrap_or(1e-6);
    let m_iters = parse_optional_usize(max_iterations, 1000, "maxIterations")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let deg_norm = degree_normalize.unwrap_or(true);
    let layout: LayoutNd<f32> = RdMds::with_dims(d)
        .eig_eps(eps)
        .tol(tol_val)
        .max_iters(m_iters)
        .seed(seed_u64)
        .degree_normalize(deg_norm)
        .embedding(graph.inner())
        .map_err(core_err_to_js)?;
    Ok(JsLayoutNd::from_inner(layout))
}

/// Sanitizes non-finite entries, centers `layout` in place, and scales isotropically to `targetStd` (default `1.0`).
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

/// Median-centers and Gaussian-calibrated radial quantile-scales `layout` in place so core spread matches `targetStd` (default `1.0`).
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

/// Computes a `dim`-dimensional Representative / Landmark MDS `LayoutNd` (`f32`) from a flat
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

/// Computes effective-resistance spectral embedding $D^{-1/2} U \Lambda^{-1/2}$.
#[wasm_bindgen(js_name = computeResistanceEmbedding)]
pub fn js_compute_resistance_embedding(
    graph: &JsCsrGraph,
    dim: Option<f64>,
    tol: Option<f64>,
    max_iterations: Option<f64>,
    seed: Option<f64>,
    eig_eps: Option<f64>,
) -> Result<JsSpectralEmbeddingResult, JsError> {
    let d = parse_optional_usize(dim, DEFAULT_N_MODES, "dim")?;
    let tol_val = tol.unwrap_or(DEFAULT_TOL);
    let m_iters = parse_optional_usize(max_iterations, DEFAULT_MAX_ITERS, "maxIterations")?;
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

/// Computes heat-diffusion spectral embedding $D^{-1/2} U \exp(-0.5 t \Lambda)$.
#[wasm_bindgen(js_name = computeDiffusionEmbedding)]
pub fn js_compute_diffusion_embedding(
    graph: &JsCsrGraph,
    dim: Option<f64>,
    p: Option<f64>,
    t: Option<f64>,
    tol: Option<f64>,
    max_iterations: Option<f64>,
    seed: Option<f64>,
) -> Result<JsSpectralEmbeddingResult, JsError> {
    let d = parse_optional_usize(dim, DEFAULT_N_MODES, "dim")?;
    let p_val = p.unwrap_or(0.05);
    let tol_val = tol.unwrap_or(DEFAULT_TOL);
    let m_iters = parse_optional_usize(max_iterations, DEFAULT_MAX_ITERS, "maxIterations")?;
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
