//! WebAssembly bindings for spectral graph layouts, MDS variants,
//! Laplacian eigensolvers, and 2D layout normalization utilities.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    DEFAULT_SEED, core_err_to_js, ffi_err_to_js, invalid_input_err, parse_non_negative_usize,
    parse_seed_u64,
};
use crate::js_options::{
    check_options_arg, get_prop, opt_bool_prop, opt_f64_prop, opt_maybe_f64_prop,
    opt_maybe_usize_prop, opt_seed_prop, opt_usize_prop,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::linalg_matrix::JsEigenResult;
use crate::linalg_mds::parse_optional_target_std;
use egraph::core::{Layout2d, LayoutGenerator, LayoutNd};
use egraph::linalg::{
    ClassicalMds, KorenSpectral, KorenSpectralConfig, PivotMds, PivotMdsConfig, RdMds, RdMdsConfig,
    normalize_layout_2d, robust_normalize_layout_2d, select_maxmin_sp_pivots,
    smallest_sym_laplacian_eigenpairs, symmetric_eigen_jacobi,
};
use egraph::neighborhood::{compute_diffusion_embedding, compute_resistance_embedding};
use wasm_bindgen::prelude::*;

const DEFAULT_NUM_PIVOTS: usize = 50;
const DEFAULT_MIN_DEGREE: usize = 2;
const DEFAULT_TOL: f64 = 1e-4;
const DEFAULT_MAX_ITERS: usize = 300;
const DEFAULT_N_MODES: usize = 16;

/// Selects well-spaced shortest-path pivots via Core-First MaxMin BFS.
#[wasm_bindgen(js_name = selectMaxminSpPivots)]
pub fn js_select_maxmin_sp_pivots(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<Vec<u32>, JsValue> {
    let (pivots, seed_u64, min_deg) = if check_options_arg(options, "selectMaxminSpPivots")? {
        (
            opt_usize_prop(options, "numPivots", DEFAULT_NUM_PIVOTS)?,
            opt_seed_prop(options)?,
            opt_usize_prop(options, "minDegree", DEFAULT_MIN_DEGREE)?,
        )
    } else {
        (DEFAULT_NUM_PIVOTS, DEFAULT_SEED, DEFAULT_MIN_DEGREE)
    };
    select_maxmin_sp_pivots(graph.inner(), pivots, seed_u64, min_deg).map_err(core_err_to_js)
}

/// Computes a 2D `Layout2d` (`f32`) via Core-First MaxMin PivotMDS.
#[wasm_bindgen(js_name = pivotMdsInit)]
pub fn js_pivot_mds_init(graph: &JsCsrGraph, options: &JsValue) -> Result<JsLayout2d, JsValue> {
    let mut cfg = PivotMdsConfig::default();
    if check_options_arg(options, "pivotMdsInit")? {
        cfg.num_pivots = opt_usize_prop(options, "numPivots", DEFAULT_NUM_PIVOTS)?;
        cfg.seed = opt_seed_prop(options)?;
        cfg.min_degree = opt_usize_prop(options, "minDegree", DEFAULT_MIN_DEGREE)?;
        let ts_val = get_prop(options, "targetStd")?;
        cfg.target_std = parse_optional_target_std(&ts_val)?;
    }
    let layout = PivotMds::<Layout2d<f32>, f32>::from_config(graph.inner(), cfg)
        .generate()
        .map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&layout))
}

/// Computes Koren's degree-normalized 2D spectral initialization `Layout2d` (`f32`).
#[wasm_bindgen(js_name = korenSpectralInit)]
pub fn js_koren_spectral_init(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsLayout2d, JsValue> {
    let mut cfg = KorenSpectralConfig::default();
    if check_options_arg(options, "korenSpectralInit")? {
        let ts_val = get_prop(options, "targetStd")?;
        cfg.target_std = parse_optional_target_std(&ts_val)?;
        cfg.tol = opt_f64_prop(options, "tol", DEFAULT_TOL)?;
        cfg.max_iters = opt_usize_prop(options, "maxIterations", DEFAULT_MAX_ITERS)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    if !cfg.tol.is_finite() || cfg.tol <= 0.0 {
        return Err(invalid_input_err(
            "tol must be finite and strictly positive",
        ));
    }
    if cfg.max_iters == 0 {
        return Err(invalid_input_err("maxIterations must be >= 1"));
    }
    let layout = KorenSpectral::<Layout2d<f32>, f32>::from_config(graph.inner(), cfg)
        .generate()
        .map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&layout))
}

/// Computes the smallest `nModes` nontrivial eigenpairs of $L_{\mathrm{sym}}$.
#[wasm_bindgen(js_name = smallestLaplacianEigenpairs)]
pub fn js_smallest_laplacian_eigenpairs(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsEigenResult, JsValue> {
    let (modes, tol_val, max_iters_val, seed_u64) =
        if check_options_arg(options, "smallestLaplacianEigenpairs")? {
            (
                opt_usize_prop(options, "nModes", DEFAULT_N_MODES)?,
                opt_f64_prop(options, "tol", DEFAULT_TOL)?,
                opt_usize_prop(options, "maxIterations", DEFAULT_MAX_ITERS)?,
                opt_seed_prop(options)?,
            )
        } else {
            (
                DEFAULT_N_MODES,
                DEFAULT_TOL,
                DEFAULT_MAX_ITERS,
                DEFAULT_SEED,
            )
        };
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
    options: &JsValue,
) -> Result<JsEigenResult, JsValue> {
    let n_val = parse_non_negative_usize(n, "n")?;
    let (requested_modes, sweeps, tol_val) = if check_options_arg(options, "symmetricEigenJacobi")?
    {
        let m = opt_maybe_usize_prop(options, "nModes")?;
        if m == Some(0) {
            return Err(invalid_input_err("nModes must be >= 1"));
        }
        (
            m,
            opt_usize_prop(options, "maxSweeps", 120)?,
            opt_f64_prop(options, "tol", 1e-12)?,
        )
    } else {
        (None, 120, 1e-12)
    };
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
    Ok(JsEigenResult::from_jacobi(lambdas, vectors, n_val, k))
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
pub fn js_classical_mds_2d(graph: &JsCsrGraph) -> Result<JsLayout2d, JsValue> {
    let layout = ClassicalMds::<Layout2d<f32>, f32>::new(graph.inner())
        .generate()
        .map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&layout))
}

/// Computes a `LayoutNd` (`f32`) zero-guarded resistance-distance spectral embedding (`RdMds`).
#[wasm_bindgen(js_name = rdMds)]
pub fn js_rd_mds(graph: &JsCsrGraph, options: &JsValue) -> Result<JsLayoutNd, JsValue> {
    let mut cfg = RdMdsConfig::default();
    let mut dims = 2usize;
    if check_options_arg(options, "rdMds")? {
        dims = opt_usize_prop(options, "dims", 2)?;
        cfg.eig_eps = opt_f64_prop(options, "eigEps", cfg.eig_eps)?;
        cfg.tol = opt_f64_prop(options, "tol", cfg.tol)?;
        cfg.max_iters = opt_usize_prop(options, "maxIterations", cfg.max_iters)?;
        cfg.seed = opt_seed_prop(options)?;
        cfg.degree_normalize = opt_bool_prop(options, "degreeNormalize", cfg.degree_normalize)?;
        let ts_val = get_prop(options, "targetStd")?;
        if !ts_val.is_undefined() {
            cfg.target_std = parse_optional_target_std(&ts_val)?;
        }
    }
    cfg.dims = Some(dims);
    let layout = RdMds::<LayoutNd<f32>, f32>::from_config(graph.inner(), cfg)
        .generate()
        .map_err(core_err_to_js)?;
    Ok(JsLayoutNd::from_inner(layout))
}

/// Sanitizes non-finite entries, centers `layout` in place, and scales isotropically to `targetStd` (default `1.0`).
#[wasm_bindgen(js_name = normalizeLayout2d)]
pub fn js_normalize_layout_2d(
    layout: &mut JsLayout2d,
    target_std: Option<f64>,
    seed: &JsValue,
) -> Result<(), JsValue> {
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
    options: &JsValue,
) -> Result<(), JsValue> {
    let (t_std, q_val, seed_u64) = if check_options_arg(options, "robustNormalizeLayout2d")? {
        (
            opt_f64_prop(options, "targetStd", 1.0)?,
            opt_f64_prop(options, "quantile", 0.95)?,
            opt_seed_prop(options)?,
        )
    } else {
        (1.0, 0.95, DEFAULT_SEED)
    };
    let mut soa = layout.inner().to_layout2d();
    robust_normalize_layout_2d(&mut soa, t_std, q_val, seed_u64).map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(())
}

/// Computes a `dims`-dimensional Representative / Landmark MDS `LayoutNd` (`f32`) from a flat
/// row-major `(numNodes, numReps)` distance matrix.
#[wasm_bindgen(js_name = representativeMds)]
pub fn js_representative_mds(
    distances_to_reps: &[f64],
    num_nodes: f64,
    num_reps: f64,
    dims: f64,
) -> Result<JsLayoutNd, JsValue> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let r = parse_non_negative_usize(num_reps, "numReps")?;
    let d = parse_non_negative_usize(dims, "dims")?;
    let layout = PivotMds::<LayoutNd<f32>, f64>::from_distances(distances_to_reps, n, r)
        .with_dims(d)
        .with_target_std(None)
        .generate()
        .map_err(core_err_to_js)?;
    Ok(JsLayoutNd::from_inner(layout))
}

/// Computes effective-resistance spectral embedding $D^{-1/2} U \Lambda^{-1/2}$.
#[wasm_bindgen(js_name = computeResistanceEmbedding)]
pub fn js_compute_resistance_embedding(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsSpectralEmbeddingResult, JsValue> {
    let (d, tol_val, m_iters, seed_u64, eps) =
        if check_options_arg(options, "computeResistanceEmbedding")? {
            (
                opt_usize_prop(options, "dim", DEFAULT_N_MODES)?,
                opt_f64_prop(options, "tol", DEFAULT_TOL)?,
                opt_usize_prop(options, "maxIterations", DEFAULT_MAX_ITERS)?,
                opt_seed_prop(options)?,
                opt_f64_prop(options, "eigEps", 1e-10)?,
            )
        } else {
            (
                DEFAULT_N_MODES,
                DEFAULT_TOL,
                DEFAULT_MAX_ITERS,
                DEFAULT_SEED,
                1e-10,
            )
        };
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
    options: &JsValue,
) -> Result<JsSpectralEmbeddingResult, JsValue> {
    let (d, p_val, t, tol_val, m_iters, seed_u64) =
        if check_options_arg(options, "computeDiffusionEmbedding")? {
            (
                opt_usize_prop(options, "dim", DEFAULT_N_MODES)?,
                opt_f64_prop(options, "p", 0.05)?,
                opt_maybe_f64_prop(options, "t")?,
                opt_f64_prop(options, "tol", DEFAULT_TOL)?,
                opt_usize_prop(options, "maxIterations", DEFAULT_MAX_ITERS)?,
                opt_seed_prop(options)?,
            )
        } else {
            (
                DEFAULT_N_MODES,
                0.05,
                None,
                DEFAULT_TOL,
                DEFAULT_MAX_ITERS,
                DEFAULT_SEED,
            )
        };
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
