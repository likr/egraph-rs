//! Chebyshev spectral filtering, heat-kernel SpMV, and DoG-JL embedding bindings.

use egraph::core::{CoreError, CoreResult};
use egraph::linalg::{
    ChebyshevWorkspace, DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR, DEFAULT_SPECTRUM_LANCZOS_STEPS,
    SpectralMomentStats, adaptive_chebyshev_degree_scaled, build_dog_terms,
    compute_chebyshev_bandpass_coefficients, project_psd_spectral_filter,
    spectral_moment_stats as linalg_spectral_moment_stats, symmetric_normalized_laplacian,
};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::matrix::PyCsrMatrix;
use crate::array_extract::extract_f64_1d_or_2d;
use crate::csr_graph::PyCsrGraph;
use crate::errors::{dimension_mismatch_err, invalid_input_err};
use crate::ffi_convert::{
    StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_optional_weights_1d,
};

const SPECTRAL_MOMENT_KEYS: [&str; 5] = [
    "sigma_lambda_sq",
    "sigma_lambda",
    "deg_cv",
    "mean_degree",
    "leaf_fraction",
];

/// Frozen exact `O(|E|)` spectral moment and degree statistics of `L_sym`.
#[pyclass(
    name = "SpectralMomentStats",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PySpectralMomentStats {
    pub(super) inner: SpectralMomentStats,
}

impl PySpectralMomentStats {
    pub(super) fn from_stats(inner: SpectralMomentStats) -> Self {
        Self { inner }
    }

    fn lookup(&self, py: Python<'_>, key: &str) -> PyResult<Option<Py<PyAny>>> {
        let val = match key {
            "sigma_lambda_sq" => self.inner.sigma_lambda_sq,
            "sigma_lambda" => self.inner.sigma_lambda,
            "deg_cv" => self.inner.deg_cv,
            "mean_degree" => self.inner.mean_degree,
            "leaf_fraction" => self.inner.leaf_fraction,
            _ => return Ok(None),
        };
        Ok(Some(val.into_py_any(py)?))
    }
}

#[pymethods]
impl PySpectralMomentStats {
    /// Spectral variance `sigma_lambda^2` of the symmetric normalized Laplacian.
    #[getter]
    fn sigma_lambda_sq(&self) -> f64 {
        self.inner.sigma_lambda_sq
    }

    /// Spectral standard deviation `sigma_lambda`.
    #[getter]
    fn sigma_lambda(&self) -> f64 {
        self.inner.sigma_lambda
    }

    /// Coefficient of variation of vertex degrees.
    #[getter]
    fn deg_cv(&self) -> f64 {
        self.inner.deg_cv
    }

    /// Arithmetic mean vertex degree.
    #[getter]
    fn mean_degree(&self) -> f64 {
        self.inner.mean_degree
    }

    /// Fraction of vertices with degree <= 1.
    #[getter]
    fn leaf_fraction(&self) -> f64 {
        self.inner.leaf_fraction
    }

    /// Return spectral moment statistics as a Python dictionary.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for k in SPECTRAL_MOMENT_KEYS {
            if let Some(v) = self.lookup(py, k)? {
                dict.set_item(k, v)?;
            }
        }
        Ok(dict.unbind())
    }

    /// Return ordered field keys for dictionary compatibility.
    fn keys(&self) -> Vec<&'static str> {
        SPECTRAL_MOMENT_KEYS.to_vec()
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        self.lookup(py, key)?
            .ok_or_else(|| PyKeyError::new_err(key.to_owned()))
    }

    fn __contains__(&self, key: &str) -> bool {
        SPECTRAL_MOMENT_KEYS.contains(&key)
    }

    fn __repr__(&self) -> String {
        format!(
            "SpectralMomentStats(sigma_lambda={:.6}, deg_cv={:.6}, mean_degree={:.3}, leaf_fraction={:.4})",
            self.inner.sigma_lambda,
            self.inner.deg_cv,
            self.inner.mean_degree,
            self.inner.leaf_fraction,
        )
    }
}

pub(super) fn resolve_dog_r1_defaults(
    stats: &SpectralMomentStats,
    t1: Option<f64>,
    t2: Option<f64>,
    alpha: Option<f64>,
) -> (f64, f64, f64) {
    let s = stats.sigma_lambda;
    (
        t1.unwrap_or(1.0 / (1.0 + s)),
        t2.unwrap_or(4.0 / (1.0 + s)),
        alpha.unwrap_or((1.0 - s).max(0.0)),
    )
}

pub(super) fn validate_dog_jl_params(degree: usize, jl_dim: usize, eps_psd: f64) -> CoreResult<()> {
    if degree == 0 {
        return Err(CoreError::invalid_input("degree must be >= 1"));
    }
    if jl_dim == 0 {
        return Err(CoreError::invalid_input("jl_dim must be >= 1"));
    }
    if !eps_psd.is_finite() {
        return Err(CoreError::non_finite("chebyshev_dog_jl_embedding::eps_psd"));
    }
    if eps_psd < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "eps_psd must be >= 0, got {eps_psd}"
        )));
    }
    Ok(())
}

/// Compute exact ``O(|E|)`` spectral variance and degree summary statistics of ``L_sym``.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph), text_signature = "(graph)")]
fn spectral_moment_stats(py: Python<'_>, graph: &PyCsrGraph) -> PySpectralMomentStats {
    let g = graph.inner();
    let stats = py.detach(|| linalg_spectral_moment_stats(&*g));
    PySpectralMomentStats::from_stats(stats)
}

/// Apply the Chebyshev heat-kernel approximation `exp(-t * L_sym)` to 1D `(N,)` or 2D `(N, C)` vectors.
///
/// Releases the Python GIL during computation and preserves the input array dimensionality.
#[pyfunction]
#[pyo3(
    signature = (graph, vectors, t, *, degree = None, lambda_max = StrictF64(2.0)),
    text_signature = "(graph, vectors, t, *, degree=None, lambda_max=2.0)"
)]
fn chebyshev_heat_spmv<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    vectors: &Bound<'_, PyAny>,
    t: StrictF64,
    degree: Option<StrictUsize>,
    lambda_max: StrictF64,
) -> PyResult<Bound<'py, PyAny>> {
    let (v_flat, nrows, ncols, is_2d) = extract_f64_1d_or_2d(vectors, "vectors")?;
    let g = graph.inner();
    let n = g.num_nodes();
    if nrows != n {
        return Err(dimension_mismatch_err(format!(
            "vectors row count {nrows} does not match graph.num_nodes ({n})"
        )));
    }
    let deg_opt = degree.map(|d| d.0);
    let t_val = t.0;
    let lmax_val = lambda_max.0;

    let out = py
        .detach(|| -> CoreResult<Vec<f64>> {
            let deg = deg_opt.unwrap_or_else(|| adaptive_chebyshev_degree_scaled(t_val, lmax_val));
            let coeffs = compute_chebyshev_bandpass_coefficients(&[(1.0, t_val)], lmax_val, deg)?;
            let (l_sym, _) = symmetric_normalized_laplacian(&*g);
            let mut out = vec![0.0_f64; v_flat.len()];
            let mut ws = ChebyshevWorkspace::with_capacity(v_flat.len());
            ws.apply_batch_with_lambda_max(&l_sym, lmax_val, &coeffs, &v_flat, &mut out, ncols)?;
            Ok(out)
        })
        .map_err(core_error_to_py)?;

    if is_2d {
        let arr = Array2::from_shape_vec((n, ncols), out)
            .map_err(|e| invalid_input_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, arr).into_any())
    } else {
        Ok(PyArray1::from_vec(py, out).into_any())
    }
}

/// Apply the Chebyshev heat-kernel approximation `exp(-t * A)` to 1D `(N,)` or 2D `(N, C)`
/// vectors for an arbitrary symmetric PSD `CsrMatrix` with dynamic `[0, lambda_max]` scaling.
///
/// Releases the Python GIL during computation and preserves the input array dimensionality.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (matrix, vectors, t, *, degree = None, lambda_max = None, lanczos_steps = StrictUsize(DEFAULT_SPECTRUM_LANCZOS_STEPS), safety_factor = StrictF64(DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR), seed = StrictU64(42)),
    text_signature = "(matrix, vectors, t, *, degree=None, lambda_max=None, lanczos_steps=16, safety_factor=1.05, seed=42)"
)]
fn chebyshev_heat_spmv_matrix<'py>(
    py: Python<'py>,
    matrix: &PyCsrMatrix,
    vectors: &Bound<'_, PyAny>,
    t: StrictF64,
    degree: Option<StrictUsize>,
    lambda_max: Option<StrictF64>,
    lanczos_steps: StrictUsize,
    safety_factor: StrictF64,
    seed: StrictU64,
) -> PyResult<Bound<'py, PyAny>> {
    let (v_flat, nrows, ncols, is_2d) = extract_f64_1d_or_2d(vectors, "vectors")?;
    let mat = matrix.inner();
    let n = mat.num_nodes();
    if nrows != n {
        return Err(dimension_mismatch_err(format!(
            "vectors row count {nrows} does not match matrix.num_nodes ({n})"
        )));
    }
    let deg_opt = degree.map(|d| d.0);
    let t_val = t.0;
    let lmax_opt = lambda_max.map(|v| v.0);
    let steps = lanczos_steps.0;
    let sf = safety_factor.0;
    let seed_val = seed.0;

    let out = py
        .detach(|| -> CoreResult<Vec<f64>> {
            let mut out = vec![0.0_f64; v_flat.len()];
            let mut ws = ChebyshevWorkspace::with_capacity(v_flat.len());
            ws.apply_heat_dynamic(
                mat, &v_flat, &mut out, ncols, t_val, deg_opt, lmax_opt, steps, sf, seed_val,
            )?;
            Ok(out)
        })
        .map_err(core_error_to_py)?;

    if is_2d {
        let arr = Array2::from_shape_vec((n, ncols), out)
            .map_err(|e| invalid_input_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, arr).into_any())
    } else {
        Ok(PyArray1::from_vec(py, out).into_any())
    }
}

/// Compute an `(N, jl_dim)` Johnson-Lindenstrauss regularized DoG spectral embedding
/// on an arbitrary symmetric PSD `CsrMatrix` with dynamic `[0, lambda_max]` scaling.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (matrix, *, t1 = None, t2 = None, alpha = None, degree = None, jl_dim = StrictUsize(64), seed = StrictU64(42), eps_psd = StrictF64(1e-4), lambda_max = None, lanczos_steps = StrictUsize(DEFAULT_SPECTRUM_LANCZOS_STEPS), safety_factor = StrictF64(DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR), scale_vec = None),
    text_signature = "(matrix, *, t1=None, t2=None, alpha=None, degree=None, jl_dim=64, seed=42, eps_psd=1e-4, lambda_max=None, lanczos_steps=16, safety_factor=1.05, scale_vec=None)"
)]
fn chebyshev_dog_jl_embedding_matrix<'py>(
    py: Python<'py>,
    matrix: &PyCsrMatrix,
    t1: Option<StrictF64>,
    t2: Option<StrictF64>,
    alpha: Option<StrictF64>,
    degree: Option<StrictUsize>,
    jl_dim: StrictUsize,
    seed: StrictU64,
    eps_psd: StrictF64,
    lambda_max: Option<StrictF64>,
    lanczos_steps: StrictUsize,
    safety_factor: StrictF64,
    scale_vec: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let sv_opt = extract_optional_weights_1d(scale_vec, "scale_vec")?;
    let mat = matrix.inner();
    let n = mat.num_nodes();
    let t1_opt = t1.map(|v| v.0);
    let t2_opt = t2.map(|v| v.0);
    let alpha_opt = alpha.map(|v| v.0);
    let deg_opt = degree.map(|d| d.0);
    let dim = jl_dim.0;
    let seed_val = seed.0;
    let eps = eps_psd.0;
    let lmax_opt = lambda_max.map(|v| v.0);
    let steps = lanczos_steps.0;
    let sf = safety_factor.0;

    let emb = py
        .detach(|| -> CoreResult<Vec<f64>> {
            let s = if n > 0 {
                let n_f = n as f64;
                let mu1 = mat.trace() / n_f;
                let frob_sq: f64 = mat.values().iter().map(|&v| v * v).sum();
                (frob_sq / n_f - mu1 * mu1).max(1e-6).sqrt()
            } else {
                1e-3
            };
            let t1_eff = t1_opt.unwrap_or(1.0 / (1.0 + s));
            let t2_eff = t2_opt.unwrap_or(4.0 / (1.0 + s));
            let alpha_eff = alpha_opt.unwrap_or((1.0 - s).max(0.0));
            let base_terms = build_dog_terms(t1_eff, t2_eff, alpha_eff, 0.0)?;
            let mut ws = ChebyshevWorkspace::with_capacity(n.saturating_mul(dim));
            let (out, _) = ws.compute_dog_jl_dynamic(
                mat,
                &base_terms,
                lmax_opt,
                deg_opt,
                dim,
                eps,
                steps,
                sf,
                seed_val,
                sv_opt.as_deref(),
            )?;
            Ok(out)
        })
        .map_err(core_error_to_py)?;

    let arr =
        Array2::from_shape_vec((n, dim), emb).map_err(|e| invalid_input_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Compute an `(N, jl_dim)` Johnson-Lindenstrauss regularized DoG spectral embedding.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (graph, *, t1 = None, t2 = None, alpha = None, degree = StrictUsize(25), jl_dim = StrictUsize(64), seed = StrictU64(42), degree_normalize = true, eps_psd = StrictF64(1e-4)),
    text_signature = "(graph, *, t1=None, t2=None, alpha=None, degree=25, jl_dim=64, seed=42, degree_normalize=True, eps_psd=1e-4)"
)]
fn chebyshev_dog_jl_embedding<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    t1: Option<StrictF64>,
    t2: Option<StrictF64>,
    alpha: Option<StrictF64>,
    degree: StrictUsize,
    jl_dim: StrictUsize,
    seed: StrictU64,
    degree_normalize: bool,
    eps_psd: StrictF64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    validate_dog_jl_params(degree.0, jl_dim.0, eps_psd.0).map_err(core_error_to_py)?;
    let g = graph.inner();
    let n = g.num_nodes();
    let t1_opt = t1.map(|v| v.0);
    let t2_opt = t2.map(|v| v.0);
    let alpha_opt = alpha.map(|v| v.0);
    let deg = degree.0;
    let dim = jl_dim.0;
    let seed_val = seed.0;
    let eps = eps_psd.0;

    let emb = py
        .detach(|| -> CoreResult<Vec<f64>> {
            let stats = linalg_spectral_moment_stats(&*g);
            let (t1_eff, t2_eff, alpha_eff) =
                resolve_dog_r1_defaults(&stats, t1_opt, t2_opt, alpha_opt);
            let base_terms = build_dog_terms(t1_eff, t2_eff, alpha_eff, 0.0)?;
            let (terms_psd, _, _) = project_psd_spectral_filter(&base_terms, 2.0, deg, eps)?;
            let (l_sym, inv_sqrt_deg) = symmetric_normalized_laplacian(&*g);
            let mut ws = ChebyshevWorkspace::with_capacity(n.saturating_mul(dim));
            ws.compute_kernel_jl_embedding(
                &l_sym,
                &terms_psd,
                2.0,
                deg,
                dim,
                seed_val,
                degree_normalize.then_some(inv_sqrt_deg.as_slice()),
            )
        })
        .map_err(core_error_to_py)?;

    let arr =
        Array2::from_shape_vec((n, dim), emb).map_err(|e| invalid_input_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpectralMomentStats>()?;
    m.add_function(wrap_pyfunction!(spectral_moment_stats, m)?)?;
    m.add_function(wrap_pyfunction!(chebyshev_heat_spmv, m)?)?;
    m.add_function(wrap_pyfunction!(chebyshev_heat_spmv_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(chebyshev_dog_jl_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(chebyshev_dog_jl_embedding_matrix, m)?)?;
    Ok(())
}
