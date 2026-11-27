//! `ChebyshevDogKernel` and Chebyshev spectral filter coefficient/degree primitives.

use egraph::core::CoreResult;
use egraph::linalg::{
    ChebyshevDogConfig, ChebyshevDogKernel, ChebyshevWorkspace,
    adaptive_chebyshev_degree as linalg_adaptive_chebyshev_degree,
    adaptive_chebyshev_degree_scaled as linalg_adaptive_chebyshev_degree_scaled,
    build_dog_terms as linalg_build_dog_terms,
    build_regularized_dog_terms as linalg_build_regularized_dog_terms,
    compute_chebyshev_bandpass_coefficients as linalg_compute_chebyshev_bandpass_coefficients,
    project_psd_spectral_filter as linalg_project_psd_spectral_filter,
};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::chebyshev::spectral_moment_fields;
use super::matrix::PyCsrMatrix;
use crate::array_extract::extract_f64_1d_or_2d;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d, is_non_sequence_container,
};

fn extract_filter_terms(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(f64, f64)>> {
    if is_non_sequence_container(obj) {
        return Err(PyTypeError::new_err(
            "terms must be a sequence of (weight, t) pairs",
        ));
    }
    let iter = obj
        .try_iter()
        .map_err(|_| PyTypeError::new_err("terms must be a sequence of (weight, t) pairs"))?;
    let mut terms = Vec::new();
    for item_res in iter {
        let item = item_res?;
        let (w, t): (StrictF64, StrictF64) = item
            .extract()
            .map_err(|_| PyTypeError::new_err("each term must be a (weight, t) float pair"))?;
        terms.push((w.0, t.0));
    }
    Ok(terms)
}

#[allow(clippy::too_many_arguments)]
fn build_dog_config(
    t1: Option<StrictF64>,
    t2: Option<StrictF64>,
    alpha: Option<StrictF64>,
    beta_floor: Option<StrictF64>,
    degree: Option<StrictUsize>,
    n_landmarks: StrictUsize,
    min_degree: StrictUsize,
    jl_dim: StrictUsize,
    eps_psd: StrictF64,
    lambda_max: StrictF64,
    seed: StrictU64,
) -> ChebyshevDogConfig {
    ChebyshevDogConfig {
        t1: t1.map(|v| v.0),
        t2: t2.map(|v| v.0),
        alpha: alpha.map(|v| v.0),
        beta_floor: beta_floor.map(|v| v.0),
        degree: degree.map(|v| v.0),
        n_landmarks: n_landmarks.0,
        min_degree: min_degree.0,
        jl_dim: jl_dim.0,
        eps_psd: eps_psd.0,
        lambda_max: lambda_max.0,
        seed: seed.0,
    }
}

/// Regularized Difference-of-Gaussians (DoG) Chebyshev spectral kernel and JL embedding.
#[pyclass(name = "ChebyshevDogKernel", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PyChebyshevDogKernel {
    inner: ChebyshevDogKernel,
}

#[pymethods]
impl PyChebyshevDogKernel {
    /// Build a `ChebyshevDogKernel` from `graph` using Core-First MaxMin landmarks.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, t1 = None, t2 = None, alpha = None, beta_floor = None, degree = None, n_landmarks = StrictUsize(64), min_degree = StrictUsize(2), jl_dim = StrictUsize(64), eps_psd = StrictF64(1e-4), lambda_max = StrictF64(2.0), seed = StrictU64(42)),
        text_signature = "(graph, *, t1=None, t2=None, alpha=None, beta_floor=None, degree=None, n_landmarks=64, min_degree=2, jl_dim=64, eps_psd=1e-4, lambda_max=2.0, seed=42)"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        t1: Option<StrictF64>,
        t2: Option<StrictF64>,
        alpha: Option<StrictF64>,
        beta_floor: Option<StrictF64>,
        degree: Option<StrictUsize>,
        n_landmarks: StrictUsize,
        min_degree: StrictUsize,
        jl_dim: StrictUsize,
        eps_psd: StrictF64,
        lambda_max: StrictF64,
        seed: StrictU64,
    ) -> PyResult<Self> {
        let cfg = build_dog_config(
            t1,
            t2,
            alpha,
            beta_floor,
            degree,
            n_landmarks,
            min_degree,
            jl_dim,
            eps_psd,
            lambda_max,
            seed,
        );
        let g = graph.inner();
        let inner = py
            .detach(|| ChebyshevDogKernel::build(&*g, &cfg))
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Build a `ChebyshevDogKernel` using explicitly provided `landmarks`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, landmarks, *, t1 = None, t2 = None, alpha = None, beta_floor = None, degree = None, jl_dim = StrictUsize(64), eps_psd = StrictF64(1e-4), lambda_max = StrictF64(2.0), seed = StrictU64(42)),
        text_signature = "(graph, landmarks, *, t1=None, t2=None, alpha=None, beta_floor=None, degree=None, jl_dim=64, eps_psd=1e-4, lambda_max=2.0, seed=42)"
    )]
    fn build_with_landmarks(
        py: Python<'_>,
        graph: &PyCsrGraph,
        landmarks: &Bound<'_, PyAny>,
        t1: Option<StrictF64>,
        t2: Option<StrictF64>,
        alpha: Option<StrictF64>,
        beta_floor: Option<StrictF64>,
        degree: Option<StrictUsize>,
        jl_dim: StrictUsize,
        eps_psd: StrictF64,
        lambda_max: StrictF64,
        seed: StrictU64,
    ) -> PyResult<Self> {
        let lm_vec = extract_u32_1d(landmarks, "landmarks")?;
        let cfg = build_dog_config(
            t1,
            t2,
            alpha,
            beta_floor,
            degree,
            StrictUsize(lm_vec.len().max(1)),
            StrictUsize(1),
            jl_dim,
            eps_psd,
            lambda_max,
            seed,
        );
        let g = graph.inner();
        let inner = py
            .detach(|| ChebyshevDogKernel::build_with_landmarks(&*g, &cfg, &lm_vec))
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    #[getter]
    fn num_landmarks(&self) -> usize {
        self.inner.num_landmarks()
    }

    #[getter]
    fn jl_dim(&self) -> usize {
        self.inner.jl_dim()
    }

    #[getter]
    fn t1(&self) -> f64 {
        self.inner.t1()
    }

    #[getter]
    fn t2(&self) -> f64 {
        self.inner.t2()
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }

    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    #[getter]
    fn delta_psd(&self) -> f64 {
        self.inner.delta_psd()
    }

    #[getter]
    fn h_min(&self) -> f64 {
        self.inner.h_min()
    }

    #[getter]
    fn beta_auto(&self) -> f64 {
        self.inner.beta_auto()
    }

    #[getter]
    fn beta_eff(&self) -> f64 {
        self.inner.beta_eff()
    }

    /// Return selected landmark vertex IDs as a `uint32` NumPy array.
    fn landmarks<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, self.inner.landmarks())
    }

    /// Return the `(num_nodes, num_landmarks)` effective kernel matrix `K_eff[:, P]`.
    fn landmark_kernel_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = Array2::from_shape_vec(
            (self.inner.num_nodes(), self.inner.num_landmarks()),
            self.inner.landmark_kernel_matrix().to_vec(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, arr))
    }

    /// Return the length-`num_nodes` JL diagonal estimate `diag(K_eff)`.
    fn diag_estimate<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.diag_estimate())
    }

    /// Return the `(num_nodes, jl_dim)` spectral square-root JL embedding.
    fn jl_embedding<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = Array2::from_shape_vec(
            (self.inner.num_nodes(), self.inner.jl_dim()),
            self.inner.jl_embedding().to_vec(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, arr))
    }

    /// Return the exact spectral moment summary statistics used to parameterize R1 DoG.
    fn moment_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (k, v) in spectral_moment_fields(self.inner.moment_stats()) {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    /// Return the PSD-projected spectral filter terms `[(weight, t), ...]`.
    fn terms_psd(&self) -> Vec<(f64, f64)> {
        self.inner.terms_psd().to_vec()
    }

    /// Compute the `(num_nodes, num_landmarks)` Normalized Varadhan Geodesic distance table.
    #[pyo3(
        signature = (p_exp = StrictF64(0.5), degree_normalize = true, eps_sim = StrictF64(1e-6)),
        text_signature = "($self, p_exp=0.5, degree_normalize=True, eps_sim=1e-6)"
    )]
    fn varadhan_landmark_table<'py>(
        &self,
        py: Python<'py>,
        p_exp: StrictF64,
        degree_normalize: bool,
        eps_sim: StrictF64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let flat = py
            .detach(|| {
                self.inner
                    .varadhan_landmark_table(p_exp.0, degree_normalize, eps_sim.0)
            })
            .map_err(core_error_to_py)?;
        let arr =
            Array2::from_shape_vec((self.inner.num_nodes(), self.inner.num_landmarks()), flat)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, arr))
    }

    /// Apply the full regularized DoG kernel operator to 1D `(N,)` or 2D `(N, C)` vectors.
    #[pyo3(signature = (l_sym, v), text_signature = "($self, l_sym, v)")]
    fn apply_batch<'py>(
        &self,
        py: Python<'py>,
        l_sym: &PyCsrMatrix,
        v: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (v_flat, nrows, ncols, is_2d) = extract_f64_1d_or_2d(v, "v")?;
        if nrows != self.inner.num_nodes() {
            return Err(PyValueError::new_err(format!(
                "v row count {nrows} does not match num_nodes ({})",
                self.inner.num_nodes()
            )));
        }
        let mat = l_sym.inner();
        let inner = &self.inner;
        let out = py
            .detach(|| -> CoreResult<Vec<f64>> {
                let mut out = vec![0.0_f64; v_flat.len()];
                let mut ws = ChebyshevWorkspace::with_capacity(v_flat.len());
                inner.apply_batch(mat, &v_flat, &mut out, ncols, &mut ws)?;
                Ok(out)
            })
            .map_err(core_error_to_py)?;
        if is_2d {
            let arr = Array2::from_shape_vec((nrows, ncols), out)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(PyArray2::from_owned_array(py, arr).into_any())
        } else {
            Ok(PyArray1::from_vec(py, out).into_any())
        }
    }

    /// Return a concise string representation of the `ChebyshevDogKernel`.
    fn __repr__(&self) -> String {
        format!(
            "ChebyshevDogKernel(num_nodes={}, num_landmarks={}, jl_dim={}, degree={})",
            self.inner.num_nodes(),
            self.inner.num_landmarks(),
            self.inner.jl_dim(),
            self.inner.degree()
        )
    }
}

/// Build Difference-of-Gaussians spectral filter terms `[(w1, t1), (w2, t2)]`.
#[pyfunction]
#[pyo3(
    signature = (t1, t2, alpha, beta_floor = StrictF64(0.0)),
    text_signature = "(t1, t2, alpha, beta_floor=0.0)"
)]
fn build_dog_terms(
    t1: StrictF64,
    t2: StrictF64,
    alpha: StrictF64,
    beta_floor: StrictF64,
) -> PyResult<Vec<(f64, f64)>> {
    linalg_build_dog_terms(t1.0, t2.0, alpha.0, beta_floor.0).map_err(core_error_to_py)
}

/// Build parameter-free R1 regularized DoG terms from spectral standard deviation `sigma_lambda`.
#[pyfunction]
#[pyo3(
    signature = (sigma_lambda, beta_floor = StrictF64(0.0)),
    text_signature = "(sigma_lambda, beta_floor=0.0)"
)]
fn build_regularized_dog_terms(
    sigma_lambda: StrictF64,
    beta_floor: StrictF64,
) -> PyResult<Vec<(f64, f64)>> {
    linalg_build_regularized_dog_terms(sigma_lambda.0, beta_floor.0).map_err(core_error_to_py)
}

/// Compute the adaptive Chebyshev polynomial degree for diffusion time `max_t`.
#[pyfunction]
#[pyo3(signature = (max_t), text_signature = "(max_t)")]
fn adaptive_chebyshev_degree(max_t: StrictF64) -> usize {
    linalg_adaptive_chebyshev_degree(max_t.0)
}

/// Compute the adaptive Chebyshev polynomial degree scaled for `[0, lambda_max]`.
#[pyfunction]
#[pyo3(
    signature = (max_t, lambda_max = StrictF64(2.0)),
    text_signature = "(max_t, lambda_max=2.0)"
)]
fn adaptive_chebyshev_degree_scaled(max_t: StrictF64, lambda_max: StrictF64) -> usize {
    linalg_adaptive_chebyshev_degree_scaled(max_t.0, lambda_max.0)
}

/// Compute Chebyshev expansion coefficients for spectral bandpass terms over `[0, lambda_max]`.
#[pyfunction]
#[pyo3(
    signature = (terms, lambda_max = StrictF64(2.0), degree = StrictUsize(40)),
    text_signature = "(terms, lambda_max=2.0, degree=40)"
)]
fn compute_chebyshev_bandpass_coefficients<'py>(
    py: Python<'py>,
    terms: &Bound<'_, PyAny>,
    lambda_max: StrictF64,
    degree: StrictUsize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let terms_vec = extract_filter_terms(terms)?;
    let coeffs = linalg_compute_chebyshev_bandpass_coefficients(&terms_vec, lambda_max.0, degree.0)
        .map_err(core_error_to_py)?;
    Ok(PyArray1::from_vec(py, coeffs))
}

type PsdProjectResult = (Vec<(f64, f64)>, f64, f64);

/// Project spectral filter `terms` onto the PSD cone, returning `(terms_psd, delta_psd, h_min)`.
#[pyfunction]
#[pyo3(
    signature = (terms, lambda_max = StrictF64(2.0), degree = StrictUsize(40), eps_psd = StrictF64(1e-4)),
    text_signature = "(terms, lambda_max=2.0, degree=40, eps_psd=1e-4)"
)]
fn project_psd_spectral_filter(
    terms: &Bound<'_, PyAny>,
    lambda_max: StrictF64,
    degree: StrictUsize,
    eps_psd: StrictF64,
) -> PyResult<PsdProjectResult> {
    let terms_vec = extract_filter_terms(terms)?;
    linalg_project_psd_spectral_filter(&terms_vec, lambda_max.0, degree.0, eps_psd.0)
        .map_err(core_error_to_py)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyChebyshevDogKernel>()?;
    m.add_function(wrap_pyfunction!(build_dog_terms, m)?)?;
    m.add_function(wrap_pyfunction!(build_regularized_dog_terms, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_chebyshev_degree, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_chebyshev_degree_scaled, m)?)?;
    m.add_function(wrap_pyfunction!(
        compute_chebyshev_bandpass_coefficients,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(project_psd_spectral_filter, m)?)?;
    Ok(())
}
