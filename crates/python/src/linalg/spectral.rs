//! Spectral initialization, normalization, and Laplacian eigenpair bindings.

use egraph::core::{CoreError, CoreResult};
use egraph::linalg::{
    RdMds, koren_spectral_init as linalg_koren, normalize_layout_2d as linalg_normalize_layout_2d,
    robust_normalize_layout_2d as linalg_robust_normalize_layout_2d,
    select_maxmin_sp_pivots as linalg_select_maxmin_sp_pivots,
    smallest_sym_laplacian_eigenpairs as linalg_smallest_eigenpairs,
    symmetric_eigen_jacobi as linalg_symmetric_eigen_jacobi,
};
use egraph::neighborhood::{
    compute_diffusion_embedding as nb_compute_diffusion_embedding,
    compute_resistance_embedding as nb_compute_resistance_embedding,
};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array_extract::extract_f64_2d;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, ffi_error_to_py,
};
use crate::layout_buffers::{Layout2dInner, PyLayout2d};
use crate::layout_nd::PyLayoutNd;

pub(super) fn validate_koren_params(
    tol: f64,
    max_iterations: usize,
    target_std: Option<f64>,
) -> CoreResult<()> {
    if !tol.is_finite() {
        return Err(CoreError::non_finite("koren_spectral_init::tol"));
    }
    if tol <= 0.0 {
        return Err(CoreError::invalid_input("tol must be strictly positive"));
    }
    if max_iterations == 0 {
        return Err(CoreError::invalid_input("max_iterations must be >= 1"));
    }
    if let Some(t_std) = target_std {
        if !t_std.is_finite() {
            return Err(CoreError::non_finite("koren_spectral_init::target_std"));
        }
        if t_std <= 0.0 {
            return Err(CoreError::invalid_input(
                "target_std must be strictly positive",
            ));
        }
    }
    Ok(())
}

/// Select well-spaced shortest-path pivots via Core-First MaxMin BFS.
#[pyfunction]
#[pyo3(
    signature = (graph, num_pivots = StrictUsize(50), *, seed = StrictU64(42), min_degree = StrictUsize(2)),
    text_signature = "(graph, num_pivots=50, *, seed=42, min_degree=2)"
)]
fn select_maxmin_sp_pivots<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    num_pivots: StrictUsize,
    seed: StrictU64,
    min_degree: StrictUsize,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let g = graph.inner();
    let pivots = py
        .detach(|| linalg_select_maxmin_sp_pivots(&*g, num_pivots.0, seed.0, min_degree.0))
        .map_err(core_error_to_py)?;
    Ok(PyArray1::from_vec(py, pivots))
}

/// Compute Koren's degree-normalized 2D spectral initialization from `L_sym`.
#[pyfunction]
#[pyo3(
    signature = (graph, *, target_std = Some(StrictF64(1.0)), tol = StrictF64(1e-4), max_iterations = StrictUsize(300), seed = StrictU64(42), dtype = ScalarDtype::F64),
    text_signature = "(graph, *, target_std=1.0, tol=1e-4, max_iterations=300, seed=42, dtype='float64')"
)]
fn koren_spectral_init(
    py: Python<'_>,
    graph: &PyCsrGraph,
    target_std: Option<StrictF64>,
    tol: StrictF64,
    max_iterations: StrictUsize,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PyLayout2d> {
    let t_std = target_std.map(|v| v.0);
    validate_koren_params(tol.0, max_iterations.0, t_std).map_err(core_error_to_py)?;
    let g = graph.inner();
    py.detach(|| match dtype {
        ScalarDtype::F64 => linalg_koren::<f64, f64>(&*g, t_std, tol.0, max_iterations.0, seed.0)
            .map(|l| PyLayout2d::from_f64_layout(&l)),
        ScalarDtype::F32 => linalg_koren::<f64, f32>(&*g, t_std, tol.0, max_iterations.0, seed.0)
            .map(|l| PyLayout2d::from_f32_layout(&l)),
    })
    .map_err(core_error_to_py)
}

type EigenpairsReturn<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    bool,
);

/// Compute the smallest `n_modes` nontrivial eigenpairs of `L_sym`.
#[pyfunction]
#[pyo3(
    signature = (graph, n_modes = StrictUsize(16), *, tol = StrictF64(1e-4), max_iterations = StrictUsize(300), seed = StrictU64(42)),
    text_signature = "(graph, n_modes=16, *, tol=1e-4, max_iterations=300, seed=42)"
)]
fn smallest_laplacian_eigenpairs<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    n_modes: StrictUsize,
    tol: StrictF64,
    max_iterations: StrictUsize,
    seed: StrictU64,
) -> PyResult<EigenpairsReturn<'py>> {
    let g = graph.inner();
    let res = py
        .detach(|| linalg_smallest_eigenpairs(&*g, n_modes.0, tol.0, max_iterations.0, seed.0))
        .map_err(core_error_to_py)?;
    let vectors_arr = Array2::from_shape_vec((res.num_nodes, res.num_modes), res.vectors)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        PyArray1::from_vec(py, res.lambdas),
        PyArray2::from_owned_array(py, vectors_arr),
        PyArray1::from_vec(py, res.inv_sqrt_deg),
        res.used_dense_fallback,
    ))
}

/// Compute a `dims`-dimensional `LayoutNd` via zero-guarded Resistance-distance MDS (`RdMds`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (graph, dims = StrictUsize(2), *, eig_eps = StrictF64(1e-10), tol = StrictF64(1e-6), max_iterations = StrictUsize(1000), seed = StrictU64(42), degree_normalize = true, dtype = ScalarDtype::F64),
    text_signature = "(graph, dims=2, *, eig_eps=1e-10, tol=1e-6, max_iterations=1000, seed=42, degree_normalize=True, dtype='float64')"
)]
fn rd_mds(
    py: Python<'_>,
    graph: &PyCsrGraph,
    dims: StrictUsize,
    eig_eps: StrictF64,
    tol: StrictF64,
    max_iterations: StrictUsize,
    seed: StrictU64,
    degree_normalize: bool,
    dtype: ScalarDtype,
) -> PyResult<PyLayoutNd> {
    let g = graph.inner();
    let cfg = RdMds::with_dims(dims.0)
        .eig_eps(eig_eps.0)
        .tol(tol.0)
        .max_iters(max_iterations.0)
        .seed(seed.0)
        .degree_normalize(degree_normalize);
    py.detach(|| match dtype {
        ScalarDtype::F64 => cfg.embedding::<f64, f64>(&*g).map(PyLayoutNd::from_f64_nd),
        ScalarDtype::F32 => cfg.embedding::<f64, f32>(&*g).map(PyLayoutNd::from_f32_nd),
    })
    .map_err(core_error_to_py)
}

/// Sanitize non-finite entries, center, and normalize a `Layout2d` in place to `target_std`.
#[pyfunction]
#[pyo3(
    signature = (layout, *, target_std = StrictF64(1.0), seed = StrictU64(42)),
    text_signature = "(layout, *, target_std=1.0, seed=42)"
)]
fn normalize_layout_2d(
    py: Python<'_>,
    layout: &Bound<'_, PyLayout2d>,
    target_std: StrictF64,
    seed: StrictU64,
) -> PyResult<()> {
    let t_std = target_std.0;
    let seed_val = seed.0;
    let mut borrowed = layout.try_borrow_mut()?;
    match borrowed.inner_mut() {
        Layout2dInner::F64(buf) => {
            let mut soa = buf.to_layout2d();
            py.detach(|| linalg_normalize_layout_2d(&mut soa, t_std, seed_val))
                .map_err(core_error_to_py)?;
            buf.copy_from_layout2d(&soa).map_err(ffi_error_to_py)
        }
        Layout2dInner::F32(buf) => {
            let mut soa = buf.to_layout2d();
            py.detach(|| linalg_normalize_layout_2d(&mut soa, t_std, seed_val))
                .map_err(core_error_to_py)?;
            buf.copy_from_layout2d(&soa).map_err(ffi_error_to_py)
        }
    }
}

/// Median-center and Gaussian-calibrated radial quantile-scale a `Layout2d` in place.
#[pyfunction]
#[pyo3(
    signature = (layout, *, target_std = StrictF64(1.0), quantile = StrictF64(0.95), seed = StrictU64(42)),
    text_signature = "(layout, *, target_std=1.0, quantile=0.95, seed=42)"
)]
fn robust_normalize_layout_2d(
    py: Python<'_>,
    layout: &Bound<'_, PyLayout2d>,
    target_std: StrictF64,
    quantile: StrictF64,
    seed: StrictU64,
) -> PyResult<()> {
    let t_std = target_std.0;
    let q_val = quantile.0;
    let seed_val = seed.0;
    let mut borrowed = layout.try_borrow_mut()?;
    match borrowed.inner_mut() {
        Layout2dInner::F64(buf) => {
            let mut soa = buf.to_layout2d();
            py.detach(|| linalg_robust_normalize_layout_2d(&mut soa, t_std, q_val, seed_val))
                .map_err(core_error_to_py)?;
            buf.copy_from_layout2d(&soa).map_err(ffi_error_to_py)
        }
        Layout2dInner::F32(buf) => {
            let mut soa = buf.to_layout2d();
            py.detach(|| linalg_robust_normalize_layout_2d(&mut soa, t_std, q_val, seed_val))
                .map_err(core_error_to_py)?;
            buf.copy_from_layout2d(&soa).map_err(ffi_error_to_py)
        }
    }
}

type ResistanceEmbeddingPy<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>);
type DiffusionEmbeddingPy<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, f64);

/// Compute effective-resistance spectral embedding `(coords, lambdas_kept)`.
#[pyfunction]
#[pyo3(
    signature = (graph, dim = StrictUsize(16), *, tol = StrictF64(1e-4), max_iterations = StrictUsize(300), seed = StrictU64(42), eig_eps = StrictF64(1e-10)),
    text_signature = "(graph, dim=16, *, tol=1e-4, max_iterations=300, seed=42, eig_eps=1e-10)"
)]
fn compute_resistance_embedding<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    dim: StrictUsize,
    tol: StrictF64,
    max_iterations: StrictUsize,
    seed: StrictU64,
    eig_eps: StrictF64,
) -> PyResult<ResistanceEmbeddingPy<'py>> {
    let g = graph.inner();
    let n = g.num_nodes();
    let (coords, lambdas) = py
        .detach(|| {
            nb_compute_resistance_embedding(&*g, dim.0, tol.0, max_iterations.0, seed.0, eig_eps.0)
        })
        .map_err(core_error_to_py)?;
    let k = lambdas.len();
    let arr =
        Array2::from_shape_vec((n, k), coords).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        PyArray2::from_owned_array(py, arr),
        PyArray1::from_vec(py, lambdas),
    ))
}

/// Compute heat-diffusion spectral embedding `(coords, lambdas, t_used)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (graph, dim = StrictUsize(16), *, p = StrictF64(0.5), t = None, tol = StrictF64(1e-4), max_iterations = StrictUsize(300), seed = StrictU64(42)),
    text_signature = "(graph, dim=16, *, p=0.5, t=None, tol=1e-4, max_iterations=300, seed=42)"
)]
fn compute_diffusion_embedding<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    dim: StrictUsize,
    p: StrictF64,
    t: Option<StrictF64>,
    tol: StrictF64,
    max_iterations: StrictUsize,
    seed: StrictU64,
) -> PyResult<DiffusionEmbeddingPy<'py>> {
    let g = graph.inner();
    let n = g.num_nodes();
    let t_opt = t.map(|v| v.0);
    let (coords, lambdas, t_used) = py
        .detach(|| {
            nb_compute_diffusion_embedding(&*g, dim.0, p.0, t_opt, tol.0, max_iterations.0, seed.0)
        })
        .map_err(core_error_to_py)?;
    let k = lambdas.len();
    let arr =
        Array2::from_shape_vec((n, k), coords).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        PyArray2::from_owned_array(py, arr),
        PyArray1::from_vec(py, lambdas),
        t_used,
    ))
}

type JacobiEigenPy<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>);

/// Compute all eigenpairs `(eigenvalues, eigenvectors)` of a symmetric `(N, N)` matrix
/// via cyclic Jacobi rotations in `f64`, sorted in ascending eigenvalue order.
#[pyfunction]
#[pyo3(
    signature = (matrix, *, max_sweeps = StrictUsize(120), tol = StrictF64(1e-12)),
    text_signature = "(matrix, *, max_sweeps=120, tol=1e-12)"
)]
fn symmetric_eigen_jacobi<'py>(
    py: Python<'py>,
    matrix: &Bound<'_, PyAny>,
    max_sweeps: StrictUsize,
    tol: StrictF64,
) -> PyResult<JacobiEigenPy<'py>> {
    let (flat, nrows, ncols) = extract_f64_2d(matrix, "matrix")?;
    if nrows != ncols {
        return Err(PyValueError::new_err(format!(
            "matrix must be a square (N, N) matrix, got ({nrows}, {ncols})"
        )));
    }
    let (lambdas, vectors) = py
        .detach(|| linalg_symmetric_eigen_jacobi(&flat, nrows, max_sweeps.0, tol.0))
        .map_err(core_error_to_py)?;
    let vectors_arr = Array2::from_shape_vec((nrows, nrows), vectors)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        PyArray1::from_vec(py, lambdas),
        PyArray2::from_owned_array(py, vectors_arr),
    ))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(select_maxmin_sp_pivots, m)?)?;
    m.add_function(wrap_pyfunction!(koren_spectral_init, m)?)?;
    m.add_function(wrap_pyfunction!(rd_mds, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_layout_2d, m)?)?;
    m.add_function(wrap_pyfunction!(robust_normalize_layout_2d, m)?)?;
    m.add_function(wrap_pyfunction!(smallest_laplacian_eigenpairs, m)?)?;
    m.add_function(wrap_pyfunction!(symmetric_eigen_jacobi, m)?)?;
    m.add_function(wrap_pyfunction!(compute_resistance_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(compute_diffusion_embedding, m)?)?;
    Ok(())
}
