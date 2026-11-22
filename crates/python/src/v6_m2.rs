//! v6 linear algebra, spectral diffusion, neighborhood graph, adaptive perplexity,
//! fuzzy simplicial set, and regularized DoG Varadhan landmark bindings backed by
//! `egraph-linalg` and `egraph-neighborhood`.

use egraph_core::{CoreError, CoreResult, Layout2d};
use egraph_ffi::InterleavedLayout2d;
use egraph_linalg::{
    ChebyshevWorkspace, PivotMdsConfig, SpectralMomentStats, adaptive_chebyshev_degree,
    build_dog_terms, compute_chebyshev_bandpass_coefficients, koren_spectral_init as linalg_koren,
    pivot_mds as linalg_pivot_mds, project_psd_spectral_filter,
    select_maxmin_sp_pivots as linalg_select_maxmin_sp_pivots,
    smallest_sym_laplacian_eigenpairs as linalg_smallest_eigenpairs,
    spectral_moment_stats as linalg_spectral_moment_stats, symmetric_normalized_laplacian,
};
use egraph_neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, DogVaradhanStats, HybridKnnConfig, KnnGraph, KnnStats,
    PerplexityConfig, SparseProbabilities, SymmetrizationMode,
    build_hybrid_adaptive_knn as nb_build_hybrid_adaptive_knn,
    build_pure_2hop_knn as nb_build_pure_2hop_knn,
    compute_2hop_neighborhood_sizes as nb_compute_2hop_neighborhood_sizes,
    compute_fuzzy_simplicial_set as nb_compute_fuzzy_simplicial_set,
    compute_horvitz_thompson_gamma as nb_compute_horvitz_thompson_gamma,
    compute_knn_probabilities as nb_compute_knn_probabilities,
    compute_median_2hop_perplexity as nb_compute_median_2hop_perplexity,
    compute_nodewise_2hop_perplexities as nb_compute_nodewise_2hop_perplexities,
    compute_regularized_dog_varadhan_landmarks as nb_compute_regularized_dog_varadhan_landmarks,
    extract_1hop_2hop_csr as nb_extract_1hop_2hop_csr, find_ab_params as nb_find_ab_params,
    partial_bfs_knn as nb_partial_bfs_knn,
};
use numpy::ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{
    PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

use super::{
    Layout2dInner, PyCsrGraph, PyLayout2d, ScalarDtype, StrictF64, StrictU64, StrictUsize,
    core_error_to_py, extract_non_negative_usize, extract_optional_weights_1d, extract_u32_1d,
    is_non_sequence_container,
};

fn convert_int_slice_to_usize<T, F>(view: ArrayView1<'_, T>, mut convert: F) -> PyResult<Vec<usize>>
where
    T: Copy,
    F: FnMut(T) -> PyResult<usize>,
{
    let mut out = Vec::with_capacity(view.len());
    for &val in view.iter() {
        out.push(convert(val)?);
    }
    Ok(out)
}

fn extract_usize_1d(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<usize>> {
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        if untyped.ndim() != 1 {
            return Err(PyValueError::new_err(format!(
                "{name} must be a 1D array, got {}D",
                untyped.ndim()
            )));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, usize>>() {
            return Ok(match arr.as_slice() {
                Ok(slice) => slice.to_vec(),
                Err(_) => arr.as_array().iter().copied().collect(),
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u64>>() {
            return convert_int_slice_to_usize(arr.as_array(), |v| {
                usize::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for usize"))
                })
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i64>>() {
            return convert_int_slice_to_usize(arr.as_array(), |v| {
                usize::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for usize"))
                })
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u32>>() {
            return Ok(arr.as_array().iter().map(|&v| v as usize).collect());
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i32>>() {
            return convert_int_slice_to_usize(arr.as_array(), |v| {
                usize::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for usize"))
                })
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, isize>>() {
            return convert_int_slice_to_usize(arr.as_array(), |v| {
                usize::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for usize"))
                })
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u16>>() {
            return Ok(arr.as_array().iter().map(|&v| usize::from(v)).collect());
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u8>>() {
            return Ok(arr.as_array().iter().map(|&v| usize::from(v)).collect());
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i16>>() {
            return convert_int_slice_to_usize(arr.as_array(), |v| {
                usize::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for usize"))
                })
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i8>>() {
            return convert_int_slice_to_usize(arr.as_array(), |v| {
                usize::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for usize"))
                })
            });
        }
        return Err(PyTypeError::new_err(format!(
            "{name} must be a 1D integer NumPy array"
        )));
    }

    if is_non_sequence_container(obj) {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a 1D integer array or sequence"
        )));
    }
    let iter = obj.try_iter().map_err(|_| {
        PyTypeError::new_err(format!("{name} must be a 1D integer array or sequence"))
    })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item = item_res?;
        out.push(extract_non_negative_usize(&item, name)?);
    }
    Ok(out)
}

fn extract_f64_1d(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    if obj.is_none() {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a 1D numeric array or sequence, not None"
        )));
    }
    match extract_optional_weights_1d(Some(obj), name)? {
        Some(v) => Ok(v),
        None => Err(PyTypeError::new_err(format!(
            "{name} must be a 1D numeric array or sequence"
        ))),
    }
}

fn extract_perplexities(obj: &Bound<'_, PyAny>, num_nodes: usize) -> PyResult<Vec<f64>> {
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "perplexities must be a 1D numeric array or scalar float, not bool",
        ));
    }
    if obj.cast::<PyUntypedArray>().is_err()
        && let Ok(scalar) = obj.extract::<f64>()
    {
        if !scalar.is_finite() {
            return Err(PyValueError::new_err(
                "non-finite floating-point value in perplexities",
            ));
        }
        if scalar <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "perplexities must be > 0, got {scalar}"
            )));
        }
        return Ok(vec![scalar; num_nodes]);
    }
    extract_f64_1d(obj, "perplexities")
}

fn convert_numeric_view2_to_f64<T, F>(
    view: ArrayView2<'_, T>,
    name: &str,
    mut to_f64: F,
) -> PyResult<(Vec<f64>, usize, usize)>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    let nrows = view.nrows();
    let ncols = view.ncols();
    let mut flat = Vec::with_capacity(nrows.saturating_mul(ncols));
    for (r_idx, row) in view.rows().into_iter().enumerate() {
        for &val in row.iter() {
            let f = to_f64(val);
            if !f.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "{name} contains non-finite value at row {r_idx}"
                )));
            }
            flat.push(f);
        }
    }
    Ok((flat, nrows, ncols))
}

fn extract_numeric_ndarray_2d(
    obj: &Bound<'_, PyAny>,
    untyped: &Bound<'_, PyUntypedArray>,
    name: &str,
) -> PyResult<(Vec<f64>, usize, usize)> {
    if untyped.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "{name} must be a 2D numeric array, got {}D",
            untyped.ndim()
        )));
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f64>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, |v| v);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f32>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, f64::from);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i64>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, |v| v as f64);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i32>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, f64::from);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u64>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, |v| v as f64);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u32>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, f64::from);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, isize>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, |v| v as f64);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, usize>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, |v| v as f64);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i16>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, f64::from);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u16>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, f64::from);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i8>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, f64::from);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u8>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, f64::from);
    }
    Err(PyTypeError::new_err(format!(
        "{name} NumPy array must have a numeric dtype"
    )))
}

fn extract_f64_2d(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<(Vec<f64>, usize, usize)> {
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        return extract_numeric_ndarray_2d(obj, untyped, name);
    }
    if is_non_sequence_container(obj) {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a 2D numeric array or sequence of sequences"
        )));
    }
    let outer = obj.try_iter().map_err(|_| {
        PyTypeError::new_err(format!(
            "{name} must be a 2D numeric array or sequence of sequences"
        ))
    })?;
    let mut flat = Vec::new();
    let mut nrows = 0usize;
    let mut ncols: Option<usize> = None;

    for row_res in outer {
        let row = row_res?;
        if is_non_sequence_container(&row) {
            return Err(PyTypeError::new_err(format!(
                "{name} rows must be numeric sequences"
            )));
        }
        let inner = row
            .try_iter()
            .map_err(|_| PyTypeError::new_err(format!("{name} rows must be numeric sequences")))?;
        let mut row_len = 0usize;
        for elem_res in inner {
            let elem = elem_res?;
            if elem.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(format!(
                    "{name} entries must be numeric, not bool"
                )));
            }
            let val = elem
                .extract::<f64>()
                .map_err(|_| PyTypeError::new_err(format!("{name} entries must be numeric")))?;
            if !val.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "{name} contains non-finite value at row {nrows}"
                )));
            }
            flat.push(val);
            row_len += 1;
        }
        match ncols {
            None => ncols = Some(row_len),
            Some(expected) if row_len != expected => {
                return Err(PyValueError::new_err(format!(
                    "{name} has ragged row {nrows}: expected {expected} columns, got {row_len}"
                )));
            }
            Some(_) => {}
        }
        nrows += 1;
    }

    Ok((flat, nrows, ncols.unwrap_or(0)))
}

fn extract_f64_1d_or_2d(
    obj: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(Vec<f64>, usize, usize, bool)> {
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        return match untyped.ndim() {
            1 => {
                let vec = extract_f64_1d(obj, name)?;
                for (idx, &val) in vec.iter().enumerate() {
                    if !val.is_finite() {
                        return Err(PyValueError::new_err(format!(
                            "{name} contains non-finite value at index {idx}"
                        )));
                    }
                }
                let n = vec.len();
                Ok((vec, n, 1, false))
            }
            2 => {
                let (flat, nrows, ncols) = extract_numeric_ndarray_2d(obj, untyped, name)?;
                Ok((flat, nrows, ncols, true))
            }
            other => Err(PyValueError::new_err(format!(
                "{name} must be a 1D (N,) or 2D (N, C) array, got {other}D"
            ))),
        };
    }

    if is_non_sequence_container(obj) {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a 1D or 2D numeric array or sequence"
        )));
    }
    let iter = obj.try_iter().map_err(|_| {
        PyTypeError::new_err(format!(
            "{name} must be a 1D or 2D numeric array or sequence"
        ))
    })?;
    let mut items = Vec::new();
    for item_res in iter {
        items.push(item_res?);
    }
    let Some(first) = items.first() else {
        return Ok((Vec::new(), 0, 1, false));
    };
    if !is_non_sequence_container(first) && first.try_iter().is_ok() {
        let (flat, nrows, ncols) = extract_f64_2d(obj, name)?;
        return Ok((flat, nrows, ncols, true));
    }

    let mut out = Vec::with_capacity(items.len());
    for (idx, item) in items.iter().enumerate() {
        if item.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(format!(
                "{name} entries must be numeric, not bool"
            )));
        }
        let val = item
            .extract::<f64>()
            .map_err(|_| PyTypeError::new_err(format!("{name} entries must be numeric")))?;
        if !val.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{name} contains non-finite value at index {idx}"
            )));
        }
        out.push(val);
    }
    let n = out.len();
    Ok((out, n, 1, false))
}

#[inline]
fn wrap_layout_f64(layout: Layout2d<f64>) -> PyLayout2d {
    PyLayout2d {
        inner: Layout2dInner::F64(InterleavedLayout2d::from_layout2d(&layout)),
    }
}

#[inline]
fn wrap_layout_f32(layout: Layout2d<f32>) -> PyLayout2d {
    PyLayout2d {
        inner: Layout2dInner::F32(InterleavedLayout2d::from_layout2d(&layout)),
    }
}

fn spectral_moment_fields(stats: &SpectralMomentStats) -> [(&'static str, f64); 5] {
    [
        ("sigma_lambda_sq", stats.sigma_lambda_sq),
        ("sigma_lambda", stats.sigma_lambda),
        ("deg_cv", stats.deg_cv),
        ("mean_degree", stats.mean_degree),
        ("leaf_fraction", stats.leaf_fraction),
    ]
}

fn knn_stats_float_fields(stats: &KnnStats) -> [(&'static str, f64); 3] {
    [
        ("avg_deg1", stats.avg_deg1),
        ("avg_deg2", stats.avg_deg2),
        ("avg_row_k", stats.avg_row_k),
    ]
}

fn knn_stats_int_fields(stats: &KnnStats) -> [(&'static str, usize); 9] {
    [
        ("min_row_k", stats.min_row_k),
        ("max_row_k", stats.max_row_k),
        ("nnz", stats.nnz),
        ("n_1hop", stats.n_1hop),
        ("n_2hop", stats.n_2hop),
        ("n_truncated_1hop", stats.n_truncated_1hop),
        ("n_truncated_2hop", stats.n_truncated_2hop),
        ("n_filled_residual", stats.n_filled_residual),
        ("n_pure_2hop_rows", stats.n_pure_2hop_rows),
    ]
}

fn knn_stats_to_py_dict(py: Python<'_>, stats: &KnnStats) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in knn_stats_float_fields(stats) {
        dict.set_item(k, v)?;
    }
    for (k, v) in knn_stats_int_fields(stats) {
        dict.set_item(k, v)?;
    }
    Ok(dict.unbind())
}

fn validate_koren_params(tol: f64, max_iters: usize, target_std: Option<f64>) -> CoreResult<()> {
    if !tol.is_finite() {
        return Err(CoreError::non_finite("koren_spectral_init::tol"));
    }
    if tol <= 0.0 {
        return Err(CoreError::invalid_input("tol must be strictly positive"));
    }
    if max_iters == 0 {
        return Err(CoreError::invalid_input("max_iters must be >= 1"));
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

fn resolve_dog_r1_defaults(
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

fn validate_dog_jl_params(degree: usize, jl_dim: usize, eps_psd: f64) -> CoreResult<()> {
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

/// Validated Compressed Sparse Row (CSR) `k`-nearest-neighbor graph in `float64`.
#[pyclass(name = "KnnGraph", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PyKnnGraph {
    inner: KnnGraph,
}

#[pymethods]
impl PyKnnGraph {
    /// Construct a validated `KnnGraph` from raw CSR arrays `(num_nodes, indptr, indices, distances)`.
    #[staticmethod]
    #[pyo3(
        signature = (num_nodes, indptr, indices, distances),
        text_signature = "(num_nodes, indptr, indices, distances)"
    )]
    fn from_csr(
        py: Python<'_>,
        num_nodes: StrictUsize,
        indptr: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        distances: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let indptr_vec = extract_usize_1d(indptr, "indptr")?;
        let indices_vec = extract_u32_1d(indices, "indices")?;
        let distances_vec = extract_f64_1d(distances, "distances")?;

        let inner = py
            .detach(|| KnnGraph::new_validated(num_nodes.0, indptr_vec, indices_vec, distances_vec))
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Number of vertices in the `k`-NN graph.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Total number of stored directed neighbor entries.
    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Return a `uintp` NumPy copy of the CSR row pointer array of length `num_nodes + 1`.
    fn indptr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        PyArray1::from_slice(py, self.inner.indptr())
    }

    /// Return a `uint32` NumPy copy of the CSR neighbor index array.
    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, self.inner.indices())
    }

    /// Return a `float64` NumPy copy of neighbor distances aligned with `indices()`.
    fn distances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.distances())
    }

    /// Return the number of vertices in the `k`-NN graph.
    fn __len__(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Return a concise string representation of the `KnnGraph`.
    fn __repr__(&self) -> String {
        format!(
            "KnnGraph(num_nodes={}, nnz={})",
            self.inner.num_nodes(),
            self.inner.nnz()
        )
    }
}

/// Canonical sorted symmetric CSR joint probability matrix `P`.
#[pyclass(name = "SparseProbabilities", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PySparseProbabilities {
    inner: SparseProbabilities,
}

#[pymethods]
impl PySparseProbabilities {
    /// Number of vertices in the probability matrix.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Total number of stored symmetric probability entries.
    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Return a `uintp` NumPy copy of the CSR row pointer array of length `num_nodes + 1`.
    fn indptr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        PyArray1::from_slice(py, &self.inner.indptr)
    }

    /// Return a `uint32` NumPy copy of the sorted CSR neighbor column index array.
    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, &self.inner.indices)
    }

    /// Return a `float64` NumPy copy of joint probabilities aligned with `indices()`.
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.values)
    }

    /// Return the number of vertices in the probability matrix.
    fn __len__(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Return a concise string representation of the `SparseProbabilities`.
    fn __repr__(&self) -> String {
        format!(
            "SparseProbabilities(num_nodes={}, nnz={})",
            self.inner.num_nodes(),
            self.inner.nnz()
        )
    }
}

/// Regularized DoG Varadhan landmark distance table (plus optional coarse diffusive scaffold)
/// supporting fast ALT lower-bound distance queries.
#[pyclass(name = "DogVaradhanLandmarks", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PyDogVaradhanLandmarks {
    inner: DogVaradhanLandmarks,
    stats: DogVaradhanStats,
}

#[pymethods]
impl PyDogVaradhanLandmarks {
    /// Number of vertices covered by the landmark table.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Number of landmark columns in the table.
    #[getter]
    fn num_landmarks(&self) -> usize {
        self.inner.num_landmarks()
    }

    /// Return a `uint32` NumPy copy of selected landmark vertex IDs.
    fn landmarks<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, self.inner.landmarks())
    }

    /// Return the `(num_nodes, num_landmarks)` `float64` DoG Varadhan distance table.
    fn table<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = Array2::from_shape_vec(
            (self.inner.num_nodes(), self.inner.num_landmarks()),
            self.inner.d_dog().to_vec(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, arr))
    }

    /// Return the `(num_nodes, num_landmarks)` `float64` coarse heat scaffold table, or `None`.
    fn coarse_table<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        let Some(dc) = self.inner.d_coarse() else {
            return Ok(None);
        };
        let arr = Array2::from_shape_vec(
            (self.inner.num_nodes(), self.inner.num_landmarks()),
            dc.to_vec(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Some(PyArray2::from_owned_array(py, arr)))
    }

    /// Effective short-range DoG diffusion time `t1`.
    #[getter]
    fn t1(&self) -> f64 {
        self.stats.t1
    }

    /// Effective long-range DoG diffusion time `t2`.
    #[getter]
    fn t2(&self) -> f64 {
        self.stats.t2
    }

    /// Effective DoG attenuation parameter `alpha`.
    #[getter]
    fn alpha(&self) -> f64 {
        self.stats.alpha
    }

    /// PSD shift `delta_psd` applied during R2 spectral projection.
    #[getter]
    fn delta_psd(&self) -> f64 {
        self.stats.delta_psd
    }

    /// Effective rank-1 positivity floor `beta_eff` actually applied.
    #[getter]
    fn beta_eff(&self) -> f64 {
        self.stats.beta_eff
    }

    /// Parameter-free automatic positivity floor `beta_auto`.
    #[getter]
    fn beta_auto(&self) -> f64 {
        self.stats.beta_auto
    }

    /// Whether the coarse diffusive heat scaffold was computed and attached.
    #[getter]
    fn used_coarse_scaffold(&self) -> bool {
        self.stats.used_coarse_scaffold
    }

    /// Evaluate landmark ALT lower-bound distances for pairs `(u[i], v[i])`.
    ///
    /// Releases the Python GIL during computation.
    #[pyo3(signature = (u, v), text_signature = "($self, u, v)")]
    fn alt_distances<'py>(
        &self,
        py: Python<'py>,
        u: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let u_vec = extract_u32_1d(u, "u")?;
        let v_vec = extract_u32_1d(v, "v")?;
        let out = py
            .detach(|| self.inner.alt_pair_distances(&u_vec, &v_vec))
            .map_err(core_error_to_py)?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// Return the number of vertices in the landmark table.
    fn __len__(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Return a concise string representation of the `DogVaradhanLandmarks`.
    fn __repr__(&self) -> String {
        format!(
            "DogVaradhanLandmarks(num_nodes={}, num_landmarks={}, used_coarse_scaffold={})",
            self.inner.num_nodes(),
            self.inner.num_landmarks(),
            if self.stats.used_coarse_scaffold {
                "True"
            } else {
                "False"
            }
        )
    }
}

/// Select well-spaced shortest-path pivots via Core-First MaxMin BFS.
///
/// Releases the Python GIL during computation.
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
        .detach(|| linalg_select_maxmin_sp_pivots(g, num_pivots.0, seed.0, min_degree.0))
        .map_err(core_error_to_py)?;
    Ok(PyArray1::from_vec(py, pivots))
}

/// Compute a 2D `Layout2d` via Core-First MaxMin PivotMDS with optional isotropic std normalization.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, *, num_pivots = StrictUsize(50), seed = StrictU64(42), min_degree = StrictUsize(2), target_std = Some(StrictF64(1.0)), dtype = ScalarDtype::F64),
    text_signature = "(graph, *, num_pivots=50, seed=42, min_degree=2, target_std=1.0, dtype='float64')"
)]
fn pivot_mds_init(
    py: Python<'_>,
    graph: &PyCsrGraph,
    num_pivots: StrictUsize,
    seed: StrictU64,
    min_degree: StrictUsize,
    target_std: Option<StrictF64>,
    dtype: ScalarDtype,
) -> PyResult<PyLayout2d> {
    let cfg = PivotMdsConfig {
        num_pivots: num_pivots.0,
        min_degree: min_degree.0,
        seed: seed.0,
        target_std: target_std.map(|v| v.0),
    };
    let g = graph.inner();
    py.detach(|| match dtype {
        ScalarDtype::F64 => linalg_pivot_mds::<f64, f64>(g, &cfg).map(wrap_layout_f64),
        ScalarDtype::F32 => linalg_pivot_mds::<f64, f32>(g, &cfg).map(wrap_layout_f32),
    })
    .map_err(core_error_to_py)
}

/// Compute Koren's degree-normalized 2D spectral initialization from the smallest nontrivial
/// eigenvectors of `L_sym`, with optional isotropic std normalization.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, *, target_std = Some(StrictF64(1.0)), tol = StrictF64(1e-4), max_iters = StrictUsize(300), seed = StrictU64(42), dtype = ScalarDtype::F64),
    text_signature = "(graph, *, target_std=1.0, tol=1e-4, max_iters=300, seed=42, dtype='float64')"
)]
fn koren_spectral_init(
    py: Python<'_>,
    graph: &PyCsrGraph,
    target_std: Option<StrictF64>,
    tol: StrictF64,
    max_iters: StrictUsize,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PyLayout2d> {
    let t_std = target_std.map(|v| v.0);
    validate_koren_params(tol.0, max_iters.0, t_std).map_err(core_error_to_py)?;
    let g = graph.inner();
    py.detach(|| match dtype {
        ScalarDtype::F64 => {
            linalg_koren::<f64, f64>(g, t_std, tol.0, max_iters.0, seed.0).map(wrap_layout_f64)
        }
        ScalarDtype::F32 => {
            linalg_koren::<f64, f32>(g, t_std, tol.0, max_iters.0, seed.0).map(wrap_layout_f32)
        }
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
///
/// Returns `(lambdas, vectors, inv_sqrt_deg, used_dense_fallback)`.
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, n_modes = StrictUsize(16), *, tol = StrictF64(1e-4), max_iters = StrictUsize(300), seed = StrictU64(42)),
    text_signature = "(graph, n_modes=16, *, tol=1e-4, max_iters=300, seed=42)"
)]
fn smallest_laplacian_eigenpairs<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    n_modes: StrictUsize,
    tol: StrictF64,
    max_iters: StrictUsize,
    seed: StrictU64,
) -> PyResult<EigenpairsReturn<'py>> {
    let g = graph.inner();
    let res = py
        .detach(|| linalg_smallest_eigenpairs(g, n_modes.0, tol.0, max_iters.0, seed.0))
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

/// Compute exact O(|E|) spectral variance and degree summary statistics of `L_sym`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph), text_signature = "(graph)")]
fn spectral_moment_stats(py: Python<'_>, graph: &PyCsrGraph) -> PyResult<Py<PyDict>> {
    let g = graph.inner();
    let stats = py.detach(|| linalg_spectral_moment_stats(g));
    let dict = PyDict::new(py);
    for (k, v) in spectral_moment_fields(&stats) {
        dict.set_item(k, v)?;
    }
    Ok(dict.unbind())
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
        return Err(PyValueError::new_err(format!(
            "vectors row count {nrows} does not match graph.num_nodes ({n})"
        )));
    }
    let deg_opt = degree.map(|d| d.0);
    let t_val = t.0;
    let lmax_val = lambda_max.0;

    let out = py
        .detach(|| -> CoreResult<Vec<f64>> {
            let deg = deg_opt.unwrap_or_else(|| adaptive_chebyshev_degree(t_val));
            let coeffs = compute_chebyshev_bandpass_coefficients(&[(1.0, t_val)], lmax_val, deg)?;
            let (l_sym, _) = symmetric_normalized_laplacian(g);
            let mut out = vec![0.0_f64; v_flat.len()];
            let mut ws = ChebyshevWorkspace::with_capacity(v_flat.len());
            ws.apply_batch_with_lambda_max(&l_sym, lmax_val, &coeffs, &v_flat, &mut out, ncols)?;
            Ok(out)
        })
        .map_err(core_error_to_py)?;

    if is_2d {
        let arr = Array2::from_shape_vec((n, ncols), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, arr).into_any())
    } else {
        Ok(PyArray1::from_vec(py, out).into_any())
    }
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
            let stats = linalg_spectral_moment_stats(g);
            let (t1_eff, t2_eff, alpha_eff) =
                resolve_dog_r1_defaults(&stats, t1_opt, t2_opt, alpha_opt);
            let base_terms = build_dog_terms(t1_eff, t2_eff, alpha_eff, 0.0)?;
            let (terms_psd, _, _) = project_psd_spectral_filter(&base_terms, 2.0, deg, eps)?;
            let (l_sym, inv_sqrt_deg) = symmetric_normalized_laplacian(g);
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
        Array2::from_shape_vec((n, dim), emb).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Extract disjoint, sorted, self-loop-free unweighted 1-hop and pure 2-hop `CsrGraph`s.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph), text_signature = "(graph)")]
fn extract_1hop_2hop_csr(py: Python<'_>, graph: &PyCsrGraph) -> PyResult<(PyCsrGraph, PyCsrGraph)> {
    let g = graph.inner();
    let (adj1, adj2) = py
        .detach(|| nb_extract_1hop_2hop_csr(g))
        .map_err(core_error_to_py)?;
    Ok((PyCsrGraph { inner: adj1 }, PyCsrGraph { inner: adj2 }))
}

/// Build a pure 1+2-hop `KnnGraph` with `{1.0, 2.0}` distances, `k2_deg`, and summary `stats`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph), text_signature = "(graph)")]
fn build_pure_2hop_knn<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
) -> PyResult<(PyKnnGraph, Bound<'py, PyArray1<u32>>, Py<PyDict>)> {
    let g = graph.inner();
    let (knn, k2_deg, stats) = py
        .detach(|| nb_build_pure_2hop_knn(g))
        .map_err(core_error_to_py)?;
    let stats_dict = knn_stats_to_py_dict(py, &stats)?;
    Ok((
        PyKnnGraph { inner: knn },
        PyArray1::from_vec(py, k2_deg),
        stats_dict,
    ))
}

/// Compute exact distinct 1-hop + 2-hop neighborhood sizes for every vertex in `graph`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph), text_signature = "(graph)")]
fn compute_2hop_neighborhood_sizes<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
) -> Bound<'py, PyArray1<u32>> {
    let g = graph.inner();
    let sizes = py.detach(|| nb_compute_2hop_neighborhood_sizes(g));
    PyArray1::from_vec(py, sizes)
}

/// Compute per-node 2-hop adaptive perplexities `clip(c * k2_i, min_perp, upper)`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, *, c = StrictF64(0.9), min_perplexity = StrictF64(5.0), max_perplexity = StrictF64(1200.0)),
    text_signature = "(graph, *, c=0.9, min_perplexity=5.0, max_perplexity=1200.0)"
)]
fn compute_nodewise_2hop_perplexities<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    c: StrictF64,
    min_perplexity: StrictF64,
    max_perplexity: StrictF64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let g = graph.inner();
    let out = py
        .detach(|| {
            nb_compute_nodewise_2hop_perplexities(g, c.0, min_perplexity.0, max_perplexity.0)
        })
        .map_err(core_error_to_py)?;
    Ok(PyArray1::from_vec(py, out))
}

/// Compute scalar median 2-hop adaptive perplexity `clip(c * median(k2), min_perp, upper)`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, *, c = StrictF64(1.5), min_perplexity = StrictF64(15.0), max_perplexity = StrictF64(1200.0)),
    text_signature = "(graph, *, c=1.5, min_perplexity=15.0, max_perplexity=1200.0)"
)]
fn compute_median_2hop_perplexity(
    py: Python<'_>,
    graph: &PyCsrGraph,
    c: StrictF64,
    min_perplexity: StrictF64,
    max_perplexity: StrictF64,
) -> PyResult<f64> {
    let g = graph.inner();
    py.detach(|| nb_compute_median_2hop_perplexity(g, c.0, min_perplexity.0, max_perplexity.0))
        .map_err(core_error_to_py)
}

/// Build a hybrid 1+2-hop + 2-stage spectral rerank `KnnGraph` and summary `stats`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (graph, coords, perplexities, *, k_factor = StrictF64(3.0), k = None, margin_gap = StrictF64(0.1), coarse_dim = StrictUsize(10), rerank_factor = StrictUsize(16), max_row_k = None),
    text_signature = "(graph, coords, perplexities, *, k_factor=3.0, k=None, margin_gap=0.1, coarse_dim=10, rerank_factor=16, max_row_k=None)"
)]
fn build_hybrid_adaptive_knn(
    py: Python<'_>,
    graph: &PyCsrGraph,
    coords: &Bound<'_, PyAny>,
    perplexities: &Bound<'_, PyAny>,
    k_factor: StrictF64,
    k: Option<StrictUsize>,
    margin_gap: StrictF64,
    coarse_dim: StrictUsize,
    rerank_factor: StrictUsize,
    max_row_k: Option<StrictUsize>,
) -> PyResult<(PyKnnGraph, Py<PyDict>)> {
    let g = graph.inner();
    let n = g.num_nodes();
    let (coords_flat, nrows, dim) = extract_f64_2d(coords, "coords")?;
    if nrows != n {
        return Err(PyValueError::new_err(format!(
            "coords row count {nrows} does not match graph.num_nodes ({n})"
        )));
    }
    let perps_vec = extract_perplexities(perplexities, n)?;
    let cfg = HybridKnnConfig {
        k_factor: k_factor.0,
        k_override: k.map(|v| v.0),
        margin_gap: margin_gap.0,
        coarse_dim: coarse_dim.0,
        rerank_factor: rerank_factor.0,
        max_row_k: max_row_k.map(|v| v.0),
    };
    let (knn, stats) = py
        .detach(|| nb_build_hybrid_adaptive_knn(g, &coords_flat, dim, &perps_vec, &cfg))
        .map_err(core_error_to_py)?;
    let stats_dict = knn_stats_to_py_dict(py, &stats)?;
    Ok((PyKnnGraph { inner: knn }, stats_dict))
}

/// Build an unweighted hop-shell `KnnGraph` via Partial BFS with deterministic tie-breaking.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, k = StrictUsize(30), *, seed = StrictU64(42)),
    text_signature = "(graph, k=30, *, seed=42)"
)]
fn partial_bfs_knn(
    py: Python<'_>,
    graph: &PyCsrGraph,
    k: StrictUsize,
    seed: StrictU64,
) -> PyResult<PyKnnGraph> {
    let g = graph.inner();
    let knn = py
        .detach(|| nb_partial_bfs_knn(g, k.0, seed.0))
        .map_err(core_error_to_py)?;
    Ok(PyKnnGraph { inner: knn })
}

/// Compute symmetric high-dimensional joint probabilities `P` from a `KnnGraph` and target perplexities.
///
/// Accepts either a 1D array of length `knn.num_nodes` or a scalar float perplexity broadcast to all vertices.
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (knn, perplexities, *, power = StrictF64(2.0), sigma_iters = StrictUsize(50), sigma_tolerance = StrictF64(1e-5), row_normalize = false),
    text_signature = "(knn, perplexities, *, power=2.0, sigma_iters=50, sigma_tolerance=1e-5, row_normalize=False)"
)]
fn compute_knn_probabilities(
    py: Python<'_>,
    knn: &PyKnnGraph,
    perplexities: &Bound<'_, PyAny>,
    power: StrictF64,
    sigma_iters: StrictUsize,
    sigma_tolerance: StrictF64,
    row_normalize: bool,
) -> PyResult<PySparseProbabilities> {
    let perps_vec = extract_perplexities(perplexities, knn.inner.num_nodes())?;
    let cfg = PerplexityConfig {
        power: power.0,
        sigma_iters: sigma_iters.0,
        sigma_tolerance: sigma_tolerance.0,
        symmetrization: if row_normalize {
            SymmetrizationMode::RowNormalized
        } else {
            SymmetrizationMode::StandardTsne
        },
    };
    let probs = py
        .detach(|| nb_compute_knn_probabilities(&knn.inner, &perps_vec, &cfg))
        .map_err(core_error_to_py)?;
    Ok(PySparseProbabilities { inner: probs })
}

/// Compute the UMAP / GuMap fuzzy simplicial set probability matrix from a `KnnGraph`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (knn, *, sigma_iters = StrictUsize(64), sigma_tolerance = StrictF64(1e-5)),
    text_signature = "(knn, *, sigma_iters=64, sigma_tolerance=1e-5)"
)]
fn compute_fuzzy_simplicial_set(
    py: Python<'_>,
    knn: &PyKnnGraph,
    sigma_iters: StrictUsize,
    sigma_tolerance: StrictF64,
) -> PyResult<PySparseProbabilities> {
    let probs = py
        .detach(|| nb_compute_fuzzy_simplicial_set(&knn.inner, sigma_iters.0, sigma_tolerance.0))
        .map_err(core_error_to_py)?;
    Ok(PySparseProbabilities { inner: probs })
}

/// Fit low-dimensional UMAP curve parameters `(a, b)` from `min_dist` and `spread`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (min_dist = StrictF64(0.1), spread = StrictF64(1.0)),
    text_signature = "(min_dist=0.1, spread=1.0)"
)]
fn find_ab_params(py: Python<'_>, min_dist: StrictF64, spread: StrictF64) -> PyResult<(f64, f64)> {
    py.detach(|| nb_find_ab_params(min_dist.0, spread.0))
        .map_err(core_error_to_py)
}

/// Compute R3 Horvitz-Thompson repulsion density scaling `gamma_non2 * sqrt(N / max(k_neg, 1))`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (gamma_non2, num_nodes, neg_per_node),
    text_signature = "(gamma_non2, num_nodes, neg_per_node)"
)]
fn compute_horvitz_thompson_gamma(
    py: Python<'_>,
    gamma_non2: StrictF64,
    num_nodes: StrictUsize,
    neg_per_node: StrictUsize,
) -> PyResult<f64> {
    py.detach(|| nb_compute_horvitz_thompson_gamma(gamma_non2.0, num_nodes.0, neg_per_node.0))
        .map_err(core_error_to_py)
}

/// Compute regularized DoG Varadhan landmark tables and optional coarse diffusive scaffold.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (graph, *, degree = StrictUsize(25), num_landmarks = StrictUsize(64), seed = StrictU64(42), degree_normalize = true, p_exp = StrictF64(0.5), eps_psd = StrictF64(1e-4), t_coarse = StrictF64(48.0), t1 = None, t2 = None, alpha = None, beta_floor = None),
    text_signature = "(graph, *, degree=25, num_landmarks=64, seed=42, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0, t1=None, t2=None, alpha=None, beta_floor=None)"
)]
fn compute_regularized_dog_varadhan_landmarks(
    py: Python<'_>,
    graph: &PyCsrGraph,
    degree: StrictUsize,
    num_landmarks: StrictUsize,
    seed: StrictU64,
    degree_normalize: bool,
    p_exp: StrictF64,
    eps_psd: StrictF64,
    t_coarse: StrictF64,
    t1: Option<StrictF64>,
    t2: Option<StrictF64>,
    alpha: Option<StrictF64>,
    beta_floor: Option<StrictF64>,
) -> PyResult<PyDogVaradhanLandmarks> {
    let cfg = DogVaradhanConfig {
        t1: t1.map(|v| v.0),
        t2: t2.map(|v| v.0),
        alpha: alpha.map(|v| v.0),
        beta_floor: beta_floor.map(|v| v.0),
        degree: Some(degree.0),
        n_landmarks: num_landmarks.0,
        eps_psd: eps_psd.0,
        degree_normalize,
        p_exp: p_exp.0,
        t_coarse: t_coarse.0,
        seed: seed.0,
        ..DogVaradhanConfig::default()
    };
    let g = graph.inner();
    let (inner, stats) = py
        .detach(|| nb_compute_regularized_dog_varadhan_landmarks(g, &cfg))
        .map_err(core_error_to_py)?;
    Ok(PyDogVaradhanLandmarks { inner, stats })
}

/// Register M2 `egraph-linalg` and `egraph-neighborhood` classes and functions on `egraph._core`.
pub(super) fn register_m2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKnnGraph>()?;
    m.add_class::<PySparseProbabilities>()?;
    m.add_class::<PyDogVaradhanLandmarks>()?;
    m.add_function(wrap_pyfunction!(select_maxmin_sp_pivots, m)?)?;
    m.add_function(wrap_pyfunction!(pivot_mds_init, m)?)?;
    m.add_function(wrap_pyfunction!(koren_spectral_init, m)?)?;
    m.add_function(wrap_pyfunction!(smallest_laplacian_eigenpairs, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_moment_stats, m)?)?;
    m.add_function(wrap_pyfunction!(chebyshev_heat_spmv, m)?)?;
    m.add_function(wrap_pyfunction!(chebyshev_dog_jl_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(extract_1hop_2hop_csr, m)?)?;
    m.add_function(wrap_pyfunction!(build_pure_2hop_knn, m)?)?;
    m.add_function(wrap_pyfunction!(compute_2hop_neighborhood_sizes, m)?)?;
    m.add_function(wrap_pyfunction!(compute_nodewise_2hop_perplexities, m)?)?;
    m.add_function(wrap_pyfunction!(compute_median_2hop_perplexity, m)?)?;
    m.add_function(wrap_pyfunction!(build_hybrid_adaptive_knn, m)?)?;
    m.add_function(wrap_pyfunction!(partial_bfs_knn, m)?)?;
    m.add_function(wrap_pyfunction!(compute_knn_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(compute_fuzzy_simplicial_set, m)?)?;
    m.add_function(wrap_pyfunction!(find_ab_params, m)?)?;
    m.add_function(wrap_pyfunction!(compute_horvitz_thompson_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(
        compute_regularized_dog_varadhan_landmarks,
        m
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        KnnStats, SpectralMomentStats, knn_stats_float_fields, knn_stats_int_fields,
        resolve_dog_r1_defaults, spectral_moment_fields, validate_dog_jl_params,
        validate_koren_params,
    };

    #[test]
    fn test_pure_rust_m2_helpers() {
        let stats = SpectralMomentStats {
            sigma_lambda_sq: 0.25,
            sigma_lambda: 0.5,
            deg_cv: 0.1,
            mean_degree: 4.0,
            leaf_fraction: 0.0,
        };
        let fields = spectral_moment_fields(&stats);
        assert_eq!(fields[0], ("sigma_lambda_sq", 0.25));
        assert_eq!(fields[1], ("sigma_lambda", 0.5));
        assert_eq!(fields[4], ("leaf_fraction", 0.0));

        let (t1, t2, alpha) = resolve_dog_r1_defaults(&stats, None, Some(2.0), None);
        assert!((t1 - 1.0 / 1.5).abs() < 1e-12);
        assert!((t2 - 2.0).abs() < 1e-12);
        assert!((alpha - 0.5).abs() < 1e-12);

        let knn_stats = KnnStats {
            avg_deg1: 2.0,
            avg_deg2: 5.0,
            avg_row_k: 6.0,
            min_row_k: 4,
            max_row_k: 8,
            nnz: 60,
            n_1hop: 20,
            n_2hop: 30,
            n_truncated_1hop: 0,
            n_truncated_2hop: 0,
            n_filled_residual: 3,
            n_pure_2hop_rows: 7,
        };
        assert_eq!(knn_stats_float_fields(&knn_stats).len(), 3);
        assert_eq!(knn_stats_int_fields(&knn_stats)[2], ("nnz", 60));

        assert!(validate_koren_params(1e-4, 300, Some(1.0)).is_ok());
        assert!(validate_koren_params(0.0, 300, None).is_err());
        assert!(validate_koren_params(f64::NAN, 300, None).is_err());
        assert!(validate_koren_params(1e-4, 0, None).is_err());
        assert!(validate_koren_params(1e-4, 300, Some(0.0)).is_err());

        assert!(validate_dog_jl_params(25, 64, 1e-4).is_ok());
        assert!(validate_dog_jl_params(0, 64, 1e-4).is_err());
        assert!(validate_dog_jl_params(25, 0, 1e-4).is_err());
        assert!(validate_dog_jl_params(25, 64, -1e-4).is_err());
        assert!(validate_dog_jl_params(25, 64, f64::NAN).is_err());
    }
}
