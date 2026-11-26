//! Sparse symmetric CSR matrix (`CsrMatrix`) and anisotropic graph Laplacian bindings.

use egraph::core::CoreResult;
use egraph::linalg::{
    AnisotropicLaplacianConfig, CsrMatrix, DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR,
    DEFAULT_SPECTRUM_LANCZOS_STEPS, DEFAULT_SPECTRUM_POWER_ITERS, DEFAULT_SPECTRUM_SAFETY_FACTOR,
    anisotropic_laplacian as linalg_anisotropic_laplacian,
};
use numpy::PyArray1;
use numpy::PyArray2;
use numpy::ndarray::Array2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array_extract::extract_f64_1d;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d};

/// Double-precision square Compressed Sparse Row matrix (`CsrMatrix`) with spectral bounds
/// and matrix-vector operations.
#[pyclass(name = "CsrMatrix", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PyCsrMatrix {
    inner: CsrMatrix,
}

impl PyCsrMatrix {
    pub(crate) fn inner(&self) -> &CsrMatrix {
        &self.inner
    }
}

#[pymethods]
impl PyCsrMatrix {
    /// Construct a `CsrMatrix` from raw CSR buffers `(num_nodes, indptr, indices, values)`.
    #[new]
    #[pyo3(
        signature = (num_nodes, indptr, indices, values),
        text_signature = "(num_nodes, indptr, indices, values)"
    )]
    fn new(
        num_nodes: StrictUsize,
        indptr: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let indptr_v = extract_u32_1d(indptr, "indptr")?;
        let indices_v = extract_u32_1d(indices, "indices")?;
        let values_v = extract_f64_1d(values, "values")?;
        let inner = CsrMatrix::from_csr(num_nodes.0, indptr_v, indices_v, values_v)
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `CsrMatrix` from raw CSR buffers `(num_nodes, indptr, indices, values)`.
    #[staticmethod]
    #[pyo3(
        signature = (num_nodes, indptr, indices, values),
        text_signature = "(num_nodes, indptr, indices, values)"
    )]
    fn from_csr(
        num_nodes: StrictUsize,
        indptr: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Self::new(num_nodes, indptr, indices, values)
    }

    /// Matrix dimension `num_nodes`.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Number of stored non-zero entries.
    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Return a `uint32` NumPy copy of CSR `indptr` (`length == num_nodes + 1`).
    fn indptr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, self.inner.indptr())
    }

    /// Return a `uint32` NumPy copy of sorted CSR column `indices` (`length == nnz`).
    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, self.inner.indices())
    }

    /// Return a `float64` NumPy copy of CSR `values` (`length == nnz`).
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.values())
    }

    /// Return a `float64` NumPy copy of the main diagonal entries (`length == num_nodes`).
    fn diagonal<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.diagonal())
    }

    /// Return the matrix trace `sum(diagonal)`.
    fn trace(&self) -> f64 {
        self.inner.trace()
    }

    /// Return the symmetric Gershgorin upper spectral bound `max_i (|A_ii| + sum_{j!=i} |A_ij|)`.
    fn gershgorin_spectral_bound(&self) -> f64 {
        self.inner.gershgorin_spectral_bound()
    }

    /// Estimate `lambda_max` via deterministic Rademacher power iteration clamped by Gershgorin.
    #[pyo3(
        signature = (*, num_iters = StrictUsize(DEFAULT_SPECTRUM_POWER_ITERS), safety_factor = StrictF64(DEFAULT_SPECTRUM_SAFETY_FACTOR), seed = StrictU64(42)),
        text_signature = "($self, *, num_iters=10, safety_factor=1.02, seed=42)"
    )]
    fn estimate_lambda_max(
        &self,
        py: Python<'_>,
        num_iters: StrictUsize,
        safety_factor: StrictF64,
        seed: StrictU64,
    ) -> PyResult<f64> {
        let inner = &self.inner;
        let iters = num_iters.0;
        let sf = safety_factor.0;
        let s = seed.0;
        py.detach(|| inner.estimate_lambda_max(iters, sf, s))
            .map_err(core_error_to_py)
    }

    /// Estimate `lambda_max` via deterministic Lanczos tridiagonalization clamped by Gershgorin.
    #[pyo3(
        signature = (*, num_steps = StrictUsize(DEFAULT_SPECTRUM_LANCZOS_STEPS), safety_factor = StrictF64(DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR), seed = StrictU64(42)),
        text_signature = "($self, *, num_steps=16, safety_factor=1.05, seed=42)"
    )]
    fn estimate_lambda_max_lanczos(
        &self,
        py: Python<'_>,
        num_steps: StrictUsize,
        safety_factor: StrictF64,
        seed: StrictU64,
    ) -> PyResult<f64> {
        let inner = &self.inner;
        let steps = num_steps.0;
        let sf = safety_factor.0;
        let s = seed.0;
        py.detach(|| inner.estimate_lambda_max_lanczos(steps, sf, s))
            .map_err(core_error_to_py)
    }

    /// Compute sparse matrix-vector product `y = A x`.
    #[pyo3(signature = (x), text_signature = "($self, x)")]
    fn spmv<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_vec = extract_f64_1d(x, "x")?;
        let n = self.inner.num_nodes();
        let inner = &self.inner;
        let y = py
            .detach(|| -> CoreResult<Vec<f64>> {
                let mut out = vec![0.0_f64; n];
                inner.spmv(&x_vec, &mut out)?;
                Ok(out)
            })
            .map_err(core_error_to_py)?;
        Ok(PyArray1::from_vec(py, y))
    }

    /// Return `scale * A + shift * I`.
    #[pyo3(signature = (scale, shift), text_signature = "($self, scale, shift)")]
    fn scale_and_shift(
        &self,
        py: Python<'_>,
        scale: StrictF64,
        shift: StrictF64,
    ) -> PyResult<Self> {
        let inner = &self.inner;
        let sc = scale.0;
        let sh = shift.0;
        let mat = py
            .detach(|| inner.scale_and_shift(sc, sh))
            .map_err(core_error_to_py)?;
        Ok(Self { inner: mat })
    }

    /// Return `True` if `|A_ij - A_ji| <= tol` for all entries.
    #[pyo3(signature = (tol = StrictF64(1e-10)), text_signature = "($self, tol=1e-10)")]
    fn is_symmetric(&self, tol: StrictF64) -> bool {
        self.inner.is_symmetric(tol.0)
    }

    /// Convert to a dense `(num_nodes, num_nodes)` `float64` NumPy array.
    fn to_dense<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n = self.inner.num_nodes();
        let dense = py
            .detach(|| self.inner.to_dense())
            .map_err(core_error_to_py)?;
        let arr = Array2::from_shape_vec((n, n), dense)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, arr))
    }

    /// Return `num_nodes`.
    fn __len__(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Return a concise string representation of the `CsrMatrix`.
    fn __repr__(&self) -> String {
        format!(
            "CsrMatrix(num_nodes={}, nnz={})",
            self.inner.num_nodes(),
            self.inner.nnz()
        )
    }
}

/// Construct a symmetric PSD Coifman-Lafon anisotropic graph Laplacian `L_aniso` and
/// its near-nullspace vector `u` (`L_aniso * u == 0` on every non-isolated component).
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, *, alpha = StrictF64(0.5), beta = StrictF64(0.0), degree_reg = StrictF64(0.0), degree_cap = None, reweighted_degree_norm = true),
    text_signature = "(graph, *, alpha=0.5, beta=0.0, degree_reg=0.0, degree_cap=None, reweighted_degree_norm=True)"
)]
fn anisotropic_laplacian<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    alpha: StrictF64,
    beta: StrictF64,
    degree_reg: StrictF64,
    degree_cap: Option<StrictF64>,
    reweighted_degree_norm: bool,
) -> PyResult<(PyCsrMatrix, Bound<'py, PyArray1<f64>>)> {
    let cfg = AnisotropicLaplacianConfig {
        alpha: alpha.0,
        beta: beta.0,
        degree_reg: degree_reg.0,
        degree_cap: degree_cap.map(|v| v.0),
        reweighted_degree_norm,
    };
    let g = graph.inner();
    let res = py
        .detach(|| linalg_anisotropic_laplacian(g, cfg))
        .map_err(core_error_to_py)?;
    Ok((
        PyCsrMatrix { inner: res.matrix },
        PyArray1::from_vec(py, res.nullspace),
    ))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCsrMatrix>()?;
    m.add_function(wrap_pyfunction!(anisotropic_laplacian, m)?)?;
    Ok(())
}
