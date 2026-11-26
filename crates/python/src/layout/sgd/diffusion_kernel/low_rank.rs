//! Low-rank spectral diffusion kernel and builder bindings.

use super::{resolve_length_and_rng, with_optional_rng_ref_mut};
use crate::{
    FloatType,
    array::{PyArray1, PyArray2},
};
use petgraph::visit::EdgeRef;
use petgraph_linalg_kernel::{
    Kernel, LowRankDiffusionKernel, LowRankDiffusionKernelBuilder,
    LowRankMultiscaleDiffusionKernel, LowRankMultiscaleDiffusionKernelBuilder,
};
use pyo3::prelude::*;

/// Python class for querying low-rank spectral heat kernel matrix elements
#[pyclass]
#[pyo3(name = "LowRankDiffusionKernel")]
pub struct PyLowRankDiffusionKernel {
    pub(crate) kernel: LowRankDiffusionKernel<FloatType>,
}

#[pymethods]
impl PyLowRankDiffusionKernel {
    #[new]
    #[pyo3(signature = (t, eigenvalues, eigenvectors))]
    fn new(t: FloatType, eigenvalues: &PyArray1, eigenvectors: &PyArray2) -> PyResult<Self> {
        if !t.is_finite() || t < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "t must be finite and >= 0",
            ));
        }
        let eigvals = eigenvalues.as_array();
        let eigvecs = eigenvectors.as_array();
        if eigvals.len() != eigvecs.ncols() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "eigenvalues length ({}) must match eigenvectors columns ({})",
                eigvals.len(),
                eigvecs.ncols()
            )));
        }
        if !eigvals.iter().all(|v| v.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "eigenvalues must contain only finite values",
            ));
        }
        if !eigvecs.iter().all(|v| v.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "eigenvectors must contain only finite values",
            ));
        }
        let kernel = LowRankDiffusionKernel::new(t, eigvals.clone(), eigvecs.clone());
        Ok(PyLowRankDiffusionKernel { kernel })
    }

    /// Queries the (i, j) element of the low-rank heat kernel matrix
    fn get(&self, i: usize, j: usize) -> PyResult<FloatType> {
        let (rows, cols) = self.kernel.shape();
        self.kernel.get(i, j).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "Index ({i}, {j}) out of bounds for kernel shape ({rows}, {cols})"
            ))
        })
    }

    /// Returns the number of nodes in the graph
    fn n(&self) -> usize {
        self.kernel.shape().0
    }
}

/// Python class for querying low-rank spectral multiscale diffusion kernel matrix elements
#[pyclass]
#[pyo3(name = "LowRankMultiscaleDiffusionKernel")]
pub struct PyLowRankMultiscaleDiffusionKernel {
    pub(crate) kernel: LowRankMultiscaleDiffusionKernel<FloatType>,
}

#[pymethods]
impl PyLowRankMultiscaleDiffusionKernel {
    #[new]
    #[pyo3(signature = (alpha, eigenvalues, eigenvectors))]
    fn new(alpha: FloatType, eigenvalues: &PyArray1, eigenvectors: &PyArray2) -> PyResult<Self> {
        if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "alpha must be finite and in [0, 1)",
            ));
        }
        let eigvals = eigenvalues.as_array();
        let eigvecs = eigenvectors.as_array();
        if eigvals.len() != eigvecs.ncols() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "eigenvalues length ({}) must match eigenvectors columns ({})",
                eigvals.len(),
                eigvecs.ncols()
            )));
        }
        if !eigvals.iter().all(|v| v.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "eigenvalues must contain only finite values",
            ));
        }
        if !eigvecs.iter().all(|v| v.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "eigenvectors must contain only finite values",
            ));
        }
        let kernel = LowRankMultiscaleDiffusionKernel::new(alpha, eigvals.clone(), eigvecs.clone());
        Ok(PyLowRankMultiscaleDiffusionKernel { kernel })
    }

    /// Queries the (i, j) element of the low-rank multiscale kernel matrix
    fn get(&self, i: usize, j: usize) -> PyResult<FloatType> {
        let (rows, cols) = self.kernel.shape();
        self.kernel.get(i, j).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "Index ({i}, {j}) out of bounds for kernel shape ({rows}, {cols})"
            ))
        })
    }

    /// Returns the number of nodes in the graph
    fn n(&self) -> usize {
        self.kernel.shape().0
    }
}

/// Builder for PyLowRankDiffusionKernel
#[pyclass]
#[pyo3(name = "LowRankDiffusionKernelBuilder")]
pub struct PyLowRankDiffusionKernelBuilder {
    pub(crate) builder: LowRankDiffusionKernelBuilder<FloatType>,
}

#[pymethods]
impl PyLowRankDiffusionKernelBuilder {
    #[new]
    fn new() -> Self {
        Self {
            builder: LowRankDiffusionKernelBuilder::new(),
        }
    }

    fn t(mut slf: PyRefMut<'_, Self>, t: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().t(t);
        slf
    }

    fn rank(mut slf: PyRefMut<'_, Self>, rank: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().rank(rank);
        slf
    }

    fn shift(mut slf: PyRefMut<'_, Self>, shift: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().shift(shift);
        slf
    }

    fn eigenvalue_max_iterations(
        mut slf: PyRefMut<'_, Self>,
        eigenvalue_max_iterations: usize,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf
            .builder
            .clone()
            .eigenvalue_max_iterations(eigenvalue_max_iterations);
        slf
    }

    fn cg_max_iterations(
        mut slf: PyRefMut<'_, Self>,
        cg_max_iterations: usize,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().cg_max_iterations(cg_max_iterations);
        slf
    }

    fn eigenvalue_tolerance(
        mut slf: PyRefMut<'_, Self>,
        eigenvalue_tolerance: FloatType,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf
            .builder
            .clone()
            .eigenvalue_tolerance(eigenvalue_tolerance);
        slf
    }

    fn cg_tolerance(mut slf: PyRefMut<'_, Self>, cg_tolerance: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().cg_tolerance(cg_tolerance);
        slf
    }

    /// Builds a `LowRankDiffusionKernel` using the unnormalized combinatorial Laplacian.
    #[pyo3(name = "build_unnormalized_laplacian", signature = (graph, length = None, rng = None))]
    fn build_laplacian(
        &self,
        graph: &crate::graph::PyGraphAdapter,
        length: Option<&Bound<'_, PyAny>>,
        rng: Option<&Bound<'_, crate::rng::PyRng>>,
    ) -> PyResult<PyLowRankDiffusionKernel> {
        let kernel = build_low_rank_from_graph(graph, length, rng, |g, f, r| {
            self.builder.clone().build_laplacian(g, f, r)
        })?;
        Ok(PyLowRankDiffusionKernel { kernel })
    }

    /// Builds a `LowRankDiffusionKernel` using the symmetric normalized Laplacian.
    #[pyo3(name = "build_symmetric_normalized_laplacian", signature = (graph, length = None, rng = None))]
    fn build_normalized_laplacian(
        &self,
        graph: &crate::graph::PyGraphAdapter,
        length: Option<&Bound<'_, PyAny>>,
        rng: Option<&Bound<'_, crate::rng::PyRng>>,
    ) -> PyResult<PyLowRankDiffusionKernel> {
        let kernel = build_low_rank_from_graph(graph, length, rng, |g, f, r| {
            self.builder.clone().build_normalized_laplacian(g, f, r)
        })?;
        Ok(PyLowRankDiffusionKernel { kernel })
    }
}

fn build_low_rank_from_graph<K, F>(
    graph: &crate::graph::PyGraphAdapter,
    length: Option<&Bound<'_, PyAny>>,
    rng: Option<&Bound<'_, crate::rng::PyRng>>,
    build_fn: F,
) -> PyResult<K>
where
    F: FnOnce(
        &petgraph::graph::UnGraph<Py<PyAny>, Py<PyAny>, crate::graph::IndexType>,
        &mut dyn FnMut(petgraph::graph::EdgeReference<'_, Py<PyAny>>) -> FloatType,
        &mut rand::rngs::StdRng,
    ) -> Result<K, String>,
{
    let (edge_length, rng_opt) = resolve_length_and_rng(length, rng)?;
    match graph.graph() {
        crate::graph::GraphType::Graph(native_graph) => {
            let resolved_len = edge_length.resolve_for_graph(native_graph)?;
            let mut length_closure = |edge: petgraph::graph::EdgeReference<'_, Py<PyAny>>| {
                resolved_len.get(edge.id().index())
            };
            with_optional_rng_ref_mut(rng_opt, |r| build_fn(native_graph, &mut length_closure, r))
                .map_err(pyo3::exceptions::PyValueError::new_err)
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "Unsupported graph type",
        )),
    }
}

/// Builder for PyLowRankMultiscaleDiffusionKernel
#[pyclass]
#[pyo3(name = "LowRankMultiscaleDiffusionKernelBuilder")]
pub struct PyLowRankMultiscaleDiffusionKernelBuilder {
    pub(crate) builder: LowRankMultiscaleDiffusionKernelBuilder<FloatType>,
}

#[pymethods]
impl PyLowRankMultiscaleDiffusionKernelBuilder {
    #[new]
    fn new() -> Self {
        Self {
            builder: LowRankMultiscaleDiffusionKernelBuilder::new(),
        }
    }

    fn alpha(mut slf: PyRefMut<'_, Self>, alpha: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().alpha(alpha);
        slf
    }

    fn rank(mut slf: PyRefMut<'_, Self>, rank: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().rank(rank);
        slf
    }

    fn shift(mut slf: PyRefMut<'_, Self>, shift: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().shift(shift);
        slf
    }

    fn eigenvalue_max_iterations(
        mut slf: PyRefMut<'_, Self>,
        eigenvalue_max_iterations: usize,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf
            .builder
            .clone()
            .eigenvalue_max_iterations(eigenvalue_max_iterations);
        slf
    }

    fn cg_max_iterations(
        mut slf: PyRefMut<'_, Self>,
        cg_max_iterations: usize,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().cg_max_iterations(cg_max_iterations);
        slf
    }

    fn eigenvalue_tolerance(
        mut slf: PyRefMut<'_, Self>,
        eigenvalue_tolerance: FloatType,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf
            .builder
            .clone()
            .eigenvalue_tolerance(eigenvalue_tolerance);
        slf
    }

    fn cg_tolerance(mut slf: PyRefMut<'_, Self>, cg_tolerance: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().cg_tolerance(cg_tolerance);
        slf
    }

    /// Builds a `LowRankMultiscaleDiffusionKernel` using the unnormalized combinatorial Laplacian.
    #[pyo3(name = "build_unnormalized_laplacian", signature = (graph, length = None, rng = None))]
    fn build_laplacian(
        &self,
        graph: &crate::graph::PyGraphAdapter,
        length: Option<&Bound<'_, PyAny>>,
        rng: Option<&Bound<'_, crate::rng::PyRng>>,
    ) -> PyResult<PyLowRankMultiscaleDiffusionKernel> {
        let kernel = build_low_rank_from_graph(graph, length, rng, |g, f, r| {
            self.builder.clone().build_laplacian(g, f, r)
        })?;
        Ok(PyLowRankMultiscaleDiffusionKernel { kernel })
    }

    /// Builds a `LowRankMultiscaleDiffusionKernel` using the symmetric normalized Laplacian.
    #[pyo3(name = "build_symmetric_normalized_laplacian", signature = (graph, length = None, rng = None))]
    fn build_normalized_laplacian(
        &self,
        graph: &crate::graph::PyGraphAdapter,
        length: Option<&Bound<'_, PyAny>>,
        rng: Option<&Bound<'_, crate::rng::PyRng>>,
    ) -> PyResult<PyLowRankMultiscaleDiffusionKernel> {
        let kernel = build_low_rank_from_graph(graph, length, rng, |g, f, r| {
            self.builder.clone().build_normalized_laplacian(g, f, r)
        })?;
        Ok(PyLowRankMultiscaleDiffusionKernel { kernel })
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLowRankDiffusionKernel>()?;
    m.add_class::<PyLowRankDiffusionKernelBuilder>()?;
    m.add_class::<PyLowRankMultiscaleDiffusionKernel>()?;
    m.add_class::<PyLowRankMultiscaleDiffusionKernelBuilder>()?;
    Ok(())
}
