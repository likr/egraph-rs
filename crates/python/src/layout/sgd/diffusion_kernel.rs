use crate::{
    array::{PyArray1, PyArray2},
    distance_matrix::PyLaplacian,
    FloatType,
};
use petgraph_distance::Kernel;
use petgraph_linalg_diffusion_kernel::{
    DiffusionKernel, DiffusionKernelBuilder, LowRankDiffusionKernel, LowRankDiffusionKernelBuilder,
    LowRankMultiscaleDiffusionKernel, LowRankMultiscaleDiffusionKernelBuilder,
    MultiscaleDiffusionKernel, MultiscaleDiffusionKernelBuilder, PivotedDiffusionKernel,
    PivotedDiffusionKernelBuilder, PivotedKernel, PivotedMultiscaleDiffusionKernel,
    PivotedMultiscaleDiffusionKernelBuilder,
};
use pyo3::prelude::*;

/// Python class for querying diffusion kernel matrix elements
#[pyclass]
#[pyo3(name = "DiffusionKernel")]
pub struct PyDiffusionKernel {
    pub(crate) kernel: DiffusionKernel<FloatType>,
}

#[pymethods]
impl PyDiffusionKernel {
    /// Creates a new DiffusionKernel
    #[new]
    fn new(
        laplacian: &PyLaplacian,
        t: FloatType,
        degree: usize,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let kernel = DiffusionKernelBuilder::new(&laplacian.matrix, t, degree)
            .build(rng.get_mut())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyDiffusionKernel { kernel })
    }

    /// Creates a new DiffusionKernel with externally provided lambda_max
    #[staticmethod]
    fn new_with_lambda_max(
        laplacian: &PyLaplacian,
        t: FloatType,
        degree: usize,
        lambda_max: FloatType,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let kernel = DiffusionKernelBuilder::new(&laplacian.matrix, t, degree)
            .lambda_max(lambda_max)
            .build(rng.get_mut())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyDiffusionKernel { kernel })
    }

    /// Queries the (i, j) element of the diffusion kernel matrix
    fn get(&self, i: usize, j: usize) -> FloatType {
        self.kernel.get(i, j)
    }

    /// Returns the number of nodes in the graph
    fn n(&self) -> usize {
        self.kernel.n()
    }
}

/// Python class for computing multiscale diffusion distance between nodes
#[pyclass]
#[pyo3(name = "MultiscaleDiffusionKernel")]
pub struct PyMultiscaleDiffusionKernel {
    pub(crate) kernel: MultiscaleDiffusionKernel<FloatType>,
}

#[pymethods]
impl PyMultiscaleDiffusionKernel {
    /// Creates a new MultiscaleDiffusionKernel
    #[new]
    fn new(laplacian: &PyLaplacian, alpha: FloatType) -> PyResult<Self> {
        let kernel = MultiscaleDiffusionKernelBuilder::new(&laplacian.matrix, alpha)
            .build()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyMultiscaleDiffusionKernel { kernel })
    }

    /// Queries the multiscale diffusion kernel between nodes i and j
    fn get(&self, i: usize, j: usize) -> FloatType {
        self.kernel.get(i, j)
    }

    /// Returns the number of nodes in the graph
    fn n(&self) -> usize {
        self.kernel.n()
    }
}

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
        let kernel = LowRankDiffusionKernel::new(
            t,
            eigenvalues.as_array().clone(),
            eigenvectors.as_array().clone(),
        );
        Ok(PyLowRankDiffusionKernel { kernel })
    }

    /// Queries the (i, j) element of the low-rank heat kernel matrix
    fn get(&self, i: usize, j: usize) -> FloatType {
        self.kernel.get(i, j)
    }

    /// Returns the number of nodes in the graph
    fn n(&self) -> usize {
        self.kernel.n()
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
        let kernel = LowRankMultiscaleDiffusionKernel::new(
            alpha,
            eigenvalues.as_array().clone(),
            eigenvectors.as_array().clone(),
        );
        Ok(PyLowRankMultiscaleDiffusionKernel { kernel })
    }

    /// Queries the (i, j) element of the low-rank multiscale kernel matrix
    fn get(&self, i: usize, j: usize) -> FloatType {
        self.kernel.get(i, j)
    }

    /// Returns the number of nodes in the graph
    fn n(&self) -> usize {
        self.kernel.n()
    }
}

/// Python class for pivoted diffusion kernel
#[pyclass]
#[pyo3(name = "PivotedDiffusionKernel")]
pub struct PyPivotedDiffusionKernel {
    pub(crate) kernel: PivotedDiffusionKernel<FloatType>,
}

#[pymethods]
impl PyPivotedDiffusionKernel {
    /// Creates a new PivotedDiffusionKernel
    #[new]
    fn new(
        laplacian: &PyLaplacian,
        t: FloatType,
        degree: usize,
        pivots: Vec<usize>,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let kernel = PivotedDiffusionKernelBuilder::new(&laplacian.matrix, t, degree, pivots)
            .build(rng.get_mut())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyPivotedDiffusionKernel { kernel })
    }

    /// Creates a new PivotedDiffusionKernel with lambda_max
    #[staticmethod]
    fn new_with_lambda_max(
        laplacian: &PyLaplacian,
        t: FloatType,
        degree: usize,
        pivots: Vec<usize>,
        lambda_max: FloatType,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let kernel = PivotedDiffusionKernelBuilder::new(&laplacian.matrix, t, degree, pivots)
            .lambda_max(lambda_max)
            .build(rng.get_mut())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyPivotedDiffusionKernel { kernel })
    }

    /// Returns pivot indices
    fn pivots(&self) -> Vec<usize> {
        self.kernel.pivots().to_vec()
    }

    /// Queries element from pivot
    fn get_from_pivot(&self, pivot_idx: usize, j: usize) -> FloatType {
        self.kernel.get_from_pivot(pivot_idx, j)
    }
}

/// Python class for pivoted multiscale diffusion kernel
#[pyclass]
#[pyo3(name = "PivotedMultiscaleDiffusionKernel")]
pub struct PyPivotedMultiscaleDiffusionKernel {
    pub(crate) kernel: PivotedMultiscaleDiffusionKernel<FloatType>,
}

#[pymethods]
impl PyPivotedMultiscaleDiffusionKernel {
    /// Creates a new PivotedMultiscaleDiffusionKernel
    #[new]
    fn new(laplacian: &PyLaplacian, alpha: FloatType, pivots: Vec<usize>) -> PyResult<Self> {
        let kernel = PivotedMultiscaleDiffusionKernelBuilder::new(&laplacian.matrix, alpha, pivots)
            .build()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyPivotedMultiscaleDiffusionKernel { kernel })
    }

    /// Returns pivot indices
    fn pivots(&self) -> Vec<usize> {
        self.kernel.pivots().to_vec()
    }

    /// Queries element from pivot
    fn get_from_pivot(&self, pivot_idx: usize, j: usize) -> FloatType {
        self.kernel.get_from_pivot(pivot_idx, j)
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

    fn eigenvalue_max_iterations(mut slf: PyRefMut<'_, Self>, eigenvalue_max_iterations: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().eigenvalue_max_iterations(eigenvalue_max_iterations);
        slf
    }

    fn cg_max_iterations(mut slf: PyRefMut<'_, Self>, cg_max_iterations: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().cg_max_iterations(cg_max_iterations);
        slf
    }

    fn eigenvalue_tolerance(mut slf: PyRefMut<'_, Self>, eigenvalue_tolerance: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().eigenvalue_tolerance(eigenvalue_tolerance);
        slf
    }

    fn cg_tolerance(mut slf: PyRefMut<'_, Self>, cg_tolerance: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().cg_tolerance(cg_tolerance);
        slf
    }

    fn build_unnormalized_laplacian(
        &self,
        laplacian: &PyLaplacian,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<PyLowRankDiffusionKernel> {
        let kernel = self.builder.clone().build_unnormalized_laplacian(&laplacian.matrix, rng.get_mut())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyLowRankDiffusionKernel { kernel })
    }

    fn build_symmetric_normalized_laplacian(
        &self,
        laplacian: &PyLaplacian,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<PyLowRankDiffusionKernel> {
        let kernel = self.builder.clone().build_symmetric_normalized_laplacian(&laplacian.matrix, rng.get_mut())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyLowRankDiffusionKernel { kernel })
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

    fn eigenvalue_max_iterations(mut slf: PyRefMut<'_, Self>, eigenvalue_max_iterations: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().eigenvalue_max_iterations(eigenvalue_max_iterations);
        slf
    }

    fn cg_max_iterations(mut slf: PyRefMut<'_, Self>, cg_max_iterations: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().cg_max_iterations(cg_max_iterations);
        slf
    }

    fn eigenvalue_tolerance(mut slf: PyRefMut<'_, Self>, eigenvalue_tolerance: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().eigenvalue_tolerance(eigenvalue_tolerance);
        slf
    }

    fn cg_tolerance(mut slf: PyRefMut<'_, Self>, cg_tolerance: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().cg_tolerance(cg_tolerance);
        slf
    }

    fn build_unnormalized_laplacian(
        &self,
        laplacian: &PyLaplacian,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<PyLowRankMultiscaleDiffusionKernel> {
        let kernel = self.builder.clone().build_unnormalized_laplacian(&laplacian.matrix, rng.get_mut())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyLowRankMultiscaleDiffusionKernel { kernel })
    }

    fn build_symmetric_normalized_laplacian(
        &self,
        laplacian: &PyLaplacian,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<PyLowRankMultiscaleDiffusionKernel> {
        let kernel = self.builder.clone().build_symmetric_normalized_laplacian(&laplacian.matrix, rng.get_mut())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyLowRankMultiscaleDiffusionKernel { kernel })
    }
}
