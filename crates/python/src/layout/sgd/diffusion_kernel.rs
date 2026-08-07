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
    /// Creates a new LowRankDiffusionKernel by computing smallest eigenvalues and eigenvectors
    #[new]
    #[pyo3(signature = (laplacian, t, rank, rng))]
    fn new(
        laplacian: &PyLaplacian,
        t: FloatType,
        rank: usize,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let kernel = LowRankDiffusionKernelBuilder::new()
            .t(t)
            .rank(rank)
            .build(&laplacian.matrix, rng.get_mut())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyLowRankDiffusionKernel { kernel })
    }

    /// Creates a LowRankDiffusionKernel directly from eigenvalues and eigenvectors
    #[staticmethod]
    fn new_from_eigen(
        t: FloatType,
        eigenvalues: &PyArray1,
        eigenvectors: &PyArray2,
    ) -> PyResult<Self> {
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
    /// Creates a new LowRankMultiscaleDiffusionKernel by computing smallest eigenvalues and eigenvectors
    #[new]
    #[pyo3(signature = (laplacian, alpha, rank, rng))]
    fn new(
        laplacian: &PyLaplacian,
        alpha: FloatType,
        rank: usize,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let kernel = LowRankMultiscaleDiffusionKernelBuilder::new()
            .alpha(alpha)
            .rank(rank)
            .build(&laplacian.matrix, rng.get_mut())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyLowRankMultiscaleDiffusionKernel { kernel })
    }

    /// Creates a LowRankMultiscaleDiffusionKernel directly from eigenvalues and eigenvectors
    #[staticmethod]
    fn new_from_eigen(
        alpha: FloatType,
        eigenvalues: &PyArray1,
        eigenvectors: &PyArray2,
    ) -> PyResult<Self> {
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
