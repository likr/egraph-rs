use crate::{distance_matrix::PyLaplacian, FloatType};
use petgraph_distance::Kernel;
use petgraph_linalg_diffusion_kernel::{
    DiffusionKernel, DiffusionKernelBuilder, LowRankDiffusionKernel, LowRankDiffusionKernelBuilder,
    MultiscaleDiffusionKernel, MultiscaleDiffusionKernelBuilder,
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

    /// Queries the (i, j) element of the low-rank heat kernel matrix
    fn get(&self, i: usize, j: usize) -> FloatType {
        self.kernel.get(i, j)
    }

    /// Returns the number of nodes in the graph
    fn n(&self) -> usize {
        self.kernel.n()
    }
}
