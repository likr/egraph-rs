//! Full and pivoted Chebyshev heat and multiscale diffusion kernel bindings.

use crate::{FloatType, distance_matrix::PyLaplacian};
use petgraph_linalg_kernel::{
    DiffusionKernel, DiffusionKernelBuilder, Kernel, MultiscaleDiffusionKernel,
    MultiscaleDiffusionKernelBuilder, PivotedDiffusionKernel, PivotedDiffusionKernelBuilder,
    PivotedKernel, PivotedMultiscaleDiffusionKernel, PivotedMultiscaleDiffusionKernelBuilder,
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
    #[pyo3(signature = (laplacian, t, degree, pivots, rng, num_hutchinson = None))]
    fn new(
        laplacian: &PyLaplacian,
        t: FloatType,
        degree: usize,
        pivots: Vec<usize>,
        rng: &mut crate::rng::PyRng,
        num_hutchinson: Option<usize>,
    ) -> PyResult<Self> {
        let kernel = PivotedDiffusionKernelBuilder::new(&laplacian.matrix, t, degree, pivots)
            .num_hutchinson(num_hutchinson)
            .build(rng.get_mut())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyPivotedDiffusionKernel { kernel })
    }

    /// Creates a new PivotedDiffusionKernel with lambda_max
    #[staticmethod]
    #[pyo3(signature = (laplacian, t, degree, pivots, lambda_max, rng, num_hutchinson = None))]
    fn new_with_lambda_max(
        laplacian: &PyLaplacian,
        t: FloatType,
        degree: usize,
        pivots: Vec<usize>,
        lambda_max: FloatType,
        rng: &mut crate::rng::PyRng,
        num_hutchinson: Option<usize>,
    ) -> PyResult<Self> {
        let kernel = PivotedDiffusionKernelBuilder::new(&laplacian.matrix, t, degree, pivots)
            .lambda_max(lambda_max)
            .num_hutchinson(num_hutchinson)
            .build(rng.get_mut())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyPivotedDiffusionKernel { kernel })
    }

    /// Returns pivot indices
    fn pivots(&self) -> Vec<usize> {
        self.kernel.pivots().to_vec()
    }

    /// Queries element from pivot
    fn get_from_pivot(&self, pivot_idx: usize, j: usize) -> PyResult<FloatType> {
        let n = self.kernel.vectors.first().map_or(0, |v| v.len());
        if pivot_idx >= self.kernel.pivots().len() || j >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index ({pivot_idx}, {j}) out of bounds for pivoted kernel shape ({}, {n})",
                self.kernel.pivots().len(),
            )));
        }
        Ok(self.kernel.get_from_pivot(pivot_idx, j))
    }

    /// Returns the estimated diagonal entry `K[j, j]` if Hutchinson estimation was enabled
    fn diagonal(&self, j: usize) -> Option<FloatType> {
        self.kernel.diagonal(j)
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
    fn get_from_pivot(&self, pivot_idx: usize, j: usize) -> PyResult<FloatType> {
        let n = self.kernel.vectors.first().map_or(0, |v| v.len());
        if pivot_idx >= self.kernel.pivots().len() || j >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index ({pivot_idx}, {j}) out of bounds for pivoted kernel shape ({}, {n})",
                self.kernel.pivots().len(),
            )));
        }
        Ok(self.kernel.get_from_pivot(pivot_idx, j))
    }

    /// Returns diagonal entry if available (always `None` for multiscale pivoted kernels).
    fn diagonal(&self, j: usize) -> Option<FloatType> {
        self.kernel.diagonal(j)
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDiffusionKernel>()?;
    m.add_class::<PyMultiscaleDiffusionKernel>()?;
    m.add_class::<PyPivotedDiffusionKernel>()?;
    m.add_class::<PyPivotedMultiscaleDiffusionKernel>()?;
    Ok(())
}
