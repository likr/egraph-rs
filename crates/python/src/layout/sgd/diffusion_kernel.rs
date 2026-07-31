//! Python bindings for DiffusionKernel
//!
//! This module provides Python access to the diffusion kernel matrix exp(-tL),
//! allowing random access to individual matrix elements.

use crate::{distance_matrix::PyLaplacian, FloatType};
use petgraph_linalg_diffusion_kernel::{DiffusionKernel, MultiscaleDiffusionKernel};
use pyo3::prelude::*;

/// Python class for querying diffusion kernel matrix elements
#[pyclass]
#[pyo3(name = "DiffusionKernel")]
pub struct PyDiffusionKernel {
    pub(crate) kernel: DiffusionKernel<FloatType>,
}

#[pymethods]
impl PyDiffusionKernel {
    /// Creates a new DiffusionKernel with automatic lambda_max estimation
    #[new]
    fn new(
        laplacian: &PyLaplacian,
        t: FloatType,
        degree: usize,
        num_vectors: usize,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let kernel = DiffusionKernel::new(&laplacian.matrix, t, degree, num_vectors, rng.get_mut());

        Ok(PyDiffusionKernel { kernel })
    }

    /// Creates a new DiffusionKernel with externally provided lambda_max
    #[staticmethod]
    fn new_with_lambda_max(
        laplacian: &PyLaplacian,
        t: FloatType,
        degree: usize,
        lambda_max: FloatType,
        num_vectors: usize,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let kernel = DiffusionKernel::new_with_lambda_max(
            &laplacian.matrix,
            t,
            degree,
            lambda_max,
            num_vectors,
            rng.get_mut(),
        );

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
    pub(crate) kernel: MultiscaleDiffusionKernel,
}

#[pymethods]
impl PyMultiscaleDiffusionKernel {
    /// Creates a new MultiscaleDiffusionKernel and builds its Hutchinson index
    #[new]
    fn new(
        laplacian: &PyLaplacian,
        alpha: FloatType,
        num_samples: usize,
        tol: FloatType,
        max_iter: usize,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let mut kernel = MultiscaleDiffusionKernel::new(
            laplacian.matrix.clone(),
            alpha,
            num_samples,
            tol,
            max_iter,
        );

        kernel.build_index_with_rng(rng.get_mut());

        Ok(PyMultiscaleDiffusionKernel { kernel })
    }

    /// Queries the multiscale diffusion distance between nodes i and j
    fn get(&self, i: usize, j: usize) -> FloatType {
        self.kernel.sample_distance(i, j).unwrap_or(0.0)
    }

    /// Computes multiscale diffusion distance between nodes i and j
    fn sample_distance(&self, i: usize, j: usize) -> PyResult<FloatType> {
        self.kernel
            .sample_distance(i, j)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Re-builds the Hutchinson index using a custom random number generator
    fn build_index(&mut self, rng: &mut crate::rng::PyRng) {
        self.kernel.build_index_with_rng(rng.get_mut());
    }

    /// Returns the number of nodes in the graph
    fn n(&self) -> usize {
        self.kernel.n()
    }
}
