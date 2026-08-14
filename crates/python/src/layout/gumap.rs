//! GUMAP layout algorithm bindings for Python
//!
//! This module provides Python bindings for the GUMAP (Graph UMAP) graph layout algorithm and its builder.

use crate::{distance_matrix::with_distance, drawing::PyDrawingEuclidean2d, rng::PyRng, FloatType};
use petgraph_layout_gumap::{GuMap, GuMapBuilder};
use pyo3::prelude::*;

/// Python class for constructing and configuring the `GuMap` algorithm via the Builder pattern.
#[pyclass(from_py_object)]
#[pyo3(name = "GuMapBuilder")]
#[derive(Clone)]
pub struct PyGuMapBuilder {
    pub(crate) builder: GuMapBuilder<FloatType>,
}

impl Default for PyGuMapBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyGuMapBuilder {
    /// Creates a new `GuMapBuilder` with default hyperparameters.
    ///
    /// :return: A new GuMapBuilder instance
    /// :rtype: GuMapBuilder
    #[new]
    pub fn new() -> Self {
        Self {
            builder: GuMapBuilder::new(),
        }
    }

    /// Sets the number of nearest neighbors `k` (default: 15).
    ///
    /// :param k: Number of nearest neighbors
    /// :type k: int
    /// :return: Self for method chaining
    /// :rtype: GuMapBuilder
    pub fn k(mut slf: PyRefMut<'_, Self>, k: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().k(k);
        slf
    }

    /// Sets the minimum distance in low-dimensional space (default: 0.1).
    ///
    /// :param min_dist: Minimum distance
    /// :type min_dist: float
    /// :return: Self for method chaining
    /// :rtype: GuMapBuilder
    pub fn min_dist(mut slf: PyRefMut<'_, Self>, min_dist: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().min_dist(min_dist);
        slf
    }

    /// Sets the scale of spread in low-dimensional space (default: 1.0).
    ///
    /// :param spread: Spread scale
    /// :type spread: float
    /// :return: Self for method chaining
    /// :rtype: GuMapBuilder
    pub fn spread(mut slf: PyRefMut<'_, Self>, spread: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().spread(spread);
        slf
    }

    /// Sets the number of SGD optimization epochs (default: 300).
    ///
    /// :param n_epochs: Number of epochs
    /// :type n_epochs: int
    /// :return: Self for method chaining
    /// :rtype: GuMapBuilder
    pub fn n_epochs(mut slf: PyRefMut<'_, Self>, n_epochs: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().n_epochs(n_epochs);
        slf
    }

    /// Sets the number of negative samples per attractive edge (default: 5).
    ///
    /// :param negative_sample_rate: Negative sampling rate
    /// :type negative_sample_rate: int
    /// :return: Self for method chaining
    /// :rtype: GuMapBuilder
    pub fn negative_sample_rate(
        mut slf: PyRefMut<'_, Self>,
        negative_sample_rate: usize,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf
            .builder
            .clone()
            .negative_sample_rate(negative_sample_rate);
        slf
    }

    /// Sets the initial learning rate (default: 1.0).
    ///
    /// :param learning_rate: Initial learning rate
    /// :type learning_rate: float
    /// :return: Self for method chaining
    /// :rtype: GuMapBuilder
    pub fn learning_rate(
        mut slf: PyRefMut<'_, Self>,
        learning_rate: FloatType,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().learning_rate(learning_rate);
        slf
    }

    /// Explicitly sets the `a` parameter for low-dimensional probability $1 / (1 + a d^{2b})$.
    ///
    /// :param a: Parameter a
    /// :type a: float
    /// :return: Self for method chaining
    /// :rtype: GuMapBuilder
    pub fn a(mut slf: PyRefMut<'_, Self>, a: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().a(a);
        slf
    }

    /// Explicitly sets the `b` parameter for low-dimensional probability $1 / (1 + a d^{2b})$.
    ///
    /// :param b: Parameter b
    /// :type b: float
    /// :return: Self for method chaining
    /// :rtype: GuMapBuilder
    pub fn b(mut slf: PyRefMut<'_, Self>, b: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().b(b);
        slf
    }

    /// Sets the maximum number of binary search iterations for finding sigma_i.
    ///
    /// :param sigma_iters: Maximum iterations
    /// :type sigma_iters: int
    /// :return: Self for method chaining
    /// :rtype: GuMapBuilder
    pub fn sigma_iters(mut slf: PyRefMut<'_, Self>, sigma_iters: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().sigma_iters(sigma_iters);
        slf
    }

    /// Sets the tolerance threshold for perplexity binary search convergence.
    ///
    /// :param sigma_tolerance: Tolerance value
    /// :type sigma_tolerance: float
    /// :return: Self for method chaining
    /// :rtype: GuMapBuilder
    pub fn sigma_tolerance(
        mut slf: PyRefMut<'_, Self>,
        sigma_tolerance: FloatType,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().sigma_tolerance(sigma_tolerance);
        slf
    }

    /// Builds a configured `GuMap` layout instance.
    ///
    /// :return: A new configured GuMap instance
    /// :rtype: GuMap
    pub fn build(&self) -> PyResult<PyGuMap> {
        let gumap = self
            .builder
            .clone()
            .build()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyGuMap { gumap })
    }
}

/// Python class for the GUMAP (Graph UMAP) layout algorithm.
///
/// GUMAP optimizes graph layouts using fuzzy simplicial sets and SGD with negative sampling.
/// It accepts any distance matrix implementation.
#[pyclass]
#[pyo3(name = "GuMap")]
pub struct PyGuMap {
    gumap: GuMap<FloatType>,
}

impl Default for PyGuMap {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyGuMap {
    /// Creates a new `GuMap` instance with default parameters.
    ///
    /// :return: A new GuMap instance
    /// :rtype: GuMap
    #[new]
    pub fn new() -> Self {
        Self {
            gumap: GuMap::new(),
        }
    }

    /// Creates a new `GuMapBuilder` instance.
    ///
    /// :return: A new GuMapBuilder instance
    /// :rtype: GuMapBuilder
    #[staticmethod]
    pub fn builder() -> PyGuMapBuilder {
        PyGuMapBuilder::new()
    }

    /// Runs the GUMAP layout algorithm on the drawing using the provided distance matrix and RNG.
    ///
    /// :param drawing: The drawing to update with optimized node coordinates
    /// :type drawing: DrawingEuclidean2d
    /// :param distance_matrix: The distance matrix to use (DistanceMatrix, DiffusionDistanceMatrix, etc.)
    /// :type distance_matrix: DistanceMatrix or DiffusionDistanceMatrix or EmbeddingDistanceMatrix or KernelDistance
    /// :param rng: Random number generator for SGD sampling
    /// :type rng: Rng
    /// :return: None
    /// :rtype: None
    pub fn run(
        &self,
        drawing: &mut PyDrawingEuclidean2d,
        distance_matrix: &Bound<PyAny>,
        rng: &mut PyRng,
    ) -> PyResult<()> {
        with_distance(distance_matrix, |d| {
            self.gumap.run(drawing.drawing_mut(), &d, rng.get_mut());
        })
    }
}

/// Registers GuMap and GuMapBuilder classes with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGuMapBuilder>()?;
    m.add_class::<PyGuMap>()?;
    Ok(())
}
