//! tsNET layout algorithm bindings for Python
//!
//! This module provides Python bindings for the tsNET graph layout algorithm and its builder.

use crate::{distance_matrix::with_distance, drawing::PyDrawingEuclidean2d, FloatType};
use petgraph_layout_ts_net::{TsNet, TsNetBuilder};
use pyo3::prelude::*;

/// Python class for constructing and configuring the `TsNet` algorithm via the Builder pattern.
#[pyclass(from_py_object)]
#[pyo3(name = "TsNetBuilder")]
#[derive(Clone)]
pub struct PyTsNetBuilder {
    pub(crate) builder: TsNetBuilder<FloatType>,
}

impl Default for PyTsNetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyTsNetBuilder {
    /// Creates a new `TsNetBuilder` with default hyperparameters.
    ///
    /// :return: A new TsNetBuilder instance
    /// :rtype: TsNetBuilder
    #[new]
    pub fn new() -> Self {
        Self {
            builder: TsNetBuilder::new(),
        }
    }

    /// Sets the target perplexity.
    ///
    /// :param perplexity: Target perplexity for t-SNE probability distribution
    /// :type perplexity: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn perplexity(mut slf: PyRefMut<'_, Self>, perplexity: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().perplexity(perplexity);
        slf
    }

    /// Sets the number of iterations for Stage 1 (early exaggeration).
    ///
    /// :param iterations: Number of iterations
    /// :type iterations: int
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn iterations_stage1(mut slf: PyRefMut<'_, Self>, iterations: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().iterations_stage1(iterations);
        slf
    }

    /// Sets the exaggeration factor for Stage 1.
    ///
    /// :param exaggeration: Exaggeration factor
    /// :type exaggeration: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn exaggeration(
        mut slf: PyRefMut<'_, Self>,
        exaggeration: FloatType,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().exaggeration(exaggeration);
        slf
    }

    /// Sets the number of iterations for Stage 2 (early compression / untangling).
    ///
    /// :param iterations: Number of iterations
    /// :type iterations: int
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn iterations_stage2(mut slf: PyRefMut<'_, Self>, iterations: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().iterations_stage2(iterations);
        slf
    }

    /// Sets the compression penalty parameter lambda_c for Stage 2.
    ///
    /// :param lambda_c: Compression penalty parameter
    /// :type lambda_c: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn lambda_c_stage2(mut slf: PyRefMut<'_, Self>, lambda_c: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().lambda_c_stage2(lambda_c);
        slf
    }

    /// Sets the number of iterations for Stage 3 (final refinement).
    ///
    /// :param iterations: Number of iterations
    /// :type iterations: int
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn iterations_stage3(mut slf: PyRefMut<'_, Self>, iterations: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().iterations_stage3(iterations);
        slf
    }

    /// Sets the compression penalty parameter lambda_c for Stage 3.
    ///
    /// :param lambda_c: Compression penalty parameter
    /// :type lambda_c: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn lambda_c_stage3(mut slf: PyRefMut<'_, Self>, lambda_c: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().lambda_c_stage3(lambda_c);
        slf
    }

    /// Sets the repulsion penalty parameter lambda_r for Stage 3.
    ///
    /// :param lambda_r: Repulsion penalty parameter
    /// :type lambda_r: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn lambda_r_stage3(mut slf: PyRefMut<'_, Self>, lambda_r: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().lambda_r_stage3(lambda_r);
        slf
    }

    /// Sets the repulsion penalty parameter lambda_r for Stage 3 (alias for backward compatibility).
    pub fn lambda_r(slf: PyRefMut<'_, Self>, lambda_r: FloatType) -> PyRefMut<'_, Self> {
        Self::lambda_r_stage3(slf, lambda_r)
    }

    /// Sets the learning rate for gradient descent.
    ///
    /// :param learning_rate: Learning rate value
    /// :type learning_rate: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn learning_rate(
        mut slf: PyRefMut<'_, Self>,
        learning_rate: FloatType,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().learning_rate(learning_rate);
        slf
    }

    /// Sets the momentum parameter for gradient descent.
    ///
    /// :param momentum: Momentum parameter in `[0, 1)`
    /// :type momentum: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn momentum(mut slf: PyRefMut<'_, Self>, momentum: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().momentum(momentum);
        slf
    }

    /// Sets the power exponent for distance matrix.
    ///
    /// :param power: Power exponent value (e.g. 2.0)
    /// :type power: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn power(mut slf: PyRefMut<'_, Self>, power: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().power(power);
        slf
    }

    /// Sets the repulsion parameter epsilon_r to prevent division by zero.
    ///
    /// :param epsilon_r: Repulsion parameter offset
    /// :type epsilon_r: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn epsilon_r(mut slf: PyRefMut<'_, Self>, epsilon_r: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().epsilon_r(epsilon_r);
        slf
    }

    /// Sets the maximum number of binary search iterations for finding sigma_i.
    ///
    /// :param sigma_iters: Number of binary search iterations
    /// :type sigma_iters: int
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn sigma_iters(mut slf: PyRefMut<'_, Self>, sigma_iters: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().sigma_iters(sigma_iters);
        slf
    }

    /// Sets the tolerance threshold for perplexity binary search convergence.
    ///
    /// :param sigma_tolerance: Convergence tolerance
    /// :type sigma_tolerance: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn sigma_tolerance(
        mut slf: PyRefMut<'_, Self>,
        sigma_tolerance: FloatType,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().sigma_tolerance(sigma_tolerance);
        slf
    }

    /// Builds a configured `TsNet` layout instance.
    ///
    /// :return: A new configured TsNet instance
    /// :rtype: TsNet
    pub fn build(&self) -> PyResult<PyTsNet> {
        let ts_net = self
            .builder
            .clone()
            .build()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyTsNet { ts_net })
    }
}

/// Python class for the tsNET layout algorithm.
///
/// tsNET optimizes graph layouts using a t-SNE-like objective function with early compression
/// and node repulsion. It accepts any distance matrix implementation.
#[pyclass]
#[pyo3(name = "TsNet")]
pub struct PyTsNet {
    ts_net: TsNet<FloatType>,
}

impl Default for PyTsNet {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyTsNet {
    /// Creates a new `TsNet` instance with default parameters.
    ///
    /// :return: A new TsNet instance
    /// :rtype: TsNet
    #[new]
    pub fn new() -> Self {
        Self {
            ts_net: TsNet::new(),
        }
    }

    /// Creates a new `TsNetBuilder` instance.
    ///
    /// :return: A new TsNetBuilder instance
    /// :rtype: TsNetBuilder
    #[staticmethod]
    pub fn builder() -> PyTsNetBuilder {
        PyTsNetBuilder::new()
    }

    /// Runs the tsNET layout algorithm on the drawing using the provided distance matrix.
    ///
    /// :param drawing: The drawing to update with optimized node coordinates
    /// :type drawing: DrawingEuclidean2d
    /// :param distance_matrix: The distance matrix to use (DistanceMatrix, DiffusionDistanceMatrix, etc.)
    /// :type distance_matrix: DistanceMatrix or DiffusionDistanceMatrix or EmbeddingDistanceMatrix or KernelDistance
    /// :return: None
    /// :rtype: None
    pub fn run(
        &self,
        drawing: &mut PyDrawingEuclidean2d,
        distance_matrix: &Bound<PyAny>,
    ) -> PyResult<()> {
        with_distance(distance_matrix, |d| {
            self.ts_net.run(drawing.drawing_mut(), &d);
        })
    }
}

/// Registers TsNet and TsNetBuilder classes with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTsNetBuilder>()?;
    m.add_class::<PyTsNet>()?;
    Ok(())
}
