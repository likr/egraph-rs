//! tsNET layout algorithm bindings for Python
//!
//! This module provides Python bindings for the tsNET graph layout algorithm.

use crate::{distance_matrix::with_distance, drawing::PyDrawingEuclidean2d, FloatType};
use petgraph_layout_ts_net::TsNet;
use pyo3::prelude::*;

/// Python class for the tsNET layout algorithm
///
/// tsNET optimizes graph layouts using a t-SNE-like objective function with early compression
/// and entropy-based node repulsion. It accepts any distance matrix implementation.
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
    /// Creates a new TsNet instance with default parameters
    ///
    /// :return: A new TsNet instance
    /// :rtype: TsNet
    #[new]
    pub fn new() -> Self {
        Self {
            ts_net: TsNet::new(),
        }
    }

    /// Sets the target perplexity
    ///
    /// :param perplexity: Target perplexity for t-SNE probability distribution
    /// :type perplexity: float
    /// :return: Self for method chaining
    /// :rtype: TsNet
    pub fn perplexity(mut slf: PyRefMut<Self>, perplexity: FloatType) -> Py<Self> {
        slf.ts_net.perplexity(perplexity);
        slf.into()
    }

    /// Sets the number of iterations for Stage 1 (early exaggeration)
    ///
    /// :param iterations: Number of iterations
    /// :type iterations: int
    /// :return: Self for method chaining
    /// :rtype: TsNet
    pub fn iterations_stage1(mut slf: PyRefMut<Self>, iterations: usize) -> Py<Self> {
        slf.ts_net.iterations_stage1(iterations);
        slf.into()
    }

    /// Sets the exaggeration factor for Stage 1
    ///
    /// :param exaggeration: Exaggeration factor
    /// :type exaggeration: float
    /// :return: Self for method chaining
    /// :rtype: TsNet
    pub fn exaggeration(mut slf: PyRefMut<Self>, exaggeration: FloatType) -> Py<Self> {
        slf.ts_net.exaggeration(exaggeration);
        slf.into()
    }

    /// Sets the power exponent for distance matrix
    ///
    /// :param power: Power exponent value (e.g. 1.0, 2.0)
    /// :type power: float
    /// :return: Self for method chaining
    /// :rtype: TsNet
    pub fn power(mut slf: PyRefMut<Self>, power: FloatType) -> Py<Self> {
        slf.ts_net.power(power);
        slf.into()
    }

    /// Sets the repulsion weight lambda_r for Stage 3
    ///
    /// :param lambda_r: Repulsion weight
    /// :type lambda_r: float
    /// :return: Self for method chaining
    /// :rtype: TsNet
    pub fn lambda_r(mut slf: PyRefMut<Self>, lambda_r: FloatType) -> Py<Self> {
        slf.ts_net.lambda_r(lambda_r);
        slf.into()
    }

    /// Sets the number of iterations for Stage 2 (compression)
    ///
    /// :param iterations: Number of iterations
    /// :type iterations: int
    /// :return: Self for method chaining
    /// :rtype: TsNet
    pub fn iterations_stage2(mut slf: PyRefMut<Self>, iterations: usize) -> Py<Self> {
        slf.ts_net.iterations_stage2(iterations);
        slf.into()
    }

    /// Sets the number of iterations for Stage 3 (refinement)
    ///
    /// :param iterations: Number of iterations
    /// :type iterations: int
    /// :return: Self for method chaining
    /// :rtype: TsNet
    pub fn iterations_stage3(mut slf: PyRefMut<Self>, iterations: usize) -> Py<Self> {
        slf.ts_net.iterations_stage3(iterations);
        slf.into()
    }

    /// Sets the learning rate for gradient descent
    ///
    /// :param learning_rate: Learning rate value
    /// :type learning_rate: float
    /// :return: Self for method chaining
    /// :rtype: TsNet
    pub fn learning_rate(mut slf: PyRefMut<Self>, learning_rate: FloatType) -> Py<Self> {
        slf.ts_net.learning_rate(learning_rate);
        slf.into()
    }

    /// Sets the momentum parameter for gradient descent
    ///
    /// :param momentum: Momentum value
    /// :type momentum: float
    /// :return: Self for method chaining
    /// :rtype: TsNet
    pub fn momentum(mut slf: PyRefMut<Self>, momentum: FloatType) -> Py<Self> {
        slf.ts_net.momentum(momentum);
        slf.into()
    }

    /// Sets the minimum distance parameter epsilon_d for input space
    ///
    /// :param epsilon_d: Minimum distance value
    /// :type epsilon_d: float
    /// :return: Self for method chaining
    /// :rtype: TsNet
    pub fn epsilon_d(mut slf: PyRefMut<Self>, epsilon_d: FloatType) -> Py<Self> {
        slf.ts_net.epsilon_d(epsilon_d);
        slf.into()
    }

    /// Sets the repulsion parameter epsilon_r to prevent zero division
    ///
    /// :param epsilon_r: Repulsion parameter value
    /// :type epsilon_r: float
    /// :return: Self for method chaining
    /// :rtype: TsNet
    pub fn epsilon_r(mut slf: PyRefMut<Self>, epsilon_r: FloatType) -> Py<Self> {
        slf.ts_net.epsilon_r(epsilon_r);
        slf.into()
    }

    /// Runs the tsNET layout algorithm on the drawing using the provided distance matrix
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

/// Registers TsNet class with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTsNet>()?;
    Ok(())
}
