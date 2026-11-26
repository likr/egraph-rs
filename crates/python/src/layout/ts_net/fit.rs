//! Fast Interpolation tsNET (`FitTsNet`) layout algorithm bindings for Python.

use crate::{
    FloatType,
    drawing::PyDrawingEuclidean2d,
    graph::{GraphType, PyGraphAdapter},
    layout::knn::{
        extract_f64_array_1d, extract_index_array_1d_with_isize, extract_knn_rows, f64_array_as_cow,
    },
    rng::PyRng,
};
use pyo3::prelude::*;

/// Python class for constructing and configuring the `FitTsNet` algorithm via the Builder pattern.
#[pyclass(from_py_object)]
#[pyo3(name = "FitTsNetBuilder")]
#[derive(Clone)]
pub struct PyFitTsNetBuilder {
    pub(crate) builder: petgraph_layout_ts_net::FitTsNetBuilder<FloatType>,
}

impl Default for PyFitTsNetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyFitTsNetBuilder {
    /// Creates a new `FitTsNetBuilder` with default hyperparameters.
    ///
    /// :return: A new FitTsNetBuilder instance
    /// :rtype: FitTsNetBuilder
    #[new]
    pub fn new() -> Self {
        Self {
            builder: petgraph_layout_ts_net::FitTsNetBuilder::new(),
        }
    }

    /// Sets the number of intervals `I` for FFT interpolation (default: 25).
    pub fn intervals(mut slf: PyRefMut<'_, Self>, intervals: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().intervals(intervals);
        slf
    }

    /// Sets the number of interpolation points `P` per interval (default: 3).
    pub fn interpolation_points(
        mut slf: PyRefMut<'_, Self>,
        interpolation_points: usize,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf
            .builder
            .clone()
            .interpolation_points(interpolation_points);
        slf
    }

    /// Sets the target perplexity (default: 40.0).
    pub fn perplexity(mut slf: PyRefMut<'_, Self>, perplexity: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().perplexity(perplexity);
        slf
    }

    /// Sets the Barnes-Hut opening angle threshold theta (default: 0.5).
    pub fn theta(mut slf: PyRefMut<'_, Self>, theta: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().theta(theta);
        slf
    }

    /// Sets the number of nearest neighbors `k` for Partial BFS. Defaults to `3 * perplexity`.
    pub fn k(mut slf: PyRefMut<'_, Self>, k: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().k(k);
        slf
    }

    /// Sets the number of iterations for Stage 1 (early exaggeration).
    pub fn iterations_stage1(mut slf: PyRefMut<'_, Self>, iterations: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().iterations_stage1(iterations);
        slf
    }

    /// Sets the exaggeration factor for Stage 1.
    pub fn exaggeration(
        mut slf: PyRefMut<'_, Self>,
        exaggeration: FloatType,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().exaggeration(exaggeration);
        slf
    }

    /// Sets the number of iterations for Stage 2 (early compression / untangling).
    pub fn iterations_stage2(mut slf: PyRefMut<'_, Self>, iterations: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().iterations_stage2(iterations);
        slf
    }

    /// Sets the compression penalty parameter lambda_c for Stage 2.
    pub fn lambda_c_stage2(mut slf: PyRefMut<'_, Self>, lambda_c: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().lambda_c_stage2(lambda_c);
        slf
    }

    /// Sets the number of iterations for Stage 3 (final refinement).
    pub fn iterations_stage3(mut slf: PyRefMut<'_, Self>, iterations: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().iterations_stage3(iterations);
        slf
    }

    /// Sets the compression penalty parameter lambda_c for Stage 3.
    pub fn lambda_c_stage3(mut slf: PyRefMut<'_, Self>, lambda_c: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().lambda_c_stage3(lambda_c);
        slf
    }

    /// Sets the repulsion penalty parameter lambda_r for Stage 3.
    pub fn lambda_r_stage3(mut slf: PyRefMut<'_, Self>, lambda_r: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().lambda_r_stage3(lambda_r);
        slf
    }

    /// Sets the repulsion penalty parameter lambda_r for Stage 3 (alias).
    pub fn lambda_r(slf: PyRefMut<'_, Self>, lambda_r: FloatType) -> PyRefMut<'_, Self> {
        Self::lambda_r_stage3(slf, lambda_r)
    }

    /// Sets the learning rate for gradient descent.
    pub fn learning_rate(
        mut slf: PyRefMut<'_, Self>,
        learning_rate: FloatType,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().learning_rate(learning_rate);
        slf
    }

    /// Sets the momentum parameter for gradient descent.
    pub fn momentum(mut slf: PyRefMut<'_, Self>, momentum: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().momentum(momentum);
        slf
    }

    /// Sets the power exponent for distance matrix.
    pub fn power(mut slf: PyRefMut<'_, Self>, power: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().power(power);
        slf
    }

    /// Sets the repulsion parameter epsilon_r to prevent division by zero.
    pub fn epsilon_r(mut slf: PyRefMut<'_, Self>, epsilon_r: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().epsilon_r(epsilon_r);
        slf
    }

    /// Sets the maximum number of binary search iterations for finding sigma_i.
    pub fn sigma_iters(mut slf: PyRefMut<'_, Self>, sigma_iters: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().sigma_iters(sigma_iters);
        slf
    }

    /// Sets the tolerance threshold for perplexity binary search convergence.
    pub fn sigma_tolerance(
        mut slf: PyRefMut<'_, Self>,
        sigma_tolerance: FloatType,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().sigma_tolerance(sigma_tolerance);
        slf
    }

    /// Builds a configured `FitTsNet` layout instance.
    ///
    /// :return: A new configured FitTsNet instance
    /// :rtype: FitTsNet
    pub fn build(&self) -> PyResult<PyFitTsNet> {
        let fit_ts_net = self
            .builder
            .clone()
            .build()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyFitTsNet { fit_ts_net })
    }
}

/// Python class for the FIt-tsNET (Fast Interpolation tsNET) layout algorithm.
///
/// Reduces the time complexity by combining Partial BFS, 2D FFT interpolation for KL divergence, and Quadtree for entropy.
#[pyclass]
#[pyo3(name = "FitTsNet")]
pub struct PyFitTsNet {
    fit_ts_net: petgraph_layout_ts_net::FitTsNet<FloatType>,
}

impl Default for PyFitTsNet {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyFitTsNet {
    /// Creates a new `FitTsNet` instance with default parameters.
    ///
    /// :return: A new FitTsNet instance
    /// :rtype: FitTsNet
    #[new]
    pub fn new() -> Self {
        Self {
            fit_ts_net: petgraph_layout_ts_net::FitTsNet::new(),
        }
    }

    /// Creates a new `FitTsNetBuilder` instance.
    ///
    /// :return: A new FitTsNetBuilder instance
    /// :rtype: FitTsNetBuilder
    #[staticmethod]
    pub fn builder() -> PyFitTsNetBuilder {
        PyFitTsNetBuilder::new()
    }

    /// Runs the FIt-tsNET layout algorithm on the drawing using the provided graph and RNG.
    ///
    /// :param drawing: The drawing to update with optimized node coordinates
    /// :type drawing: DrawingEuclidean2d
    /// :param graph: The input graph
    /// :type graph: Graph
    /// :param rng: Random number generator for tie-breaking
    /// :type rng: Rng
    /// :return: None
    /// :rtype: None
    pub fn run(
        &self,
        drawing: &mut PyDrawingEuclidean2d,
        graph: &PyGraphAdapter,
        rng: &mut PyRng,
    ) -> PyResult<()> {
        match graph.graph() {
            GraphType::Graph(native_graph) => self
                .fit_ts_net
                .try_run(drawing.drawing_mut(), native_graph, rng.get_mut())
                .map_err(pyo3::exceptions::PyValueError::new_err),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "unsupported graph type for FitTsNet",
            )),
        }
    }

    /// Runs the FIt-tsNET layout algorithm on the drawing using precomputed k-nearest neighbors.
    ///
    /// :param drawing: The drawing to update with optimized node coordinates
    /// :type drawing: DrawingEuclidean2d
    /// :param knn_indices: 2D integer array of shape (n, k) containing neighbor node indices
    /// :type knn_indices: numpy.ndarray
    /// :param knn_distances: 2D float64 array of shape (n, k) containing neighbor distances
    /// :type knn_distances: numpy.ndarray
    /// :return: None
    /// :rtype: None
    #[pyo3(signature = (drawing, knn_indices, knn_distances))]
    pub fn run_with_knn(
        &self,
        drawing: &mut PyDrawingEuclidean2d,
        knn_indices: &Bound<'_, PyAny>,
        knn_distances: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let knn = extract_knn_rows(drawing.len(), knn_indices, knn_distances)?;
        self.fit_ts_net
            .try_run_with_knn(drawing.drawing_mut(), &knn)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Runs the FIt-tsNET layout algorithm on the drawing using variable-length CSR k-nearest
    /// neighbors and per-node target perplexities.
    ///
    /// :param drawing: The drawing to update with optimized node coordinates
    /// :type drawing: DrawingEuclidean2d
    /// :param indptr: 1D integer array of length n + 1 containing CSR row pointers
    /// :type indptr: numpy.ndarray
    /// :param indices: 1D integer array containing neighbor node indices
    /// :type indices: numpy.ndarray
    /// :param distances: 1D float64 array containing neighbor distances
    /// :type distances: numpy.ndarray
    /// :param perplexities: 1D float64 array of length n containing per-node target perplexities
    /// :type perplexities: numpy.ndarray
    /// :return: None
    /// :rtype: None
    #[pyo3(signature = (drawing, indptr, indices, distances, perplexities))]
    pub fn run_with_knn_adaptive(
        &self,
        drawing: &mut PyDrawingEuclidean2d,
        indptr: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        distances: &Bound<'_, PyAny>,
        perplexities: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let indptr_vec = extract_index_array_1d_with_isize(indptr, "indptr")?;
        let indices_vec = extract_index_array_1d_with_isize(indices, "indices")?;
        let dist_arr = extract_f64_array_1d(distances, "distances")?;
        let perp_arr = extract_f64_array_1d(perplexities, "perplexities")?;
        let dist_cow = f64_array_as_cow(&dist_arr);
        let perp_cow = f64_array_as_cow(&perp_arr);
        self.fit_ts_net
            .try_run_with_csr_knn_adaptive(
                drawing.drawing_mut(),
                &indptr_vec,
                &indices_vec,
                &dist_cow,
                &perp_cow,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFitTsNetBuilder>()?;
    m.add_class::<PyFitTsNet>()?;
    Ok(())
}
