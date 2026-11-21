//! tsNET and BH-tsNET layout algorithm bindings for Python
//!
//! This module provides Python bindings for the tsNET and BH-tsNET graph layout algorithms and their builders.

use crate::{
    distance_matrix::with_distance,
    drawing::PyDrawingEuclidean2d,
    graph::{GraphType, PyGraphAdapter},
    rng::PyRng,
    FloatType,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods};
use petgraph_layout_ts_net::{BhTsNet, BhTsNetBuilder, TsNet, TsNetBuilder};
use pyo3::prelude::*;
use std::borrow::Cow;

fn build_knn_rows_from_view<'a, T, F>(
    indices: numpy::ndarray::ArrayView2<'a, T>,
    distances: numpy::ndarray::ArrayView2<'a, FloatType>,
    n: usize,
    mut convert_idx: F,
) -> PyResult<Vec<Vec<(usize, FloatType)>>>
where
    T: Copy,
    F: FnMut(T) -> PyResult<usize>,
{
    let (rows, cols) = indices.dim();
    if rows != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "knn_indices row count ({}) must match drawing node count ({})",
            rows, n
        )));
    }
    if distances.dim() != (rows, cols) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "knn_distances shape {:?} must match knn_indices shape {:?}",
            distances.dim(),
            (rows, cols)
        )));
    }

    let mut knn = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(cols);
        for j_pos in 0..cols {
            let idx = convert_idx(indices[[i, j_pos]])?;
            if idx >= n {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "knn_indices[{}, {}] = {} is out of range for drawing with {} nodes",
                    i, j_pos, idx, n
                )));
            }
            let dist = distances[[i, j_pos]];
            if !dist.is_finite() || dist < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "knn_distances[{}, {}] = {} must be finite and non-negative",
                    i, j_pos, dist
                )));
            }
            row.push((idx, dist));
        }
        knn.push(row);
    }
    Ok(knn)
}

fn check_2d_array(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        if untyped.ndim() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{} must be a 2D array, got {}D",
                name,
                untyped.ndim()
            )));
        }
    }
    Ok(())
}

fn extract_knn_rows(
    n: usize,
    knn_indices: &Bound<'_, PyAny>,
    knn_distances: &Bound<'_, PyAny>,
) -> PyResult<Vec<Vec<(usize, FloatType)>>> {
    check_2d_array(knn_indices, "knn_indices")?;
    check_2d_array(knn_distances, "knn_distances")?;

    let dist_arr = knn_distances
        .extract::<PyReadonlyArray2<'_, FloatType>>()
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("knn_distances must be a 2D float64 numpy array")
        })?;
    let dist_view = dist_arr.as_array();

    if let Ok(arr) = knn_indices.extract::<PyReadonlyArray2<'_, i64>>() {
        return build_knn_rows_from_view(arr.as_array(), dist_view, n, |v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "knn_indices must contain non-negative integers, got {}",
                    v
                ))
            })
        });
    }
    if let Ok(arr) = knn_indices.extract::<PyReadonlyArray2<'_, i32>>() {
        return build_knn_rows_from_view(arr.as_array(), dist_view, n, |v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "knn_indices must contain non-negative integers, got {}",
                    v
                ))
            })
        });
    }
    if let Ok(arr) = knn_indices.extract::<PyReadonlyArray2<'_, usize>>() {
        return build_knn_rows_from_view(arr.as_array(), dist_view, n, Ok);
    }
    if let Ok(arr) = knn_indices.extract::<PyReadonlyArray2<'_, isize>>() {
        return build_knn_rows_from_view(arr.as_array(), dist_view, n, |v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "knn_indices must contain non-negative integers, got {}",
                    v
                ))
            })
        });
    }
    if let Ok(arr) = knn_indices.extract::<PyReadonlyArray2<'_, u64>>() {
        return build_knn_rows_from_view(arr.as_array(), dist_view, n, |v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "knn_indices value {} overflows usize",
                    v
                ))
            })
        });
    }
    if let Ok(arr) = knn_indices.extract::<PyReadonlyArray2<'_, u32>>() {
        return build_knn_rows_from_view(arr.as_array(), dist_view, n, |v| Ok(v as usize));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "knn_indices must be a 2D numpy integer array (int64, int32, or uintp/usize)",
    ))
}

fn check_1d_array(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        if untyped.ndim() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{} must be a 1D array, got {}D",
                name,
                untyped.ndim()
            )));
        }
    }
    Ok(())
}

fn convert_index_view_1d<T, F>(
    view: numpy::ndarray::ArrayView1<'_, T>,
    mut convert: F,
) -> PyResult<Vec<usize>>
where
    T: Copy,
    F: FnMut(T) -> PyResult<usize>,
{
    let mut out = Vec::with_capacity(view.len());
    for &val in view.iter() {
        out.push(convert(val)?);
    }
    Ok(out)
}

fn extract_index_array_1d_with_isize(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<usize>> {
    // Check ndim before dtype extraction so wrong-dimension arrays consistently raise PyValueError.
    check_1d_array(obj, name)?;

    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i64>>() {
        return convert_index_view_1d(arr.as_array(), |v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "{} must contain non-negative integers, got {}",
                    name, v
                ))
            })
        });
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i32>>() {
        return convert_index_view_1d(arr.as_array(), |v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "{} must contain non-negative integers, got {}",
                    name, v
                ))
            })
        });
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, usize>>() {
        return convert_index_view_1d(arr.as_array(), Ok);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, isize>>() {
        return convert_index_view_1d(arr.as_array(), |v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "{} must contain non-negative integers, got {}",
                    name, v
                ))
            })
        });
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u64>>() {
        return convert_index_view_1d(arr.as_array(), |v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "{} value {} overflows usize",
                    name, v
                ))
            })
        });
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u32>>() {
        return convert_index_view_1d(arr.as_array(), |v| Ok(v as usize));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{} must be a 1D numpy integer array (int64, int32, uintp/usize, intp/isize, uint64, or uint32)",
        name
    )))
}

fn extract_f64_array_1d<'py>(
    obj: &'py Bound<'py, PyAny>,
    name: &str,
) -> PyResult<PyReadonlyArray1<'py, FloatType>> {
    check_1d_array(obj, name)?;
    obj.extract::<PyReadonlyArray1<'py, FloatType>>()
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "{} must be a 1D float64 numpy array",
                name
            ))
        })
}

fn f64_array_as_cow<'a, 'py>(arr: &'a PyReadonlyArray1<'py, FloatType>) -> Cow<'a, [FloatType]> {
    match arr.as_slice() {
        Ok(slice) => Cow::Borrowed(slice),
        Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

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

    /// Creates a new `TsNetBuilder` initialized with tsNET* hyperparameters (`initial_l_c = 0.1`).
    ///
    /// :return: A new TsNetBuilder instance for tsNET*
    /// :rtype: TsNetBuilder
    #[staticmethod]
    pub fn new_star() -> Self {
        Self {
            builder: TsNetBuilder::new_star(),
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

    /// Sets the total number of epochs (iterations).
    ///
    /// :param n_epochs: Number of epochs
    /// :type n_epochs: int
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn n_epochs(mut slf: PyRefMut<'_, Self>, n_epochs: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().n_epochs(n_epochs);
        slf
    }

    /// Sets the initial learning rate.
    ///
    /// :param initial_lr: Initial learning rate
    /// :type initial_lr: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn initial_lr(mut slf: PyRefMut<'_, Self>, initial_lr: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().initial_lr(initial_lr);
        slf
    }

    /// Sets the final learning rate.
    ///
    /// :param final_lr: Final learning rate
    /// :type final_lr: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn final_lr(mut slf: PyRefMut<'_, Self>, final_lr: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().final_lr(final_lr);
        slf
    }

    /// Sets the epoch at which learning rate switches from `initial_lr` to `final_lr`.
    ///
    /// :param lr_switch: Switch epoch index (or None for n_epochs / 2)
    /// :type lr_switch: int or None
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn lr_switch(mut slf: PyRefMut<'_, Self>, lr_switch: Option<usize>) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().lr_switch(lr_switch);
        slf
    }

    /// Sets both the initial and final learning rate to the same value.
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

    /// Sets the initial momentum parameter.
    ///
    /// :param initial_momentum: Initial momentum in `[0, 1)`
    /// :type initial_momentum: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn initial_momentum(
        mut slf: PyRefMut<'_, Self>,
        initial_momentum: FloatType,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().initial_momentum(initial_momentum);
        slf
    }

    /// Sets the final momentum parameter.
    ///
    /// :param final_momentum: Final momentum in `[0, 1)`
    /// :type final_momentum: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn final_momentum(
        mut slf: PyRefMut<'_, Self>,
        final_momentum: FloatType,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().final_momentum(final_momentum);
        slf
    }

    /// Sets the epoch at which momentum switches.
    ///
    /// :param momentum_switch: Switch epoch index (or None for n_epochs / 2)
    /// :type momentum_switch: int or None
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn momentum_switch(
        mut slf: PyRefMut<'_, Self>,
        momentum_switch: Option<usize>,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().momentum_switch(momentum_switch);
        slf
    }

    /// Sets both the initial and final momentum parameter to the same value.
    ///
    /// :param momentum: Momentum parameter in `[0, 1)`
    /// :type momentum: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn momentum(mut slf: PyRefMut<'_, Self>, momentum: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().momentum(momentum);
        slf
    }

    /// Sets the initial KL divergence weight.
    ///
    /// :param initial_l_kl: Initial weight for KL divergence term (or None)
    /// :type initial_l_kl: float or None
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn initial_l_kl(
        mut slf: PyRefMut<'_, Self>,
        initial_l_kl: Option<FloatType>,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().initial_l_kl(initial_l_kl);
        slf
    }

    /// Sets the final KL divergence weight.
    ///
    /// :param final_l_kl: Final weight for KL divergence term (or None)
    /// :type final_l_kl: float or None
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn final_l_kl(
        mut slf: PyRefMut<'_, Self>,
        final_l_kl: Option<FloatType>,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().final_l_kl(final_l_kl);
        slf
    }

    /// Sets the epoch at which lambda_kl switches.
    ///
    /// :param l_kl_switch: Switch epoch index (or None for n_epochs / 2)
    /// :type l_kl_switch: int or None
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn l_kl_switch(
        mut slf: PyRefMut<'_, Self>,
        l_kl_switch: Option<usize>,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().l_kl_switch(l_kl_switch);
        slf
    }

    /// Sets the initial compression weight.
    ///
    /// :param initial_l_c: Initial compression weight (or None)
    /// :type initial_l_c: float or None
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn initial_l_c(
        mut slf: PyRefMut<'_, Self>,
        initial_l_c: Option<FloatType>,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().initial_l_c(initial_l_c);
        slf
    }

    /// Sets the final compression weight.
    ///
    /// :param final_l_c: Final compression weight (or None)
    /// :type final_l_c: float or None
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn final_l_c(
        mut slf: PyRefMut<'_, Self>,
        final_l_c: Option<FloatType>,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().final_l_c(final_l_c);
        slf
    }

    /// Sets the epoch at which lambda_c switches.
    ///
    /// :param l_c_switch: Switch epoch index (or None for n_epochs / 2)
    /// :type l_c_switch: int or None
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn l_c_switch(
        mut slf: PyRefMut<'_, Self>,
        l_c_switch: Option<usize>,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().l_c_switch(l_c_switch);
        slf
    }

    /// Sets the initial repulsion weight.
    ///
    /// :param initial_l_r: Initial repulsion weight (or None)
    /// :type initial_l_r: float or None
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn initial_l_r(
        mut slf: PyRefMut<'_, Self>,
        initial_l_r: Option<FloatType>,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().initial_l_r(initial_l_r);
        slf
    }

    /// Sets the final repulsion weight.
    ///
    /// :param final_l_r: Final repulsion weight (or None)
    /// :type final_l_r: float or None
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn final_l_r(
        mut slf: PyRefMut<'_, Self>,
        final_l_r: Option<FloatType>,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().final_l_r(final_l_r);
        slf
    }

    /// Sets the epoch at which lambda_r switches.
    ///
    /// :param l_r_switch: Switch epoch index (or None for n_epochs / 2)
    /// :type l_r_switch: int or None
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn l_r_switch(
        mut slf: PyRefMut<'_, Self>,
        l_r_switch: Option<usize>,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().l_r_switch(l_r_switch);
        slf
    }

    /// Sets the repulsion offset parameter `r_eps` to prevent division by zero in repulsion force.
    ///
    /// :param r_eps: Repulsion parameter offset
    /// :type r_eps: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn r_eps(mut slf: PyRefMut<'_, Self>, r_eps: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().r_eps(r_eps);
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

    /// Sets the autostop convergence tolerance threshold.
    ///
    /// :param autostop: Autostop convergence tolerance
    /// :type autostop: float
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn autostop(mut slf: PyRefMut<'_, Self>, autostop: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().autostop(autostop);
        slf
    }

    /// Sets the sliding window size for checking autostop convergence.
    ///
    /// :param window_size: Window size in epochs
    /// :type window_size: int
    /// :return: Self for method chaining
    /// :rtype: TsNetBuilder
    pub fn window_size(mut slf: PyRefMut<'_, Self>, window_size: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().window_size(window_size);
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

    /// Creates a new `TsNet` instance with tsNET* parameters (`initial_l_c = 0.1`).
    ///
    /// :return: A new TsNet instance for tsNET*
    /// :rtype: TsNet
    #[staticmethod]
    pub fn new_star() -> Self {
        Self {
            ts_net: TsNet::new_star(),
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

    /// Creates a new `TsNetBuilder` instance with tsNET* hyperparameters.
    ///
    /// :return: A new TsNetBuilder instance for tsNET*
    /// :rtype: TsNetBuilder
    #[staticmethod]
    pub fn builder_star() -> PyTsNetBuilder {
        PyTsNetBuilder::new_star()
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

/// Python class for constructing and configuring the `BhTsNet` algorithm via the Builder pattern.
#[pyclass(from_py_object)]
#[pyo3(name = "BhTsNetBuilder")]
#[derive(Clone)]
pub struct PyBhTsNetBuilder {
    pub(crate) builder: BhTsNetBuilder<FloatType>,
}

impl Default for PyBhTsNetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyBhTsNetBuilder {
    /// Creates a new `BhTsNetBuilder` with default hyperparameters.
    ///
    /// :return: A new BhTsNetBuilder instance
    /// :rtype: BhTsNetBuilder
    #[new]
    pub fn new() -> Self {
        Self {
            builder: BhTsNetBuilder::new(),
        }
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

    /// Builds a configured `BhTsNet` layout instance.
    ///
    /// :return: A new configured BhTsNet instance
    /// :rtype: BhTsNet
    pub fn build(&self) -> PyResult<PyBhTsNet> {
        let bh_ts_net = self
            .builder
            .clone()
            .build()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyBhTsNet { bh_ts_net })
    }
}

/// Python class for the BH-tsNET (Barnes-Hut tsNET) layout algorithm.
///
/// Reduces the time complexity to O(N log N) using Partial BFS and Quadtree Barnes-Hut spatial force approximation.
#[pyclass]
#[pyo3(name = "BhTsNet")]
pub struct PyBhTsNet {
    bh_ts_net: BhTsNet<FloatType>,
}

impl Default for PyBhTsNet {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyBhTsNet {
    /// Creates a new `BhTsNet` instance with default parameters.
    ///
    /// :return: A new BhTsNet instance
    /// :rtype: BhTsNet
    #[new]
    pub fn new() -> Self {
        Self {
            bh_ts_net: BhTsNet::new(),
        }
    }

    /// Creates a new `BhTsNetBuilder` instance.
    ///
    /// :return: A new BhTsNetBuilder instance
    /// :rtype: BhTsNetBuilder
    #[staticmethod]
    pub fn builder() -> PyBhTsNetBuilder {
        PyBhTsNetBuilder::new()
    }

    /// Runs the BH-tsNET layout algorithm on the drawing using the provided graph and RNG.
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
            GraphType::Graph(native_graph) => {
                self.bh_ts_net
                    .run(drawing.drawing_mut(), native_graph, rng.get_mut());
                Ok(())
            }
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "unsupported graph type for BhTsNet",
            )),
        }
    }

    /// Runs the BH-tsNET layout algorithm on the drawing using precomputed k-nearest neighbors.
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
        self.bh_ts_net.run_with_knn(drawing.drawing_mut(), &knn);
        Ok(())
    }

    /// Runs the BH-tsNET layout algorithm on the drawing using variable-length CSR k-nearest
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
        self.bh_ts_net
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
            GraphType::Graph(native_graph) => {
                self.fit_ts_net
                    .run(drawing.drawing_mut(), native_graph, rng.get_mut());
                Ok(())
            }
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
        self.fit_ts_net.run_with_knn(drawing.drawing_mut(), &knn);
        Ok(())
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

/// Python class for constructing and configuring the `LTsNet` algorithm via the Builder pattern.
#[pyclass(from_py_object)]
#[pyo3(name = "LTsNetBuilder")]
#[derive(Clone)]
pub struct PyLTsNetBuilder {
    pub(crate) builder: petgraph_layout_ts_net::LTsNetBuilder<FloatType>,
}

impl Default for PyLTsNetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyLTsNetBuilder {
    /// Creates a new `LTsNetBuilder` with default hyperparameters.
    ///
    /// :return: A new LTsNetBuilder instance
    /// :rtype: LTsNetBuilder
    #[new]
    pub fn new() -> Self {
        Self {
            builder: petgraph_layout_ts_net::LTsNetBuilder::new(),
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

    /// Sets the repulsion parameter epsilon_r (default: 0.05).
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

    /// Builds a configured `LTsNet` layout instance.
    ///
    /// :return: A new configured LTsNet instance
    /// :rtype: LTsNet
    pub fn build(&self) -> PyResult<PyLTsNet> {
        let l_ts_net = self
            .builder
            .clone()
            .build()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyLTsNet { l_ts_net })
    }
}

/// Python class for the L-tsNET (Linear-tsNET) layout algorithm.
///
/// Achieves strictly linear O(N) time complexity by extending 2D FFT interpolation to both KL divergence and entropy.
#[pyclass]
#[pyo3(name = "LTsNet")]
pub struct PyLTsNet {
    l_ts_net: petgraph_layout_ts_net::LTsNet<FloatType>,
}

impl Default for PyLTsNet {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyLTsNet {
    /// Creates a new `LTsNet` instance with default parameters.
    ///
    /// :return: A new LTsNet instance
    /// :rtype: LTsNet
    #[new]
    pub fn new() -> Self {
        Self {
            l_ts_net: petgraph_layout_ts_net::LTsNet::new(),
        }
    }

    /// Creates a new `LTsNetBuilder` instance.
    ///
    /// :return: A new LTsNetBuilder instance
    /// :rtype: LTsNetBuilder
    #[staticmethod]
    pub fn builder() -> PyLTsNetBuilder {
        PyLTsNetBuilder::new()
    }

    /// Runs the L-tsNET layout algorithm on the drawing using the provided graph and RNG.
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
            GraphType::Graph(native_graph) => {
                self.l_ts_net
                    .run(drawing.drawing_mut(), native_graph, rng.get_mut());
                Ok(())
            }
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "unsupported graph type for LTsNet",
            )),
        }
    }

    /// Runs the L-tsNET layout algorithm on the drawing using precomputed k-nearest neighbors.
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
        self.l_ts_net.run_with_knn(drawing.drawing_mut(), &knn);
        Ok(())
    }

    /// Runs the L-tsNET layout algorithm on the drawing using variable-length CSR k-nearest
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
        self.l_ts_net
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

/// Registers TsNet, TsNetBuilder, BhTsNet, BhTsNetBuilder, FitTsNet, FitTsNetBuilder, LTsNet, and LTsNetBuilder classes with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTsNetBuilder>()?;
    m.add_class::<PyTsNet>()?;
    m.add_class::<PyBhTsNetBuilder>()?;
    m.add_class::<PyBhTsNet>()?;
    m.add_class::<PyFitTsNetBuilder>()?;
    m.add_class::<PyFitTsNet>()?;
    m.add_class::<PyLTsNetBuilder>()?;
    m.add_class::<PyLTsNet>()?;
    Ok(())
}
