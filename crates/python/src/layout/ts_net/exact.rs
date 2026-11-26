//! Exact tsNET layout algorithm bindings for Python.

use crate::{FloatType, distance_matrix::with_distance, drawing::PyDrawingEuclidean2d};
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
            self.ts_net
                .try_run(drawing.drawing_mut(), &d)
                .map_err(pyo3::exceptions::PyValueError::new_err)
        })?
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTsNetBuilder>()?;
    m.add_class::<PyTsNet>()?;
    Ok(())
}
