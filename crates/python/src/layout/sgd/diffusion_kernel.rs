use crate::{
    array::{PyArray1, PyArray2},
    distance_matrix::{EdgeLengthFn, PyLaplacian},
    FloatType,
};
use petgraph::visit::EdgeRef;
use petgraph_linalg_kernel::Kernel;
use petgraph_linalg_kernel::{
    dog_bandpass_terms, BandPassDiffusionKernel, BandPassDiffusionKernelBuilder, DiffusionKernel,
    DiffusionKernelBuilder, LowRankDiffusionKernel, LowRankDiffusionKernelBuilder,
    LowRankMultiscaleDiffusionKernel, LowRankMultiscaleDiffusionKernelBuilder,
    MultiscaleDiffusionKernel, MultiscaleDiffusionKernelBuilder, PivotedBandPassDiffusionKernel,
    PivotedBandPassDiffusionKernelBuilder, PivotedDiffusionKernel, PivotedDiffusionKernelBuilder,
    PivotedKernel, PivotedMultiscaleDiffusionKernel, PivotedMultiscaleDiffusionKernelBuilder,
};
use pyo3::prelude::*;
use rand::SeedableRng;

fn with_optional_rng<R>(
    rng: Option<&mut crate::rng::PyRng>,
    f: impl FnOnce(&mut rand::rngs::StdRng) -> R,
) -> R {
    match rng {
        Some(r) => f(r.get_mut()),
        None => {
            let mut local = rand::rngs::StdRng::from_entropy();
            f(&mut local)
        }
    }
}

fn with_optional_rng_ref_mut<R>(
    rng: Option<PyRefMut<'_, crate::rng::PyRng>>,
    f: impl FnOnce(&mut rand::rngs::StdRng) -> R,
) -> R {
    match rng {
        Some(mut r) => f(r.get_mut()),
        None => {
            let mut local = rand::rngs::StdRng::from_entropy();
            f(&mut local)
        }
    }
}

fn resolve_length_and_rng<'py>(
    length: Option<&Bound<'py, PyAny>>,
    rng: Option<&Bound<'py, crate::rng::PyRng>>,
) -> PyResult<(EdgeLengthFn, Option<PyRefMut<'py, crate::rng::PyRng>>)> {
    if let Some(rng_bound) = rng {
        return Ok((
            EdgeLengthFn::from_bound(length)?,
            Some(rng_bound.borrow_mut()),
        ));
    }
    if let Some(first) = length {
        if let Ok(rng_bound) = first.cast::<crate::rng::PyRng>() {
            return Ok((EdgeLengthFn::Constant(1.0), Some(rng_bound.borrow_mut())));
        }
    }
    Ok((EdgeLengthFn::from_bound(length)?, None))
}

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
        self.kernel.get(i, j).unwrap()
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
    fn get(&self, i: usize, j: usize) -> FloatType {
        self.kernel.get(i, j).unwrap()
    }

    /// Returns the number of nodes in the graph
    fn n(&self) -> usize {
        self.kernel.shape().0
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
    #[new]
    #[pyo3(signature = (t, eigenvalues, eigenvectors))]
    fn new(t: FloatType, eigenvalues: &PyArray1, eigenvectors: &PyArray2) -> PyResult<Self> {
        let kernel = LowRankDiffusionKernel::new(
            t,
            eigenvalues.as_array().clone(),
            eigenvectors.as_array().clone(),
        );
        Ok(PyLowRankDiffusionKernel { kernel })
    }

    /// Queries the (i, j) element of the low-rank heat kernel matrix
    fn get(&self, i: usize, j: usize) -> FloatType {
        self.kernel.get(i, j).unwrap()
    }

    /// Returns the number of nodes in the graph
    fn n(&self) -> usize {
        self.kernel.shape().0
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
    #[new]
    #[pyo3(signature = (alpha, eigenvalues, eigenvectors))]
    fn new(alpha: FloatType, eigenvalues: &PyArray1, eigenvectors: &PyArray2) -> PyResult<Self> {
        if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "alpha must be finite and in [0, 1)",
            ));
        }
        let kernel = LowRankMultiscaleDiffusionKernel::new(
            alpha,
            eigenvalues.as_array().clone(),
            eigenvectors.as_array().clone(),
        );
        Ok(PyLowRankMultiscaleDiffusionKernel { kernel })
    }

    /// Queries the (i, j) element of the low-rank multiscale kernel matrix
    fn get(&self, i: usize, j: usize) -> FloatType {
        self.kernel.get(i, j).unwrap()
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
    fn get_from_pivot(&self, pivot_idx: usize, j: usize) -> FloatType {
        self.kernel.get_from_pivot(pivot_idx, j)
    }

    /// Returns the estimated diagonal entry `K[j, j]` if Hutchinson estimation was enabled
    fn diagonal(&self, j: usize) -> Option<FloatType> {
        self.kernel.diagonal(j)
    }
}

fn build_bandpass_kernel(
    laplacian: &PyLaplacian,
    terms: Vec<(FloatType, FloatType)>,
    degree: usize,
    lambda_max: Option<FloatType>,
    rng: Option<&mut crate::rng::PyRng>,
) -> PyResult<PyBandPassDiffusionKernel> {
    let mut builder = BandPassDiffusionKernelBuilder::new(&laplacian.matrix, terms, degree);
    if let Some(lm) = lambda_max {
        builder = builder.lambda_max(lm);
    }
    let kernel = with_optional_rng(rng, |r| builder.build(r))
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyBandPassDiffusionKernel { kernel })
}

fn build_pivoted_bandpass_kernel(
    laplacian: &PyLaplacian,
    terms: Vec<(FloatType, FloatType)>,
    degree: usize,
    pivots: Vec<usize>,
    lambda_max: Option<FloatType>,
    num_hutchinson: Option<usize>,
    rng: Option<&mut crate::rng::PyRng>,
) -> PyResult<PyPivotedBandPassDiffusionKernel> {
    let mut builder =
        PivotedBandPassDiffusionKernelBuilder::new(&laplacian.matrix, terms, degree, pivots)
            .num_hutchinson(num_hutchinson);
    if let Some(lm) = lambda_max {
        builder = builder.lambda_max(lm);
    }
    let kernel = with_optional_rng(rng, |r| builder.build(r))
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyPivotedBandPassDiffusionKernel { kernel })
}

/// Python class for querying Chebyshev band-pass diffusion kernel matrix elements.
#[pyclass]
#[pyo3(name = "BandPassDiffusionKernel")]
pub struct PyBandPassDiffusionKernel {
    pub(crate) kernel: BandPassDiffusionKernel<FloatType>,
}

#[pymethods]
impl PyBandPassDiffusionKernel {
    /// Creates a new `BandPassDiffusionKernel` from `(coeff, t)` terms.
    #[new]
    #[pyo3(signature = (laplacian, terms, degree, rng = None, lambda_max = None))]
    fn new(
        laplacian: &PyLaplacian,
        terms: Vec<(FloatType, FloatType)>,
        degree: usize,
        rng: Option<&mut crate::rng::PyRng>,
        lambda_max: Option<FloatType>,
    ) -> PyResult<Self> {
        build_bandpass_kernel(laplacian, terms, degree, lambda_max, rng)
    }

    /// Creates a new `BandPassDiffusionKernel` with externally provided `lambda_max`.
    #[staticmethod]
    #[pyo3(signature = (laplacian, terms, degree, lambda_max, rng = None))]
    fn new_with_lambda_max(
        laplacian: &PyLaplacian,
        terms: Vec<(FloatType, FloatType)>,
        degree: usize,
        lambda_max: FloatType,
        rng: Option<&mut crate::rng::PyRng>,
    ) -> PyResult<Self> {
        build_bandpass_kernel(laplacian, terms, degree, Some(lambda_max), rng)
    }

    /// Creates a DC-retained 3-term Difference-of-Gaussians `BandPassDiffusionKernel`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (laplacian, t1, t2, alpha, beta_floor, degree, rng = None, lambda_max = None))]
    fn dog(
        laplacian: &PyLaplacian,
        t1: FloatType,
        t2: FloatType,
        alpha: FloatType,
        beta_floor: FloatType,
        degree: usize,
        rng: Option<&mut crate::rng::PyRng>,
        lambda_max: Option<FloatType>,
    ) -> PyResult<Self> {
        let terms = dog_bandpass_terms(t1, t2, alpha, beta_floor)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        build_bandpass_kernel(laplacian, terms, degree, lambda_max, rng)
    }

    /// Queries the `(i, j)` element of the band-pass kernel matrix.
    fn get(&self, i: usize, j: usize) -> PyResult<FloatType> {
        self.kernel.get(i, j).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "Index ({i}, {j}) out of bounds for kernel shape ({}, {})",
                self.kernel.n, self.kernel.n
            ))
        })
    }

    /// Returns the number of nodes in the graph.
    fn n(&self) -> usize {
        self.kernel.shape().0
    }
}

/// Builder for [`BandPassDiffusionKernel`].
#[pyclass]
#[pyo3(name = "BandPassDiffusionKernelBuilder")]
pub struct PyBandPassDiffusionKernelBuilder {
    terms: Vec<(FloatType, FloatType)>,
    degree: usize,
    lambda_max: Option<FloatType>,
}

#[pymethods]
impl PyBandPassDiffusionKernelBuilder {
    /// Creates a new `BandPassDiffusionKernelBuilder`.
    #[new]
    #[pyo3(signature = (terms = None, degree = 30, lambda_max = None))]
    fn new(
        terms: Option<Vec<(FloatType, FloatType)>>,
        degree: usize,
        lambda_max: Option<FloatType>,
    ) -> Self {
        Self {
            terms: terms.unwrap_or_default(),
            degree,
            lambda_max,
        }
    }

    /// Creates a builder configured for a DC-retained 3-term Difference-of-Gaussians kernel.
    #[staticmethod]
    #[pyo3(signature = (t1, t2, alpha, beta_floor, degree = 30, lambda_max = None))]
    fn dog(
        t1: FloatType,
        t2: FloatType,
        alpha: FloatType,
        beta_floor: FloatType,
        degree: usize,
        lambda_max: Option<FloatType>,
    ) -> PyResult<Self> {
        let terms = dog_bandpass_terms(t1, t2, alpha, beta_floor)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            terms,
            degree,
            lambda_max,
        })
    }

    /// Sets `(coeff, t)` bandpass terms.
    fn terms(
        mut slf: PyRefMut<'_, Self>,
        terms: Vec<(FloatType, FloatType)>,
    ) -> PyRefMut<'_, Self> {
        slf.terms = terms;
        slf
    }

    /// Sets the Chebyshev polynomial degree.
    fn degree(mut slf: PyRefMut<'_, Self>, degree: usize) -> PyRefMut<'_, Self> {
        slf.degree = degree;
        slf
    }

    /// Sets an explicit spectral upper bound `lambda_max`.
    fn lambda_max(
        mut slf: PyRefMut<'_, Self>,
        lambda_max: Option<FloatType>,
    ) -> PyRefMut<'_, Self> {
        slf.lambda_max = lambda_max;
        slf
    }

    /// Builds a `BandPassDiffusionKernel` for `laplacian`.
    #[pyo3(signature = (laplacian, rng = None))]
    fn build(
        &self,
        laplacian: &PyLaplacian,
        rng: Option<&mut crate::rng::PyRng>,
    ) -> PyResult<PyBandPassDiffusionKernel> {
        build_bandpass_kernel(
            laplacian,
            self.terms.clone(),
            self.degree,
            self.lambda_max,
            rng,
        )
    }
}

/// Python class for Chebyshev pivoted band-pass diffusion kernel with optional Hutchinson diagonal.
#[pyclass]
#[pyo3(name = "PivotedBandPassDiffusionKernel")]
pub struct PyPivotedBandPassDiffusionKernel {
    pub(crate) kernel: PivotedBandPassDiffusionKernel<FloatType>,
}

#[pymethods]
impl PyPivotedBandPassDiffusionKernel {
    /// Creates a new `PivotedBandPassDiffusionKernel` from `(coeff, t)` terms.
    #[new]
    #[pyo3(signature = (laplacian, terms, degree, pivots, rng = None, num_hutchinson = Some(32), lambda_max = None))]
    fn new(
        laplacian: &PyLaplacian,
        terms: Vec<(FloatType, FloatType)>,
        degree: usize,
        pivots: Vec<usize>,
        rng: Option<&mut crate::rng::PyRng>,
        num_hutchinson: Option<usize>,
        lambda_max: Option<FloatType>,
    ) -> PyResult<Self> {
        build_pivoted_bandpass_kernel(
            laplacian,
            terms,
            degree,
            pivots,
            lambda_max,
            num_hutchinson,
            rng,
        )
    }

    /// Creates a new `PivotedBandPassDiffusionKernel` with externally provided `lambda_max`.
    #[staticmethod]
    #[pyo3(signature = (laplacian, terms, degree, pivots, lambda_max, rng = None, num_hutchinson = Some(32)))]
    fn new_with_lambda_max(
        laplacian: &PyLaplacian,
        terms: Vec<(FloatType, FloatType)>,
        degree: usize,
        pivots: Vec<usize>,
        lambda_max: FloatType,
        rng: Option<&mut crate::rng::PyRng>,
        num_hutchinson: Option<usize>,
    ) -> PyResult<Self> {
        build_pivoted_bandpass_kernel(
            laplacian,
            terms,
            degree,
            pivots,
            Some(lambda_max),
            num_hutchinson,
            rng,
        )
    }

    /// Creates a DC-retained 3-term Difference-of-Gaussians `PivotedBandPassDiffusionKernel`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (laplacian, t1, t2, alpha, beta_floor, degree, pivots, rng = None, num_hutchinson = Some(32), lambda_max = None))]
    fn dog(
        laplacian: &PyLaplacian,
        t1: FloatType,
        t2: FloatType,
        alpha: FloatType,
        beta_floor: FloatType,
        degree: usize,
        pivots: Vec<usize>,
        rng: Option<&mut crate::rng::PyRng>,
        num_hutchinson: Option<usize>,
        lambda_max: Option<FloatType>,
    ) -> PyResult<Self> {
        let terms = dog_bandpass_terms(t1, t2, alpha, beta_floor)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        build_pivoted_bandpass_kernel(
            laplacian,
            terms,
            degree,
            pivots,
            lambda_max,
            num_hutchinson,
            rng,
        )
    }

    /// Returns pivot node indices.
    fn pivots(&self) -> Vec<usize> {
        self.kernel.pivots().to_vec()
    }

    /// Queries kernel entry `K(pivots[pivot_idx], j)`.
    fn get_from_pivot(&self, pivot_idx: usize, j: usize) -> PyResult<FloatType> {
        let n = self.n();
        if pivot_idx >= self.kernel.pivots().len() || j >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index ({pivot_idx}, {j}) out of bounds for pivoted kernel shape ({}, {n})",
                self.kernel.pivots().len(),
            )));
        }
        Ok(self.kernel.get_from_pivot(pivot_idx, j))
    }

    /// Returns the estimated diagonal entry `K[j, j]` if Hutchinson estimation was enabled.
    fn diagonal(&self, j: usize) -> Option<FloatType> {
        self.kernel.diagonal(j)
    }

    /// Returns the number of nodes in the graph.
    fn n(&self) -> usize {
        self.kernel.vectors.first().map_or(0, |v| v.len())
    }
}

/// Builder for [`PivotedBandPassDiffusionKernel`].
#[pyclass]
#[pyo3(name = "PivotedBandPassDiffusionKernelBuilder")]
pub struct PyPivotedBandPassDiffusionKernelBuilder {
    terms: Vec<(FloatType, FloatType)>,
    degree: usize,
    pivots: Vec<usize>,
    lambda_max: Option<FloatType>,
    num_hutchinson: Option<usize>,
}

#[pymethods]
impl PyPivotedBandPassDiffusionKernelBuilder {
    /// Creates a new `PivotedBandPassDiffusionKernelBuilder` (default `num_hutchinson=32`).
    #[new]
    #[pyo3(signature = (terms = None, degree = 30, pivots = None, lambda_max = None, num_hutchinson = Some(32)))]
    fn new(
        terms: Option<Vec<(FloatType, FloatType)>>,
        degree: usize,
        pivots: Option<Vec<usize>>,
        lambda_max: Option<FloatType>,
        num_hutchinson: Option<usize>,
    ) -> Self {
        Self {
            terms: terms.unwrap_or_default(),
            degree,
            pivots: pivots.unwrap_or_default(),
            lambda_max,
            num_hutchinson,
        }
    }

    /// Creates a builder configured for a DC-retained 3-term Difference-of-Gaussians pivoted kernel.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (t1, t2, alpha, beta_floor, degree = 30, pivots = None, lambda_max = None, num_hutchinson = Some(32)))]
    fn dog(
        t1: FloatType,
        t2: FloatType,
        alpha: FloatType,
        beta_floor: FloatType,
        degree: usize,
        pivots: Option<Vec<usize>>,
        lambda_max: Option<FloatType>,
        num_hutchinson: Option<usize>,
    ) -> PyResult<Self> {
        let terms = dog_bandpass_terms(t1, t2, alpha, beta_floor)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            terms,
            degree,
            pivots: pivots.unwrap_or_default(),
            lambda_max,
            num_hutchinson,
        })
    }

    /// Sets `(coeff, t)` bandpass terms.
    fn terms(
        mut slf: PyRefMut<'_, Self>,
        terms: Vec<(FloatType, FloatType)>,
    ) -> PyRefMut<'_, Self> {
        slf.terms = terms;
        slf
    }

    /// Sets the Chebyshev polynomial degree.
    fn degree(mut slf: PyRefMut<'_, Self>, degree: usize) -> PyRefMut<'_, Self> {
        slf.degree = degree;
        slf
    }

    /// Sets the pivot node indices.
    fn pivots(mut slf: PyRefMut<'_, Self>, pivots: Vec<usize>) -> PyRefMut<'_, Self> {
        slf.pivots = pivots;
        slf
    }

    /// Sets an explicit spectral upper bound `lambda_max`.
    fn lambda_max(
        mut slf: PyRefMut<'_, Self>,
        lambda_max: Option<FloatType>,
    ) -> PyRefMut<'_, Self> {
        slf.lambda_max = lambda_max;
        slf
    }

    /// Sets the number of Hutchinson probe vectors (or `None` to disable diagonal estimation).
    fn num_hutchinson(
        mut slf: PyRefMut<'_, Self>,
        num_hutchinson: Option<usize>,
    ) -> PyRefMut<'_, Self> {
        slf.num_hutchinson = num_hutchinson;
        slf
    }

    /// Builds a `PivotedBandPassDiffusionKernel` for `laplacian`.
    #[pyo3(signature = (laplacian, rng = None))]
    fn build(
        &self,
        laplacian: &PyLaplacian,
        rng: Option<&mut crate::rng::PyRng>,
    ) -> PyResult<PyPivotedBandPassDiffusionKernel> {
        build_pivoted_bandpass_kernel(
            laplacian,
            self.terms.clone(),
            self.degree,
            self.pivots.clone(),
            self.lambda_max,
            self.num_hutchinson,
            rng,
        )
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

    /// Returns diagonal entry if available (always `None` for multiscale pivoted kernels).
    fn diagonal(&self, j: usize) -> Option<FloatType> {
        self.kernel.diagonal(j)
    }
}

/// Builder for PyLowRankDiffusionKernel
#[pyclass]
#[pyo3(name = "LowRankDiffusionKernelBuilder")]
pub struct PyLowRankDiffusionKernelBuilder {
    pub(crate) builder: LowRankDiffusionKernelBuilder<FloatType>,
}

#[pymethods]
impl PyLowRankDiffusionKernelBuilder {
    #[new]
    fn new() -> Self {
        Self {
            builder: LowRankDiffusionKernelBuilder::new(),
        }
    }

    fn t(mut slf: PyRefMut<'_, Self>, t: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().t(t);
        slf
    }

    fn rank(mut slf: PyRefMut<'_, Self>, rank: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().rank(rank);
        slf
    }

    fn shift(mut slf: PyRefMut<'_, Self>, shift: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().shift(shift);
        slf
    }

    fn eigenvalue_max_iterations(
        mut slf: PyRefMut<'_, Self>,
        eigenvalue_max_iterations: usize,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf
            .builder
            .clone()
            .eigenvalue_max_iterations(eigenvalue_max_iterations);
        slf
    }

    fn cg_max_iterations(
        mut slf: PyRefMut<'_, Self>,
        cg_max_iterations: usize,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().cg_max_iterations(cg_max_iterations);
        slf
    }

    fn eigenvalue_tolerance(
        mut slf: PyRefMut<'_, Self>,
        eigenvalue_tolerance: FloatType,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf
            .builder
            .clone()
            .eigenvalue_tolerance(eigenvalue_tolerance);
        slf
    }

    fn cg_tolerance(mut slf: PyRefMut<'_, Self>, cg_tolerance: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().cg_tolerance(cg_tolerance);
        slf
    }

    /// Builds a `LowRankDiffusionKernel` using the unnormalized combinatorial Laplacian.
    #[pyo3(name = "build_unnormalized_laplacian", signature = (graph, length = None, rng = None))]
    fn build_laplacian(
        &self,
        graph: &crate::graph::PyGraphAdapter,
        length: Option<&Bound<'_, PyAny>>,
        rng: Option<&Bound<'_, crate::rng::PyRng>>,
    ) -> PyResult<PyLowRankDiffusionKernel> {
        let kernel = build_low_rank_from_graph(graph, length, rng, |g, f, r| {
            self.builder.clone().build_laplacian(g, f, r)
        })?;
        Ok(PyLowRankDiffusionKernel { kernel })
    }

    /// Builds a `LowRankDiffusionKernel` using the symmetric normalized Laplacian.
    #[pyo3(name = "build_symmetric_normalized_laplacian", signature = (graph, length = None, rng = None))]
    fn build_normalized_laplacian(
        &self,
        graph: &crate::graph::PyGraphAdapter,
        length: Option<&Bound<'_, PyAny>>,
        rng: Option<&Bound<'_, crate::rng::PyRng>>,
    ) -> PyResult<PyLowRankDiffusionKernel> {
        let kernel = build_low_rank_from_graph(graph, length, rng, |g, f, r| {
            self.builder.clone().build_normalized_laplacian(g, f, r)
        })?;
        Ok(PyLowRankDiffusionKernel { kernel })
    }
}

fn build_low_rank_from_graph<K, F>(
    graph: &crate::graph::PyGraphAdapter,
    length: Option<&Bound<'_, PyAny>>,
    rng: Option<&Bound<'_, crate::rng::PyRng>>,
    build_fn: F,
) -> PyResult<K>
where
    F: FnOnce(
        &petgraph::graph::UnGraph<Py<PyAny>, Py<PyAny>, crate::graph::IndexType>,
        &mut dyn FnMut(petgraph::graph::EdgeReference<'_, Py<PyAny>>) -> FloatType,
        &mut rand::rngs::StdRng,
    ) -> Result<K, String>,
{
    let (edge_length, rng_opt) = resolve_length_and_rng(length, rng)?;
    let mut length_closure =
        |edge: petgraph::graph::EdgeReference<'_, Py<PyAny>>| edge_length.eval(edge.id().index());
    match graph.graph() {
        crate::graph::GraphType::Graph(native_graph) => {
            with_optional_rng_ref_mut(rng_opt, |r| build_fn(native_graph, &mut length_closure, r))
                .map_err(pyo3::exceptions::PyValueError::new_err)
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "Unsupported graph type",
        )),
    }
}

/// Builder for PyLowRankMultiscaleDiffusionKernel
#[pyclass]
#[pyo3(name = "LowRankMultiscaleDiffusionKernelBuilder")]
pub struct PyLowRankMultiscaleDiffusionKernelBuilder {
    pub(crate) builder: LowRankMultiscaleDiffusionKernelBuilder<FloatType>,
}

#[pymethods]
impl PyLowRankMultiscaleDiffusionKernelBuilder {
    #[new]
    fn new() -> Self {
        Self {
            builder: LowRankMultiscaleDiffusionKernelBuilder::new(),
        }
    }

    fn alpha(mut slf: PyRefMut<'_, Self>, alpha: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().alpha(alpha);
        slf
    }

    fn rank(mut slf: PyRefMut<'_, Self>, rank: usize) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().rank(rank);
        slf
    }

    fn shift(mut slf: PyRefMut<'_, Self>, shift: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().shift(shift);
        slf
    }

    fn eigenvalue_max_iterations(
        mut slf: PyRefMut<'_, Self>,
        eigenvalue_max_iterations: usize,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf
            .builder
            .clone()
            .eigenvalue_max_iterations(eigenvalue_max_iterations);
        slf
    }

    fn cg_max_iterations(
        mut slf: PyRefMut<'_, Self>,
        cg_max_iterations: usize,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().cg_max_iterations(cg_max_iterations);
        slf
    }

    fn eigenvalue_tolerance(
        mut slf: PyRefMut<'_, Self>,
        eigenvalue_tolerance: FloatType,
    ) -> PyRefMut<'_, Self> {
        slf.builder = slf
            .builder
            .clone()
            .eigenvalue_tolerance(eigenvalue_tolerance);
        slf
    }

    fn cg_tolerance(mut slf: PyRefMut<'_, Self>, cg_tolerance: FloatType) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().cg_tolerance(cg_tolerance);
        slf
    }

    /// Builds a `LowRankMultiscaleDiffusionKernel` using the unnormalized combinatorial Laplacian.
    #[pyo3(name = "build_unnormalized_laplacian", signature = (graph, length = None, rng = None))]
    fn build_laplacian(
        &self,
        graph: &crate::graph::PyGraphAdapter,
        length: Option<&Bound<'_, PyAny>>,
        rng: Option<&Bound<'_, crate::rng::PyRng>>,
    ) -> PyResult<PyLowRankMultiscaleDiffusionKernel> {
        let kernel = build_low_rank_from_graph(graph, length, rng, |g, f, r| {
            self.builder.clone().build_laplacian(g, f, r)
        })?;
        Ok(PyLowRankMultiscaleDiffusionKernel { kernel })
    }

    /// Builds a `LowRankMultiscaleDiffusionKernel` using the symmetric normalized Laplacian.
    #[pyo3(name = "build_symmetric_normalized_laplacian", signature = (graph, length = None, rng = None))]
    fn build_normalized_laplacian(
        &self,
        graph: &crate::graph::PyGraphAdapter,
        length: Option<&Bound<'_, PyAny>>,
        rng: Option<&Bound<'_, crate::rng::PyRng>>,
    ) -> PyResult<PyLowRankMultiscaleDiffusionKernel> {
        let kernel = build_low_rank_from_graph(graph, length, rng, |g, f, r| {
            self.builder.clone().build_normalized_laplacian(g, f, r)
        })?;
        Ok(PyLowRankMultiscaleDiffusionKernel { kernel })
    }
}
