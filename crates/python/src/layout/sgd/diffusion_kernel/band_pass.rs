//! Non-pivoted Chebyshev band-pass diffusion kernel and builder bindings.

use super::with_optional_rng;
use crate::{FloatType, distance_matrix::PyLaplacian};
use petgraph_linalg_kernel::{
    BandPassDiffusionKernel, BandPassDiffusionKernelBuilder, Kernel, dog_bandpass_terms,
    multiscale_heat_terms,
};
use pyo3::prelude::*;

fn build_bandpass_kernel(
    laplacian: &PyLaplacian,
    terms: Vec<(FloatType, FloatType)>,
    degree: usize,
    lambda_max: Option<FloatType>,
    off_diagonal_floor: FloatType,
    rng: Option<&mut crate::rng::PyRng>,
) -> PyResult<PyBandPassDiffusionKernel> {
    let mut builder = BandPassDiffusionKernelBuilder::new(&laplacian.matrix, terms, degree)
        .off_diagonal_floor(off_diagonal_floor);
    if let Some(lm) = lambda_max {
        builder = builder.lambda_max(lm);
    }
    let kernel = with_optional_rng(rng, |r| builder.build(r))
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyBandPassDiffusionKernel { kernel })
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
    #[pyo3(signature = (laplacian, terms, degree, rng = None, lambda_max = None, off_diagonal_floor = 0.0))]
    fn new(
        laplacian: &PyLaplacian,
        terms: Vec<(FloatType, FloatType)>,
        degree: usize,
        rng: Option<&mut crate::rng::PyRng>,
        lambda_max: Option<FloatType>,
        off_diagonal_floor: FloatType,
    ) -> PyResult<Self> {
        build_bandpass_kernel(
            laplacian,
            terms,
            degree,
            lambda_max,
            off_diagonal_floor,
            rng,
        )
    }

    /// Creates a new `BandPassDiffusionKernel` with externally provided `lambda_max`.
    #[staticmethod]
    #[pyo3(signature = (laplacian, terms, degree, lambda_max, rng = None, off_diagonal_floor = 0.0))]
    fn new_with_lambda_max(
        laplacian: &PyLaplacian,
        terms: Vec<(FloatType, FloatType)>,
        degree: usize,
        lambda_max: FloatType,
        rng: Option<&mut crate::rng::PyRng>,
        off_diagonal_floor: FloatType,
    ) -> PyResult<Self> {
        build_bandpass_kernel(
            laplacian,
            terms,
            degree,
            Some(lambda_max),
            off_diagonal_floor,
            rng,
        )
    }

    /// Creates a Difference-of-Gaussians `BandPassDiffusionKernel` with uniform rank-1 `beta_floor`.
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
        build_bandpass_kernel(laplacian, terms, degree, lambda_max, beta_floor, rng)
    }

    /// Creates a positive-coefficient multi-scale heat `BandPassDiffusionKernel`.
    #[staticmethod]
    #[pyo3(signature = (laplacian, terms, degree, rng = None, lambda_max = None, off_diagonal_floor = 0.0))]
    fn multiscale_heat(
        laplacian: &PyLaplacian,
        terms: Vec<(FloatType, FloatType)>,
        degree: usize,
        rng: Option<&mut crate::rng::PyRng>,
        lambda_max: Option<FloatType>,
        off_diagonal_floor: FloatType,
    ) -> PyResult<Self> {
        let validated =
            multiscale_heat_terms(&terms).map_err(pyo3::exceptions::PyValueError::new_err)?;
        build_bandpass_kernel(
            laplacian,
            validated,
            degree,
            lambda_max,
            off_diagonal_floor,
            rng,
        )
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
    off_diagonal_floor: FloatType,
}

#[pymethods]
impl PyBandPassDiffusionKernelBuilder {
    /// Creates a new `BandPassDiffusionKernelBuilder`.
    #[new]
    #[pyo3(signature = (terms = None, degree = 30, lambda_max = None, off_diagonal_floor = 0.0))]
    fn new(
        terms: Option<Vec<(FloatType, FloatType)>>,
        degree: usize,
        lambda_max: Option<FloatType>,
        off_diagonal_floor: FloatType,
    ) -> Self {
        Self {
            terms: terms.unwrap_or_default(),
            degree,
            lambda_max,
            off_diagonal_floor,
        }
    }

    /// Creates a builder configured for a Difference-of-Gaussians kernel with uniform `beta_floor`.
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
            off_diagonal_floor: beta_floor,
        })
    }

    /// Creates a builder configured for a positive-coefficient multi-scale heat kernel.
    #[staticmethod]
    #[pyo3(signature = (terms, degree = 30, lambda_max = None, off_diagonal_floor = 0.0))]
    fn multiscale_heat(
        terms: Vec<(FloatType, FloatType)>,
        degree: usize,
        lambda_max: Option<FloatType>,
        off_diagonal_floor: FloatType,
    ) -> PyResult<Self> {
        let validated =
            multiscale_heat_terms(&terms).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            terms: validated,
            degree,
            lambda_max,
            off_diagonal_floor,
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

    /// Sets the uniform rank-1 floor added to every kernel entry (`K + floor * 1 1^T`).
    fn off_diagonal_floor(mut slf: PyRefMut<'_, Self>, floor: FloatType) -> PyRefMut<'_, Self> {
        slf.off_diagonal_floor = floor;
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
            self.off_diagonal_floor,
            rng,
        )
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBandPassDiffusionKernel>()?;
    m.add_class::<PyBandPassDiffusionKernelBuilder>()?;
    Ok(())
}
