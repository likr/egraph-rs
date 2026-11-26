//! Pivoted Chebyshev band-pass diffusion kernel and builder bindings.

use super::with_optional_rng;
use crate::{FloatType, distance_matrix::PyLaplacian};
use petgraph_linalg_kernel::{
    PivotedBandPassDiffusionKernel, PivotedBandPassDiffusionKernelBuilder, PivotedKernel,
    dog_bandpass_terms, multiscale_heat_terms,
};
use pyo3::prelude::*;

#[allow(clippy::too_many_arguments)]
fn build_pivoted_bandpass_kernel(
    laplacian: &PyLaplacian,
    terms: Vec<(FloatType, FloatType)>,
    degree: usize,
    pivots: Vec<usize>,
    lambda_max: Option<FloatType>,
    num_hutchinson: Option<usize>,
    off_diagonal_floor: FloatType,
    rng: Option<&mut crate::rng::PyRng>,
) -> PyResult<PyPivotedBandPassDiffusionKernel> {
    let mut builder =
        PivotedBandPassDiffusionKernelBuilder::new(&laplacian.matrix, terms, degree, pivots)
            .num_hutchinson(num_hutchinson)
            .off_diagonal_floor(off_diagonal_floor);
    if let Some(lm) = lambda_max {
        builder = builder.lambda_max(lm);
    }
    let kernel = with_optional_rng(rng, |r| builder.build(r))
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyPivotedBandPassDiffusionKernel { kernel })
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
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (laplacian, terms, degree, pivots, rng = None, num_hutchinson = Some(32), lambda_max = None, off_diagonal_floor = 0.0))]
    fn new(
        laplacian: &PyLaplacian,
        terms: Vec<(FloatType, FloatType)>,
        degree: usize,
        pivots: Vec<usize>,
        rng: Option<&mut crate::rng::PyRng>,
        num_hutchinson: Option<usize>,
        lambda_max: Option<FloatType>,
        off_diagonal_floor: FloatType,
    ) -> PyResult<Self> {
        build_pivoted_bandpass_kernel(
            laplacian,
            terms,
            degree,
            pivots,
            lambda_max,
            num_hutchinson,
            off_diagonal_floor,
            rng,
        )
    }

    /// Creates a new `PivotedBandPassDiffusionKernel` with externally provided `lambda_max`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (laplacian, terms, degree, pivots, lambda_max, rng = None, num_hutchinson = Some(32), off_diagonal_floor = 0.0))]
    fn new_with_lambda_max(
        laplacian: &PyLaplacian,
        terms: Vec<(FloatType, FloatType)>,
        degree: usize,
        pivots: Vec<usize>,
        lambda_max: FloatType,
        rng: Option<&mut crate::rng::PyRng>,
        num_hutchinson: Option<usize>,
        off_diagonal_floor: FloatType,
    ) -> PyResult<Self> {
        build_pivoted_bandpass_kernel(
            laplacian,
            terms,
            degree,
            pivots,
            Some(lambda_max),
            num_hutchinson,
            off_diagonal_floor,
            rng,
        )
    }

    /// Creates a Difference-of-Gaussians `PivotedBandPassDiffusionKernel` with uniform `beta_floor`.
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
            beta_floor,
            rng,
        )
    }

    /// Creates a positive-coefficient multi-scale heat `PivotedBandPassDiffusionKernel`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (laplacian, terms, degree, pivots, rng = None, num_hutchinson = Some(32), lambda_max = None, off_diagonal_floor = 0.0))]
    fn multiscale_heat(
        laplacian: &PyLaplacian,
        terms: Vec<(FloatType, FloatType)>,
        degree: usize,
        pivots: Vec<usize>,
        rng: Option<&mut crate::rng::PyRng>,
        num_hutchinson: Option<usize>,
        lambda_max: Option<FloatType>,
        off_diagonal_floor: FloatType,
    ) -> PyResult<Self> {
        let validated =
            multiscale_heat_terms(&terms).map_err(pyo3::exceptions::PyValueError::new_err)?;
        build_pivoted_bandpass_kernel(
            laplacian,
            validated,
            degree,
            pivots,
            lambda_max,
            num_hutchinson,
            off_diagonal_floor,
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
    off_diagonal_floor: FloatType,
}

#[pymethods]
impl PyPivotedBandPassDiffusionKernelBuilder {
    /// Creates a new `PivotedBandPassDiffusionKernelBuilder` (default `num_hutchinson=32`).
    #[new]
    #[pyo3(signature = (terms = None, degree = 30, pivots = None, lambda_max = None, num_hutchinson = Some(32), off_diagonal_floor = 0.0))]
    fn new(
        terms: Option<Vec<(FloatType, FloatType)>>,
        degree: usize,
        pivots: Option<Vec<usize>>,
        lambda_max: Option<FloatType>,
        num_hutchinson: Option<usize>,
        off_diagonal_floor: FloatType,
    ) -> Self {
        Self {
            terms: terms.unwrap_or_default(),
            degree,
            pivots: pivots.unwrap_or_default(),
            lambda_max,
            num_hutchinson,
            off_diagonal_floor,
        }
    }

    /// Creates a builder configured for a Difference-of-Gaussians pivoted kernel with uniform `beta_floor`.
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
            off_diagonal_floor: beta_floor,
        })
    }

    /// Creates a builder configured for a positive-coefficient multi-scale heat pivoted kernel.
    #[staticmethod]
    #[pyo3(signature = (terms, degree = 30, pivots = None, lambda_max = None, num_hutchinson = Some(32), off_diagonal_floor = 0.0))]
    fn multiscale_heat(
        terms: Vec<(FloatType, FloatType)>,
        degree: usize,
        pivots: Option<Vec<usize>>,
        lambda_max: Option<FloatType>,
        num_hutchinson: Option<usize>,
        off_diagonal_floor: FloatType,
    ) -> PyResult<Self> {
        let validated =
            multiscale_heat_terms(&terms).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            terms: validated,
            degree,
            pivots: pivots.unwrap_or_default(),
            lambda_max,
            num_hutchinson,
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

    /// Sets the uniform rank-1 floor added to every kernel entry (`K + floor * 1 1^T`).
    fn off_diagonal_floor(mut slf: PyRefMut<'_, Self>, floor: FloatType) -> PyRefMut<'_, Self> {
        slf.off_diagonal_floor = floor;
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
            self.off_diagonal_floor,
            rng,
        )
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPivotedBandPassDiffusionKernel>()?;
    m.add_class::<PyPivotedBandPassDiffusionKernelBuilder>()?;
    Ok(())
}
