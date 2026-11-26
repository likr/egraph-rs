//! Chebyshev and low-rank diffusion kernel bindings for Python.

mod band_pass;
mod full;
mod low_rank;
mod pivoted_band_pass;

use crate::distance_matrix::EdgeLengthFn;
use pyo3::prelude::*;
use rand::SeedableRng;

pub use self::band_pass::{PyBandPassDiffusionKernel, PyBandPassDiffusionKernelBuilder};
pub use self::full::{
    PyDiffusionKernel, PyMultiscaleDiffusionKernel, PyPivotedDiffusionKernel,
    PyPivotedMultiscaleDiffusionKernel,
};
pub use self::low_rank::{
    PyLowRankDiffusionKernel, PyLowRankDiffusionKernelBuilder, PyLowRankMultiscaleDiffusionKernel,
    PyLowRankMultiscaleDiffusionKernelBuilder,
};
pub use self::pivoted_band_pass::{
    PyPivotedBandPassDiffusionKernel, PyPivotedBandPassDiffusionKernelBuilder,
};

pub(super) fn with_optional_rng<R>(
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

pub(super) fn with_optional_rng_ref_mut<R>(
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

pub(super) fn resolve_length_and_rng<'py>(
    length: Option<&Bound<'py, PyAny>>,
    rng: Option<&Bound<'py, crate::rng::PyRng>>,
) -> PyResult<(EdgeLengthFn, Option<PyRefMut<'py, crate::rng::PyRng>>)> {
    if let Some(rng_bound) = rng {
        return Ok((
            EdgeLengthFn::from_bound(length)?,
            Some(rng_bound.borrow_mut()),
        ));
    }
    if let Some(first) = length
        && let Ok(rng_bound) = first.cast::<crate::rng::PyRng>()
    {
        return Ok((EdgeLengthFn::Constant(1.0), Some(rng_bound.borrow_mut())));
    }
    Ok((EdgeLengthFn::from_bound(length)?, None))
}

/// Registers all diffusion kernel and kernel builder classes with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    full::register(m)?;
    band_pass::register(m)?;
    low_rank::register(m)?;
    pivoted_band_pass::register(m)?;
    Ok(())
}
