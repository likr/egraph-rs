//! SGD (Stochastic Gradient Descent) layout algorithms
//!
//! This module provides Python bindings for various SGD-based layout algorithms.

#![allow(clippy::module_inception)]

mod diffusion_kernel;
mod full;
mod omega;
mod random_pair_sparse_sgd;
mod schedulers;
mod sgd;
mod sparse;

use pyo3::prelude::*;

pub use self::diffusion_kernel::{PyDiffusionKernel, PyMultiscaleDiffusionKernel};
pub use self::full::PyFullSgd;
pub use self::omega::PyOmega;
pub use self::random_pair_sparse_sgd::PyRandomPairSparseSgd;
pub use self::schedulers::{
    PySchedulerConstant, PySchedulerExponential, PySchedulerLinear, PySchedulerQuadratic,
    PySchedulerReciprocal,
};
pub use self::sgd::PySgd;
pub use self::sparse::PySparseSgd;

/// Register all SGD-related classes with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register scheduler classes
    m.add_class::<PySchedulerConstant>()?;
    m.add_class::<PySchedulerLinear>()?;
    m.add_class::<PySchedulerQuadratic>()?;
    m.add_class::<PySchedulerExponential>()?;
    m.add_class::<PySchedulerReciprocal>()?;

    // Register SGD algorithm classes
    m.add_class::<PyFullSgd>()?;
    m.add_class::<PySparseSgd>()?;
    m.add_class::<PyRandomPairSparseSgd>()?;
    m.add_class::<PyOmega>()?;
    m.add_class::<PySgd>()?;
    m.add_class::<PyDiffusionKernel>()?;
    m.add_class::<PyMultiscaleDiffusionKernel>()?;

    Ok(())
}
