//! SGD (Stochastic Gradient Descent) layout algorithms
//!
//! This module provides Python bindings for various SGD-based layout algorithms.

mod full;
mod omega;
mod schedulers;
mod sgd;
mod sparse;

use pyo3::prelude::*;

pub use self::full::PyFullSgd;
pub use self::omega::PyOmega;
pub use self::schedulers::{
    PySchedulerConstant, PySchedulerExponential, PySchedulerLinear, PySchedulerQuadratic,
    PySchedulerReciprocal,
};
pub use self::sgd::PySgd;
pub use self::sparse::PySparseSgd;

/// Register all SGD-related classes with the Python module
pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // Register scheduler classes
    m.add_class::<PySchedulerConstant>()?;
    m.add_class::<PySchedulerLinear>()?;
    m.add_class::<PySchedulerQuadratic>()?;
    m.add_class::<PySchedulerExponential>()?;
    m.add_class::<PySchedulerReciprocal>()?;

    // Register SGD algorithm classes
    m.add_class::<PyFullSgd>()?;
    m.add_class::<PySparseSgd>()?;
    m.add_class::<PyOmega>()?;
    m.add_class::<PySgd>()?;

    Ok(())
}
