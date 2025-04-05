/// Graph layout algorithms for the Python bindings
///
/// This module provides various graph layout algorithms that compute positions for graph nodes
/// in different geometric spaces, optimizing for various aesthetic criteria like distance preservation,
/// stress minimization, and minimal edge crossings.
///
/// # Submodules
///
/// - `mds`: Multidimensional Scaling algorithms
/// - `kamada_kawai`: Kamada-Kawai force-directed layout algorithm
/// - `overwrap_removal`: Algorithms to remove overlaps between nodes
/// - `stress_majorization`: Stress majorization layout algorithm
/// - `sgd`: Stochastic Gradient Descent based layout algorithms
mod kamada_kawai;
mod mds;
mod overwrap_removal;
mod sgd;
mod stress_majorization;

use pyo3::prelude::*;

/// Registers layout-related classes and functions with the Python module
///
/// This function adds all the graph layout algorithms to the Python module,
/// making them available to be instantiated and used from Python code.
/// These algorithms determine the positions of nodes in a graph drawing,
/// optimizing for various aesthetic criteria.
pub fn register(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // Register various layout algorithm implementations
    mds::register(py, m)?;
    kamada_kawai::register(py, m)?;
    overwrap_removal::register(py, m)?;
    stress_majorization::register(py, m)?;
    sgd::register(py, m)?;
    Ok(())
}
