/// Graph layout algorithms for the Python bindings
///
/// This module provides various graph layout algorithms that compute positions for graph nodes
/// in different geometric spaces, optimizing for various aesthetic criteria like distance preservation,
/// stress minimization, and minimal edge crossings.
///
/// # Submodules
///
/// - `mds`: Multidimensional Scaling algorithms
/// - `rdmds`: Resistance-distance MDS for spectral embeddings
/// - `kamada_kawai`: Kamada-Kawai force-directed layout algorithm
/// - `overwrap_removal`: Algorithms to remove overlaps between nodes
/// - `stress_majorization`: Stress majorization layout algorithm
/// - `sgd`: Stochastic Gradient Descent based layout algorithms
mod kamada_kawai;
mod mds;
mod overwrap_removal;
mod rdmds;
pub(crate) mod rdmds_solvers;
pub(crate) mod sgd;
mod stress_majorization;
mod ts_net;

use pyo3::prelude::*;

/// Registers layout-related classes and functions with the Python module
///
/// This function adds all the graph layout algorithms to the Python module,
/// making them available to be instantiated and used from Python code.
/// These algorithms determine the positions of nodes in a graph drawing,
/// optimizing for various aesthetic criteria.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register various layout algorithm implementations
    mds::register(m)?;
    rdmds::register(m)?;
    rdmds_solvers::register(m)?;
    kamada_kawai::register(m)?;
    overwrap_removal::register(m)?;
    stress_majorization::register(m)?;
    sgd::register(m)?;
    ts_net::register(m)?;
    Ok(())
}
