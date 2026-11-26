//! Distance matrix, kernel, and Laplacian bindings for Python.

mod kernels;
mod laplacian;
mod matrix;
mod neg_log;

use pyo3::prelude::*;

#[allow(unused_imports)]
pub use kernels::{
    InnerKernel, PyEmbeddingKernel, PyEmbeddingKernelBuilder, PyExponentialKernel,
    PyExponentialKernelBuilder, PyGaussianKernel, PyGaussianKernelBuilder, PyTKernel,
    PyTKernelBuilder, extract_inner_kernel,
};
pub(crate) use laplacian::EdgeLengthFn;
#[allow(unused_imports)]
pub use laplacian::{PyLaplacian, PyStandardLaplacian, PySymmetricNormalizedLaplacian};
#[allow(unused_imports)]
pub use matrix::{
    DistanceMatrixType, PyDistanceMatrix, PyKernelDistance, PyKernelDistanceBuilder,
    distance_node_count, with_distance, with_full_distance, with_pivot_mds_distance,
};
#[allow(unused_imports)]
pub use neg_log::{
    InnerPivotedKernel, PyNegLogDistance, PyNegLogDistanceBuilder, PyNegLogSimDistance,
    PyNegLogSimDistanceBuilder, PyPivotedNegLogDistance, PyPivotedNegLogDistanceBuilder,
    PyPivotedNegLogSimDistance, PyPivotedNegLogSimDistanceBuilder, extract_inner_pivoted_kernel,
};

/// Registers all distance matrix, kernel, and Laplacian classes with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    matrix::register(m)?;
    neg_log::register(m)?;
    kernels::register(m)?;
    laplacian::register(m)?;
    Ok(())
}
