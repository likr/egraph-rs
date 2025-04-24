/// Python bindings for petgraph-clustering.
///
/// This module provides Python bindings for the petgraph-clustering Rust crate,
/// which implements various community detection algorithms for graphs.
use pyo3::prelude::*;

mod coarsen;
mod infomap;
mod label_propagation;
mod louvain;
mod spectral;

use coarsen::py_coarsen;
use infomap::PyInfoMap;
use label_propagation::PyLabelPropagation;
use louvain::PyLouvain;
use spectral::PySpectralClustering;

/// Register the clustering module and its classes with Python.
pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyLouvain>()?;
    m.add_class::<PyLabelPropagation>()?;
    m.add_class::<PySpectralClustering>()?;
    m.add_class::<PyInfoMap>()?;
    m.add_function(wrap_pyfunction!(py_coarsen, m)?)?;
    Ok(())
}
