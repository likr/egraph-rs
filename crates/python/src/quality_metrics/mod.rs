//! Graph layout quality metrics for evaluating drawing aesthetics.
//!
//! This module provides a collection of metrics to quantitatively assess the quality
//! of graph layouts across both CSR (`CsrGraph`, `Layout2d`) and legacy (`Graph`, `Drawing`) types.

mod crossings;
mod neighborhood_geometry;

use crate::{
    csr_graph::PyCsrGraph, drawing::PyDrawingEuclidean2d, graph::PyGraphAdapter,
    layout_buffers::PyLayout2d,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

#[allow(unused_imports)]
pub use crossings::PyCrossingEdges;

pub(super) enum DualBinaryInput<'py> {
    Csr(PyRef<'py, PyCsrGraph>, PyRef<'py, PyLayout2d>),
    Legacy(PyRef<'py, PyGraphAdapter>, PyRef<'py, PyDrawingEuclidean2d>),
}

pub(super) fn extract_dual_binary<'py>(
    graph: &Bound<'py, PyAny>,
    drawing: &Bound<'py, PyAny>,
    fn_name: &str,
) -> PyResult<DualBinaryInput<'py>> {
    if let Ok(csr) = graph.cast::<PyCsrGraph>() {
        let layout = drawing.cast::<PyLayout2d>().map_err(|_| {
            PyTypeError::new_err(format!(
                "{fn_name}: expected Layout2d when graph is CsrGraph"
            ))
        })?;
        return Ok(DualBinaryInput::Csr(csr.borrow(), layout.borrow()));
    }
    if let Ok(legacy_graph) = graph.cast::<PyGraphAdapter>() {
        let legacy_drawing = drawing.cast::<PyDrawingEuclidean2d>().map_err(|_| {
            PyTypeError::new_err(format!(
                "{fn_name}: expected DrawingEuclidean2d when graph is Graph or DiGraph"
            ))
        })?;
        return Ok(DualBinaryInput::Legacy(
            legacy_graph.borrow(),
            legacy_drawing.borrow(),
        ));
    }
    Err(PyTypeError::new_err(format!(
        "{fn_name}: graph must be CsrGraph, Graph, or DiGraph"
    )))
}

/// Registers all layout quality metric functions with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crossings::register(m)?;
    neighborhood_geometry::register(m)?;
    Ok(())
}
