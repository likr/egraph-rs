//! Edge crossing detection, crossing number, and crossing angle quality metrics.

use super::{DualBinaryInput, extract_dual_binary};
use crate::{
    FloatType,
    csr_graph::PyCsrGraph,
    drawing::{DrawingType, PyDrawing, PyDrawingEuclidean2d, PyDrawingTorus2d},
    graph::{GraphType, PyGraphAdapter},
    layout_buffers::PyLayout2d,
    metrics::{csr_crossing_angle_impl, csr_crossing_edges_impl, csr_crossing_number_impl},
};
use petgraph_quality_metrics::{
    CrossingEdges, crossing_angle_with_crossing_edges, crossing_number_with_crossing_edges,
    try_crossing_angle, try_crossing_edges, try_crossing_edges_torus, try_crossing_number,
};
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Python class for storing information about crossing edges in a graph drawing
///
/// This class is used to efficiently cache the computation of edge crossings,
/// which can be expensive for large graphs. Once computed, the crossing information
/// can be reused for multiple metrics like crossing number and crossing angle.
#[pyclass]
#[pyo3(name = "CrossingEdges")]
pub struct PyCrossingEdges {
    crossing_edges: CrossingEdges<FloatType>,
}

/// Computes the crossing edges in a graph drawing
///
/// This function identifies all pairs of edges that cross in the drawing.
/// The result is cached in a CrossingEdges object that can be reused for
/// other metrics like crossing number and crossing angle.
///
/// :param graph: The graph being drawn
/// :type graph: Graph or DiGraph
/// :param drawing: The drawing to analyze
/// :type drawing: Drawing
/// :return: An object containing information about crossing edges
/// :rtype: CrossingEdges
/// :raises: NotImplementedError if the drawing type is not supported
#[pyfunction]
#[pyo3(name = "crossing_edges")]
fn py_crossing_edges<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    drawing: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(csr) = graph.cast::<PyCsrGraph>() {
        let layout = drawing.cast::<PyLayout2d>().map_err(|_| {
            PyTypeError::new_err("crossing_edges(CsrGraph, layout) expects a Layout2d drawing")
        })?;
        let list = csr_crossing_edges_impl(py, &csr.borrow(), &layout.borrow())?;
        return Ok(pyo3::types::PyList::new(py, list)?.into_any());
    }
    let graph = graph.cast::<PyGraphAdapter>().map_err(|_| {
        PyTypeError::new_err(
            "crossing_edges expects (CsrGraph, Layout2d) or legacy (Graph, Drawing)",
        )
    })?;
    let drawing = drawing.cast::<PyDrawing>().map_err(|_| {
        PyTypeError::new_err(
            "crossing_edges expects (CsrGraph, Layout2d) or legacy (Graph, Drawing)",
        )
    })?;
    let graph_borrowed = graph.borrow();
    let drawing_type = drawing.borrow().drawing_type();
    let crossing_edges = match drawing_type {
        DrawingType::Euclidean2d => {
            let drawing = drawing
                .clone()
                .cast::<PyDrawingEuclidean2d>()
                .map_err(|e| PyTypeError::new_err(e.to_string()))?
                .borrow();
            match graph_borrowed.graph() {
                GraphType::Graph(native_graph) => {
                    try_crossing_edges(native_graph, drawing.drawing())
                        .map_err(PyValueError::new_err)?
                }
                GraphType::DiGraph(native_graph) => {
                    try_crossing_edges(native_graph, drawing.drawing())
                        .map_err(PyValueError::new_err)?
                }
            }
        }
        DrawingType::Torus2d => {
            let drawing = drawing
                .clone()
                .cast::<PyDrawingTorus2d>()
                .map_err(|e| PyTypeError::new_err(e.to_string()))?
                .borrow();
            match graph_borrowed.graph() {
                GraphType::Graph(native_graph) => {
                    try_crossing_edges_torus(native_graph, drawing.drawing())
                        .map_err(PyValueError::new_err)?
                }
                GraphType::DiGraph(native_graph) => {
                    try_crossing_edges_torus(native_graph, drawing.drawing())
                        .map_err(PyValueError::new_err)?
                }
            }
        }
        _ => {
            return Err(PyNotImplementedError::new_err(
                "unsupported drawing type for crossing_edges",
            ));
        }
    };
    let py_obj = Py::new(py, PyCrossingEdges { crossing_edges })?;
    Ok(py_obj.into_bound(py).into_any())
}

/// Computes the crossing-angle penalty in a graph drawing
///
/// Crossing angle measures the angles at which edges cross. Angles closer to 90 degrees
/// (perpendicular crossings) yield lower `cos^2(theta)` penalties.
///
/// :param graph: The graph being drawn
/// :type graph: Graph, DiGraph, or CsrGraph
/// :param drawing: The drawing or layout to analyze
/// :type drawing: DrawingEuclidean2d or Layout2d
/// :return: The crossing-angle penalty
/// :rtype: float
#[pyfunction]
#[pyo3(name = "crossing_angle")]
fn py_crossing_angle(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
    drawing: &Bound<'_, PyAny>,
) -> PyResult<FloatType> {
    match extract_dual_binary(graph, drawing, "crossing_angle")? {
        DualBinaryInput::Csr(csr, layout) => csr_crossing_angle_impl(py, &csr, &layout),
        DualBinaryInput::Legacy(g, d) => match g.graph() {
            GraphType::Graph(native_graph) => try_crossing_angle(native_graph, d.drawing()),
            GraphType::DiGraph(native_graph) => try_crossing_angle(native_graph, d.drawing()),
        }
        .map_err(PyValueError::new_err),
    }
}

/// Computes the average crossing angle using pre-computed crossing edges
///
/// This function is more efficient when crossing edges have already been computed.
///
/// :param crossing_edges: Pre-computed crossing edges information
/// :type crossing_edges: CrossingEdges
/// :return: The average crossing angle in radians
/// :rtype: float
#[pyfunction]
#[pyo3(name = "crossing_angle_with_crossing_edges")]
fn py_crossing_angle_with_crossing_edges(crossing_edges: &PyCrossingEdges) -> FloatType {
    crossing_angle_with_crossing_edges(&crossing_edges.crossing_edges)
}

/// Computes the number of edge crossings in a graph drawing
///
/// Crossing number is the total number of pairs of edges that cross in the drawing.
/// Lower values generally indicate clearer, more readable layouts.
///
/// :param graph: The graph being drawn
/// :type graph: Graph, DiGraph, or CsrGraph
/// :param drawing: The drawing or layout to analyze
/// :type drawing: DrawingEuclidean2d or Layout2d
/// :return: The number of edge crossings
/// :rtype: float
#[pyfunction]
#[pyo3(name = "crossing_number")]
fn py_crossing_number(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
    drawing: &Bound<'_, PyAny>,
) -> PyResult<FloatType> {
    match extract_dual_binary(graph, drawing, "crossing_number")? {
        DualBinaryInput::Csr(csr, layout) => csr_crossing_number_impl(py, &csr, &layout),
        DualBinaryInput::Legacy(g, d) => match g.graph() {
            GraphType::Graph(native_graph) => try_crossing_number(native_graph, d.drawing()),
            GraphType::DiGraph(native_graph) => try_crossing_number(native_graph, d.drawing()),
        }
        .map_err(PyValueError::new_err),
    }
}

/// Computes the number of edge crossings using pre-computed crossing edges
///
/// This function is more efficient when crossing edges have already been computed.
///
/// :param crossing_edges: Pre-computed crossing edges information
/// :type crossing_edges: CrossingEdges
/// :return: The number of edge crossings
/// :rtype: float
#[pyfunction]
#[pyo3(name = "crossing_number_with_crossing_edges")]
fn py_crossing_number_with_crossing_edges(crossing_edges: &PyCrossingEdges) -> FloatType {
    crossing_number_with_crossing_edges(&crossing_edges.crossing_edges)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_crossing_angle, m)?)?;
    m.add_function(wrap_pyfunction!(py_crossing_angle_with_crossing_edges, m)?)?;
    m.add_function(wrap_pyfunction!(py_crossing_edges, m)?)?;
    m.add_function(wrap_pyfunction!(py_crossing_number, m)?)?;
    m.add_function(wrap_pyfunction!(py_crossing_number_with_crossing_edges, m)?)?;
    Ok(())
}
