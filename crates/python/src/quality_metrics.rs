/// Graph layout quality metrics for evaluating drawing aesthetics
///
/// This module provides a collection of metrics to quantitatively assess the quality
/// of graph layouts. These metrics can be used to compare different layout algorithms,
/// tune parameters, or optimize layouts for specific aesthetic criteria.
///
/// The metrics include various aspects of drawing quality such as:
/// - Stress: How well distances in the layout match the graph-theoretical distances
/// - Angular resolution: The minimum angle between edges at a node
/// - Crossing angle: The angles at which edges cross
/// - Crossing number: The number of edge crossings in the layout
/// - Aspect ratio: The balance between width and height of the drawing
/// - Neighborhood preservation: How well local neighborhoods are preserved in the layout
/// - Node resolution: How evenly distributed the nodes are in the drawing space
/// - Gabriel graph property: A measure of whether the layout respects theoretical constraints
use crate::{
    FloatType,
    csr_graph::PyCsrGraph,
    distance_matrix::{DistanceMatrixType, PyDistanceMatrix},
    drawing::{DrawingType, PyDrawing, PyDrawingEuclidean2d, PyDrawingTorus2d},
    graph::{GraphType, PyGraphAdapter},
    layout_buffers::PyLayout2d,
    metrics::{
        csr_angular_resolution_impl, csr_aspect_ratio_impl, csr_crossing_angle_impl,
        csr_crossing_edges_impl, csr_crossing_number_impl, csr_gabriel_graph_property_impl,
        csr_ideal_edge_lengths_impl, csr_neighborhood_preservation_1hop_impl,
        csr_neighborhood_preservation_2hop_impl, csr_node_resolution_impl,
    },
};
use petgraph_algorithm_shortest_path::DistanceMatrix;
use petgraph_drawing::Drawing;
use petgraph_quality_metrics::{
    CrossingEdges, angular_resolution, aspect_ratio, crossing_angle_with_crossing_edges,
    crossing_number_with_crossing_edges, gabriel_graph_property, ideal_edge_lengths,
    neighborhood_preservation, neighborhood_preservation_2hop, node_resolution, stress,
    try_crossing_angle, try_crossing_edges, try_crossing_edges_torus, try_crossing_number,
};
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;

enum DualBinaryInput<'py> {
    Csr(PyRef<'py, PyCsrGraph>, PyRef<'py, PyLayout2d>),
    Legacy(PyRef<'py, PyGraphAdapter>, PyRef<'py, PyDrawingEuclidean2d>),
}

fn extract_dual_binary<'py>(
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

fn check_graph_drawing_len(graph: &PyGraphAdapter, drawing_len: usize) -> PyResult<()> {
    let n = match graph.graph() {
        GraphType::Graph(g) => g.node_count(),
        GraphType::DiGraph(g) => g.node_count(),
    };
    if n != drawing_len {
        return Err(PyValueError::new_err(format!(
            "graph node_count ({n}) does not match drawing length ({drawing_len})"
        )));
    }
    Ok(())
}

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

/// Computes the angular resolution of a graph drawing
///
/// Angular resolution is the minimum angle between any two edges incident on the same node.
/// Higher values indicate better readability as edges are more clearly separated.
///
/// :param graph: The graph being drawn
/// :type graph: Graph, DiGraph, or CsrGraph
/// :param drawing: The drawing or layout to analyze
/// :type drawing: DrawingEuclidean2d or Layout2d
/// :return: The angular resolution metric
/// :rtype: float
#[pyfunction]
#[pyo3(name = "angular_resolution")]
fn py_angular_resolution(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
    drawing: &Bound<'_, PyAny>,
) -> PyResult<FloatType> {
    match extract_dual_binary(graph, drawing, "angular_resolution")? {
        DualBinaryInput::Csr(csr, layout) => csr_angular_resolution_impl(py, &csr, &layout),
        DualBinaryInput::Legacy(g, d) => {
            check_graph_drawing_len(&g, d.len())?;
            Ok(match g.graph() {
                GraphType::Graph(native_graph) => angular_resolution(native_graph, d.drawing()),
                GraphType::DiGraph(native_graph) => angular_resolution(native_graph, d.drawing()),
            })
        }
    }
}

/// Computes the aspect ratio of a graph drawing
///
/// Aspect ratio is the ratio of the width to the height of the drawing.
/// Values close to 1.0 indicate a balanced drawing that uses space efficiently.
///
/// :param drawing: The drawing or layout to analyze
/// :type drawing: DrawingEuclidean2d or Layout2d
/// :return: The aspect ratio (`sigma_2 / sigma_1` in `[0.0, 1.0]`)
/// :rtype: float
#[pyfunction]
#[pyo3(name = "aspect_ratio")]
fn py_aspect_ratio(py: Python<'_>, drawing: &Bound<'_, PyAny>) -> PyResult<FloatType> {
    if let Ok(layout) = drawing.cast::<PyLayout2d>() {
        return csr_aspect_ratio_impl(py, &layout.borrow());
    }
    if let Ok(legacy_drawing) = drawing.cast::<PyDrawingEuclidean2d>() {
        return Ok(aspect_ratio(legacy_drawing.borrow().drawing()));
    }
    Err(PyTypeError::new_err(
        "aspect_ratio: drawing must be DrawingEuclidean2d or Layout2d",
    ))
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

/// Computes how well a drawing satisfies the Gabriel graph property
///
/// The Gabriel graph property states that for each edge (u,v), the disk with diameter
/// from u to v should not contain any other nodes. This metric measures squared violations
/// of this property.
///
/// :param graph: The graph being drawn
/// :type graph: Graph, DiGraph, or CsrGraph
/// :param drawing: The drawing or layout to analyze
/// :type drawing: DrawingEuclidean2d or Layout2d
/// :return: The Gabriel graph property violation sum
/// :rtype: float
#[pyfunction]
#[pyo3(name = "gabriel_graph_property")]
fn py_gabriel_graph_property(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
    drawing: &Bound<'_, PyAny>,
) -> PyResult<FloatType> {
    match extract_dual_binary(graph, drawing, "gabriel_graph_property")? {
        DualBinaryInput::Csr(csr, layout) => csr_gabriel_graph_property_impl(py, &csr, &layout),
        DualBinaryInput::Legacy(g, d) => {
            check_graph_drawing_len(&g, d.len())?;
            Ok(match g.graph() {
                GraphType::Graph(native_graph) => gabriel_graph_property(native_graph, d.drawing()),
                GraphType::DiGraph(native_graph) => {
                    gabriel_graph_property(native_graph, d.drawing())
                }
            })
        }
    }
}

/// Measures how well edge lengths in the drawing match their ideal lengths
///
/// Ideal edge lengths are proportional to the graph-theoretical distances (or edge weights /
/// unit length `1.0` for `(CsrGraph, Layout2d)`).
///
/// :param graph: The graph being drawn
/// :type graph: Graph, DiGraph, or CsrGraph
/// :param drawing: The drawing or layout to analyze
/// :type drawing: Drawing or Layout2d
/// :param distance_matrix: Matrix of ideal distances between nodes (required for legacy `Graph`/`Drawing`, omitted for `CsrGraph`/`Layout2d`)
/// :type distance_matrix: DistanceMatrix or None
/// :return: The relative squared deviation from ideal edge lengths
/// :rtype: float
/// :raises: ValueError if the distance matrix type or dimensions are invalid
/// :raises: NotImplementedError if the drawing type is not supported
#[pyfunction]
#[pyo3(name = "ideal_edge_lengths", signature = (graph, drawing, distance_matrix = None))]
fn py_ideal_edge_lengths(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
    drawing: &Bound<'_, PyAny>,
    distance_matrix: Option<&Bound<'_, PyAny>>,
) -> PyResult<FloatType> {
    if let Ok(csr) = graph.cast::<PyCsrGraph>() {
        let layout = drawing.cast::<PyLayout2d>().map_err(|_| {
            PyTypeError::new_err("ideal_edge_lengths: expected Layout2d when graph is CsrGraph")
        })?;
        if let Some(dm) = distance_matrix
            && !dm.is_none()
        {
            return Err(PyTypeError::new_err(
                "ideal_edge_lengths(CsrGraph, Layout2d) does not accept a distance_matrix argument",
            ));
        }
        return csr_ideal_edge_lengths_impl(py, &csr.borrow(), &layout.borrow());
    }

    let legacy_graph = graph.cast::<PyGraphAdapter>().map_err(|_| {
        PyTypeError::new_err("ideal_edge_lengths: graph must be CsrGraph, Graph, or DiGraph")
    })?;
    let legacy_drawing = drawing.cast::<PyDrawing>().map_err(|_| {
        PyTypeError::new_err("ideal_edge_lengths: expected Drawing when graph is Graph or DiGraph")
    })?;
    let dm_any = distance_matrix
        .filter(|dm| !dm.is_none())
        .ok_or_else(|| {
            PyTypeError::new_err(
                "ideal_edge_lengths() missing required 'distance_matrix' argument for legacy Graph/Drawing",
            )
        })?;
    let dm_ref = dm_any
        .cast::<PyDistanceMatrix>()
        .map_err(|_| PyTypeError::new_err("distance_matrix must be a DistanceMatrix"))?;

    let g = legacy_graph.borrow();
    let dm = dm_ref.borrow();
    let drawing_type = legacy_drawing.borrow().drawing_type();
    match drawing_type {
        DrawingType::Euclidean2d => {
            let drawing = legacy_drawing
                .clone()
                .cast::<PyDrawingEuclidean2d>()
                .map_err(|e| PyTypeError::new_err(e.to_string()))?
                .borrow();
            check_graph_drawing_len(&g, drawing.len())?;
            match dm.distance_matrix() {
                DistanceMatrixType::Full(d) => {
                    if d.shape() != (drawing.len(), drawing.len()) {
                        return Err(PyValueError::new_err(format!(
                            "distance matrix shape {:?} does not match drawing length {}",
                            d.shape(),
                            drawing.len()
                        )));
                    }
                    Ok(match g.graph() {
                        GraphType::Graph(native_graph) => {
                            ideal_edge_lengths(native_graph, drawing.drawing(), d)
                        }
                        GraphType::DiGraph(native_graph) => {
                            ideal_edge_lengths(native_graph, drawing.drawing(), d)
                        }
                    })
                }
                _ => Err(PyValueError::new_err("unsupported distance matrix type")),
            }
        }
        DrawingType::Torus2d => {
            let drawing = legacy_drawing
                .clone()
                .cast::<PyDrawingTorus2d>()
                .map_err(|e| PyTypeError::new_err(e.to_string()))?
                .borrow();
            let drawing_len = drawing.drawing().len();
            check_graph_drawing_len(&g, drawing_len)?;
            match dm.distance_matrix() {
                DistanceMatrixType::Full(d) => {
                    if d.shape() != (drawing_len, drawing_len) {
                        return Err(PyValueError::new_err(format!(
                            "distance matrix shape {:?} does not match drawing length {drawing_len}",
                            d.shape()
                        )));
                    }
                    Ok(match g.graph() {
                        GraphType::Graph(native_graph) => {
                            ideal_edge_lengths(native_graph, drawing.drawing(), d)
                        }
                        GraphType::DiGraph(native_graph) => {
                            ideal_edge_lengths(native_graph, drawing.drawing(), d)
                        }
                    })
                }
                _ => Err(PyValueError::new_err("unsupported distance matrix type")),
            }
        }
        _ => Err(PyNotImplementedError::new_err(
            "unsupported drawing type for ideal_edge_lengths",
        )),
    }
}

/// Measures how well the drawing preserves node neighborhoods
///
/// This metric computes how well the layout preserves the local structure of the graph.
/// It measures whether nodes that are close in the graph structure are also
/// positioned close together in the drawing.
///
/// :param graph: The graph being drawn
/// :type graph: Graph, DiGraph, or CsrGraph
/// :param drawing: The drawing or layout to analyze
/// :type drawing: DrawingEuclidean2d or Layout2d
/// :return: The neighborhood preservation score (higher is better)
/// :rtype: float
#[pyfunction]
#[pyo3(name = "neighborhood_preservation")]
fn py_neighborhood_preservation(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
    drawing: &Bound<'_, PyAny>,
) -> PyResult<FloatType> {
    match extract_dual_binary(graph, drawing, "neighborhood_preservation")? {
        DualBinaryInput::Csr(csr, layout) => {
            csr_neighborhood_preservation_1hop_impl(py, &csr, &layout)
        }
        DualBinaryInput::Legacy(g, d) => {
            check_graph_drawing_len(&g, d.len())?;
            Ok(match g.graph() {
                GraphType::Graph(native_graph) => {
                    neighborhood_preservation(native_graph, d.drawing())
                }
                GraphType::DiGraph(native_graph) => {
                    neighborhood_preservation(native_graph, d.drawing())
                }
            })
        }
    }
}

/// Measures how well the drawing preserves 2-hop node neighborhoods
///
/// This metric computes how well the layout preserves the 2-hop local structure of the graph.
/// It measures whether nodes that are within shortest path distance <= 2 in the graph structure are
/// positioned close together in the drawing.
///
/// :param graph: The graph being drawn
/// :type graph: Graph, DiGraph, or CsrGraph
/// :param drawing: The drawing or layout to analyze
/// :type drawing: DrawingEuclidean2d or Layout2d
/// :return: The 2-hop neighborhood preservation score (higher is better)
/// :rtype: float
#[pyfunction]
#[pyo3(name = "neighborhood_preservation_2hop")]
fn py_neighborhood_preservation_2hop(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
    drawing: &Bound<'_, PyAny>,
) -> PyResult<FloatType> {
    match extract_dual_binary(graph, drawing, "neighborhood_preservation_2hop")? {
        DualBinaryInput::Csr(csr, layout) => {
            csr_neighborhood_preservation_2hop_impl(py, &csr, &layout)
        }
        DualBinaryInput::Legacy(g, d) => {
            check_graph_drawing_len(&g, d.len())?;
            Ok(match g.graph() {
                GraphType::Graph(native_graph) => {
                    neighborhood_preservation_2hop(native_graph, d.drawing())
                }
                GraphType::DiGraph(native_graph) => {
                    neighborhood_preservation_2hop(native_graph, d.drawing())
                }
            })
        }
    }
}

/// Measures how evenly nodes are distributed in the drawing space
///
/// Node resolution evaluates minimum node separation violations relative to the drawing diameter.
///
/// :param drawing: The drawing or layout to analyze
/// :type drawing: Drawing or Layout2d
/// :return: The node resolution penalty
/// :rtype: float
/// :raises: NotImplementedError if the drawing type is not supported
#[pyfunction]
#[pyo3(name = "node_resolution")]
fn py_node_resolution(py: Python<'_>, drawing: &Bound<'_, PyAny>) -> PyResult<FloatType> {
    if let Ok(layout) = drawing.cast::<PyLayout2d>() {
        return csr_node_resolution_impl(py, &layout.borrow());
    }
    let legacy_drawing = drawing.cast::<PyDrawing>().map_err(|_| {
        PyTypeError::new_err("node_resolution: drawing must be Drawing or Layout2d")
    })?;
    let drawing_type = legacy_drawing.borrow().drawing_type();
    match drawing_type {
        DrawingType::Euclidean2d => {
            let drawing = legacy_drawing
                .clone()
                .cast::<PyDrawingEuclidean2d>()
                .map_err(|e| PyTypeError::new_err(e.to_string()))?
                .borrow();
            Ok(node_resolution(drawing.drawing()))
        }
        DrawingType::Torus2d => {
            let drawing = legacy_drawing
                .clone()
                .cast::<PyDrawingTorus2d>()
                .map_err(|e| PyTypeError::new_err(e.to_string()))?
                .borrow();
            Ok(node_resolution(drawing.drawing()))
        }
        _ => Err(PyNotImplementedError::new_err(
            "unsupported drawing type for node_resolution",
        )),
    }
}

/// Computes the stress of a graph drawing
///
/// Stress measures how well the Euclidean distances in the drawing match
/// the graph-theoretical distances. Lower values indicate a better match.
///
/// :param drawing: The drawing to analyze
/// :type drawing: Drawing
/// :param distance_matrix: Matrix of ideal distances between nodes
/// :type distance_matrix: DistanceMatrix
/// :return: The stress value (lower is better)
/// :rtype: float
/// :raises: ValueError if the distance matrix type is not supported
/// :raises: NotImplementedError if the drawing type is not supported
#[pyfunction]
#[pyo3(name = "stress")]
fn py_stress(
    drawing: &Bound<PyDrawing>,
    distance_matrix: &PyDistanceMatrix,
) -> PyResult<FloatType> {
    Python::attach(|_py| {
        let drawing_type = drawing.borrow().drawing_type();
        match distance_matrix.distance_matrix() {
            DistanceMatrixType::Full(d) => match drawing_type {
                DrawingType::Euclidean2d => {
                    let drawing = drawing
                        .clone()
                        .cast::<PyDrawingEuclidean2d>()
                        .map_err(|e| PyTypeError::new_err(e.to_string()))?
                        .borrow();
                    if d.shape() != (drawing.len(), drawing.len()) {
                        return Err(PyValueError::new_err(format!(
                            "distance matrix shape {:?} does not match drawing length {}",
                            d.shape(),
                            drawing.len()
                        )));
                    }
                    Ok(stress(drawing.drawing(), d))
                }
                DrawingType::Torus2d => {
                    let drawing = drawing
                        .clone()
                        .cast::<PyDrawingTorus2d>()
                        .map_err(|e| PyTypeError::new_err(e.to_string()))?
                        .borrow();
                    let drawing_len = drawing.drawing().len();
                    if d.shape() != (drawing_len, drawing_len) {
                        return Err(PyValueError::new_err(format!(
                            "distance matrix shape {:?} does not match drawing length {drawing_len}",
                            d.shape()
                        )));
                    }
                    Ok(stress(drawing.drawing(), d))
                }
                _ => Err(PyNotImplementedError::new_err(
                    "unsupported drawing type for stress",
                )),
            },
            _ => Err(PyValueError::new_err("unsupported distance matrix type")),
        }
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_angular_resolution, m)?)?;
    m.add_function(wrap_pyfunction!(py_aspect_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(py_crossing_angle, m)?)?;
    m.add_function(wrap_pyfunction!(py_crossing_angle_with_crossing_edges, m)?)?;
    m.add_function(wrap_pyfunction!(py_crossing_edges, m)?)?;
    m.add_function(wrap_pyfunction!(py_crossing_number, m)?)?;
    m.add_function(wrap_pyfunction!(py_crossing_number_with_crossing_edges, m)?)?;
    m.add_function(wrap_pyfunction!(py_gabriel_graph_property, m)?)?;
    m.add_function(wrap_pyfunction!(py_ideal_edge_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(py_neighborhood_preservation, m)?)?;
    m.add_function(wrap_pyfunction!(py_neighborhood_preservation_2hop, m)?)?;
    m.add_function(wrap_pyfunction!(py_node_resolution, m)?)?;

    m.add_function(wrap_pyfunction!(py_stress, m)?)?;
    Ok(())
}
