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
    distance_matrix::{DistanceMatrixType, PyDistanceMatrix},
    drawing::{DrawingType, PyDrawing, PyDrawingEuclidean2d, PyDrawingTorus2d},
    graph::{GraphType, PyGraphAdapter},
};
use petgraph_quality_metrics::{
    angular_resolution, aspect_ratio, crossing_angle, crossing_angle_with_crossing_edges,
    crossing_edges, crossing_edges_torus, crossing_number, crossing_number_with_crossing_edges,
    gabriel_graph_property, ideal_edge_lengths, neighborhood_preservation, node_resolution, stress,
    CrossingEdges,
};
use pyo3::prelude::*;

/// Python class for storing information about crossing edges in a graph drawing
///
/// This class is used to efficiently cache the computation of edge crossings,
/// which can be expensive for large graphs. Once computed, the crossing information
/// can be reused for multiple metrics like crossing number and crossing angle.
#[pyclass]
#[pyo3(name = "CrossingEdges")]
pub struct PyCrossingEdges {
    crossing_edges: CrossingEdges,
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
fn py_crossing_edges(graph: &PyGraphAdapter, drawing: &Bound<PyDrawing>) -> PyCrossingEdges {
    Python::with_gil(|py| {
        let drawing_type = drawing.borrow().drawing_type();
        let crossing_edges = match drawing_type {
            DrawingType::Euclidean2d => {
                let drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingEuclidean2d>(py)
                    .unwrap()
                    .borrow_mut();
                match graph.graph() {
                    GraphType::Graph(native_graph) => {
                        crossing_edges(native_graph, drawing.drawing())
                    }
                    GraphType::DiGraph(native_graph) => {
                        crossing_edges(native_graph, drawing.drawing())
                    }
                }
            }
            DrawingType::Torus2d => {
                let drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingTorus2d>(py)
                    .unwrap()
                    .borrow_mut();
                match graph.graph() {
                    GraphType::Graph(native_graph) => {
                        crossing_edges_torus(native_graph, drawing.drawing())
                    }
                    GraphType::DiGraph(native_graph) => {
                        crossing_edges_torus(native_graph, drawing.drawing())
                    }
                }
            }
            _ => unimplemented!(),
        };
        PyCrossingEdges { crossing_edges }
    })
}

/// Computes the angular resolution of a graph drawing
///
/// Angular resolution is the minimum angle between any two edges incident on the same node.
/// Higher values indicate better readability as edges are more clearly separated.
///
/// :param graph: The graph being drawn
/// :type graph: Graph or DiGraph
/// :param drawing: The drawing to analyze
/// :type drawing: DrawingEuclidean2d
/// :return: The angular resolution in radians
/// :rtype: float
#[pyfunction]
#[pyo3(name = "angular_resolution")]
fn py_angular_resolution(graph: &PyGraphAdapter, drawing: &PyDrawingEuclidean2d) -> f32 {
    match graph.graph() {
        GraphType::Graph(native_graph) => angular_resolution(native_graph, drawing.drawing()),
        GraphType::DiGraph(native_graph) => angular_resolution(native_graph, drawing.drawing()),
    }
}

/// Computes the aspect ratio of a graph drawing
///
/// Aspect ratio is the ratio of the width to the height of the drawing.
/// Values close to 1.0 indicate a balanced drawing that uses space efficiently.
///
/// :param drawing: The drawing to analyze
/// :type drawing: DrawingEuclidean2d
/// :return: The aspect ratio (width/height)
/// :rtype: float
#[pyfunction]
#[pyo3(name = "aspect_ratio")]
fn py_aspect_ratio(drawing: &PyDrawingEuclidean2d) -> f32 {
    aspect_ratio(drawing.drawing())
}

/// Computes the average crossing angle in a graph drawing
///
/// Crossing angle is the angle at which edges cross. Angles closer to 90 degrees
/// (perpendicular crossings) are generally more readable than shallow angles.
///
/// :param graph: The graph being drawn
/// :type graph: Graph or DiGraph
/// :param drawing: The drawing to analyze
/// :type drawing: DrawingEuclidean2d
/// :return: The average crossing angle in radians
/// :rtype: float
#[pyfunction]
#[pyo3(name = "crossing_angle")]
fn py_crossing_angle(graph: &PyGraphAdapter, drawing: &PyDrawingEuclidean2d) -> f32 {
    match graph.graph() {
        GraphType::Graph(native_graph) => crossing_angle(native_graph, drawing.drawing()),
        GraphType::DiGraph(native_graph) => crossing_angle(native_graph, drawing.drawing()),
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
fn py_crossing_angle_with_crossing_edges(crossing_edges: &PyCrossingEdges) -> f32 {
    crossing_angle_with_crossing_edges(&crossing_edges.crossing_edges)
}

/// Computes the number of edge crossings in a graph drawing
///
/// Crossing number is the total number of pairs of edges that cross in the drawing.
/// Lower values generally indicate clearer, more readable layouts.
///
/// :param graph: The graph being drawn
/// :type graph: Graph or DiGraph
/// :param drawing: The drawing to analyze
/// :type drawing: DrawingEuclidean2d
/// :return: The number of edge crossings
/// :rtype: float
#[pyfunction]
#[pyo3(name = "crossing_number")]
fn py_crossing_number(graph: &PyGraphAdapter, drawing: &PyDrawingEuclidean2d) -> f32 {
    match graph.graph() {
        GraphType::Graph(native_graph) => crossing_number(native_graph, drawing.drawing()),
        GraphType::DiGraph(native_graph) => crossing_number(native_graph, drawing.drawing()),
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
fn py_crossing_number_with_crossing_edges(crossing_edges: &PyCrossingEdges) -> f32 {
    crossing_number_with_crossing_edges(&crossing_edges.crossing_edges)
}

/// Computes how well a drawing satisfies the Gabriel graph property
///
/// The Gabriel graph property states that for each edge (u,v), the disk with diameter
/// from u to v should not contain any other nodes. This metric measures the percentage
/// of edges that satisfy this property.
///
/// :param graph: The graph being drawn
/// :type graph: Graph or DiGraph
/// :param drawing: The drawing to analyze
/// :type drawing: DrawingEuclidean2d
/// :return: The percentage of edges satisfying the Gabriel graph property (0.0-1.0)
/// :rtype: float
#[pyfunction]
#[pyo3(name = "gabriel_graph_property")]
fn py_gabriel_graph_property(graph: &PyGraphAdapter, drawing: &PyDrawingEuclidean2d) -> f32 {
    match graph.graph() {
        GraphType::Graph(native_graph) => gabriel_graph_property(native_graph, drawing.drawing()),
        GraphType::DiGraph(native_graph) => gabriel_graph_property(native_graph, drawing.drawing()),
    }
}

/// Measures how well edge lengths in the drawing match their ideal lengths
///
/// Ideal edge lengths are proportional to the graph-theoretical distances.
/// This metric computes the average deviation between actual edge lengths
/// in the drawing and their ideal lengths based on the distance matrix.
///
/// :param graph: The graph being drawn
/// :type graph: Graph or DiGraph
/// :param drawing: The drawing to analyze
/// :type drawing: Drawing
/// :param distance_matrix: Matrix of ideal distances between nodes
/// :type distance_matrix: DistanceMatrix
/// :return: The average deviation from ideal edge lengths
/// :rtype: float
/// :raises: ValueError if the distance matrix type is not supported
/// :raises: NotImplementedError if the drawing type is not supported
#[pyfunction]
#[pyo3(name = "ideal_edge_lengths")]
fn py_ideal_edge_lengths(
    graph: &PyGraphAdapter,
    drawing: &Bound<PyDrawing>,
    distance_matrix: &PyDistanceMatrix,
) -> f32 {
    Python::with_gil(|py| {
        let drawing_type = drawing.borrow().drawing_type();
        match drawing_type {
            DrawingType::Euclidean2d => {
                let drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingEuclidean2d>(py)
                    .unwrap()
                    .borrow_mut();
                match distance_matrix.distance_matrix() {
                    DistanceMatrixType::Full(d) => match graph.graph() {
                        GraphType::Graph(native_graph) => {
                            ideal_edge_lengths(native_graph, drawing.drawing(), d)
                        }
                        GraphType::DiGraph(native_graph) => {
                            ideal_edge_lengths(native_graph, drawing.drawing(), d)
                        }
                    },
                    _ => panic!("unsupported distance matrix type"),
                }
            }
            DrawingType::Torus2d => {
                let drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingTorus2d>(py)
                    .unwrap()
                    .borrow_mut();
                match distance_matrix.distance_matrix() {
                    DistanceMatrixType::Full(d) => match graph.graph() {
                        GraphType::Graph(native_graph) => {
                            ideal_edge_lengths(native_graph, drawing.drawing(), d)
                        }
                        GraphType::DiGraph(native_graph) => {
                            ideal_edge_lengths(native_graph, drawing.drawing(), d)
                        }
                    },
                    _ => panic!("unsupported distance matrix type"),
                }
            }
            _ => {
                unimplemented!()
            }
        }
    })
}

/// Measures how well the drawing preserves node neighborhoods
///
/// This metric computes how well the layout preserves the local structure of the graph.
/// It measures whether nodes that are close in the graph structure are also
/// positioned close together in the drawing.
///
/// :param graph: The graph being drawn
/// :type graph: Graph or DiGraph
/// :param drawing: The drawing to analyze
/// :type drawing: DrawingEuclidean2d
/// :return: The neighborhood preservation score (higher is better)
/// :rtype: float
#[pyfunction]
#[pyo3(name = "neighborhood_preservation")]
fn py_neighborhood_preservation(graph: &PyGraphAdapter, drawing: &PyDrawingEuclidean2d) -> f32 {
    match graph.graph() {
        GraphType::Graph(native_graph) => {
            neighborhood_preservation(native_graph, drawing.drawing())
        }
        GraphType::DiGraph(native_graph) => {
            neighborhood_preservation(native_graph, drawing.drawing())
        }
    }
}

/// Measures how evenly nodes are distributed in the drawing space
///
/// Node resolution is the minimum distance between any two nodes, normalized
/// by the average distance between all pairs of nodes. Higher values indicate
/// a more even distribution of nodes.
///
/// :param drawing: The drawing to analyze
/// :type drawing: Drawing
/// :return: The node resolution score (higher is better)
/// :rtype: float
/// :raises: NotImplementedError if the drawing type is not supported
#[pyfunction]
#[pyo3(name = "node_resolution")]
fn py_node_resolution(drawing: &Bound<PyDrawing>) -> f32 {
    let drawing_type = drawing.borrow().drawing_type();
    Python::with_gil(|py| match drawing_type {
        DrawingType::Euclidean2d => {
            let drawing = drawing
                .into_py(py)
                .downcast_bound::<PyDrawingEuclidean2d>(py)
                .unwrap()
                .borrow();
            node_resolution(drawing.drawing())
        }
        DrawingType::Torus2d => {
            let drawing = drawing
                .into_py(py)
                .downcast_bound::<PyDrawingTorus2d>(py)
                .unwrap()
                .borrow();
            node_resolution(drawing.drawing())
        }
        _ => unimplemented!(),
    })
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
fn py_stress(drawing: &Bound<PyDrawing>, distance_matrix: &PyDistanceMatrix) -> f32 {
    Python::with_gil(|py| {
        let drawing_type = drawing.borrow().drawing_type();
        match distance_matrix.distance_matrix() {
            DistanceMatrixType::Full(d) => match drawing_type {
                DrawingType::Euclidean2d => {
                    let drawing = drawing
                        .into_py(py)
                        .downcast_bound::<PyDrawingEuclidean2d>(py)
                        .unwrap()
                        .borrow_mut();
                    stress(drawing.drawing(), d)
                }
                DrawingType::Torus2d => {
                    let drawing = drawing
                        .into_py(py)
                        .downcast_bound::<PyDrawingTorus2d>(py)
                        .unwrap()
                        .borrow_mut();
                    stress(drawing.drawing(), d)
                }
                _ => unimplemented!(),
            },
            _ => panic!("unsupported distance matrix type"),
        }
    })
}

pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(py_node_resolution, m)?)?;
    m.add_function(wrap_pyfunction!(py_stress, m)?)?;
    Ok(())
}
