use crate::{
    distance_matrix::{DistanceMatrixType, PyDistanceMatrix},
    drawing::{DrawingType, PyDrawing},
    graph::{GraphType, PyGraphAdapter},
};
use petgraph_quality_metrics::{
    angular_resolution, aspect_ratio, crossing_angle, crossing_angle_with_crossing_edges,
    crossing_edges, crossing_edges_torus, crossing_number, crossing_number_with_crossing_edges,
    gabriel_graph_property, ideal_edge_lengths, neighborhood_preservation, node_resolution, stress,
    CrossingEdges,
};
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "CrossingEdges")]
pub struct PyCrossingEdges {
    crossing_edges: CrossingEdges,
}

#[pyfunction]
#[pyo3(name = "crossing_edges")]
fn py_crossing_edges(graph: &PyGraphAdapter, drawing: &PyDrawing) -> PyCrossingEdges {
    match drawing.drawing() {
        DrawingType::Drawing2D(drawing) => PyCrossingEdges {
            crossing_edges: match graph.graph() {
                GraphType::Graph(native_graph) => crossing_edges(native_graph, drawing),
                GraphType::DiGraph(native_graph) => crossing_edges(native_graph, drawing),
            },
        },
        DrawingType::DrawingTorus(drawing) => PyCrossingEdges {
            crossing_edges: match graph.graph() {
                GraphType::Graph(native_graph) => crossing_edges_torus(native_graph, drawing),
                GraphType::DiGraph(native_graph) => crossing_edges_torus(native_graph, drawing),
            },
        },
    }
}

#[pyfunction]
#[pyo3(name = "angular_resolution")]
fn py_angular_resolution(graph: &PyGraphAdapter, drawing: &PyDrawing) -> f32 {
    match drawing.drawing() {
        DrawingType::Drawing2D(drawing) => match graph.graph() {
            GraphType::Graph(native_graph) => angular_resolution(native_graph, drawing),
            GraphType::DiGraph(native_graph) => angular_resolution(native_graph, drawing),
        },
        _ => unimplemented!(),
    }
}

#[pyfunction]
#[pyo3(name = "aspect_ratio")]
fn py_aspect_ratio(drawing: &PyDrawing) -> f32 {
    match drawing.drawing() {
        DrawingType::Drawing2D(drawing) => aspect_ratio(drawing),
        _ => unimplemented!(),
    }
}

#[pyfunction]
#[pyo3(name = "crossing_angle")]
fn py_crossing_angle(
    graph: &PyGraphAdapter,
    drawing: &PyDrawing,
    crossing_edges: Option<&PyCrossingEdges>,
) -> f32 {
    if let Some(ce) = crossing_edges {
        crossing_angle_with_crossing_edges(&ce.crossing_edges)
    } else {
        match drawing.drawing() {
            DrawingType::Drawing2D(drawing) => match graph.graph() {
                GraphType::Graph(native_graph) => crossing_angle(native_graph, drawing),
                GraphType::DiGraph(native_graph) => crossing_angle(native_graph, drawing),
            },
            _ => unimplemented!(),
        }
    }
}

#[pyfunction]
#[pyo3(name = "crossing_number")]
fn py_crossing_number(
    graph: &PyGraphAdapter,
    drawing: &PyDrawing,
    crossing_edges: Option<&PyCrossingEdges>,
) -> f32 {
    if let Some(ce) = crossing_edges {
        crossing_number_with_crossing_edges(&ce.crossing_edges)
    } else {
        match drawing.drawing() {
            DrawingType::Drawing2D(drawing) => match graph.graph() {
                GraphType::Graph(native_graph) => crossing_number(native_graph, drawing),
                GraphType::DiGraph(native_graph) => crossing_number(native_graph, drawing),
            },
            _ => unimplemented!(),
        }
    }
}

#[pyfunction]
#[pyo3(name = "gabriel_graph_property")]
fn py_gabriel_graph_property(graph: &PyGraphAdapter, drawing: &PyDrawing) -> f32 {
    match drawing.drawing() {
        DrawingType::Drawing2D(drawing) => match graph.graph() {
            GraphType::Graph(native_graph) => gabriel_graph_property(native_graph, drawing),
            GraphType::DiGraph(native_graph) => gabriel_graph_property(native_graph, drawing),
        },
        _ => unimplemented!(),
    }
}

#[pyfunction]
#[pyo3(name = "ideal_edge_lengths")]
fn py_ideal_edge_lengths(
    graph: &PyGraphAdapter,
    drawing: &PyDrawing,
    distance_matrix: &PyDistanceMatrix,
) -> f32 {
    match distance_matrix.distance_matrix() {
        DistanceMatrixType::Full(d) => match drawing.drawing() {
            DrawingType::Drawing2D(drawing) => match graph.graph() {
                GraphType::Graph(native_graph) => ideal_edge_lengths(native_graph, drawing, d),
                GraphType::DiGraph(native_graph) => ideal_edge_lengths(native_graph, drawing, d),
            },
            _ => unimplemented!(),
        },
        _ => panic!("unsupported distance matrix type"),
    }
}

#[pyfunction]
#[pyo3(name = "neighborhood_preservation")]
fn py_neighborhood_preservation(graph: &PyGraphAdapter, drawing: &PyDrawing) -> f32 {
    match drawing.drawing() {
        DrawingType::Drawing2D(drawing) => match graph.graph() {
            GraphType::Graph(native_graph) => neighborhood_preservation(native_graph, drawing),
            GraphType::DiGraph(native_graph) => neighborhood_preservation(native_graph, drawing),
        },
        _ => unimplemented!(),
    }
}

#[pyfunction]
#[pyo3(name = "node_resolution")]
fn py_node_resolution(drawing: &PyDrawing) -> f32 {
    match drawing.drawing() {
        DrawingType::Drawing2D(drawing) => node_resolution(drawing),
        DrawingType::DrawingTorus(drawing) => node_resolution(drawing),
    }
}

#[pyfunction]
#[pyo3(name = "stress")]
fn py_stress(drawing: &PyDrawing, distance_matrix: &PyDistanceMatrix) -> f32 {
    match distance_matrix.distance_matrix() {
        DistanceMatrixType::Full(d) => match drawing.drawing() {
            DrawingType::Drawing2D(drawing) => stress(drawing, d),
            DrawingType::DrawingTorus(drawing) => stress(drawing, d),
        },
        _ => panic!("unsupported distance matrix type"),
    }
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_angular_resolution, m)?)?;
    m.add_function(wrap_pyfunction!(py_aspect_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(py_crossing_angle, m)?)?;
    m.add_function(wrap_pyfunction!(py_crossing_edges, m)?)?;
    m.add_function(wrap_pyfunction!(py_crossing_number, m)?)?;
    m.add_function(wrap_pyfunction!(py_gabriel_graph_property, m)?)?;
    m.add_function(wrap_pyfunction!(py_ideal_edge_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(py_neighborhood_preservation, m)?)?;
    m.add_function(wrap_pyfunction!(py_node_resolution, m)?)?;
    m.add_function(wrap_pyfunction!(py_stress, m)?)?;
    Ok(())
}
