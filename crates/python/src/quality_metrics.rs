use crate::{
    coordinates::PyCoordinates,
    distance_matrix::PyDistanceMatrix,
    graph::{GraphType, IndexType, PyGraphAdapter},
};
use petgraph::graph::EdgeIndex;
use petgraph_quality_metrics::{
    angular_resolution, aspect_ratio, crossing_angle, crossing_angle_with_crossing_edges,
    crossing_edges, crossing_number, crossing_number_with_crossing_edges, gabriel_graph_property,
    ideal_edge_lengths, neighborhood_preservation, node_resolution, stress,
};
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "CrossingEdges")]
pub struct PyCrossingEdges {
    crossing_edges: Vec<(EdgeIndex<IndexType>, EdgeIndex<IndexType>)>,
}

#[pyfunction]
#[pyo3(name = "crossing_edges")]
fn py_crossing_edges(graph: &PyGraphAdapter, coordinates: &PyCoordinates) -> PyCrossingEdges {
    PyCrossingEdges {
        crossing_edges: match graph.graph() {
            GraphType::Graph(native_graph) => {
                crossing_edges(native_graph, coordinates.coordinates())
            }
            GraphType::DiGraph(native_graph) => {
                crossing_edges(native_graph, coordinates.coordinates())
            }
        },
    }
}

#[pyfunction]
#[pyo3(name = "angular_resolution")]
fn py_angular_resolution(graph: &PyGraphAdapter, coordinates: &PyCoordinates) -> f32 {
    match graph.graph() {
        GraphType::Graph(native_graph) => {
            angular_resolution(native_graph, coordinates.coordinates())
        }
        GraphType::DiGraph(native_graph) => {
            angular_resolution(native_graph, coordinates.coordinates())
        }
    }
}

#[pyfunction]
#[pyo3(name = "aspect_ratio")]
fn py_aspect_ratio(coordinates: &PyCoordinates) -> f32 {
    aspect_ratio(coordinates.coordinates())
}

#[pyfunction]
#[pyo3(name = "crossing_angle")]
fn py_crossing_angle(
    graph: &PyGraphAdapter,
    coordinates: &PyCoordinates,
    crossing_edges: Option<&PyCrossingEdges>,
) -> f32 {
    if let Some(ce) = crossing_edges {
        match graph.graph() {
            GraphType::Graph(native_graph) => crossing_angle_with_crossing_edges(
                native_graph,
                coordinates.coordinates(),
                &ce.crossing_edges,
            ),
            GraphType::DiGraph(native_graph) => crossing_angle_with_crossing_edges(
                native_graph,
                coordinates.coordinates(),
                &ce.crossing_edges,
            ),
        }
    } else {
        match graph.graph() {
            GraphType::Graph(native_graph) => {
                crossing_angle(native_graph, coordinates.coordinates())
            }
            GraphType::DiGraph(native_graph) => {
                crossing_angle(native_graph, coordinates.coordinates())
            }
        }
    }
}

#[pyfunction]
#[pyo3(name = "crossing_number")]
fn py_crossing_number(
    graph: &PyGraphAdapter,
    coordinates: &PyCoordinates,
    crossing_edges: Option<&PyCrossingEdges>,
) -> f32 {
    if let Some(ce) = crossing_edges {
        crossing_number_with_crossing_edges(&ce.crossing_edges)
    } else {
        match graph.graph() {
            GraphType::Graph(native_graph) => {
                crossing_number(native_graph, coordinates.coordinates())
            }
            GraphType::DiGraph(native_graph) => {
                crossing_number(native_graph, coordinates.coordinates())
            }
        }
    }
}

#[pyfunction]
#[pyo3(name = "gabriel_graph_property")]
fn py_gabriel_graph_property(graph: &PyGraphAdapter, coordinates: &PyCoordinates) -> f32 {
    match graph.graph() {
        GraphType::Graph(native_graph) => {
            gabriel_graph_property(native_graph, coordinates.coordinates())
        }
        GraphType::DiGraph(native_graph) => {
            gabriel_graph_property(native_graph, coordinates.coordinates())
        }
    }
}

#[pyfunction]
#[pyo3(name = "ideal_edge_lengths")]
fn py_ideal_edge_lengths(
    graph: &PyGraphAdapter,
    coordinates: &PyCoordinates,
    distance_matrix: &PyDistanceMatrix,
) -> f32 {
    match graph.graph() {
        GraphType::Graph(native_graph) => ideal_edge_lengths(
            native_graph,
            coordinates.coordinates(),
            distance_matrix.distance_matrix(),
        ),
        GraphType::DiGraph(native_graph) => ideal_edge_lengths(
            native_graph,
            coordinates.coordinates(),
            distance_matrix.distance_matrix(),
        ),
    }
}

#[pyfunction]
#[pyo3(name = "neighborhood_preservation")]
fn py_neighborhood_preservation(graph: &PyGraphAdapter, coordinates: &PyCoordinates) -> f32 {
    match graph.graph() {
        GraphType::Graph(native_graph) => {
            neighborhood_preservation(native_graph, coordinates.coordinates())
        }
        GraphType::DiGraph(native_graph) => {
            neighborhood_preservation(native_graph, coordinates.coordinates())
        }
    }
}

#[pyfunction]
#[pyo3(name = "node_resolution")]
fn py_node_resolution(graph: &PyGraphAdapter, coordinates: &PyCoordinates) -> f32 {
    match graph.graph() {
        GraphType::Graph(native_graph) => node_resolution(native_graph, coordinates.coordinates()),
        GraphType::DiGraph(native_graph) => {
            node_resolution(native_graph, coordinates.coordinates())
        }
    }
}

#[pyfunction]
#[pyo3(name = "stress")]
fn py_stress(coordinates: &PyCoordinates, distance_matrix: &PyDistanceMatrix) -> f32 {
    stress(coordinates.coordinates(), distance_matrix.distance_matrix())
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
