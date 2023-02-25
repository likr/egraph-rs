use crate::{
    coordinates::PyCoordinates,
    distance_matrix::PyDistanceMatrix,
    graph::{IndexType, PyGraph},
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
fn py_crossing_edges(graph: &PyGraph, coordinates: &PyCoordinates) -> PyCrossingEdges {
    PyCrossingEdges {
        crossing_edges: crossing_edges(graph.graph(), coordinates.coordinates()),
    }
}

#[pyfunction]
#[pyo3(name = "angular_resolution")]
fn py_angular_resolution(graph: &PyGraph, coordinates: &PyCoordinates) -> f32 {
    angular_resolution(graph.graph(), coordinates.coordinates())
}

#[pyfunction]
#[pyo3(name = "aspect_ratio")]
fn py_aspect_ratio(coordinates: &PyCoordinates) -> f32 {
    aspect_ratio(coordinates.coordinates())
}

#[pyfunction]
#[pyo3(name = "crossing_angle")]
fn py_crossing_angle(
    graph: &PyGraph,
    coordinates: &PyCoordinates,
    crossing_edges: Option<&PyCrossingEdges>,
) -> f32 {
    if let Some(ce) = crossing_edges {
        crossing_angle_with_crossing_edges(
            graph.graph(),
            coordinates.coordinates(),
            &ce.crossing_edges,
        )
    } else {
        crossing_angle(graph.graph(), coordinates.coordinates())
    }
}

#[pyfunction]
#[pyo3(name = "crossing_number")]
fn py_crossing_number(
    graph: &PyGraph,
    coordinates: &PyCoordinates,
    crossing_edges: Option<&PyCrossingEdges>,
) -> f32 {
    if let Some(ce) = crossing_edges {
        crossing_number_with_crossing_edges(&ce.crossing_edges)
    } else {
        crossing_number(graph.graph(), coordinates.coordinates())
    }
}

#[pyfunction]
#[pyo3(name = "gabriel_graph_property")]
fn py_gabriel_graph_property(graph: &PyGraph, coordinates: &PyCoordinates) -> f32 {
    gabriel_graph_property(graph.graph(), coordinates.coordinates())
}

#[pyfunction]
#[pyo3(name = "ideal_edge_lengths")]
fn py_ideal_edge_lengths(
    graph: &PyGraph,
    coordinates: &PyCoordinates,
    distance_matrix: &PyDistanceMatrix,
) -> f32 {
    ideal_edge_lengths(
        graph.graph(),
        coordinates.coordinates(),
        distance_matrix.distance_matrix(),
    )
}

#[pyfunction]
#[pyo3(name = "neighborhood_preservation")]
fn py_neighborhood_preservation(graph: &PyGraph, coordinates: &PyCoordinates) -> f32 {
    neighborhood_preservation(graph.graph(), coordinates.coordinates())
}

#[pyfunction]
#[pyo3(name = "node_resolution")]
fn py_node_resolution(graph: &PyGraph, coordinates: &PyCoordinates) -> f32 {
    node_resolution(graph.graph(), coordinates.coordinates())
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
