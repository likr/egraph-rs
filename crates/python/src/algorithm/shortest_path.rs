use crate::{distance_matrix::PyDistanceMatrix, graph::PyGraph};
use petgraph::visit::EdgeRef;
use petgraph_algorithm_shortest_path::{all_sources_bfs, all_sources_dijkstra, warshall_floyd};
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyfunction]
#[pyo3(name = "all_sources_bfs")]
fn py_all_sources_bfs(graph: &PyGraph, unit_edge_length: f32) -> PyDistanceMatrix {
    let distance_matrix = all_sources_bfs(graph.graph(), unit_edge_length);
    PyDistanceMatrix::new(distance_matrix)
}

#[pyfunction]
#[pyo3(name = "all_sources_dijkstra")]
fn py_all_sources_dijkstra(graph: &PyGraph, f: &PyAny) -> PyDistanceMatrix {
    let mut distance = HashMap::new();
    for e in graph.graph().edge_indices() {
        let v = f.call1((e.index(),)).unwrap().extract().unwrap();
        distance.insert(e, v);
    }
    let distance_matrix = all_sources_dijkstra(graph.graph(), &mut |e| distance[&e.id()]);
    PyDistanceMatrix::new(distance_matrix)
}

#[pyfunction]
#[pyo3(name = "warshall_floyd")]
fn py_warshall_floyd(graph: &PyGraph, f: &PyAny) -> PyDistanceMatrix {
    let mut distance = HashMap::new();
    for e in graph.graph().edge_indices() {
        let v = f.call1((e.index(),)).unwrap().extract().unwrap();
        distance.insert(e, v);
    }
    let distance_matrix = warshall_floyd(graph.graph(), &mut |e| distance[&e.id()]);
    PyDistanceMatrix::new(distance_matrix)
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_all_sources_bfs, m)?)?;
    m.add_function(wrap_pyfunction!(py_all_sources_dijkstra, m)?)?;
    m.add_function(wrap_pyfunction!(py_warshall_floyd, m)?)?;
    Ok(())
}
