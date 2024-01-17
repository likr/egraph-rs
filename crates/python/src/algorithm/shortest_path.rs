use crate::{
    distance_matrix::PyDistanceMatrix,
    graph::{GraphType, PyGraphAdapter},
};
use petgraph::visit::EdgeRef;
use petgraph_algorithm_shortest_path::{all_sources_bfs, all_sources_dijkstra, warshall_floyd};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "all_sources_bfs")]
fn py_all_sources_bfs(graph: &PyGraphAdapter, unit_edge_length: f32) -> PyDistanceMatrix {
    let distance_matrix = match graph.graph() {
        GraphType::Graph(g) => all_sources_bfs(g, unit_edge_length),
        GraphType::DiGraph(g) => all_sources_bfs(g, unit_edge_length),
    };
    PyDistanceMatrix::new_with_full_distance_matrix(distance_matrix)
}

#[pyfunction]
#[pyo3(name = "all_sources_dijkstra")]
fn py_all_sources_dijkstra(graph: &PyGraphAdapter, f: &PyAny) -> PyDistanceMatrix {
    let distance_matrix = match graph.graph() {
        GraphType::Graph(g) => all_sources_dijkstra(g, |e| {
            f.call1((e.id().index(),)).unwrap().extract().unwrap()
        }),
        GraphType::DiGraph(g) => all_sources_dijkstra(g, |e| {
            f.call1((e.id().index(),)).unwrap().extract().unwrap()
        }),
    };
    PyDistanceMatrix::new_with_full_distance_matrix(distance_matrix)
}

#[pyfunction]
#[pyo3(name = "warshall_floyd")]
fn py_warshall_floyd(graph: &PyGraphAdapter, f: &PyAny) -> PyDistanceMatrix {
    let distance_matrix = match graph.graph() {
        GraphType::Graph(g) => warshall_floyd(g, |e| {
            f.call1((e.id().index(),)).unwrap().extract().unwrap()
        }),
        GraphType::DiGraph(g) => warshall_floyd(g, |e| {
            f.call1((e.id().index(),)).unwrap().extract().unwrap()
        }),
    };
    PyDistanceMatrix::new_with_full_distance_matrix(distance_matrix)
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_all_sources_bfs, m)?)?;
    m.add_function(wrap_pyfunction!(py_all_sources_dijkstra, m)?)?;
    m.add_function(wrap_pyfunction!(py_warshall_floyd, m)?)?;
    Ok(())
}
