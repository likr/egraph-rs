mod graph;

use graph::*;
use petgraph::prelude::*;
use pyo3::prelude::*;

pub type Node = PyObject;
pub type Edge = PyObject;
pub type IndexType = u32;

pub enum GraphType {
    Graph(Graph<Node, Edge, Undirected, IndexType>),
    DiGraph(Graph<Node, Edge, Directed, IndexType>),
}

#[pyclass(subclass)]
#[pyo3(name = "GraphAdapter")]
pub struct PyGraphAdapter {
    graph: GraphType,
}

impl PyGraphAdapter {
    pub fn graph(&self) -> &GraphType {
        &self.graph
    }

    pub fn graph_mut(&mut self) -> &mut GraphType {
        &mut self.graph
    }
}

#[pymethods]
impl PyGraphAdapter {
    pub fn node_count(&self) -> usize {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_node_count(native_graph),
            GraphType::DiGraph(native_graph) => graph_node_count(native_graph),
        }
    }

    pub fn edge_count(&self) -> usize {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_edge_count(native_graph),
            GraphType::DiGraph(native_graph) => graph_edge_count(native_graph),
        }
    }

    pub fn add_node(&mut self, value: PyObject) -> usize {
        match self.graph_mut() {
            GraphType::Graph(native_graph) => graph_add_node(native_graph, value),
            GraphType::DiGraph(native_graph) => graph_add_node(native_graph, value),
        }
    }

    pub fn node_weight(&self, a: usize) -> PyResult<PyObject> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_node_weight(native_graph, a),
            GraphType::DiGraph(native_graph) => graph_node_weight(native_graph, a),
        }
    }

    pub fn add_edge(&mut self, a: usize, b: usize, value: PyObject) -> usize {
        match self.graph_mut() {
            GraphType::Graph(native_graph) => graph_add_edge(native_graph, a, b, value),
            GraphType::DiGraph(native_graph) => graph_add_edge(native_graph, a, b, value),
        }
    }

    pub fn edge_weight(&mut self, e: usize) -> PyResult<PyObject> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_edge_weight(native_graph, e),
            GraphType::DiGraph(native_graph) => graph_edge_weight(native_graph, e),
        }
    }

    pub fn edge_endpoints(&self, e: usize) -> PyResult<(usize, usize)> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_edge_endpoints(native_graph, e),
            GraphType::DiGraph(native_graph) => graph_edge_endpoints(native_graph, e),
        }
    }

    pub fn remove_node(&mut self, a: usize) -> PyResult<PyObject> {
        match self.graph_mut() {
            GraphType::Graph(native_graph) => graph_remove_node(native_graph, a),
            GraphType::DiGraph(native_graph) => graph_remove_node(native_graph, a),
        }
    }

    pub fn remove_edge(&mut self, e: usize) -> PyResult<PyObject> {
        match self.graph_mut() {
            GraphType::Graph(native_graph) => graph_remove_edge(native_graph, e),
            GraphType::DiGraph(native_graph) => graph_remove_edge(native_graph, e),
        }
    }

    pub fn neighbors(&self, a: usize) -> Vec<usize> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_neighbors(native_graph, a),
            GraphType::DiGraph(native_graph) => graph_neighbors(native_graph, a),
        }
    }

    pub fn neighbors_directed(&self, a: usize, dir: usize) -> Vec<usize> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_neighbors_directed(native_graph, a, dir),
            GraphType::DiGraph(native_graph) => graph_neighbors_directed(native_graph, a, dir),
        }
    }

    pub fn neighbors_undirected(&self, a: usize) -> Vec<usize> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_neighbors_undirected(native_graph, a),
            GraphType::DiGraph(native_graph) => graph_neighbors_undirected(native_graph, a),
        }
    }

    pub fn edges(&self, a: usize) -> Vec<PyObject> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_edges(native_graph, a),
            GraphType::DiGraph(native_graph) => graph_edges(native_graph, a),
        }
    }

    pub fn contains_edge(&self, a: usize, b: usize) -> bool {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_contains_edge(native_graph, a, b),
            GraphType::DiGraph(native_graph) => graph_contains_edge(native_graph, a, b),
        }
    }

    pub fn find_edge(&self, a: usize, b: usize) -> PyResult<usize> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_find_edge(native_graph, a, b),
            GraphType::DiGraph(native_graph) => graph_find_edge(native_graph, a, b),
        }
    }

    pub fn externals(&self, dir: usize) -> Vec<usize> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_externals(native_graph, dir),
            GraphType::DiGraph(native_graph) => graph_externals(native_graph, dir),
        }
    }

    pub fn node_indices(&self) -> Vec<usize> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_node_indices(native_graph),
            GraphType::DiGraph(native_graph) => graph_node_indices(native_graph),
        }
    }

    pub fn edge_indices(&self) -> Vec<usize> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_edge_indices(native_graph),
            GraphType::DiGraph(native_graph) => graph_edge_indices(native_graph),
        }
    }

    pub fn map(&self, node_map: &PyAny, edge_map: &PyAny) -> Self {
        Self {
            graph: match self.graph() {
                GraphType::Graph(native_graph) => {
                    GraphType::Graph(graph_map(native_graph, node_map, edge_map))
                }
                GraphType::DiGraph(native_graph) => {
                    GraphType::DiGraph(graph_map(native_graph, node_map, edge_map))
                }
            },
        }
    }

    pub fn filter_map(&self, node_map: &PyAny, edge_map: &PyAny) -> Self {
        Self {
            graph: match self.graph() {
                GraphType::Graph(native_graph) => {
                    GraphType::Graph(graph_filter_map(native_graph, node_map, edge_map))
                }
                GraphType::DiGraph(native_graph) => {
                    GraphType::DiGraph(graph_filter_map(native_graph, node_map, edge_map))
                }
            },
        }
    }
}

pub fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGraphAdapter>()?;
    graph::register(py, m)?;
    Ok(())
}
