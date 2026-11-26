//! Python `GraphAdapter` base class wrapping undirected and directed petgraph instances.

use super::edge_length::{add_edges_to_graph, extract_edges_2col};
use super::graph_base::{
    graph_add_edge, graph_add_node, graph_contains_edge, graph_edge_count, graph_edge_endpoints,
    graph_edge_indices, graph_edge_weight, graph_edges, graph_externals, graph_filter_map,
    graph_find_edge, graph_map, graph_neighbors, graph_neighbors_directed,
    graph_neighbors_undirected, graph_node_count, graph_node_indices, graph_node_weight,
    graph_remove_edge, graph_remove_node,
};
use super::{Edge, GraphType, IndexType, Node};
use petgraph::prelude::*;
use pyo3::prelude::*;

/// Base class for Python graph types
///
/// This class serves as an adapter between Python graph objects and Rust's petgraph.
/// It handles both directed and undirected graphs through the `GraphType` enum.
#[pyclass(subclass)]
#[pyo3(name = "GraphAdapter")]
pub struct PyGraphAdapter {
    pub(super) graph: GraphType,
}

impl PyGraphAdapter {
    pub fn new(graph: Graph<Node, Edge, Undirected, IndexType>) -> Self {
        PyGraphAdapter {
            graph: GraphType::Graph(graph),
        }
    }

    pub fn new_digraph(graph: Graph<Node, Edge, Directed, IndexType>) -> Self {
        PyGraphAdapter {
            graph: GraphType::DiGraph(graph),
        }
    }

    /// Returns a reference to the underlying graph
    pub fn graph(&self) -> &GraphType {
        &self.graph
    }

    /// Returns a mutable reference to the underlying graph
    pub fn graph_mut(&mut self) -> &mut GraphType {
        &mut self.graph
    }
}

#[pymethods]
impl PyGraphAdapter {
    /// Add edges in bulk from an `(M, 2)` integer NumPy array with `None` edge weights.
    pub fn add_edges_from_numpy(
        &mut self,
        py: Python<'_>,
        edges: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let pairs = extract_edges_2col(edges)?;
        match &mut self.graph {
            GraphType::Graph(g) => add_edges_to_graph(py, g, &pairs),
            GraphType::DiGraph(g) => add_edges_to_graph(py, g, &pairs),
        }
    }

    /// Returns the number of nodes in the graph
    ///
    /// :return: The number of nodes in the graph
    /// :rtype: int
    pub fn node_count(&self) -> usize {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_node_count(native_graph),
            GraphType::DiGraph(native_graph) => graph_node_count(native_graph),
        }
    }

    /// Returns the number of edges in the graph
    ///
    /// :return: The number of edges in the graph
    /// :rtype: int
    pub fn edge_count(&self) -> usize {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_edge_count(native_graph),
            GraphType::DiGraph(native_graph) => graph_edge_count(native_graph),
        }
    }

    /// Adds a new node to the graph with the given value
    ///
    /// :param value: The Python object to store at this node
    /// :type value: object
    /// :return: The index of the newly added node
    /// :rtype: int
    pub fn add_node(&mut self, value: Py<PyAny>) -> usize {
        match self.graph_mut() {
            GraphType::Graph(native_graph) => graph_add_node(native_graph, value),
            GraphType::DiGraph(native_graph) => graph_add_node(native_graph, value),
        }
    }

    /// Returns the value associated with a node
    ///
    /// :param a: The index of the node
    /// :type a: int
    /// :return: The Python object stored at the node
    /// :rtype: object
    /// :raises ValueError: If the node index is invalid
    pub fn node_weight(&self, a: usize) -> PyResult<Py<PyAny>> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_node_weight(native_graph, a),
            GraphType::DiGraph(native_graph) => graph_node_weight(native_graph, a),
        }
    }

    /// Adds a new edge to the graph with the given value
    ///
    /// :param a: The source node index
    /// :type a: int
    /// :param b: The target node index
    /// :type b: int
    /// :param value: The Python object to store at this edge
    /// :type value: object
    /// :return: The index of the newly added edge
    /// :rtype: int
    pub fn add_edge(&mut self, a: usize, b: usize, value: Py<PyAny>) -> usize {
        match self.graph_mut() {
            GraphType::Graph(native_graph) => graph_add_edge(native_graph, a, b, value),
            GraphType::DiGraph(native_graph) => graph_add_edge(native_graph, a, b, value),
        }
    }

    /// Returns the value associated with an edge
    ///
    /// :param e: The index of the edge
    /// :type e: int
    /// :return: The Python object stored at the edge
    /// :rtype: object
    /// :raises ValueError: If the edge index is invalid
    pub fn edge_weight(&mut self, e: usize) -> PyResult<Py<PyAny>> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_edge_weight(native_graph, e),
            GraphType::DiGraph(native_graph) => graph_edge_weight(native_graph, e),
        }
    }

    /// Returns the endpoint nodes of an edge
    ///
    /// :param e: The index of the edge
    /// :type e: int
    /// :return: A tuple of (source, target) node indices
    /// :rtype: tuple[int, int]
    /// :raises ValueError: If the edge index is invalid
    pub fn edge_endpoints(&self, e: usize) -> PyResult<(usize, usize)> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_edge_endpoints(native_graph, e),
            GraphType::DiGraph(native_graph) => graph_edge_endpoints(native_graph, e),
        }
    }

    /// Removes a node from the graph
    ///
    /// :param a: The index of the node to remove
    /// :type a: int
    /// :return: The Python object that was stored at the node
    /// :rtype: object
    /// :raises ValueError: If the node index is invalid
    pub fn remove_node(&mut self, a: usize) -> PyResult<Py<PyAny>> {
        match self.graph_mut() {
            GraphType::Graph(native_graph) => graph_remove_node(native_graph, a),
            GraphType::DiGraph(native_graph) => graph_remove_node(native_graph, a),
        }
    }

    /// Removes an edge from the graph
    ///
    /// :param e: The index of the edge to remove
    /// :type e: int
    /// :return: The Python object that was stored at the edge
    /// :rtype: object
    /// :raises ValueError: If the edge index is invalid
    pub fn remove_edge(&mut self, e: usize) -> PyResult<Py<PyAny>> {
        match self.graph_mut() {
            GraphType::Graph(native_graph) => graph_remove_edge(native_graph, e),
            GraphType::DiGraph(native_graph) => graph_remove_edge(native_graph, e),
        }
    }

    /// Returns all neighbors of a node
    ///
    /// :param a: The index of the node
    /// :type a: int
    /// :return: A list of indices of neighboring nodes
    /// :rtype: list[int]
    pub fn neighbors(&self, a: usize) -> Vec<usize> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_neighbors(native_graph, a),
            GraphType::DiGraph(native_graph) => graph_neighbors(native_graph, a),
        }
    }

    /// Returns neighbors of a node in a specific direction
    ///
    /// :param a: The index of the node
    /// :type a: int
    /// :param dir: The direction: 0 for outgoing, any other value for incoming
    /// :type dir: int
    /// :return: A list of indices of neighboring nodes in the specified direction
    /// :rtype: list[int]
    pub fn neighbors_directed(&self, a: usize, dir: usize) -> Vec<usize> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_neighbors_directed(native_graph, a, dir),
            GraphType::DiGraph(native_graph) => graph_neighbors_directed(native_graph, a, dir),
        }
    }

    /// Returns all neighbors of a node, ignoring edge direction
    ///
    /// :param a: The index of the node
    /// :type a: int
    /// :return: A list of indices of all neighboring nodes
    /// :rtype: list[int]
    pub fn neighbors_undirected(&self, a: usize) -> Vec<usize> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_neighbors_undirected(native_graph, a),
            GraphType::DiGraph(native_graph) => graph_neighbors_undirected(native_graph, a),
        }
    }

    /// Returns all edges connected to a node
    ///
    /// :param a: The index of the node
    /// :type a: int
    /// :return: A list of edge values (Python objects)
    /// :rtype: list[object]
    pub fn edges(&self, a: usize) -> Vec<Py<PyAny>> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_edges(native_graph, a),
            GraphType::DiGraph(native_graph) => graph_edges(native_graph, a),
        }
    }

    /// Checks if an edge exists between two nodes
    ///
    /// :param a: The source node index
    /// :type a: int
    /// :param b: The target node index
    /// :type b: int
    /// :return: True if an edge exists, False otherwise
    /// :rtype: bool
    pub fn contains_edge(&self, a: usize, b: usize) -> bool {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_contains_edge(native_graph, a, b),
            GraphType::DiGraph(native_graph) => graph_contains_edge(native_graph, a, b),
        }
    }

    /// Finds the edge between two nodes
    ///
    /// :param a: The source node index
    /// :type a: int
    /// :param b: The target node index
    /// :type b: int
    /// :return: The edge index if found
    /// :rtype: int
    /// :raises ValueError: If no edge exists between the nodes
    pub fn find_edge(&self, a: usize, b: usize) -> PyResult<usize> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_find_edge(native_graph, a, b),
            GraphType::DiGraph(native_graph) => graph_find_edge(native_graph, a, b),
        }
    }

    /// Returns nodes with no incoming or outgoing edges
    ///
    /// :param dir: The direction: 0 for outgoing (nodes with no outgoing edges),
    ///             any other value for incoming (nodes with no incoming edges)
    /// :type dir: int
    /// :return: A list of node indices that have no edges in the specified direction
    /// :rtype: list[int]
    pub fn externals(&self, dir: usize) -> Vec<usize> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_externals(native_graph, dir),
            GraphType::DiGraph(native_graph) => graph_externals(native_graph, dir),
        }
    }

    /// Returns all node indices in the graph
    ///
    /// :return: A list of all node indices
    /// :rtype: list[int]
    pub fn node_indices(&self) -> Vec<usize> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_node_indices(native_graph),
            GraphType::DiGraph(native_graph) => graph_node_indices(native_graph),
        }
    }

    /// Returns all edge indices in the graph
    ///
    /// :return: A list of all edge indices
    /// :rtype: list[int]
    pub fn edge_indices(&self) -> Vec<usize> {
        match self.graph() {
            GraphType::Graph(native_graph) => graph_edge_indices(native_graph),
            GraphType::DiGraph(native_graph) => graph_edge_indices(native_graph),
        }
    }

    /// Creates a new graph by applying mapping functions to all nodes and edges
    ///
    /// :param node_map: A Python function that takes (node_index, node_value) and returns a new node value
    /// :type node_map: callable
    /// :param edge_map: A Python function that takes (edge_index, edge_value) and returns a new edge value
    /// :type edge_map: callable
    /// :return: A new graph with the mapped values
    /// :rtype: GraphAdapter
    pub fn map(&self, node_map: &Bound<PyAny>, edge_map: &Bound<PyAny>) -> Self {
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

    /// Creates a new graph by selectively mapping nodes and edges
    ///
    /// :param node_map: A Python function that takes (node_index, node_value) and returns a new node value or None
    /// :type node_map: callable
    /// :param edge_map: A Python function that takes (edge_index, edge_value) and returns a new edge value or None
    /// :type edge_map: callable
    /// :return: A new graph containing only the nodes and edges for which the mapping functions returned non-None values
    /// :rtype: GraphAdapter
    pub fn filter_map(&self, node_map: &Bound<PyAny>, edge_map: &Bound<PyAny>) -> Self {
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

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraphAdapter>()?;
    Ok(())
}
