/// Graph data structures for the Python bindings
///
/// This module provides the core graph data structures for the egraph-rs Python
/// bindings. It offers both undirected and directed graph implementations that
/// efficiently wrap Rust's petgraph library, exposing its functionality to Python.
///
/// The graph implementations are designed to be flexible, allowing arbitrary Python
/// objects to be stored as node and edge data. This enables seamless integration
/// with existing Python code and data structures.
///
/// # Main components
///
/// - `Graph`: Undirected graph implementation where edges have no direction
/// - `DiGraph`: Directed graph implementation where edges have a source and target
/// - `GraphAdapter`: Base class providing common functionality for both graph types
///
/// These graph classes support common graph operations such as adding/removing nodes
/// and edges, querying neighbors, finding paths, and traversal.
mod graph_base;

use graph_base::*;
use petgraph::prelude::*;
use pyo3::prelude::*;

/// Type alias for node data in a graph, which can be any Python object
pub type Node = PyObject;
/// Type alias for edge data in a graph, which can be any Python object
pub type Edge = PyObject;
/// Type alias for the index type used in graphs
pub type IndexType = u32;
/// Type alias for node indices in a graph
pub type NodeId = NodeIndex<IndexType>;

/// Enum representing either an undirected graph or a directed graph
///
/// This enum allows the code to work with both directed and undirected graphs
/// through a common interface, while preserving the specific behavior of each
/// graph type.
///
/// # Variants
///
/// * `Graph` - An undirected graph, where edges have no direction
/// * `DiGraph` - A directed graph, where edges have a source and target node
pub enum GraphType {
    /// An undirected graph, where edges have no direction
    Graph(Graph<Node, Edge, Undirected, IndexType>),
    /// A directed graph, where edges have a source and target
    DiGraph(Graph<Node, Edge, Directed, IndexType>),
}

/// Base class for Python graph types
///
/// This class serves as an adapter between Python graph objects and Rust's petgraph.
/// It handles both directed and undirected graphs through the `GraphType` enum.
#[pyclass(subclass)]
#[pyo3(name = "GraphAdapter")]
pub struct PyGraphAdapter {
    graph: GraphType,
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
    pub fn add_node(&mut self, value: PyObject) -> usize {
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
    pub fn node_weight(&self, a: usize) -> PyResult<PyObject> {
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
    pub fn add_edge(&mut self, a: usize, b: usize, value: PyObject) -> usize {
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
    pub fn edge_weight(&mut self, e: usize) -> PyResult<PyObject> {
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
    pub fn remove_node(&mut self, a: usize) -> PyResult<PyObject> {
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
    pub fn remove_edge(&mut self, e: usize) -> PyResult<PyObject> {
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
    pub fn edges(&self, a: usize) -> Vec<PyObject> {
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

/// Registers graph-related classes with the Python module
pub fn register(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyGraphAdapter>()?;
    graph_base::register(py, m)?;
    Ok(())
}
