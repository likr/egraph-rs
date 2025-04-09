use crate::graph::{Edge, GraphType, IndexType, Node, PyGraphAdapter};
use petgraph::{
    graph::{edge_index, node_index},
    prelude::*,
    EdgeType,
};
use pyo3::{exceptions::PyValueError, prelude::*};

/// Returns the number of nodes in a graph
///
/// # Parameters
/// * `graph` - The graph to query
pub fn graph_node_count<Ty: EdgeType>(graph: &Graph<Node, Edge, Ty, IndexType>) -> usize {
    graph.node_count()
}

/// Returns the number of edges in a graph
///
/// # Parameters
/// * `graph` - The graph to query
pub fn graph_edge_count<Ty: EdgeType>(graph: &Graph<Node, Edge, Ty, IndexType>) -> usize {
    graph.edge_count()
}

/// Adds a node to a graph with the given value
///
/// # Parameters
/// * `graph` - The graph to modify
/// * `value` - The Python object to store at the new node
///
/// # Returns
/// The index of the newly added node
pub fn graph_add_node<Ty: EdgeType>(
    graph: &mut Graph<Node, Edge, Ty, IndexType>,
    value: PyObject,
) -> usize {
    graph.add_node(value).index()
}

/// Retrieves the value stored at a node
///
/// # Parameters
/// * `graph` - The graph to query
/// * `a` - The index of the node
///
/// # Returns
/// The Python object stored at the node
///
/// # Errors
/// Returns a `PyValueError` if the node index is invalid
pub fn graph_node_weight<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    a: usize,
) -> PyResult<PyObject> {
    let a = node_index(a);
    graph
        .node_weight(a)
        .cloned()
        .ok_or_else(|| PyValueError::new_err("invalid node index"))
}

/// Adds an edge to a graph with the given value
///
/// # Parameters
/// * `graph` - The graph to modify
/// * `a` - The source node index
/// * `b` - The target node index
/// * `value` - The Python object to store at the new edge
///
/// # Returns
/// The index of the newly added edge
pub fn graph_add_edge<Ty: EdgeType>(
    graph: &mut Graph<Node, Edge, Ty, IndexType>,
    a: usize,
    b: usize,
    value: PyObject,
) -> usize {
    let a = node_index(a);
    let b = node_index(b);
    graph.add_edge(a, b, value).index()
}

/// Retrieves the value stored at an edge
///
/// # Parameters
/// * `graph` - The graph to query
/// * `e` - The index of the edge
///
/// # Returns
/// The Python object stored at the edge
///
/// # Errors
/// Returns a `PyValueError` if the edge index is invalid
pub fn graph_edge_weight<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    e: usize,
) -> PyResult<PyObject> {
    let e = edge_index(e);
    graph
        .edge_weight(e)
        .cloned()
        .ok_or_else(|| PyValueError::new_err("invalid edge index"))
}

/// Returns the endpoint nodes of an edge
///
/// # Parameters
/// * `graph` - The graph to query
/// * `e` - The index of the edge
///
/// # Returns
/// A tuple of (source, target) node indices
///
/// # Errors
/// Returns a `PyValueError` if the edge index is invalid
pub fn graph_edge_endpoints<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    e: usize,
) -> PyResult<(usize, usize)> {
    let e = edge_index(e);
    graph
        .edge_endpoints(e)
        .map(|(u, v)| (u.index(), v.index()))
        .ok_or_else(|| PyValueError::new_err("invalid edge index"))
}

/// Removes a node from a graph
///
/// # Parameters
/// * `graph` - The graph to modify
/// * `a` - The index of the node to remove
///
/// # Returns
/// The Python object that was stored at the node
///
/// # Errors
/// Returns a `PyValueError` if the node index is invalid
pub fn graph_remove_node<Ty: EdgeType>(
    graph: &mut Graph<Node, Edge, Ty, IndexType>,
    a: usize,
) -> PyResult<PyObject> {
    let a = node_index(a);
    graph
        .remove_node(a)
        .ok_or_else(|| PyValueError::new_err("invalid node index"))
}

/// Removes an edge from a graph
///
/// # Parameters
/// * `graph` - The graph to modify
/// * `e` - The index of the edge to remove
///
/// # Returns
/// The Python object that was stored at the edge
///
/// # Errors
/// Returns a `PyValueError` if the edge index is invalid
pub fn graph_remove_edge<Ty: EdgeType>(
    graph: &mut Graph<Node, Edge, Ty, IndexType>,
    e: usize,
) -> PyResult<PyObject> {
    let e = edge_index(e);
    graph
        .remove_edge(e)
        .ok_or_else(|| PyValueError::new_err("invalid node index"))
}

/// Returns all neighbors of a node
///
/// # Parameters
/// * `graph` - The graph to query
/// * `a` - The index of the node
///
/// # Returns
/// A vector of indices of neighboring nodes
pub fn graph_neighbors<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    a: usize,
) -> Vec<usize> {
    graph
        .neighbors(node_index(a))
        .map(|u| u.index())
        .collect::<Vec<_>>()
}

/// Returns neighbors of a node in a specific direction
///
/// # Parameters
/// * `graph` - The graph to query
/// * `a` - The index of the node
/// * `dir` - The direction: 0 for outgoing, any other value for incoming
///
/// # Returns
/// A vector of indices of neighboring nodes in the specified direction
pub fn graph_neighbors_directed<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    a: usize,
    dir: usize,
) -> Vec<usize> {
    let a = node_index(a);
    let dir = match dir {
        0 => Direction::Outgoing,
        _ => Direction::Incoming,
    };
    graph
        .neighbors_directed(a, dir)
        .map(|u| u.index())
        .collect::<Vec<_>>()
}

/// Returns all neighbors of a node, ignoring edge direction
///
/// # Parameters
/// * `graph` - The graph to query
/// * `a` - The index of the node
///
/// # Returns
/// A vector of indices of all neighboring nodes
pub fn graph_neighbors_undirected<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    a: usize,
) -> Vec<usize> {
    graph
        .neighbors_undirected(node_index(a))
        .map(|u| u.index())
        .collect::<Vec<_>>()
}

/// Returns all edges connected to a node
///
/// # Parameters
/// * `graph` - The graph to query
/// * `a` - The index of the node
///
/// # Returns
/// A vector of edge values (Python objects)
pub fn graph_edges<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    a: usize,
) -> Vec<PyObject> {
    graph
        .edges(node_index(a))
        .map(|e| e.weight().clone())
        .collect::<Vec<_>>()
}

/// Checks if an edge exists between two nodes
///
/// # Parameters
/// * `graph` - The graph to query
/// * `a` - The source node index
/// * `b` - The target node index
///
/// # Returns
/// `true` if an edge exists, `false` otherwise
pub fn graph_contains_edge<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    a: usize,
    b: usize,
) -> bool {
    let a = node_index(a);
    let b = node_index(b);
    graph.contains_edge(a, b)
}

/// Finds the edge between two nodes
///
/// # Parameters
/// * `graph` - The graph to query
/// * `a` - The source node index
/// * `b` - The target node index
///
/// # Returns
/// The edge index if found
///
/// # Errors
/// Returns a `PyValueError` if no edge exists between the nodes
pub fn graph_find_edge<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    a: usize,
    b: usize,
) -> PyResult<usize> {
    let a = node_index(a);
    let b = node_index(b);
    graph
        .find_edge(a, b)
        .map(|e| e.index())
        .ok_or_else(|| PyValueError::new_err("invalid edge index"))
}

/// Returns nodes with no incoming or outgoing edges
///
/// # Parameters
/// * `graph` - The graph to query
/// * `dir` - The direction: 0 for outgoing (nodes with no outgoing edges),
///           any other value for incoming (nodes with no incoming edges)
///
/// # Returns
/// A vector of node indices that have no edges in the specified direction
pub fn graph_externals<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    dir: usize,
) -> Vec<usize> {
    let dir = match dir {
        0 => Direction::Outgoing,
        _ => Direction::Incoming,
    };
    graph.externals(dir).map(|u| u.index()).collect::<Vec<_>>()
}

/// Returns all node indices in a graph
///
/// # Parameters
/// * `graph` - The graph to query
///
/// # Returns
/// A vector of all node indices
pub fn graph_node_indices<Ty: EdgeType>(graph: &Graph<Node, Edge, Ty, IndexType>) -> Vec<usize> {
    graph.node_indices().map(|u| u.index()).collect::<Vec<_>>()
}

/// Returns all edge indices in a graph
///
/// # Parameters
/// * `graph` - The graph to query
///
/// # Returns
/// A vector of all edge indices
pub fn graph_edge_indices<Ty: EdgeType>(graph: &Graph<Node, Edge, Ty, IndexType>) -> Vec<usize> {
    graph.edge_indices().map(|e| e.index()).collect::<Vec<_>>()
}

/// Creates a new graph by applying mapping functions to all nodes and edges
///
/// # Parameters
/// * `graph` - The source graph
/// * `node_map` - A Python function that takes (node_index, node_value) and returns a new node value
/// * `edge_map` - A Python function that takes (edge_index, edge_value) and returns a new edge value
///
/// # Returns
/// A new graph with the mapped values
pub fn graph_map<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    node_map: &Bound<PyAny>,
    edge_map: &Bound<PyAny>,
) -> Graph<Node, Edge, Ty, IndexType> {
    graph.map(
        |u, node| PyObject::from(node_map.call1((u.index(), node)).unwrap()),
        |e, edge| PyObject::from(edge_map.call1((e.index(), edge)).unwrap()),
    )
}

/// Creates a new graph by selectively mapping nodes and edges
///
/// # Parameters
/// * `graph` - The source graph
/// * `node_map` - A Python function that takes (node_index, node_value) and returns a new node value or None
/// * `edge_map` - A Python function that takes (edge_index, edge_value) and returns a new edge value or None
///
/// # Returns
/// A new graph containing only the nodes and edges for which the mapping functions returned non-None values
pub fn graph_filter_map<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    node_map: &Bound<PyAny>,
    edge_map: &Bound<PyAny>,
) -> Graph<Node, Edge, Ty, IndexType> {
    graph.filter_map(
        |u, node| {
            let result = node_map.call1((u.index(), node)).unwrap();
            if result.is_none() {
                None
            } else {
                Some(PyObject::from(result))
            }
        },
        |e, edge| {
            let result = edge_map.call1((e.index(), edge)).unwrap();
            if result.is_none() {
                None
            } else {
                Some(PyObject::from(result))
            }
        },
    )
}

/// Python class for undirected graphs
///
/// This class represents an undirected graph where edges have no direction.
/// Nodes and edges can store any Python object.
#[pyclass(extends = PyGraphAdapter)]
#[pyo3(name = "Graph")]
pub struct PyGraph;

#[pymethods]
impl PyGraph {
    /// Creates a new empty undirected graph
    ///
    /// :return: A new empty undirected graph
    /// :rtype: Graph
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyGraphAdapter {
            graph: GraphType::Graph(Graph::<Node, Edge, Undirected, IndexType>::with_capacity(
                0, 0,
            )),
        })
        .add_subclass(Self)
    }
}

/// Python class for directed graphs
///
/// This class represents a directed graph where edges have a source and target.
/// Nodes and edges can store any Python object.
#[pyclass(extends = PyGraphAdapter)]
#[pyo3(name = "DiGraph")]
pub struct PyDiGraph;

#[pymethods]
impl PyDiGraph {
    /// Creates a new empty directed graph
    ///
    /// :return: A new empty directed graph
    /// :rtype: DiGraph
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyGraphAdapter {
            graph: GraphType::DiGraph(Graph::<Node, Edge, Directed, IndexType>::with_capacity(
                0, 0,
            )),
        })
        .add_subclass(Self)
    }
}

/// Registers graph classes with the Python module
pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyDiGraph>()?;
    Ok(())
}
