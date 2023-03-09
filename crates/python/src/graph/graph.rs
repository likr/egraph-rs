use crate::graph::{Edge, GraphType, IndexType, Node, PyGraphAdapter};
use petgraph::{
    graph::{edge_index, node_index},
    prelude::*,
    EdgeType,
};
use pyo3::{exceptions::PyValueError, prelude::*};

pub fn graph_node_count<Ty: EdgeType>(graph: &Graph<Node, Edge, Ty, IndexType>) -> usize {
    graph.node_count()
}

pub fn graph_edge_count<Ty: EdgeType>(graph: &Graph<Node, Edge, Ty, IndexType>) -> usize {
    graph.edge_count()
}

pub fn graph_add_node<Ty: EdgeType>(
    graph: &mut Graph<Node, Edge, Ty, IndexType>,
    value: PyObject,
) -> usize {
    graph.add_node(value).index()
}

pub fn graph_node_weight<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    a: usize,
) -> PyResult<PyObject> {
    let a = node_index(a);
    graph
        .node_weight(a)
        .map(|node| node.clone())
        .ok_or_else(|| PyValueError::new_err("invalid node index"))
}

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

pub fn graph_edge_weight<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    e: usize,
) -> PyResult<PyObject> {
    let e = edge_index(e);
    graph
        .edge_weight(e)
        .map(|edge| edge.clone())
        .ok_or_else(|| PyValueError::new_err("invalid edge index"))
}

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

pub fn graph_remove_node<Ty: EdgeType>(
    graph: &mut Graph<Node, Edge, Ty, IndexType>,
    a: usize,
) -> PyResult<PyObject> {
    let a = node_index(a);
    graph
        .remove_node(a)
        .ok_or_else(|| PyValueError::new_err("invalid node index"))
}

pub fn graph_remove_edge<Ty: EdgeType>(
    graph: &mut Graph<Node, Edge, Ty, IndexType>,
    e: usize,
) -> PyResult<PyObject> {
    let e = edge_index(e);
    graph
        .remove_edge(e)
        .ok_or_else(|| PyValueError::new_err("invalid node index"))
}

pub fn graph_neighbors<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    a: usize,
) -> Vec<usize> {
    graph
        .neighbors(node_index(a))
        .map(|u| u.index())
        .collect::<Vec<_>>()
}

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

pub fn graph_neighbors_undirected<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    a: usize,
) -> Vec<usize> {
    graph
        .neighbors_undirected(node_index(a))
        .map(|u| u.index())
        .collect::<Vec<_>>()
}

pub fn graph_edges<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    a: usize,
) -> Vec<PyObject> {
    graph
        .edges(node_index(a))
        .map(|e| e.weight().clone())
        .collect::<Vec<_>>()
}

pub fn graph_contains_edge<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    a: usize,
    b: usize,
) -> bool {
    let a = node_index(a);
    let b = node_index(b);
    graph.contains_edge(a, b)
}

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

pub fn graph_node_indices<Ty: EdgeType>(graph: &Graph<Node, Edge, Ty, IndexType>) -> Vec<usize> {
    graph.node_indices().map(|u| u.index()).collect::<Vec<_>>()
}

pub fn graph_edge_indices<Ty: EdgeType>(graph: &Graph<Node, Edge, Ty, IndexType>) -> Vec<usize> {
    graph.edge_indices().map(|e| e.index()).collect::<Vec<_>>()
}

pub fn graph_map<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    node_map: &PyAny,
    edge_map: &PyAny,
) -> Graph<Node, Edge, Ty, IndexType> {
    graph.map(
        |u, node| PyObject::from(node_map.call1((u.index(), node)).unwrap()),
        |e, edge| PyObject::from(edge_map.call1((e.index(), edge)).unwrap()),
    )
}

pub fn graph_filter_map<Ty: EdgeType>(
    graph: &Graph<Node, Edge, Ty, IndexType>,
    node_map: &PyAny,
    edge_map: &PyAny,
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

#[pyclass(extends = PyGraphAdapter)]
#[pyo3(name = "Graph")]
pub struct PyGraph;

#[pymethods]
impl PyGraph {
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

#[pyclass(extends = PyGraphAdapter)]
#[pyo3(name = "DiGraph")]
pub struct PyDiGraph;

#[pymethods]
impl PyDiGraph {
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

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyDiGraph>()?;
    Ok(())
}
