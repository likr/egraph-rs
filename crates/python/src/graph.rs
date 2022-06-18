use petgraph::graph::{edge_index, node_index};
use petgraph::{Directed, Direction};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub type Node = PyObject;
pub type Edge = PyObject;
pub type EdgeType = Directed;
pub type IndexType = u32;
type GraphType = petgraph::Graph<Node, Edge, EdgeType, IndexType>;

#[pyclass]
#[pyo3(name = "Graph")]
pub struct PyGraph {
    graph: GraphType,
}

impl PyGraph {
    pub fn new_from_graph(graph: GraphType) -> PyGraph {
        PyGraph { graph }
    }

    pub fn graph(&self) -> &GraphType {
        &self.graph
    }
}

#[pymethods]
impl PyGraph {
    #[new]
    pub fn new() -> PyGraph {
        PyGraph::new_from_graph(GraphType::with_capacity(0, 0))
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn add_node(&mut self, value: PyObject) -> usize {
        self.graph.add_node(value).index()
    }

    pub fn node_weight(&self, a: usize) -> PyResult<PyObject> {
        let a = node_index(a);
        self.graph
            .node_weight(a)
            .map(|node| node.clone())
            .ok_or_else(|| PyValueError::new_err("invalid node index"))
    }

    pub fn add_edge(&mut self, a: usize, b: usize, value: PyObject) -> usize {
        let a = node_index(a);
        let b = node_index(b);
        self.graph.add_edge(a, b, value).index()
    }

    pub fn edge_weight(&mut self, e: usize) -> PyResult<PyObject> {
        let e = edge_index(e);
        self.graph
            .edge_weight(e)
            .map(|edge| edge.clone())
            .ok_or_else(|| PyValueError::new_err("invalid edge index"))
    }

    pub fn edge_endpoints(&self, e: usize) -> PyResult<(usize, usize)> {
        let e = edge_index(e);
        self.graph
            .edge_endpoints(e)
            .map(|(u, v)| (u.index(), v.index()))
            .ok_or_else(|| PyValueError::new_err("invalid edge index"))
    }

    pub fn remove_node(&mut self, a: usize) -> PyResult<PyObject> {
        let a = node_index(a);
        self.graph
            .remove_node(a)
            .ok_or_else(|| PyValueError::new_err("invalid node index"))
    }

    pub fn remove_edge(&mut self, e: usize) -> PyResult<PyObject> {
        let e = edge_index(e);
        self.graph
            .remove_edge(e)
            .ok_or_else(|| PyValueError::new_err("invalid node index"))
    }

    pub fn neighbors(&self, a: usize) -> Vec<usize> {
        self.graph
            .neighbors(node_index(a))
            .map(|u| u.index())
            .collect::<Vec<_>>()
    }

    pub fn neighbors_directed(&self, a: usize, dir: usize) -> Vec<usize> {
        let a = node_index(a);
        let dir = match dir {
            0 => Direction::Outgoing,
            _ => Direction::Incoming,
        };
        self.graph
            .neighbors_directed(a, dir)
            .map(|u| u.index())
            .collect::<Vec<_>>()
    }

    pub fn neighbors_undirected(&self, a: usize) -> Vec<usize> {
        self.graph
            .neighbors_undirected(node_index(a))
            .map(|u| u.index())
            .collect::<Vec<_>>()
    }

    pub fn edges(&self, a: usize) -> Vec<PyObject> {
        self.graph
            .edges(node_index(a))
            .map(|e| e.weight().clone())
            .collect::<Vec<_>>()
    }

    pub fn contains_edge(&self, a: usize, b: usize) -> bool {
        let a = node_index(a);
        let b = node_index(b);
        self.graph.contains_edge(a, b)
    }

    pub fn find_edge(&self, a: usize, b: usize) -> PyResult<usize> {
        let a = node_index(a);
        let b = node_index(b);
        self.graph
            .find_edge(a, b)
            .map(|e| e.index())
            .ok_or_else(|| PyValueError::new_err("invalid edge index"))
    }

    pub fn externals(&self, dir: usize) -> Vec<usize> {
        let dir = match dir {
            0 => Direction::Outgoing,
            _ => Direction::Incoming,
        };
        self.graph
            .externals(dir)
            .map(|u| u.index())
            .collect::<Vec<_>>()
    }

    pub fn node_indices(&self) -> Vec<usize> {
        self.graph
            .node_indices()
            .map(|u| u.index())
            .collect::<Vec<_>>()
    }

    pub fn edge_indices(&self) -> Vec<usize> {
        self.graph
            .edge_indices()
            .map(|e| e.index())
            .collect::<Vec<_>>()
    }

    pub fn map(&self, node_map: &PyAny, edge_map: &PyAny) -> PyGraph {
        PyGraph {
            graph: self.graph.map(
                |u, node| PyObject::from(node_map.call1((u.index(), node)).unwrap()),
                |e, edge| PyObject::from(edge_map.call1((e.index(), edge)).unwrap()),
            ),
        }
    }

    pub fn filter_map(&self, node_map: &PyAny, edge_map: &PyAny) -> PyGraph {
        PyGraph {
            graph: self.graph.filter_map(
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
            ),
        }
    }
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    Ok(())
}
