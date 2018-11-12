#![feature(specialization)]

extern crate petgraph;
extern crate pyo3;

pub use petgraph::prelude::*;
use pyo3::prelude::*;

#[derive(Default, Clone)]
pub struct Node {
    pub x: f64,
    pub y: f64,
}

impl Node {
    pub fn new(x: f64, y: f64) -> Node {
        Node { x, y }
    }

    pub fn empty() -> Node {
        Node { x: 0., y: 0. }
    }
}

#[derive(Default, Clone)]
pub struct Edge {}

impl Edge {
    pub fn new() -> Edge {
        Edge {}
    }
}

type GraphType = petgraph::Graph<Node, Edge, Undirected>;

#[pyclass]
struct Graph {
    graph: GraphType,
}

#[pymethods]
impl Graph {
    #[new]
    fn __new(obj: &PyRawObject) -> PyResult<()> {
        obj.init(|_| Graph {
            graph: GraphType::with_capacity(0, 0),
        })
    }

    fn add_node(&mut self) -> PyResult<usize> {
        Ok(self.graph.add_node(Node::default()).index())
    }

    fn add_edge(&mut self, u: usize, v: usize) -> PyResult<usize> {
        let u = NodeIndex::new(u);
        let v = NodeIndex::new(v);
        Ok(self.graph.add_edge(u, v, Edge::default()).index())
    }
}

#[pymodinit]
fn egraph(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Graph>()?;

    Ok(())
}
