#![feature(specialization)]

extern crate petgraph;
extern crate pyo3;

use petgraph::graph::node_index;
use petgraph::prelude::*;
use pyo3::class::iter::PyIterProtocol;
use pyo3::prelude::*;
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

type GraphType = petgraph::Graph<(), (), Directed, usize>;

#[pyclass]
struct Neighbors {
    iter: petgraph::graph::WalkNeighbors<usize>,
    graph: Rc<RefCell<GraphType>>,
}

#[pyproto]
impl PyIterProtocol for Neighbors {
    fn __iter__(&mut self) -> PyResult<PyObject> {
        Ok(self.into())
    }

    fn __next__(&mut self) -> PyResult<Option<PyObject>> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        Ok(self
            .iter
            .next_node(&self.graph.borrow())
            .map(|index| index.index().to_object(py)))
    }
}

#[pyclass]
struct NodeIndices {
    iter: petgraph::graph::NodeIndices<usize>,
}

#[pyproto]
impl PyIterProtocol for NodeIndices {
    fn __iter__(&mut self) -> PyResult<PyObject> {
        Ok(self.into())
    }

    fn __next__(&mut self) -> PyResult<Option<PyObject>> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        Ok(self.iter.next().map(|index| index.index().to_object(py)))
    }
}

#[pyclass]
struct EdgeIndices {
    iter: petgraph::graph::EdgeIndices<usize>,
}

#[pyproto]
impl PyIterProtocol for EdgeIndices {
    fn __iter__(&mut self) -> PyResult<PyObject> {
        Ok(self.into())
    }

    fn __next__(&mut self) -> PyResult<Option<PyObject>> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        Ok(self.iter.next().map(|index| index.index().to_object(py)))
    }
}

#[pyclass]
struct Graph {
    graph: Rc<RefCell<GraphType>>,
}

impl Graph {
    fn graph(&self) -> Ref<GraphType> {
        self.graph.borrow()
    }

    fn graph_mut(&self) -> RefMut<GraphType> {
        self.graph.borrow_mut()
    }
}

#[pymethods]
impl Graph {
    #[new]
    fn __new__(obj: &PyRawObject) -> PyResult<()> {
        obj.init(|_| Graph {
            graph: Rc::new(RefCell::new(GraphType::with_capacity(0, 0))),
        })
    }

    fn add_node(&mut self) -> PyResult<usize> {
        Ok(self.graph_mut().add_node(()).index())
    }

    fn add_edge(&mut self, u: usize, v: usize) -> PyResult<usize> {
        let u = NodeIndex::new(u);
        let v = NodeIndex::new(v);
        Ok(self.graph_mut().add_edge(u, v, ()).index())
    }

    fn remove_node(&mut self, u: usize) -> PyResult<()> {
        self.graph_mut().remove_node(NodeIndex::new(u));
        Ok(())
    }

    fn remove_edge(&mut self, u: usize) -> PyResult<()> {
        self.graph_mut().remove_edge(EdgeIndex::new(u));
        Ok(())
    }

    fn node_count(&self) -> PyResult<usize> {
        Ok(self.graph().node_count())
    }

    fn edge_count(&self) -> PyResult<usize> {
        Ok(self.graph().edge_count())
    }

    fn neighbors(&self, a: usize) -> PyResult<Neighbors> {
        let iter = self.graph().neighbors(node_index(a)).detach();
        Ok(Neighbors {
            iter,
            graph: self.graph.clone(),
        })
    }

    fn node_indices(&self) -> PyResult<NodeIndices> {
        Ok(NodeIndices {
            iter: self.graph().node_indices(),
        })
    }

    fn edge_indices(&self) -> PyResult<EdgeIndices> {
        Ok(EdgeIndices {
            iter: self.graph().edge_indices(),
        })
    }
}

#[pyclass]
struct FM3 {
    fm3: egraph::layout::fm3::FM3,
}

#[pymethods]
impl FM3 {
    #[new]
    fn __new__(obj: &PyRawObject) -> PyResult<()> {
        obj.init(|_| FM3 {
            fm3: egraph::layout::fm3::FM3::new(),
        })
    }

    #[call]
    fn __call__(&self, graph: &Graph) -> PyResult<()> {
        let points = self.fm3.call(&graph.graph());
        for point in points.iter() {
            println!("{} {}", point.x, point.y);
        }
        Ok(())
    }
}

#[pymodinit]
fn egraph(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Graph>()?;
    m.add_class::<FM3>()?;

    Ok(())
}
