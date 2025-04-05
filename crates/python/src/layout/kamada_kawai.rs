/// Kamada-Kawai layout algorithm implementation for Python
///
/// This module provides a Python binding for the Kamada-Kawai force-directed
/// graph layout algorithm. The algorithm models a graph as a spring system where
/// spring lengths are based on shortest path distances, and iteratively positions
/// nodes to minimize the energy of this system.
///
/// The implementation allows both running the entire algorithm at once, or
/// stepping through it node by node for more fine-grained control.
use crate::{
    drawing::PyDrawingEuclidean2d,
    graph::{GraphType, PyGraphAdapter},
};
use petgraph::visit::EdgeRef;
use petgraph_layout_kamada_kawai::KamadaKawai;
use pyo3::prelude::*;

/// Python class for the Kamada-Kawai layout algorithm
///
/// This class implements the Kamada-Kawai force-directed graph layout algorithm.
/// It models a graph as a system of springs where spring lengths are proportional
/// to the shortest path distances between nodes. The algorithm iteratively adjusts
/// node positions to minimize the energy of this spring system.
///
/// The algorithm selects nodes based on their energy gradient and optimizes
/// their positions one at a time, continuing until the maximum energy gradient
/// falls below a specified threshold (eps).
///
/// Reference: Kamada, T., & Kawai, S. (1989). An algorithm for drawing general
/// undirected graphs. Information Processing Letters, 31(1), 7-15.
#[pyclass]
#[pyo3(name = "KamadaKawai")]
struct PyKamadaKawai {
    kamada_kawai: KamadaKawai<f32>,
}

#[pymethods]
impl PyKamadaKawai {
    /// Creates a new Kamada-Kawai layout algorithm instance
    ///
    /// This constructor initializes a Kamada-Kawai layout algorithm using a graph
    /// and a function that provides edge weights.
    ///
    /// # Parameters
    /// * `graph` - The graph to layout
    /// * `f` - A Python function that takes an edge index and returns its weight
    ///
    /// # Returns
    /// A new KamadaKawai instance
    #[new]
    fn new(graph: &PyGraphAdapter, f: &Bound<PyAny>) -> PyKamadaKawai {
        PyKamadaKawai {
            kamada_kawai: match graph.graph() {
                GraphType::Graph(native_graph) => KamadaKawai::new(native_graph, |e| {
                    f.call1((e.id().index(),)).unwrap().extract().unwrap()
                }),
                _ => panic!("unsupported graph type"),
            },
        }
    }

    /// Selects the node with the highest energy gradient magnitude
    ///
    /// This method identifies the node that should be moved next in the algorithm
    /// by finding the one with the largest energy gradient. The node with the largest
    /// gradient is the one that, when moved, will reduce the overall energy the most.
    ///
    /// # Parameters
    /// * `drawing` - The current drawing of the graph
    ///
    /// # Returns
    /// The index of the selected node if any node has a gradient larger than eps,
    /// None otherwise
    fn select_node(&self, drawing: &PyDrawingEuclidean2d) -> Option<usize> {
        self.kamada_kawai.select_node(drawing.drawing())
    }

    /// Optimizes the position of a single node
    ///
    /// This method moves a specified node to reduce the energy of the layout,
    /// while keeping all other nodes fixed.
    ///
    /// # Parameters
    /// * `m` - The index of the node to optimize
    /// * `drawing` - The drawing to modify
    fn apply_to_node(&self, m: usize, drawing: &mut PyDrawingEuclidean2d) {
        self.kamada_kawai.apply_to_node(m, drawing.drawing_mut())
    }

    /// Runs the complete Kamada-Kawai algorithm
    ///
    /// This method repeatedly selects the node with the largest energy gradient
    /// and optimizes its position until all nodes have gradients smaller than eps.
    ///
    /// # Parameters
    /// * `drawing` - The drawing to optimize
    fn run(&self, drawing: &mut PyDrawingEuclidean2d) {
        self.kamada_kawai.run(drawing.drawing_mut())
    }

    /// Gets the convergence threshold for the algorithm
    ///
    /// The eps parameter determines when the algorithm stops. When all nodes
    /// have energy gradients smaller than eps, the layout is considered converged.
    ///
    /// # Returns
    /// The current convergence threshold
    #[getter]
    fn eps(&self) -> f32 {
        self.kamada_kawai.eps
    }

    /// Sets the convergence threshold for the algorithm
    ///
    /// # Parameters
    /// * `value` - The new convergence threshold
    #[setter]
    fn set_eps(&mut self, value: f32) {
        self.kamada_kawai.eps = value;
    }
}

pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyKamadaKawai>()?;
    Ok(())
}
