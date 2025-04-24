use crate::graph::{GraphType, PyGraphAdapter};
use petgraph_algorithm_layering::LongestPath;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Longest Path layering algorithm.
///
/// This algorithm assigns layers to nodes in a directed graph by determining the
/// longest path from any source node (node with no incoming edges) to each node.
/// It is commonly used in hierarchical graph layouts.
///
/// Example:
///     >>> import egraph as eg
///     >>> graph = eg.DiGraph()
///     >>> # Add nodes and edges...
///     >>> lp = eg.algorithm.LongestPath()
///     >>> layers = lp.assign_layers(graph)
///     >>> # layers is a dict mapping node indices to layer numbers
#[pyclass(name = "LongestPath")]
pub struct PyLongestPath {
    instance: LongestPath,
}

#[pymethods]
impl PyLongestPath {
    #[new]
    fn new() -> Self {
        PyLongestPath {
            instance: LongestPath::new(),
        }
    }

    /// Assign layers to nodes in the given directed graph.
    ///
    /// The algorithm assigns layer 0 to source nodes (those with no incoming edges),
    /// and then assigns other nodes to layers based on the longest path from any source.
    ///
    /// Args:
    ///     graph: A directed graph.
    ///
    /// Returns:
    ///     dict: A dictionary mapping node indices to layer numbers (starting from 0).
    ///
    /// Raises:
    ///     ValueError: If the graph contains cycles. Use remove_cycle() first to make the graph acyclic.
    #[pyo3(signature = (graph))]
    fn assign_layers(&self, graph: &PyGraphAdapter) -> PyResult<HashMap<usize, usize>> {
        match graph.graph() {
            GraphType::Graph(_) => Err(pyo3::exceptions::PyValueError::new_err(
                "LongestPath only works with directed graphs",
            )),
            GraphType::DiGraph(graph) => {
                // Try to assign layers, which may fail if the graph has cycles
                let layers = self.instance.assign_layers(graph);

                // Convert from NodeIndex keys to usize keys for Python
                let result = layers
                    .into_iter()
                    .map(|(node_idx, layer)| (node_idx.index(), layer))
                    .collect();

                Ok(result)
            }
        }
    }
}
