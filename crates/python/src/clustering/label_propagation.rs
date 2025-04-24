use crate::graph::{GraphType, PyGraphAdapter};
use petgraph_clustering::{CommunityDetection, LabelPropagation};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Label Propagation community detection algorithm.
///
/// Label Propagation is a fast, near-linear time algorithm for detecting communities
/// in networks. It works by iteratively updating each node's label to the most common
/// label among its neighbors until convergence.
///
/// Parameters:
///     max_iterations (int, optional): Maximum number of iterations to perform. Default is 100.
///     seed (int, optional): Seed for random number generation. Default is None.
///
/// Example:
///     >>> import egraph as eg
///     >>> graph = eg.Graph()
///     >>> # Add nodes and edges...
///     >>> lp = eg.clustering.LabelPropagation(max_iterations=100, seed=42)
///     >>> communities = lp.detect_communities(graph)
///     >>> # communities is a dict mapping node indices to community IDs
#[pyclass(name = "LabelPropagation")]
pub struct PyLabelPropagation {
    instance: LabelPropagation,
}

#[pymethods]
impl PyLabelPropagation {
    #[new]
    #[pyo3(signature = ())]
    fn new() -> Self {
        PyLabelPropagation {
            instance: LabelPropagation::new(),
        }
    }

    /// Detect communities in the given graph.
    ///
    /// Returns:
    ///     dict: A dictionary mapping node indices to community IDs.
    #[pyo3(signature = (graph))]
    fn detect_communities(&self, graph: &PyGraphAdapter) -> HashMap<usize, usize> {
        let map = match graph.graph() {
            GraphType::Graph(graph) => self.instance.detect_communities(graph),
            GraphType::DiGraph(graph) => self.instance.detect_communities(graph),
        };
        map.into_iter()
            .map(|(u, c)| (u.index() as usize, c))
            .collect::<HashMap<_, _>>()
    }
}
