use crate::graph::{GraphType, PyGraphAdapter};
use petgraph_clustering::{CommunityDetection, InfoMap};
use pyo3::prelude::*;
use std::collections::HashMap;

/// InfoMap community detection algorithm.
///
/// InfoMap is an information-theoretic approach to community detection that
/// minimizes the expected description length of a random walk on the graph.
///
/// Parameters:
///     num_trials (int, optional): Number of trials to perform. Default is 10.
///     seed (int, optional): Seed for random number generation. Default is None.
///
/// Example:
///     >>> import egraph as eg
///     >>> graph = eg.Graph()
///     >>> # Add nodes and edges...
///     >>> infomap = eg.clustering.InfoMap(num_trials=10, seed=42)
///     >>> communities = infomap.detect_communities(graph)
///     >>> # communities is a dict mapping node indices to community IDs
#[pyclass(name = "InfoMap")]
pub struct PyInfoMap {
    instance: InfoMap,
}

#[pymethods]
impl PyInfoMap {
    #[new]
    #[pyo3(signature = ())]
    fn new() -> Self {
        PyInfoMap {
            instance: InfoMap::new(),
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
