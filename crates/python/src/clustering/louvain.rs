use crate::graph::{GraphType, PyGraphAdapter};
use petgraph_clustering::{CommunityDetection, Louvain};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Louvain community detection algorithm.
///
/// The Louvain method is a heuristic algorithm for detecting communities in networks.
/// It works by optimizing modularity in a greedy manner, where modularity measures
/// the density of connections within communities compared to connections between them.
///
/// Parameters:
///     resolution (float, optional): Resolution parameter that affects the size of communities.
///         Higher values lead to smaller communities. Default is 1.0.
///     seed (int, optional): Seed for random number generation. Default is None.
///
/// Example:
///     >>> import egraph as eg
///     >>> graph = eg.Graph()
///     >>> # Add nodes and edges...
///     >>> louvain = eg.clustering.Louvain(resolution=1.0, seed=42)
///     >>> communities = louvain.detect_communities(graph)
///     >>> # communities is a dict mapping node indices to community IDs
#[pyclass(name = "Louvain")]
pub struct PyLouvain {
    instance: Louvain,
}

#[pymethods]
impl PyLouvain {
    #[new]
    #[pyo3(signature = ())]
    fn new() -> Self {
        PyLouvain {
            instance: Louvain::new(),
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
            .map(|(u, c)| (u.index(), c))
            .collect::<HashMap<_, _>>()
    }
}
