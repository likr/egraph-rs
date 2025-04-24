use crate::graph::{GraphType, PyGraphAdapter};
use petgraph_clustering::{CommunityDetection, Spectral};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Spectral Clustering community detection algorithm.
///
/// Spectral Clustering uses the eigenvectors of the graph Laplacian matrix to
/// partition the graph into communities.
///
/// Parameters:
///     n_clusters (int, optional): Number of clusters to identify. Default is 2.
///     seed (int, optional): Seed for random number generation. Default is None.
///
/// Example:
///     >>> import egraph as eg
///     >>> graph = eg.Graph()
///     >>> # Add nodes and edges...
///     >>> spectral = eg.clustering.SpectralClustering(n_clusters=5, seed=42)
///     >>> communities = spectral.detect_communities(graph)
///     >>> # communities is a dict mapping node indices to community IDs
#[pyclass(name = "SpectralClustering")]
pub struct PySpectralClustering {
    instance: Spectral,
}

#[pymethods]
impl PySpectralClustering {
    #[new]
    #[pyo3(signature = (k))]
    fn new(k: usize) -> Self {
        PySpectralClustering {
            instance: Spectral::new(k),
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
