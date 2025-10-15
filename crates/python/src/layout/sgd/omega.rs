//! Omega SGD layout algorithm
//!
//! This module provides Python bindings for the Omega algorithm,
//! which generates node pairs for SGD from precomputed spectral embeddings.

use crate::{
    array::PyArray2,
    graph::{GraphType, PyGraphAdapter},
    layout::sgd::PySgd,
    FloatType,
};
use petgraph_layout_omega::Omega;
use pyo3::prelude::*;

/// Python class for configuring the Omega algorithm
///
/// Omega generates node pairs for SGD optimization from precomputed spectral embeddings.
/// It does not compute embeddings itself - use RdMds for that purpose.
///
/// The algorithm generates node pairs from both graph edges and random sampling,
/// using distances computed from the provided spectral embedding coordinates.
///
/// :param k: Number of random pairs per node (default: 30)
/// :type k: int
/// :param min_dist: Minimum distance between node pairs (default: 1e-3)
/// :type min_dist: float
#[pyclass]
#[pyo3(name = "Omega")]
pub struct PyOmega {
    omega: Omega<FloatType>,
}

#[pymethods]
impl PyOmega {
    /// Creates a new Omega with default parameters
    ///
    /// Default values:
    /// - k: 30 (random pairs per node)
    /// - min_dist: 1e-3 (minimum distance)
    ///
    /// :return: A new Omega instance
    /// :rtype: Omega
    #[new]
    fn new() -> Self {
        PyOmega {
            omega: Omega::new(),
        }
    }

    /// Sets the number of random pairs per node
    ///
    /// :param k: Number of random pairs per node
    /// :type k: int
    /// :return: Self for method chaining
    /// :rtype: Omega
    fn k(mut slf: PyRefMut<Self>, k: usize) -> Py<Self> {
        slf.omega.k(k);
        slf.into()
    }

    /// Sets the minimum distance between node pairs
    ///
    /// :param min_dist: Minimum distance between node pairs
    /// :type min_dist: float
    /// :return: Self for method chaining
    /// :rtype: Omega
    fn min_dist(mut slf: PyRefMut<Self>, min_dist: FloatType) -> Py<Self> {
        slf.omega.min_dist(min_dist);
        slf.into()
    }

    /// Builds an Sgd instance from precomputed embedding
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param embedding: Precomputed spectral coordinates as a 2D Array2 (from RdMds)
    /// :type embedding: Array2
    /// :param rng: Random number generator for selecting random node pairs
    /// :type rng: Rng
    /// :return: A new Sgd instance configured with node pairs from the embedding
    /// :rtype: Sgd
    /// :raises: ValueError if the graph type is not supported
    fn build(
        &self,
        graph: &PyGraphAdapter,
        embedding: &PyArray2,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<PySgd> {
        // Get the ndarray directly from PyArray2
        let coordinates = embedding.as_array();

        let sgd = match graph.graph() {
            GraphType::Graph(native_graph) => {
                self.omega.build(native_graph, coordinates, rng.get_mut())
            }
            _ => panic!("unsupported graph type"),
        };

        Ok(PySgd::new_with_sgd(sgd))
    }
}
