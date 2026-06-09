//! Python bindings for RandomPairSparseSgd
//!
//! This module provides Python access to RandomPairSparseSgd which builds
//! SGD instances using edge pairs and random node pairs from a Distance metric.

use crate::{
    graph::{GraphType, PyGraphAdapter},
    layout::sgd::PySgd,
    rng::PyRng,
};
use petgraph_layout_sgd::RandomPairSparseSgd;
use pyo3::prelude::*;

/// Python class for configuring the RandomPairSparseSgd algorithm
///
/// RandomPairSparseSgd generates node pairs for SGD optimization from a distance matrix,
/// including all edge-based pairs and a number of random pairs per node.
///
/// :param k: Number of random pairs per node (default: 30)
/// :type k: int
#[pyclass]
#[pyo3(name = "RandomPairSparseSgd")]
pub struct PyRandomPairSparseSgd {
    builder: RandomPairSparseSgd,
}

#[pymethods]
impl PyRandomPairSparseSgd {
    /// Creates a new RandomPairSparseSgd with default parameters
    ///
    /// :return: A new RandomPairSparseSgd instance
    /// :rtype: RandomPairSparseSgd
    #[new]
    fn new() -> Self {
        PyRandomPairSparseSgd {
            builder: RandomPairSparseSgd::new(),
        }
    }

    /// Sets the number of random pairs per node
    ///
    /// :param k: Number of random pairs per node
    /// :type k: int
    /// :return: Self for method chaining
    /// :rtype: RandomPairSparseSgd
    fn k(mut slf: PyRefMut<Self>, k: usize) -> Py<Self> {
        slf.builder.k(k);
        slf.into()
    }

    /// Builds an Sgd instance from a distance matrix
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param distance_matrix: The distance matrix to query distances from (e.g. DistanceMatrix, DiffusionDistanceMatrix, etc.)
    /// :type distance_matrix: DistanceMatrix or DiffusionDistanceMatrix or EmbeddingDistanceMatrix or KernelDistance
    /// :param rng: Random number generator for selecting random node pairs
    /// :type rng: Rng
    /// :return: A new Sgd instance configured with node pairs from the distance matrix
    /// :rtype: Sgd
    /// :raises: ValueError if the graph type is not supported
    fn build(
        &self,
        graph: &PyGraphAdapter,
        distance_matrix: &Bound<PyAny>,
        rng: &mut PyRng,
    ) -> PyResult<PySgd> {
        match graph.graph() {
            GraphType::Graph(native_graph) => {
                let sgd = crate::distance_matrix::with_distance(distance_matrix, |distance| {
                    self.builder.build(native_graph, distance, rng.get_mut())
                })?;
                Ok(PySgd::new_with_sgd(sgd))
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "unsupported graph type",
            )),
        }
    }
}
