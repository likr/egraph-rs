//! DiffusionKernel provides random access to exp(-tL) matrix elements.
//!
//! This module provides a clean interface for querying individual elements of the
//! diffusion kernel matrix K = exp(-tL), where L is the graph Laplacian. The kernel
//! is approximated using Chebyshev polynomials and element queries are performed
//! using the Hutchinson trace estimator with symmetry optimization.

use crate::chebyshev::chebyshev_approximation;
use crate::hutchinson::{generate_rademacher_vectors, HutchinsonEstimator};
use crate::power_method::estimate_lambda_max;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_linalg_spmv::SparseSymmetricMatrix;
use rand::Rng;
use std::collections::HashMap;

/// DiffusionKernel provides random access to exp(-tL) matrix elements.
///
/// This structure encapsulates the diffusion kernel computation using Chebyshev
/// polynomial approximation and Hutchinson trace estimation. Once constructed,
/// it allows efficient queries of individual kernel matrix elements K[i,j].
///
/// # Example
///
/// ```
/// use petgraph::Graph;
/// use petgraph_layout_kernel_sgd::DiffusionKernel;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let mut graph = Graph::new_undirected();
/// let n0 = graph.add_node(());
/// let n1 = graph.add_node(());
/// graph.add_edge(n0, n1, ());
///
/// let mut rng = StdRng::seed_from_u64(42);
///
/// // Create kernel with automatic lambda_max estimation
/// let dk = DiffusionKernel::new(
///     &graph,
///     |_| 1.0f32,    // edge length function
///     1000.0,        // diffusion time t
///     10,            // Chebyshev degree
///     50,            // number of random vectors
///     &mut rng
/// );
///
/// // Query kernel elements
/// let k_00 = dk.get(0, 0);
/// let k_01 = dk.get(0, 1);
/// ```
#[derive(Debug)]
pub struct DiffusionKernel<S> {
    estimator: HutchinsonEstimator<S>,
}

impl<S> DiffusionKernel<S>
where
    S: DrawingValue,
{
    /// Creates a new DiffusionKernel with automatic lambda_max estimation.
    ///
    /// Uses the power method to estimate the maximum eigenvalue of the Laplacian
    /// matrix before computing the diffusion kernel.
    ///
    /// # Parameters
    /// * `graph` - The input graph
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `t` - Diffusion time parameter
    /// * `degree` - Degree of Chebyshev polynomial approximation
    /// * `num_vectors` - Number of Hutchinson random vectors (effective 2x due to symmetry)
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// A new DiffusionKernel instance
    pub fn new<G, F, R>(
        graph: G,
        mut length: F,
        t: S,
        degree: usize,
        num_vectors: usize,
        rng: &mut R,
    ) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        S: std::iter::Sum + Default,
    {
        // Build Laplacian matrix
        let laplacian = build_laplacian(graph, &mut length);

        // Estimate lambda_max using power method
        let lambda_max = estimate_lambda_max(&laplacian, rng, 100, S::from_f32(1e-6).unwrap());

        Self::new_with_lambda_max(graph, length, t, degree, lambda_max, num_vectors, rng)
    }

    /// Creates a new DiffusionKernel with externally provided lambda_max.
    ///
    /// Use this method when you have already computed the maximum eigenvalue of
    /// the Laplacian matrix, avoiding redundant computation.
    ///
    /// # Parameters
    /// * `graph` - The input graph
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `t` - Diffusion time parameter
    /// * `degree` - Degree of Chebyshev polynomial approximation
    /// * `lambda_max` - Maximum eigenvalue of the Laplacian
    /// * `num_vectors` - Number of Hutchinson random vectors (effective 2x due to symmetry)
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// A new DiffusionKernel instance
    pub fn new_with_lambda_max<G, F, R>(
        graph: G,
        mut length: F,
        t: S,
        degree: usize,
        lambda_max: S,
        num_vectors: usize,
        rng: &mut R,
    ) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        S: std::iter::Sum + Default + ndarray::ScalarOperand,
    {
        let n = graph.node_count();

        // Build Laplacian matrix
        let laplacian = build_laplacian(graph, &mut length);

        // Generate Rademacher random vectors
        let v = generate_rademacher_vectors(n, num_vectors, rng);

        // Compute K @ V using Chebyshev approximation
        let kv = chebyshev_approximation(&laplacian, t, degree, lambda_max, &v);

        // Create Hutchinson estimator
        let estimator = HutchinsonEstimator::new(v, kv);

        Self { estimator }
    }

    /// Queries the (i, j) element of the diffusion kernel matrix.
    ///
    /// Returns an approximation of exp(-tL)[i,j] using the Hutchinson estimator
    /// with symmetry optimization.
    ///
    /// # Parameters
    /// * `i` - Row index
    /// * `j` - Column index
    ///
    /// # Returns
    /// Approximated kernel value at (i, j)
    ///
    /// # Example
    ///
    /// ```
    /// # use petgraph::Graph;
    /// # use petgraph_layout_kernel_sgd::DiffusionKernel;
    /// # use rand::rngs::StdRng;
    /// # use rand::SeedableRng;
    /// # let mut graph = Graph::new_undirected();
    /// # let n0 = graph.add_node(());
    /// # let n1 = graph.add_node(());
    /// # graph.add_edge(n0, n1, ());
    /// # let mut rng = StdRng::seed_from_u64(42);
    /// # let dk = DiffusionKernel::new(&graph, |_| 1.0f32, 1000.0, 10, 50, &mut rng);
    /// // Get diagonal element (should be close to 1)
    /// let k_00 = dk.get(0, 0);
    ///
    /// // Get off-diagonal element
    /// let k_01 = dk.get(0, 1);
    ///
    /// // Symmetry: K[i,j] == K[j,i]
    /// let k_10 = dk.get(1, 0);
    /// assert!((k_01 - k_10).abs() < 1e-10);
    /// ```
    pub fn get(&self, i: usize, j: usize) -> S
    where
        S: std::iter::Sum,
    {
        self.estimator.query(i, j)
    }

    /// Returns the number of nodes in the graph.
    ///
    /// # Returns
    /// The dimension of the kernel matrix (number of nodes)
    pub fn n(&self) -> usize {
        self.estimator.n()
    }
}

/// Builds a graph Laplacian matrix from edge weights.
fn build_laplacian<G, F, S>(graph: G, length: &mut F) -> SparseSymmetricMatrix<S>
where
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
    G::NodeId: DrawingIndex,
    F: FnMut(G::EdgeRef) -> S,
    S: DrawingValue + Default,
{
    let n = graph.node_count();

    // Create node index mapping
    let node_indices: HashMap<G::NodeId, usize> = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, node_id)| (node_id, i))
        .collect();

    let mut matrix = SparseSymmetricMatrix::new(n);
    let mut degrees = vec![S::zero(); n];

    // Process edges
    for edge in graph.edge_references() {
        let i = node_indices[&edge.source()];
        let j = node_indices[&edge.target()];
        let weight = length(edge);

        if i != j {
            let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
            matrix.add_edge(min_idx, max_idx, -weight);
            degrees[i] += weight;
            degrees[j] += weight;
        }
    }

    // Set diagonal elements (degrees)
    for (i, &deg) in degrees.iter().enumerate().take(n) {
        matrix.set_diagonal(i, deg);
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_diffusion_kernel_new() {
        let mut graph = Graph::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n0, n1, ());
        graph.add_edge(n1, n2, ());

        let mut rng = StdRng::seed_from_u64(42);

        let dk = DiffusionKernel::new(&graph, |_| 1.0f32, 1000.0, 10, 50, &mut rng);

        assert_eq!(dk.n(), 3);

        // Diagonal elements should be positive
        // (Note: with large t and Hutchinson estimation, values may vary significantly)
        for i in 0..3 {
            let k_ii = dk.get(i, i);
            assert!(k_ii > 0.0, "K[{},{}] = {} should be positive", i, i, k_ii);
        }

        // Off-diagonal elements should be positive (for connected components)
        let k_01 = dk.get(0, 1);
        let k_12 = dk.get(1, 2);
        assert!(k_01 > 0.0, "K[0,1] should be positive");
        assert!(k_12 > 0.0, "K[1,2] should be positive");
    }

    #[test]
    fn test_diffusion_kernel_new_with_lambda_max() {
        let mut graph = Graph::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        graph.add_edge(n0, n1, ());

        let mut rng = StdRng::seed_from_u64(42);

        let dk =
            DiffusionKernel::new_with_lambda_max(&graph, |_| 1.0f64, 1000.0, 10, 2.0, 50, &mut rng);

        assert_eq!(dk.n(), 2);

        // Test symmetry
        let k_01 = dk.get(0, 1);
        let k_10 = dk.get(1, 0);
        assert!(
            (k_01 - k_10).abs() < 1e-10,
            "K[0,1]={} != K[1,0]={}",
            k_01,
            k_10
        );
    }

    #[test]
    fn test_diffusion_kernel_symmetry() {
        let mut graph = Graph::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n0, n1, ());
        graph.add_edge(n1, n2, ());
        graph.add_edge(n2, n0, ());

        let mut rng = StdRng::seed_from_u64(42);

        let dk = DiffusionKernel::new(&graph, |_| 1.0f32, 1000.0, 10, 100, &mut rng);

        // Test symmetry for all pairs
        for i in 0..3 {
            for j in 0..3 {
                let k_ij = dk.get(i, j);
                let k_ji = dk.get(j, i);
                assert!(
                    (k_ij - k_ji).abs() < 1e-6,
                    "K[{},{}]={} != K[{},{}]={}",
                    i,
                    j,
                    k_ij,
                    j,
                    i,
                    k_ji
                );
            }
        }
    }

    #[test]
    fn test_build_laplacian() {
        let mut graph = Graph::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n0, n1, ());
        graph.add_edge(n1, n2, ());

        let laplacian: SparseSymmetricMatrix<f64> = build_laplacian(&graph, &mut |_| 1.0);

        assert_eq!(laplacian.dim(), 3);
        // Diagonal should be degrees: [1, 2, 1]
        assert_eq!(laplacian.diagonal()[0], 1.0);
        assert_eq!(laplacian.diagonal()[1], 2.0);
        assert_eq!(laplacian.diagonal()[2], 1.0);
        // Should have 2 edges
        assert_eq!(laplacian.edges().len(), 2);
    }
}
