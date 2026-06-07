//! KernelSgd builder for creating SGD instances using diffusion kernel distances.

use crate::diffusion_kernel::DiffusionKernel;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_layout_sgd::Sgd;
use rand::{Rng, RngExt};
use std::collections::HashSet;

/// KernelSgd builder for creating SGD instances from diffusion kernel distances.
///
/// This structure uses the diffusion kernel exp(-tL) to compute ideal distances
/// between nodes, where L is the graph Laplacian. The kernel is approximated using
/// Chebyshev polynomials and element queries are performed using the Hutchinson
/// trace estimator with symmetry optimization.
#[derive(Debug, Clone)]
pub struct KernelSgd<S> {
    /// Diffusion time parameter
    pub t: S,
    /// Number of Hutchinson random vectors (effective 2x due to symmetry)
    pub num_vectors: usize,
    /// Degree of Chebyshev polynomial approximation
    pub degree: usize,
    /// Number of random pairs per node
    pub k: usize,
    /// Minimum distance between node pairs
    pub min_dist: S,
}

impl<S> KernelSgd<S>
where
    S: DrawingValue,
{
    /// Creates a new KernelSgd with default values.
    ///
    /// Default values:
    /// - t: 1000.0 (diffusion time)
    /// - num_vectors: 50 (Hutchinson vectors, effective 100 with symmetry)
    /// - degree: 10 (Chebyshev polynomial degree)
    /// - k: 30 (random pairs per node)
    /// - min_dist: 1e-3 (minimum distance)
    pub fn new() -> Self {
        Self {
            t: S::from_f32(1000.0).unwrap(),
            num_vectors: 50,
            degree: 10,
            k: 30,
            min_dist: S::from_f32(1e-3).unwrap(),
        }
    }

    /// Sets the diffusion time parameter.
    pub fn t(&mut self, t: S) -> &mut Self {
        self.t = t;
        self
    }

    /// Sets the number of Hutchinson random vectors.
    ///
    /// Note: Due to symmetry optimization, effective sample count is 2x this value.
    pub fn num_vectors(&mut self, num_vectors: usize) -> &mut Self {
        self.num_vectors = num_vectors;
        self
    }

    /// Sets the degree of Chebyshev polynomial approximation.
    pub fn degree(&mut self, degree: usize) -> &mut Self {
        self.degree = degree;
        self
    }

    /// Sets the number of random pairs per node.
    pub fn k(&mut self, k: usize) -> &mut Self {
        self.k = k;
        self
    }

    /// Sets the minimum distance between node pairs.
    pub fn min_dist(&mut self, min_dist: S) -> &mut Self {
        self.min_dist = min_dist;
        self
    }

    /// Builds an SGD instance with automatic lambda_max estimation.
    ///
    /// Uses power method to estimate the maximum eigenvalue of the Laplacian.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// A new SGD instance configured with kernel-based distances
    pub fn build<G, F, R>(&self, graph: G, length: F, rng: &mut R) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        S: std::iter::Sum + Default,
    {
        // Create DiffusionKernel with automatic lambda_max estimation
        let kernel =
            DiffusionKernel::new(graph, length, self.t, self.degree, self.num_vectors, rng);

        // Generate node pairs with kernel-based distances
        let node_pairs = generate_node_pairs(graph, &kernel, self.min_dist, self.k, rng);

        Sgd::new(node_pairs)
    }

    /// Builds an SGD instance with externally provided lambda_max.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `lambda_max` - Maximum eigenvalue of the Laplacian
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// A new SGD instance configured with kernel-based distances
    pub fn build_with_lambda_max<G, F, R>(
        &self,
        graph: G,
        length: F,
        lambda_max: S,
        rng: &mut R,
    ) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        S: std::iter::Sum + Default + ndarray::ScalarOperand,
    {
        // Create DiffusionKernel with provided lambda_max
        let kernel = DiffusionKernel::new_with_lambda_max(
            graph,
            length,
            self.t,
            self.degree,
            lambda_max,
            self.num_vectors,
            rng,
        );

        // Generate node pairs with kernel-based distances
        let node_pairs = generate_node_pairs(graph, &kernel, self.min_dist, self.k, rng);

        Sgd::new(node_pairs)
    }
}

impl<S> Default for KernelSgd<S>
where
    S: DrawingValue,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Generates node pairs with distances computed from the diffusion kernel.
fn generate_node_pairs<G, S, R>(
    graph: G,
    kernel: &DiffusionKernel<S>,
    min_dist: S,
    k: usize,
    rng: &mut R,
) -> Vec<(usize, usize, S, S, S, S)>
where
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
    G::NodeId: DrawingIndex,
    S: DrawingValue + std::iter::Sum,
    R: Rng,
{
    use std::collections::HashMap;

    let n = graph.node_count();

    // Create node index mapping
    let node_indices: HashMap<G::NodeId, usize> = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, node_id)| (node_id, i))
        .collect();

    let mut node_pairs = Vec::new();
    let mut used_pairs = HashSet::new();

    // Step 1: Add edge-based node pairs with kernel distances
    for edge in graph.edge_references() {
        let i = node_indices[&edge.source()];
        let j = node_indices[&edge.target()];

        if i != j {
            let pair_key = if i < j { (i, j) } else { (j, i) };

            if !used_pairs.contains(&pair_key) {
                used_pairs.insert(pair_key);
                // Compute distance from kernel matrix elements
                let k_ii = kernel.get(i, i);
                let k_jj = kernel.get(j, j);
                let k_ij = kernel.get(i, j);
                let distance = (k_ii + k_jj - S::from_f32(2.0).unwrap() * k_ij)
                    .sqrt()
                    .max(min_dist);
                let weight = S::one() / (distance * distance);
                node_pairs.push((i, j, distance, distance, weight, weight));
            }
        }
    }

    // Step 2: Add random node pairs with kernel distances
    for i in 0..n {
        for _ in 0..k {
            let j = rng.random_range(0..n);
            if i != j {
                let pair_key = if i < j { (i, j) } else { (j, i) };

                if !used_pairs.contains(&pair_key) {
                    used_pairs.insert(pair_key);
                    // Compute distance from kernel matrix elements
                    let k_ii = kernel.get(i, i);
                    let k_jj = kernel.get(j, j);
                    let k_ij = kernel.get(i, j);
                    let distance = (k_ii + k_jj - S::from_f32(2.0).unwrap() * k_ij)
                        .sqrt()
                        .max(min_dist);
                    let weight = S::one() / (distance * distance);
                    node_pairs.push((i, j, distance, distance, weight, weight));
                }
            }
        }
    }

    node_pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_kernel_sgd_new() {
        let kernel_sgd: KernelSgd<f32> = KernelSgd::new();
        assert_eq!(kernel_sgd.t, 1000.0);
        assert_eq!(kernel_sgd.num_vectors, 50);
        assert_eq!(kernel_sgd.degree, 10);
        assert_eq!(kernel_sgd.k, 30);
        assert_eq!(kernel_sgd.min_dist, 1e-3);
    }

    #[test]
    fn test_kernel_sgd_builder() {
        let mut kernel_sgd: KernelSgd<f32> = KernelSgd::new();
        kernel_sgd
            .t(500.0)
            .num_vectors(100)
            .degree(20)
            .k(50)
            .min_dist(1e-2);

        assert_eq!(kernel_sgd.t, 500.0);
        assert_eq!(kernel_sgd.num_vectors, 100);
        assert_eq!(kernel_sgd.degree, 20);
        assert_eq!(kernel_sgd.k, 50);
        assert_eq!(kernel_sgd.min_dist, 1e-2);
    }

    #[test]
    fn test_kernel_sgd_build() {
        let mut graph = Graph::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n0, n1, ());
        graph.add_edge(n1, n2, ());

        let mut rng = StdRng::seed_from_u64(42);
        let kernel_sgd: KernelSgd<f32> = KernelSgd::new();

        let sgd = kernel_sgd.build(&graph, |_| 1.0, &mut rng);

        // Should have node pairs from edges + some random pairs
        // Exact count depends on random sampling, but should be > 2 (edges)
        assert!(sgd.node_pairs().len() >= 2);
    }
}
