//! Omega implementation of the SGD trait for graph layout using spectral coordinates.

use crate::eigenvalue::EigenSolver;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_layout_sgd::Sgd;
use rand::Rng;
use std::collections::{HashMap, HashSet};

/// Omega Stochastic Gradient Descent implementation for graph layout.
///
/// This implementation differs from FullSgd and SparseSgd in how it constructs node pairs.
/// It uses spectral analysis of the graph Laplacian to create d-dimensional coordinates
/// for nodes, then uses these coordinates to compute distances for both edge-based and
/// random node pairs.
///
/// The algorithm follows these steps:
/// 1. Compute graph Laplacian eigenvalues and eigenvectors
/// 2. Generate d-dimensional coordinates from eigenvectors
/// 3. Add edge-based node pairs with Euclidean distances
/// 4. Add random node pairs with Euclidean distances (avoiding duplicates)
pub struct Omega<S> {
    /// List of node pairs to be considered during layout optimization.
    /// Each tuple contains (i, j, distance_ij, distance_ji, weight_ij, weight_ji)
    node_pairs: Vec<(usize, usize, S, S, S, S)>,
}

impl<S> Omega<S>
where
    S: DrawingValue,
{
    /// Creates a new Omega instance from a graph using spectral coordinates.
    ///
    /// This constructor implements the 4-step Omega algorithm:
    /// 1. Computes the smallest d non-zero eigenvalues and eigenvectors of the graph Laplacian
    /// 2. Creates d-dimensional coordinates by dividing eigenvectors by sqrt of eigenvalues
    /// 3. Adds edge-based node pairs using Euclidean distances from coordinates
    /// 4. Adds k random node pairs per node using Euclidean distances (avoiding duplicates)
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths (currently unused but kept for API consistency)
    /// * `d` - The number of dimensions for spectral coordinates
    /// * `k` - The number of random node pairs to add per node
    /// * `rng` - Random number generator for selecting random node pairs
    ///
    /// # Returns
    /// A new Omega instance configured with appropriate node pairs
    ///
    /// # Computational Complexity
    /// - Step 1: O(d(|V| + |E|)) - Eigenvalue computation
    /// - Step 2: O(d|V|) - Coordinate generation
    /// - Step 3: O(|E|) - Edge-based pairs
    /// - Step 4: O(k|V|) - Random pairs
    pub fn new<G, F, R>(graph: G, mut _length: F, d: usize, k: usize, rng: &mut R) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        let n = graph.node_count();

        // Create node index mapping
        let node_indices: HashMap<G::NodeId, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id, i))
            .collect();

        // Step 1 & 2: Compute spectral coordinates
        let coordinates = Self::compute_spectral_coordinates(&graph, d);

        let mut node_pairs = Vec::new();
        let mut used_pairs = HashSet::new();

        // Step 3: Add edge-based node pairs with Euclidean distances
        for edge in graph.edge_references() {
            let i = node_indices[&edge.source()];
            let j = node_indices[&edge.target()];
            let pair_key = if i < j { (i, j) } else { (j, i) };

            if !used_pairs.contains(&pair_key) {
                used_pairs.insert(pair_key);
                let distance = Self::euclidean_distance(&coordinates[i], &coordinates[j]);
                let weight = S::one() / (distance * distance);
                node_pairs.push((i, j, distance, distance, weight, weight));
            }
        }

        // Step 4: Add random node pairs with Euclidean distances (avoiding duplicates)
        for i in 0..n {
            for _ in 0..k {
                let j = rng.gen_range(0..n);
                if i != j {
                    let pair_key = if i < j { (i, j) } else { (j, i) };

                    if !used_pairs.contains(&pair_key) {
                        used_pairs.insert(pair_key);
                        let distance = Self::euclidean_distance(&coordinates[i], &coordinates[j]);
                        let weight = S::one() / (distance * distance);
                        node_pairs.push((i, j, distance, distance, weight, weight));
                    }
                    // Skip if duplicate - no re-sampling
                }
            }
        }

        Omega { node_pairs }
    }

    /// Computes d-dimensional spectral coordinates for all nodes in the graph.
    ///
    /// Uses the eigenvalue solver to compute the smallest d non-zero eigenvalues
    /// and eigenvectors of the graph Laplacian, then creates coordinates by
    /// dividing each eigenvector by the square root of its corresponding eigenvalue.
    ///
    /// # Parameters
    /// * `graph` - The input graph
    /// * `d` - Number of dimensions for the coordinates
    ///
    /// # Returns
    /// A vector where coordinates[i] contains the d-dimensional coordinate for node i
    fn compute_spectral_coordinates<G>(graph: &G, d: usize) -> Vec<Vec<S>>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
        G::NodeId: DrawingIndex,
    {
        let n = graph.node_count();
        let solver = EigenSolver::<S>::default();

        // Step 1: Compute smallest d non-zero eigenvalues and eigenvectors
        let (eigenvalues, eigenvectors) = solver.compute_smallest_eigenvalues(graph, d);

        // Step 2: Create coordinates by dividing eigenvectors by sqrt of eigenvalues
        let mut coordinates = vec![vec![S::zero(); d]; n];

        for (dim, (eigenvalue, eigenvector)) in
            eigenvalues.iter().zip(eigenvectors.iter()).enumerate()
        {
            let sqrt_eigenvalue = eigenvalue.sqrt();
            for node in 0..n {
                coordinates[node][dim] = eigenvector[node] / sqrt_eigenvalue;
            }
        }

        coordinates
    }

    /// Computes the Euclidean distance between two d-dimensional coordinates.
    ///
    /// # Parameters
    /// * `coord1` - First coordinate vector
    /// * `coord2` - Second coordinate vector
    ///
    /// # Returns
    /// The Euclidean distance between the two coordinates
    fn euclidean_distance(coord1: &[S], coord2: &[S]) -> S {
        coord1
            .iter()
            .zip(coord2.iter())
            .map(|(&x1, &x2)| {
                let diff = x1 - x2;
                diff * diff
            })
            .fold(S::zero(), |acc, x| acc + x)
            .sqrt()
    }
}

/// Implementation of the Sgd trait for Omega
///
/// This provides the core SGD functionality for the Omega graph layout algorithm,
/// allowing it to work with the common SGD framework.
impl<S> Sgd<S> for Omega<S> {
    /// Returns a reference to the node pairs data structure.
    ///
    /// This implementation uses a combination of edge-based and random node pairs,
    /// all computed using spectral coordinates derived from the graph Laplacian.
    fn node_pairs(&self) -> &Vec<(usize, usize, S, S, S, S)> {
        &self.node_pairs
    }

    /// Returns a mutable reference to the node pairs data structure.
    ///
    /// This allows the algorithm to modify the node pairs during execution,
    /// such as updating distances or weights based on current layout.
    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, S, S, S, S)> {
        &mut self.node_pairs
    }
}
