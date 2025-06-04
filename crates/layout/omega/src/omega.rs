//! Omega implementation of the SGD trait for graph layout using spectral coordinates.

use crate::eigenvalue::{EigenSolver, LaplacianStructure};
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_layout_sgd::Sgd;
use rand::Rng;
use std::collections::{HashMap, HashSet};

/// Configuration options for the Omega algorithm using Builder pattern.
///
/// This structure contains all parameters needed to configure the Omega algorithm,
/// including spectral dimensions, random pairs, distance constraints, and
/// eigenvalue solver parameters.
#[derive(Debug, Clone)]
pub struct OmegaOption<S> {
    /// Number of spectral dimensions
    pub d: usize,
    /// Number of random pairs per node  
    pub k: usize,
    /// Minimum distance between node pairs
    pub min_dist: S,
    /// Maximum number of iterations for inverse power method
    pub max_iterations: usize,
    /// Maximum number of iterations for CG method
    pub cg_max_iterations: usize,
    /// Convergence tolerance for eigenvalue computation
    pub tolerance: S,
    /// Convergence tolerance for CG method
    pub cg_tolerance: S,
    /// Convergence tolerance for eigenvector changes
    pub vector_tolerance: S,
}

impl<S> OmegaOption<S>
where
    S: DrawingValue,
{
    /// Creates a new OmegaOption with default values.
    ///
    /// Default values:
    /// - d: 2 (spectral dimensions)
    /// - k: 30 (random pairs per node)
    /// - min_dist: 1e-3 (minimum distance)
    /// - max_iterations: 1000 (eigenvalue solver)
    /// - cg_max_iterations: 100 (CG solver)
    /// - tolerance: 1e-4 (eigenvalue convergence)
    /// - cg_tolerance: 1e-4 (CG convergence)
    /// - vector_tolerance: 1e-4 (eigenvector convergence)
    pub fn new() -> Self {
        Self {
            d: 2,
            k: 30,
            min_dist: S::from_f32(1e-3).unwrap(),
            max_iterations: 1000,
            cg_max_iterations: 100,
            tolerance: S::from_f32(1e-4).unwrap(),
            cg_tolerance: S::from_f32(1e-4).unwrap(),
            vector_tolerance: S::from_f32(1e-4).unwrap(),
        }
    }

    /// Sets the number of spectral dimensions.
    pub fn d(mut self, d: usize) -> Self {
        self.d = d;
        self
    }

    /// Sets the number of random pairs per node.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the minimum distance between node pairs.
    pub fn min_dist(mut self, min_dist: S) -> Self {
        self.min_dist = min_dist;
        self
    }

    /// Sets maximum iterations for inverse power method.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets maximum iterations for CG method.
    pub fn cg_max_iterations(mut self, cg_max_iterations: usize) -> Self {
        self.cg_max_iterations = cg_max_iterations;
        self
    }

    /// Sets convergence tolerance for eigenvalues.
    pub fn tolerance(mut self, tolerance: S) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets convergence tolerance for CG method.
    pub fn cg_tolerance(mut self, cg_tolerance: S) -> Self {
        self.cg_tolerance = cg_tolerance;
        self
    }

    /// Sets convergence tolerance for eigenvectors.
    pub fn vector_tolerance(mut self, vector_tolerance: S) -> Self {
        self.vector_tolerance = vector_tolerance;
        self
    }
}

impl<S> Default for OmegaOption<S>
where
    S: DrawingValue,
{
    fn default() -> Self {
        Self::new()
    }
}

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
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `options` - Configuration options for the Omega algorithm
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
    pub fn new<G, F, R>(graph: G, length: F, options: OmegaOption<S>, rng: &mut R) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
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

        // Step 1 & 2: Compute spectral coordinates using edge weights
        let coordinates =
            Self::compute_spectral_coordinates_with_weights(graph, length, &options, rng);

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
                let distance = distance.max(options.min_dist);
                let weight = S::one() / (distance * distance);
                node_pairs.push((i, j, distance, distance, weight, weight));
            }
        }

        // Step 4: Add random node pairs with Euclidean distances (avoiding duplicates)
        for i in 0..n {
            for _ in 0..options.k {
                let j = rng.gen_range(0..n);
                if i != j {
                    let pair_key = if i < j { (i, j) } else { (j, i) };

                    if !used_pairs.contains(&pair_key) {
                        used_pairs.insert(pair_key);
                        let distance = Self::euclidean_distance(&coordinates[i], &coordinates[j]);
                        let distance = distance.max(options.min_dist);
                        let weight = S::one() / (distance * distance);
                        node_pairs.push((i, j, distance, distance, weight, weight));
                    }
                    // Skip if duplicate - no re-sampling
                }
            }
        }

        Omega { node_pairs }
    }

    /// Computes d-dimensional spectral coordinates using edge weights and custom options.
    ///
    /// Uses a weighted Laplacian based on edge lengths/weights and configurable eigenvalue solver
    /// to compute the smallest d non-zero eigenvalues and eigenvectors, then creates coordinates
    /// by dividing each eigenvector by the square root of its corresponding eigenvalue.
    ///
    /// # Parameters
    /// * `graph` - The input graph
    /// * `length` - Function to extract edge weights/lengths
    /// * `options` - Configuration options containing solver parameters and dimensions
    /// * `rng` - Random number generator for eigenvalue computation
    ///
    /// # Returns
    /// A vector where coordinates[i] contains the d-dimensional coordinate for node i
    fn compute_spectral_coordinates_with_weights<G, F, R>(
        graph: G,
        length: F,
        options: &OmegaOption<S>,
        rng: &mut R,
    ) -> Vec<Vec<S>>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        let n = graph.node_count();

        // Create weighted Laplacian structure
        let laplacian = LaplacianStructure::new(graph, length);

        // Create custom eigenvalue solver from options
        let solver = EigenSolver::new(
            options.max_iterations,
            options.cg_max_iterations,
            options.tolerance,
            options.cg_tolerance,
            options.vector_tolerance,
        );

        // Step 1: Compute smallest d non-zero eigenvalues and eigenvectors
        let (eigenvalues, eigenvectors) =
            solver.compute_smallest_eigenvalues_with_laplacian(&laplacian, options.d, rng);

        // Step 2: Create coordinates by dividing eigenvectors by sqrt of eigenvalues
        let mut coordinates = vec![vec![S::zero(); options.d]; n];

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
