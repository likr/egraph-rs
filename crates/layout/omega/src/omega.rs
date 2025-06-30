//! Omega implementation of the SGD trait for graph layout using spectral coordinates.

use crate::eigenvalue::{compute_smallest_eigenvalues_with_laplacian, LaplacianStructure};
use ndarray::{Array2, Zip};
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_layout_sgd::Sgd;
use rand::Rng;
use std::collections::{HashMap, HashSet};

/// Builder for configuring the Omega algorithm.
///
/// This structure contains all parameters needed to configure the Omega algorithm,
/// including spectral dimensions, random pairs, distance constraints, and
/// eigenvalue solver parameters.
#[derive(Debug, Clone)]
pub struct Omega<S> {
    pub eps: S,
    /// Number of spectral dimensions
    pub d: usize,
    /// Number of random pairs per node  
    pub k: usize,
    /// Minimum distance between node pairs
    pub min_dist: S,
    /// Maximum number of iterations for eigenvalue computation using inverse power method
    pub eigenvalue_max_iterations: usize,
    /// Maximum number of iterations for CG method
    pub cg_max_iterations: usize,
    /// Convergence tolerance for eigenvalue computation
    pub eigenvalue_tolerance: S,
    /// Convergence tolerance for CG method
    pub cg_tolerance: S,
}

impl<S> Omega<S>
where
    S: DrawingValue,
{
    /// Creates a new OmegaBuilder with default values.
    ///
    /// Default values:
    /// - d: 2 (spectral dimensions)
    /// - k: 30 (random pairs per node)
    /// - min_dist: 1e-3 (minimum distance)
    /// - eigenvalue_max_iterations: 1000 (eigenvalue solver)
    /// - cg_max_iterations: 100 (CG solver)
    /// - eigenvalue_tolerance: 1e-4 (eigenvalue convergence)
    /// - cg_tolerance: 1e-4 (CG convergence)
    pub fn new() -> Self {
        Self {
            eps: S::from_f32(0.1).unwrap(),
            d: 2,
            k: 30,
            min_dist: S::from_f32(1e-3).unwrap(),
            eigenvalue_max_iterations: 1000,
            cg_max_iterations: 100,
            eigenvalue_tolerance: S::from_f32(1e-4).unwrap(),
            cg_tolerance: S::from_f32(1e-4).unwrap(),
        }
    }

    pub fn eps(&mut self, eps: S) -> &mut Self {
        self.eps = eps;
        self
    }

    /// Sets the number of spectral dimensions.
    pub fn d(&mut self, d: usize) -> &mut Self {
        self.d = d;
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

    /// Sets maximum iterations for eigenvalue computation using inverse power method.
    pub fn eigenvalue_max_iterations(&mut self, eigenvalue_max_iterations: usize) -> &mut Self {
        self.eigenvalue_max_iterations = eigenvalue_max_iterations;
        self
    }

    /// Sets maximum iterations for CG method.
    pub fn cg_max_iterations(&mut self, cg_max_iterations: usize) -> &mut Self {
        self.cg_max_iterations = cg_max_iterations;
        self
    }

    /// Sets convergence tolerance for eigenvalue computation.
    pub fn eigenvalue_tolerance(&mut self, eigenvalue_tolerance: S) -> &mut Self {
        self.eigenvalue_tolerance = eigenvalue_tolerance;
        self
    }

    /// Sets convergence tolerance for CG method.
    pub fn cg_tolerance(&mut self, cg_tolerance: S) -> &mut Self {
        self.cg_tolerance = cg_tolerance;
        self
    }

    /// Builds an SGD instance using the configured parameters.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `rng` - Random number generator for selecting random node pairs
    ///
    /// # Returns
    /// A new SGD instance configured with the builder's parameters
    pub fn build<G, F, R>(&self, graph: G, length: F, rng: &mut R) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        let node_pairs = compute_omega_node_pairs(graph, length, self, rng);
        Sgd::new(node_pairs, self.eps)
    }
}

/// Computes node pairs for the Omega algorithm using spectral coordinates.
///
/// This function implements the core Omega algorithm logic, extracting it from
/// the previous Omega::new method to work with the new builder pattern.
///
/// # Parameters
/// * `graph` - The input graph to be laid out
/// * `length` - A function that maps edges to their lengths/weights
/// * `options` - Configuration options for the Omega algorithm
/// * `rng` - Random number generator for selecting random node pairs
///
/// # Returns
/// A vector of node pairs ready for SGD processing
fn compute_omega_node_pairs<S, G, F, R>(
    graph: G,
    length: F,
    options: &Omega<S>,
    rng: &mut R,
) -> Vec<(usize, usize, S, S, S, S)>
where
    S: DrawingValue,
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
    G::NodeId: DrawingIndex,
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
    let coordinates = compute_spectral_coordinates_with_weights(graph, length, options, rng);

    let mut node_pairs = Vec::new();
    let mut used_pairs = HashSet::new();

    // Step 3: Add edge-based node pairs with Euclidean distances
    for edge in graph.edge_references() {
        let i = node_indices[&edge.source()];
        let j = node_indices[&edge.target()];
        let pair_key = if i < j { (i, j) } else { (j, i) };

        if !used_pairs.contains(&pair_key) {
            used_pairs.insert(pair_key);
            let distance = euclidean_distance(coordinates.row(i), coordinates.row(j));
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
                    let distance = euclidean_distance(coordinates.row(i), coordinates.row(j));
                    let distance = distance.max(options.min_dist);
                    let weight = S::one() / (distance * distance);
                    node_pairs.push((i, j, distance, distance, weight, weight));
                }
                // Skip if duplicate - no re-sampling
            }
        }
    }

    node_pairs
}

impl<S> Default for Omega<S>
where
    S: DrawingValue,
{
    fn default() -> Self {
        Self::new()
    }
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
/// * `options` - Configuration builder containing solver parameters and dimensions
/// * `rng` - Random number generator for eigenvalue computation
///
/// # Returns
/// An Array2 where coordinates.row(i) contains the d-dimensional coordinate for node i
fn compute_spectral_coordinates_with_weights<S, G, F, R>(
    graph: G,
    length: F,
    options: &Omega<S>,
    rng: &mut R,
) -> Array2<S>
where
    S: DrawingValue,
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
    G::NodeId: DrawingIndex,
    F: FnMut(G::EdgeRef) -> S,
    R: Rng,
{
    // Create weighted Laplacian structure
    let laplacian = LaplacianStructure::new(graph, length);

    // Step 1: Compute smallest d non-zero eigenvalues and eigenvectors
    let (eigenvalues, mut eigenvectors) = compute_smallest_eigenvalues_with_laplacian(
        &laplacian,
        options.d,
        options.eigenvalue_max_iterations,
        options.cg_max_iterations,
        options.eigenvalue_tolerance,
        options.cg_tolerance,
        rng,
    );

    // Step 2: Create coordinates by dividing eigenvectors by sqrt of eigenvalues
    eigenvectors.column_mut(0).fill(S::zero());
    for dim in 1..=options.d {
        let mut eigenvector = eigenvectors.column_mut(dim);
        eigenvector /= eigenvalues[dim];
    }

    eigenvectors
}

/// Computes the Euclidean distance between two d-dimensional coordinates.
///
/// # Parameters
/// * `coord1` - First coordinate vector (ndarray row view)
/// * `coord2` - Second coordinate vector (ndarray row view)
///
/// # Returns
/// The Euclidean distance between the two coordinates
fn euclidean_distance<S>(coord1: ndarray::ArrayView1<S>, coord2: ndarray::ArrayView1<S>) -> S
where
    S: DrawingValue,
{
    let mut sum = S::zero();
    Zip::from(coord1).and(coord2).for_each(|&a, &b| {
        let diff = a - b;
        sum += diff * diff
    });
    sum.sqrt()
}
