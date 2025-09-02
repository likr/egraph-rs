//! Omega implementation of the SGD trait for graph layout using spectral coordinates.

use crate::eigenvalue::{
    compute_spectral_coordinates, compute_spectral_coordinates_and_eigenvalues,
};
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
    /// Number of spectral dimensions
    pub d: usize,
    /// Number of random pairs per node  
    pub k: usize,
    /// Minimum distance between node pairs
    pub min_dist: S,
    /// Shift parameter for creating positive definite matrix L + cI
    pub shift: S,
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
    /// - shift: 1e-3 (shift parameter for positive definite matrix)
    /// - eigenvalue_max_iterations: 1000 (eigenvalue solver)
    /// - cg_max_iterations: 100 (CG solver)
    /// - eigenvalue_tolerance: 1e-1 (eigenvalue convergence)
    /// - cg_tolerance: 1e-4 (CG convergence)
    pub fn new() -> Self {
        Self {
            d: 2,
            k: 30,
            min_dist: S::from_f32(1e-3).unwrap(),
            shift: S::from_f32(1e-3).unwrap(),
            eigenvalue_max_iterations: 1000,
            cg_max_iterations: 100,
            eigenvalue_tolerance: S::from_f32(1e-1).unwrap(),
            cg_tolerance: S::from_f32(1e-4).unwrap(),
        }
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

    /// Sets the shift parameter for creating positive definite matrix L + cI.
    pub fn shift(&mut self, shift: S) -> &mut Self {
        self.shift = shift;
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

    /// Computes spectral coordinates using the configured parameters.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `rng` - Random number generator for spectral coordinate computation
    ///
    /// # Returns
    /// An Array2 where coordinates.row(i) contains the d-dimensional coordinate for node i
    pub fn embedding<G, F, R>(&self, graph: G, length: F, rng: &mut R) -> Array2<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        compute_spectral_coordinates(
            graph,
            length,
            self.shift,
            self.eigenvalue_max_iterations,
            self.cg_max_iterations,
            self.eigenvalue_tolerance,
            self.cg_tolerance,
            self.d,
            rng,
        )
    }

    /// Computes spectral coordinates and eigenvalues using the configured parameters.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `rng` - Random number generator for spectral coordinate computation
    ///
    /// # Returns
    /// A tuple containing:
    /// - Array2 where coordinates.row(i) contains the d-dimensional coordinate for node i
    /// - Array1 of eigenvalues (λ_0, λ_1, ..., λ_d)
    pub fn embedding_and_eigenvalues<G, F, R>(
        &self,
        graph: G,
        length: F,
        rng: &mut R,
    ) -> (Array2<S>, ndarray::Array1<S>)
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        compute_spectral_coordinates_and_eigenvalues(
            graph,
            length,
            self.shift,
            self.eigenvalue_max_iterations,
            self.cg_max_iterations,
            self.eigenvalue_tolerance,
            self.cg_tolerance,
            self.d,
            rng,
        )
    }

    /// Builds an SGD instance using precomputed embedding.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `embedding` - Precomputed spectral coordinates
    /// * `rng` - Random number generator for selecting random node pairs
    ///
    /// # Returns
    /// A new SGD instance configured with node pairs derived from the embedding
    pub fn build_with_embedding<G, R>(&self, graph: G, embedding: &Array2<S>, rng: &mut R) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        R: Rng,
    {
        let node_pairs = compute_omega_node_pairs(graph, embedding, self.min_dist, self.k, rng);
        Sgd::new(node_pairs)
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
        let embedding = self.embedding(graph, length, rng);
        self.build_with_embedding(graph, &embedding, rng)
    }
}

/// Computes node pairs for the Omega algorithm using precomputed spectral coordinates.
///
/// This function generates node pairs from both edges and random sampling, using
/// distances computed from the provided spectral embedding coordinates.
///
/// # Parameters
/// * `graph` - The input graph to be laid out
/// * `embedding` - Precomputed spectral coordinates where embedding.row(i) is the coordinate for node i
/// * `min_dist` - Minimum distance between node pairs
/// * `k` - Number of random pairs per node
/// * `rng` - Random number generator for selecting random node pairs
///
/// # Returns
/// A vector of node pairs ready for SGD processing
fn compute_omega_node_pairs<S, G, R>(
    graph: G,
    embedding: &Array2<S>,
    min_dist: S,
    k: usize,
    rng: &mut R,
) -> Vec<(usize, usize, S, S, S, S)>
where
    S: DrawingValue,
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
    G::NodeId: DrawingIndex,
    R: Rng,
{
    let n = graph.node_count();

    // Create node index mapping
    let node_indices: HashMap<G::NodeId, usize> = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, node_id)| (node_id, i))
        .collect();

    let mut node_pairs = Vec::new();
    let mut used_pairs = HashSet::new();

    // Step 1: Add edge-based node pairs with Euclidean distances
    for edge in graph.edge_references() {
        let i = node_indices[&edge.source()];
        let j = node_indices[&edge.target()];
        let pair_key = if i < j { (i, j) } else { (j, i) };

        if !used_pairs.contains(&pair_key) {
            used_pairs.insert(pair_key);
            let distance = euclidean_distance(embedding.row(i), embedding.row(j));
            let distance = distance.max(min_dist);
            let weight = S::one() / (distance * distance);
            node_pairs.push((i, j, distance, distance, weight, weight));
        }
    }

    // Step 2: Add random node pairs with Euclidean distances (avoiding duplicates)
    for i in 0..n {
        for _ in 0..k {
            let j = rng.gen_range(0..n);
            if i != j {
                let pair_key = if i < j { (i, j) } else { (j, i) };

                if !used_pairs.contains(&pair_key) {
                    used_pairs.insert(pair_key);
                    let distance = euclidean_distance(embedding.row(i), embedding.row(j));
                    let distance = distance.max(min_dist);
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
