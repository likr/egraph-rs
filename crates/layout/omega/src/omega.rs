//! Omega implementation for creating SGD instances from spectral embeddings.

use ndarray::{Array2, Zip};
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_layout_sgd::Sgd;
use rand::Rng;
use std::collections::{HashMap, HashSet};

/// Omega builder for creating SGD instances from spectral embeddings.
///
/// This structure takes precomputed spectral coordinates (embeddings) and generates
/// node pairs for SGD optimization. It does not compute embeddings itself - use
/// RdMds from petgraph-linalg-rdmds for that purpose.
#[derive(Debug, Clone)]
pub struct Omega<S> {
    /// Number of random pairs per node  
    pub k: usize,
    /// Minimum distance between node pairs
    pub min_dist: S,
}

impl<S> Omega<S>
where
    S: DrawingValue,
{
    /// Creates a new Omega with default values.
    ///
    /// Default values:
    /// - k: 30 (random pairs per node)
    /// - min_dist: 1e-3 (minimum distance)
    pub fn new() -> Self {
        Self {
            k: 30,
            min_dist: S::from_f32(1e-3).unwrap(),
        }
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

    /// Builds an SGD instance from precomputed embedding.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `embedding` - Precomputed spectral coordinates (from RdMds)
    /// * `rng` - Random number generator for selecting random node pairs
    ///
    /// # Returns
    /// A new SGD instance configured with node pairs derived from the embedding
    pub fn build<G, R>(&self, graph: G, embedding: &Array2<S>, rng: &mut R) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        R: Rng,
    {
        let node_pairs =
            compute_node_pairs_from_embedding(graph, embedding, self.min_dist, self.k, rng);
        Sgd::new(node_pairs)
    }
}

impl<S> Default for Omega<S>
where
    S: DrawingValue,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Computes node pairs from precomputed spectral embedding coordinates.
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
fn compute_node_pairs_from_embedding<S, G, R>(
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
