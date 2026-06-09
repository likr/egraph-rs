//! Omega implementation for creating SGD instances from spectral embeddings.

use ndarray::Array2;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_distance::Distance;
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_layout_sgd::Sgd;
use petgraph_linalg_embedding_distance::EmbeddingDistanceMatrix;
use rand::Rng;
use std::collections::{HashMap, HashSet};

/// Omega builder for creating SGD instances from spectral embeddings.
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
    pub fn build<G, R>(&self, graph: G, embedding: &Array2<S>, rng: &mut R) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex + std::hash::Hash + Eq,
        R: Rng,
    {
        let distance_matrix = EmbeddingDistanceMatrix::new(graph, embedding.clone(), self.min_dist);
        let node_pairs = compute_node_pairs_from_distance(graph, &distance_matrix, self.k, rng);
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

/// Computes node pairs from a Distance implementation.
fn compute_node_pairs_from_distance<S, G, D, R>(
    graph: G,
    distance_matrix: &D,
    k: usize,
    rng: &mut R,
) -> Vec<(usize, usize, S, S, S, S)>
where
    S: DrawingValue,
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
    G::NodeId: DrawingIndex + std::hash::Hash + Eq,
    D: Distance<G::NodeId, S>,
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

    // Step 1: Add edge-based node pairs with distances
    for edge in graph.edge_references() {
        let i = node_indices[&edge.source()];
        let j = node_indices[&edge.target()];
        let pair_key = if i < j { (i, j) } else { (j, i) };

        if !used_pairs.contains(&pair_key) {
            used_pairs.insert(pair_key);
            let distance = distance_matrix.get_by_index(i, j);
            let weight = S::one() / (distance * distance);
            node_pairs.push((i, j, distance, distance, weight, weight));
        }
    }

    // Step 2: Add random node pairs with distances (avoiding duplicates)
    for i in 0..n {
        for _ in 0..k {
            let j = rng.gen_range(0..n);
            if i != j {
                let pair_key = if i < j { (i, j) } else { (j, i) };

                if !used_pairs.contains(&pair_key) {
                    used_pairs.insert(pair_key);
                    let distance = distance_matrix.get_by_index(i, j);
                    let weight = S::one() / (distance * distance);
                    node_pairs.push((i, j, distance, distance, weight, weight));
                }
            }
        }
    }

    node_pairs
}
