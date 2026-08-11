use crate::Sgd;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_linalg_kernel::Distance;
use rand::Rng;
use std::collections::{HashMap, HashSet};

/// RandomPairSparseSgd generates SGD instances with all edges and `k` random node pairs per node.
pub struct RandomPairSparseSgd {
    pub k: usize,
}

impl Default for RandomPairSparseSgd {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomPairSparseSgd {
    pub fn new() -> Self {
        Self { k: 30 }
    }

    pub fn k(&mut self, k: usize) -> &mut Self {
        self.k = k;
        self
    }

    pub fn build<G, R, S>(
        &self,
        graph: G,
        distance_matrix: &dyn Distance<G::NodeId, S>,
        rng: &mut R,
    ) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        R: Rng,
        S: DrawingValue,
    {
        let n = graph.node_count();

        // Create node index mapping
        let node_indices: HashMap<G::NodeId, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id, i))
            .collect();
        let nodes = graph.node_identifiers().collect::<Vec<_>>();

        let mut node_pairs = Vec::new();
        let mut used_pairs = HashSet::new();

        // Step 1: Add edge-based node pairs with distances
        for edge in graph.edge_references() {
            let u = edge.source();
            let v = edge.target();
            let i = node_indices[&u];
            let j = node_indices[&v];

            if i != j {
                let pair_key = if i < j { (i, j) } else { (j, i) };

                if !used_pairs.contains(&pair_key) {
                    used_pairs.insert(pair_key);
                    if let Some(distance) = distance_matrix.get(u, v) {
                        let weight = S::one() / (distance * distance);
                        node_pairs.push((i, j, distance, distance, weight, weight));
                    }
                }
            }
        }

        // Step 2: Add random node pairs with distances
        for i in 0..n {
            for _ in 0..self.k {
                let j = rng.gen_range(0..n);
                if i != j {
                    let pair_key = if i < j { (i, j) } else { (j, i) };

                    if !used_pairs.contains(&pair_key) {
                        used_pairs.insert(pair_key);
                        let u = nodes[i];
                        let v = nodes[j];
                        if let Some(distance) = distance_matrix.get(u, v) {
                            let weight = S::one() / (distance * distance);
                            node_pairs.push((i, j, distance, distance, weight, weight));
                        }
                    }
                }
            }
        }

        Sgd::new(node_pairs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;
    use petgraph_algorithm_shortest_path::{all_sources_dijkstra, FullDistanceMatrix};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_random_pair_sparse_sgd_basic() {
        let mut graph = Graph::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n0, n1, ());
        graph.add_edge(n1, n2, ());

        let distance_matrix: FullDistanceMatrix<_, f32> = all_sources_dijkstra(&graph, |_| 1.0f32);
        let mut rng = StdRng::seed_from_u64(42);

        let layout = RandomPairSparseSgd::new()
            .k(5)
            .build(&graph, &distance_matrix, &mut rng);
        assert!(layout.node_pairs().len() >= 2);
    }
}
