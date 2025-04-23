use crate::{utils::renumber_communities, CommunityDetection};
use petgraph::visit::{EdgeCount, IntoNeighbors, IntoNodeIdentifiers};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::collections::HashMap;
use std::hash::Hash;

/// Label Propagation community detection algorithm implementation.
///
/// This algorithm detects communities in a graph by propagating labels through
/// the network. Each node adopts the label that most of its neighbors have.
/// The algorithm is simple and fast.
///
/// # Examples
///
/// ```
/// use petgraph::graph::UnGraph;
/// use petgraph_clustering::{LabelPropagation, CommunityDetection};
///
/// // Create a simple graph with two communities
/// let mut graph = UnGraph::<(), ()>::new_undirected();
/// let n1 = graph.add_node(());
/// let n2 = graph.add_node(());
/// let n3 = graph.add_node(());
/// let n4 = graph.add_node(());
///
/// // Community 1: n1, n2 are densely connected
/// graph.add_edge(n1, n2, ());
///
/// // Community 2: n3, n4 are densely connected
/// graph.add_edge(n3, n4, ());
///
/// // Weak connection between communities
/// graph.add_edge(n2, n3, ());
///
/// // Detect communities
/// let label_prop = LabelPropagation::new();
/// let communities = label_prop.detect_communities(&graph);
/// ```
pub struct LabelPropagation {
    max_iterations: usize,
    seed: Option<u64>,
}

impl LabelPropagation {
    /// Creates a new Label Propagation community detection algorithm instance.
    ///
    /// Default maximum iterations is set to 100.
    pub fn new() -> Self {
        Self {
            max_iterations: 100,
            seed: None,
        }
    }

    /// Creates a new Label Propagation community detection algorithm instance with
    /// the specified maximum number of iterations.
    pub fn with_max_iterations(max_iterations: usize) -> Self {
        Self {
            max_iterations,
            seed: None,
        }
    }

    /// Creates a new Label Propagation community detection algorithm instance with
    /// the specified random seed for reproducibility.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            max_iterations: 100,
            seed: Some(seed),
        }
    }

    /// Creates a new Label Propagation community detection algorithm instance with
    /// the specified maximum number of iterations and random seed.
    pub fn with_max_iterations_and_seed(max_iterations: usize, seed: u64) -> Self {
        Self {
            max_iterations,
            seed: Some(seed),
        }
    }
}

impl<G> CommunityDetection<G> for LabelPropagation
where
    G: EdgeCount + IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Clone,
{
    fn detect_communities(&self, graph: G) -> HashMap<G::NodeId, usize> {
        // Initialize each node with a unique label
        let mut labels: HashMap<G::NodeId, usize> = HashMap::new();
        for (i, node) in graph.node_identifiers().enumerate() {
            labels.insert(node.clone(), i);
        }

        // Create a Random Number Generator with seed if provided
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Store all nodes in a vector for random shuffling
        let mut nodes: Vec<G::NodeId> = graph.node_identifiers().collect();

        // Perform label propagation for max_iterations or until convergence
        let mut iteration = 0;
        let mut changed = true;

        while changed && iteration < self.max_iterations {
            changed = false;

            // Shuffle nodes to avoid bias in propagation order
            nodes.shuffle(&mut rng);

            for node in &nodes {
                let neighbor_labels = collect_neighbor_labels(graph, *node, &labels);

                if let Some(most_common_label) = find_most_common_label(&neighbor_labels, &mut rng)
                {
                    if labels[node] != most_common_label {
                        *labels.get_mut(node).unwrap() = most_common_label;
                        changed = true;
                    }
                }
            }

            iteration += 1;
        }

        // Renumber communities from 0 to n-1
        renumber_communities(&labels)
    }
}

/// Collects labels from a node's neighbors.
fn collect_neighbor_labels<G>(
    graph: G,
    node: G::NodeId,
    labels: &HashMap<G::NodeId, usize>,
) -> Vec<usize>
where
    G: IntoNeighbors,
    G::NodeId: Eq + Hash + Clone,
{
    graph
        .neighbors(node.clone())
        .map(|neighbor| labels[&neighbor])
        .collect()
}

/// Finds the most common label among neighbors.
/// In case of a tie, randomly selects one of the most common labels.
fn find_most_common_label<R: rand::Rng>(neighbor_labels: &[usize], rng: &mut R) -> Option<usize> {
    if neighbor_labels.is_empty() {
        return None;
    }

    // Count occurrences of each label
    let mut label_counts: HashMap<usize, usize> = HashMap::new();
    for &label in neighbor_labels {
        *label_counts.entry(label).or_insert(0) += 1;
    }

    // Find the maximum count
    let max_count = label_counts.values().cloned().max().unwrap_or(0);

    // Collect all labels that have the maximum count
    let most_common_labels: Vec<usize> = label_counts
        .iter()
        .filter(|&(_, &count)| count == max_count)
        .map(|(&label, _)| label)
        .collect();

    // Randomly select one of the most common labels
    most_common_labels.choose(rng).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;

    #[test]
    fn test_label_propagation_simple_graph() {
        // Create a simple graph with two communities
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        let n4 = graph.add_node(());

        // Community 1: n1, n2 are densely connected
        graph.add_edge(n1, n2, ());

        // Community 2: n3, n4 are densely connected
        graph.add_edge(n3, n4, ());

        // Weak connection between communities
        graph.add_edge(n2, n3, ());

        // Use a fixed seed for reproducibility
        let label_prop = LabelPropagation::with_seed(42);
        let communities = label_prop.detect_communities(&graph);

        // Verify that nodes in the same group are in the same community
        assert_eq!(
            communities[&n1], communities[&n2],
            "Nodes 1 and 2 should be in the same community"
        );
        assert_eq!(
            communities[&n3], communities[&n4],
            "Nodes 3 and 4 should be in the same community"
        );
    }

    #[test]
    fn test_label_propagation_empty_graph() {
        // Create an empty graph
        let graph = UnGraph::<(), ()>::new_undirected();

        // Detect communities
        let label_prop = LabelPropagation::new();
        let communities = label_prop.detect_communities(&graph);

        // Verify that we get an empty community assignment
        assert!(
            communities.is_empty(),
            "Empty graph should have no communities"
        );
    }

    #[test]
    fn test_label_propagation_single_node() {
        // Create a graph with a single node
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let n1 = graph.add_node(());

        // Detect communities
        let label_prop = LabelPropagation::new();
        let communities = label_prop.detect_communities(&graph);

        // Verify that the node is assigned to community 0
        assert_eq!(
            communities[&n1], 0,
            "Single node should be assigned to community 0"
        );
    }

    #[test]
    fn test_label_propagation_one_community() {
        // Create a graph with one community (complete graph)
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());

        // All nodes are connected to each other
        graph.add_edge(n1, n2, ());
        graph.add_edge(n1, n3, ());
        graph.add_edge(n2, n3, ());

        // Detect communities
        let label_prop = LabelPropagation::new();
        let communities = label_prop.detect_communities(&graph);

        // Verify that all nodes are in the same community
        assert_eq!(
            communities[&n1], communities[&n2],
            "Nodes 1 and 2 should be in the same community"
        );
        assert_eq!(
            communities[&n2], communities[&n3],
            "Nodes 2 and 3 should be in the same community"
        );
    }

    #[test]
    fn test_label_propagation_reproducibility() {
        // Create a graph where randomization affects the outcome
        let mut graph = UnGraph::<(), ()>::new_undirected();
        for _ in 0..10 {
            graph.add_node(());
        }

        // Add some random edges
        for i in 0..9 {
            graph.add_edge(i.into(), (i + 1).into(), ());
        }
        graph.add_edge(0.into(), 5.into(), ());
        graph.add_edge(5.into(), 9.into(), ());

        // Run the algorithm twice with the same seed
        let label_prop1 = LabelPropagation::with_seed(123);
        let communities1 = label_prop1.detect_communities(&graph);

        let label_prop2 = LabelPropagation::with_seed(123);
        let communities2 = label_prop2.detect_communities(&graph);

        // Verify that the results are the same
        for node in graph.node_identifiers() {
            assert_eq!(
                communities1[&node], communities2[&node],
                "Results should be reproducible with the same seed"
            );
        }

        // Run with a different seed
        let label_prop3 = LabelPropagation::with_seed(456);
        let _communities3 = label_prop3.detect_communities(&graph);

        // Note: We don't verify that communities3 is different because
        // there's a small chance they might be the same even with different seeds
    }
}
