use crate::{CommunityDetection, utils::renumber_communities};
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::Array2;
use petgraph::visit::{EdgeCount, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::DrawingIndex;
use petgraph_linalg_rdmds::RdMds;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::hash::Hash;

/// Spectral clustering algorithm implementation.
///
/// This algorithm uses the eigenvectors of the graph Laplacian matrix
/// to identify communities in the graph. It's particularly effective
/// for finding well-separated communities.
///
/// The implementation uses:
/// - RdMds (Resistance-distance MDS) for computing spectral coordinates with
///   high-quality eigenvalue computation using IC(0) preconditioning and
///   conjugate gradient solver
/// - linfa's k-means clustering with k-means++ initialization for robust
///   cluster assignment
///
/// # Examples
///
/// ```
/// use petgraph::graph::UnGraph;
/// use petgraph_clustering::{Spectral, CommunityDetection};
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
/// let spectral = Spectral::new(2); // Specify 2 communities
/// let communities = spectral.detect_communities(&graph);
/// ```
pub struct Spectral {
    k: usize, // Number of clusters/communities
}

impl Spectral {
    /// Creates a new Spectral clustering algorithm instance with
    /// the specified number of communities to detect.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of communities to detect. Must be greater than 0.
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "Number of communities must be greater than 0");
        Self { k }
    }
}

impl<G> CommunityDetection<G> for Spectral
where
    G: EdgeCount + IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
    G::NodeId: Eq + Hash + Clone + DrawingIndex,
{
    fn detect_communities(&self, graph: G) -> HashMap<G::NodeId, usize> {
        let node_count = graph.node_count();

        // For empty graphs or too many clusters, return empty map
        if node_count == 0 || self.k > node_count {
            return HashMap::new();
        }

        // For single-node graphs, assign to community 0
        if node_count == 1 {
            let node = graph.node_identifiers().next().unwrap();
            let mut communities = HashMap::new();
            communities.insert(node, 0);
            return communities;
        }

        // Collect nodes for indexing
        let nodes: Vec<G::NodeId> = graph.node_identifiers().collect();

        // Use RdMds to compute spectral coordinates
        // The eigenvectors of the Laplacian provide the embedding for clustering
        // Use a seeded RNG for reproducibility
        let mut rng = StdRng::seed_from_u64(42);
        let embedding = RdMds::new()
            .d(self.k) // Use k dimensions for k-way clustering
            .shift(1e-3f32)
            .eigenvalue_max_iterations(1000)
            .cg_max_iterations(100)
            .eigenvalue_tolerance(1e-1)
            .cg_tolerance(1e-4)
            .embedding(&graph, |_| 1.0f32, &mut rng);

        // Convert embedding to f64 for linfa (linfa uses f64)
        let data_array =
            Array2::from_shape_fn((node_count, self.k), |(i, j)| embedding[[i, j]] as f64);

        // Create dataset for k-means
        let dataset = Dataset::from(data_array);

        // Perform k-means clustering using linfa
        // Use k-means++ initialization for better cluster quality
        let model = KMeans::params(self.k)
            .max_n_iterations(100)
            .tolerance(1e-4)
            .fit(&dataset)
            .expect("K-means clustering failed");

        // Get cluster assignments
        let predictions = model.predict(&dataset);
        let clusters: Vec<usize> = predictions.as_targets().iter().copied().collect();

        // Map back to node IDs
        let mut communities = HashMap::new();
        for (i, &cluster) in clusters.iter().enumerate() {
            communities.insert(nodes[i], cluster);
        }

        // Renumber communities from 0 to n-1 (in case some clusters are empty)
        renumber_communities(&communities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;

    #[test]
    fn test_spectral_simple_graph() {
        // Create a simple graph with two communities
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        let n4 = graph.add_node(());
        let n5 = graph.add_node(());
        let n6 = graph.add_node(());

        // Community 1: n1, n2 are densely connected
        graph.add_edge(n1, n2, ());
        graph.add_edge(n1, n3, ());
        graph.add_edge(n2, n3, ());

        // Community 2: n3, n4 are densely connected
        graph.add_edge(n4, n5, ());
        graph.add_edge(n4, n6, ());
        graph.add_edge(n5, n6, ());

        // Weak connection between communities
        graph.add_edge(n3, n4, ());

        // Detect communities
        let spectral = Spectral::new(2);
        let communities = spectral.detect_communities(&graph);

        // Verify communities
        assert_eq!(
            communities[&n1], communities[&n2],
            "Nodes 1 and 2 should be in the same community"
        );
        assert_ne!(
            communities[&n3], communities[&n4],
            "Nodes 3 and 4 should be in different communities"
        );
        assert_eq!(
            communities[&n5], communities[&n6],
            "Nodes 5 and 6 should be in the same community"
        );
    }

    #[test]
    fn test_spectral_empty_graph() {
        // Create an empty graph
        let graph = UnGraph::<(), ()>::new_undirected();

        // Detect communities
        let spectral = Spectral::new(2);
        let communities = spectral.detect_communities(&graph);

        // Verify that we get an empty community assignment
        assert!(
            communities.is_empty(),
            "Empty graph should have no communities"
        );
    }

    #[test]
    fn test_spectral_single_node() {
        // Create a graph with a single node
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let n1 = graph.add_node(());

        // Detect communities
        let spectral = Spectral::new(1);
        let communities = spectral.detect_communities(&graph);

        // Verify that the node is assigned to community 0
        assert_eq!(
            communities[&n1], 0,
            "Single node should be assigned to community 0"
        );
    }

    #[test]
    fn test_spectral_one_community() {
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
        let spectral = Spectral::new(1);
        let communities = spectral.detect_communities(&graph);

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
}
