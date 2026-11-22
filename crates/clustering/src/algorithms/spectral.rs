use crate::{CommunityDetection, utils::renumber_communities};
use egraph_clustering::{KMeansConfig, kmeans_plus_plus};
use petgraph::visit::{EdgeCount, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::DrawingIndex;
use petgraph_linalg_kernel::RdMds;
use petgraph_linalg_kernel::solvers::Ic0CgSolver;
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
/// - Pure-Rust `egraph_clustering::kmeans_plus_plus` for deterministic cluster assignment
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
            let mut communities = HashMap::new();
            if let Some(node) = graph.node_identifiers().next() {
                communities.insert(node, 0);
            }
            return communities;
        }

        // Collect nodes for indexing
        let nodes: Vec<G::NodeId> = graph.node_identifiers().collect();

        // Use RdMds to compute spectral coordinates
        // The eigenvectors of the Laplacian provide the embedding for clustering
        // Use a seeded RNG for reproducibility
        let mut rng = StdRng::seed_from_u64(42);
        let mut embedding_algo = RdMds::new();
        embedding_algo
            .d(self.k) // Use k dimensions for k-way clustering
            .shift(1e-3f32)
            .eigenvalue_max_iterations(1000)
            .eigenvalue_tolerance(1e-1);

        let solver = Ic0CgSolver {
            max_iterations: 100,
            tolerance: 1e-4f32,
        };

        let embedding = embedding_algo.embedding(&graph, |_| 1.0f32, &solver, &mut rng);

        // Flatten spectral coordinates to f64 for pure-Rust K-Means++ clustering.
        let dims = embedding.ncols();
        if dims == 0 {
            return HashMap::new();
        }
        let mut points_flat = Vec::with_capacity(node_count * dims);
        for i in 0..node_count {
            for j in 0..dims {
                let val = embedding[[i, j]] as f64;
                points_flat.push(if val.is_finite() { val } else { 0.0 });
            }
        }

        let config = KMeansConfig {
            k: self.k.min(node_count),
            max_iters: 100,
            tol: 1e-4,
            n_init: 1,
            seed: 42,
        };
        let Ok(result) = kmeans_plus_plus(&points_flat, node_count, dims, &config) else {
            return HashMap::new();
        };

        // Map back to node IDs
        let mut communities = HashMap::new();
        for (i, &cluster) in result.labels.iter().enumerate() {
            communities.insert(nodes[i], cluster as usize);
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

    #[test]
    fn test_spectral_too_many_clusters_and_determinism() {
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n1, n2, ());

        let spectral_too_many = Spectral::new(3);
        assert!(spectral_too_many.detect_communities(&graph).is_empty());

        let spectral = Spectral::new(2);
        assert_eq!(
            spectral.detect_communities(&graph),
            spectral.detect_communities(&graph)
        );
    }
}
