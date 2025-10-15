use crate::{CommunityDetection, utils::renumber_communities};
use petgraph::visit::{EdgeCount, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::DrawingIndex;
use petgraph_linalg_rdmds::RdMds;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::hash::Hash;

/// Spectral clustering algorithm implementation.
///
/// This algorithm uses the eigenvectors of the graph Laplacian matrix
/// to identify communities in the graph. It's particularly effective
/// for finding well-separated communities.
///
/// The implementation uses RdMds (Resistance-distance MDS) for computing
/// spectral coordinates, which provides high-quality eigenvalue computation
/// with IC(0) preconditioning and conjugate gradient solver.
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

        // Convert Array2 to Vec<Vec<f64>> for k-means clustering
        let eigenvectors: Vec<Vec<f64>> = (0..node_count)
            .map(|i| (0..self.k).map(|j| embedding[[i, j]] as f64).collect())
            .collect();

        // Perform k-means clustering on the eigenvectors with seeded RNG
        let mut kmeans_rng = StdRng::seed_from_u64(123);
        let clusters = kmeans_clustering(&eigenvectors, self.k, &mut kmeans_rng);

        // Map back to node IDs
        let mut communities = HashMap::new();
        for (i, &cluster) in clusters.iter().enumerate() {
            communities.insert(nodes[i], cluster);
        }

        // Renumber communities from 0 to n-1 (in case some clusters are empty)
        renumber_communities(&communities)
    }
}

/// Performs k-means clustering on the eigenvectors.
///
/// This is a simplified implementation that works for small datasets.
/// For real-world applications, use a specialized clustering library.
fn kmeans_clustering<R: Rng>(data: &[Vec<f64>], k: usize, _rng: &mut R) -> Vec<usize> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }

    let dim = data[0].len();

    // Initialize centroids (use first k data points as initial centroids)
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    for i in 0..k {
        if i < n {
            centroids.push(data[i].clone());
        } else {
            // If k > n, duplicate some centroids
            centroids.push(data[i % n].clone());
        }
    }

    // Initialize cluster assignments
    let mut clusters = vec![0; n];

    // Iterative refinement
    let max_iterations = 20; // Limit iterations for simplicity
    for _ in 0..max_iterations {
        let mut changed = false;

        // Assign points to nearest centroid
        for i in 0..n {
            let mut min_dist = f64::MAX;
            let mut min_cluster = 0;

            for c in 0..k {
                let mut dist = 0.0;
                for j in 0..dim {
                    let diff = data[i][j] - centroids[c][j];
                    dist += diff * diff;
                }

                if dist < min_dist {
                    min_dist = dist;
                    min_cluster = c;
                }
            }

            if clusters[i] != min_cluster {
                clusters[i] = min_cluster;
                changed = true;
            }
        }

        if !changed {
            break; // Convergence reached
        }

        // Update centroids
        let mut new_centroids = vec![vec![0.0; dim]; k];
        let mut counts = vec![0; k];

        for i in 0..n {
            let cluster = clusters[i];
            counts[cluster] += 1;

            for j in 0..dim {
                new_centroids[cluster][j] += data[i][j];
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..dim {
                    new_centroids[c][j] /= counts[c] as f64;
                }
            } else {
                // Handle empty clusters by keeping the old centroid
                new_centroids[c] = centroids[c].clone();
            }
        }

        centroids = new_centroids;
    }

    clusters
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

        // Community 1: n1, n2 are densely connected
        graph.add_edge(n1, n2, ());

        // Community 2: n3, n4 are densely connected
        graph.add_edge(n3, n4, ());

        // Weak connection between communities
        graph.add_edge(n2, n3, ());

        // Detect communities
        let spectral = Spectral::new(2);
        let communities = spectral.detect_communities(&graph);

        // Verify communities
        assert_eq!(
            communities[&n1], communities[&n2],
            "Nodes 1 and 2 should be in the same community"
        );
        assert_eq!(
            communities[&n3], communities[&n4],
            "Nodes 3 and 4 should be in the same community"
        );
        assert_ne!(
            communities[&n1], communities[&n3],
            "Nodes 1 and 3 should be in different communities"
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
