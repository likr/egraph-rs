use crate::{utils::renumber_communities, CommunityDetection};
use petgraph::visit::{EdgeCount, IntoNeighbors, IntoNodeIdentifiers};
use std::collections::HashMap;
use std::hash::Hash;

/// Spectral clustering algorithm implementation.
///
/// This algorithm uses the eigenvectors of the graph Laplacian matrix
/// to identify communities in the graph. It's particularly effective
/// for finding well-separated communities.
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
    G: EdgeCount + IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Clone,
{
    fn detect_communities(&self, graph: G) -> HashMap<G::NodeId, usize> {
        let node_count = graph.node_identifiers().count();

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

        // Build adjacency matrix representation
        let nodes: Vec<G::NodeId> = graph.node_identifiers().collect();
        let node_indices: HashMap<G::NodeId, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (*node, i))
            .collect();

        // Build adjacency matrix (simple unweighted representation)
        let mut adjacency = vec![vec![0.0; node_count]; node_count];
        for (i, node) in nodes.iter().enumerate() {
            for neighbor in graph.neighbors(*node) {
                let j = node_indices[&neighbor];
                adjacency[i][j] = 1.0;
            }
        }

        // Build degree matrix
        let mut degree = vec![0.0; node_count];
        for i in 0..node_count {
            degree[i] = adjacency[i].iter().sum();
        }

        // Build Laplacian matrix: L = D - A
        let mut laplacian = vec![vec![0.0; node_count]; node_count];
        for i in 0..node_count {
            for j in 0..node_count {
                if i == j {
                    laplacian[i][j] = degree[i] - adjacency[i][j];
                } else {
                    laplacian[i][j] = -adjacency[i][j];
                }
            }
        }

        // Compute eigenvalues and eigenvectors using power iteration
        // For simplicity, we implement a basic approach that works for small graphs
        // In practice, you would use a specialized library like nalgebra or ndarray
        let eigenvectors = compute_first_k_eigenvectors(&laplacian, self.k);

        // Perform k-means clustering on the eigenvectors
        let clusters = kmeans_clustering(&eigenvectors, self.k);

        // Map back to node IDs
        let mut communities = HashMap::new();
        for (i, &cluster) in clusters.iter().enumerate() {
            communities.insert(nodes[i], cluster);
        }

        // Renumber communities from 0 to n-1 (in case some clusters are empty)
        renumber_communities(&communities)
    }
}

/// Computes the first k eigenvectors of the Laplacian matrix using power iteration.
///
/// This is a simplified implementation that works for small graphs.
/// For real-world applications, use a specialized linear algebra library.
fn compute_first_k_eigenvectors(matrix: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut eigenvectors: Vec<Vec<f64>> = Vec::with_capacity(k);

    // Very simplified approach to find approximate eigenvectors
    for i in 0..k {
        // Initialize with random vector (here we use a simple approach)
        let mut v: Vec<f64> = vec![0.0; n];
        v[i % n] = 1.0; // Simple initialization, not truly random

        // Orthogonalize against previous eigenvectors
        for prev_v in &eigenvectors {
            let dot_product: f64 = v.iter().zip(prev_v.iter()).map(|(a, b)| a * b).sum();
            for j in 0..n {
                v[j] -= dot_product * prev_v[j];
            }
        }

        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for j in 0..n {
                v[j] /= norm;
            }
        }

        // Power iteration
        for _ in 0..20 {
            // Small number of iterations for simplicity
            let mut next_v: Vec<f64> = vec![0.0; n];

            // Matrix-vector multiplication
            for i in 0..n {
                for j in 0..n {
                    next_v[i] += matrix[i][j] * v[j];
                }
            }

            // Orthogonalize against previous eigenvectors
            for prev_v in &eigenvectors {
                let dot_product: f64 = next_v.iter().zip(prev_v.iter()).map(|(a, b)| a * b).sum();
                for j in 0..n {
                    next_v[j] -= dot_product * prev_v[j];
                }
            }

            // Normalize
            let norm: f64 = next_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for j in 0..n {
                    next_v[j] /= norm;
                }
                v = next_v;
            } else {
                break; // Vector collapsed, likely orthogonal to all eigenvectors
            }
        }

        eigenvectors.push(v);
    }

    eigenvectors
}

/// Performs k-means clustering on the eigenvectors.
///
/// This is a simplified implementation that works for small datasets.
/// For real-world applications, use a specialized clustering library.
fn kmeans_clustering(data: &[Vec<f64>], k: usize) -> Vec<usize> {
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
