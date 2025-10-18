use crate::{CommunityDetection, utils::renumber_communities};
use petgraph::visit::{EdgeCount, IntoNeighbors, IntoNodeIdentifiers};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Louvain community detection algorithm implementation.
///
/// This algorithm detects communities in a graph by optimizing modularity.
/// It works by iteratively moving nodes between communities to maximize modularity gain.
///
/// # Examples
///
/// ```
/// use petgraph::graph::UnGraph;
/// use petgraph_clustering::{Louvain, CommunityDetection};
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
/// let louvain = Louvain::new();
/// let communities = louvain.detect_communities(&graph);
///
/// // Nodes n1 and n2 should be in the same community
/// assert_eq!(communities[&n1], communities[&n2]);
///
/// // Nodes n3 and n4 should be in the same community
/// assert_eq!(communities[&n3], communities[&n4]);
///
/// // Nodes n1 and n3 should be in different communities
/// assert_ne!(communities[&n1], communities[&n3]);
/// ```
pub struct Louvain {
    max_iterations: usize,
}

impl Default for Louvain {
    fn default() -> Self {
        Self::new()
    }
}

impl Louvain {
    /// Creates a new Louvain community detection algorithm instance.
    ///
    /// Default maximum iterations is set to 100.
    pub fn new() -> Self {
        Self {
            max_iterations: 100,
        }
    }

    /// Creates a new Louvain community detection algorithm instance with
    /// the specified maximum number of iterations.
    pub fn with_max_iterations(max_iterations: usize) -> Self {
        Self { max_iterations }
    }
}

impl<G> CommunityDetection<G> for Louvain
where
    G: EdgeCount + IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Clone,
{
    fn detect_communities(&self, graph: G) -> HashMap<G::NodeId, usize> {
        let mut current_communities: HashMap<G::NodeId, G::NodeId> =
            graph.node_identifiers().map(|u| (u, u)).collect();

        let mut iteration = 0;
        while iteration < self.max_iterations {
            if let Some(new_communities) = louvain_step(graph, &current_communities) {
                current_communities = new_communities;
                iteration += 1;
            } else {
                // No more improvement possible
                break;
            }
        }

        // Normalize community assignments: ensure all nodes in the same community
        // point to the same representative by finding unique community representatives
        let mut community_representatives: HashMap<G::NodeId, G::NodeId> = HashMap::new();

        // Find all unique community IDs (the values in current_communities)
        for community_id in current_communities.values() {
            if !community_representatives.contains_key(community_id) {
                community_representatives.insert(*community_id, *community_id);
            }
        }

        // Assign usize IDs to each unique community
        let mut community_id_map: HashMap<G::NodeId, usize> = HashMap::new();
        for (idx, community_id) in community_representatives.keys().enumerate() {
            community_id_map.insert(*community_id, idx);
        }

        // Map each node to its community's usize ID
        let mut result: HashMap<G::NodeId, usize> = HashMap::new();
        for (node, community_id) in &current_communities {
            result.insert(*node, community_id_map[community_id]);
        }

        renumber_communities(&result)
    }
}

/// Performs one step of the Louvain algorithm for community detection.
///
/// This function iterates through each node in the graph and evaluates the
/// modularity gain of moving the node to one of its neighboring communities.
/// If moving a node increases the overall modularity, the node's community
/// assignment is updated.
///
/// # Arguments
///
/// * `graph` - A reference to a graph `G` implementing `petgraph`'s `EdgeCount`,
///   `IntoNeighbors`, and `IntoNodeIdentifiers` traits. `G::NodeId` must
///   implement `Eq` and `Hash`.
/// * `initial_communities` - Optional initial community assignments. If None,
///   each node is assigned to its own community.
///
/// # Returns
///
/// * `Some(communities)` - If at least one node was moved to a different
///   community (improving modularity), returns a `HashMap` mapping each node's
///   `NodeId` to the `NodeId` representing its assigned community.
/// * `None` - If no node movement improved modularity during this step.
pub fn louvain_step<G>(
    graph: G,
    initial_communities: &HashMap<G::NodeId, G::NodeId>,
) -> Option<HashMap<G::NodeId, G::NodeId>>
where
    G: EdgeCount + IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Clone,
{
    let m = 2.0 * graph.edge_count() as f32;
    if m == 0.0 {
        return None; // Empty graph has no communities to optimize
    }

    let k = graph
        .node_identifiers()
        .map(|u| (u, graph.neighbors(u).count() as f32))
        .collect::<HashMap<_, _>>();

    let mut sigma_total = HashMap::new();
    for (node, degree) in &k {
        let community = &initial_communities[node];
        *sigma_total.entry(*community).or_insert(0.0) += degree;
    }

    let mut communities = initial_communities.clone();
    let mut community_nodes: HashMap<G::NodeId, HashSet<G::NodeId>> = HashMap::new();

    for (node, community) in &communities {
        community_nodes
            .entry(*community)
            .or_insert_with(HashSet::new)
            .insert(*node);
    }

    let mut improve = false;

    for u in graph.node_identifiers() {
        let mut neighboring_communities = HashSet::new();
        for v in graph.neighbors(u) {
            neighboring_communities.insert(communities[&v]);
        }

        let current_community = communities[&u];
        neighboring_communities.remove(&current_community);

        // Find the best community to move to
        let mut best_delta_q = 0.0;
        let mut best_community = None;

        for c in neighboring_communities {
            // Calculate k_in: edges from u to community c
            let mut k_in_new = 0.;
            for v in graph.neighbors(u) {
                if communities[&v] == c {
                    k_in_new += 1.;
                }
            }

            // Calculate k_in_old: edges from u to its current community (excluding u itself)
            let mut k_in_old = 0.;
            for v in graph.neighbors(u) {
                if communities[&v] == current_community && v != u {
                    k_in_old += 1.;
                }
            }

            // Modularity gain formula:
            // ΔQ = [k_in_new - k[u] * sigma_total[c] / m] / m - [k_in_old - k[u] * (sigma_total[current] - k[u]) / m] / m
            let sigma_current = sigma_total[&current_community];
            let delta_q = (k_in_new - k[&u] * sigma_total[&c] / m) / m
                - (k_in_old - k[&u] * (sigma_current - k[&u]) / m) / m;

            if delta_q > best_delta_q {
                best_delta_q = delta_q;
                best_community = Some(c);
            }
        }

        // Move to the best community if it improves modularity
        if let Some(c) = best_community {
            let prev_c = communities[&u];

            // Only update community_nodes when we actually move the node
            community_nodes.get_mut(&prev_c).unwrap().remove(&u);
            community_nodes.get_mut(&c).unwrap().insert(u);

            *sigma_total.get_mut(&c).unwrap() += k[&u];
            *sigma_total.get_mut(&prev_c).unwrap() -= k[&u];
            *communities.get_mut(&u).unwrap() = c;
            improve = true;
        }
    }

    if improve { Some(communities) } else { None }
}

/// Original louvain_step function that initializes each node to its own community.
/// Maintains backward compatibility for existing code.
///
/// # Arguments
///
/// * `graph` - A reference to a graph `G`
///
/// # Returns
///
/// * `Some(communities)` - If at least one node was moved to a different
///   community (improving modularity), returns a `HashMap` mapping each node's
///   `NodeId` to the `NodeId` representing its assigned community.
/// * `None` - If no node movement improved modularity during this step.
pub fn louvain_step_legacy<G>(graph: G) -> Option<HashMap<G::NodeId, G::NodeId>>
where
    G: EdgeCount + IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Clone,
{
    // Initialize each node in its own community
    let initial_communities: HashMap<G::NodeId, G::NodeId> =
        graph.node_identifiers().map(|u| (u, u)).collect();

    louvain_step(graph, &initial_communities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;

    #[test]
    fn test_louvain_simple_graph() {
        // Create a simple graph with two communities
        // To ensure two distinct communities, we need stronger internal connections
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        let n4 = graph.add_node(());
        let n5 = graph.add_node(());
        let n6 = graph.add_node(());

        // Community 1: n1, n2, n3 form a triangle (densely connected)
        graph.add_edge(n1, n2, ());
        graph.add_edge(n1, n3, ());
        graph.add_edge(n2, n3, ());

        // Community 2: n4, n5, n6 form a triangle (densely connected)
        graph.add_edge(n4, n5, ());
        graph.add_edge(n4, n6, ());
        graph.add_edge(n5, n6, ());

        // Single weak connection between communities
        graph.add_edge(n3, n4, ());

        // Detect communities - must use reference to graph
        let louvain = Louvain::new();
        let communities = louvain.detect_communities(&graph);

        // Verify communities
        assert_eq!(
            communities[&n1], communities[&n2],
            "Nodes 1 and 2 should be in the same community"
        );
        assert_eq!(
            communities[&n2], communities[&n3],
            "Nodes 2 and 3 should be in the same community"
        );
        assert_eq!(
            communities[&n4], communities[&n5],
            "Nodes 4 and 5 should be in the same community"
        );
        assert_eq!(
            communities[&n5], communities[&n6],
            "Nodes 5 and 6 should be in the same community"
        );
        assert_ne!(
            communities[&n1], communities[&n4],
            "Nodes from different communities should have different IDs"
        );
    }

    #[test]
    fn test_louvain_empty_graph() {
        // Create an empty graph
        let graph = UnGraph::<(), ()>::new_undirected();

        // Detect communities - must use reference to graph
        let louvain = Louvain::new();
        let communities = louvain.detect_communities(&graph);

        // Verify that we get an empty community assignment
        assert!(
            communities.is_empty(),
            "Empty graph should have no communities"
        );
    }

    #[test]
    fn test_louvain_single_node() {
        // Create a graph with a single node
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let n1 = graph.add_node(());

        // Detect communities - must use reference to graph
        let louvain = Louvain::new();
        let communities = louvain.detect_communities(&graph);

        // Verify that the node is assigned to community 0
        assert_eq!(
            communities[&n1], 0,
            "Single node should be assigned to community 0"
        );
    }

    #[test]
    fn test_louvain_one_community() {
        // Create a graph with one community (complete graph)
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());

        // All nodes are connected to each other
        graph.add_edge(n1, n2, ());
        graph.add_edge(n1, n3, ());
        graph.add_edge(n2, n3, ());

        // Detect communities - must use reference to graph
        let louvain = Louvain::new();
        let communities = louvain.detect_communities(&graph);

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
    fn test_louvain_step_legacy_compatibility() {
        // Create a simple graph
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n1, n2, ());

        // Verify that the legacy function works - must use reference to graph
        let communities = louvain_step_legacy(&graph);
        assert!(
            communities.is_some(),
            "louvain_step_legacy should return Some"
        );

        // Verify louvain_step with manual initialization - must use reference to graph
        let initial_communities: HashMap<_, _> = graph.node_identifiers().map(|u| (u, u)).collect();
        let communities2 = louvain_step(&graph, &initial_communities);

        // Results should be equivalent
        assert_eq!(
            communities, communities2,
            "Both functions should give the same result"
        );
    }
}
