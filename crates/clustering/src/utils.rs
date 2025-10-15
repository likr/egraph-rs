use petgraph::visit::{EdgeCount, IntoNeighbors, IntoNodeIdentifiers};
use std::collections::HashMap;
use std::hash::Hash;

/// Calculate modularity for a graph with given community assignments.
///
/// Modularity is a measure of the structure of networks which measures the strength of
/// division of a network into communities/clusters.
///
/// # Arguments
///
/// * `graph` - A reference to a graph implementing required traits
/// * `communities` - A hashmap that maps each node to its community ID
///
/// # Returns
///
/// The modularity value for the given community structure, ranging from -0.5 to 1.0.
/// Higher values indicate better community structure.
pub fn modularity<G>(graph: G, communities: &HashMap<G::NodeId, usize>) -> f64
where
    G: EdgeCount + IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Clone,
{
    let m = graph.edge_count() as f64 * 2.0; // Each edge counted twice in the formula
    if m == 0.0 {
        return 0.0; // Empty graph
    }

    // Calculate the degree of each node
    let degrees: HashMap<_, _> = graph
        .node_identifiers()
        .map(|node| (node, graph.neighbors(node).count() as f64))
        .collect();

    // Calculate the sum of the degrees of nodes in each community
    let mut community_degrees: HashMap<usize, f64> = HashMap::new();
    for (node, degree) in &degrees {
        let community = communities.get(node).unwrap_or(&0);
        *community_degrees.entry(*community).or_insert(0.0) += degree;
    }

    let mut q = 0.0;

    // For each pair of nodes, calculate contribution to modularity
    for i in graph.node_identifiers() {
        for j in graph.neighbors(i) {
            // Check if nodes are in the same community
            let ci = communities.get(&i).unwrap_or(&0);
            let cj = communities.get(&j).unwrap_or(&0);

            if ci == cj {
                // Count the edge contribution to modularity
                let ki = degrees.get(&i).unwrap_or(&0.0);
                let kj = degrees.get(&j).unwrap_or(&0.0);

                // Actual connection - expected connection
                q += 1.0 - (ki * kj) / m;
            }
        }
    }

    q / m
}

/// Renumber communities from 0 to n-1, where n is the number of communities.
///
/// This function is useful when the community IDs are not consecutive or do not start from 0.
///
/// # Arguments
///
/// * `communities` - A hashmap that maps each node to its community ID
///
/// # Returns
///
/// A new hashmap with renumbered community IDs
pub fn renumber_communities<T>(communities: &HashMap<T, usize>) -> HashMap<T, usize>
where
    T: Eq + Hash + Clone,
{
    let mut id_map = HashMap::new();
    let mut next_id = 0;

    let mut result = HashMap::new();

    for (node, &old_id) in communities {
        let new_id = match id_map.get(&old_id) {
            Some(&id) => id,
            None => {
                let id = next_id;
                id_map.insert(old_id, id);
                next_id += 1;
                id
            }
        };

        result.insert(node.clone(), new_id);
    }

    result
}
