use crate::distance_matrix::{DistanceMatrix, FullDistanceMatrix, PivotedDistanceMatrix};
use ndarray::prelude::*;
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers};
use std::{collections::VecDeque, hash::Hash};

/// Performs a Breadth-First Search (BFS) starting from a single source node `s`
/// and populates the corresponding row in the provided `distance_matrix`.
///
/// This function assumes all edge lengths are equal to `unit_edge_length`.
/// It modifies the `distance_matrix` in place.
///
/// # Type Parameters
///
/// * `G`: The graph type, implementing `IntoNeighbors`.
/// * `S`: The scalar type for distances, implementing `NdFloat` (e.g., `f32`, `f64`).
/// * `D`: The distance matrix type, implementing `DistanceMatrix<G::NodeId, S>`.
///
/// # Arguments
///
/// * `graph`: The graph to perform BFS on.
/// * `unit_edge_length`: The length assigned to each edge during the traversal.
/// * `s`: The starting node ID for the BFS.
/// * `distance_matrix`: A mutable reference to the distance matrix to be populated.
///   The distances from `s` to all reachable nodes will be written into the row corresponding to `s`.
pub fn bfs_with_distance_matrix<G, S, D>(
    graph: G,
    unit_edge_length: S,
    s: G::NodeId,
    distance_matrix: &mut D,
) where
    G: IntoNeighbors,
    G::NodeId: Eq + Hash,
    D: DistanceMatrix<G::NodeId, S>,
    S: NdFloat,
{
    let n = distance_matrix.shape().1;
    let k = distance_matrix.row_index(s).unwrap();
    let s_col = distance_matrix.col_index(s).unwrap();
    let mut visited = vec![false; n];
    visited[s_col] = true;
    let mut queue = VecDeque::new();
    queue.push_back(s);
    distance_matrix.set_by_index(k, s_col, S::zero());
    while let Some(u) = queue.pop_front() {
        let i = distance_matrix.col_index(u).unwrap();
        for v in graph.neighbors(u) {
            let j = distance_matrix.col_index(v).unwrap();
            if !visited[j] {
                visited[j] = true;
                queue.push_back(v);
                distance_matrix.set_by_index(
                    k,
                    j,
                    distance_matrix.get_by_index(k, i) + unit_edge_length,
                );
            }
        }
    }
}

/// Computes the shortest path distances from a set of source nodes to all other nodes
/// in the graph using Breadth-First Search (BFS).
///
/// Assumes all edge lengths are equal to `unit_edge_length`.
///
/// # Type Parameters
///
/// * `G`: The graph type, implementing `IntoNeighbors` and `IntoNodeIdentifiers`.
/// * `S`: The scalar type for distances, implementing `NdFloat`.
///
/// # Arguments
///
/// * `graph`: The graph to perform BFS on.
/// * `unit_edge_length`: The length assigned to each edge.
/// * `sources`: A slice of node IDs representing the starting points for the BFS computations.
///
/// # Returns
///
/// A `PivotedDistanceMatrix` containing the shortest path distances from each source node
/// to all nodes present in the `distance_matrix`'s column index mapping.
pub fn multi_source_bfs<G, S>(
    graph: G,
    unit_edge_length: S,
    sources: &[G::NodeId],
) -> PivotedDistanceMatrix<G::NodeId, S>
where
    G: IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    S: NdFloat,
{
    let mut distance_matrix = PivotedDistanceMatrix::new(graph, sources);
    for &u in sources.iter() {
        bfs_with_distance_matrix(graph, unit_edge_length, u, &mut distance_matrix);
    }
    distance_matrix
}

/// Computes the shortest path distances between all pairs of nodes in the graph
/// using Breadth-First Search (BFS).
///
/// Assumes all edge lengths are equal to `unit_edge_length`.
///
/// # Type Parameters
///
/// * `G`: The graph type, implementing `IntoNeighbors` and `IntoNodeIdentifiers`.
/// * `S`: The scalar type for distances, implementing `NdFloat`.
///
/// # Arguments
///
/// * `graph`: The graph to perform BFS on.
/// * `unit_edge_length`: The length assigned to each edge.
///
/// # Returns
///
/// A `FullDistanceMatrix` containing the shortest path distances between all pairs of nodes.
pub fn all_sources_bfs<G, S>(graph: G, unit_edge_length: S) -> FullDistanceMatrix<G::NodeId, S>
where
    G: IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    S: NdFloat,
{
    let mut distance_matrix = FullDistanceMatrix::new(graph);
    for u in graph.node_identifiers() {
        bfs_with_distance_matrix(graph, unit_edge_length, u, &mut distance_matrix);
    }
    distance_matrix
}

/// Computes the shortest path distances from a single source node `s` to all other nodes,
/// assuming a specified unit edge length.
///
/// # Type Parameters
///
/// * `G`: The graph type, implementing `IntoNeighbors` and `IntoNodeIdentifiers`.
/// * `S`: The scalar type for distances, implementing `NdFloat`.
///
/// # Arguments
///
/// * `graph`: The graph to perform BFS on.
/// * `unit_edge_length`: The length assigned to each edge.
/// * `s`: The starting node ID for the BFS.
///
/// # Returns
///
/// A `PivotedDistanceMatrix` containing the shortest path distances from the source node `s`.
pub fn bfs_with_unit_edge_length<G, S>(
    graph: G,
    unit_edge_length: S,
    s: G::NodeId,
) -> PivotedDistanceMatrix<G::NodeId, S>
where
    G: IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    S: NdFloat,
{
    multi_source_bfs(graph, unit_edge_length, &[s])
}

/// Computes the shortest path distances from a single source node `s` to all other nodes,
/// assuming a unit edge length of 1.
///
/// This is a convenience wrapper around `bfs_with_unit_edge_length` with `unit_edge_length = 1`.
///
/// # Type Parameters
///
/// * `G`: The graph type, implementing `IntoNeighbors` and `IntoNodeIdentifiers`.
/// * `S`: The scalar type for distances, implementing `NdFloat`.
///
/// # Arguments
///
/// * `graph`: The graph to perform BFS on.
/// * `s`: The starting node ID for the BFS.
///
/// # Returns
///
/// A `PivotedDistanceMatrix` containing the shortest path distances (hop count) from `s`.
pub fn bfs<G, S>(graph: G, s: G::NodeId) -> PivotedDistanceMatrix<G::NodeId, S>
where
    G: IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    S: NdFloat,
{
    bfs_with_unit_edge_length(graph, S::one(), s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;

    #[test]
    fn test_bfs_with_pivoted_distance_matrix_non_leading_pivots() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        let n2 = g.add_node(());
        let n3 = g.add_node(());
        let n4 = g.add_node(());
        let n5 = g.add_node(());

        g.add_edge(n0, n1, ());
        g.add_edge(n1, n2, ());
        g.add_edge(n2, n3, ());
        g.add_edge(n3, n4, ());
        g.add_edge(n4, n5, ());
        g.add_edge(n1, n4, ());

        // Use pivots not located at node indices 0 or 1 so row index != column index
        // and shape().0 (2) < shape().1 (6).
        let dm = multi_source_bfs(&g, 1.0_f64, &[n3, n5]);
        assert_eq!(dm.shape(), (2, 6));

        let expected_from_n3 = [
            (n0, 3.0),
            (n1, 2.0),
            (n2, 1.0),
            (n3, 0.0),
            (n4, 1.0),
            (n5, 2.0),
        ];
        for (node, dist) in expected_from_n3 {
            assert_eq!(dm.get(n3, node), Some(dist));
        }

        let expected_from_n5 = [
            (n0, 3.0),
            (n1, 2.0),
            (n2, 3.0),
            (n3, 2.0),
            (n4, 1.0),
            (n5, 0.0),
        ];
        for (node, dist) in expected_from_n5 {
            assert_eq!(dm.get(n5, node), Some(dist));
        }

        // Also check scaled unit edge length via direct bfs_with_distance_matrix call.
        let mut scaled = PivotedDistanceMatrix::new(&g, &[n4, n2]);
        bfs_with_distance_matrix(&g, 2.5_f64, n4, &mut scaled);
        bfs_with_distance_matrix(&g, 2.5_f64, n2, &mut scaled);
        assert_eq!(scaled.get(n4, n0), Some(5.0));
        assert_eq!(scaled.get(n4, n4), Some(0.0));
        assert_eq!(scaled.get(n2, n5), Some(7.5));
        assert_eq!(scaled.get(n2, n2), Some(0.0));
    }
}
