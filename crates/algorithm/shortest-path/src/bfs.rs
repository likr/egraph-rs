use crate::distance_matrix::{DistanceMatrix, FullDistanceMatrix, SubDistanceMatrix};
use ndarray::prelude::*;
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers};
use std::{collections::VecDeque, hash::Hash};

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
    let n = distance_matrix.shape().0;
    let k = distance_matrix.row_index(s).unwrap();
    let mut visited = vec![false; n];
    visited[k] = true;
    let mut queue = VecDeque::new();
    queue.push_back(s);
    distance_matrix.set(s, s, S::zero());
    while let Some(u) = queue.pop_front() {
        let i = distance_matrix.row_index(u).unwrap();
        for v in graph.neighbors(u) {
            let j = distance_matrix.row_index(v).unwrap();
            if !visited[j] {
                visited[j] = true;
                queue.push_back(v);
                distance_matrix.set_by_index(
                    j,
                    k,
                    distance_matrix.get_by_index(i, k) + unit_edge_length,
                );
            }
        }
    }
}

pub fn multi_source_bfs<G, S>(
    graph: G,
    unit_edge_length: S,
    sources: &[G::NodeId],
) -> SubDistanceMatrix<G::NodeId, S>
where
    G: IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    S: NdFloat,
{
    let mut distance_matrix = SubDistanceMatrix::new(graph, sources);
    for &u in sources.iter() {
        bfs_with_distance_matrix(graph, unit_edge_length, u, &mut distance_matrix);
    }
    distance_matrix
}

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

pub fn bfs_with_unit_edge_length<G, S>(
    graph: G,
    unit_edge_length: S,
    s: G::NodeId,
) -> SubDistanceMatrix<G::NodeId, S>
where
    G: IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    S: NdFloat,
{
    multi_source_bfs(graph, unit_edge_length, &[s])
}

pub fn bfs<G, S>(graph: G, s: G::NodeId) -> SubDistanceMatrix<G::NodeId, S>
where
    G: IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    S: NdFloat,
{
    bfs_with_unit_edge_length(graph, S::one(), s)
}
