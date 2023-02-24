use ndarray::prelude::*;
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers};
use std::{
    collections::{HashMap, VecDeque},
    f32::INFINITY,
    hash::Hash,
};

pub fn bfs_with_distance_matrix<G>(
    graph: G,
    indices: &HashMap<G::NodeId, usize>,
    unit_edge_length: f32,
    s: G::NodeId,
    distance_matrix: &mut Array2<f32>,
    k: usize,
) where
    G: IntoNeighbors,
    G::NodeId: Eq + Hash + Ord,
{
    let mut visited = vec![false; indices.len()];
    visited[indices[&s]] = true;
    let mut queue = VecDeque::new();
    queue.push_back(s);
    distance_matrix[[indices[&s], k]] = 0.;
    while let Some(u) = queue.pop_front() {
        for v in graph.neighbors(u) {
            if !visited[indices[&v]] {
                visited[indices[&v]] = true;
                queue.push_back(v);
                distance_matrix[[indices[&v], k]] =
                    distance_matrix[[indices[&u], k]] + unit_edge_length;
            }
        }
    }
}

pub fn multi_source_bfs<G>(graph: G, unit_edge_length: f32, sources: &[G::NodeId]) -> Array2<f32>
where
    G: IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Ord,
{
    let indices = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, u)| (u, i))
        .collect::<HashMap<_, _>>();
    let n = indices.len();
    let k = sources.len();
    let mut distance_matrix = Array::from_elem((n, k), INFINITY);
    for c in 0..k {
        bfs_with_distance_matrix(
            graph,
            &indices,
            unit_edge_length,
            sources[c],
            &mut distance_matrix,
            c,
        );
    }
    distance_matrix
}

pub fn all_sources_bfs<G>(graph: G, unit_edge_length: f32) -> Array2<f32>
where
    G: IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Ord,
{
    let sources = graph.node_identifiers().collect::<Vec<_>>();
    multi_source_bfs(graph, unit_edge_length, &sources)
}

pub fn bfs_with_unit_edge_length<G>(graph: G, unit_edge_length: f32, s: G::NodeId) -> Array1<f32>
where
    G: IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Ord,
{
    let distance_matrix = multi_source_bfs(graph, unit_edge_length, &[s]);
    let n = distance_matrix.shape()[0];
    distance_matrix.into_shape(n).unwrap()
}

pub fn bfs<G>(graph: G, s: G::NodeId) -> Array1<f32>
where
    G: IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Ord,
{
    bfs_with_unit_edge_length(graph, 1., s)
}
