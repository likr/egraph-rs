use ndarray::prelude::*;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeIdentifiers};
use std::{collections::HashMap, f32::INFINITY, hash::Hash};

pub fn warshall_floyd<G, F>(graph: G, length: &mut F) -> Array2<f32>
where
    G: IntoEdgeReferences + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
    F: FnMut(G::EdgeRef) -> f32,
{
    let indices = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, u)| (u, i))
        .collect::<HashMap<_, _>>();
    let n = indices.len();
    let mut distance = Array::from_elem((n, n), INFINITY);

    for e in graph.edge_references() {
        let i = indices[&e.source()];
        let j = indices[&e.target()];
        let d = length(e);
        distance[[i, j]] = d;
        distance[[j, i]] = d;
    }
    for i in 0..n {
        distance[[i, i]] = 0.;
    }

    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let d = distance[[i, k]] + distance[[k, j]];
                if d < distance[[i, j]] {
                    distance[[i, j]] = d;
                }
            }
        }
    }

    distance
}
