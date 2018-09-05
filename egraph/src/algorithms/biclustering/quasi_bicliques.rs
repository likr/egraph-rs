use std::collections::{HashMap, HashSet};
use petgraph::{Graph, EdgeType};
use petgraph::graph::{IndexType, NodeIndex};
use super::Bicluster;

fn hash_key(vertices: &Vec<usize>) -> String {
    vertices.iter()
        .map(|u| format!("{}", u.index()))
        .collect::<Vec<_>>()
        .join(",")
}

pub fn find_quasi_bicliques<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    source: &HashSet<usize>,
    target: &HashSet<usize>,
    mu: f64
) -> Vec<Bicluster> {
    let mut biclusters = Vec::new();
    let mut keys = HashSet::new();
    for &u in source {
        let mut u_neighbors = graph.neighbors(NodeIndex::new(u))
            .filter(|v| target.contains(&v.index()))
            .map(|v| v.index())
            .collect::<Vec<_>>();
        u_neighbors.sort();
        let key = hash_key(&mut u_neighbors);
        if keys.contains(&key) {
            continue;
        }
        let mut bicluster = Bicluster::new();
        for v in u_neighbors {
            bicluster.target.insert(v);
        }
        biclusters.push(bicluster);
        keys.insert(key);
    }

    for bicluster in biclusters.iter_mut() {
        let mut m = HashMap::new();
        for &v in bicluster.target.iter() {
            for u in graph.neighbors(NodeIndex::new(v)) {
                if !source.contains(&u.index()) {
                    continue;
                }
                if !m.contains_key(&u.index()) {
                    m.insert(u.index(), 0);
                }
                if let Some(count) = m.get_mut(&u.index()) {
                    *count += 1;
                }
            }
        }
        for (u, count) in m {
            if count >= (mu * bicluster.target.len() as f64) as usize {
                bicluster.source.insert(u);
            }
        }
    }

    biclusters
}
