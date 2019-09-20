use crate::algorithm::biclustering::Bicluster;
use crate::Graph;
use std::collections::{HashMap, HashSet};

fn hash_key(vertices: &Vec<usize>) -> String {
    vertices
        .iter()
        .map(|u| format!("{}", u))
        .collect::<Vec<_>>()
        .join(",")
}

pub fn mu_quasi_bicliques<D, G: Graph<D>>(
    graph: &G,
    source: &Vec<usize>,
    target: &Vec<usize>,
    mu: f64,
) -> Vec<Bicluster> {
    let mut biclusters = Vec::new();
    let mut keys = HashSet::new();
    for &u in source {
        let mut u_neighbors = graph
            .neighbors(u)
            .filter(|v| target.contains(&v))
            .map(|v| v)
            .collect::<Vec<_>>();
        u_neighbors.sort();
        let key = hash_key(&mut u_neighbors);
        if keys.contains(&key) {
            continue;
        }
        let mut bicluster = Bicluster::new();
        for v in u_neighbors {
            bicluster.target.push(v);
        }
        biclusters.push(bicluster);
        keys.insert(key);
    }

    for bicluster in biclusters.iter_mut() {
        let mut m = HashMap::new();
        for &v in bicluster.target.iter() {
            for u in graph.neighbors(v) {
                if !source.contains(&u) {
                    continue;
                }
                if !m.contains_key(&u) {
                    m.insert(u, 0);
                }
                if let Some(count) = m.get_mut(&u) {
                    *count += 1;
                }
            }
        }
        for (u, count) in m {
            if count >= (mu * bicluster.target.len() as f64) as usize {
                bicluster.source.push(u);
            }
        }
    }

    biclusters
}
