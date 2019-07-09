use crate::algorithm::biclustering::{filter_by_size, maximal_biclusters, Bicluster, Biclustering};
use crate::graph::{neighbors, Graph};
use std::collections::{HashMap, HashSet};

fn hash_key(vertices: &Vec<usize>) -> String {
    vertices
        .iter()
        .map(|u| format!("{}", u))
        .collect::<Vec<_>>()
        .join(",")
}

pub fn find_quasi_bicliques<D, G: Graph<D>>(
    graph: &G,
    source: &HashSet<usize>,
    target: &HashSet<usize>,
    mu: f64,
) -> Vec<Bicluster> {
    let mut biclusters = Vec::new();
    let mut keys = HashSet::new();
    for &u in source {
        let mut u_neighbors = neighbors(graph, u)
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
            bicluster.target.insert(v);
        }
        biclusters.push(bicluster);
        keys.insert(key);
    }

    for bicluster in biclusters.iter_mut() {
        let mut m = HashMap::new();
        for &v in bicluster.target.iter() {
            for u in neighbors(graph, v) {
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
                bicluster.source.insert(u);
            }
        }
    }

    biclusters
}

pub struct QuasiBiclique {
    pub mu: f64,
    pub min_size: usize,
}

impl QuasiBiclique {
    pub fn new() -> QuasiBiclique {
        QuasiBiclique {
            mu: 0.5,
            min_size: 4,
        }
    }
}

impl Biclustering for QuasiBiclique {
    fn call<D, G: Graph<D>>(
        &self,
        graph: &G,
        source: &HashSet<usize>,
        target: &HashSet<usize>,
    ) -> Vec<Bicluster> {
        let biclusters = find_quasi_bicliques(graph, source, target, self.mu);
        let biclusters = filter_by_size(graph, &biclusters, self.min_size);
        maximal_biclusters(&biclusters)
    }
}
