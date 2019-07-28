pub mod quasi_biclique;

use crate::Graph;
use std::collections::HashSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bicluster {
    pub source: Vec<usize>,
    pub target: Vec<usize>,
}

impl Bicluster {
    pub fn new() -> Bicluster {
        Bicluster {
            source: Vec::new(),
            target: Vec::new(),
        }
    }
}

pub fn maximal_biclusters(biclusters: &Vec<Bicluster>) -> Vec<Bicluster> {
    let source_set = biclusters
        .iter()
        .map(|bicluster| bicluster.source.iter().collect::<HashSet<_>>())
        .collect::<Vec<_>>();
    let target_set = biclusters
        .iter()
        .map(|bicluster| bicluster.target.iter().collect::<HashSet<_>>())
        .collect::<Vec<_>>();
    biclusters
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            !biclusters.iter().enumerate().any(|(j, _)| {
                return *i != j
                    && source_set[j].is_superset(&source_set[*i])
                    && target_set[j].is_superset(&target_set[*i]);
            })
        })
        .map(|(_, bicluster)| bicluster.clone())
        .collect::<Vec<_>>()
}

pub fn filter_by_size<D, G: Graph<D>>(
    graph: &G,
    biclusters: &Vec<Bicluster>,
    min_size: usize,
) -> Vec<Bicluster> {
    biclusters
        .iter()
        .filter(|bicluster| {
            let mut size = 0;
            for &u in bicluster.source.iter() {
                for &v in bicluster.target.iter() {
                    if graph.has_edge(u, v) {
                        size += 1;
                    }
                }
            }
            size >= min_size
        })
        .map(|bicluster| bicluster.clone())
        .collect::<Vec<_>>()
}

pub use self::quasi_biclique::mu_quasi_bicliques;
