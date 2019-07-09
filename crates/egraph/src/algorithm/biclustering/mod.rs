pub mod quasi_biclique;

use crate::Graph;
use std::collections::HashSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bicluster {
    pub source: HashSet<usize>,
    pub target: HashSet<usize>,
}

impl Bicluster {
    pub fn new() -> Bicluster {
        Bicluster {
            source: HashSet::new(),
            target: HashSet::new(),
        }
    }
}

pub trait Biclustering {
    fn call<D, G: Graph<D>>(
        &self,
        graph: &G,
        source: &HashSet<usize>,
        target: &HashSet<usize>,
    ) -> Vec<Bicluster>;
}

pub fn maximal_biclusters(biclusters: &Vec<Bicluster>) -> Vec<Bicluster> {
    biclusters
        .iter()
        .enumerate()
        .filter(|(i, bicluster1)| {
            !biclusters.iter().enumerate().any(|(j, bicluster2)| {
                return *i != j
                    && bicluster2.source.is_superset(&bicluster1.source)
                    && bicluster2.target.is_superset(&bicluster1.target);
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

pub use self::quasi_biclique::QuasiBiclique;
