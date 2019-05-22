pub mod quasi_biclique;

use petgraph::graph::{IndexType, NodeIndex};
use petgraph::{EdgeType, Graph};
use std::collections::HashSet;

#[derive(Clone, Debug)]
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
    fn call<N, E, Ty: EdgeType, Ix: IndexType>(
        &self,
        graph: &Graph<N, E, Ty, Ix>,
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

pub fn filter_by_size<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    biclusters: &Vec<Bicluster>,
    min_size: usize,
) -> Vec<Bicluster> {
    biclusters
        .iter()
        .filter(|bicluster| {
            let mut size = 0;
            for &u in bicluster.source.iter() {
                for &v in bicluster.target.iter() {
                    if let Some(_) = graph.find_edge(NodeIndex::new(u), NodeIndex::new(v)) {
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
