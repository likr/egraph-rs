pub mod quasi_bicliques;

use std::collections::HashSet;
use petgraph::{Graph, EdgeType};
use petgraph::graph::{IndexType, NodeIndex};

#[derive(Clone)]
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

pub fn maximal_biclusters(
    biclusters: &Vec<Bicluster>,
) -> Vec<Bicluster> {
    biclusters.iter()
        .filter(|bicluster1| {
            biclusters.iter().all(|bicluster2| {
                if bicluster1.source.iter().any(|u| !bicluster2.source.contains(u)) {
                    return true
                }
                if bicluster1.target.iter().any(|v| !bicluster2.target.contains(v)) {
                    return true
                }
                false
            })
        })
        .map(|bicluster| bicluster.clone())
        .collect::<Vec<_>>()
}

pub fn filter_by_size<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    biclusters: &Vec<Bicluster>,
    min_size: usize,
) -> Vec<Bicluster> {
    biclusters.iter()
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
