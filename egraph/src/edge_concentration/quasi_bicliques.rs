use std::collections::HashSet;
use petgraph::{Graph, EdgeType};
use petgraph::graph::{IndexType};
use ::algorithms::biclustering::{Bicluster, maximal_biclusters, filter_by_size};
use ::algorithms::biclustering::quasi_bicliques::find_quasi_bicliques;

pub struct QuasiBicliqueEdgeConcentration {
    pub mu: f64,
    pub min_size: usize,
}

impl QuasiBicliqueEdgeConcentration {
    pub fn new() -> QuasiBicliqueEdgeConcentration {
        QuasiBicliqueEdgeConcentration {
            mu: 0.5,
            min_size: 4,
        }
    }

    pub fn call<N, E, Ty: EdgeType, Ix: IndexType>(
        &self,
        graph: &Graph<N, E, Ty, Ix>,
        source: &HashSet<usize>,
        target: &HashSet<usize>,
    ) -> Vec<Bicluster> {
        let biclusters = find_quasi_bicliques(graph, source, target, self.mu);
        let biclusters = filter_by_size(graph, &biclusters, self.min_size);
        maximal_biclusters(&biclusters)
    }
}
