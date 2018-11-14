use petgraph::graph::NodeIndex;
use petgraph::Graph;
use ranking::RankingModule;
use std::collections::{HashMap, HashSet};

pub struct MinWidthRanking {}

impl MinWidthRanking {
    pub fn new() -> MinWidthRanking {
        MinWidthRanking {}
    }
}

impl<N, E> RankingModule<N, E> for MinWidthRanking {
    fn call(&self, graph: &Graph<N, E>) -> HashMap<NodeIndex, usize> {
        let mut result = HashMap::new();
        let mut remains = graph.node_indices().collect::<HashSet<_>>();
        while !remains.is_empty() {}
        // TODO
        result
    }
}
