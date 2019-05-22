use petgraph::graph::IndexType;
use petgraph::prelude::*;
use std::collections::HashMap;

pub trait RankingModule<N, E, Ix: IndexType> {
    fn call(&self, graph: &Graph<N, E, Directed, Ix>) -> HashMap<NodeIndex<Ix>, usize>;
}
