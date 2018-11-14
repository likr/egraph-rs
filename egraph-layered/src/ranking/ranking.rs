use petgraph::prelude::*;
use std::collections::HashMap;

pub trait RankingModule<N, E> {
    fn call(&self, graph: &Graph<N, E, Directed>) -> HashMap<NodeIndex, usize>;
}
