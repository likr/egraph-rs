use crate::{Graph, NodeIndex};
use std::collections::HashMap;

pub trait RankingModule<D, G: Graph<D>> {
    fn call(&self, graph: &G) -> HashMap<NodeIndex, usize>;
}
