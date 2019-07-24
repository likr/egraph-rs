use super::RankingModule;
use crate::algorithm::ranking::min_width;
use crate::{Graph, NodeIndex};
use std::collections::HashMap;

pub struct MinWidthRanking {
    pub width_limit: usize,
}

impl MinWidthRanking {
    pub fn new() -> MinWidthRanking {
        MinWidthRanking { width_limit: 10 }
    }
}

impl<D, G: Graph<D>> RankingModule<D, G> for MinWidthRanking {
    fn call(&self, graph: &G) -> HashMap<NodeIndex, usize> {
        min_width(graph, self.width_limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_petgraph_adapter::PetgraphWrapper;
    use petgraph::Graph;

    #[test]
    fn test_min_width_layering() {
        let mut graph = Graph::<_, _>::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        let e = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(d, c, ());
        graph.add_edge(d, e, ());
        let graph = PetgraphWrapper::new(graph);
        let layers = min_width(&graph, 1);
        assert_eq!(layers[&a.index()], 1);
        assert_eq!(layers[&b.index()], 2);
        assert_eq!(layers[&c.index()], 3);
        assert_eq!(layers[&d.index()], 0);
        assert_eq!(layers[&e.index()], 4);
    }
}
