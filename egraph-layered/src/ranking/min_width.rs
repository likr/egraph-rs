use petgraph::graph::{IndexType, NodeIndex};
use petgraph::{Directed, Direction, Graph};
use ranking::RankingModule;
use std::collections::HashMap;

fn dfs<N, E, Ix: IndexType>(
    graph: &Graph<N, E, Directed, Ix>,
    layers: &mut HashMap<NodeIndex<Ix>, usize>,
    widths: &mut HashMap<usize, usize>,
    u: NodeIndex<Ix>,
    width_limit: usize,
) -> usize {
    if let Some(&layer) = layers.get(&u) {
        return layer;
    }

    let mut max_layer = 0;
    for v in graph.neighbors_directed(u, Direction::Incoming) {
        let layer = dfs(graph, layers, widths, v, width_limit);
        if layer >= max_layer {
            max_layer = layer + 1;
        }
    }
    while let Some(&width) = widths.get(&max_layer) {
        if width < width_limit {
            break;
        }
        max_layer += 1;
    }
    if !widths.contains_key(&max_layer) {
        widths.insert(max_layer, 0);
    }
    *widths.get_mut(&max_layer).unwrap() += 1;
    layers.insert(u, max_layer);
    max_layer
}

fn min_width<N, E, Ix: IndexType>(
    graph: &Graph<N, E, Directed, Ix>,
    width_limit: usize,
) -> HashMap<NodeIndex<Ix>, usize> {
    let mut layering = HashMap::new();
    let mut widths = HashMap::new();
    for u in graph.externals(Direction::Outgoing) {
        dfs(graph, &mut layering, &mut widths, u, width_limit);
    }
    layering
}

pub struct MinWidthRanking {
    pub width_limit: usize,
}

impl MinWidthRanking {
    pub fn new() -> MinWidthRanking {
        MinWidthRanking { width_limit: 10 }
    }
}

impl<N, E, Ix: IndexType> RankingModule<N, E, Ix> for MinWidthRanking {
    fn call(&self, graph: &Graph<N, E, Directed, Ix>) -> HashMap<NodeIndex<Ix>, usize> {
        min_width(graph, self.width_limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;

    #[test]
    fn test_min_width_layering() {
        let mut graph = Graph::<(), ()>::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        let e = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(d, c, ());
        graph.add_edge(d, e, ());
        let layers = min_width(&graph, 1);
        assert_eq!(*layers.get(&a).unwrap(), 1);
        assert_eq!(*layers.get(&b).unwrap(), 2);
        assert_eq!(*layers.get(&c).unwrap(), 3);
        assert_eq!(*layers.get(&d).unwrap(), 0);
        assert_eq!(*layers.get(&e).unwrap(), 4);
    }
}
