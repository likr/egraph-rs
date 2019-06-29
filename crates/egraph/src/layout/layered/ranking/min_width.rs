use super::RankingModule;
use crate::graph::sink_nodes;
use crate::{Graph, NodeIndex};
use std::collections::HashMap;

fn dfs<D, G: Graph<D>>(
    graph: &G,
    layers: &mut HashMap<NodeIndex, usize>,
    widths: &mut HashMap<usize, usize>,
    u: NodeIndex,
    width_limit: usize,
) -> usize {
    if let Some(&layer) = layers.get(&u) {
        return layer;
    }

    let mut max_layer = 0;
    for v in graph.in_nodes(u) {
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

fn min_width<D, G: Graph<D>>(graph: &G, width_limit: usize) -> HashMap<NodeIndex, usize> {
    let mut layering = HashMap::new();
    let mut widths = HashMap::new();
    for u in sink_nodes(graph) {
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
