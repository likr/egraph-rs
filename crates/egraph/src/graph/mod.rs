pub use egraph_adapter::{Graph, NodeIndex};

pub fn degree<D, G: Graph<D>>(graph: &G, u: NodeIndex) -> usize {
    graph.in_degree(u) + graph.out_degree(u)
}

pub fn neighbors<'a, D, G: Graph<D>>(
    graph: &'a G,
    u: NodeIndex,
) -> Box<dyn Iterator<Item = NodeIndex> + 'a> {
    Box::new(graph.out_nodes(u).chain(graph.in_nodes(u)))
}

pub fn source_nodes<'a, D, G: Graph<D>>(graph: &'a G) -> Box<dyn Iterator<Item = NodeIndex> + 'a> {
    Box::new(graph.nodes().filter(move |&u| graph.in_degree(u) == 0))
}

pub fn sink_nodes<'a, D, G: Graph<D>>(graph: &'a G) -> Box<dyn Iterator<Item = NodeIndex> + 'a> {
    Box::new(graph.nodes().filter(move |&u| graph.out_degree(u) == 0))
}
