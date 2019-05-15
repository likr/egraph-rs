extern crate egraph_interface;

use egraph_interface::{Graph, NodeIndex};

pub fn degree(graph: &Graph, u: NodeIndex) -> usize {
    graph.in_degree(u) + graph.out_degree(u)
}

pub fn neighbors<'a>(graph: &'a Graph, u: NodeIndex) -> Box<Iterator<Item = NodeIndex> + 'a> {
    Box::new(graph.out_nodes(u).chain(graph.in_nodes(u)))
}
