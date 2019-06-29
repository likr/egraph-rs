pub type NodeIndex = usize;

pub trait Graph<D> {
    fn data(&self) -> &D;
    fn data_mut(&mut self) -> &mut D;
    fn nodes(&self) -> Box<dyn Iterator<Item = NodeIndex>>;
    fn edges<'a>(&'a self) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex)> + 'a>;
    fn out_nodes<'a>(&'a self, u: NodeIndex) -> Box<dyn Iterator<Item = NodeIndex> + 'a>;
    fn in_nodes<'a>(&'a self, u: NodeIndex) -> Box<dyn Iterator<Item = NodeIndex> + 'a>;
    fn node_count(&self) -> usize;
    fn edge_count(&self) -> usize;
    fn out_degree<'a>(&'a self, u: NodeIndex) -> usize;
    fn in_degree<'a>(&'a self, u: NodeIndex) -> usize;
}

pub fn degree<D, G: Graph<D>>(graph: &G, u: NodeIndex) -> usize {
    graph.in_degree(u) + graph.out_degree(u)
}

pub fn neighbors<'a, D, G: Graph<D>>(
    graph: &'a G,
    u: NodeIndex,
) -> Box<dyn Iterator<Item = NodeIndex> + 'a> {
    Box::new(graph.out_nodes(u).chain(graph.in_nodes(u)))
}
