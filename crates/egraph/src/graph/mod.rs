pub type NodeIndex = usize;

pub trait Graph<G> {
    fn data(&self) -> &G;
    fn data_mut(&mut self) -> &mut G;
    fn nodes(&self) -> Box<Iterator<Item = NodeIndex>>;
    fn edges<'a>(&'a self) -> Box<Iterator<Item = (NodeIndex, NodeIndex)> + 'a>;
    fn out_nodes<'a>(&'a self, u: NodeIndex) -> Box<Iterator<Item = NodeIndex> + 'a>;
    fn in_nodes<'a>(&'a self, u: NodeIndex) -> Box<Iterator<Item = NodeIndex> + 'a>;
    fn node_count(&self) -> usize;
    fn edge_count(&self) -> usize;
    fn out_degree<'a>(&'a self, u: NodeIndex) -> usize;
    fn in_degree<'a>(&'a self, u: NodeIndex) -> usize;
}

pub fn degree<G>(graph: &Graph<G>, u: NodeIndex) -> usize {
    graph.in_degree(u) + graph.out_degree(u)
}

pub fn neighbors<'a, G>(graph: &'a Graph<G>, u: NodeIndex) -> Box<Iterator<Item = NodeIndex> + 'a> {
    Box::new(graph.out_nodes(u).chain(graph.in_nodes(u)))
}