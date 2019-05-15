pub type NodeIndex = usize;

pub trait Graph {
    fn nodes(&self) -> Box<Iterator<Item = NodeIndex>>;
    fn edges<'a>(&'a self) -> Box<Iterator<Item = (NodeIndex, NodeIndex)> + 'a>;
    fn out_nodes<'a>(&'a self, u: NodeIndex) -> Box<Iterator<Item = NodeIndex> + 'a>;
    fn in_nodes<'a>(&'a self, u: NodeIndex) -> Box<Iterator<Item = NodeIndex> + 'a>;
    fn node_count(&self) -> usize;
    fn edge_count(&self) -> usize;
    fn out_degree<'a>(&'a self, u: NodeIndex) -> usize;
    fn in_degree<'a>(&'a self, u: NodeIndex) -> usize;
}
