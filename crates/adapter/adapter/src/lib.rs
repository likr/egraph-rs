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
    fn has_edge(&self, u: NodeIndex, v: NodeIndex) -> bool;
}
