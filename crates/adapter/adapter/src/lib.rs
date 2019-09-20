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

    fn degree(&self, u: NodeIndex) -> usize {
        self.in_degree(u) + self.out_degree(u)
    }

    fn neighbors<'a>(&'a self, u: NodeIndex) -> Box<dyn Iterator<Item = NodeIndex> + 'a> {
        Box::new(self.out_nodes(u).chain(self.in_nodes(u)))
    }

    fn source_nodes<'a>(&'a self) -> Box<dyn Iterator<Item = NodeIndex> + 'a> {
        Box::new(self.nodes().filter(move |&u| self.in_degree(u) == 0))
    }

    fn sink_nodes<'a>(&'a self) -> Box<dyn Iterator<Item = NodeIndex> + 'a> {
        Box::new(self.nodes().filter(move |&u| self.out_degree(u) == 0))
    }

    fn nodes_with_index<'a>(&'a self) -> Box<dyn Iterator<Item = (NodeIndex, usize)> + 'a> {
        Box::new(self.nodes().enumerate().map(|(i, u)| (u, i)))
    }
}
