use petgraph::graph::{node_index, IndexType};
use petgraph::prelude::*;
use petgraph::EdgeType;

pub struct PetgraphWrapper<N, E, Ty: EdgeType, Ix: IndexType> {
    graph: Graph<N, E, Ty, Ix>,
}

impl<N, E, Ty: EdgeType, Ix: IndexType> PetgraphWrapper<N, E, Ty, Ix> {
    pub fn new(graph: Graph<N, E, Ty, Ix>) -> PetgraphWrapper<N, E, Ty, Ix> {
        PetgraphWrapper { graph }
    }
}

impl<N, E, Ty: EdgeType, Ix: IndexType> egraph_adapter::Graph<Graph<N, E, Ty, Ix>>
    for PetgraphWrapper<N, E, Ty, Ix>
{
    fn data(&self) -> &Graph<N, E, Ty, Ix> {
        &self.graph
    }

    fn data_mut(&mut self) -> &mut Graph<N, E, Ty, Ix> {
        &mut self.graph
    }

    fn nodes(&self) -> Box<Iterator<Item = usize>> {
        Box::new(self.graph.node_indices().map(|i| i.index()))
    }

    fn edges<'a>(&'a self) -> Box<Iterator<Item = (usize, usize)> + 'a> {
        Box::new(
            self.graph
                .edge_indices()
                .map(move |e| self.graph.edge_endpoints(e).unwrap())
                .map(|(u, v)| (u.index(), v.index())),
        )
    }

    fn out_nodes<'a>(&'a self, u: usize) -> Box<Iterator<Item = usize> + 'a> {
        Box::new(
            self.graph
                .neighbors_directed(node_index(u), Outgoing)
                .map(|i| i.index()),
        )
    }

    fn in_nodes<'a>(&'a self, u: usize) -> Box<Iterator<Item = usize> + 'a> {
        Box::new(
            self.graph
                .neighbors_directed(node_index(u), Incoming)
                .map(|i| i.index()),
        )
    }

    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    fn out_degree(&self, u: usize) -> usize {
        self.graph
            .neighbors_directed(node_index(u), Outgoing)
            .count()
    }

    fn in_degree(&self, u: usize) -> usize {
        self.graph
            .neighbors_directed(node_index(u), Incoming)
            .count()
    }

    fn has_edge(&self, u: usize, v: usize) -> bool {
        if let Some(_) = self.graph.find_edge(node_index(u), node_index(v)) {
            return true;
        }
        false
    }
}
