use super::force::{Force, ForceContext};
use super::link_force::{Link, LinkForceContext};
use petgraph::graph::IndexType;
use petgraph::prelude::*;
use petgraph::EdgeType;

pub struct GroupLinkForce<N, E, Ty: EdgeType, Ix: IndexType> {
    pub intra_group: f32,
    pub inter_group: f32,
    pub group: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> usize>,
    pub distance: Box<Fn(&Graph<N, E, Ty, Ix>, EdgeIndex<Ix>) -> f32>,
}

impl<N, E, Ty: EdgeType, Ix: IndexType> GroupLinkForce<N, E, Ty, Ix> {
    pub fn new() -> GroupLinkForce<N, E, Ty, Ix> {
        GroupLinkForce {
            inter_group: 0.5,
            intra_group: 0.01,
            group: Box::new(|_, _| 0),
            distance: Box::new(|_, _| 30.0),
        }
    }
}

impl<N, E, Ty: EdgeType, Ix: IndexType> Force<N, E, Ty, Ix> for GroupLinkForce<N, E, Ty, Ix> {
    fn build(&self, graph: &Graph<N, E, Ty, Ix>) -> Box<ForceContext> {
        let group_accessor = &self.group;
        let groups = graph
            .node_indices()
            .map(|a| group_accessor(graph, a))
            .collect::<Vec<_>>();

        let distance_accessor = &self.distance;
        let links = graph
            .edge_indices()
            .map(|e| {
                let (source, target) = graph.edge_endpoints(e).unwrap();
                let distance = distance_accessor(graph, e);
                let strength = if groups[source.index()] == groups[target.index()] {
                    self.intra_group
                } else {
                    self.inter_group
                };
                let source_degree = graph.neighbors_undirected(source).count() as f32;
                let target_degree = graph.neighbors_undirected(target).count() as f32;
                let bias = source_degree / (source_degree + target_degree);
                Link::new(source.index(), target.index(), distance, strength, bias)
            })
            .collect();
        Box::new(LinkForceContext::new(links))
    }
}
