use crate::graph::{degree, Graph, NodeIndex};
use crate::layout::force_directed::force::link_force::{Link, LinkForceContext};
use crate::layout::force_directed::force::{Force, ForceContext};

pub struct GroupLinkForce<G> {
    pub intra_group: f32,
    pub inter_group: f32,
    pub group: Box<Fn(&Graph<G>, NodeIndex) -> usize>,
    pub distance: Box<Fn(&Graph<G>, NodeIndex, NodeIndex) -> f32>,
}

impl<G> GroupLinkForce<G> {
    pub fn new() -> GroupLinkForce<G> {
        GroupLinkForce {
            inter_group: 0.01,
            intra_group: 0.5,
            group: Box::new(|_, _| 0),
            distance: Box::new(|_, _, _| 30.0),
        }
    }
}

impl<G> Force<G> for GroupLinkForce<G> {
    fn build(&self, graph: &Graph<G>) -> Box<ForceContext> {
        let group_accessor = &self.group;
        let groups = graph
            .nodes()
            .map(|u| group_accessor(graph, u))
            .collect::<Vec<_>>();

        let distance_accessor = &self.distance;
        let links = graph
            .edges()
            .map(|(u, v)| {
                let distance = distance_accessor(graph, u, v);
                let strength = if groups[u] == groups[v] {
                    self.intra_group
                } else {
                    self.inter_group
                };
                let source_degree = degree(graph, u) as f32;
                let target_degree = degree(graph, v) as f32;
                let bias = source_degree / (source_degree + target_degree);
                Link::new(u, v, distance, strength, bias)
            })
            .collect();
        Box::new(LinkForceContext::new(links))
    }
}
