use super::force::{Force, ForceContext};
use super::link_force::{Link, LinkForceContext};
use egraph_graph::degree;
use egraph_interface::{Graph, NodeIndex};

pub struct GroupLinkForce {
    pub intra_group: f32,
    pub inter_group: f32,
    pub group: Box<Fn(&Graph, NodeIndex) -> usize>,
    pub distance: Box<Fn(&Graph, NodeIndex, NodeIndex) -> f32>,
}

impl GroupLinkForce {
    pub fn new() -> GroupLinkForce {
        GroupLinkForce {
            inter_group: 0.01,
            intra_group: 0.5,
            group: Box::new(|_, _| 0),
            distance: Box::new(|_, _, _| 30.0),
        }
    }
}

impl Force for GroupLinkForce {
    fn build(&self, graph: &Graph) -> Box<ForceContext> {
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
