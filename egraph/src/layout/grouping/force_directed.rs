use super::{Group, Grouping};
use layout::force_directed::force::{CenterForce, CollideForce, LinkForce, ManyBodyForce};
use layout::force_directed::initial_placement;
use layout::force_directed::simulation::Simulation;
use petgraph::graph::IndexType;
use petgraph::prelude::*;
use petgraph::EdgeType;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub struct GroupNode {
    pub group: usize,
    pub size: f32,
    radius: f32,
}

impl GroupNode {
    fn new(group: usize, size: f32) -> Self {
        GroupNode {
            group,
            size,
            radius: size.sqrt(),
        }
    }
}

pub struct GroupLink {
    pub weight: f32,
}

impl GroupLink {
    fn new(weight: f32) -> Self {
        GroupLink { weight }
    }
}

pub struct ForceDirectedGrouping<N, E, Ty: EdgeType, Ix: IndexType> {
    pub group: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> usize>,
    pub node_size: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32>,
    pub link_weight: Box<Fn(&Graph<N, E, Ty, Ix>, EdgeIndex<Ix>) -> f32>,
    pub link_force_strength: Box<Fn(&Graph<GroupNode, GroupLink>, EdgeIndex) -> f32>,
    pub many_body_force_strength: Box<Fn(&Graph<GroupNode, GroupLink>, NodeIndex) -> f32>,
}

impl<N, E, Ty: EdgeType, Ix: IndexType> ForceDirectedGrouping<N, E, Ty, Ix> {
    pub fn new() -> ForceDirectedGrouping<N, E, Ty, Ix> {
        ForceDirectedGrouping {
            group: Box::new(|_, _| 0),
            node_size: Box::new(|_, _| 1000.),
            link_weight: Box::new(|_, _| 1.),
            link_force_strength: Box::new(|graph, e| (graph[e].weight + 1.).ln()),
            many_body_force_strength: Box::new(|graph, a| -graph[a].size),
        }
    }
}

impl<N, E, Ty: EdgeType, Ix: IndexType> Grouping<N, E, Ty, Ix>
    for ForceDirectedGrouping<N, E, Ty, Ix>
{
    fn call(
        &self,
        graph: &Graph<N, E, Ty, Ix>,
        _width: f32,
        _height: f32,
    ) -> HashMap<usize, Group> {
        let values_map = self.group_size(graph, &self.group, &self.node_size);

        let mut group_ids = HashMap::new();
        let mut group_graph = Graph::new();
        for (&group, &size) in values_map.iter() {
            let index = group_graph.add_node(GroupNode::new(group, size));
            group_ids.insert(group, index);
        }
        let mut group_links = HashMap::new();
        for e in graph.edge_indices() {
            let (source, target) = graph.edge_endpoints(e).unwrap();
            let key = {
                let source_group = (self.group)(graph, source);
                let target_group = (self.group)(graph, target);
                if source_group == target_group {
                    continue;
                }
                if source_group < target_group {
                    (source_group, target_group)
                } else {
                    (target_group, source_group)
                }
            };
            if !group_links.contains_key(&key) {
                group_links.insert(key, 0.);
            }
            *group_links.get_mut(&key).unwrap() += (self.link_weight)(graph, e);
        }
        for ((g1, g2), count) in group_links {
            group_graph.add_edge(
                *group_ids.get(&g1).unwrap(),
                *group_ids.get(&g2).unwrap(),
                GroupLink::new(count),
            );
        }

        let many_body_force_strength = group_graph
            .node_indices()
            .map(|a| (self.many_body_force_strength)(&group_graph, a))
            .collect::<Vec<_>>();
        let link_force_strength = group_graph
            .edge_indices()
            .map(|e| (self.link_force_strength)(&group_graph, e))
            .collect::<Vec<_>>();

        let many_body_force = Rc::new(RefCell::new(ManyBodyForce::new()));
        many_body_force.borrow_mut().strength =
            Box::new(move |_, a| many_body_force_strength[a.index()]);
        let link_force = Rc::new(RefCell::new(LinkForce::new()));
        link_force.borrow_mut().distance = Box::new(|g: &Graph<GroupNode, GroupLink>, e| {
            let (s, t) = g.edge_endpoints(e).unwrap();
            g[s].radius + g[t].radius
        });
        link_force.borrow_mut().strength = Box::new(move |_, e| link_force_strength[e.index()]);
        let collide_force = Rc::new(RefCell::new(CollideForce::new()));
        collide_force.borrow_mut().radius =
            Box::new(|g: &Graph<GroupNode, GroupLink>, a| g[a].radius);
        let center_force = Rc::new(RefCell::new(CenterForce::new()));
        let mut simulation = Simulation::new();
        simulation.add(many_body_force);
        simulation.add(link_force);
        simulation.add(collide_force);
        simulation.add(center_force);

        let mut context = simulation.build(&group_graph);
        let mut points = initial_placement(group_graph.node_count());
        context.start(&mut points);

        let mut result = HashMap::new();
        for g in group_graph.node_indices() {
            result.insert(
                group_graph[g].group,
                Group::new(
                    points[g.index()].x,
                    points[g.index()].y,
                    group_graph[g].radius,
                    group_graph[g].radius,
                ),
            );
        }
        result
    }
}
