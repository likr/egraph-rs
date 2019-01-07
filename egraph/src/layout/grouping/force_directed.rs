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

struct GroupNode {
    group: usize,
    size: f32,
}

impl GroupNode {
    fn new(group: usize, size: f32) -> Self {
        GroupNode { group, size }
    }
}

struct GroupLink {
    count: usize,
}

impl GroupLink {
    fn new(count: usize) -> Self {
        GroupLink { count }
    }
}

pub struct ForceDirectedGrouping<N, E, Ty: EdgeType, Ix: IndexType> {
    pub group: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> usize>,
    pub size: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32>,
}

impl<N, E, Ty: EdgeType, Ix: IndexType> ForceDirectedGrouping<N, E, Ty, Ix> {
    pub fn new() -> ForceDirectedGrouping<N, E, Ty, Ix> {
        ForceDirectedGrouping {
            group: Box::new(|_, _| 0),
            size: Box::new(|_, _| 1.),
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
        let values_map = self.group_size(graph, &self.group, &self.size);

        let mut group_ids = HashMap::new();
        let mut group_graph = Graph::new();
        for (&group, &size) in values_map.iter() {
            let index = group_graph.add_node(GroupNode::new(group, size.sqrt()));
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
                group_links.insert(key, 0);
            }
            *group_links.get_mut(&key).unwrap() += 1;
        }
        for ((g1, g2), count) in group_links {
            group_graph.add_edge(
                *group_ids.get(&g1).unwrap(),
                *group_ids.get(&g2).unwrap(),
                GroupLink::new(count),
            );
        }

        let many_body_force = Rc::new(RefCell::new(ManyBodyForce::new()));
        many_body_force.borrow_mut().strength = Box::new(|_, _| -1000.);
        let link_force = Rc::new(RefCell::new(LinkForce::new()));
        link_force.borrow_mut().distance = Box::new(|g: &Graph<GroupNode, GroupLink>, e| {
            let (s, t) = g.edge_endpoints(e).unwrap();
            g[s].size + g[t].size
        });
        let collide_force = Rc::new(RefCell::new(CollideForce::new()));
        collide_force.borrow_mut().radius =
            Box::new(|g: &Graph<GroupNode, GroupLink>, a| g[a].size);
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
                    group_graph[g].size,
                    group_graph[g].size,
                ),
            );
        }
        result
    }
}
