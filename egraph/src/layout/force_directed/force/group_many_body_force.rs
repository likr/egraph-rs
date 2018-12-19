use super::force::{Force, ForceContext, Point};
use super::many_body_force::ManyBodyForceContext;
use petgraph::graph::IndexType;
use petgraph::prelude::*;
use petgraph::EdgeType;
use std::collections::HashMap;

pub struct GroupManyBodyForceContext {
    groups: HashMap<usize, Vec<usize>>,
    contexts: HashMap<usize, ManyBodyForceContext>,
}

impl GroupManyBodyForceContext {
    pub fn new(
        groups: HashMap<usize, Vec<usize>>,
        contexts: HashMap<usize, ManyBodyForceContext>,
    ) -> GroupManyBodyForceContext {
        GroupManyBodyForceContext { groups, contexts }
    }
}

impl ForceContext for GroupManyBodyForceContext {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        for k in self.groups.keys() {
            let group = self.groups.get(k).unwrap();
            let mut group_points = group.iter().map(|&i| points[i]).collect();
            let context = self.contexts.get(k).unwrap();
            context.apply(&mut group_points, alpha);
        }
    }
}

pub struct GroupManyBodyForce<N, E, Ty: EdgeType, Ix: IndexType> {
    pub strength: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32>,
    pub group: Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> usize>,
}

impl<N, E, Ty: EdgeType, Ix: IndexType> GroupManyBodyForce<N, E, Ty, Ix> {
    pub fn new() -> GroupManyBodyForce<N, E, Ty, Ix> {
        GroupManyBodyForce {
            strength: Box::new(|_, _| -30.),
            group: Box::new(|_, _| 0),
        }
    }
}

impl<N, E, Ty: EdgeType, Ix: IndexType> Force<N, E, Ty, Ix> for GroupManyBodyForce<N, E, Ty, Ix> {
    fn build(&self, graph: &Graph<N, E, Ty, Ix>) -> Box<ForceContext> {
        let strength_accessor = &self.strength;
        let strength = graph
            .node_indices()
            .map(|a| strength_accessor(graph, a))
            .collect::<Vec<_>>();

        let group_accessor = &self.group;
        let mut groups = HashMap::new();
        for a in graph.node_indices() {
            let g = group_accessor(graph, a);
            if !groups.contains_key(&g) {
                groups.insert(g, Vec::new());
            }
            let ids = groups.get_mut(&g).unwrap();
            ids.push(a.index());
        }

        let mut contexts = HashMap::new();
        for (&g, ids) in &groups {
            contexts.insert(
                g,
                ManyBodyForceContext::new(ids.iter().map(|&i| strength[i]).collect()),
            );
        }

        Box::new(GroupManyBodyForceContext::new(groups, contexts))
    }
}
