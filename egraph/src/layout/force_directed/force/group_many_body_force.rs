use super::force::{Force, ForceContext, Point};
use super::group_indices;
use petgraph::graph::IndexType;
use petgraph::prelude::*;
use petgraph::EdgeType;
use std::collections::HashMap;

pub struct GroupManyBodyForceContext {
    groups: HashMap<usize, Vec<usize>>,
    strength: Vec<f32>,
}

impl GroupManyBodyForceContext {
    pub fn new(
        groups: HashMap<usize, Vec<usize>>,
        strength: Vec<f32>,
    ) -> GroupManyBodyForceContext {
        GroupManyBodyForceContext { groups, strength }
    }
}

impl ForceContext for GroupManyBodyForceContext {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        for indices in self.groups.values() {
            let n = indices.len();
            for i in 0..n {
                let a = indices[i];
                let x0 = points[a].x;
                let y0 = points[a].y;
                let mut dvx = 0.;
                let mut dvy = 0.;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let b = indices[j];
                    let strength = self.strength[b];
                    let dx = points[b].x - x0;
                    let dy = points[b].y - y0;
                    let l = (dx * dx + dy * dy).max(1e-6);
                    dvx += dx * strength * alpha / l;
                    dvy += dy * strength * alpha / l;
                }
                points[a].vx += dvx;
                points[a].vy += dvy
            }
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

        let groups = {
            let group_accessor = &self.group;
            let groups = graph
                .node_indices()
                .map(|a| group_accessor(graph, a))
                .collect::<Vec<_>>();
            group_indices(&groups)
        };

        Box::new(GroupManyBodyForceContext::new(groups, strength))
    }
}
