use super::force::{Force, ForceContext, Point};
use super::group_indices;
use egraph_interface::{Graph, NodeIndex};
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

pub struct GroupManyBodyForce {
    pub strength: Box<Fn(&Graph, NodeIndex) -> f32>,
    pub group: Box<Fn(&Graph, NodeIndex) -> usize>,
}

impl GroupManyBodyForce {
    pub fn new() -> GroupManyBodyForce {
        GroupManyBodyForce {
            strength: Box::new(|_, _| -30.),
            group: Box::new(|_, _| 0),
        }
    }
}

impl Force for GroupManyBodyForce {
    fn build(&self, graph: &Graph) -> Box<ForceContext> {
        let strength_accessor = &self.strength;
        let strength = graph
            .nodes()
            .map(|i| strength_accessor(graph, i))
            .collect::<Vec<_>>();

        let groups = {
            let group_accessor = &self.group;
            let groups = graph
                .nodes()
                .map(|i| group_accessor(graph, i))
                .collect::<Vec<_>>();
            group_indices(&groups)
        };

        Box::new(GroupManyBodyForceContext::new(groups, strength))
    }
}
