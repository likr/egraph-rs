use super::group_indices;
use crate::graph::{Graph, NodeIndex};
use crate::layout::force_directed::force::{Force, ForceContext, Point};
use std::collections::HashMap;
use std::marker::PhantomData;

pub struct GroupCenterForceContext {
    groups: HashMap<usize, Vec<usize>>,
    group_x: HashMap<usize, f32>,
    group_y: HashMap<usize, f32>,
}

impl GroupCenterForceContext {
    pub fn new(
        groups: HashMap<usize, Vec<usize>>,
        group_x: HashMap<usize, f32>,
        group_y: HashMap<usize, f32>,
    ) -> GroupCenterForceContext {
        GroupCenterForceContext {
            groups,
            group_x,
            group_y,
        }
    }
}

impl ForceContext for GroupCenterForceContext {
    fn apply(&self, points: &mut Vec<Point>, _alpha: f32) {
        for (group, indices) in self.groups.iter() {
            let group_x = self.group_x[group];
            let group_y = self.group_y[group];
            let mut center_x = 0.;
            let mut center_y = 0.;
            for &a in indices.iter() {
                center_x += points[a].x;
                center_y += points[a].y;
            }
            center_x /= indices.len() as f32;
            center_y /= indices.len() as f32;
            center_x -= group_x;
            center_y -= group_y;
            for &a in indices.iter() {
                points[a].x -= center_x;
                points[a].y -= center_y;
            }
        }
    }
}

pub struct GroupCenterForce<D, G: Graph<D>> {
    pub group: Box<dyn Fn(&G, NodeIndex) -> usize>,
    pub group_x: Box<dyn Fn(usize) -> f32>,
    pub group_y: Box<dyn Fn(usize) -> f32>,
    phantom: PhantomData<D>,
}

impl<D, G: Graph<D>> GroupCenterForce<D, G> {
    pub fn new() -> GroupCenterForce<D, G> {
        GroupCenterForce {
            group: Box::new(|_, _| 0),
            group_x: Box::new(|_| 0.),
            group_y: Box::new(|_| 0.),
            phantom: PhantomData,
        }
    }
}

impl<D, G: Graph<D>> Force<D, G> for GroupCenterForce<D, G> {
    fn build(&self, graph: &G) -> Box<dyn ForceContext> {
        let group_accessor = &self.group;
        let groups = graph
            .nodes()
            .map(|u| group_accessor(graph, u))
            .collect::<Vec<_>>();

        let group_x_accessor = &self.group_x;
        let group_y_accessor = &self.group_y;
        let mut group_x = HashMap::new();
        let mut group_y = HashMap::new();
        for &group in groups.iter() {
            if group_x.contains_key(&group) {
                continue;
            }
            group_x.insert(group, group_x_accessor(group));
            group_y.insert(group, group_y_accessor(group));
        }

        let groups = group_indices(&groups);

        Box::new(GroupCenterForceContext::new(groups, group_x, group_y))
    }
}
