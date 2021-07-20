use super::group_indices;
use crate::{Force, Point};
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use std::collections::HashMap;

pub struct GroupCenterForce {
    groups: HashMap<usize, Vec<usize>>,
    group_x: HashMap<usize, f32>,
    group_y: HashMap<usize, f32>,
}

impl GroupCenterForce {
    pub fn new<
        N,
        E,
        Ty: EdgeType,
        Ix: IndexType,
        F1: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> usize,
        F2: FnMut(usize) -> f32,
        F3: FnMut(usize) -> f32,
    >(
        graph: &Graph<N, E, Ty, Ix>,
        mut group_accessor: F1,
        mut group_x_accessor: F2,
        mut group_y_accessor: F3,
    ) -> GroupCenterForce {
        let groups = graph
            .node_indices()
            .map(|u| group_accessor(graph, u))
            .collect::<Vec<_>>();
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
        GroupCenterForce {
            groups,
            group_x,
            group_y,
        }
    }
}

impl Force for GroupCenterForce {
    fn apply(&self, points: &mut [Point], _alpha: f32) {
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
