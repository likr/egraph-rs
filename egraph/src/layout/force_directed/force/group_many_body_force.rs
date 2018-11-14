use super::force::{Force, Group, Point};
use super::many_body_force::ManyBodyForce;

pub struct GroupManyBodyForce {
    pub groups: Vec<Group>,
    pub node_groups: Vec<usize>,
    pub strength: f32,
}

impl GroupManyBodyForce {
    pub fn new(groups: &Vec<Group>, node_groups: &Vec<usize>) -> GroupManyBodyForce {
        GroupManyBodyForce {
            groups: groups.clone().to_vec(),
            node_groups: node_groups.clone().to_vec(),
            strength: 0.1,
        }
    }
}

impl Force for GroupManyBodyForce {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        let groups = &self.groups;
        let node_groups = &self.node_groups;

        let many_body = ManyBodyForce::new();
        for g in 0..groups.len() {
            let group_point_indices = (0..points.len())
                .filter(|&i| node_groups[i] == g)
                .collect::<Vec<_>>();
            let mut group_points = group_point_indices
                .iter()
                .map(|&i| points[i])
                .collect::<Vec<_>>();
            many_body.apply(&mut group_points, alpha * 5.0);
            for (&i, &point) in group_point_indices.iter().zip(group_points.iter()) {
                points[i] = point
            }
        }
    }

    fn get_strength(&self) -> f32 {
        self.strength
    }

    fn set_strength(&mut self, strength: f32) {
        self.strength = strength;
    }
}
