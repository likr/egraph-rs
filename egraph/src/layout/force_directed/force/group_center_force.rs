use super::force::{Force, Point, Group};

pub struct GroupCenterForce {
    pub groups: Vec<Group>,
    pub node_groups: Vec<usize>,
    pub strength: f32,
}

impl GroupCenterForce {
    pub fn new(groups: &Vec<Group>, node_groups: &Vec<usize>) -> GroupCenterForce {
        GroupCenterForce {
            groups: groups.clone().to_vec(),
            node_groups: node_groups.clone().to_vec(),
            strength: 0.1,
        }
    }
}

impl Force for GroupCenterForce {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        let groups = &self.groups;
        let node_groups = &self.node_groups;
        let k = self.strength * alpha;
        for (point, &g) in points.iter_mut().zip(node_groups) {
            point.vx += (groups[g].x - point.x) * k;
            point.vy += (groups[g].y - point.y) * k;
        }
    }
}
