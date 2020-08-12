pub mod group_center_force;
pub mod group_link_force;
pub mod group_many_body_force;
pub mod group_position_force;

pub use self::group_center_force::GroupCenterForce;
pub use self::group_link_force::GroupLinkForce;
pub use self::group_many_body_force::GroupManyBodyForce;
pub use self::group_position_force::GroupPositionForce;
use std::collections::HashMap;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Group {
    pub x: f32,
    pub y: f32,
}

impl Group {
    pub fn new(x: f32, y: f32) -> Group {
        Group { x: x, y: y }
    }
}

pub fn group_indices(groups: &Vec<usize>) -> HashMap<usize, Vec<usize>> {
    let mut result = HashMap::new();
    for (i, &group) in groups.iter().enumerate() {
        if !result.contains_key(&group) {
            result.insert(group, Vec::new());
        }
        let ids = result.get_mut(&group).unwrap();
        ids.push(i);
    }
    result
}
