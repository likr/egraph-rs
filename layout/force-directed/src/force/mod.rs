mod center_force;
mod collide_force;
mod force;
mod group_center_force;
mod group_link_force;
mod group_many_body_force;
mod group_position_force;
mod link_force;
mod many_body_force;
mod position_force;

pub use self::center_force::CenterForce;
pub use self::collide_force::CollideForce;
pub use self::force::{Force, ForceContext, Group, Point};
pub use self::group_center_force::GroupCenterForce;
pub use self::group_link_force::GroupLinkForce;
pub use self::group_many_body_force::GroupManyBodyForce;
pub use self::group_position_force::GroupPositionForce;
pub use self::link_force::LinkForce;
pub use self::many_body_force::ManyBodyForce;
pub use self::position_force::PositionForce;

use std::collections::HashMap;

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
