pub mod center_force;
pub mod collide_force;
pub mod force;
pub mod group_center_force;
pub mod group_link_force;
pub mod group_many_body_force;
pub mod group_position_force;
pub mod link_force;
pub mod many_body_force;
pub mod position_force;

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
