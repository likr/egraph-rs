use std::f32::consts::PI;
use std::mem::forget;
use std::os::raw::{c_double, c_uint};
use egraph::layout::force_directed::force::{Point, CenterForce, Group, GroupCenterForce, GroupLinkForce, GroupManyBodyForce, LinkForce, ManyBodyForce};
use egraph::layout::force_directed::simulation::Simulation;
use graph::Graph;
use super::copy_to_vec;

#[no_mangle]
pub unsafe fn simulation_new() -> *mut Simulation {
    let simulation = Box::new(Simulation::new());
    Box::into_raw(simulation)
}

#[no_mangle]
pub unsafe fn simulation_add_center_force(p_simulation: *mut Simulation) -> c_uint {
    (*p_simulation).forces.push(Box::new(CenterForce::new()));
    ((*p_simulation).forces.len() - 1) as c_uint
}

#[no_mangle]
pub unsafe fn simulation_add_group_center_force(p_simulation: *mut Simulation, p_groups: *mut Group, num_groups: c_uint, p_node_groups: *mut c_uint, num_nodes: c_uint) -> c_uint {
    let groups = Vec::from_raw_parts(p_groups, num_groups as usize, num_groups as usize);
    let node_groups = copy_to_vec(p_node_groups, num_nodes as usize);
    (*p_simulation).forces.push(Box::new(GroupCenterForce::new(&groups, &node_groups)));
    forget(groups);
    ((*p_simulation).forces.len() - 1) as c_uint
}

#[no_mangle]
pub unsafe fn simulation_add_group_link_force(p_simulation: *mut Simulation, p_graph: *mut Graph, p_node_groups: *mut c_uint, intra_group: c_double, inter_group: c_double) -> c_uint {
    let node_groups = copy_to_vec(p_node_groups, (*p_graph).node_count());
    let force = GroupLinkForce::new_with_strength(&(*p_graph), &node_groups, intra_group as f32, inter_group as f32);
    (*p_simulation).forces.push(Box::new(force));
    ((*p_simulation).forces.len() - 1) as c_uint
}

#[no_mangle]
pub unsafe fn simulation_add_group_many_body_force(p_simulation: *mut Simulation, p_groups: *mut Group, num_groups: c_uint, p_node_groups: *mut c_uint, num_nodes: c_uint) -> c_uint {
    let groups = Vec::from_raw_parts(p_groups, num_groups as usize, num_groups as usize);
    let node_groups = copy_to_vec(p_node_groups, num_nodes as usize);
    (*p_simulation).forces.push(Box::new(GroupManyBodyForce::new(&groups, &node_groups)));
    forget(groups);
    ((*p_simulation).forces.len() - 1) as c_uint
}

#[no_mangle]
pub unsafe fn simulation_add_link_force(p_simulation: *mut Simulation, p_graph: *mut Graph) -> c_uint {
    (*p_simulation).forces.push(Box::new(LinkForce::new(&(*p_graph))));
    ((*p_simulation).forces.len() - 1) as c_uint
}

#[no_mangle]
pub unsafe fn simulation_add_many_body_force(p_simulation: *mut Simulation) -> c_uint {
    (*p_simulation).forces.push(Box::new(ManyBodyForce::new()));
    ((*p_simulation).forces.len() - 1) as c_uint
}

#[no_mangle]
pub unsafe fn simulation_start(p_simulation: *mut Simulation, p_graph: *mut Graph) {
    let mut points = (*p_graph).node_indices()
        .map(|node| {
            let weight = (*p_graph).node_weight(node).unwrap();
            if weight.x != 0. && weight.y != 0. {
                return Point::new(weight.x as f32, weight.y as f32)
            }

            let i = node.index();
            let r = 10. * (i as usize as f32).sqrt();
            let theta = PI * (3. - (5. as f32).sqrt()) * (i as usize as f32);
            let x = r * theta.cos();
            let y = r * theta.sin();
            Point::new(x, y)
        })
        .collect::<Vec<_>>();
    (*p_simulation).start(&mut points);
    for (node, point) in (*p_graph).node_indices().zip(points) {
        let mut node = (*p_graph).node_weight_mut(node).unwrap();
        node.x = point.x as f64;
        node.y = point.y as f64;
    }
}

#[no_mangle]
pub unsafe fn simulation_get_strength(p_simulation: *mut Simulation, i: c_uint) -> c_double {
    (*p_simulation).forces[i as usize].get_strength() as c_double
}

#[no_mangle]
pub unsafe fn simulation_set_strength(p_simulation: *mut Simulation, i: c_uint, strength: c_double) {
    (*p_simulation).forces[i as usize].set_strength(strength as f32);
}

#[no_mangle]
pub unsafe fn simulation_get_alpha(p_simulation: *mut Simulation) -> c_double {
    (*p_simulation).alpha as c_double
}

#[no_mangle]
pub unsafe fn simulation_set_alpha(p_simulation: *mut Simulation, value: c_double) {
    (*p_simulation).alpha = value as f32;
}

#[no_mangle]
pub unsafe fn simulation_get_alpha_min(p_simulation: *mut Simulation) -> c_double {
    (*p_simulation).alpha_min as c_double
}

#[no_mangle]
pub unsafe fn simulation_set_alpha_min(p_simulation: *mut Simulation, value: c_double) {
    (*p_simulation).alpha_min = value as f32;
}

#[no_mangle]
pub unsafe fn simulation_get_alpha_target(p_simulation: *mut Simulation) -> c_double {
    (*p_simulation).alpha_target as c_double
}

#[no_mangle]
pub unsafe fn simulation_set_alpha_target(p_simulation: *mut Simulation, value: c_double) {
    (*p_simulation).alpha_target = value as f32;
}

#[no_mangle]
pub unsafe fn simulation_get_velocity_decay(p_simulation: *mut Simulation) -> c_double {
    (*p_simulation).velocity_decay as c_double
}

#[no_mangle]
pub unsafe fn simulation_set_velocity_decay(p_simulation: *mut Simulation, value: c_double) {
    (*p_simulation).velocity_decay = value as f32;
}

#[no_mangle]
pub unsafe fn point_x(point: *mut Point) -> c_double {
    (*point).x as c_double
}

#[no_mangle]
pub unsafe fn point_y(point: *mut Point) -> c_double {
    (*point).y as c_double
}
