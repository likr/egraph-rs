use std::f32::consts::PI;
use std::mem::forget;
use std::os::raw::{c_uint};
use egraph::layout::force_directed::force::{Force, Point, CenterForce, Group, GroupForce, GroupLinkForce, LinkForce, ManyBodyForce};
use egraph::layout::force_directed::simulation::start_simulation;
use egraph::layout::force_directed::group::treemap;
use graph::Graph;

unsafe fn copy_to_vec(pointer: *mut c_uint, size: usize) -> Vec<usize> {
    let vec1 = Vec::from_raw_parts(pointer, size, size);
    let vec2 = vec1.iter().map(|&item| item as usize).collect::<Vec<_>>();
    forget(vec1);
    vec2
}

pub struct Simulation {
    forces: Vec<Box<Force>>,
}

impl Simulation {
    fn new() -> Simulation {
        Simulation {
            forces: Vec::new(),
        }
    }
}

#[no_mangle]
pub unsafe fn simulation_new() -> *mut Simulation {
    let simulation = Box::new(Simulation::new());
    Box::into_raw(simulation)
}

#[no_mangle]
pub unsafe fn simulation_add_center_force(p_simulation: *mut Simulation) {
    (*p_simulation).forces.push(Box::new(CenterForce::new()));
}

#[no_mangle]
pub unsafe fn simulation_add_group_force(p_simulation: *mut Simulation, p_groups: *mut Group, num_groups: c_uint, p_node_groups: *mut c_uint, num_nodes: c_uint) {
    let groups = Vec::from_raw_parts(p_groups, num_groups as usize, num_groups as usize);
    let node_groups = copy_to_vec(p_node_groups, num_nodes as usize);
    (*p_simulation).forces.push(Box::new(GroupForce::new(groups, node_groups)));
}

#[no_mangle]
pub unsafe fn simulation_add_group_link_force(p_simulation: *mut Simulation, p_graph: *mut Graph, p_node_groups: *mut c_uint) {
    let node_groups = copy_to_vec(p_node_groups, (*p_graph).node_count());
    let force = GroupLinkForce::new(&(*p_graph), &node_groups);
    (*p_simulation).forces.push(Box::new(force));
}

#[no_mangle]
pub unsafe fn simulation_add_link_force(p_simulation: *mut Simulation, p_graph: *mut Graph) {
    (*p_simulation).forces.push(Box::new(LinkForce::new(&(*p_graph))));
}

#[no_mangle]
pub unsafe fn simulation_add_many_body_force(p_simulation: *mut Simulation) {
    (*p_simulation).forces.push(Box::new(ManyBodyForce::new()));
}

#[no_mangle]
pub unsafe fn simulation_start(p_simulation: *mut Simulation, p_graph: *mut Graph) {
    let mut points = (*p_graph).node_indices()
        .map(|node| {
            let i = node.index();
            let r = 10. * (i as usize as f32).sqrt();
            let theta = PI * (3. - (5. as f32).sqrt()) * (i as usize as f32);
            let x = r * theta.cos();
            let y = r * theta.sin();
            Point::new(x, y)
        })
        .collect::<Vec<_>>();
    start_simulation(&mut points, &(*p_simulation).forces);
    for (node, point) in (*p_graph).node_indices().zip(points) {
        let mut node = (*p_graph).node_weight_mut(node).unwrap();
        node.x = point.x as f64;
        node.y = point.y as f64;
    }
}

#[no_mangle]
pub unsafe fn group_assign_treemap(width: c_uint, height: c_uint, num_groups: c_uint, p_node_groups: *mut c_uint, num_nodes: c_uint) -> *mut Group {
    let node_groups = copy_to_vec(p_node_groups, num_nodes as usize);
    let mut groups = treemap::assign(width as usize, height as usize, num_groups as usize, &node_groups);
    let pointer = groups.as_mut_ptr();
    forget(groups);
    pointer
}
