use std::f32::consts::PI;
use std::mem::forget;
use std::os::raw::{c_uint};
use egraph::layout::force_directed::force::{Force, Link, Point, CenterForce, Group, GroupForce, LinkForce, ManyBodyForce};
use egraph::layout::force_directed::simulation::start_simulation;
use egraph::layout::force_directed::group::treemap;
use graph::Graph;

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
    let node_groups = Vec::from_raw_parts(p_node_groups, num_nodes as usize, num_nodes as usize);
    let node_groups = node_groups.iter()
        .map(|&g| g as usize)
        .collect::<Vec<_>>();
    (*p_simulation).forces.push(Box::new(GroupForce::new(groups, node_groups)));
}

#[no_mangle]
pub unsafe fn simulation_add_link_force(p_simulation: *mut Simulation, p_graph: *mut Graph) {
    let links = (*p_graph).edge_indices()
        .map(|edge| {
            let (source, target) = (*p_graph).edge_endpoints(edge).unwrap();
            Link::new(source.index(), target.index())
        })
        .collect::<Vec<_>>();
    (*p_simulation).forces.push(Box::new(LinkForce::new(&links)));
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
    let node_groups = Vec::from_raw_parts(p_node_groups, num_nodes as usize, num_nodes as usize);
    let node_groups2 = node_groups.iter()
        .map(|&g| g as usize)
        .collect::<Vec<_>>();
    let mut groups = treemap::assign(width as usize, height as usize, num_groups as usize, &node_groups2);
    let pointer = groups.as_mut_ptr();
    forget(groups);
    forget(node_groups);
    pointer
}
