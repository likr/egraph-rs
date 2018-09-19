use std::os::raw::{c_double, c_uint};
use egraph::layout::fm3::FM3;
use ::graph::Graph;

#[no_mangle]
pub unsafe fn layout_fm3_new() -> *mut FM3 {
    let fm3 = Box::new(FM3::new());
    Box::into_raw(fm3)
}

#[no_mangle]
pub unsafe fn layout_fm3_get_min_size(p_fm3: *mut FM3) -> c_uint {
    (*p_fm3).min_size as c_uint
}

#[no_mangle]
pub unsafe fn layout_fm3_set_min_size(p_fm3: *mut FM3, value: c_uint) {
    (*p_fm3).min_size = value as usize;
}

#[no_mangle]
pub unsafe fn layout_fm3_get_step_iteration(p_fm3: *mut FM3) -> c_uint {
    (*p_fm3).step_iteration as c_uint
}

#[no_mangle]
pub unsafe fn layout_fm3_set_step_iteration(p_fm3: *mut FM3, value: c_uint) {
    (*p_fm3).step_iteration = value as usize;
}

#[no_mangle]
pub unsafe fn layout_fm3_get_unit_edge_length(p_fm3: *mut FM3) -> c_double {
    (*p_fm3).unit_edge_length as c_double
}

#[no_mangle]
pub unsafe fn layout_fm3_set_unit_edge_length(p_fm3: *mut FM3, value: c_double) {
    (*p_fm3).unit_edge_length = value as f32;
}

#[no_mangle]
pub unsafe fn layout_fm3_call(p_fm3: *mut FM3, p_graph: *mut Graph) {
    let points = (*p_fm3).call(&*p_graph);
    for (node, point) in (*p_graph).node_indices().zip(points) {
        let mut node = (*p_graph).node_weight_mut(node).unwrap();
        node.x = point.x as f64;
        node.y = point.y as f64;
    }
}
