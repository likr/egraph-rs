use std::mem::forget;
use std::os::raw::{c_double, c_uint};
use egraph::layout::force_directed::force::{Point, Link};
use egraph::layout::force_directed::edge_bundling::{EdgeBundling, Line};
use graph::Graph;

#[no_mangle]
pub unsafe fn edge_bundling_new() -> *mut EdgeBundling {
    let edge_bundling = Box::new(EdgeBundling::new());
    Box::into_raw(edge_bundling)
}

#[no_mangle]
pub unsafe fn edge_bundling_get_cycles(p_edge_bundling: *mut EdgeBundling) -> c_uint {
    (*p_edge_bundling).cycles as c_uint
}

#[no_mangle]
pub unsafe fn edge_bundling_get_s0(p_edge_bundling: *mut EdgeBundling) -> c_double {
    (*p_edge_bundling).s0 as c_double
}

#[no_mangle]
pub unsafe fn edge_bundling_get_i0(p_edge_bundling: *mut EdgeBundling) -> c_uint {
    (*p_edge_bundling).i0 as c_uint
}

#[no_mangle]
pub unsafe fn edge_bundling_get_s_step(p_edge_bundling: *mut EdgeBundling) -> c_double {
    (*p_edge_bundling).s_step as c_double
}

#[no_mangle]
pub unsafe fn edge_bundling_get_i_step(p_edge_bundling: *mut EdgeBundling) -> c_double {
    (*p_edge_bundling).i_step as c_double
}

#[no_mangle]
pub unsafe fn edge_bundling_set_cycles(p_edge_bundling: *mut EdgeBundling, cycles: c_uint) {
    (*p_edge_bundling).cycles = cycles as usize;
}

#[no_mangle]
pub unsafe fn edge_bundling_set_s0(p_edge_bundling: *mut EdgeBundling, s0: c_double) {
    (*p_edge_bundling).s0 = s0 as f32;
}

#[no_mangle]
pub unsafe fn edge_bundling_set_i0(p_edge_bundling: *mut EdgeBundling, i0: c_uint) {
    (*p_edge_bundling).i0 = i0 as usize;
}

#[no_mangle]
pub unsafe fn edge_bundling_set_s_step(p_edge_bundling: *mut EdgeBundling, s_step: c_double) {
    (*p_edge_bundling).s_step = s_step as f32;
}

#[no_mangle]
pub unsafe fn edge_bundling_set_i_step(p_edge_bundling: *mut EdgeBundling, i_step: c_double) {
    (*p_edge_bundling).i_step = i_step as f32;
}

#[no_mangle]
pub unsafe fn edge_bundling_call(p_edge_bundling: *mut EdgeBundling, p_graph: *mut Graph) -> *mut Line {
    let points = (*(*p_graph).raw_nodes())
        .iter()
        .map(|node| Point::new(node.weight.x as f32, node.weight.y as f32))
        .collect::<Vec<_>>();
    let links = (*p_graph).edge_indices()
        .map(|edge| {
            let (source, target) = (*p_graph).edge_endpoints(edge).unwrap();
            Link::new(source.index(), target.index())
        })
        .collect::<Vec<_>>();
    let mut lines = (*p_edge_bundling).call(&points, &links);
    let pointer = lines.as_mut_ptr();
    forget(lines);
    pointer
}

#[no_mangle]
pub unsafe fn lines_at(line: *mut Line, i: c_uint) -> *mut Line {
    line.add(i as usize)
}

#[no_mangle]
pub unsafe fn line_points(line: *mut Line) -> *mut Point {
    (*line).points.as_mut_ptr()
}

#[no_mangle]
pub unsafe fn line_points_at(line: *mut Line, i: c_uint) -> *mut Point {
    line_points(line).add(i as usize)
}

#[no_mangle]
pub unsafe fn line_points_length(line: *mut Line) -> c_uint {
    (*line).points.len() as c_uint
}

