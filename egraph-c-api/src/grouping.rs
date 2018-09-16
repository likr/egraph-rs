use std::os::raw::{c_double, c_uint};
use std::mem::forget;
use egraph::layout::grouping::{
    Group,
    Grouping,
    ForceDirectedGrouping,
    RadialGrouping,
    TreemapGrouping,
};
use graph::Graph;

#[no_mangle]
pub unsafe fn force_directed_grouping_new(p_graph: *mut Graph) -> *mut ForceDirectedGrouping {
    let force_directed_grouping = Box::new(ForceDirectedGrouping::new(&(*p_graph)));
    Box::into_raw(force_directed_grouping)
}

#[no_mangle]
pub unsafe fn force_directed_grouping_call(
    p_force_directed_grouping: *mut ForceDirectedGrouping,
    width: c_double,
    height: c_double,
    p_values: *mut c_double,
    num_values: c_uint
) -> *mut Group {
    let values = Vec::from_raw_parts(p_values, num_values as usize, num_values as usize);
    let mut groups = (*p_force_directed_grouping).call(width, height, &values);
    forget(values);
    let pointer = groups.as_mut_ptr();
    forget(groups);
    pointer
}

#[no_mangle]
pub unsafe fn force_directed_grouping_get_link_length(
    p_force_directed_grouping: *const ForceDirectedGrouping,
) -> c_double {
    (*p_force_directed_grouping).link_length as c_double
}

#[no_mangle]
pub unsafe fn force_directed_grouping_set_link_length(
    p_force_directed_grouping: *mut ForceDirectedGrouping,
    value: c_double,
) {
    (*p_force_directed_grouping).link_length = value as f64;
}

#[no_mangle]
pub unsafe fn force_directed_grouping_get_many_body_force_strength(
    p_force_directed_grouping: *const ForceDirectedGrouping,
) -> c_double {
    (*p_force_directed_grouping).many_body_force_strength as c_double
}

#[no_mangle]
pub unsafe fn force_directed_grouping_set_many_body_force_strength(
    p_force_directed_grouping: *mut ForceDirectedGrouping,
    value: c_double,
) {
    (*p_force_directed_grouping).many_body_force_strength = value as f64;
}

#[no_mangle]
pub unsafe fn force_directed_grouping_get_link_force_strength(
    p_force_directed_grouping: *const ForceDirectedGrouping,
) -> c_double {
    (*p_force_directed_grouping).link_force_strength as c_double
}

#[no_mangle]
pub unsafe fn force_directed_grouping_set_link_force_strength(
    p_force_directed_grouping: *mut ForceDirectedGrouping,
    value: c_double,
) {
    (*p_force_directed_grouping).link_force_strength = value as f64;
}

#[no_mangle]
pub unsafe fn force_directed_grouping_get_center_force_strength(
    p_force_directed_grouping: *const ForceDirectedGrouping,
) -> c_double {
    (*p_force_directed_grouping).center_force_strength as c_double
}

#[no_mangle]
pub unsafe fn force_directed_grouping_set_center_force_strength(
    p_force_directed_grouping: *mut ForceDirectedGrouping,
    value: c_double,
) {
    (*p_force_directed_grouping).center_force_strength = value as f64;
}

#[no_mangle]
pub unsafe fn radial_grouping_new() -> *mut RadialGrouping {
    let radial_grouping = Box::new(RadialGrouping::new());
    Box::into_raw(radial_grouping)
}

#[no_mangle]
pub unsafe fn radial_grouping_call(p_radial_grouping: *mut RadialGrouping, width: c_double, height: c_double, p_values: *mut c_double, num_values: c_uint) -> *mut Group {
    let values = Vec::from_raw_parts(p_values, num_values as usize, num_values as usize);
    let mut groups = (*p_radial_grouping).call(width, height, &values);
    forget(values);
    let pointer = groups.as_mut_ptr();
    forget(groups);
    pointer
}

#[no_mangle]
pub unsafe fn treemap_grouping_new() -> *mut TreemapGrouping {
    let treemap_grouping = Box::new(TreemapGrouping::new());
    Box::into_raw(treemap_grouping)
}

#[no_mangle]
pub unsafe fn treemap_grouping_call(p_treemap_grouping: *mut TreemapGrouping, width: c_double, height: c_double, p_values: *mut c_double, num_values: c_uint) -> *mut Group {
    let values = Vec::from_raw_parts(p_values, num_values as usize, num_values as usize);
    let mut groups = (*p_treemap_grouping).call(width, height, &values);
    forget(values);
    let pointer = groups.as_mut_ptr();
    forget(groups);
    pointer
}

#[no_mangle]
pub unsafe fn group_x(p_group: *const Group, i: c_uint) -> c_double {
    (*p_group.add(i as usize)).x
}

#[no_mangle]
pub unsafe fn group_y(p_group: *const Group, i: c_uint) -> c_double {
    (*p_group.add(i as usize)).y
}

#[no_mangle]
pub unsafe fn group_width(p_group: *const Group, i: c_uint) -> c_double {
    (*p_group.add(i as usize)).width
}

#[no_mangle]
pub unsafe fn group_height(p_group: *const Group, i: c_uint) -> c_double {
    (*p_group.add(i as usize)).height
}

#[no_mangle]
pub unsafe fn groups_at(p_group: *mut Group, i: c_uint) -> *mut Group {
    p_group.add(i)
}

#[no_mangle]
pub unsafe fn group_get_x(p_group: *const Group) -> c_double {
    (*p_group).x
}

#[no_mangle]
pub unsafe fn group_get_y(p_group: *const Group) -> c_double {
    (*p_group).y
}

#[no_mangle]
pub unsafe fn group_get_width(p_group: *const Group) -> c_double {
    (*p_group).width
}

#[no_mangle]
pub unsafe fn group_get_height(p_group: *const Group) -> c_double {
    (*p_group).height
}

#[no_mangle]
pub unsafe fn group_set_x(p_group: *mut Group, value: c_double) {
    (*p_group).x = value;
}

#[no_mangle]
pub unsafe fn group_set_y(p_group: *mut Group, value: c_double) {
    (*p_group). = value;y
}

#[no_mangle]
pub unsafe fn group_set_width(p_group: *mut Group, value: c_double) {
    (*p_group).width = value;
}

#[no_mangle]
pub unsafe fn group_set_height(p_group: *mut Group, value: c_double) {
    (*p_group).height = value;
}
