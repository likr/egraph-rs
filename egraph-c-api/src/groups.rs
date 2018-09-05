use std::os::raw::{c_double, c_uint};
use std::mem::forget;
use egraph::layout::groups::{Group, RadialLayout};

#[no_mangle]
pub unsafe fn radial_grouping_new() -> *mut RadialLayout {
    let radial_layout = Box::new(RadialLayout::new());
    Box::into_raw(radial_layout)
}

#[no_mangle]
pub unsafe fn radial_grouping_call(p_radial_layout: *mut RadialLayout, width: c_double, height: c_double, p_values: *mut c_double, num_values: c_uint) -> *mut Group {
    let values = Vec::from_raw_parts(p_values, num_values as usize, num_values as usize);
    let mut groups = (*p_radial_layout).call(width, height, &values);
    forget(values);
    let pointer = groups.as_mut_ptr();
    forget(groups);
    pointer
}

#[no_mangle]
pub unsafe fn group_x(p_group: *mut Group, i: c_uint) -> c_double {
    (*p_group.add(i as usize)).x
}

#[no_mangle]
pub unsafe fn group_y(p_group: *mut Group, i: c_uint) -> c_double {
    (*p_group.add(i as usize)).y
}

#[no_mangle]
pub unsafe fn group_width(p_group: *mut Group, i: c_uint) -> c_double {
    (*p_group.add(i as usize)).width
}

#[no_mangle]
pub unsafe fn group_height(p_group: *mut Group, i: c_uint) -> c_double {
    (*p_group.add(i as usize)).height
}
