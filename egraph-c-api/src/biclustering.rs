use std::collections::HashSet;
use std::mem::forget;
use std::os::raw::{c_double, c_uint};
use egraph::algorithms::biclustering::{Bicluster, QuasiBiclique};
use graph::Graph;

pub type Biclusters = Vec<Bicluster>;

#[no_mangle]
pub unsafe fn quasi_biclique_new() -> *mut QuasiBiclique {
    let quasi_biclique = Box::new(QuasiBiclique::new());
    Box::into_raw(quasi_biclique)
}

#[no_mangle]
pub unsafe fn quasi_biclique_call(
    p_quasi_biclique: *mut QuasiBiclique,
    p_graph: *mut Graph,
    p_source: *mut c_uint,
    source_size: c_uint,
    p_target: *mut c_uint,
    target_size: c_uint,
) -> *mut Biclusters {
    let source = Vec::from_raw_parts(p_source, source_size as usize, source_size as usize);
    let target = Vec::from_raw_parts(p_target, target_size as usize, target_size as usize);
    let source_set = source.iter().map(|&u| u as usize).collect::<HashSet<_>>();
    let target_set = target.iter().map(|&u| u as usize).collect::<HashSet<_>>();
    let biclusters = Box::new((*p_quasi_biclique).call(&(*p_graph), &source_set, &target_set));
    forget(source);
    forget(target);
    Box::into_raw(biclusters)
}

#[no_mangle]
pub unsafe fn quasi_biclique_get_mu(p_quasi_biclique: *mut QuasiBiclique) -> c_double {
    (*p_quasi_biclique).mu
}

#[no_mangle]
pub unsafe fn quasi_biclique_set_mu(p_quasi_biclique: *mut QuasiBiclique, value: c_double) {
    (*p_quasi_biclique).mu = value;
}

#[no_mangle]
pub unsafe fn quasi_biclique_get_min_size(p_quasi_biclique: *mut QuasiBiclique) -> c_uint {
    (*p_quasi_biclique).min_size as c_uint
}

#[no_mangle]
pub unsafe fn quasi_biclique_set_min_size(p_quasi_biclique: *mut QuasiBiclique, value: c_uint) {
    (*p_quasi_biclique).min_size = value as usize;
}

#[no_mangle]
pub unsafe fn bicluster_length(p_biclusters: *mut Biclusters) -> c_uint {
    (*p_biclusters).len() as c_uint
}

#[no_mangle]
pub unsafe fn bicluster_source(p_biclusters: *mut Biclusters, i: c_uint) -> *mut c_uint {
    let mut vertices = (*p_biclusters)[i as usize].source.iter().map(|&index| index as c_uint).collect::<Vec<_>>();
    let pointer = vertices.as_mut_ptr();
    forget(vertices);
    pointer
}

#[no_mangle]
pub unsafe fn bicluster_source_length(p_biclusters: *mut Biclusters, i: c_uint) -> c_uint {
    (*p_biclusters)[i as usize].source.len() as c_uint
}

#[no_mangle]
pub unsafe fn bicluster_target(p_biclusters: *mut Biclusters, i: c_uint) -> *mut c_uint {
    let mut vertices = (*p_biclusters)[i as usize].target.iter().map(|&index| index as c_uint).collect::<Vec<_>>();
    let pointer = vertices.as_mut_ptr();
    forget(vertices);
    pointer
}

#[no_mangle]
pub unsafe fn bicluster_target_length(p_biclusters: *mut Biclusters, i: c_uint) -> c_uint {
    (*p_biclusters)[i as usize].target.len() as c_uint
}
