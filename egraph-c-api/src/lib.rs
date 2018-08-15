extern crate clp;
extern crate egraph;
extern crate egraph_force_directed;
extern crate petgraph;

pub mod graph;
pub mod force_directed;

use std::os::raw::{c_uchar, c_uint};
use self::graph::Graph;

#[no_mangle]
pub fn hoge() {
    let mut model = clp::Model::new();
    model.resize(3, 3);
    println!("{} {}", model.number_rows(), model.number_columns());
}

#[no_mangle]
pub unsafe fn connected_components(p_graph: *mut Graph) -> *mut c_uint {
    egraph::algorithms::connected_components(&*p_graph)
        .iter()
        .map(|&c| c as c_uint)
        .collect::<Vec<_>>()
        .as_mut_ptr()
}

#[no_mangle]
pub unsafe fn rust_alloc(bytes: c_uint) -> *mut c_uchar {
    let mut v = vec![0; bytes as usize];
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[no_mangle]
pub unsafe fn rust_free(pointer: *mut c_uint) {
    let _b = Box::from_raw(pointer);
}
