extern crate clp;
extern crate egraph;
extern crate petgraph;

pub mod graph;
pub mod force_directed;
pub mod edge_bundling;

use std::os::raw::{c_double, c_uchar, c_uint};
use std::mem::forget;
use egraph::utils::treemap::{squarify, Tile};
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
pub unsafe fn squarified_treemap(width: c_double, height: c_double, p_values: *mut c_double, num_values: c_uint) -> *mut Tile {
    let values = Vec::from_raw_parts(p_values, num_values as usize, num_values as usize);
    let mut tiles = squarify(width, height, &values);
    forget(values);
    let pointer = tiles.as_mut_ptr();
    forget(tiles);
    pointer
}

#[no_mangle]
pub unsafe fn rust_alloc(bytes: c_uint) -> *mut c_uchar {
    let mut v = vec![0 as c_uchar; bytes as usize];
    let pointer = v.as_mut_ptr();
    forget(v);
    pointer
}

#[no_mangle]
pub unsafe fn rust_free(pointer: *mut c_uint) {
    let _b = Box::from_raw(pointer);
}
