// pub mod force_directed;
// pub mod radial;
pub mod treemap;

use crate::Graph;
use std::collections::HashMap;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Group {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Group {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Group {
        Group {
            x,
            y,
            width,
            height,
        }
    }
}

pub fn group_size<G>(
    graph: &Graph<G>,
    group: &Box<Fn(&Graph<G>, usize) -> usize>,
    size: &Box<Fn(&Graph<G>, usize) -> f32>,
) -> HashMap<usize, f32> {
    let mut result = HashMap::new();
    for a in graph.nodes() {
        let g = group(graph, a);
        if !result.contains_key(&g) {
            result.insert(g, 0.);
        }
        *result.get_mut(&g).unwrap() += size(graph, a);
    }
    result
}
// }

// pub use self::force_directed::ForceDirectedGrouping;
// pub use self::radial::RadialGrouping;
pub use self::treemap::TreemapGrouping;
