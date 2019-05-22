pub mod force_directed;
pub mod radial;
pub mod treemap;

use petgraph::graph::IndexType;
use petgraph::prelude::*;
use petgraph::EdgeType;
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

pub trait Grouping<N, E, Ty: EdgeType, Ix: IndexType> {
    fn call(&self, graph: &Graph<N, E, Ty, Ix>, width: f32, height: f32) -> HashMap<usize, Group>;

    fn group_size(
        &self,
        graph: &Graph<N, E, Ty, Ix>,
        group: &Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> usize>,
        size: &Box<Fn(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32>,
    ) -> HashMap<usize, f32> {
        let mut result = HashMap::new();
        for a in graph.node_indices() {
            let g = group(graph, a);
            if !result.contains_key(&g) {
                result.insert(g, 0.);
            }
            *result.get_mut(&g).unwrap() += size(graph, a);
        }
        result
    }
}

pub use self::force_directed::ForceDirectedGrouping;
pub use self::radial::RadialGrouping;
pub use self::treemap::TreemapGrouping;
