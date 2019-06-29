pub mod force_directed;
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

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GroupNode {
    pub id: usize,
    pub weight: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GroupLink {
    pub source: usize,
    pub target: usize,
    pub weight: f32,
}

pub fn aggregate_nodes<D, G: Graph<D>>(
    graph: &G,
    group: &Box<dyn Fn(&G, usize) -> usize>,
    weight: &Box<dyn Fn(&G, usize) -> f32>,
) -> Vec<GroupNode> {
    let mut result = HashMap::new();
    for a in graph.nodes() {
        let g = group(graph, a);
        if !result.contains_key(&g) {
            result.insert(g, 0.);
        }
        *result.get_mut(&g).unwrap() += weight(graph, a);
    }
    result
        .iter()
        .map(|(&id, &weight)| GroupNode { id, weight })
        .collect::<Vec<_>>()
}

pub fn aggregate_edges<D, G: Graph<D>>(
    graph: &G,
    group: &Box<dyn Fn(&G, usize) -> usize>,
    weight: &Box<dyn Fn(&G, usize, usize) -> f32>,
) -> Vec<GroupLink> {
    let group_ids = graph
        .nodes()
        .map(|u| (u, group(graph, u)))
        .collect::<HashMap<usize, usize>>();
    let mut result = HashMap::new();

    for (u, v) in graph.edges() {
        let key = {
            let source_group = group_ids[&u];
            let target_group = group_ids[&v];
            if source_group == target_group {
                continue;
            }
            if source_group < target_group {
                (source_group, target_group)
            } else {
                (target_group, source_group)
            }
        };
        if !result.contains_key(&key) {
            result.insert(key, 0.);
        }
        *result.get_mut(&key).unwrap() += (weight)(graph, u, v);
    }
    result
        .iter()
        .map(|(&(source, target), &weight)| GroupLink {
            source,
            target,
            weight,
        })
        .collect::<Vec<_>>()
}

pub use self::force_directed::ForceDirectedGrouping;
// pub use self::radial::RadialGrouping;
pub use self::treemap::TreemapGrouping;
