pub mod force_directed;
pub mod treemap;

pub use self::force_directed::force_directed_grouping;
// pub use self::radial::RadialGrouping;
pub use self::treemap::treemap_grouping;

use petgraph::graph::{Graph, IndexType, NodeIndex};
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

pub fn node_group<
    N,
    E,
    Ty: EdgeType,
    Ix: IndexType,
    F: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> usize,
>(
    graph: &Graph<N, E, Ty, Ix>,
    mut group_accessor: F,
) -> HashMap<NodeIndex<Ix>, usize> {
    graph
        .node_indices()
        .map(|u| (u, group_accessor(graph, u)))
        .collect::<HashMap<_, _>>()
}

pub fn aggregate_nodes<
    N,
    E,
    Ty: EdgeType,
    Ix: IndexType,
    F: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32,
>(
    graph: &Graph<N, E, Ty, Ix>,
    groups: &HashMap<NodeIndex<Ix>, usize>,
    mut weight: F,
) -> Vec<GroupNode> {
    let mut result = HashMap::new();
    for u in graph.node_indices() {
        let g = groups[&u];
        *result.entry(g).or_insert(0.) += weight(graph, u);
    }
    result
        .iter()
        .map(|(&id, &weight)| GroupNode { id, weight })
        .collect::<Vec<_>>()
}

pub fn aggregate_edges<
    N,
    E,
    Ty: EdgeType,
    Ix: IndexType,
    F: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>, NodeIndex<Ix>) -> f32,
>(
    graph: &Graph<N, E, Ty, Ix>,
    groups: &HashMap<NodeIndex<Ix>, usize>,
    mut weight: F,
) -> Vec<GroupLink> {
    let mut result = HashMap::new();
    for e in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(e).unwrap();
        let key = {
            let source_group = groups[&u];
            let target_group = groups[&v];
            if source_group == target_group {
                continue;
            }
            if source_group < target_group {
                (source_group, target_group)
            } else {
                (target_group, source_group)
            }
        };
        *result.entry(key).or_insert(0.) += weight(graph, u, v)
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
