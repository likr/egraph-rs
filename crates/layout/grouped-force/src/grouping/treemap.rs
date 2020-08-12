use super::{aggregate_nodes, node_group, Group};
use ordered_float::OrderedFloat;
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use std::cmp::Reverse;
use std::collections::HashMap;
use treemap::{normalize, squarify};

pub fn treemap_grouping<
    N,
    E,
    Ty: EdgeType,
    Ix: IndexType,
    F1: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> usize,
    F2: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32,
>(
    graph: &Graph<N, E, Ty, Ix>,
    group_accessor: F1,
    weight_accessor: F2,
    width: f32,
    height: f32,
) -> HashMap<usize, Group> {
    let groups = node_group(graph, group_accessor);
    let mut items = aggregate_nodes(graph, &groups, weight_accessor);
    items.sort_by_key(|item| Reverse(OrderedFloat::from(item.weight)));
    let mut values = items
        .iter()
        .map(|item| item.weight as f64)
        .collect::<Vec<_>>();
    normalize(&mut values, (width * height) as f64);

    let mut result = HashMap::new();
    for (tile, item) in squarify(width as f64, height as f64, &values)
        .iter()
        .zip(items)
    {
        let g = item.id;
        result.insert(
            g,
            Group::new(
                (tile.x + tile.dx / 2.) as f32 - width / 2.,
                (tile.y + tile.dy / 2.) as f32 - height / 2.,
                tile.dx as f32,
                tile.dy as f32,
            ),
        );
    }
    result
}
