use crate::{Graph, NodeIndex};
use std::collections::HashMap;

fn dfs<D, G: Graph<D>>(
    graph: &G,
    layers: &mut HashMap<NodeIndex, usize>,
    widths: &mut HashMap<usize, usize>,
    u: NodeIndex,
    width_limit: usize,
) -> usize {
    if let Some(&layer) = layers.get(&u) {
        return layer;
    }

    let mut max_layer = 0;
    for v in graph.in_nodes(u) {
        let layer = dfs(graph, layers, widths, v, width_limit);
        if layer >= max_layer {
            max_layer = layer + 1;
        }
    }
    while let Some(&width) = widths.get(&max_layer) {
        if width < width_limit {
            break;
        }
        max_layer += 1;
    }
    if !widths.contains_key(&max_layer) {
        widths.insert(max_layer, 0);
    }
    *widths.get_mut(&max_layer).unwrap() += 1;
    layers.insert(u, max_layer);
    max_layer
}

pub fn min_width<D, G: Graph<D>>(graph: &G, width_limit: usize) -> HashMap<NodeIndex, usize> {
    let mut layering = HashMap::new();
    let mut widths = HashMap::new();
    for u in graph.sink_nodes() {
        dfs(graph, &mut layering, &mut widths, u, width_limit);
    }
    layering
}
