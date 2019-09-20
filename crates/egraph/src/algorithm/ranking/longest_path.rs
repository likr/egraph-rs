use crate::{Graph, NodeIndex};
use std::collections::HashMap;

fn dfs<D, G: Graph<D>>(
    graph: &G,
    layers: &mut HashMap<NodeIndex, usize>,
    u: NodeIndex,
    depth: usize,
) {
    for v in graph.out_nodes(u) {
        if layers.contains_key(&v) {
            let layer = layers.get_mut(&v).unwrap();
            if *layer <= depth {
                *layer = depth + 1
            }
        } else {
            layers.insert(v, depth + 1);
        }
        dfs(graph, layers, v, depth + 1);
    }
}

pub fn longest_path<D, G: Graph<D>>(graph: &G) -> HashMap<NodeIndex, usize> {
    let mut result = HashMap::new();
    for u in graph.source_nodes() {
        result.insert(u, 0);
        dfs(graph, &mut result, u, 0);
    }
    result
}
