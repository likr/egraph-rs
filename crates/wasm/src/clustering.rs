//! Graph clustering algorithms for WebAssembly.
//!
//! This module provides WebAssembly bindings for graph clustering algorithms,
//! which can be used to identify communities, reduce graph complexity, or
//! analyze the hierarchical structure of networks.
//!
//! The main operation provided is graph coarsening, which simplifies a graph
//! by merging nodes according to a grouping function.

use crate::graph::JsGraph;
use js_sys::{Array, Function};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// Coarsens a graph by merging nodes into groups.
///
/// Graph coarsening is a process that simplifies a graph by merging nodes according
/// to some grouping criteria. Nodes in the same group are collapsed into a single
/// node in the resulting coarsened graph, and multiple edges between groups are
/// combined into a single edge.
///
/// Takes a graph to coarsen, a grouping function that assigns each node to a group,
/// and functions for creating the merged node and edge values.
///
/// Returns an array containing the coarsened graph and a mapping from group IDs
/// to node indices in the coarsened graph.
///
/// Throws an error if any group ID is not a number.
#[wasm_bindgen(js_name = coarsen)]
pub fn js_coarsen(
    graph: &JsGraph,
    groups: &Function,
    shrink_node: &Function,
    shrink_edge: &Function,
) -> Result<JsValue, JsValue> {
    let graph = graph.graph();
    let mut group_map = HashMap::new();
    for u in graph.node_indices() {
        let group = groups
            .call1(&JsValue::null(), &JsValue::from_f64(u.index() as f64))?
            .as_f64()
            .ok_or_else(|| format!("group[{}] is not a number", u.index()))?
            as usize;
        group_map.insert(u, group);
    }
    let (coarsened_graph, group_ids) = petgraph_clustering::coarsen(
        graph,
        &mut |_, u| {
            let u = JsValue::from_f64(u.index() as f64);
            groups
                .call1(&JsValue::null(), &u)
                .unwrap()
                .as_f64()
                .unwrap() as usize
        },
        &mut |_, node_ids| {
            let node_ids = node_ids
                .iter()
                .map(|u| JsValue::from_f64(u.index() as f64))
                .collect::<Array>();
            shrink_node.call1(&JsValue::null(), &node_ids).unwrap()
        },
        &mut |_, edge_ids| {
            let edge_ids = edge_ids
                .iter()
                .map(|u| JsValue::from_f64(u.index() as f64))
                .collect::<Array>();
            shrink_edge.call1(&JsValue::null(), &edge_ids).unwrap()
        },
    );

    let group_ids = group_ids
        .into_iter()
        .map(|(group, node_id)| (group, node_id.index()))
        .collect::<HashMap<_, _>>();

    let result = Array::new();
    result.push(&JsGraph::new_from_graph(coarsened_graph).into());
    result.push(&serde_wasm_bindgen::to_value(&group_ids).unwrap());
    Ok(result.into())
}
