use crate::graph::JsGraph;
use js_sys::{Array, Function};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = coarsen)]
pub fn js_coarsen(
  graph: &JsGraph,
  groups: &Function,
  shrink_node: &Function,
  shrink_edge: &Function,
) -> Result<JsGraph, JsValue> {
  let graph = graph.graph();
  let mut group_map = HashMap::new();
  for u in graph.node_indices() {
    let group = groups
      .call1(&JsValue::null(), &JsValue::from_f64(u.index() as f64))?
      .as_f64()
      .ok_or_else(|| format!("group[{}] is not a number", u.index()))? as usize;
    group_map.insert(u, group);
  }
  let coarsened_graph = petgraph_clustering::coarsen(
    graph,
    &group_map,
    &mut |_, node_ids| {
      let node_ids = node_ids
        .iter()
        .map(|u| JsValue::from_f64(u.index() as f64))
        .collect::<Array>();
      let result = shrink_node.call1(&JsValue::null(), &node_ids).ok().unwrap();
      result
    },
    &mut |_, edge_ids| {
      let edge_ids = edge_ids
        .iter()
        .map(|u| JsValue::from_f64(u.index() as f64))
        .collect::<Array>();
      let result = shrink_edge.call1(&JsValue::null(), &edge_ids).ok().unwrap();
      result
    },
  );
  Ok(JsGraph::new_from_graph(coarsened_graph))
}
