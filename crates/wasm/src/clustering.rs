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
) -> Result<JsValue, JsValue> {
  let graph = graph.graph();
  let mut group_map = HashMap::new();
  for u in graph.node_indices() {
    let group = groups
      .call1(&JsValue::null(), &JsValue::from_f64(u.index() as f64))?
      .as_f64()
      .ok_or_else(|| format!("group[{}] is not a number", u.index()))? as usize;
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
  result.push(&JsValue::from_serde(&group_ids).unwrap());
  Ok(result.into())
}
