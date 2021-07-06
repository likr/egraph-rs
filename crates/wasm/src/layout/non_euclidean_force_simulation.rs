use crate::graph::JsGraph;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = nonEuclideanFruchtermanReingold)]
pub fn non_euclidean_fruchterman_reingold(
  graph: &JsGraph,
  coordinates: JsValue,
  repeat: usize,
  k: f32,
) -> Result<JsValue, JsValue> {
  let mut coordinates = JsValue::into_serde::<HashMap<usize, (f32, f32)>>(&coordinates)
    .unwrap()
    .into_iter()
    .map(|(k, v)| (NodeIndex::new(k), v))
    .collect::<HashMap<_, _>>();
  petgraph_layout_non_euclidean_force_simulation::non_euclidean_fruchterman_reingold(
    graph.graph(),
    &mut coordinates,
    repeat,
    k,
  );
  Ok(
    JsValue::from_serde(
      &coordinates
        .into_iter()
        .map(|(k, v)| (k.index(), v))
        .collect::<HashMap<_, _>>(),
    )
    .unwrap(),
  )
}
