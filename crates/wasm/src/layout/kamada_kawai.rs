use crate::graph::JsGraph;
use js_sys::Function;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = kamadaKawai)]
pub fn kamada_kawai(
  graph: &JsGraph,
  coordinates: JsValue,
  _length: Function,
  eps: f32,
  width: f32,
  height: f32,
) -> JsValue {
  let mut coordinates = JsValue::into_serde::<HashMap<usize, (f32, f32)>>(&coordinates)
    .unwrap()
    .into_iter()
    .map(|(k, v)| (NodeIndex::new(k), v))
    .collect::<HashMap<_, _>>();
  petgraph_layout_kamada_kawai::kamada_kawai(
    graph.graph(),
    &mut coordinates,
    &mut |_| 1.,
    eps,
    width,
    height,
  );
  JsValue::from_serde(
    &coordinates
      .into_iter()
      .map(|(k, v)| (k.index(), v))
      .collect::<HashMap<_, _>>(),
  )
  .unwrap()
}
