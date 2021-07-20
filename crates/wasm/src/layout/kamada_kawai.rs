use crate::graph::JsGraph;
use crate::layout::force_simulation::coordinates::JsCoordinates;
use js_sys::Function;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = kamadaKawai)]
pub fn kamada_kawai(
  graph: &JsGraph,
  coordinates: &JsCoordinates,
  _length: Function,
  eps: f32,
  width: f32,
  height: f32,
) -> JsValue {
  let mut result = coordinates
    .coordinates()
    .iter()
    .map(|(u, p)| (u, (p.x, p.y)))
    .collect::<HashMap<_, _>>();
  petgraph_layout_kamada_kawai::kamada_kawai(
    graph.graph(),
    &mut result,
    &mut |_| 1.,
    eps,
    width,
    height,
  );
  JsValue::from_serde(
    &result
      .into_iter()
      .map(|(k, v)| (k.index(), v))
      .collect::<HashMap<_, _>>(),
  )
  .unwrap()
}
