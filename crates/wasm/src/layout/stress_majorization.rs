use crate::graph::JsGraph;
use js_sys::{Function, Reflect};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = stressMajorization)]
pub fn stress_majorization(
  graph: &JsGraph,
  coordinates: JsValue,
  f: Function,
) -> Result<JsValue, JsValue> {
  let mut distance = HashMap::new();
  for e in graph.graph().edge_indices() {
    let result = f.call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))?;
    let d = Reflect::get(&result, &"distance".into())?
      .as_f64()
      .ok_or_else(|| format!("links[{}].distance is not a Number.", e.index()))?;
    distance.insert(e, d as f32);
  }

  let mut coordinates = JsValue::into_serde::<HashMap<usize, (f32, f32)>>(&coordinates)
    .unwrap()
    .into_iter()
    .map(|(k, v)| (NodeIndex::new(k), v))
    .collect::<HashMap<_, _>>();
  petgraph_layout_stress_majorization::stress_majorization(
    graph.graph(),
    &mut coordinates,
    &mut |e| distance[&e.id()],
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
