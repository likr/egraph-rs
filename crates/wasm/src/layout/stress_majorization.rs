use crate::graph::JsGraph;
use crate::layout::force_simulation::coordinates::JsCoordinates;
use js_sys::{Function, Reflect};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = stressMajorization)]
pub fn stress_majorization(
  graph: &JsGraph,
  coordinates: &JsCoordinates,
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

  let mut result = coordinates
    .coordinates()
    .iter()
    .map(|(u, p)| (u, (p.x, p.y)))
    .collect::<HashMap<_, _>>();
  petgraph_layout_stress_majorization::stress_majorization(graph.graph(), &mut result, &mut |e| {
    distance[&e.id()]
  });
  Ok(
    JsValue::from_serde(
      &result
        .into_iter()
        .map(|(k, v)| (k.index(), v))
        .collect::<HashMap<_, _>>(),
    )
    .unwrap(),
  )
}
