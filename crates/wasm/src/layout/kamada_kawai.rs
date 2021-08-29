use crate::graph::JsGraph;
use crate::layout::force_simulation::coordinates::JsCoordinates;
use js_sys::{Function, Reflect};
use petgraph::visit::EdgeRef;
use petgraph_layout_kamada_kawai::KamadaKawai;
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

#[wasm_bindgen(js_name = KamadaKawai)]
pub struct JsKamadaKawai {
  kamada_kawai: KamadaKawai,
}

#[wasm_bindgen(js_class = KamadaKawai)]
impl JsKamadaKawai {
  #[wasm_bindgen(constructor)]
  pub fn new(graph: &JsGraph, f: &Function) -> Result<JsKamadaKawai, JsValue> {
    let mut distance = HashMap::new();
    for e in graph.graph().edge_indices() {
      let result = f.call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))?;
      let d = Reflect::get(&result, &"distance".into())?
        .as_f64()
        .ok_or_else(|| format!("links[{}].distance is not a Number.", e.index()))?;
      distance.insert(e, d as f32);
    }
    Ok(JsKamadaKawai {
      kamada_kawai: KamadaKawai::new(graph.graph(), &mut |e| distance[&e.id()]),
    })
  }

  #[wasm_bindgen(js_name = selectNode)]
  pub fn select_node(&self, coordinates: &JsCoordinates) -> Option<usize> {
    self.kamada_kawai.select_node(coordinates.coordinates())
  }

  #[wasm_bindgen(js_name = applyToNode)]
  pub fn apply_to_node(&self, m: usize, coordinates: &mut JsCoordinates) {
    self
      .kamada_kawai
      .apply_to_node(m, coordinates.coordinates_mut());
  }

  pub fn run(&self, coordinates: &mut JsCoordinates) {
    self.kamada_kawai.run(coordinates.coordinates_mut());
  }
}
