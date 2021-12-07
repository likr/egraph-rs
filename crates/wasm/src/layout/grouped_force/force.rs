use crate::graph::JsGraph;
use crate::layout::force_simulation::coordinates::JsCoordinates;
use js_sys::{Function, Reflect};
use petgraph_layout_force::link_force::LinkArgument;
use petgraph_layout_force_simulation::Force;
use petgraph_layout_grouped_force::force::group_many_body_force::GroupManyBodyForceArgument;
use petgraph_layout_grouped_force::force::group_position_force;
use petgraph_layout_grouped_force::force::{
  GroupLinkForce, GroupManyBodyForce, GroupPositionForce,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = GroupLinkForce)]
pub struct JsGroupLinkForce {
  force: GroupLinkForce,
}

#[wasm_bindgen(js_class = GroupLinkForce)]
impl JsGroupLinkForce {
  #[wasm_bindgen(constructor)]
  pub fn new(
    graph: &JsGraph,
    group_accessor: &Function,
    inter_link_accessor: &Function,
    intra_link_accessor: &Function,
  ) -> Result<JsGroupLinkForce, JsValue> {
    Ok(JsGroupLinkForce {
      force: GroupLinkForce::new(
        graph.graph(),
        |_, u| {
          let obj = group_accessor
            .call1(&JsValue::null(), &JsValue::from_f64(u.index() as f64))
            .unwrap();
          Reflect::get(&obj, &"group".into())
            .unwrap()
            .as_f64()
            .unwrap() as usize
        },
        |_, e| {
          let obj = inter_link_accessor
            .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
            .unwrap();
          LinkArgument {
            distance: Some(
              Reflect::get(&obj, &"distance".into())
                .unwrap()
                .as_f64()
                .unwrap() as f32,
            ),
            strength: Some(
              Reflect::get(&obj, &"strength".into())
                .unwrap()
                .as_f64()
                .unwrap() as f32,
            ),
          }
        },
        |_, e| {
          let obj = intra_link_accessor
            .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
            .unwrap();
          LinkArgument {
            distance: Some(
              Reflect::get(&obj, &"distance".into())
                .unwrap()
                .as_f64()
                .unwrap() as f32,
            ),
            strength: Some(
              Reflect::get(&obj, &"strength".into())
                .unwrap()
                .as_f64()
                .unwrap() as f32,
            ),
          }
        },
      ),
    })
  }

  pub fn apply(&self, coordinates: &mut JsCoordinates, alpha: f32) {
    self.force.apply(coordinates.points_mut(), alpha);
  }
}

#[wasm_bindgen(js_name = GroupManyBodyForce)]
pub struct JsGroupManyBodyForce {
  force: GroupManyBodyForce,
}

#[wasm_bindgen(js_class = GroupManyBodyForce)]
impl JsGroupManyBodyForce {
  #[wasm_bindgen(constructor)]
  pub fn new(graph: &JsGraph, node_accessor: &Function) -> Result<JsGroupManyBodyForce, JsValue> {
    Ok(JsGroupManyBodyForce {
      force: GroupManyBodyForce::new(graph.graph(), |_, u| {
        let obj = node_accessor
          .call1(&JsValue::null(), &JsValue::from_f64(u.index() as f64))
          .unwrap();
        GroupManyBodyForceArgument {
          group: Reflect::get(&obj, &"group".into())
            .unwrap()
            .as_f64()
            .unwrap() as usize,
          strength: Some(
            Reflect::get(&obj, &"strength".into())
              .unwrap()
              .as_f64()
              .unwrap() as f32,
          ),
        }
      }),
    })
  }

  pub fn apply(&self, coordinates: &mut JsCoordinates, alpha: f32) {
    self.force.apply(coordinates.points_mut(), alpha);
  }
}

#[wasm_bindgen(js_name = GroupPositionForce)]
pub struct JsGroupPositionForce {
  force: GroupPositionForce,
}

#[wasm_bindgen(js_class=GroupPositionForce)]
impl JsGroupPositionForce {
  #[wasm_bindgen(constructor)]
  pub fn new(
    graph: &JsGraph,
    node_accessor: &Function,
    group_accessor: &Function,
  ) -> Result<JsGroupPositionForce, JsValue> {
    Ok(JsGroupPositionForce {
      force: GroupPositionForce::new(
        graph.graph(),
        |_, u| {
          let obj = node_accessor
            .call1(&JsValue::null(), &JsValue::from_f64(u.index() as f64))
            .unwrap();
          group_position_force::NodeArgument {
            group: Reflect::get(&obj, &"group".into())
              .unwrap()
              .as_f64()
              .unwrap() as usize,
            strength: Reflect::get(&obj, &"strength".into())
              .unwrap()
              .as_f64()
              .unwrap() as f32,
          }
        },
        |_, g| {
          let obj = group_accessor
            .call1(&JsValue::null(), &JsValue::from_f64(g as f64))
            .unwrap();
          group_position_force::GroupArgument {
            x: Reflect::get(&obj, &"x".into()).unwrap().as_f64().unwrap() as f32,
            y: Reflect::get(&obj, &"y".into()).unwrap().as_f64().unwrap() as f32,
          }
        },
      ),
    })
  }

  pub fn apply(&self, coordinates: &mut JsCoordinates, alpha: f32) {
    self.force.apply(coordinates.points_mut(), alpha);
  }
}
