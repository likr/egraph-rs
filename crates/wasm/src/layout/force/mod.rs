use crate::graph::JsGraph;
use crate::layout::force_simulation::coordinates::JsCoordinates;
use js_sys::{Function, Reflect};
use petgraph_layout_force::link_force::LinkArgument;
use petgraph_layout_force::position_force::NodeArgument;
use petgraph_layout_force::{CollideForce, LinkForce, ManyBodyForce, PositionForce, RadialForce};
use petgraph_layout_force_simulation::{Force, ForceToNode};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = CollideForce)]
pub struct JsCollideForce {
    force: CollideForce,
}

#[wasm_bindgen(js_class = CollideForce)]
impl JsCollideForce {
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &JsGraph,
        f: &Function,
        options: &JsValue,
    ) -> Result<JsCollideForce, JsValue> {
        let mut radii = HashMap::new();
        for u in graph.graph().node_indices() {
            let result = f.call1(&JsValue::null(), &JsValue::from_f64(u.index() as f64))?;
            let radius = Reflect::get(&result, &"radius".into())?
                .as_f64()
                .ok_or_else(|| format!("nodes[{}].radius is not a number", u.index()))?;
            radii.insert(u, radius as f32);
        }
        let strength = Reflect::get(&options, &"strength".into())?
            .as_f64()
            .ok_or_else(|| format!("options.strength is not a number"))?;
        let iterations = Reflect::get(&options, &"strength".into())?
            .as_f64()
            .ok_or_else(|| format!("options.iterations is not a number"))?;
        Ok(JsCollideForce {
            force: CollideForce::new(
                graph.graph(),
                |_, u| radii[&u],
                strength as f32,
                iterations as usize,
            ),
        })
    }

    pub fn apply(&self, context: &mut JsCoordinates, alpha: f32) {
        self.force.apply(context.points_mut(), alpha);
    }
}

#[wasm_bindgen(js_name = LinkForce)]
pub struct JsLinkForce {
    force: LinkForce,
}

#[wasm_bindgen(js_class = LinkForce)]
impl JsLinkForce {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, f: &Function) -> Result<JsLinkForce, JsValue> {
        if f.is_undefined() {
            return Ok(JsLinkForce {
                force: LinkForce::new(graph.graph()),
            });
        }
        let mut link_arguments = HashMap::new();
        for e in graph.graph().edge_indices() {
            let result = f.call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))?;
            let distance = Reflect::get(&result, &"distance".into())
                .ok()
                .map(|v| v.as_f64().map(|v| v as f32))
                .flatten();
            let strength = Reflect::get(&result, &"strength".into())
                .ok()
                .map(|v| v.as_f64().map(|v| v as f32))
                .flatten();
            link_arguments.insert(e, LinkArgument { distance, strength });
        }
        Ok(JsLinkForce {
            force: LinkForce::new_with_accessor(graph.graph(), |_, e| link_arguments[&e]),
        })
    }

    pub fn apply(&self, coordinates: &mut JsCoordinates, alpha: f32) {
        self.force.apply(coordinates.points_mut(), alpha);
    }

    #[wasm_bindgen(js_name = applyToNode)]
    pub fn apply_to_node(&self, u: usize, coordinates: &mut JsCoordinates, alpha: f32) {
        self.force.apply_to_node(u, coordinates.points_mut(), alpha)
    }
}

#[wasm_bindgen(js_name = ManyBodyForce)]
pub struct JsManyBodyForce {
    force: ManyBodyForce,
}

#[wasm_bindgen(js_class = ManyBodyForce)]
impl JsManyBodyForce {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, f: &Function) -> Result<JsManyBodyForce, JsValue> {
        if f.is_undefined() {
            return Ok(JsManyBodyForce {
                force: ManyBodyForce::new(graph.graph()),
            });
        }
        let mut strengths = HashMap::new();
        for u in graph.graph().node_indices() {
            let result = f.call1(&JsValue::null(), &(u.index() as f64).into())?;
            let strength = Reflect::get(&result, &"strength".into())
                .ok()
                .map(|v| v.as_f64().map(|v| v as f32))
                .flatten();
            strengths.insert(u, strength);
        }
        Ok(JsManyBodyForce {
            force: ManyBodyForce::new_with_accessor(graph.graph(), |_, u| strengths[&u]),
        })
    }

    pub fn apply(&self, coordinates: &mut JsCoordinates, alpha: f32) {
        self.force.apply(coordinates.points_mut(), alpha);
    }

    #[wasm_bindgen(js_name = applyToNode)]
    pub fn apply_to_node(&self, u: usize, coordinates: &mut JsCoordinates, alpha: f32) {
        self.force.apply_to_node(u, coordinates.points_mut(), alpha)
    }
}

#[wasm_bindgen(js_name = PositionForce)]
pub struct JsPositionForce {
    force: PositionForce,
}

#[wasm_bindgen(js_class=PositionForce)]
impl JsPositionForce {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, f: &Function) -> Result<JsPositionForce, JsValue> {
        let mut node_arguments = HashMap::new();
        for u in graph.graph().node_indices() {
            let result = f.call1(&JsValue::null(), &JsValue::from_f64(u.index() as f64))?;
            let strength = Reflect::get(&result, &"strength".into())
                .ok()
                .map(|v| v.as_f64().map(|v| v as f32))
                .flatten();
            let x = Reflect::get(&result, &"x".into())
                .ok()
                .map(|v| v.as_f64().map(|v| v as f32))
                .flatten();
            let y = Reflect::get(&result, &"y".into())
                .ok()
                .map(|v| v.as_f64().map(|v| v as f32))
                .flatten();
            node_arguments.insert(u, NodeArgument { strength, x, y });
        }
        Ok(JsPositionForce {
            force: PositionForce::new(graph.graph(), |_, u| node_arguments[&u]),
        })
    }

    pub fn apply(&self, coordinates: &mut JsCoordinates, alpha: f32) {
        self.force.apply(coordinates.points_mut(), alpha);
    }

    #[wasm_bindgen(js_name = applyToNode)]
    pub fn apply_to_node(&self, u: usize, coordinates: &mut JsCoordinates, alpha: f32) {
        self.force.apply_to_node(u, coordinates.points_mut(), alpha)
    }
}

#[wasm_bindgen(js_name = RadialForce)]
pub struct JsRadialForce {
    force: RadialForce,
}

#[wasm_bindgen(js_class=RadialForce)]
impl JsRadialForce {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, f: Function) -> Result<JsRadialForce, JsValue> {
        let mut node_arguments = HashMap::new();
        for u in graph.graph().node_indices() {
            let result = f.call1(&JsValue::null(), &JsValue::from_f64(u.index() as f64))?;
            let strength = Reflect::get(&result, &"strength".into())?
                .as_f64()
                .ok_or_else(|| format!("nodes[{}].strength is not a number", u.index()))?;
            let radius = Reflect::get(&result, &"radius".into())?
                .as_f64()
                .ok_or_else(|| format!("nodes[{}].radius is not a number", u.index()))?;
            let x = Reflect::get(&result, &"x".into())?
                .as_f64()
                .ok_or_else(|| format!("nodes[{}].x is not a number", u.index()))?;
            let y = Reflect::get(&result, &"y".into())?
                .as_f64()
                .ok_or_else(|| format!("nodes[{}].y is not a number", u.index()))?;
            node_arguments.insert(
                u,
                Some((strength as f32, radius as f32, x as f32, y as f32)),
            );
        }
        Ok(JsRadialForce {
            force: RadialForce::new(graph.graph(), |_, u| node_arguments[&u]),
        })
    }

    pub fn apply(&self, coordinates: &mut JsCoordinates, alpha: f32) {
        self.force.apply(coordinates.points_mut(), alpha);
    }

    #[wasm_bindgen(js_name = applyToNode)]
    pub fn apply_to_node(&self, u: usize, coordinates: &mut JsCoordinates, alpha: f32) {
        self.force.apply_to_node(u, coordinates.points_mut(), alpha)
    }
}
