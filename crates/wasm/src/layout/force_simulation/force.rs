use crate::graph::JsGraph;
use js_sys::{Function, Reflect};
use petgraph_layout_force_simulation::force::link_force::LinkArgument;
use petgraph_layout_force_simulation::force::position_force;
use petgraph_layout_force_simulation::force::{
    CenterForce, CollideForce, LinkForce, ManyBodyForce, PositionForce, RadialForce,
};
use petgraph_layout_force_simulation::Force;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = Force)]
pub struct JsForce {
    force: Box<dyn Force>,
}

impl JsForce {
    pub fn new<F: Force + 'static>(force: F) -> JsForce {
        JsForce {
            force: Box::new(force),
        }
    }

    pub fn with_box(force: Box<dyn Force>) -> JsForce {
        JsForce { force }
    }
}

impl AsRef<dyn Force> for JsForce {
    fn as_ref(&self) -> &(dyn Force + 'static) {
        self.force.as_ref()
    }
}

#[wasm_bindgen(js_name = CenterForce)]
pub struct JsCenterForce {}

#[wasm_bindgen(js_class = CenterForce)]
impl JsCenterForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsForce {
        JsForce::new(CenterForce::new())
    }
}

#[wasm_bindgen(js_name = CollideForce)]
pub struct JsCollideForce {}

#[wasm_bindgen(js_class = CollideForce)]
impl JsCollideForce {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, f: &Function, options: &JsValue) -> Result<JsForce, JsValue> {
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
        Ok(JsForce::new(CollideForce::new(
            graph.graph(),
            |_, u| radii[&u],
            strength as f32,
            iterations as usize,
        )))
    }
}

#[wasm_bindgen(js_name = LinkForce)]
pub struct JsLinkForce {}

#[wasm_bindgen(js_class = LinkForce)]
impl JsLinkForce {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, f: &Function) -> Result<JsForce, JsValue> {
        if f.is_undefined() {
            return Ok(JsForce::new(LinkForce::new(graph.graph())));
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
        Ok(JsForce::new(LinkForce::new_with_accessor(
            graph.graph(),
            |_, e| link_arguments[&e],
        )))
    }
}

#[wasm_bindgen(js_name = ManyBodyForce)]
pub struct JsManyBodyForce {}

#[wasm_bindgen(js_class = ManyBodyForce)]
impl JsManyBodyForce {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, f: &Function) -> Result<JsForce, JsValue> {
        if f.is_undefined() {
            return Ok(JsForce::new(ManyBodyForce::new(graph.graph())));
        }
        let mut strengths = HashMap::new();
        for u in graph.graph().node_indices() {
            let result = f.call1(&JsValue::null(), &JsValue::from_f64(u.index() as f64))?;
            let strength = Reflect::get(&result, &"strength".into())
                .ok()
                .map(|v| v.as_f64().map(|v| v as f32))
                .flatten();
            strengths.insert(u, strength);
        }
        Ok(JsForce::new(ManyBodyForce::new_with_accessor(
            graph.graph(),
            |_, u| strengths[&u],
        )))
    }
}

#[wasm_bindgen(js_name = PositionForce)]
pub struct JsPositionForce {}

#[wasm_bindgen(js_class=PositionForce)]
impl JsPositionForce {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, f: &Function) -> Result<JsForce, JsValue> {
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
            node_arguments.insert(u, position_force::NodeArgument { strength, x, y });
        }
        Ok(JsForce::new(PositionForce::new(graph.graph(), |_, u| {
            node_arguments[&u]
        })))
    }
}

#[wasm_bindgen(js_name = RadialForce)]
pub struct JsRadialForce {}

#[wasm_bindgen(js_class=RadialForce)]
impl JsRadialForce {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, f: Function) -> Result<JsForce, JsValue> {
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
        Ok(JsForce::new(RadialForce::new(graph.graph(), |_, u| {
            node_arguments[&u]
        })))
    }
}
