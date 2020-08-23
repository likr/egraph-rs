use super::force::JsForce;
use crate::graph::JsGraph;
use core::ops::Deref;
use js_sys::{Array, Function, Reflect};
use petgraph::graph::NodeIndex;
use petgraph_layout_force_simulation::Simulation;
use std::collections::HashMap;
use wasm_bindgen::convert::RefFromWasmAbi;
use wasm_bindgen::prelude::*;

fn convert_forces(
    forces: &Box<[JsValue]>,
) -> Result<
    Vec<wasm_bindgen::__rt::Ref<'_, crate::layout::force_simulation::force::JsForce>>,
    JsValue,
> {
    let mut result = vec![];
    for force in forces.iter() {
        let ptr = Reflect::get(&force, &"ptr".into())?
            .as_f64()
            .ok_or_else(|| "xxx")? as u32;
        let force = unsafe { JsForce::ref_from_abi(ptr) };
        result.push(force);
    }
    Ok(result)
}

#[wasm_bindgen]
extern "C" {
    pub type ForceArray;
    #[wasm_bindgen(method, structural, indexing_getter)]
    pub fn get(this: &ForceArray, prop: &str) -> JsForce;
}

fn convert_coordinates(coordinates: &HashMap<NodeIndex, (f32, f32)>) -> JsValue {
    let coordinates = coordinates
        .iter()
        .map(|(u, &(x, y))| (u.index(), (x, y)))
        .collect::<HashMap<usize, (f32, f32)>>();
    JsValue::from_serde(&coordinates).unwrap()
}

#[wasm_bindgen(js_name = Simulation)]
pub struct JsSimulation {
    simulation: Simulation<u32>,
}

#[wasm_bindgen(js_class = Simulation)]
impl JsSimulation {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, f: &Function) -> Result<JsSimulation, JsValue> {
        let mut initial_position = HashMap::new();
        for u in graph.graph().node_indices() {
            let xy = f.call1(&JsValue::null(), &JsValue::from_f64(u.index() as f64))?;
            let xy: Array = xy.into();
            let x = xy
                .get(0)
                .as_f64()
                .ok_or_else(|| format!("x[{}] is not a number", u.index()))?;
            let y = xy
                .get(1)
                .as_f64()
                .ok_or_else(|| format!("y[{}] is not a number", u.index()))?;
            initial_position.insert(u, (x as f32, y as f32));
        }
        Ok(JsSimulation {
            simulation: Simulation::new(graph.graph(), |_, u| initial_position[&u]),
        })
    }

    pub fn run(&mut self, js_forces: Box<[JsValue]>) -> Result<JsValue, JsValue> {
        let forces = convert_forces(&js_forces)?;
        let force_refs = forces.iter().map(|f| f.deref()).collect::<Vec<_>>();
        Ok(convert_coordinates(
            &self.simulation.run(&force_refs.as_slice()),
        ))
    }

    #[wasm_bindgen(js_name = runStep)]
    pub fn run_step(&mut self, n: usize, js_forces: Box<[JsValue]>) -> Result<JsValue, JsValue> {
        let forces = convert_forces(&js_forces)?;
        let force_refs = forces.iter().map(|f| f.deref()).collect::<Vec<_>>();
        Ok(convert_coordinates(
            &self.simulation.run_step(n, &force_refs.as_slice()),
        ))
    }

    #[wasm_bindgen(js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.simulation.is_finished()
    }

    pub fn reset(&mut self, alpha_start: f32) {
        self.simulation.reset(alpha_start);
    }

    #[wasm_bindgen(getter = alphaStart)]
    pub fn alpha(&mut self) -> f32 {
        self.simulation.alpha
    }

    #[wasm_bindgen(setter = alphaStart)]
    pub fn set_alpha(&mut self, value: f32) {
        self.simulation.alpha = value;
    }

    #[wasm_bindgen(getter = alphaMin)]
    pub fn alpha_min(&mut self) -> f32 {
        self.simulation.alpha_min
    }

    #[wasm_bindgen(setter = alphaMin)]
    pub fn set_alpha_min(&mut self, value: f32) {
        self.simulation.alpha_min = value;
    }

    #[wasm_bindgen(getter = alphaTarget)]
    pub fn alpha_target(&mut self) -> f32 {
        self.simulation.alpha_target
    }

    #[wasm_bindgen(setter = alphaTarget)]
    pub fn set_alpha_target(&mut self, value: f32) {
        self.simulation.alpha_target = value;
    }

    #[wasm_bindgen(getter = velocityDecay)]
    pub fn velocity_decay(&mut self) -> f32 {
        self.simulation.velocity_decay
    }

    #[wasm_bindgen(setter = velocityDecay)]
    pub fn set_velocity_decay(&mut self, value: f32) {
        self.simulation.velocity_decay = value;
    }

    #[wasm_bindgen(getter = iterations)]
    pub fn iterations(&mut self) -> usize {
        self.simulation.iterations
    }

    #[wasm_bindgen(setter = iterations)]
    pub fn set_iterations(&mut self, value: usize) {
        self.simulation.iterations = value;
    }
}
