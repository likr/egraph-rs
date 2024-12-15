use crate::{
    drawing::{
        JsDrawingEuclidean, JsDrawingEuclidean2d, JsDrawingHyperbolic2d, JsDrawingSpherical2d,
        JsDrawingTorus2d,
    },
    graph::JsGraph,
    rng::JsRng,
};
use js_sys::{Array, Function};
use petgraph::visit::EdgeRef;
use petgraph_layout_sgd::{
    DistanceAdjustedSgd, FullSgd, Scheduler, SchedulerConstant, SchedulerExponential,
    SchedulerLinear, SchedulerQuadratic, SchedulerReciprocal, Sgd, SparseSgd,
};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = "SchedulerConstant")]
pub struct JsSchedulerConstant {
    scheduler: SchedulerConstant<f32>,
}

#[wasm_bindgen(js_class = "SchedulerConstant")]
impl JsSchedulerConstant {
    pub fn run(&mut self, f: &Function) {
        self.scheduler.run(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    pub fn step(&mut self, f: &Function) {
        self.scheduler.step(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    #[wasm_bindgen(js_name = "isFinished")]
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

#[wasm_bindgen(js_name = "SchedulerLinear")]
pub struct JsSchedulerLinear {
    scheduler: SchedulerLinear<f32>,
}

#[wasm_bindgen(js_class = "SchedulerLinear")]
impl JsSchedulerLinear {
    pub fn run(&mut self, f: &Function) {
        self.scheduler.run(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    pub fn step(&mut self, f: &Function) {
        self.scheduler.step(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    #[wasm_bindgen(js_name = "isFinished")]
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

#[wasm_bindgen(js_name = "SchedulerQuadratic")]
pub struct JsSchedulerQuadratic {
    scheduler: SchedulerQuadratic<f32>,
}

#[wasm_bindgen(js_class = "SchedulerQuadratic")]
impl JsSchedulerQuadratic {
    pub fn run(&mut self, f: &Function) {
        self.scheduler.run(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    pub fn step(&mut self, f: &Function) {
        self.scheduler.step(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    #[wasm_bindgen(js_name = "isFinished")]
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

#[wasm_bindgen(js_name = "SchedulerExponential")]
pub struct JsSchedulerExponential {
    scheduler: SchedulerExponential<f32>,
}

#[wasm_bindgen(js_class = "SchedulerExponential")]
impl JsSchedulerExponential {
    pub fn run(&mut self, f: &Function) {
        self.scheduler.run(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    pub fn step(&mut self, f: &Function) {
        self.scheduler.step(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    #[wasm_bindgen(js_name = "isFinished")]
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

#[wasm_bindgen(js_name = "SchedulerReciprocal")]
pub struct JsSchedulerReciprocal {
    scheduler: SchedulerReciprocal<f32>,
}

#[wasm_bindgen(js_class = "SchedulerReciprocal")]
impl JsSchedulerReciprocal {
    pub fn run(&mut self, f: &Function) {
        self.scheduler.run(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    pub fn step(&mut self, f: &Function) {
        self.scheduler.step(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    #[wasm_bindgen(js_name = "isFinished")]
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

#[wasm_bindgen(js_name = "FullSgd")]
pub struct JsFullSgd {
    sgd: FullSgd<f32>,
}

#[wasm_bindgen(js_class = "FullSgd")]
impl JsFullSgd {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, length: &Function) -> JsFullSgd {
        let mut length_map = HashMap::new();
        for e in graph.graph().edge_indices() {
            let c = length
                .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            length_map.insert(e, c);
        }
        JsFullSgd {
            sgd: FullSgd::new(graph.graph(), |e| length_map[&e.id()]),
        }
    }

    pub fn shuffle(&mut self, rng: &mut JsRng) {
        self.sgd.shuffle(rng.get_mut());
    }

    #[wasm_bindgen(js_name = "applyWithDrawingEuclidean2d")]
    pub fn apply_with_drawing_euclidean_2d(&self, drawing: &mut JsDrawingEuclidean2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    #[wasm_bindgen(js_name = "applyWithDrawingEuclidean")]
    pub fn apply_with_drawing_euclidean(&self, drawing: &mut JsDrawingEuclidean, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    #[wasm_bindgen(js_name = "applyWithDrawingHyperbolic2d")]
    pub fn apply_with_drawing_hyperbolic_2d(&self, drawing: &mut JsDrawingHyperbolic2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    #[wasm_bindgen(js_name = "applyWithDrawingSpherical2d")]
    pub fn apply_with_drawing_spherical_2d(&self, drawing: &mut JsDrawingSpherical2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    #[wasm_bindgen(js_name = "applyWithDrawingTorus2d")]
    pub fn apply_with_drawing_torus_2d(&self, drawing: &mut JsDrawingTorus2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> JsSchedulerExponential {
        self.scheduler_exponential(t_max, epsilon)
    }

    #[wasm_bindgen(js_name = "schedulerConstant")]
    pub fn scheduler_constant(&self, t_max: usize, epsilon: f32) -> JsSchedulerConstant {
        JsSchedulerConstant {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    #[wasm_bindgen(js_name = "schedulerLinear")]
    pub fn scheduler_linear(&self, t_max: usize, epsilon: f32) -> JsSchedulerLinear {
        JsSchedulerLinear {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    #[wasm_bindgen(js_name = "schedulerQuadratic")]
    pub fn scheduler_quadratic(&self, t_max: usize, epsilon: f32) -> JsSchedulerQuadratic {
        JsSchedulerQuadratic {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    #[wasm_bindgen(js_name = "schedulerExponential")]
    pub fn scheduler_exponential(&self, t_max: usize, epsilon: f32) -> JsSchedulerExponential {
        JsSchedulerExponential {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    #[wasm_bindgen(js_name = "schedulerReciprocal")]
    pub fn scheduler_reciprocal(&self, t_max: usize, epsilon: f32) -> JsSchedulerReciprocal {
        JsSchedulerReciprocal {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    #[wasm_bindgen(js_name = "updateDistance")]
    pub fn update_distance(&mut self, distance: &Function) {
        self.sgd.update_distance(|i, j, dij, wij| {
            let args = Array::new();
            args.push(&JsValue::from_f64(i as f64));
            args.push(&JsValue::from_f64(j as f64));
            args.push(&JsValue::from_f64(dij as f64));
            args.push(&JsValue::from_f64(wij as f64));
            distance
                .apply(&JsValue::null(), &args)
                .unwrap()
                .as_f64()
                .unwrap() as f32
        })
    }

    #[wasm_bindgen(js_name = "updateWeight")]
    pub fn update_weight(&mut self, weight: &Function) {
        self.sgd.update_weight(|i, j, d, w| {
            let args = Array::new();
            args.push(&JsValue::from_f64(i as f64));
            args.push(&JsValue::from_f64(j as f64));
            args.push(&JsValue::from_f64(d as f64));
            args.push(&JsValue::from_f64(w as f64));
            weight
                .apply(&JsValue::null(), &args)
                .unwrap()
                .as_f64()
                .unwrap() as f32
        })
    }
}

#[wasm_bindgen(js_name = "SparseSgd")]
pub struct JsSparseSgd {
    sgd: SparseSgd<f32>,
}

#[wasm_bindgen(js_class = "SparseSgd")]
impl JsSparseSgd {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, length: &Function, h: usize, rng: &mut JsRng) -> JsSparseSgd {
        let mut length_map = HashMap::new();
        for e in graph.graph().edge_indices() {
            let c = length
                .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            length_map.insert(e, c);
        }
        JsSparseSgd {
            sgd: SparseSgd::new_with_rng(graph.graph(), |e| length_map[&e.id()], h, rng.get_mut()),
        }
    }

    pub fn shuffle(&mut self, rng: &mut JsRng) {
        self.sgd.shuffle(rng.get_mut());
    }

    #[wasm_bindgen(js_name = "applyWithDrawingEuclidean2d")]
    pub fn apply_with_drawing_euclidean_2d(&self, drawing: &mut JsDrawingEuclidean2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    #[wasm_bindgen(js_name = "applyWithDrawingEuclidean")]
    pub fn apply_with_drawing_euclidean(&self, drawing: &mut JsDrawingEuclidean, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    #[wasm_bindgen(js_name = "applyWithDrawingHyperbolic2d")]
    pub fn apply_with_drawing_hyperbolic_2d(&self, drawing: &mut JsDrawingHyperbolic2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    #[wasm_bindgen(js_name = "applyWithDrawingSpherical2d")]
    pub fn apply_with_drawing_spherical_2d(&self, drawing: &mut JsDrawingSpherical2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    #[wasm_bindgen(js_name = "applyWithDrawingTorus2d")]
    pub fn apply_with_drawing_torus_2d(&self, drawing: &mut JsDrawingTorus2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> JsSchedulerExponential {
        self.scheduler_exponential(t_max, epsilon)
    }

    #[wasm_bindgen(js_name = "schedulerConstant")]
    pub fn scheduler_constant(&self, t_max: usize, epsilon: f32) -> JsSchedulerConstant {
        JsSchedulerConstant {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    #[wasm_bindgen(js_name = "schedulerLinear")]
    pub fn scheduler_linear(&self, t_max: usize, epsilon: f32) -> JsSchedulerLinear {
        JsSchedulerLinear {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    #[wasm_bindgen(js_name = "schedulerQuadratic")]
    pub fn scheduler_quadratic(&self, t_max: usize, epsilon: f32) -> JsSchedulerQuadratic {
        JsSchedulerQuadratic {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    #[wasm_bindgen(js_name = "schedulerExponential")]
    pub fn scheduler_exponential(&self, t_max: usize, epsilon: f32) -> JsSchedulerExponential {
        JsSchedulerExponential {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    #[wasm_bindgen(js_name = "schedulerReciprocal")]
    pub fn scheduler_reciprocal(&self, t_max: usize, epsilon: f32) -> JsSchedulerReciprocal {
        JsSchedulerReciprocal {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    #[wasm_bindgen(js_name = "updateDistance")]
    pub fn update_distance(&mut self, distance: &Function) {
        self.sgd.update_distance(|i, j, dij, wij| {
            let args = Array::new();
            args.push(&JsValue::from_f64(i as f64));
            args.push(&JsValue::from_f64(j as f64));
            args.push(&JsValue::from_f64(dij as f64));
            args.push(&JsValue::from_f64(wij as f64));
            distance
                .apply(&JsValue::null(), &args)
                .unwrap()
                .as_f64()
                .unwrap() as f32
        })
    }

    #[wasm_bindgen(js_name = "updateWeight")]
    pub fn update_weight(&mut self, weight: &Function) {
        self.sgd.update_weight(|i, j, d, w| {
            let args = Array::new();
            args.push(&JsValue::from_f64(i as f64));
            args.push(&JsValue::from_f64(j as f64));
            args.push(&JsValue::from_f64(d as f64));
            args.push(&JsValue::from_f64(w as f64));
            weight
                .apply(&JsValue::null(), &args)
                .unwrap()
                .as_f64()
                .unwrap() as f32
        })
    }
}

#[wasm_bindgen(js_name = "DistanceAdjustedFullSgd")]
pub struct JsDistanceAdjustedFullSgd {
    sgd: DistanceAdjustedSgd<FullSgd<f32>, f32>,
}

#[wasm_bindgen(js_class = "DistanceAdjustedFullSgd")]
impl JsDistanceAdjustedFullSgd {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, length: &Function) -> Self {
        let mut length_map = HashMap::new();
        for e in graph.graph().edge_indices() {
            let c = length
                .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            length_map.insert(e, c);
        }
        Self {
            sgd: DistanceAdjustedSgd::new(FullSgd::new(graph.graph(), |e| length_map[&e.id()])),
        }
    }

    pub fn shuffle(&mut self, rng: &mut JsRng) {
        self.sgd.shuffle(rng.get_mut());
    }

    pub fn apply(&self, drawing: &mut JsDrawingEuclidean2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    #[wasm_bindgen(js_name = "applyWithDistanceAdjustment")]
    pub fn apply_with_distance_adjustment(&self, drawing: &mut JsDrawingEuclidean2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> JsSchedulerExponential {
        JsSchedulerExponential {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    #[wasm_bindgen(js_name = "updateDistance")]
    pub fn update_distance(&mut self, distance: &Function) {
        self.sgd.update_distance(|i, j, d, w| {
            let args = Array::new();
            args.push(&JsValue::from_f64(i as f64));
            args.push(&JsValue::from_f64(j as f64));
            args.push(&JsValue::from_f64(d as f64));
            args.push(&JsValue::from_f64(w as f64));
            distance
                .apply(&JsValue::null(), &args)
                .unwrap()
                .as_f64()
                .unwrap() as f32
        })
    }

    #[wasm_bindgen(js_name = "updateWeight")]
    pub fn update_weight(&mut self, weight: &Function) {
        self.sgd.update_weight(|i, j, d, w| {
            let args = Array::new();
            args.push(&JsValue::from_f64(i as f64));
            args.push(&JsValue::from_f64(j as f64));
            args.push(&JsValue::from_f64(d as f64));
            args.push(&JsValue::from_f64(w as f64));
            weight
                .apply(&JsValue::null(), &args)
                .unwrap()
                .as_f64()
                .unwrap() as f32
        })
    }

    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f32 {
        self.sgd.alpha
    }

    #[wasm_bindgen(setter)]
    pub fn set_alpha(&mut self, value: f32) {
        self.sgd.alpha = value;
    }

    #[wasm_bindgen(getter, js_name = "minimumDistance")]
    pub fn minimum_distance(&self) -> f32 {
        self.sgd.minimum_distance
    }

    #[wasm_bindgen(setter, js_name = "minimumDistance")]
    pub fn set_minimum_distance(&mut self, value: f32) {
        self.sgd.minimum_distance = value;
    }
}

#[wasm_bindgen(js_name = "DistanceAdjustedSparseSgd")]
pub struct JsDistanceAdjustedSparseSgd {
    sgd: DistanceAdjustedSgd<SparseSgd<f32>, f32>,
}

#[wasm_bindgen(js_class = "DistanceAdjustedSparseSgd")]
impl JsDistanceAdjustedSparseSgd {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, length: &Function, h: usize, rng: &mut JsRng) -> Self {
        let mut length_map = HashMap::new();
        for e in graph.graph().edge_indices() {
            let c = length
                .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            length_map.insert(e, c);
        }
        Self {
            sgd: DistanceAdjustedSgd::new(SparseSgd::new_with_rng(
                graph.graph(),
                |e| length_map[&e.id()],
                h,
                rng.get_mut(),
            )),
        }
    }

    pub fn shuffle(&mut self, rng: &mut JsRng) {
        self.sgd.shuffle(rng.get_mut());
    }

    pub fn apply(&self, drawing: &mut JsDrawingEuclidean2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    #[wasm_bindgen(js_name = "applyWithDistanceAdjustment")]
    pub fn apply_with_distance_adjustment(&self, drawing: &mut JsDrawingEuclidean2d, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> JsSchedulerExponential {
        JsSchedulerExponential {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    #[wasm_bindgen(js_name = "updateDistance")]
    pub fn update_distance(&mut self, distance: &Function) {
        self.sgd.update_distance(|i, j, d, w| {
            let args = Array::new();
            args.push(&JsValue::from_f64(i as f64));
            args.push(&JsValue::from_f64(j as f64));
            args.push(&JsValue::from_f64(d as f64));
            args.push(&JsValue::from_f64(w as f64));
            distance
                .apply(&JsValue::null(), &args)
                .unwrap()
                .as_f64()
                .unwrap() as f32
        })
    }

    #[wasm_bindgen(js_name = "updateWeight")]
    pub fn update_weight(&mut self, weight: &Function) {
        self.sgd.update_weight(|i, j, d, w| {
            let args = Array::new();
            args.push(&JsValue::from_f64(i as f64));
            args.push(&JsValue::from_f64(j as f64));
            args.push(&JsValue::from_f64(d as f64));
            args.push(&JsValue::from_f64(w as f64));
            weight
                .apply(&JsValue::null(), &args)
                .unwrap()
                .as_f64()
                .unwrap() as f32
        })
    }

    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f32 {
        self.sgd.alpha
    }

    #[wasm_bindgen(setter)]
    pub fn set_alpha(&mut self, value: f32) {
        self.sgd.alpha = value;
    }

    #[wasm_bindgen(getter, js_name = "minimumDistance")]
    pub fn minimum_distance(&self) -> f32 {
        self.sgd.minimum_distance
    }

    #[wasm_bindgen(setter, js_name = "minimumDistance")]
    pub fn set_minimum_distance(&mut self, value: f32) {
        self.sgd.minimum_distance = value;
    }
}
