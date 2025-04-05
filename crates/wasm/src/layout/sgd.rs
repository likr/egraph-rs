//! Stochastic Gradient Descent (SGD) layout algorithms for WebAssembly.
//!
//! This module provides WebAssembly bindings for SGD-based graph layout algorithms,
//! which are force-directed approaches that efficiently handle large graphs by
//! optimizing layouts through stochastic sampling of node pairs.
//!
//! The module includes:
//! * Several SGD variants: full, sparse, and distance-adjusted
//! * Multiple learning rate scheduler implementations for controlling convergence
//! * Support for different drawing geometries (Euclidean, hyperbolic, spherical, torus)

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

/// WebAssembly binding for constant learning rate scheduler.
///
/// This scheduler maintains a constant learning rate throughout the optimization
/// process, making it simple but potentially less effective for convergence compared
/// to decay-based schedulers.
#[wasm_bindgen(js_name = "SchedulerConstant")]
pub struct JsSchedulerConstant {
    scheduler: SchedulerConstant<f32>,
}

#[wasm_bindgen(js_class = "SchedulerConstant")]
impl JsSchedulerConstant {
    /// Runs the complete scheduling process, applying the learning rate to each iteration.
    ///
    /// @param {Function} f - A callback function that receives the current learning rate at each step
    pub fn run(&mut self, f: &Function) {
        self.scheduler.run(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    /// Performs a single step of the scheduler, advancing to the next iteration.
    ///
    /// @param {Function} f - A callback function that receives the current learning rate
    pub fn step(&mut self, f: &Function) {
        self.scheduler.step(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    /// Checks if the scheduling process has completed all iterations.
    ///
    /// @returns {boolean} True if all iterations have been completed
    #[wasm_bindgen(js_name = "isFinished")]
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

/// WebAssembly binding for linear decay learning rate scheduler.
///
/// This scheduler linearly decreases the learning rate from an initial value
/// to a minimum value over the specified number of iterations.
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

/// WebAssembly binding for quadratic decay learning rate scheduler.
///
/// This scheduler decreases the learning rate following a quadratic curve,
/// which provides a more aggressive decay early on compared to linear decay.
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

/// WebAssembly binding for exponential decay learning rate scheduler.
///
/// This scheduler exponentially decreases the learning rate, providing
/// rapid decay initially and much slower decay in later iterations.
/// It's often effective for helping SGD converge to good solutions.
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

/// WebAssembly binding for reciprocal decay learning rate scheduler.
///
/// This scheduler decreases the learning rate proportionally to 1/t,
/// where t is the iteration number. This decay schedule is common in
/// many optimization algorithms.
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

/// WebAssembly binding for Full SGD layout algorithm.
///
/// Full SGD is a force-directed layout algorithm that computes shortest-path
/// distances between all pairs of nodes. While accurate, it can be
/// computationally expensive for large graphs.
#[wasm_bindgen(js_name = "FullSgd")]
pub struct JsFullSgd {
    sgd: FullSgd<f32>,
}

#[wasm_bindgen(js_class = "FullSgd")]
impl JsFullSgd {
    /// Creates a new Full SGD layout instance for the given graph.
    ///
    /// @param {Graph} graph - The graph to layout
    /// @param {Function} length - A function that takes an edge index and returns its desired length
    /// @returns {FullSgd} A new Full SGD layout instance
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

    /// Shuffles the node pairs used for SGD updates to improve randomization.
    ///
    /// @param {Rng} rng - The random number generator to use for shuffling
    pub fn shuffle(&mut self, rng: &mut JsRng) {
        self.sgd.shuffle(rng.get_mut());
    }

    /// Applies a single SGD update step to a 2D Euclidean drawing.
    ///
    /// @param {DrawingEuclidean2d} drawing - The drawing to modify
    /// @param {number} eta - The learning rate to use for this update
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

    /// Updates the distance calculation function for node pairs.
    ///
    /// This allows customizing how graph-theoretic distances are transformed
    /// into target distances in the layout.
    ///
    /// @param {Function} distance - A function that takes (i, j, distance, weight) and returns a new distance
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

    /// Updates the weight calculation function for node pairs.
    ///
    /// This allows customizing how much influence each node pair has on the layout.
    ///
    /// @param {Function} weight - A function that takes (i, j, distance, weight) and returns a new weight
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

/// WebAssembly binding for Sparse SGD layout algorithm.
///
/// Sparse SGD is an efficient variant of SGD that uses pivot-based
/// approximation of graph distances, making it suitable for large graphs
/// where computing all-pairs shortest paths would be too expensive.
#[wasm_bindgen(js_name = "SparseSgd")]
pub struct JsSparseSgd {
    sgd: SparseSgd<f32>,
}

#[wasm_bindgen(js_class = "SparseSgd")]
impl JsSparseSgd {
    /// Creates a new Sparse SGD layout instance for the given graph.
    ///
    /// @param {Graph} graph - The graph to layout
    /// @param {Function} length - A function that takes an edge index and returns its desired length
    /// @param {number} h - The number of pivot nodes to use (higher gives better accuracy but slower performance)
    /// @param {Rng} rng - The random number generator to use for selecting pivots
    /// @returns {SparseSgd} A new Sparse SGD layout instance
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

/// WebAssembly binding for Distance-Adjusted Full SGD layout algorithm.
///
/// This variant enhances the standard Full SGD by dynamically adjusting
/// distances during layout, which can help prevent node collision and
/// improve the overall aesthetic of the layout.
#[wasm_bindgen(js_name = "DistanceAdjustedFullSgd")]
pub struct JsDistanceAdjustedFullSgd {
    sgd: DistanceAdjustedSgd<FullSgd<f32>, f32>,
}

#[wasm_bindgen(js_class = "DistanceAdjustedFullSgd")]
impl JsDistanceAdjustedFullSgd {
    /// Creates a new Distance-Adjusted Full SGD layout instance.
    ///
    /// @param {Graph} graph - The graph to layout
    /// @param {Function} length - A function that takes an edge index and returns its desired length
    /// @returns {DistanceAdjustedFullSgd} A new Distance-Adjusted Full SGD layout instance
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

    /// Applies a single distance-adjusted SGD update step to the layout.
    ///
    /// This variant automatically adjusts distances to avoid node collisions.
    ///
    /// @param {DrawingEuclidean2d} drawing - The drawing to modify
    /// @param {number} eta - The learning rate to use for this update
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

    /// Gets the alpha parameter that controls the strength of distance adjustment.
    ///
    /// @returns {number} The current alpha value
    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f32 {
        self.sgd.alpha
    }

    /// Sets the alpha parameter that controls the strength of distance adjustment.
    ///
    /// @param {number} value - The new alpha value
    #[wasm_bindgen(setter)]
    pub fn set_alpha(&mut self, value: f32) {
        self.sgd.alpha = value;
    }

    /// Gets the minimum distance parameter for the distance adjustment.
    ///
    /// This distance serves as a lower bound for how close nodes can get.
    ///
    /// @returns {number} The current minimum distance
    #[wasm_bindgen(getter, js_name = "minimumDistance")]
    pub fn minimum_distance(&self) -> f32 {
        self.sgd.minimum_distance
    }

    /// Sets the minimum distance parameter for the distance adjustment.
    ///
    /// @param {number} value - The new minimum distance
    #[wasm_bindgen(setter, js_name = "minimumDistance")]
    pub fn set_minimum_distance(&mut self, value: f32) {
        self.sgd.minimum_distance = value;
    }
}

/// WebAssembly binding for Distance-Adjusted Sparse SGD layout algorithm.
///
/// This variant combines the efficiency of Sparse SGD with distance adjustment
/// to improve layout quality for large graphs while preventing node overlap.
#[wasm_bindgen(js_name = "DistanceAdjustedSparseSgd")]
pub struct JsDistanceAdjustedSparseSgd {
    sgd: DistanceAdjustedSgd<SparseSgd<f32>, f32>,
}

#[wasm_bindgen(js_class = "DistanceAdjustedSparseSgd")]
impl JsDistanceAdjustedSparseSgd {
    /// Creates a new Distance-Adjusted Sparse SGD layout instance.
    ///
    /// @param {Graph} graph - The graph to layout
    /// @param {Function} length - A function that takes an edge index and returns its desired length
    /// @param {number} h - The number of pivot nodes to use
    /// @param {Rng} rng - The random number generator to use for selecting pivots
    /// @returns {DistanceAdjustedSparseSgd} A new Distance-Adjusted Sparse SGD layout instance
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
