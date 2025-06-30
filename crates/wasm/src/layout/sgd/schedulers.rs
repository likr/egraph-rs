//! Learning rate schedulers for layout algorithms in WebAssembly.
//!
//! This module provides WebAssembly bindings for various learning rate schedulers
//! used in iterative optimization algorithms like SGD. These schedulers control
//! how the learning rate changes over iterations, which affects convergence behavior.

use js_sys::Function;
use petgraph_layout_sgd::{
    Scheduler, SchedulerConstant, SchedulerExponential, SchedulerLinear, SchedulerQuadratic,
    SchedulerReciprocal,
};
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
    #[wasm_bindgen(constructor)]
    pub fn new(t_max: usize) -> Self {
        Self {
            scheduler: SchedulerConstant::new(t_max),
        }
    }

    /// Runs the complete scheduling process, applying the learning rate to each iteration.
    ///
    /// Takes a callback function that receives the current learning rate at each step.
    pub fn run(&mut self, f: &Function) {
        self.scheduler.run(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    /// Performs a single step of the scheduler, advancing to the next iteration.
    ///
    /// Takes a callback function that receives the current learning rate.
    pub fn step(&mut self, f: &Function) {
        self.scheduler.step(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    /// Checks if the scheduling process has completed all iterations.
    ///
    /// Returns true if all iterations have been completed.
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
    #[wasm_bindgen(constructor)]
    pub fn new(t_max: usize) -> Self {
        Self {
            scheduler: SchedulerLinear::new(t_max),
        }
    }

    /// Runs the complete scheduling process, applying the learning rate to each iteration.
    ///
    /// Takes a callback function that receives the current learning rate at each step.
    pub fn run(&mut self, f: &Function) {
        self.scheduler.run(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    /// Performs a single step of the scheduler, advancing to the next iteration.
    ///
    /// Takes a callback function that receives the current learning rate.
    pub fn step(&mut self, f: &Function) {
        self.scheduler.step(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    /// Checks if the scheduling process has completed all iterations.
    ///
    /// Returns true if all iterations have been completed.
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
    #[wasm_bindgen(constructor)]
    pub fn new(t_max: usize) -> Self {
        Self {
            scheduler: SchedulerQuadratic::new(t_max),
        }
    }

    /// Runs the complete scheduling process, applying the learning rate to each iteration.
    ///
    /// Takes a callback function that receives the current learning rate at each step.
    pub fn run(&mut self, f: &Function) {
        self.scheduler.run(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    /// Performs a single step of the scheduler, advancing to the next iteration.
    ///
    /// Takes a callback function that receives the current learning rate.
    pub fn step(&mut self, f: &Function) {
        self.scheduler.step(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    /// Checks if the scheduling process has completed all iterations.
    ///
    /// Returns true if all iterations have been completed.
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
    #[wasm_bindgen(constructor)]
    pub fn new(t_max: usize) -> Self {
        Self {
            scheduler: SchedulerExponential::new(t_max),
        }
    }

    /// Runs the complete scheduling process, applying the learning rate to each iteration.
    ///
    /// Takes a callback function that receives the current learning rate at each step.
    pub fn run(&mut self, f: &Function) {
        self.scheduler.run(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    /// Performs a single step of the scheduler, advancing to the next iteration.
    ///
    /// Takes a callback function that receives the current learning rate.
    pub fn step(&mut self, f: &Function) {
        self.scheduler.step(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    /// Checks if the scheduling process has completed all iterations.
    ///
    /// Returns true if all iterations have been completed.
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
    #[wasm_bindgen(constructor)]
    pub fn new(t_max: usize) -> Self {
        Self {
            scheduler: SchedulerReciprocal::new(t_max),
        }
    }

    /// Runs the complete scheduling process, applying the learning rate to each iteration.
    ///
    /// Takes a callback function that receives the current learning rate at each step.
    pub fn run(&mut self, f: &Function) {
        self.scheduler.run(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    /// Performs a single step of the scheduler, advancing to the next iteration.
    ///
    /// Takes a callback function that receives the current learning rate.
    pub fn step(&mut self, f: &Function) {
        self.scheduler.step(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    /// Checks if the scheduling process has completed all iterations.
    ///
    /// Returns true if all iterations have been completed.
    #[wasm_bindgen(js_name = "isFinished")]
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}
