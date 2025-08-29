//! Learning rate schedulers for SGD algorithms
//!
//! This module provides Python bindings for various learning rate schedulers
//! used in SGD-based layout algorithms.

use petgraph_layout_sgd::{
    Scheduler, SchedulerConstant, SchedulerExponential, SchedulerLinear, SchedulerQuadratic,
    SchedulerReciprocal,
};
use pyo3::prelude::*;

/// Python class that implements a constant learning rate scheduler
///
/// This scheduler maintains a constant learning rate throughout the optimization process.
/// It's the simplest scheduler but may not converge as effectively as decreasing schedules.
///
/// :param t_max: The maximum number of iterations
/// :type t_max: int
/// :param epsilon: The constant learning rate to use
/// :type epsilon: float
#[pyclass]
#[pyo3(name = "SchedulerConstant")]
pub struct PySchedulerConstant {
    scheduler: SchedulerConstant<f32>,
}

impl PySchedulerConstant {
    pub fn new_with_scheduler(scheduler: SchedulerConstant<f32>) -> Self {
        Self { scheduler }
    }
}

#[pymethods]
impl PySchedulerConstant {
    /// Runs the complete schedule, calling the provided function with each learning rate
    ///
    /// :param f: A Python function that takes the current learning rate as a parameter
    /// :type f: callable
    /// :return: None
    /// :rtype: None
    pub fn run(&mut self, f: &Bound<PyAny>) {
        let mut callback = |eta| {
            f.call1((eta as f64,)).ok();
        };
        self.scheduler.run(&mut callback);
    }

    /// Advances the schedule by one step and calls the provided function with the current learning rate
    ///
    /// :param f: A Python function that takes the current learning rate as a parameter
    /// :type f: callable
    /// :return: None
    /// :rtype: None
    pub fn step(&mut self, f: &Bound<PyAny>) {
        let mut callback = |eta| {
            f.call1((eta as f64,)).ok();
        };
        self.scheduler.step(&mut callback);
    }

    /// Checks if the schedule has completed all steps
    ///
    /// :return: True if the schedule is finished, False otherwise
    /// :rtype: bool
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

/// Python class that implements a linear decay learning rate scheduler
///
/// This scheduler decreases the learning rate linearly from the initial value
/// to the final value over the specified number of steps.
///
/// :param t_max: The maximum number of iterations
/// :type t_max: int
/// :param epsilon: The final learning rate (initial rate is 1.0)
/// :type epsilon: float
#[pyclass]
#[pyo3(name = "SchedulerLinear")]
pub struct PySchedulerLinear {
    scheduler: SchedulerLinear<f32>,
}

impl PySchedulerLinear {
    pub fn new_with_scheduler(scheduler: SchedulerLinear<f32>) -> Self {
        Self { scheduler }
    }
}

#[pymethods]
impl PySchedulerLinear {
    /// Runs the complete schedule, calling the provided function with each learning rate
    ///
    /// :param f: A Python function that takes the current learning rate as a parameter
    /// :type f: callable
    /// :return: None
    /// :rtype: None
    pub fn run(&mut self, f: &Bound<PyAny>) {
        let mut callback = |eta| {
            f.call1((eta as f64,)).ok();
        };
        self.scheduler.run(&mut callback);
    }

    /// Advances the schedule by one step and calls the provided function with the current learning rate
    ///
    /// :param f: A Python function that takes the current learning rate as a parameter
    /// :type f: callable
    /// :return: None
    /// :rtype: None
    pub fn step(&mut self, f: &Bound<PyAny>) {
        let mut callback = |eta| {
            f.call1((eta as f64,)).ok();
        };
        self.scheduler.step(&mut callback);
    }

    /// Checks if the schedule has completed all steps
    ///
    /// :return: True if the schedule is finished, False otherwise
    /// :rtype: bool
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

/// Python class that implements a quadratic decay learning rate scheduler
///
/// This scheduler decreases the learning rate according to a quadratic function
/// from the initial value to the final value over the specified number of steps.
///
/// :param t_max: The maximum number of iterations
/// :type t_max: int
/// :param epsilon: The final learning rate (initial rate is 1.0)
/// :type epsilon: float
#[pyclass]
#[pyo3(name = "SchedulerQuadratic")]
pub struct PySchedulerQuadratic {
    scheduler: SchedulerQuadratic<f32>,
}

impl PySchedulerQuadratic {
    pub fn new_with_scheduler(scheduler: SchedulerQuadratic<f32>) -> Self {
        Self { scheduler }
    }
}

#[pymethods]
impl PySchedulerQuadratic {
    /// Runs the complete schedule, calling the provided function with each learning rate
    ///
    /// :param f: A Python function that takes the current learning rate as a parameter
    /// :type f: callable
    /// :return: None
    /// :rtype: None
    pub fn run(&mut self, f: &Bound<PyAny>) {
        let mut callback = |eta| {
            f.call1((eta as f64,)).ok();
        };
        self.scheduler.run(&mut callback);
    }

    /// Advances the schedule by one step and calls the provided function with the current learning rate
    ///
    /// :param f: A Python function that takes the current learning rate as a parameter
    /// :type f: callable
    /// :return: None
    /// :rtype: None
    pub fn step(&mut self, f: &Bound<PyAny>) {
        let mut callback = |eta| {
            f.call1((eta as f64,)).ok();
        };
        self.scheduler.step(&mut callback);
    }

    /// Checks if the schedule has completed all steps
    ///
    /// :return: True if the schedule is finished, False otherwise
    /// :rtype: bool
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

/// Python class that implements an exponential decay learning rate scheduler
///
/// This scheduler decreases the learning rate exponentially from the initial value
/// to the final value over the specified number of steps. This is often the most
/// effective scheduler for graph layout algorithms.
///
/// :param t_max: The maximum number of iterations
/// :type t_max: int
/// :param epsilon: The final learning rate (initial rate is 1.0)
/// :type epsilon: float
#[pyclass]
#[pyo3(name = "SchedulerExponential")]
pub struct PySchedulerExponential {
    scheduler: SchedulerExponential<f32>,
}

impl PySchedulerExponential {
    pub fn new_with_scheduler(scheduler: SchedulerExponential<f32>) -> Self {
        Self { scheduler }
    }
}

#[pymethods]
impl PySchedulerExponential {
    /// Runs the complete schedule, calling the provided function with each learning rate
    ///
    /// :param f: A Python function that takes the current learning rate as a parameter
    /// :type f: callable
    /// :return: None
    /// :rtype: None
    pub fn run(&mut self, f: &Bound<PyAny>) {
        let mut callback = |eta| {
            f.call1((eta as f64,)).ok();
        };
        self.scheduler.run(&mut callback);
    }

    /// Advances the schedule by one step and calls the provided function with the current learning rate
    ///
    /// :param f: A Python function that takes the current learning rate as a parameter
    /// :type f: callable
    /// :return: None
    /// :rtype: None
    pub fn step(&mut self, f: &Bound<PyAny>) {
        let mut callback = |eta| {
            f.call1((eta as f64,)).ok();
        };
        self.scheduler.step(&mut callback);
    }

    /// Checks if the schedule has completed all steps
    ///
    /// :return: True if the schedule is finished, False otherwise
    /// :rtype: bool
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

/// Python class that implements a reciprocal decay learning rate scheduler
///
/// This scheduler decreases the learning rate according to a reciprocal function (1/t)
/// from the initial value to the final value over the specified number of steps.
///
/// :param t_max: The maximum number of iterations
/// :type t_max: int
/// :param epsilon: The final learning rate (initial rate is 1.0)
/// :type epsilon: float
#[pyclass]
#[pyo3(name = "SchedulerReciprocal")]
pub struct PySchedulerReciprocal {
    scheduler: SchedulerReciprocal<f32>,
}

impl PySchedulerReciprocal {
    pub fn new_with_scheduler(scheduler: SchedulerReciprocal<f32>) -> Self {
        Self { scheduler }
    }
}

#[pymethods]
impl PySchedulerReciprocal {
    /// Runs the complete schedule, calling the provided function with each learning rate
    ///
    /// :param f: A Python function that takes the current learning rate as a parameter
    /// :type f: callable
    /// :return: None
    /// :rtype: None
    pub fn run(&mut self, f: &Bound<PyAny>) {
        let mut callback = |eta| {
            f.call1((eta as f64,)).ok();
        };
        self.scheduler.run(&mut callback);
    }

    /// Advances the schedule by one step and calls the provided function with the current learning rate
    ///
    /// :param f: A Python function that takes the current learning rate as a parameter
    /// :type f: callable
    /// :return: None
    /// :rtype: None
    pub fn step(&mut self, f: &Bound<PyAny>) {
        let mut callback = |eta| {
            f.call1((eta as f64,)).ok();
        };
        self.scheduler.step(&mut callback);
    }

    /// Checks if the schedule has completed all steps
    ///
    /// :return: True if the schedule is finished, False otherwise
    /// :rtype: bool
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}
