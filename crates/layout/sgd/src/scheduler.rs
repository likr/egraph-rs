//! Schedulers for controlling the learning rate in SGD algorithms.
//!
//! This module provides various implementations of learning rate schedulers
//! that can be used to decrease the learning rate over time during the SGD
//! layout process. Different schedulers offer different decay patterns,
//! allowing for flexibility in controlling the convergence behavior.

mod scheduler_constant;
mod scheduler_exponential;
mod scheduler_linear;
mod scheduler_quadratic;
mod scheduler_reciprocal;

/// Trait for learning rate schedulers in SGD algorithms.
///
/// Schedulers control how the learning rate changes over time during the layout process.
/// Typically, the learning rate starts high to allow for bigger movements and then
/// gradually decreases to allow for fine-tuning the final positions.
///
/// The generic parameter `S` represents the scalar type used for calculations
/// (typically `f32` or `f64`).
pub trait Scheduler<S> {
    /// Initializes a new scheduler with the specified parameters.
    ///
    /// # Parameters
    /// * `t_max` - The maximum number of iterations the scheduler will run
    /// * `eta_min` - The minimum learning rate (typically reached at the end of the process)
    /// * `eta_max` - The maximum learning rate (typically used at the beginning of the process)
    ///
    /// # Returns
    /// A new scheduler instance configured with the provided parameters
    fn init(t_max: usize, eta_min: S, eta_max: S) -> Self;

    /// Runs the complete scheduling process from start to finish.
    ///
    /// This method repeatedly calls `step()` until `is_finished()` returns true,
    /// providing the calculated learning rate to the callback function at each step.
    ///
    /// # Parameters
    /// * `callback` - A function that receives the current learning rate at each step
    fn run<F: FnMut(S)>(&mut self, callback: &mut F) {
        while !self.is_finished() {
            self.step(callback)
        }
    }

    /// Performs a single step of the scheduling process.
    ///
    /// This method calculates the learning rate for the current step,
    /// provides it to the callback function, and updates the internal state.
    ///
    /// # Parameters
    /// * `callback` - A function that receives the calculated learning rate
    fn step<F: FnMut(S)>(&mut self, callback: &mut F);

    /// Checks if the scheduling process is complete.
    ///
    /// # Returns
    /// `true` if the scheduler has reached its maximum number of iterations,
    /// `false` otherwise
    fn is_finished(&self) -> bool;
}

pub use scheduler_constant::SchedulerConstant;
pub use scheduler_exponential::SchedulerExponential;
pub use scheduler_linear::SchedulerLinear;
pub use scheduler_quadratic::SchedulerQuadratic;
pub use scheduler_reciprocal::SchedulerReciprocal;
