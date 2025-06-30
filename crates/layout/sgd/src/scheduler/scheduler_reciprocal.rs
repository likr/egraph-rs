use std::marker::PhantomData;

use crate::scheduler::Scheduler;
use petgraph_drawing::DrawingValue;

/// A learning rate scheduler with reciprocal decay.
///
/// This scheduler decreases the learning rate following a reciprocal (hyperbolic) curve from 1.0 to 0.01,
/// using the formula: η(t) = 1 / (1 + 99*t/(t_max-1)).
///
/// Reciprocal decay provides an initially rapid decrease that slows down over time,
/// which can be effective for finding a balance between exploration and exploitation
/// in graph layout tasks.
pub struct SchedulerReciprocal<S> {
    /// Current iteration counter
    t: usize,
    /// Maximum number of iterations
    t_max: usize,
    /// Phantom data to use the generic parameter S
    phantom: std::marker::PhantomData<S>,
}

impl<S> SchedulerReciprocal<S>
where
    S: DrawingValue,
{
    pub fn new(t_max: usize) -> Self {
        Self {
            t: 0,
            t_max,
            phantom: PhantomData,
        }
    }
}

/// Implementation of the Scheduler trait for SchedulerReciprocal
impl<S> Scheduler<S> for SchedulerReciprocal<S>
where
    S: DrawingValue,
{
    /// Performs a single step of the scheduling process.
    ///
    /// This method calculates the learning rate using the reciprocal decay formula,
    /// provides it to the callback function, and increments the iteration counter.
    ///
    /// # Parameters
    /// * `callback` - A function that will be called with the calculated learning rate
    fn step<F: FnMut(S)>(&mut self, callback: &mut F) {
        let eta = if self.t_max == 1 {
            S::one()
        } else {
            // Reciprocal decay from 1.0 to 0.01: 1 / (1 + 99*t/(t_max-1))
            let progress = S::from_usize(self.t).unwrap() / S::from_usize(self.t_max - 1).unwrap();
            let decay_factor = S::from_f32(99.0).unwrap() * progress;
            S::one() / (S::one() + decay_factor)
        };
        callback(eta);
        self.t += 1;
    }

    /// Checks if the scheduling process is complete.
    ///
    /// # Returns
    /// `true` if the current iteration count has reached the maximum, `false` otherwise
    fn is_finished(&self) -> bool {
        self.t >= self.t_max
    }
}
