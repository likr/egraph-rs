use std::marker::PhantomData;

use crate::scheduler::Scheduler;
use petgraph_drawing::DrawingValue;

/// A learning rate scheduler with linear decay.
///
/// This scheduler decreases the learning rate linearly over time from 1.0 to 0.0,
/// following the formula: η(t) = 1.0 - t / (t_max - 1).
///
/// Linear decay provides a steady, predictable decrease in the learning rate,
/// which can be useful for many graph layout applications.
pub struct SchedulerLinear<S> {
    /// Current iteration counter
    t: usize,
    /// Maximum number of iterations
    t_max: usize,
    /// Phantom data to use the generic parameter S
    phantom: PhantomData<S>,
}

impl<S> SchedulerLinear<S>
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

/// Implementation of the Scheduler trait for SchedulerLinear
impl<S> Scheduler<S> for SchedulerLinear<S>
where
    S: DrawingValue,
{
    /// Performs a single step of the scheduling process.
    ///
    /// This method calculates the learning rate using the linear decay formula,
    /// provides it to the callback function, and increments the iteration counter.
    ///
    /// # Parameters
    /// * `callback` - A function that will be called with the calculated learning rate
    fn step<F: FnMut(S)>(&mut self, callback: &mut F) {
        let eta = if self.t_max == 1 {
            S::one()
        } else {
            S::one() - S::from_usize(self.t).unwrap() / S::from_usize(self.t_max - 1).unwrap()
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
