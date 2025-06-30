use crate::scheduler::Scheduler;
use petgraph_drawing::DrawingValue;
use std::marker::PhantomData;

/// A learning rate scheduler that maintains a constant learning rate.
///
/// Unlike other schedulers, `SchedulerConstant` does not decrease the learning rate
/// over time. Instead, it provides a constant learning rate of 1.0 at each step.
/// This scheduler is primarily useful for testing or for cases where a constant
/// learning rate is desired.
pub struct SchedulerConstant<S> {
    /// Current iteration counter
    t: usize,
    /// Maximum number of iterations
    t_max: usize,
    /// Phantom data to use the generic parameter S
    phantom: PhantomData<S>,
}

impl<S> SchedulerConstant<S>
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

/// Implementation of the Scheduler trait for SchedulerConstant
impl<S> Scheduler<S> for SchedulerConstant<S>
where
    S: DrawingValue,
{
    /// Performs a single step of the scheduling process.
    ///
    /// This implementation always provides a learning rate of 1.0 to the callback
    /// and increments the iteration counter.
    ///
    /// # Parameters
    /// * `callback` - A function that will be called with the constant learning rate
    fn step<F: FnMut(S)>(&mut self, callback: &mut F) {
        callback(S::one());
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
