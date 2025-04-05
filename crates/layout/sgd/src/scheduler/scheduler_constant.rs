use crate::{scheduler::Scheduler, Sgd};
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

impl<S> SchedulerConstant<S> {
    /// Creates a new constant scheduler from an SGD instance.
    ///
    /// This constructor uses the SGD's scheduler method to initialize appropriate parameters.
    ///
    /// # Parameters
    /// * `sgd` - The SGD algorithm instance to create a scheduler for
    /// * `t_max` - The maximum number of iterations to run
    /// * `epsilon` - A small value used to calculate the minimum learning rate
    ///
    /// # Returns
    /// A new constant scheduler instance
    pub fn new<SGD>(sgd: SGD, t_max: usize, epsilon: S) -> Self
    where
        SGD: Sgd<S>,
        S: DrawingValue,
    {
        sgd.scheduler(t_max, epsilon)
    }
}

/// Implementation of the Scheduler trait for SchedulerConstant
impl<S> Scheduler<S> for SchedulerConstant<S>
where
    S: DrawingValue,
{
    /// Initializes a new constant scheduler.
    ///
    /// This implementation ignores the eta_min and eta_max parameters since
    /// it always uses a constant learning rate of 1.0.
    ///
    /// # Parameters
    /// * `t_max` - The maximum number of iterations
    /// * `_eta_min` - Ignored for constant scheduler
    /// * `_eta_max` - Ignored for constant scheduler
    ///
    /// # Returns
    /// A new SchedulerConstant instance
    fn init(t_max: usize, _eta_min: S, _eta_max: S) -> Self {
        Self {
            t: 0,
            t_max,
            phantom: PhantomData,
        }
    }

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
