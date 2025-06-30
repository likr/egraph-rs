use std::marker::PhantomData;

use crate::scheduler::Scheduler;
use petgraph_drawing::DrawingValue;

/// A learning rate scheduler with exponential decay.
///
/// This scheduler decreases the learning rate exponentially over time from 1.0 to 0.01,
/// following the formula: η(t) = exp(-ln(100) * t / (t_max - 1)).
///
/// Exponential decay creates a learning rate that decreases quickly at first
/// and then more gradually, which can help achieve faster convergence in many cases.
pub struct SchedulerExponential<S> {
    /// Current iteration counter
    t: usize,
    /// Maximum number of iterations
    t_max: usize,
    /// Phantom data to use the generic parameter S
    phantom: PhantomData<S>,
}

impl<S> SchedulerExponential<S>
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

/// Implementation of the Scheduler trait for SchedulerExponential
impl<S> Scheduler<S> for SchedulerExponential<S>
where
    S: DrawingValue,
{
    /// Performs a single step of the scheduling process.
    ///
    /// This method calculates the learning rate using the exponential decay formula,
    /// provides it to the callback function, and increments the iteration counter.
    ///
    /// # Parameters
    /// * `callback` - A function that will be called with the calculated learning rate
    fn step<F: FnMut(S)>(&mut self, callback: &mut F) {
        let eta = if self.t_max == 1 {
            S::one()
        } else {
            // Exponential decay from 1.0 to 0.01: exp(-ln(100) * t / (t_max - 1))
            let progress = S::from_usize(self.t).unwrap() / S::from_usize(self.t_max - 1).unwrap();
            let ln_100 = S::from_f32(4.605170).unwrap(); // ln(100) ≈ 4.605170
            (-ln_100 * progress).exp()
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
