use crate::scheduler::Scheduler;
use petgraph_drawing::DrawingValue;

/// A learning rate scheduler with reciprocal decay.
///
/// This scheduler decreases the learning rate following a reciprocal (hyperbolic) curve,
/// using the formula: η(t) = a / (1 + b * t), where:
/// a = eta_max, b = (eta_max / eta_min - 1) / (t_max - 1).
///
/// Reciprocal decay provides an initially rapid decrease that slows down over time,
/// which can be effective for finding a balance between exploration and exploitation
/// in graph layout tasks.
pub struct SchedulerReciprocal<S> {
    /// Current iteration counter
    t: usize,
    /// Maximum number of iterations
    t_max: usize,
    /// Scale factor (a parameter)
    a: S,
    /// Decay factor (b parameter)
    b: S,
}

/// Implementation of the Scheduler trait for SchedulerReciprocal
impl<S> Scheduler<S> for SchedulerReciprocal<S>
where
    S: DrawingValue,
{
    /// Initializes a new reciprocal scheduler.
    ///
    /// This method calculates the parameters for the reciprocal decay formula
    /// based on the desired minimum and maximum learning rates and the number of iterations.
    ///
    /// # Parameters
    /// * `t_max` - The maximum number of iterations
    /// * `eta_min` - The minimum learning rate (reached at the end)
    /// * `eta_max` - The maximum learning rate (used at the beginning)
    ///
    /// # Returns
    /// A new SchedulerReciprocal instance
    fn init(t_max: usize, eta_min: S, eta_max: S) -> Self {
        let a = eta_max;
        let b = if t_max == 1 {
            S::zero()
        } else {
            (eta_max / eta_min - S::one()) / S::from_usize(t_max - 1).unwrap()
        };
        Self { t: 0, t_max, a, b }
    }

    /// Performs a single step of the scheduling process.
    ///
    /// This method calculates the learning rate using the reciprocal decay formula,
    /// provides it to the callback function, and increments the iteration counter.
    ///
    /// # Parameters
    /// * `callback` - A function that will be called with the calculated learning rate
    fn step<F: FnMut(S)>(&mut self, callback: &mut F) {
        let eta = self.a / (S::one() + self.b * S::from_usize(self.t).unwrap());
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
