use crate::scheduler::Scheduler;
use petgraph_drawing::DrawingValue;

/// A learning rate scheduler with linear decay.
///
/// This scheduler decreases the learning rate linearly over time,
/// following the formula: η(t) = a - b * t.
///
/// Linear decay provides a steady, predictable decrease in the learning rate,
/// which can be useful for many graph layout applications.
pub struct SchedulerLinear<S> {
    /// Current iteration counter
    t: usize,
    /// Maximum number of iterations
    t_max: usize,
    /// Initial learning rate (y-intercept in the linear formula)
    a: S,
    /// Rate of decrease per iteration (slope in the linear formula)
    b: S,
}

/// Implementation of the Scheduler trait for SchedulerLinear
impl<S> Scheduler<S> for SchedulerLinear<S>
where
    S: DrawingValue,
{
    /// Initializes a new linear scheduler.
    ///
    /// This method calculates the parameters for the linear decay formula
    /// based on the desired minimum and maximum learning rates and the number of iterations.
    ///
    /// # Parameters
    /// * `t_max` - The maximum number of iterations
    /// * `eta_min` - The minimum learning rate (reached at the end)
    /// * `eta_max` - The maximum learning rate (used at the beginning)
    ///
    /// # Returns
    /// A new SchedulerLinear instance
    fn init(t_max: usize, eta_min: S, eta_max: S) -> Self {
        Self {
            t: 0,
            t_max,
            a: eta_max,
            b: (eta_max - eta_min) / S::from_usize(t_max - 1).unwrap(),
        }
    }

    /// Performs a single step of the scheduling process.
    ///
    /// This method calculates the learning rate using the linear decay formula,
    /// provides it to the callback function, and increments the iteration counter.
    ///
    /// # Parameters
    /// * `callback` - A function that will be called with the calculated learning rate
    fn step<F: FnMut(S)>(&mut self, callback: &mut F) {
        let eta = self.a - self.b * S::from_usize(self.t).unwrap();
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
