use crate::scheduler::Scheduler;
use petgraph_drawing::DrawingValue;

/// A learning rate scheduler with quadratic decay.
///
/// This scheduler decreases the learning rate following a quadratic curve,
/// using the formula: η(t) = a * (1 - b * t)², where:
/// a = eta_max, b = (1 + sqrt(eta_min / eta_max)) / (t_max - 1).
///
/// Quadratic decay produces a slower initial decrease that accelerates over time,
/// which can help maintain sufficient exploration in early stages while ensuring
/// convergence in later stages.
pub struct SchedulerQuadratic<S> {
    /// Current iteration counter
    t: usize,
    /// Maximum number of iterations
    t_max: usize,
    /// Scale factor (a parameter)
    a: S,
    /// Decay factor (b parameter)
    b: S,
}

/// Implementation of the Scheduler trait for SchedulerQuadratic
impl<S> Scheduler<S> for SchedulerQuadratic<S>
where
    S: DrawingValue,
{
    /// Initializes a new quadratic scheduler.
    ///
    /// This method calculates the parameters for the quadratic decay formula
    /// based on the desired minimum and maximum learning rates and the number of iterations.
    ///
    /// # Parameters
    /// * `t_max` - The maximum number of iterations
    /// * `eta_min` - The minimum learning rate (reached at the end)
    /// * `eta_max` - The maximum learning rate (used at the beginning)
    ///
    /// # Returns
    /// A new SchedulerQuadratic instance
    fn init(t_max: usize, eta_min: S, eta_max: S) -> Self {
        let a = eta_max;
        let b = if t_max == 1 {
            S::zero()
        } else {
            (S::one() + (eta_min / eta_max).sqrt()) / S::from_usize(t_max - 1).unwrap()
        };
        Self { t: 0, t_max, a, b }
    }

    /// Performs a single step of the scheduling process.
    ///
    /// This method calculates the learning rate using the quadratic decay formula,
    /// provides it to the callback function, and increments the iteration counter.
    ///
    /// # Parameters
    /// * `callback` - A function that will be called with the calculated learning rate
    fn step<F: FnMut(S)>(&mut self, callback: &mut F) {
        let factor = S::one() - self.b * S::from_usize(self.t).unwrap();
        let eta = self.a * factor * factor;
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
