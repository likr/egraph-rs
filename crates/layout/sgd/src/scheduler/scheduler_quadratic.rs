use crate::{scheduler::Scheduler, Sgd};
use petgraph_drawing::DrawingValue;

/// A learning rate scheduler with quadratic decay.
///
/// This scheduler decreases the learning rate following a quadratic curve,
/// using the formula: η(t) = a * (1 - b*t)².
///
/// Quadratic decay produces a slower initial decrease that accelerates over time,
/// which can help maintain sufficient exploration in early stages while ensuring
/// convergence in later stages.
pub struct SchedulerQuadratic<S> {
    /// Current iteration counter
    t: usize,
    /// Maximum number of iterations
    t_max: usize,
    /// Initial learning rate (scaling factor in the quadratic formula)
    a: S,
    /// Rate parameter controlling the curvature of the quadratic decay
    b: S,
}

impl<S> SchedulerQuadratic<S> {
    /// Creates a new quadratic scheduler from an SGD instance.
    ///
    /// This constructor uses the SGD's scheduler method to initialize appropriate parameters
    /// for quadratic decay.
    ///
    /// # Parameters
    /// * `sgd` - The SGD algorithm instance to create a scheduler for
    /// * `t_max` - The maximum number of iterations to run
    /// * `epsilon` - A small value used to calculate the minimum learning rate
    ///
    /// # Returns
    /// A new quadratic scheduler instance
    pub fn new<SGD>(sgd: SGD, t_max: usize, epsilon: S) -> Self
    where
        SGD: Sgd<S>,
        S: DrawingValue,
    {
        sgd.scheduler(t_max, epsilon)
    }
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
        Self {
            t: 0,
            t_max,
            a: eta_max,
            b: (S::one() - (eta_min / eta_max).sqrt()) / S::from_usize(t_max - 1).unwrap(),
        }
    }

    /// Performs a single step of the scheduling process.
    ///
    /// This method calculates the learning rate using the quadratic decay formula,
    /// provides it to the callback function, and increments the iteration counter.
    ///
    /// # Parameters
    /// * `callback` - A function that will be called with the calculated learning rate
    fn step<F: FnMut(S)>(&mut self, callback: &mut F) {
        let eta = self.a
            * (S::one() - self.b * S::from_usize(self.t).unwrap())
            * (S::one() - self.b * S::from_usize(self.t).unwrap());
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
