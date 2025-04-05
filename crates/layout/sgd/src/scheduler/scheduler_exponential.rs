use crate::{scheduler::Scheduler, Sgd};
use petgraph_drawing::DrawingValue;

/// A learning rate scheduler with exponential decay.
///
/// This scheduler decreases the learning rate exponentially over time,
/// following the formula: η(t) = a * exp(b * t).
///
/// Exponential decay creates a learning rate that decreases quickly at first
/// and then more gradually, which can help achieve faster convergence in many cases.
pub struct SchedulerExponential<S> {
    /// Current iteration counter
    t: usize,
    /// Maximum number of iterations
    t_max: usize,
    /// Coefficient in the exponential formula
    a: S,
    /// Decay rate in the exponential formula
    b: S,
}

impl<S> SchedulerExponential<S> {
    /// Creates a new exponential scheduler from an SGD instance.
    ///
    /// This constructor uses the SGD's scheduler method to initialize appropriate parameters
    /// for exponential decay.
    ///
    /// # Parameters
    /// * `sgd` - The SGD algorithm instance to create a scheduler for
    /// * `t_max` - The maximum number of iterations to run
    /// * `epsilon` - A small value used to calculate the minimum learning rate
    ///
    /// # Returns
    /// A new exponential scheduler instance
    pub fn new<SGD>(sgd: SGD, t_max: usize, epsilon: S) -> Self
    where
        SGD: Sgd<S>,
        S: DrawingValue,
    {
        sgd.scheduler(t_max, epsilon)
    }
}

/// Implementation of the Scheduler trait for SchedulerExponential
impl<S> Scheduler<S> for SchedulerExponential<S>
where
    S: DrawingValue,
{
    /// Initializes a new exponential scheduler.
    ///
    /// This method calculates the parameters for the exponential decay formula
    /// based on the desired minimum and maximum learning rates and the number of iterations.
    ///
    /// # Parameters
    /// * `t_max` - The maximum number of iterations
    /// * `eta_min` - The minimum learning rate (reached at the end)
    /// * `eta_max` - The maximum learning rate (used at the beginning)
    ///
    /// # Returns
    /// A new SchedulerExponential instance
    fn init(t_max: usize, eta_min: S, eta_max: S) -> Self {
        Self {
            t: 0,
            t_max,
            a: eta_max,
            b: (eta_min / eta_max).ln() / S::from_usize(t_max - 1).unwrap(),
        }
    }

    /// Performs a single step of the scheduling process.
    ///
    /// This method calculates the learning rate using the exponential decay formula,
    /// provides it to the callback function, and increments the iteration counter.
    ///
    /// # Parameters
    /// * `callback` - A function that will be called with the calculated learning rate
    fn step<F: FnMut(S)>(&mut self, callback: &mut F) {
        let eta = self.a * (self.b * S::from_usize(self.t).unwrap()).exp();
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
