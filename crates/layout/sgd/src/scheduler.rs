mod scheduler_constant;
mod scheduler_exponential;
mod scheduler_linear;
mod scheduler_quadratic;
mod scheduler_reciprocal;

pub trait Scheduler<S> {
    fn init(t_max: usize, eta_min: S, eta_max: S) -> Self;

    fn run<F: FnMut(S)>(&mut self, callback: &mut F) {
        while !self.is_finished() {
            self.step(callback)
        }
    }

    fn step<F: FnMut(S)>(&mut self, callback: &mut F);

    fn is_finished(&self) -> bool;
}

pub use scheduler_constant::SchedulerConstant;
pub use scheduler_exponential::SchedulerExponential;
pub use scheduler_linear::SchedulerLinear;
pub use scheduler_quadratic::SchedulerQuadratic;
pub use scheduler_reciprocal::SchedulerReciprocal;
