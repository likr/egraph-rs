use crate::{scheduler::Scheduler, Sgd};
use petgraph_drawing::DrawingValue;

pub struct SchedulerQuadratic<S> {
    t: usize,
    t_max: usize,
    a: S,
    b: S,
}

impl<S> SchedulerQuadratic<S> {
    pub fn new<SGD>(sgd: SGD, t_max: usize, epsilon: S) -> Self
    where
        SGD: Sgd<S>,
        S: DrawingValue,
    {
        sgd.scheduler(t_max, epsilon)
    }
}

impl<S> Scheduler<S> for SchedulerQuadratic<S>
where
    S: DrawingValue,
{
    fn init(t_max: usize, eta_min: S, eta_max: S) -> Self {
        Self {
            t: 0,
            t_max,
            a: eta_max,
            b: (S::one() - (eta_min / eta_max).sqrt()) / S::from_usize(t_max - 1).unwrap(),
        }
    }

    fn step<F: FnMut(S)>(&mut self, callback: &mut F) {
        let eta = self.a
            * (S::one() - self.b * S::from_usize(self.t).unwrap())
            * (S::one() - self.b * S::from_usize(self.t).unwrap());
        callback(eta);
        self.t += 1;
    }

    fn is_finished(&self) -> bool {
        self.t >= self.t_max
    }
}
