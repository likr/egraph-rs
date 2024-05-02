use crate::{scheduler::Scheduler, Sgd};
use petgraph_drawing::DrawingValue;
use std::marker::PhantomData;

pub struct SchedulerConstant<S> {
    t: usize,
    t_max: usize,
    phantom: PhantomData<S>,
}

impl<S> SchedulerConstant<S> {
    pub fn new<SGD>(sgd: SGD, t_max: usize, epsilon: S) -> Self
    where
        SGD: Sgd<S>,
        S: DrawingValue,
    {
        sgd.scheduler(t_max, epsilon)
    }
}

impl<S> Scheduler<S> for SchedulerConstant<S>
where
    S: DrawingValue,
{
    fn init(t_max: usize, _eta_min: S, _eta_max: S) -> Self {
        Self {
            t: 0,
            t_max,
            phantom: PhantomData,
        }
    }

    fn step<F: FnMut(S)>(&mut self, callback: &mut F) {
        callback(S::one());
        self.t += 1;
    }

    fn is_finished(&self) -> bool {
        self.t >= self.t_max
    }
}
