use crate::Scheduler;
use petgraph_drawing::{Delta, Drawing, DrawingValue, Metric};
use rand::prelude::*;

pub trait Sgd<S> {
    fn node_pairs(&self) -> &Vec<(usize, usize, S, S)>;

    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, S, S)>;

    fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        self.node_pairs_mut().shuffle(rng);
    }

    fn apply<Diff, D, M>(&self, drawing: &mut D, eta: S)
    where
        D: Drawing<Item = M>,
        Diff: Delta<S = S>,
        M: Metric<D = Diff>,
        S: DrawingValue,
    {
        for &(i, j, dij, wij) in self.node_pairs().iter() {
            let mu = (eta * wij).min(S::one());
            let delta = drawing.delta(i, j);
            let norm = delta.norm();
            if norm > S::zero() {
                let r = S::from_f32(0.5).unwrap() * mu * (norm - dij) / norm;
                *drawing.raw_entry_mut(i) += delta * -r;
            }
        }
    }

    fn scheduler<SC>(&self, t_max: usize, epsilon: S) -> SC
    where
        SC: Scheduler<S>,
        S: DrawingValue,
    {
        let mut w_min = S::infinity();
        let mut w_max = S::zero();
        for &(_, _, _, wij) in self.node_pairs().iter() {
            if wij == S::zero() {
                continue;
            }
            if wij < w_min {
                w_min = wij;
            }
            if wij > w_max {
                w_max = wij;
            }
        }
        let eta_max = S::one() / w_min;
        let eta_min = epsilon / w_max;
        SC::init(t_max, eta_min, eta_max)
    }

    fn update_distance<F>(&mut self, mut distance: F)
    where
        F: FnMut(usize, usize, S, S) -> S,
        S: Copy,
    {
        for p in self.node_pairs_mut() {
            let (i, j, dij, wij) = p;
            p.2 = distance(*i, *j, *dij, *wij)
        }
    }

    fn update_weight<F>(&mut self, mut weight: F)
    where
        F: FnMut(usize, usize, S, S) -> S,
        S: Copy,
    {
        for p in self.node_pairs_mut() {
            let (i, j, dij, wij) = p;
            p.3 = weight(*i, *j, *dij, *wij)
        }
    }
}
