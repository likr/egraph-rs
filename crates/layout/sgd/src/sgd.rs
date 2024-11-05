use crate::Scheduler;
use petgraph_drawing::{Delta, Drawing, DrawingValue, Metric};
use rand::prelude::*;

pub trait Sgd<S> {
    fn node_pairs(&self) -> &Vec<(usize, usize, S, S, S)>;

    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, S, S, S)>;

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
        for &(i, j, dij, wij, wji) in self.node_pairs().iter() {
            let mu_i = (eta * wij).min(S::one());
            let mu_j = (eta * wji).min(S::one());
            let delta = drawing.delta(i, j);
            let norm = delta.norm();
            if norm > S::zero() {
                let r = S::from_f32(0.5).unwrap() * (norm - dij) / norm;
                *drawing.raw_entry_mut(i) += delta.clone() * -r * mu_i;
                *drawing.raw_entry_mut(j) += delta.clone() * r * mu_j;
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
        for &(_, _, _, wij, wji) in self.node_pairs().iter() {
            for w in [wij, wji] {
                if w == S::zero() {
                    continue;
                }
                if w < w_min {
                    w_min = w;
                }
                if w > w_max {
                    w_max = w;
                }
            }
        }
        let eta_max = S::one() / w_min;
        let eta_min = epsilon / w_max;
        SC::init(t_max, eta_min, eta_max)
    }

    fn update_distance<F>(&mut self, mut distance: F)
    where
        F: FnMut(usize, usize, S, S, S) -> S,
        S: Copy,
    {
        for p in self.node_pairs_mut() {
            let (i, j, dij, wij, wji) = p;
            p.2 = distance(*i, *j, *dij, *wij, *wji)
        }
    }

    fn update_weight<F>(&mut self, mut weight: F)
    where
        F: FnMut(usize, usize, S, S) -> S,
        S: Copy,
    {
        for p in self.node_pairs_mut() {
            let (i, j, dij, wij, wji) = p;
            p.3 = weight(*i, *j, *dij, *wij);
            p.4 = weight(*i, *j, *dij, *wji);
        }
    }
}
