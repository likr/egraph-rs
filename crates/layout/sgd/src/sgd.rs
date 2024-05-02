use petgraph_drawing::{Delta, Drawing, Metric};
use rand::prelude::*;
use std::f32::INFINITY;

pub trait Sgd {
    fn node_pairs(&self) -> &Vec<(usize, usize, f32, f32)>;

    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, f32, f32)>;

    fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        self.node_pairs_mut().shuffle(rng);
    }

    fn apply<Diff, D, M>(&self, drawing: &mut D, eta: f32)
    where
        D: Drawing<Item = M>,
        Diff: Delta<S = f32>,
        M: Metric<D = Diff>,
    {
        for &(i, j, dij, wij) in self.node_pairs().iter() {
            let mu = (eta * wij).min(1.);
            let delta = drawing.delta(i, j);
            let norm = delta.norm();
            if norm > 0. {
                let r = 0.5 * mu * (norm - dij) / norm;
                *drawing.raw_entry_mut(i) += delta * -r;
            }
        }
    }

    fn scheduler(&self, t_max: usize, epsilon: f32) -> SgdScheduler {
        let mut w_min = INFINITY;
        let mut w_max = 0.;
        for &(_, _, _, wij) in self.node_pairs().iter() {
            if wij == 0. {
                continue;
            }
            if wij < w_min {
                w_min = wij;
            }
            if wij > w_max {
                w_max = wij;
            }
        }
        let eta_max = 1. / w_min;
        let eta_min = epsilon / w_max;
        SgdScheduler {
            t: 0,
            t_max,
            a: eta_max,
            b: (eta_min / eta_max).ln() / (t_max - 1) as f32,
        }
    }

    fn update_distance<F>(&mut self, mut distance: F)
    where
        F: FnMut(usize, usize, f32, f32) -> f32,
    {
        for p in self.node_pairs_mut() {
            let (i, j, dij, wij) = p;
            p.2 = distance(*i, *j, *dij, *wij)
        }
    }

    fn update_weight<F>(&mut self, mut weight: F)
    where
        F: FnMut(usize, usize, f32, f32) -> f32,
    {
        for p in self.node_pairs_mut() {
            let (i, j, dij, wij) = p;
            p.3 = weight(*i, *j, *dij, *wij)
        }
    }
}

pub struct SgdScheduler {
    t: usize,
    t_max: usize,
    a: f32,
    b: f32,
}

impl SgdScheduler {
    pub fn run<F: FnMut(f32)>(&mut self, f: &mut F) {
        while !self.is_finished() {
            self.step(f)
        }
    }

    pub fn step<F: FnMut(f32)>(&mut self, f: &mut F) {
        let eta = self.a * (self.b * self.t as f32).exp();
        f(eta);
        self.t += 1;
    }

    pub fn is_finished(&self) -> bool {
        self.t >= self.t_max
    }
}
