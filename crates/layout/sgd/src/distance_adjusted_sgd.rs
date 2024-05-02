use crate::Sgd;
use petgraph_drawing::{Delta, Drawing, Metric};
use std::collections::HashMap;

pub struct DistanceAdjustedSgd<A>
where
    A: Sgd,
{
    pub alpha: f32,
    pub minimum_distance: f32,
    sgd: A,
    original_distance: HashMap<(usize, usize), f32>,
}

impl<A> DistanceAdjustedSgd<A>
where
    A: Sgd,
{
    pub fn new(sgd: A) -> DistanceAdjustedSgd<A> {
        let mut original_distance = HashMap::new();
        for p in sgd.node_pairs().iter() {
            original_distance.insert((p.0, p.1), p.2);
        }
        Self {
            alpha: 0.5,
            minimum_distance: 0.0,
            sgd,
            original_distance,
        }
    }

    pub fn apply_with_distance_adjustment<D, Diff, M>(&mut self, drawing: &mut D, eta: f32)
    where
        D: Drawing<Item = M>,
        Diff: Delta<S = f32>,
        M: Metric<D = Diff>,
    {
        self.sgd.apply(drawing, eta);
        self.sgd.update_distance(|i, j, _, w| {
            let delta = drawing.delta(i, j);
            let d1 = delta.norm();
            let d2 = self.original_distance[&(i, j)];
            let new_d = (self.alpha * w * d1 + 2. * (1. - self.alpha) * d2)
                / (self.alpha * w + 2. * (1. - self.alpha));
            new_d.clamp(self.minimum_distance, d2)
        });
        self.sgd.update_weight(|_, _, d, _| 1. / (d * d));
    }
}

impl<A> Sgd for DistanceAdjustedSgd<A>
where
    A: Sgd,
{
    fn node_pairs(&self) -> &Vec<(usize, usize, f32, f32)> {
        self.sgd.node_pairs()
    }

    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, f32, f32)> {
        self.sgd.node_pairs_mut()
    }
}
