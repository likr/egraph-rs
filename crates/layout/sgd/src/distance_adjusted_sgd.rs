use crate::Sgd;
use petgraph_drawing::{Delta, Drawing, DrawingValue, Metric};
use std::collections::HashMap;

pub struct DistanceAdjustedSgd<A, S>
where
    A: Sgd<S>,
{
    pub alpha: S,
    pub minimum_distance: S,
    sgd: A,
    original_distance: HashMap<(usize, usize), S>,
}

impl<A, S> DistanceAdjustedSgd<A, S>
where
    A: Sgd<S>,
{
    pub fn new(sgd: A) -> DistanceAdjustedSgd<A, S>
    where
        S: DrawingValue,
    {
        let mut original_distance = HashMap::new();
        for p in sgd.node_pairs().iter() {
            original_distance.insert((p.0, p.1), p.2);
        }
        Self {
            alpha: S::from_f32(0.5).unwrap(),
            minimum_distance: S::from(0.0).unwrap(),
            sgd,
            original_distance,
        }
    }

    pub fn apply_with_distance_adjustment<D, Diff, M>(&mut self, drawing: &mut D, eta: S)
    where
        D: Drawing<Item = M>,
        Diff: Delta<S = S>,
        M: Metric<D = Diff>,
        S: DrawingValue,
    {
        self.sgd.apply(drawing, eta);
        self.sgd.update_distance(|i, j, _, wij, wji| {
            let delta = drawing.delta(i, j);
            let d1 = delta.norm();
            let d2 = self.original_distance[&(i, j)];
            let w = wij.max(wji);
            let new_d = (self.alpha * w * d1
                + S::from_usize(2).unwrap() * (S::one() - self.alpha) * d2)
                / (self.alpha * w + S::from_usize(2).unwrap() * (S::one() - self.alpha));
            new_d.max(self.minimum_distance).min(d2)
        });
        self.sgd.update_weight(|_, _, d, _| S::one() / (d * d));
    }
}

impl<A, S> Sgd<S> for DistanceAdjustedSgd<A, S>
where
    A: Sgd<S>,
{
    fn node_pairs(&self) -> &Vec<(usize, usize, S, S, S)> {
        self.sgd.node_pairs()
    }

    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, S, S, S)> {
        self.sgd.node_pairs_mut()
    }
}
