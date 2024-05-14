use petgraph::visit::IntoNodeIdentifiers;
use petgraph_drawing::{Delta, Drawing, DrawingValue, Metric};

pub struct OverwrapRemoval<S> {
    radius: Vec<S>,
    pub strength: S,
    pub iterations: usize,
    pub min_distance: S,
}

impl<S> OverwrapRemoval<S>
where
    S: DrawingValue,
{
    pub fn new<G, F>(graph: G, radius: F) -> OverwrapRemoval<S>
    where
        G: IntoNodeIdentifiers,
        F: FnMut(G::NodeId) -> S,
    {
        let mut radius = radius;
        OverwrapRemoval {
            radius: graph
                .node_identifiers()
                .map(|u| radius(u))
                .collect::<Vec<_>>(),
            strength: S::one(),
            iterations: 1,
            min_distance: S::from_f32(1e-3).unwrap(),
        }
    }

    pub fn apply<DR, M, D>(&self, drawing: &mut DR)
    where
        DR: Drawing<Item = M>,
        M: Metric<D = D>,
        D: Delta<S = S>,
    {
        let n = drawing.len();
        for _ in 0..self.iterations {
            for i in 0..n {
                let ri = self.radius[i];
                for j in (i + 1)..n {
                    let rj = self.radius[j];
                    let delta1 = drawing.delta(i, j);
                    let delta2 = drawing.delta(i, j);
                    let r = ri + rj;
                    let l = delta1.norm().max(self.min_distance);
                    if l < r {
                        let d = (r - l) / l * self.strength;
                        let rr = (rj * rj) / (ri * ri + rj * rj);
                        *drawing.raw_entry_mut(i) += delta1 * (d * rr);
                        *drawing.raw_entry_mut(j) -= delta2 * (d * (S::one() - rr));
                    }
                }
            }
        }
    }
}
