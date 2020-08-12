use crate::{Force, Point, MIN_DISTANCE};
use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;

pub struct CollideForce {
    radius: Vec<f32>,
    strength: f32,
    iterations: usize,
}

impl CollideForce {
    pub fn new<
        N,
        E,
        Ty: EdgeType,
        Ix: IndexType,
        F: FnMut(&Graph<N, E, Ty, Ix>, NodeIndex<Ix>) -> f32,
    >(
        graph: &Graph<N, E, Ty, Ix>,
        mut radius_accessor: F,
        strength: f32,
        iterations: usize,
    ) -> CollideForce {
        let radius = graph
            .node_indices()
            .map(|u| radius_accessor(graph, u))
            .collect::<Vec<_>>();
        CollideForce {
            radius,
            strength,
            iterations,
        }
    }
}

impl Force for CollideForce {
    fn apply(&self, points: &mut Vec<Point>, _alpha: f32) {
        let n = points.len();
        for _ in 0..self.iterations {
            for i in 0..n {
                let xi = points[i].x + points[i].vx;
                let yi = points[i].y + points[i].vy;
                let ri = self.radius[i];
                for j in (i + 1)..n {
                    let xj = points[j].x + points[j].vx;
                    let yj = points[j].y + points[j].vy;
                    let rj = self.radius[j];
                    let dx = xi - xj;
                    let dy = yi - yj;
                    let r = ri + rj;
                    let l2 = (dx * dx + dy * dy).max(MIN_DISTANCE);
                    if l2 < r * r {
                        let l = l2.sqrt();
                        let d = (r - l) / l * self.strength;
                        let rr = (rj * rj) / (ri * ri + rj * rj);
                        points[i].vx += (dx * d) * rr;
                        points[i].vy += (dy * d) * rr;
                        points[j].vx -= (dx * d) * (1. - rr);
                        points[j].vy -= (dy * d) * (1. - rr);
                    }
                }
            }
        }
    }
}
