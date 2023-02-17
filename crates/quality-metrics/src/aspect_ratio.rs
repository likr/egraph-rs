use petgraph::graph::{Graph, IndexType};
use petgraph::EdgeType;
use petgraph_layout_force_simulation::Coordinates;
use std::f32::{consts::PI, MAX, MIN};

pub fn aspect_ratio<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    coordinates: &Coordinates<Ix>,
) -> f32 {
    let n = graph.node_count();

    let mut cx = 0.;
    let mut cy = 0.;
    for u in graph.node_indices() {
        let (x, y) = coordinates.position(u).unwrap();
        cx += x;
        cy += y;
    }
    cx /= n as f32;
    cy /= n as f32;

    let mut ratio = MAX;
    let repeat = 7;
    for i in 0..repeat {
        let mut left = MAX;
        let mut right = MIN;
        let mut top = MAX;
        let mut bottom = MIN;
        let t = 2. * PI * i as f32 / n as f32;
        let cost = t.cos();
        let sint = t.sin();
        for u in graph.node_indices() {
            let (x, y) = coordinates.position(u).unwrap();
            let xo = x - cx;
            let yo = y - cy;
            let rx = xo * cost - yo * sint;
            let ry = xo * sint + yo * cost;
            left = left.min(rx);
            right = right.max(rx);
            top = top.min(ry);
            bottom = bottom.max(ry);
        }
        let w = right - left;
        let h = bottom - top;
        ratio = ratio.min(w.min(h) / w.max(h));
    }
    ratio
}
