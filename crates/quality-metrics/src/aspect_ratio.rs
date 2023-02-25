use petgraph::graph::IndexType;
use petgraph_layout_force_simulation::Coordinates;

pub fn aspect_ratio<Ix: IndexType>(coordinates: &Coordinates<Ix>) -> f32 {
    let mut cx = 0.;
    let mut cy = 0.;
    for p in coordinates.points.iter() {
        let xi = p.x;
        let yi = p.y;
        cx += xi;
        cy += yi;
    }
    cx /= coordinates.len() as f32;
    cy /= coordinates.len() as f32;

    let mut xx = 0.;
    let mut xy = 0.;
    let mut yy = 0.;
    for p in coordinates.points.iter() {
        let xi = p.x - cx;
        let yi = p.y - cy;
        xx += xi * xi;
        xy += xi * yi;
        yy += yi * yi;
    }

    let tr = xx + yy;
    let det = xx * yy - xy * xy;
    let sigma1 = ((tr + (tr * tr - 4. * det).sqrt()) / 2.).sqrt();
    let sigma2 = ((tr - (tr * tr - 4. * det).sqrt()) / 2.).sqrt();
    sigma2 / sigma1
}
