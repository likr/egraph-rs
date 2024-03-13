use petgraph_drawing::{DrawingEuclidean2d, DrawingIndex};

pub fn aspect_ratio<N>(drawing: &DrawingEuclidean2d<N, f32>) -> f32
where
    N: DrawingIndex,
{
    let n = drawing.len();
    let mut cx = 0.;
    let mut cy = 0.;
    for i in 0..n {
        let xi = drawing.raw_entry(i).0;
        let yi = drawing.raw_entry(i).1;
        cx += xi;
        cy += yi;
    }
    cx /= n as f32;
    cy /= n as f32;

    let mut xx = 0.;
    let mut xy = 0.;
    let mut yy = 0.;
    for i in 0..n {
        let xi = drawing.raw_entry(i).0 - cx;
        let yi = drawing.raw_entry(i).1 - cy;
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
