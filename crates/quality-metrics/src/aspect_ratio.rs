use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex};

/// Calculates the aspect ratio metric for a graph layout.
///
/// The aspect ratio metric evaluates the balance between width and height of the drawing.
/// It is calculated as the ratio of the smaller to the larger eigenvalue of the covariance
/// matrix of node positions. A value closer to 1 indicates a more balanced, circular layout,
/// while a value closer to 0 indicates a highly elongated layout.
///
/// This metric computes the principal components of the node positions and returns
/// the ratio of the smaller to the larger eigenvalue, which represents how close
/// the layout is to being uniformly distributed in all directions.
///
/// # Parameters
///
/// * `drawing`: The 2D Euclidean layout of the graph
///
/// # Returns
///
/// An `f32` value in the range [0, 1] representing the aspect ratio metric.
/// A value of 1 indicates a perfectly balanced layout (equal spread in all directions),
/// while lower values indicate more elongated layouts.
///
/// # Type Parameters
///
/// * `N`: Node ID type that implements `DrawingIndex`
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
