use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue};

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
/// An `S` value in the range [0, 1] representing the aspect ratio metric.
/// A value of 1 indicates a perfectly balanced layout (equal spread in all directions),
/// while lower values indicate more elongated layouts.
///
/// # Type Parameters
///
/// * `N`: Node ID type that implements `DrawingIndex`
pub fn aspect_ratio<N, S>(drawing: &DrawingEuclidean2d<N, S>) -> S
where
    N: DrawingIndex,
    S: DrawingValue,
{
    let n = drawing.len();
    let mut cx = S::zero();
    let mut cy = S::zero();
    for i in 0..n {
        let xi = drawing.raw_entry(i).0;
        let yi = drawing.raw_entry(i).1;
        cx += xi;
        cy += yi;
    }
    cx /= S::from_usize(n).unwrap();
    cy /= S::from_usize(n).unwrap();

    let mut xx = S::zero();
    let mut xy = S::zero();
    let mut yy = S::zero();
    for i in 0..n {
        let xi = drawing.raw_entry(i).0 - cx;
        let yi = drawing.raw_entry(i).1 - cy;
        xx += xi * xi;
        xy += xi * yi;
        yy += yi * yi;
    }

    let tr = xx + yy;
    let det = xx * yy - xy * xy;
    let sigma1 = ((tr + (tr * tr - det * (4.).into()).sqrt()) / (2.).into()).sqrt();
    let sigma2 = ((tr - (tr * tr - det * (4.).into()).sqrt()) / (2.).into()).sqrt();
    sigma2 / sigma1
}
