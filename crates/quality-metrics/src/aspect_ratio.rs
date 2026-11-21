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
    if n <= 1 {
        return S::zero();
    }
    let mut cx = S::zero();
    let mut cy = S::zero();
    for i in 0..n {
        let xi = drawing.raw_entry(i).0;
        let yi = drawing.raw_entry(i).1;
        if !xi.is_finite() || !yi.is_finite() {
            return S::zero();
        }
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
    // Clamp numerical roundoff below 0 on collinear or degenerate zero-span drawings.
    let disc = (tr * tr - det * (4.).into()).max(S::zero());
    let sqrt_disc = disc.sqrt();
    let lam1 = (tr + sqrt_disc) / (2.).into();
    let lam2 = ((tr - sqrt_disc) / (2.).into()).max(S::zero());
    if !lam1.is_finite() || lam1 <= S::zero() {
        return S::zero();
    }
    let sigma1 = lam1.sqrt();
    let sigma2 = lam2.sqrt();
    if !sigma1.is_finite() || sigma1 <= S::zero() || !sigma2.is_finite() {
        return S::zero();
    }
    sigma2 / sigma1
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::{NodeIndex, UnGraph};
    use petgraph_drawing::MetricEuclidean2d;

    #[test]
    fn test_aspect_ratio_degenerate_cases_return_zero() {
        let g0 = UnGraph::<(), ()>::new_undirected();
        let d0 = DrawingEuclidean2d::<NodeIndex, f64>::initial_placement(&g0);
        assert_eq!(aspect_ratio(&d0), 0.0);

        let mut g1 = UnGraph::<(), ()>::new_undirected();
        g1.add_node(());
        let d1 = DrawingEuclidean2d::<NodeIndex, f64>::initial_placement(&g1);
        assert_eq!(aspect_ratio(&d1), 0.0);

        let mut g3 = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..3).map(|_| g3.add_node(())).collect();
        let mut d_zero = DrawingEuclidean2d::<NodeIndex, f64>::initial_placement(&g3);
        for &u in &nodes {
            *d_zero.position_mut(u).unwrap() = MetricEuclidean2d(0.0, 0.0);
        }
        assert_eq!(aspect_ratio(&d_zero), 0.0);

        let mut d_nan = DrawingEuclidean2d::<NodeIndex, f64>::initial_placement(&g3);
        *d_nan.position_mut(nodes[0]).unwrap() = MetricEuclidean2d(f64::NAN, 0.0);
        assert_eq!(aspect_ratio(&d_nan), 0.0);
    }
}
