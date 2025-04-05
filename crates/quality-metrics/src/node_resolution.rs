use petgraph_drawing::{Delta, Drawing, DrawingValue, Metric};

/// Calculates the node resolution metric for a graph layout.
///
/// Node resolution evaluates how well nodes are distributed in the drawing space,
/// assessing whether nodes are too close to each other which can hamper readability.
/// The metric is based on a comparison of actual inter-node distances to an ideal
/// minimum distance that depends on the number of nodes and the maximum distance
/// between any two nodes.
///
/// This implementation calculates the sum of squared violations of the minimum
/// distance rule. A lower value indicates better node resolution (fewer violations).
///
/// # Parameters
///
/// * `drawing`: The layout of the graph
///
/// # Returns
///
/// A value of type `S` representing the node resolution metric. Higher values
/// indicate better node spacing.
///
/// # Type Parameters
///
/// * `Diff`: A type for representing differences between metric values
/// * `D`: A drawing type
/// * `M`: Metric type used in the drawing
/// * `S`: Numeric type for calculations
pub fn node_resolution<Diff, D, M, S>(drawing: &D) -> S
where
    D: Drawing<Item = M>,
    Diff: Delta<S = S>,
    M: Copy + Metric<D = Diff>,
    S: DrawingValue,
{
    let n = drawing.len();
    let r = S::one() / S::from_usize(n).unwrap().sqrt();

    let mut d_max = S::zero();
    for i in 1..n {
        for j in 0..i {
            let delta = drawing.delta(i, j);
            d_max = d_max.max(delta.norm());
        }
    }

    let mut s = S::zero();
    for i in 1..n {
        for j in 0..i {
            let delta = drawing.delta(i, j);
            s += (S::one() - delta.norm() / (r * d_max))
                .powi(2)
                .max(S::zero());
        }
    }
    s
}
