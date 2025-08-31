use crate::Constraint;
use petgraph_drawing::{Delta, Drawing, DrawingValue, MetricCartesian};

/// Checks if two 1-dimensional segments overlap
fn overlap_1d<S>(x00: S, x01: S, x10: S, x11: S) -> bool
where
    S: DrawingValue,
{
    (x00 < x11 && x10 < x01) || (x10 < x01 && x00 < x11)
}

/// Checks if two rectangles overlap by comparing their intervals in all dimensions
fn overlap<S>(a: &Vec<(S, S)>, b: &Vec<(S, S)>) -> bool
where
    S: DrawingValue,
{
    a.iter()
        .zip(b.iter())
        .all(|(&(x00, x01), &(x10, x11))| overlap_1d(x00, x01, x10, x11))
}

/// Generates separation constraints to prevent rectangle overlaps.
///
/// This is a legacy function maintained for backward compatibility.
/// It directly calls generate_rectangle_no_overlap_constraints_x or
/// generate_rectangle_no_overlap_constraints_y based on the dimension parameter.
///
/// # Arguments
///
/// * `drawing` - A reference to a `DrawingEuclidean2d` that contains the positions of the nodes.
/// * `size` - A function that returns the size of a node in a given dimension.
/// * `k` - The dimension along which to apply the separation constraint (0 for x, 1 for y).
///
/// # Returns
///
/// A vector of `Constraint` objects representing the separation constraints.
pub fn generate_rectangle_no_overlap_constraints<D, Diff, M, F, S>(
    drawing: &D,
    size: F,
    k: usize,
) -> Vec<Constraint<S>>
where
    D: Drawing<Item = M>,
    D::Index: Clone,
    Diff: Delta<S = S>,
    M: MetricCartesian<D = Diff>,
    F: FnMut(D::Index, usize) -> S,
    S: DrawingValue,
{
    let mut size = size;
    let n = drawing.len();
    let d = drawing.dimension();
    let mut constraints = vec![];
    let sizes = (0..drawing.len())
        .map(|i| {
            let u = drawing.node_id(i);
            let x = drawing.raw_entry(i);
            (0..d)
                .map(|j| {
                    let xj = *x.nth(j);
                    let w = size(u.clone(), j) / (2.).into();
                    (xj - w, xj + w)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for j in 1..n {
        for i in 0..j {
            if overlap(&sizes[i], &sizes[j]) {
                constraints.push(
                    if drawing.raw_entry(i).nth(k) < drawing.raw_entry(j).nth(k) {
                        Constraint::new(
                            i,
                            j,
                            (sizes[i][k].1 - sizes[i][k].0 + sizes[j][k].1 - sizes[j][k].0)
                                / (2.).into(),
                        )
                    } else {
                        Constraint::new(
                            j,
                            i,
                            (sizes[i][k].1 - sizes[i][k].0 + sizes[j][k].1 - sizes[j][k].0)
                                / (2.).into(),
                        )
                    },
                )
            }
        }
    }
    constraints
}
