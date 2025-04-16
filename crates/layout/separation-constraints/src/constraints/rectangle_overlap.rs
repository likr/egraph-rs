use petgraph_drawing::{Delta, Drawing, MetricCartesian};

use crate::Constraint;

fn overlap_1d(x00: f32, x01: f32, x10: f32, x11: f32) -> bool {
    (x00 < x11 && x10 < x01) || (x10 < x01 && x00 < x11)
}

fn overlap(a: &Vec<(f32, f32)>, b: &Vec<(f32, f32)>) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(&(x00, x01), &(x10, x11))| overlap_1d(x00, x01, x10, x11))
}

pub fn generate_rectangle_no_overlap_constraints<D, Diff, M, F>(
    drawing: &D,
    size: F,
    k: usize,
) -> Vec<Constraint>
where
    D: Drawing<Item = M>,
    D::Index: Clone,
    Diff: Delta<S = f32>,
    M: MetricCartesian<D = Diff>,
    F: FnMut(D::Index, usize) -> f32,
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
                    let xj = x.nth(j);
                    let w = size(u.clone(), j) / 2.;
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
                            (sizes[i][k].1 - sizes[i][k].0 + sizes[j][k].1 - sizes[j][k].0) / 2.,
                        )
                    } else {
                        Constraint::new(
                            j,
                            i,
                            (sizes[i][k].1 - sizes[i][k].0 + sizes[j][k].1 - sizes[j][k].0) / 2.,
                        )
                    },
                )
            }
        }
    }
    constraints
}
