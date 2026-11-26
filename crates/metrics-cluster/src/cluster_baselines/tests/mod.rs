use super::*;

pub(super) fn assert_close(got: f64, expected: f64, tol: f64, name: &str) {
    assert!(
        (got - expected).abs() < tol,
        "{name}: got {got}, expected {expected} (diff = {})",
        (got - expected).abs()
    );
}

pub(super) fn layout_from_pairs(pts: &[(f64, f64)]) -> Layout2d<f64> {
    let xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = pts.iter().map(|p| p.1).collect();
    Layout2d::from_xy(xs, ys).unwrap()
}

pub(super) fn brute_oo_nni(orig: &Layout2d<f64>, adj: &Layout2d<f64>) -> f64 {
    let n = orig.len();
    if n <= 1 {
        return 0.0;
    }
    let mut inv = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if (orig.x()[i] - orig.x()[j]) * (adj.x()[i] - adj.x()[j]) < 0.0 {
                inv += 1;
            }
            if (orig.y()[i] - orig.y()[j]) * (adj.y()[i] - adj.y()[j]) < 0.0 {
                inv += 1;
            }
        }
    }
    inv as f64 / (n * (n - 1)) as f64
}

pub(super) fn brute_gong_edges(layout: &Layout2d<f64>, gamma: f64) -> Vec<(usize, usize)> {
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut edges = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = xs[j] - xs[i];
            let dy = ys[j] - ys[i];
            let d_ij_sq = dx * dx + dy * dy;
            if d_ij_sq <= 0.0 {
                continue;
            }
            let mx = gamma * xs[j] + (1.0 - gamma) * xs[i];
            let my = gamma * ys[j] + (1.0 - gamma) * ys[i];
            let r_sq = (1.0 - gamma).powi(2) * d_ij_sq;
            let mut ok = true;
            for q in 0..n {
                if q == i || q == j {
                    continue;
                }
                let d_mq_sq = (xs[q] - mx).powi(2) + (ys[q] - my).powi(2);
                if d_mq_sq <= r_sq * (1.0 + 1e-12) {
                    ok = false;
                    break;
                }
            }
            if ok {
                edges.push((i, j));
            }
        }
    }
    edges
}

mod nd;
mod nor_sprawlter_and_stability;
mod parity_and_agreement;
mod separation_and_boundaries;
