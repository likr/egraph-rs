mod config_and_constructors;
mod convergence_and_backends;
mod jacobi_and_lr;
mod manifold;
mod nd;

use egraph_core::{CsrGraph, DeterministicRng, Layout2d, Scalar};

use super::TsNetStage;

pub(super) fn make_stages(s1: usize, s2: usize, s3: usize, lambda_r3: f64) -> Vec<TsNetStage> {
    vec![
        TsNetStage::new()
            .with_iterations(s1)
            .with_exaggeration(4.0)
            .with_lambda_kl(1.0)
            .with_lambda_c(0.1)
            .with_lambda_r(0.0),
        TsNetStage::new()
            .with_iterations(s2)
            .with_exaggeration(1.0)
            .with_lambda_kl(1.0)
            .with_lambda_c(0.01)
            .with_lambda_r(0.0),
        TsNetStage::new()
            .with_iterations(s3)
            .with_exaggeration(1.0)
            .with_lambda_kl(1.0)
            .with_lambda_c(0.0)
            .with_lambda_r(lambda_r3),
    ]
}

pub(super) fn make_barbell_graph(clique_size: usize) -> CsrGraph<f64> {
    let n = 2 * clique_size;
    let mut edges = Vec::new();
    for c in 0..2 {
        let off = (c * clique_size) as u32;
        for i in 0..clique_size as u32 {
            for j in (i + 1)..clique_size as u32 {
                edges.push((off + i, off + j));
            }
        }
    }
    edges.push((0, clique_size as u32));
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

pub(super) fn make_grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
    let n = rows * cols;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            if c + 1 < cols {
                edges.push((u, u + 1));
            }
            if r + 1 < rows {
                edges.push((u, u + cols as u32));
            }
        }
    }
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

pub(super) fn seeded_layout<S: Scalar>(n: usize, scale: f64, seed: u64) -> Layout2d<S> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        xs.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
        ys.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
    }
    Layout2d::from_xy(xs, ys).unwrap()
}

pub(super) fn rms_radius<S: Scalar>(layout: &Layout2d<S>) -> f64 {
    let n = layout.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    for (&x, &y) in layout.x().iter().zip(layout.y().iter()) {
        let xf = x.to_f64();
        let yf = y.to_f64();
        sum_sq += xf * xf + yf * yf;
    }
    (sum_sq / (n as f64)).sqrt()
}

pub(super) fn mean_edge_and_all_distances(
    graph: &CsrGraph<f64>,
    layout: &Layout2d<f64>,
) -> (f64, f64) {
    let n = layout.len();
    let mut edge_sum = 0.0;
    let mut edge_cnt = 0usize;
    for u in 0..n {
        for &v in graph.neighbors(u) {
            let vi = v as usize;
            if u < vi {
                edge_sum += (layout.x()[u] - layout.x()[vi]).hypot(layout.y()[u] - layout.y()[vi]);
                edge_cnt += 1;
            }
        }
    }
    let mut all_sum = 0.0;
    let mut all_cnt = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            all_sum += (layout.x()[i] - layout.x()[j]).hypot(layout.y()[i] - layout.y()[j]);
            all_cnt += 1;
        }
    }
    (
        edge_sum / (edge_cnt.max(1) as f64),
        all_sum / (all_cnt.max(1) as f64),
    )
}
