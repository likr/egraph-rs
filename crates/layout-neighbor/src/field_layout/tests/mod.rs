mod config_and_guards;
mod ib_umap;
mod t_fdp;

use egraph_core::{CsrGraph, DeterministicRng, Layout2d, Scalar};

pub(super) fn grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
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

pub(super) fn barbell_graph(clique_size: usize) -> CsrGraph<f64> {
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

pub(super) fn star_graph(num_leaves: usize) -> CsrGraph<f64> {
    let n = 1 + num_leaves;
    let edges: Vec<(u32, u32)> = (1..=num_leaves as u32).map(|v| (0, v)).collect();
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

pub(super) fn seeded_layout<S: Scalar>(n: usize, scale: f64, seed: u64) -> Layout2d<S> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        x.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
        y.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
    }
    Layout2d::from_xy(x, y).unwrap()
}
