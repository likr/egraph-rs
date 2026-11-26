use std::collections::HashSet;
use std::f64::consts::TAU;

use egraph_core::{CsrGraph, Layout2d, ManifoldLayout2d, Scalar};
use egraph_linalg::BfsWorkspace;
use egraph_neighborhood::extract_1hop_2hop_csr;

pub(super) fn grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
    let n = rows * cols;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            if r + 1 < rows {
                edges.push((u, ((r + 1) * cols + c) as u32));
            }
            if c + 1 < cols {
                edges.push((u, (r * cols + c + 1) as u32));
            }
        }
    }
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

pub(super) fn barbell_graph(clique_size: usize) -> CsrGraph<f64> {
    let n = 2 * clique_size;
    let mut edges = Vec::new();
    for offset in [0, clique_size] {
        for i in 0..clique_size {
            for j in (i + 1)..clique_size {
                edges.push(((offset + i) as u32, (offset + j) as u32));
            }
        }
    }
    edges.push(((clique_size - 1) as u32, clique_size as u32));
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

pub(super) fn circle_layout<S: Scalar>(n: usize, radius: f64) -> Layout2d<S> {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let angle = TAU * (i as f64) / (n.max(1) as f64);
        let (s, c) = angle.sin_cos();
        x.push(S::from_f64(radius * c));
        y.push(S::from_f64(radius * s));
    }
    Layout2d::from_xy(x, y).unwrap()
}

pub(super) fn euclidean_apsp_stress<S: Scalar>(graph: &CsrGraph<f64>, layout: &Layout2d<S>) -> f64 {
    let n = graph.num_nodes();
    let mut ws = BfsWorkspace::new(n);
    let mut dist = vec![0.0_f64; n];
    let mut stress = 0.0_f64;
    for i in 0..n {
        ws.bfs_from(graph, i, &mut dist).unwrap();
        for (j, &d_g) in dist.iter().enumerate().skip(i + 1) {
            if d_g.is_finite() && d_g > 0.0 {
                let dx = layout.x()[i].to_f64() - layout.x()[j].to_f64();
                let dy = layout.y()[i].to_f64() - layout.y()[j].to_f64();
                let d_e = dx.hypot(dy);
                let diff = d_e - d_g;
                stress += (diff * diff) / (d_g * d_g);
            }
        }
    }
    stress
}

pub(super) fn compute_np2<S: Scalar>(graph: &CsrGraph<f64>, layout: &Layout2d<S>) -> f64 {
    let (adj1, adj2) = extract_1hop_2hop_csr(graph).unwrap();
    let n = graph.num_nodes();
    let mut total = 0.0_f64;
    let mut active = 0usize;
    for u in 0..n {
        let mut true_2hop = HashSet::new();
        for &v in adj1.neighbors(u) {
            true_2hop.insert(v);
        }
        for &v in adj2.neighbors(u) {
            true_2hop.insert(v);
        }
        let k = true_2hop.len();
        if k == 0 {
            continue;
        }
        active += 1;
        let ux = layout.x()[u].to_f64();
        let uy = layout.y()[u].to_f64();
        let mut dists: Vec<(f64, u32)> = (0..n)
            .filter(|&v| v != u)
            .map(|v| {
                let dx = ux - layout.x()[v].to_f64();
                let dy = uy - layout.y()[v].to_f64();
                (dx * dx + dy * dy, v as u32)
            })
            .collect();
        dists.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let hits = dists
            .iter()
            .take(k)
            .filter(|&&(_, v)| true_2hop.contains(&v))
            .count();
        total += (hits as f64) / (k as f64);
    }
    total / (active.max(1) as f64)
}

pub(super) fn manifold_pair_stress<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    pairs: &[(usize, usize, f64)],
) -> f64 {
    let mut total = 0.0_f64;
    for &(i, j, target) in pairs {
        let d = layout.geodesic_distance(i, j).unwrap().to_f64();
        let diff = d - target;
        let w = 1.0 / (target * target);
        total += w * diff * diff;
    }
    total
}
