use std::collections::HashSet;
use std::f64::consts::TAU;

use egraph_core::{
    CsrGraph, IterativeSolver, Layout2d, LayoutNd, ManifoldKind, ManifoldLayout2d,
    RiemannianLayout, Scalar, SolverStatus,
};
use egraph_neighborhood::extract_1hop_2hop_csr;

pub(crate) const ALL_MANIFOLDS: [ManifoldKind; 4] = [
    ManifoldKind::HyperbolicPoincare,
    ManifoldKind::HyperbolicLorentz,
    ManifoldKind::Spherical,
    ManifoldKind::Torus2d,
];

pub(crate) fn grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
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

pub(crate) fn barbell_graph(clique_size: usize) -> CsrGraph<f64> {
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

pub(crate) fn circle_layout<S: Scalar>(n: usize, radius: f64) -> Layout2d<S> {
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

pub(crate) fn layout_2d_to_nd<S: Scalar>(l2: &Layout2d<S>) -> LayoutNd<S> {
    let n = l2.len();
    let mut flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        flat.push(l2.x()[i]);
        flat.push(l2.y()[i]);
    }
    LayoutNd::from_flat(n, 2, &flat).unwrap()
}

pub(crate) fn compute_np2<S: Scalar>(graph: &CsrGraph<f64>, layout: &Layout2d<S>) -> f64 {
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

pub(crate) fn assert_manifold_step_eq_run<S, Sol>(
    mut s_run: Sol,
    mut s_step: Sol,
    kind: ManifoldKind,
    n: usize,
    label: &str,
) where
    S: Scalar,
    Sol: IterativeSolver<Target = ManifoldLayout2d<S>>,
{
    let mut l_run = ManifoldLayout2d::<S>::init(kind, n, 42, 0.35).unwrap();
    let mut l_step = l_run.clone();

    let st_run = s_run.run(&mut l_run).unwrap();
    assert_eq!(st_run, SolverStatus::Completed, "{label}");

    let first_k = (s_step.total_iterations().unwrap() / 3).max(1);
    s_step.run_steps(&mut l_step, first_k).unwrap();
    if !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    let st_step = s_step.run(&mut l_step).unwrap();
    assert_eq!(st_step, SolverStatus::Completed, "{label}");
    assert_eq!(s_run.iteration(), s_step.iteration(), "{label}");
    assert!(l_run.as_layout().is_all_finite(), "{label}: finite");

    for i in 0..n {
        assert_eq!(
            l_run.x()[i].to_f64().to_bits(),
            l_step.x()[i].to_f64().to_bits(),
            "{label} {kind:?} x[{i}]"
        );
        assert_eq!(
            l_run.y()[i].to_f64().to_bits(),
            l_step.y()[i].to_f64().to_bits(),
            "{label} {kind:?} y[{i}]"
        );
    }
}

pub(crate) fn assert_riemannian_step_eq_run<L, Sol>(
    mut s_run: Sol,
    mut s_step: Sol,
    init: &L,
    step_chunk: usize,
    label: &str,
) -> L
where
    L: RiemannianLayout + Clone + PartialEq + std::fmt::Debug,
    Sol: IterativeSolver<Target = L>,
{
    let mut l_run = init.clone();
    let mut l_step = init.clone();
    assert_eq!(s_run.status(), SolverStatus::Initialized, "{label}");
    assert_eq!(
        s_run.run(&mut l_run).unwrap(),
        SolverStatus::Completed,
        "{label}"
    );

    let total = s_step.total_iterations().unwrap();
    if step_chunk > 0 && total > step_chunk {
        assert_eq!(
            s_step.run_steps(&mut l_step, step_chunk).unwrap(),
            SolverStatus::Running,
            "{label}"
        );
        assert_eq!(s_step.iteration(), step_chunk, "{label}");
    }
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_step.status(), SolverStatus::Completed, "{label}");
    assert_eq!(s_step.iteration(), total, "{label}");
    assert_eq!(
        s_step.step(&mut l_step).unwrap(),
        SolverStatus::Completed,
        "{label}"
    );
    l_run.validate_finite().expect("finite output");
    assert_eq!(l_run, l_step, "{label}: step() != run()");
    l_run
}
