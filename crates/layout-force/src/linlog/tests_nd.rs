use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout3d, LayoutNd, Scalar, SolverStatus};

use crate::backend::{BarnesHut, Exact, ForceBackend};

use super::*;

fn planted_clusters_nd() -> CsrGraph<f64> {
    let mut edges = Vec::new();
    for i in 0..4usize {
        for j in (i + 1)..4 {
            edges.push((i as u32, j as u32));
        }
    }
    for i in 4..10usize {
        for j in (i + 1)..10 {
            edges.push((i as u32, j as u32));
        }
    }
    edges.push((3, 4));
    CsrGraph::from_edges(10, &edges, None, false, false).unwrap()
}

fn assert_step_eq_run_3d<S: Scalar>(
    cfg: LinLogConfig,
    backend: ForceBackend,
    graph: &CsrGraph<f64>,
    init: &Layout3d<S>,
) {
    let mut s_run = LinLogSolver::<Layout3d<S>, ForceBackend>::with_backend(
        graph,
        cfg.clone(),
        backend.clone(),
    )
    .unwrap();
    let mut s_step =
        LinLogSolver::<Layout3d<S>, ForceBackend>::with_backend(graph, cfg, backend).unwrap();
    let mut l_run = init.clone();
    let mut l_step = init.clone();

    assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
    s_step.run_steps(&mut l_step, 7).unwrap();
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_run.status(), s_step.status());
    assert_eq!(s_run.iteration(), s_step.iteration());
    assert_eq!(s_run.total_iterations(), s_step.total_iterations());
    for i in 0..init.len() {
        assert_eq!(
            l_run.x()[i].to_f64().to_bits(),
            l_step.x()[i].to_f64().to_bits()
        );
        assert_eq!(
            l_run.y()[i].to_f64().to_bits(),
            l_step.y()[i].to_f64().to_bits()
        );
        assert_eq!(
            l_run.z()[i].to_f64().to_bits(),
            l_step.z()[i].to_f64().to_bits()
        );
    }
}

fn assert_step_eq_run_nd<S: Scalar>(cfg: LinLogConfig, graph: &CsrGraph<f64>, init: &LayoutNd<S>) {
    let mut s_run = LinLogSolver::<LayoutNd<S>>::new(graph, cfg.clone()).unwrap();
    let mut s_step = LinLogSolver::<LayoutNd<S>, Exact>::new(graph, cfg).unwrap();
    let mut l_run = init.clone();
    let mut l_step = init.clone();

    assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
    s_step.run_steps(&mut l_step, 7).unwrap();
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_run.status(), s_step.status());
    assert_eq!(s_run.iteration(), s_step.iteration());
    assert_eq!(s_run.total_iterations(), s_step.total_iterations());
    for (&a, &b) in l_run.coords().iter().zip(l_step.coords().iter()) {
        assert_eq!(a.to_f64().to_bits(), b.to_f64().to_bits());
        assert!(a.is_finite());
    }
}

#[test]
fn test_linlog_3d_octree_nd_exact_and_unified_energy() {
    let graph = planted_clusters_nd();
    let n = graph.num_nodes();

    for use_bh in [false, true] {
        let backend = if use_bh {
            ForceBackend::barnes_hut(1.2)
        } else {
            ForceBackend::exact()
        };
        for (a_exp, r_exp) in [(0.0, -1.0), (0.5, -1.5)] {
            let cfg = LinLogConfig::new()
                .with_iterations(20)
                .with_attraction_exponent(a_exp)
                .with_repulsion_exponent(r_exp);
            let init64 = Layout3d::<f64>::init(n, 17, 3.0).unwrap();
            assert_step_eq_run_3d(cfg.clone(), backend.clone(), &graph, &init64);
            let init32 = Layout3d::<f32>::init(n, 17, 3.0).unwrap();
            assert_step_eq_run_3d(cfg, backend.clone(), &graph, &init32);
        }
    }

    // 4D exact bit-exactness across f64 & f32 and unified energy(&layout) decrease.
    let cfg_4d = LinLogConfig::new().with_iterations(30);
    let init4_64 = LayoutNd::<f64>::init(n, 4, 29, 3.0).unwrap();
    assert_step_eq_run_nd(cfg_4d.clone(), &graph, &init4_64);
    let init4_32 = LayoutNd::<f32>::init(n, 4, 29, 3.0).unwrap();
    assert_step_eq_run_nd(cfg_4d.clone(), &graph, &init4_32);

    let mut l4_e = init4_64;
    let mut s4 = LinLogSolver::<LayoutNd<f64>>::new(&graph, cfg_4d).unwrap();
    let e0 = s4.energy(&l4_e).unwrap();
    s4.run(&mut l4_e).unwrap();
    let e1 = s4.energy(&l4_e).unwrap();
    assert!(e1 < e0, "expected 4D LinLog energy decrease: {e0} -> {e1}");

    // 3D Layout3d BarnesHut theta=0.0 matches Exact for both r=-1.0 and r=-1.5.
    let init3 = Layout3d::<f64>::init(n, 63, 4.0).unwrap();
    for (a_exp, r_exp) in [(0.0, -1.0), (0.5, -1.5)] {
        let mut l_ex = init3.clone();
        let mut l_bh0 = init3.clone();
        LinLogSolver::<Layout3d<f64>, Exact>::new(
            &graph,
            LinLogConfig::new()
                .with_iterations(20)
                .with_attraction_exponent(a_exp)
                .with_repulsion_exponent(r_exp),
        )
        .unwrap()
        .run(&mut l_ex)
        .unwrap();
        let mut s_bh0 = LinLogSolver::<Layout3d<f64>>::with_backend(
            &graph,
            LinLogConfig::new()
                .with_iterations(20)
                .with_attraction_exponent(a_exp)
                .with_repulsion_exponent(r_exp),
            BarnesHut::new(0.0),
        )
        .unwrap();
        let eb0 = s_bh0.energy(&l_bh0).unwrap();
        s_bh0.run(&mut l_bh0).unwrap();
        let eb1 = s_bh0.energy(&l_bh0).unwrap();
        assert!(eb1 < eb0);
        for i in 0..n {
            assert!((l_ex.x()[i] - l_bh0.x()[i]).abs() < 1e-8);
            assert!((l_ex.y()[i] - l_bh0.y()[i]).abs() < 1e-8);
            assert!((l_ex.z()[i] - l_bh0.z()[i]).abs() < 1e-8);
        }
    }
}

#[test]
fn test_linlog_nd_fail_fast_construction_and_energy_guards() {
    let disc = CsrGraph::<f64>::from_edges(6, &[(0, 1), (2, 3)], None, false, false).unwrap();

    // Invalid BarnesHut theta fails fast at construction.
    let err = LinLogSolver::<Layout3d<f64>, BarnesHut>::with_backend(
        &disc,
        LinLogConfig::default(),
        BarnesHut::new(-0.5),
    );
    assert!(matches!(err, Err(CoreError::InvalidInput(_))));

    let mut s_ex =
        LinLogSolver::<LayoutNd<f64>>::new(&disc, LinLogConfig::new().with_iterations(5)).unwrap();
    let mut l_bad_len = LayoutNd::<f64>::zeros(2, 3).unwrap();
    assert!(matches!(
        s_ex.step(&mut l_bad_len),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        s_ex.energy(&l_bad_len),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l_inf = LayoutNd::from_flat(6, 3, &{
        let mut v = vec![0.0; 18];
        v[4] = f64::INFINITY;
        v
    })
    .unwrap();
    assert!(matches!(
        s_ex.step(&mut l_inf),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        s_ex.energy(&l_inf),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut l3 = LayoutNd::<f64>::init(6, 3, 2, 1.0).unwrap();
    let mut l4 = LayoutNd::<f64>::init(6, 4, 1, 1.0).unwrap();
    s_ex.step(&mut l3).unwrap();
    assert!(matches!(
        s_ex.step(&mut l4),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
