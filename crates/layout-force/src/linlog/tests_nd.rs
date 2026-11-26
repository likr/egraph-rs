use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, LayoutNd, Scalar, SolverStatus};

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

fn assert_step_eq_run_nd<S: Scalar>(cfg: LinLogConfig, graph: &CsrGraph<f64>, init: &LayoutNd<S>) {
    let mut s_run = LinLogSolver::<S>::new(graph, cfg).unwrap();
    let mut s_step = LinLogSolver::<S>::new(graph, cfg).unwrap();
    let mut l_run = init.copy();
    let mut l_step = init.copy();

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
        assert_eq!(a.to_f32().to_bits(), b.to_f32().to_bits());
        assert!(a.is_finite());
    }
}

#[test]
fn test_linlog_nd_step_run_bit_exact_barnes_hut_and_energy() {
    let graph = planted_clusters_nd();
    let n = graph.num_nodes();

    for use_bh in [false, true] {
        for (a_exp, r_exp) in [(0.0, -1.0), (0.5, -1.5)] {
            let cfg = LinLogConfig {
                iterations: 20,
                use_barnes_hut: use_bh,
                attraction_exponent: a_exp,
                repulsion_exponent: r_exp,
                ..Default::default()
            };
            let init64 = LayoutNd::<f64>::init(n, 3, 17, 3.0).unwrap();
            assert_step_eq_run_nd(cfg, &graph, &init64);
            let init32 = LayoutNd::<f32>::init(n, 3, 17, 3.0).unwrap();
            assert_step_eq_run_nd(cfg, &graph, &init32);
        }
    }

    // 4D exact bit-exactness across f64 & f32 and energy decrease.
    let cfg_4d = LinLogConfig {
        iterations: 30,
        use_barnes_hut: false,
        ..Default::default()
    };
    let init4_64 = LayoutNd::<f64>::init(n, 4, 29, 3.0).unwrap();
    assert_step_eq_run_nd(cfg_4d, &graph, &init4_64);
    let init4_32 = LayoutNd::<f32>::init(n, 4, 29, 3.0).unwrap();
    assert_step_eq_run_nd(cfg_4d, &graph, &init4_32);

    let mut l4_e = init4_64.copy();
    let mut s4 = LinLogSolver::<f64>::new(&graph, cfg_4d).unwrap();
    let e0 = s4.energy_nd(&l4_e).unwrap();
    s4.run(&mut l4_e).unwrap();
    let e1 = s4.energy_nd(&l4_e).unwrap();
    assert!(e1 < e0, "expected 4D LinLog energy decrease: {e0} -> {e1}");

    // 3D BarnesHut theta=0.0 matches Exact for both r=-1.0 and r=-1.5.
    let init3 = LayoutNd::<f64>::init(n, 3, 63, 4.0).unwrap();
    for (a_exp, r_exp) in [(0.0, -1.0), (0.5, -1.5)] {
        let mut l_ex = init3.copy();
        let mut l_bh0 = init3.copy();
        LinLogSolver::<f64>::new(
            &graph,
            LinLogConfig {
                iterations: 20,
                use_barnes_hut: false,
                attraction_exponent: a_exp,
                repulsion_exponent: r_exp,
                ..Default::default()
            },
        )
        .unwrap()
        .run(&mut l_ex)
        .unwrap();
        let mut s_bh0 = LinLogSolver::<f64>::new(
            &graph,
            LinLogConfig {
                iterations: 20,
                use_barnes_hut: true,
                theta: 0.0,
                attraction_exponent: a_exp,
                repulsion_exponent: r_exp,
                ..Default::default()
            },
        )
        .unwrap();
        let eb0 = s_bh0.energy_nd(&l_bh0).unwrap();
        s_bh0.run(&mut l_bh0).unwrap();
        let eb1 = s_bh0.energy_nd(&l_bh0).unwrap();
        assert!(eb1 < eb0);
        for (&a, &b) in l_ex.coords().iter().zip(l_bh0.coords().iter()) {
            assert!((a - b).abs() < 1e-8);
        }
    }

    // 2D LayoutNd energy_nd and optimization match Layout2d.
    let mut l2 = Layout2d::<f64>::init(n, 71, 2.0).unwrap();
    let flat2: Vec<f64> = l2
        .x()
        .iter()
        .zip(l2.y().iter())
        .flat_map(|(&x, &y)| [x, y])
        .collect();
    let mut lnd2 = LayoutNd::from_flat(n, 2, flat2).unwrap();
    let cfg2 = LinLogConfig {
        iterations: 15,
        use_barnes_hut: true,
        ..Default::default()
    };
    let mut s2 = LinLogSolver::<f64>::new(&graph, cfg2).unwrap();
    let mut snd2 = LinLogSolver::<f64>::new(&graph, cfg2).unwrap();
    assert!((s2.energy(&l2).unwrap() - snd2.energy_nd(&lnd2).unwrap()).abs() < 1e-12);
    s2.run(&mut l2).unwrap();
    snd2.run(&mut lnd2).unwrap();
    for i in 0..n {
        assert!((l2.x()[i] - lnd2.coords()[i * 2]).abs() < 1e-12);
        assert!((l2.y()[i] - lnd2.coords()[i * 2 + 1]).abs() < 1e-12);
    }
    assert_eq!(snd2.status(), SolverStatus::Completed);
    assert_eq!(snd2.iteration(), 15);
    assert_eq!(snd2.total_iterations(), 15);
}

#[test]
fn test_linlog_nd_degenerates_and_error_guards() {
    for n in [0usize, 1, 5] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let mut l = LayoutNd::<f64>::zeros(n, 3).unwrap();
        let mut s = LinLogSolver::<f64>::new(
            &g,
            LinLogConfig {
                iterations: 5,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(s.run(&mut l).unwrap().is_finished());
        assert!(l.is_all_finite());
        assert!(s.energy_nd(&l).unwrap().is_finite());
    }

    let disc = CsrGraph::<f64>::from_edges(6, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let l_collinear = LayoutNd::from_flat(
        6,
        3,
        (0..6)
            .flat_map(|i| [i as f64, 2.0 * i as f64, 3.0 * i as f64])
            .collect(),
    )
    .unwrap();
    let l_planar = LayoutNd::from_flat(
        6,
        3,
        vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.0, 2.0, 1.0,
            0.0,
        ],
    )
    .unwrap();
    for init in [&l_collinear, &l_planar] {
        let mut l = init.copy();
        LinLogSolver::<f64>::new(
            &disc,
            LinLogConfig {
                iterations: 5,
                use_barnes_hut: true,
                ..Default::default()
            },
        )
        .unwrap()
        .run(&mut l)
        .unwrap();
        assert!(l.is_all_finite());
    }

    let mut s_bh = LinLogSolver::<f64>::new(&disc, LinLogConfig::default()).unwrap();
    let mut l4 = LayoutNd::<f64>::init(6, 4, 1, 1.0).unwrap();
    assert!(matches!(
        s_bh.step(&mut l4),
        Err(CoreError::UnsupportedOperation(_))
    ));

    let mut s_ex = LinLogSolver::<f64>::new(
        &disc,
        LinLogConfig {
            iterations: 5,
            use_barnes_hut: false,
            ..Default::default()
        },
    )
    .unwrap();
    let mut l_bad_len = LayoutNd::<f64>::zeros(2, 3).unwrap();
    assert!(matches!(
        s_ex.step(&mut l_bad_len),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        s_ex.energy_nd(&l_bad_len),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l_inf = LayoutNd::from_flat(6, 3, {
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
        s_ex.energy_nd(&l_inf),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut l3 = LayoutNd::<f64>::init(6, 3, 2, 1.0).unwrap();
    s_ex.step(&mut l3).unwrap();
    assert!(matches!(
        s_ex.step(&mut l4),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
