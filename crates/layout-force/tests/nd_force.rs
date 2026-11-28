//! Integration tests for 3D (`Layout3d<S>`) and N-D (`LayoutNd<S>`) force-directed solvers.

use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout3d, LayoutNd, Scalar, SolverStatus};
use egraph_layout_force::{
    BarnesHut, Exact, ForceAtlas2Config, ForceAtlas2Solver, ForceBackend, YifanHuConfig,
    YifanHuSolver,
};

fn make_grid_3d_csr(nx: usize, ny: usize, nz: usize) -> CsrGraph<f64> {
    let n = nx * ny * nz;
    let mut edges = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let u = ((z * ny + y) * nx + x) as u32;
                if x + 1 < nx {
                    edges.push((u, u + 1));
                }
                if y + 1 < ny {
                    edges.push((u, u + nx as u32));
                }
                if z + 1 < nz {
                    edges.push((u, u + (nx * ny) as u32));
                }
            }
        }
    }
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

fn make_disconnected_weighted_csr() -> CsrGraph<f64> {
    CsrGraph::from_edges(
        10,
        &[
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 3),
            (7, 8),
        ],
        Some(&[1.0, 2.0, 0.5, 1.5, 3.0, 1.0, 2.5, 1.2]),
        false,
        false,
    )
    .unwrap()
}

fn assert_layout_3d_bit_exact<S: Scalar>(a: &Layout3d<S>, b: &Layout3d<S>) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert_eq!(a.x()[i].to_f64().to_bits(), b.x()[i].to_f64().to_bits());
        assert_eq!(a.y()[i].to_f64().to_bits(), b.y()[i].to_f64().to_bits());
        assert_eq!(a.z()[i].to_f64().to_bits(), b.z()[i].to_f64().to_bits());
    }
}

fn assert_layout_nd_bit_exact<S: Scalar>(a: &LayoutNd<S>, b: &LayoutNd<S>) {
    assert_eq!(a.dims(), b.dims());
    assert_eq!(a.num_nodes(), b.num_nodes());
    for (&va, &vb) in a.coords().iter().zip(b.coords().iter()) {
        assert_eq!(va.to_f64().to_bits(), vb.to_f64().to_bits());
        assert!(va.is_finite());
    }
}

fn assert_step_eq_run<L, Sol>(
    mut make_solver: impl FnMut() -> Sol,
    init: &L,
    step_chunk: usize,
    check_eq: impl Fn(&L, &L),
) where
    L: Clone,
    Sol: IterativeSolver<Target = L>,
{
    let mut s_run = make_solver();
    let mut s_step = make_solver();
    let mut l_run = init.clone();
    let mut l_step = init.clone();

    let st_run = s_run.run(&mut l_run).unwrap();
    assert!(st_run.is_finished());
    assert_eq!(
        s_step.run_steps(&mut l_step, step_chunk).unwrap(),
        s_step.status()
    );
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_run.status(), s_step.status());
    assert_eq!(s_run.iteration(), s_step.iteration());
    assert_eq!(s_run.total_iterations(), s_step.total_iterations());
    check_eq(&l_run, &l_step);
}

#[test]
fn test_force_atlas2_3d_and_nd_step_run_bit_exact() {
    let graph = make_grid_3d_csr(3, 3, 2);
    let n = graph.num_nodes();

    for use_bh in [false, true] {
        let cfg = ForceAtlas2Config::new()
            .with_iterations(18)
            .with_linlog_mode(use_bh)
            .with_strong_gravity(!use_bh)
            .with_outbound_attraction_distribution(use_bh);
        let backend = if use_bh {
            ForceBackend::barnes_hut(0.5)
        } else {
            ForceBackend::exact()
        };
        let init64 = Layout3d::<f64>::init(n, 101, 4.0).unwrap();
        assert_step_eq_run(
            || {
                ForceAtlas2Solver::<Layout3d<f64>, ForceBackend>::with_backend(
                    &graph,
                    cfg.clone(),
                    backend.clone(),
                )
                .unwrap()
            },
            &init64,
            5,
            assert_layout_3d_bit_exact,
        );

        let init32 = Layout3d::<f32>::init(n, 101, 4.0).unwrap();
        assert_step_eq_run(
            || {
                ForceAtlas2Solver::<Layout3d<f32>, ForceBackend>::with_backend(
                    &graph,
                    cfg.clone(),
                    backend.clone(),
                )
                .unwrap()
            },
            &init32,
            5,
            assert_layout_3d_bit_exact,
        );
    }

    // 5D exact repulsion bit-exactness across f64 & f32 (LayoutNd defaults to Exact).
    let cfg_5d = ForceAtlas2Config::new().with_iterations(12);
    let init5_64 = LayoutNd::<f64>::init(n, 5, 303, 2.0).unwrap();
    assert_step_eq_run(
        || ForceAtlas2Solver::<LayoutNd<f64>>::new(&graph, cfg_5d.clone()).unwrap(),
        &init5_64,
        4,
        assert_layout_nd_bit_exact,
    );
    let init5_32 = LayoutNd::<f32>::init(n, 5, 303, 2.0).unwrap();
    assert_step_eq_run(
        || ForceAtlas2Solver::<LayoutNd<f32>, Exact>::new(&graph, cfg_5d.clone()).unwrap(),
        &init5_32,
        4,
        assert_layout_nd_bit_exact,
    );

    let s = ForceAtlas2Solver::<LayoutNd<f64>>::new(&graph, cfg_5d).unwrap();
    assert_eq!(s.status(), SolverStatus::Initialized);
    assert_eq!(s.iteration(), 0);
    assert_eq!(s.total_iterations(), Some(12));
    assert_eq!(s.backend(), &Exact::new());
}

#[test]
fn test_force_atlas2_3d_overlap_and_yifan_hu_3d_nd() {
    let edgeless = CsrGraph::<()>::from_unweighted_edges(4, &[], false, false).unwrap();
    let init3 = Layout3d::from_xyz(
        vec![0.0, 0.0, 0.2, 0.0],
        vec![0.0, 0.0, 0.0, 0.3],
        vec![0.0, 0.0, 0.0, 0.4],
    )
    .unwrap();
    let cfg_ov = ForceAtlas2Config::new()
        .with_iterations(20)
        .with_prevent_overlap(true)
        .with_default_node_radius(1.0);
    assert_step_eq_run(
        || ForceAtlas2Solver::<Layout3d<f64>>::new_unweighted(&edgeless, cfg_ov.clone()).unwrap(),
        &init3,
        3,
        assert_layout_3d_bit_exact,
    );

    let mut l_ov = init3;
    ForceAtlas2Solver::<Layout3d<f64>>::new_unweighted(&edgeless, cfg_ov)
        .unwrap()
        .run(&mut l_ov)
        .unwrap();
    let d01 = ((l_ov.x()[0] - l_ov.x()[1]).powi(2)
        + (l_ov.y()[0] - l_ov.y()[1]).powi(2)
        + (l_ov.z()[0] - l_ov.z()[1]).powi(2))
    .sqrt();
    assert!(d01 > 1.0, "expected overlap separation in 3D, got {d01}");

    // 4D overlap repulsion (defaults to Exact on LayoutNd).
    let init4 = LayoutNd::<f64>::init(4, 4, 19, 0.5).unwrap();
    let cfg_ov4 = ForceAtlas2Config::new()
        .with_iterations(15)
        .with_prevent_overlap(true)
        .with_default_node_radius(1.0);
    assert_step_eq_run(
        || ForceAtlas2Solver::<LayoutNd<f64>>::new_unweighted(&edgeless, cfg_ov4.clone()).unwrap(),
        &init4,
        4,
        assert_layout_nd_bit_exact,
    );

    // Yifan Hu 3D octree + 3D exact + 4D/1D exact.
    let graph = make_grid_3d_csr(4, 4, 4);
    let cfg = YifanHuConfig::new()
        .with_max_iterations_per_level(15)
        .with_min_coarsen_size(10)
        .with_min_coarsen_ratio(0.75);
    assert_step_eq_run(
        || YifanHuSolver::<Layout3d<f64>>::new(&graph, cfg.clone()).unwrap(),
        &Layout3d::<f64>::init(64, 404, 4.0).unwrap(),
        7,
        assert_layout_3d_bit_exact,
    );
    assert_step_eq_run(
        || {
            YifanHuSolver::<Layout3d<f64>, Exact>::with_backend(&graph, cfg.clone(), Exact::new())
                .unwrap()
        },
        &Layout3d::<f64>::init(64, 404, 4.0).unwrap(),
        7,
        assert_layout_3d_bit_exact,
    );

    let disc = make_disconnected_weighted_csr();
    let cfg_small = YifanHuConfig::new()
        .with_max_iterations_per_level(12)
        .with_min_coarsen_size(4);
    assert_step_eq_run(
        || YifanHuSolver::<LayoutNd<f64>>::new(&disc, cfg_small.clone()).unwrap(),
        &LayoutNd::<f64>::init(10, 4, 505, 3.0).unwrap(),
        5,
        assert_layout_nd_bit_exact,
    );
    assert_step_eq_run(
        || YifanHuSolver::<LayoutNd<f64>>::new(&disc, cfg_small.clone()).unwrap(),
        &LayoutNd::<f64>::init(10, 1, 606, 3.0).unwrap(),
        4,
        assert_layout_nd_bit_exact,
    );
}

#[test]
fn test_nd_degenerates_and_fail_fast_guards() {
    let g_disc = make_disconnected_weighted_csr();

    // Invalid BarnesHut theta fails fast at construction.
    assert!(matches!(
        ForceAtlas2Solver::<Layout3d<f64>, BarnesHut>::with_backend(
            &g_disc,
            ForceAtlas2Config::default(),
            BarnesHut::new(-0.1),
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        YifanHuSolver::<Layout3d<f64>, BarnesHut>::with_backend(
            &g_disc,
            YifanHuConfig::default(),
            BarnesHut::new(f64::NAN),
        ),
        Err(CoreError::InvalidInput(_))
    ));

    let mut fa2 = ForceAtlas2Solver::<LayoutNd<f64>>::new(
        &g_disc,
        ForceAtlas2Config::new().with_iterations(5),
    )
    .unwrap();
    let mut yh = YifanHuSolver::<LayoutNd<f64>>::new(&g_disc, YifanHuConfig::default()).unwrap();

    let mut l_bad_len = LayoutNd::<f64>::zeros(3, 3).unwrap();
    assert!(matches!(
        fa2.run_steps(&mut l_bad_len, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        yh.run_steps(&mut l_bad_len, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l3 = LayoutNd::<f64>::init(10, 3, 2, 1.0).unwrap();
    let mut l4 = LayoutNd::<f64>::init(10, 4, 1, 1.0).unwrap();
    fa2.step(&mut l3).unwrap();
    assert!(matches!(
        fa2.step(&mut l4),
        Err(CoreError::DimensionMismatch { .. })
    ));

    yh.step(&mut l3).unwrap();
    assert!(matches!(
        yh.step(&mut l4),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
