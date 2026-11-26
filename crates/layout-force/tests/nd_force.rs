//! Integration tests for 3D and N-D force-directed and FDEB solvers (`LayoutNd<S>`).

use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, LayoutNd, Scalar,
    SolverStatus,
};
use egraph_layout_force::continuous_fdp::{
    ContinuousFdp, ForceAtlas2Config, ForceAtlas2Solver, YifanHuConfig, YifanHuSolver,
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

fn assert_layout_nd_bit_exact<S: Scalar>(a: &LayoutNd<S>, b: &LayoutNd<S>) {
    assert_eq!(a.dims(), b.dims());
    assert_eq!(a.num_nodes(), b.num_nodes());
    for (&va, &vb) in a.coords().iter().zip(b.coords().iter()) {
        assert_eq!(va.to_f64().to_bits(), vb.to_f64().to_bits());
        assert_eq!(va.to_f32().to_bits(), vb.to_f32().to_bits());
        assert!(va.is_finite());
    }
}

fn assert_step_eq_run_nd<S: Scalar, Sol>(
    mut make_solver: impl FnMut() -> Sol,
    init: &LayoutNd<S>,
    step_chunk: usize,
) where
    Sol: IterativeSolver<S, LayoutNd<S>>,
{
    let mut s_run = make_solver();
    let mut s_step = make_solver();
    let mut l_run = init.copy();
    let mut l_step = init.copy();

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
    assert_layout_nd_bit_exact(&l_run, &l_step);
}

#[test]
fn test_force_atlas2_nd_step_run_bit_exact_and_oracle_parity() {
    let graph = make_grid_3d_csr(3, 3, 2);
    let n = graph.num_nodes();

    for use_bh in [false, true] {
        let cfg = ForceAtlas2Config {
            iterations: 18,
            use_barnes_hut: use_bh,
            theta: 0.5,
            linlog_mode: use_bh,
            strong_gravity: !use_bh,
            outbound_attraction_distribution: use_bh,
            ..Default::default()
        };
        let init64 = LayoutNd::<f64>::init(n, 3, 101, 4.0).unwrap();
        assert_step_eq_run_nd(
            || ForceAtlas2Solver::<f64>::new(&graph, cfg).unwrap(),
            &init64,
            5,
        );

        let init32 = LayoutNd::<f32>::init(n, 3, 101, 4.0).unwrap();
        assert_step_eq_run_nd(
            || ForceAtlas2Solver::<f32>::new(&graph, cfg).unwrap(),
            &init32,
            5,
        );
    }

    // 5D exact repulsion bit-exactness across f64 & f32.
    let cfg_5d = ForceAtlas2Config {
        iterations: 12,
        use_barnes_hut: false,
        ..Default::default()
    };
    let init5_64 = LayoutNd::<f64>::init(n, 5, 303, 2.0).unwrap();
    assert_step_eq_run_nd(
        || ForceAtlas2Solver::<f64>::new(&graph, cfg_5d).unwrap(),
        &init5_64,
        4,
    );
    let init5_32 = LayoutNd::<f32>::init(n, 5, 303, 2.0).unwrap();
    assert_step_eq_run_nd(
        || ForceAtlas2Solver::<f32>::new(&graph, cfg_5d).unwrap(),
        &init5_32,
        4,
    );

    // 3D Barnes-Hut theta=0.0 matches exact O(N^2).
    let init3 = LayoutNd::<f64>::init(n, 3, 77, 5.0).unwrap();
    let mut l_exact = init3.copy();
    let mut l_bh0 = init3.copy();
    ForceAtlas2Solver::<f64>::new(
        &graph,
        ForceAtlas2Config {
            iterations: 15,
            use_barnes_hut: false,
            ..Default::default()
        },
    )
    .unwrap()
    .run(&mut l_exact)
    .unwrap();
    ForceAtlas2Solver::<f64>::new(
        &graph,
        ForceAtlas2Config {
            iterations: 15,
            use_barnes_hut: true,
            theta: 0.0,
            ..Default::default()
        },
    )
    .unwrap()
    .run(&mut l_bh0)
    .unwrap();
    for (&a, &b) in l_exact.coords().iter().zip(l_bh0.coords().iter()) {
        assert!((a - b).abs() < 1e-8);
    }

    // Inherent status methods remain unambiguous even with IterativeSolver in scope.
    let s = ForceAtlas2Solver::<f64>::new(&graph, cfg_5d).unwrap();
    assert_eq!(s.status(), SolverStatus::Initialized);
    assert_eq!(s.iteration(), 0);
    assert_eq!(s.total_iterations(), 12);
}

#[test]
fn test_force_atlas2_nd_overlap_and_continuous_fdp() {
    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    // Coincident and near-overlapping vertices in 3D with prevent_overlap.
    let mut coords = vec![
        0.0, 0.0, 0.0, // 0 & 1 coincident
        0.0, 0.0, 0.0, //
        0.2, 0.0, 0.0, // overlapping within radius sum = 2.0
        0.0, 0.3, 0.4,
    ];
    let init = LayoutNd::<f64>::from_flat(4, 3, coords.clone()).unwrap();
    let cfg_ov = ForceAtlas2Config {
        iterations: 20,
        use_barnes_hut: true,
        prevent_overlap: true,
        default_node_radius: 1.0,
        ..Default::default()
    };
    assert_step_eq_run_nd(
        || ContinuousFdp::<f64>::force_atlas2(&edgeless, cfg_ov).unwrap(),
        &init,
        3,
    );

    let mut l_ov = init.copy();
    let mut s_ov = ForceAtlas2Solver::<f64>::new(&edgeless, cfg_ov).unwrap();
    s_ov.run(&mut l_ov).unwrap();
    let c = l_ov.coords();
    // Coincident pair (0, 1) separated along deterministic axis 0.
    let d01 = ((c[0] - c[3]).powi(2) + (c[1] - c[4]).powi(2) + (c[2] - c[5]).powi(2)).sqrt();
    assert!(d01 > 1.0, "expected overlap separation in 3D, got {d01}");

    // 4D overlap repulsion without Barnes-Hut.
    coords.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
    let init4 = LayoutNd::<f64>::init(4, 4, 19, 0.5).unwrap();
    let cfg_ov4 = ForceAtlas2Config {
        iterations: 15,
        use_barnes_hut: false,
        prevent_overlap: true,
        default_node_radius: 1.0,
        ..Default::default()
    };
    assert_step_eq_run_nd(
        || ForceAtlas2Solver::<f64>::new(&edgeless, cfg_ov4).unwrap(),
        &init4,
        4,
    );

    // 2D LayoutNd matches Layout2d for ForceAtlas2.
    let g2 = make_grid_3d_csr(3, 3, 1);
    let mut rng = DeterministicRng::from_seed_u64(55);
    let mut xs = Vec::with_capacity(9);
    let mut ys = Vec::with_capacity(9);
    let mut flat = Vec::with_capacity(18);
    for _ in 0..9 {
        let x = 3.0 * (rng.next_f64() - 0.5);
        let y = 3.0 * (rng.next_f64() - 0.5);
        xs.push(x);
        ys.push(y);
        flat.push(x);
        flat.push(y);
    }
    let mut l2 = Layout2d::from_xy(xs, ys).unwrap();
    let mut lnd2 = LayoutNd::from_flat(9, 2, flat).unwrap();
    let cfg2 = ForceAtlas2Config {
        iterations: 12,
        use_barnes_hut: true,
        ..Default::default()
    };
    ForceAtlas2Solver::<f64>::new(&g2, cfg2)
        .unwrap()
        .run(&mut l2)
        .unwrap();
    ForceAtlas2Solver::<f64>::new(&g2, cfg2)
        .unwrap()
        .run(&mut lnd2)
        .unwrap();
    for i in 0..9 {
        assert!((l2.x()[i] - lnd2.coords()[i * 2]).abs() < 1e-12);
        assert!((l2.y()[i] - lnd2.coords()[i * 2 + 1]).abs() < 1e-12);
    }
}

#[test]
fn test_yifan_hu_nd_multilevel_and_continuous_fdp() {
    // N=64 3D grid exercises multi-level EC+MIVS coarsening and 3D prolongation jitter.
    let graph = make_grid_3d_csr(4, 4, 4);
    let cfg = YifanHuConfig {
        max_iterations_per_level: 15,
        min_coarsen_size: 10,
        min_coarsen_ratio: 0.75,
        ..Default::default()
    };

    let init3_64 = LayoutNd::<f64>::init(64, 3, 404, 4.0).unwrap();
    assert_step_eq_run_nd(
        || YifanHuSolver::<f64>::new(&graph, cfg).unwrap(),
        &init3_64,
        7,
    );
    assert_step_eq_run_nd(
        || ContinuousFdp::<f64>::yifan_hu(&graph, cfg).unwrap(),
        &init3_64,
        7,
    );

    let init3_32 = LayoutNd::<f32>::init(64, 3, 404, 4.0).unwrap();
    assert_step_eq_run_nd(
        || YifanHuSolver::<f32>::new(&graph, cfg).unwrap(),
        &init3_32,
        7,
    );

    // 4D and 1D exact Coulomb fallback.
    let disc = make_disconnected_weighted_csr();
    let cfg_small = YifanHuConfig {
        max_iterations_per_level: 12,
        min_coarsen_size: 4,
        ..Default::default()
    };
    let init4_64 = LayoutNd::<f64>::init(10, 4, 505, 3.0).unwrap();
    assert_step_eq_run_nd(
        || YifanHuSolver::<f64>::new(&disc, cfg_small).unwrap(),
        &init4_64,
        5,
    );
    let init4_32 = LayoutNd::<f32>::init(10, 4, 505, 3.0).unwrap();
    assert_step_eq_run_nd(
        || ContinuousFdp::<f32>::yifan_hu(&disc, cfg_small).unwrap(),
        &init4_32,
        5,
    );
    let init1_64 = LayoutNd::<f64>::init(10, 1, 606, 3.0).unwrap();
    assert_step_eq_run_nd(
        || YifanHuSolver::<f64>::new(&disc, cfg_small).unwrap(),
        &init1_64,
        4,
    );

    // Inherent status methods on ContinuousFdp & YifanHuSolver.
    let cfdp = ContinuousFdp::<f64>::yifan_hu(&graph, cfg).unwrap();
    assert_eq!(cfdp.status(), SolverStatus::Initialized);
    assert_eq!(cfdp.iteration(), 0);
    assert!(cfdp.total_iterations() >= 15);
}

#[test]
fn test_nd_degenerates_and_error_guards() {
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let g_edgeless = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
    let g_disc = make_disconnected_weighted_csr();
    let mut l0 = LayoutNd::<f64>::zeros(0, 3).unwrap();
    let mut l1 = LayoutNd::<f64>::zeros(1, 3).unwrap();
    let l_coincident = LayoutNd::<f64>::zeros(5, 3).unwrap();
    let l_collinear = LayoutNd::from_flat(
        5,
        3,
        vec![
            0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0,
        ],
    )
    .unwrap();
    let l_planar = LayoutNd::from_flat(
        5,
        3,
        vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.0,
        ],
    )
    .unwrap();

    assert!(matches!(
        LayoutNd::<f64>::zeros(3, 0),
        Err(CoreError::InvalidInput(_))
    ));

    for g in [&g0, &g1] {
        let mut l = if g.num_nodes() == 0 {
            l0.copy()
        } else {
            l1.copy()
        };
        assert!(
            ForceAtlas2Solver::<f64>::new(g, Default::default())
                .unwrap()
                .run(&mut l)
                .unwrap()
                .is_finished()
        );
        assert!(
            YifanHuSolver::<f64>::new(g, Default::default())
                .unwrap()
                .run(&mut l)
                .unwrap()
                .is_finished()
        );
        assert!(
            ContinuousFdp::<f64>::force_atlas2(g, Default::default())
                .unwrap()
                .run(&mut l)
                .unwrap()
                .is_finished()
        );
    }
    let _ = (&mut l0, &mut l1);

    for init in [&l_coincident, &l_collinear, &l_planar] {
        let mut l = init.copy();
        ForceAtlas2Solver::<f64>::new(
            &g_edgeless,
            ForceAtlas2Config {
                iterations: 5,
                ..Default::default()
            },
        )
        .unwrap()
        .run(&mut l)
        .unwrap();
        assert!(l.is_all_finite());

        let mut l = init.copy();
        YifanHuSolver::<f64>::new(
            &g_edgeless,
            YifanHuConfig {
                max_iterations_per_level: 5,
                ..Default::default()
            },
        )
        .unwrap()
        .run(&mut l)
        .unwrap();
        assert!(l.is_all_finite());
    }

    // Disconnected weighted graph converges cleanly in 3D.
    let mut l_d = LayoutNd::<f64>::init(10, 3, 99, 2.0).unwrap();
    YifanHuSolver::<f64>::new(
        &g_disc,
        YifanHuConfig {
            max_iterations_per_level: 10,
            ..Default::default()
        },
    )
    .unwrap()
    .run(&mut l_d)
    .unwrap();
    assert!(l_d.is_all_finite());

    // NaN, Inf, DimensionMismatch, mid-run dim change, and Barnes-Hut dims=4 error guards.
    let mut fa2 = ForceAtlas2Solver::<f64>::new(&g_disc, Default::default()).unwrap();
    let mut yh = YifanHuSolver::<f64>::new(&g_disc, Default::default()).unwrap();

    let mut l_bad_len = LayoutNd::<f64>::zeros(3, 3).unwrap();
    assert!(matches!(
        fa2.step(&mut l_bad_len),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        yh.step(&mut l_bad_len),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l_nan = LayoutNd::from_flat(10, 3, {
        let mut v = vec![0.0; 30];
        v[4] = f64::NAN;
        v
    })
    .unwrap();
    let mut l_inf = LayoutNd::from_flat(10, 3, {
        let mut v = vec![0.0; 30];
        v[7] = f64::INFINITY;
        v
    })
    .unwrap();
    assert!(matches!(
        fa2.step(&mut l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        yh.step(&mut l_inf),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Barnes-Hut dims=4 rejected with UnsupportedOperation.
    let mut l4 = LayoutNd::<f64>::init(10, 4, 1, 1.0).unwrap();
    assert!(matches!(
        fa2.step(&mut l4),
        Err(CoreError::UnsupportedOperation(_))
    ));

    // Mid-run dim change rejected across steps.
    let mut l3 = LayoutNd::<f64>::init(10, 3, 2, 1.0).unwrap();
    let mut fa2_exact = ForceAtlas2Solver::<f64>::new(
        &g_disc,
        ForceAtlas2Config {
            use_barnes_hut: false,
            iterations: 5,
            ..Default::default()
        },
    )
    .unwrap();
    fa2_exact.step(&mut l3).unwrap();
    assert!(matches!(
        fa2_exact.step(&mut l4),
        Err(CoreError::DimensionMismatch { .. })
    ));

    yh.step(&mut l3).unwrap();
    assert!(matches!(
        yh.step(&mut l4),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
