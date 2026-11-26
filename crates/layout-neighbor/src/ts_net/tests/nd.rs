use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, LayoutNd, Scalar,
    SolverStatus,
};
use egraph_neighborhood::SparseProbabilities;

use super::super::{TsNetBackend, TsNetConfig, TsNetSolver, TsNetStage};
use super::{make_barbell_graph, make_grid_graph};

fn seeded_layout_nd<S: Scalar>(n: usize, dims: usize, scale: f64, seed: u64) -> LayoutNd<S> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut coords = Vec::with_capacity(n * dims);
    for _ in 0..(n * dims) {
        coords.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
    }
    LayoutNd::from_flat(n, dims, coords).unwrap()
}

fn rms_radius_nd<S: Scalar>(layout: &LayoutNd<S>) -> f64 {
    let n = layout.num_nodes();
    if n == 0 {
        return 0.0;
    }
    let sum_sq: f64 = layout
        .coords()
        .iter()
        .map(|v| {
            let vf = v.to_f64();
            vf * vf
        })
        .sum();
    (sum_sq / (n as f64)).sqrt()
}

#[test]
fn test_tsnet_nd_step_vs_run_bit_exact_f64_and_f32() {
    let grid = make_grid_graph(4, 4);
    let n = grid.num_nodes();

    let cases = [
        (TsNetBackend::Exact, 3usize),
        (TsNetBackend::Exact, 5usize),
        (TsNetBackend::BarnesHut, 3usize),
    ];

    for (backend, dims) in cases {
        let cfg = TsNetConfig {
            backend,
            stage_iterations: [8, 8, 8],
            lambda_r_stage3: 0.6,
            ..TsNetConfig::pure_2hop()
        };

        // f64 bit-exact step == run
        let mut l_run = seeded_layout_nd::<f64>(n, dims, 0.5, 7);
        let mut l_step = l_run.clone();
        let mut s_run = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg).unwrap();
        let mut s_step = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg).unwrap();

        assert_eq!(s_run.status(), SolverStatus::Initialized);
        assert_eq!(s_run.total_iterations(), 24);
        assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);

        assert_eq!(
            s_step.run_steps(&mut l_step, 5).unwrap(),
            SolverStatus::Running
        );
        assert_eq!(s_step.iteration(), 5);
        assert_eq!(s_step.stage_index(), 0);
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(s_step.status(), SolverStatus::Completed);
        assert_eq!(s_step.iteration(), 24);
        assert_eq!(s_step.stage_index(), 2);

        for (a, b) in l_run.coords().iter().zip(l_step.coords().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }

        // f32 bit-exact step == run
        let mut l32_run = seeded_layout_nd::<f32>(n, dims, 0.5, 7);
        let mut l32_step = l32_run.clone();
        let mut s32_run = TsNetSolver::<f32>::from_graph_pure_2hop(&grid, cfg).unwrap();
        let mut s32_step = TsNetSolver::<f32>::from_graph_pure_2hop(&grid, cfg).unwrap();
        assert_eq!(s32_run.run(&mut l32_run).unwrap(), SolverStatus::Completed);
        for _ in 0..24 {
            s32_step.step(&mut l32_step).unwrap();
        }
        assert_eq!(s32_step.status(), SolverStatus::Completed);
        for (a, b) in l32_run.coords().iter().zip(l32_step.coords().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

#[test]
fn test_tsnet_nd_auto_large_3d_bit_exact_and_no_fft() {
    // N = 272 > 256: Auto resolves to LinearFft at construction, but 3D LayoutNd routes cleanly
    // to ArenaOctree3d BarnesHut without invoking FitField2d.
    let grid = make_grid_graph(17, 16);
    let n = grid.num_nodes();
    let cfg = TsNetConfig {
        backend: TsNetBackend::Auto,
        stage_iterations: [4, 4, 4],
        lambda_r_stage3: 0.4,
        ..TsNetConfig::pure_2hop()
    };

    let mut s_auto_run =
        TsNetSolver::<f64>::from_graph_row_normalized(&grid, 0.5, Some(16), cfg).unwrap();
    let mut s_auto_step =
        TsNetSolver::<f64>::from_graph_row_normalized(&grid, 0.5, Some(16), cfg).unwrap();
    assert_eq!(s_auto_run.resolved_backend(), TsNetBackend::LinearFft);

    let mut l_run = seeded_layout_nd::<f64>(n, 3, 0.5, 19);
    let mut l_step = l_run.clone();
    assert_eq!(s_auto_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
    s_auto_step.run_steps(&mut l_step, 5).unwrap();
    while !s_auto_step.status().is_finished() {
        s_auto_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_auto_run.total_fft_2d_calls(), 0);
    assert!(l_run.is_all_finite());
    for (a, b) in l_run.coords().iter().zip(l_step.coords().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // Matches explicit BarnesHut bit-for-bit on 3D LayoutNd.
    let cfg_bh = TsNetConfig {
        backend: TsNetBackend::BarnesHut,
        ..cfg
    };
    let mut s_bh =
        TsNetSolver::<f64>::from_graph_row_normalized(&grid, 0.5, Some(16), cfg_bh).unwrap();
    let mut l_bh = seeded_layout_nd::<f64>(n, 3, 0.5, 19);
    s_bh.run(&mut l_bh).unwrap();
    for (a, b) in l_run.coords().iter().zip(l_bh.coords().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn test_tsnet_nd_2d_parity_bit_exact_all_backends() {
    let grid = make_grid_graph(4, 4);
    let n = grid.num_nodes();

    for backend in [
        TsNetBackend::Exact,
        TsNetBackend::BarnesHut,
        TsNetBackend::FftInterpolation,
        TsNetBackend::LinearFft,
        TsNetBackend::Auto,
    ] {
        let cfg = TsNetConfig {
            backend,
            stage_iterations: [6, 6, 6],
            lambda_r_stage3: 0.6,
            ..TsNetConfig::pure_2hop()
        };

        let mut s2d = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg).unwrap();
        let mut snd = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg).unwrap();

        let l_init = seeded_layout_nd::<f64>(n, 2, 0.5, 55);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for i in 0..n {
            xs.push(l_init.coords()[2 * i]);
            ys.push(l_init.coords()[2 * i + 1]);
        }
        let mut l2d = Layout2d::from_xy(xs, ys).unwrap();
        let mut lnd = l_init;

        s2d.run(&mut l2d).unwrap();
        snd.run(&mut lnd).unwrap();

        assert_eq!(s2d.total_fft_2d_calls(), snd.total_fft_2d_calls());
        for i in 0..n {
            assert_eq!(l2d.x()[i].to_bits(), lnd.coords()[2 * i].to_bits());
            assert_eq!(l2d.y()[i].to_bits(), lnd.coords()[2 * i + 1].to_bits());
        }
    }
}

#[test]
fn test_tsnet_nd_barbell_3d_separation_and_entropy_expansion() {
    let barbell = make_barbell_graph(6);
    let n = barbell.num_nodes();

    for backend in [TsNetBackend::Exact, TsNetBackend::BarnesHut] {
        let cfg = TsNetConfig {
            backend,
            learning_rate: Some(15.0),
            stage_iterations: [80, 80, 100],
            lambda_r_stage3: 0.4,
            ..TsNetConfig::pure_2hop()
        };
        let mut layout = seeded_layout_nd::<f64>(n, 3, 0.2, 11);
        let mut solver = TsNetSolver::<f64>::from_graph_pure_2hop(&barbell, cfg).unwrap();
        assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
        assert!(layout.is_all_finite());

        let mut c0 = [0.0_f64; 3];
        let mut c1 = [0.0_f64; 3];
        for i in 0..6 {
            for d in 0..3 {
                c0[d] += layout.coords()[i * 3 + d] / 6.0;
                c1[d] += layout.coords()[(6 + i) * 3 + d] / 6.0;
            }
        }
        let centroid_dist =
            ((c0[0] - c1[0]).powi(2) + (c0[1] - c1[1]).powi(2) + (c0[2] - c1[2]).powi(2)).sqrt();
        let mut intra = 0.0_f64;
        for i in 0..6 {
            let d0 = ((layout.coords()[i * 3] - c0[0]).powi(2)
                + (layout.coords()[i * 3 + 1] - c0[1]).powi(2)
                + (layout.coords()[i * 3 + 2] - c0[2]).powi(2))
            .sqrt();
            let d1 = ((layout.coords()[(6 + i) * 3] - c1[0]).powi(2)
                + (layout.coords()[(6 + i) * 3 + 1] - c1[1]).powi(2)
                + (layout.coords()[(6 + i) * 3 + 2] - c1[2]).powi(2))
            .sqrt();
            intra += d0 + d1;
        }
        intra /= 12.0;
        assert!(
            centroid_dist > intra * 1.5,
            "{backend:?}: 3D barbell failed to separate (centroid_dist={centroid_dist}, intra={intra})"
        );

        // Pure entropy stage must expand outward in 3D without sign inversion.
        let probs = SparseProbabilities::empty(n);
        let stages = vec![TsNetStage {
            iterations: 20,
            exaggeration: 1.0,
            lambda_kl: 0.0,
            lambda_c: 0.0,
            lambda_r: 1.0,
        }];
        let cfg_ent = TsNetConfig {
            backend,
            learning_rate: Some(50.0),
            ..TsNetConfig::pure_2hop()
        };
        let mut s_ent =
            TsNetSolver::<f64>::from_probabilities_with_stages(probs, cfg_ent, stages).unwrap();
        let mut l_ent = seeded_layout_nd::<f64>(n, 3, 0.25, 123);
        let init_rms = rms_radius_nd(&l_ent);
        s_ent.run(&mut l_ent).unwrap();
        let final_rms = rms_radius_nd(&l_ent);
        assert!(
            final_rms > init_rms * 1.5,
            "{backend:?}: 3D pure entropy failed to expand (init={init_rms}, final={final_rms})"
        );
    }
}

#[test]
fn test_tsnet_nd_edge_cases_degenerates_and_errors() {
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let g_edgeless = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
    let grid = make_grid_graph(3, 3);

    for backend in [TsNetBackend::Exact, TsNetBackend::BarnesHut] {
        let cfg = TsNetConfig {
            backend,
            stage_iterations: [5, 5, 5],
            lambda_r_stage3: 0.5,
            ..TsNetConfig::pure_2hop()
        };

        // n = 0
        let mut s0 = TsNetSolver::<f64>::from_graph_pure_2hop(&g0, cfg).unwrap();
        let mut l0 = LayoutNd::<f64>::zeros(0, 3).unwrap();
        assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);

        // n = 1
        let mut s1 = TsNetSolver::<f64>::from_graph_pure_2hop(&g1, cfg).unwrap();
        let mut l1 = LayoutNd::from_flat(1, 3, vec![3.5, -2.0, 1.25]).unwrap();
        assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
        assert_eq!(l1.coords(), &[3.5, -2.0, 1.25]);

        // Edgeless graph (n = 5)
        let mut s_edge = TsNetSolver::<f64>::from_graph_pure_2hop(&g_edgeless, cfg).unwrap();
        let mut l_edge = seeded_layout_nd::<f64>(5, 3, 0.5, 33);
        assert_eq!(s_edge.run(&mut l_edge).unwrap(), SolverStatus::Completed);
        assert!(l_edge.is_all_finite());

        // Coincident 3D coordinates
        let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
        let mut s2 = TsNetSolver::<f64>::from_graph_pure_2hop(&g2, cfg).unwrap();
        let mut l2 = LayoutNd::<f64>::zeros(2, 3).unwrap();
        assert_eq!(s2.run(&mut l2).unwrap(), SolverStatus::Completed);
        assert!(l2.is_all_finite());

        // Collinear-in-3D layout
        let mut s_col = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg).unwrap();
        let mut col_coords = Vec::with_capacity(9 * 3);
        for i in 0..9 {
            col_coords.extend_from_slice(&[i as f64 * 0.25, 0.0, 0.0]);
        }
        let mut l_col = LayoutNd::from_flat(9, 3, col_coords).unwrap();
        assert_eq!(s_col.run(&mut l_col).unwrap(), SolverStatus::Completed);
        assert!(l_col.is_all_finite());

        // Planar-in-3D layout
        let mut s_planar = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg).unwrap();
        let mut planar_coords = Vec::with_capacity(9 * 3);
        for i in 0..9 {
            planar_coords.extend_from_slice(&[(i % 3) as f64 * 0.3, (i / 3) as f64 * 0.3, 0.0]);
        }
        let mut l_planar = LayoutNd::from_flat(9, 3, planar_coords).unwrap();
        assert_eq!(
            s_planar.run(&mut l_planar).unwrap(),
            SolverStatus::Completed
        );
        assert!(l_planar.is_all_finite());
    }

    // Explicit FFT backends in 3D/N-D return CoreError::UnsupportedOperation
    for fft_backend in [TsNetBackend::FftInterpolation, TsNetBackend::LinearFft] {
        let cfg_fft = TsNetConfig {
            backend: fft_backend,
            stage_iterations: [2, 0, 0],
            ..TsNetConfig::pure_2hop()
        };
        let mut s_fft = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg_fft).unwrap();
        let mut l_3d = seeded_layout_nd::<f64>(9, 3, 0.5, 11);
        assert!(matches!(
            s_fft.step(&mut l_3d),
            Err(CoreError::UnsupportedOperation(_))
        ));
    }

    // BarnesHut in dims != 2 && dims != 3 returns CoreError::UnsupportedOperation
    let cfg_bh = TsNetConfig {
        backend: TsNetBackend::BarnesHut,
        stage_iterations: [2, 0, 0],
        ..TsNetConfig::pure_2hop()
    };
    let mut s_bh = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg_bh).unwrap();
    let mut l_4d = seeded_layout_nd::<f64>(9, 4, 0.5, 11);
    assert!(matches!(
        s_bh.step(&mut l_4d),
        Err(CoreError::UnsupportedOperation(_))
    ));

    // Changing dims mid-run is rejected with DimensionMismatch
    let cfg_exact = TsNetConfig {
        backend: TsNetBackend::Exact,
        stage_iterations: [4, 0, 0],
        ..TsNetConfig::pure_2hop()
    };
    let mut s_dims = TsNetSolver::<f64>::from_graph_pure_2hop(&grid, cfg_exact).unwrap();
    let mut l_3d = seeded_layout_nd::<f64>(9, 3, 0.5, 11);
    let mut l_4d = seeded_layout_nd::<f64>(9, 4, 0.5, 11);
    s_dims.step(&mut l_3d).unwrap();
    assert!(matches!(
        s_dims.step(&mut l_4d),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // NaN / Inf / node-count mismatch rejection
    let mut l_nan = seeded_layout_nd::<f64>(9, 3, 0.5, 11);
    l_nan.coords_mut()[4] = f64::NAN;
    assert!(matches!(
        s_dims.step(&mut l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut l_inf = seeded_layout_nd::<f64>(9, 3, 0.5, 11);
    l_inf.coords_mut()[7] = f64::NEG_INFINITY;
    assert!(matches!(
        s_dims.step(&mut l_inf),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut l_wrong_n = LayoutNd::<f64>::zeros(8, 3).unwrap();
    assert!(matches!(
        s_dims.step(&mut l_wrong_n),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
