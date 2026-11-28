use egraph_core::{IterativeSolver, Layout2d, SolverStatus};
use egraph_neighborhood::SparseProbabilities;
use egraph_spatial_fft::{IbFftConfig, InterpolationOrder};

use super::super::{TsNetBackend, TsNetConfig, TsNetSolver, TsNetStage};
use super::{
    make_barbell_graph, make_grid_graph, make_stages, mean_edge_and_all_distances, rms_radius,
    seeded_layout,
};

#[test]
fn test_step_vs_run_bit_exact_f64_and_f32() {
    let grid = make_grid_graph(4, 4);
    let n = grid.num_nodes();

    for backend in [
        TsNetBackend::Exact,
        TsNetBackend::BarnesHut,
        TsNetBackend::FftInterpolation,
        TsNetBackend::LinearFft,
    ] {
        let cfg = TsNetConfig::pure_2hop()
            .with_backend(backend)
            .with_stages(make_stages(8, 8, 8, 0.6));

        let mut l_run = seeded_layout::<f64>(n, 0.5, 7);
        let mut l_step = l_run.clone();
        let mut s_run =
            TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&grid, cfg.clone()).unwrap();
        let mut s_step =
            TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&grid, cfg.clone()).unwrap();

        assert_eq!(s_run.status(), SolverStatus::Initialized);
        assert_eq!(s_run.total_iterations(), Some(24));
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
        assert_eq!(s_step.total_fft_2d_calls(), s_run.total_fft_2d_calls());

        for i in 0..n {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        let mut l32_run = seeded_layout::<f32>(n, 0.5, 7);
        let mut l32_step = l32_run.clone();
        let mut s32_run =
            TsNetSolver::<Layout2d<f32>>::from_graph_pure_2hop(&grid, cfg.clone()).unwrap();
        let mut s32_step = TsNetSolver::<Layout2d<f32>>::from_graph_pure_2hop(&grid, cfg).unwrap();
        assert_eq!(s32_run.run(&mut l32_run).unwrap(), SolverStatus::Completed);
        for _ in 0..24 {
            s32_step.step(&mut l32_step).unwrap();
        }
        assert_eq!(s32_step.status(), SolverStatus::Completed);
        for i in 0..n {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }
}

#[test]
fn test_all_backends_converge_on_barbell_and_grid_without_nan() {
    let barbell = make_barbell_graph(6);
    let grid_small = make_grid_graph(4, 4);
    let grid_pure = make_grid_graph(6, 6);

    for backend in [
        TsNetBackend::Exact,
        TsNetBackend::BarnesHut,
        TsNetBackend::FftInterpolation,
        TsNetBackend::LinearFft,
    ] {
        let cfg_small = TsNetConfig::pure_2hop()
            .with_backend(backend)
            .with_learning_rate(Some(15.0))
            .with_stages(make_stages(80, 80, 100, 0.4));
        let mut l_barbell = seeded_layout::<f64>(barbell.num_nodes(), 0.2, 11);
        let mut solver_b =
            TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&barbell, cfg_small.clone())
                .unwrap();
        assert_eq!(
            solver_b.run(&mut l_barbell).unwrap(),
            SolverStatus::Completed
        );
        assert!(l_barbell.is_all_finite());

        let c0_x = l_barbell.x()[..6].iter().sum::<f64>() / 6.0;
        let c0_y = l_barbell.y()[..6].iter().sum::<f64>() / 6.0;
        let c1_x = l_barbell.x()[6..].iter().sum::<f64>() / 6.0;
        let c1_y = l_barbell.y()[6..].iter().sum::<f64>() / 6.0;
        let centroid_dist = (c0_x - c1_x).hypot(c0_y - c1_y);
        let mut intra_spread = 0.0_f64;
        for i in 0..6 {
            intra_spread += (l_barbell.x()[i] - c0_x).hypot(l_barbell.y()[i] - c0_y);
            intra_spread += (l_barbell.x()[6 + i] - c1_x).hypot(l_barbell.y()[6 + i] - c1_y);
        }
        intra_spread /= 12.0;
        assert!(
            centroid_dist > intra_spread * 1.5,
            "{backend:?}: barbell failed to separate cliques (centroid_dist={centroid_dist}, intra_spread={intra_spread})"
        );

        let mut l_grid_small = seeded_layout::<f64>(grid_small.num_nodes(), 0.2, 19);
        let mut solver_gs =
            TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&grid_small, cfg_small).unwrap();
        assert_eq!(
            solver_gs.run(&mut l_grid_small).unwrap(),
            SolverStatus::Completed
        );
        assert!(l_grid_small.is_all_finite());
        let (edge_s, all_s) = mean_edge_and_all_distances(&grid_small, &l_grid_small);
        assert!(
            edge_s < 0.65 * all_s,
            "{backend:?}: 4x4 grid locality failed (mean_edge={edge_s}, mean_all={all_s})"
        );

        let grid_5x5 = make_grid_graph(5, 5);
        let cfg_classic = TsNetConfig::pure_2hop()
            .with_backend(backend)
            .with_learning_rate(Some(50.0))
            .with_stages(vec![
                TsNetStage::new()
                    .with_iterations(40)
                    .with_exaggeration(4.0)
                    .with_lambda_c(0.1),
                TsNetStage::new()
                    .with_iterations(60)
                    .with_exaggeration(1.0)
                    .with_lambda_c(1.2),
                TsNetStage::new()
                    .with_iterations(40)
                    .with_exaggeration(1.0)
                    .with_lambda_c(0.01)
                    .with_lambda_r(0.6),
            ]);
        let mut l_classic = seeded_layout::<f64>(grid_5x5.num_nodes(), 1.0, 19);
        let mut solver_c =
            TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&grid_5x5, cfg_classic).unwrap();
        assert_eq!(
            solver_c.run(&mut l_classic).unwrap(),
            SolverStatus::Completed
        );
        assert!(l_classic.is_all_finite());
        let (edge_c, all_c) = mean_edge_and_all_distances(&grid_5x5, &l_classic);
        assert!(
            edge_c < 0.6 * all_c,
            "{backend:?}: classic 5x5 grid locality failed (mean_edge={edge_c}, mean_all={all_c})"
        );

        let cfg_pure = TsNetConfig::pure_2hop().with_backend(backend);
        let mut l_grid = seeded_layout::<f64>(grid_pure.num_nodes(), 1.0, 19);
        let mut solver_g =
            TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&grid_pure, cfg_pure).unwrap();
        assert_eq!(
            solver_g.run_steps(&mut l_grid, 30).unwrap(),
            SolverStatus::Running
        );
        assert!(l_grid.is_all_finite());

        let mean_x = l_grid.x().iter().sum::<f64>() / (grid_pure.num_nodes() as f64);
        let mean_y = l_grid.y().iter().sum::<f64>() / (grid_pure.num_nodes() as f64);
        assert!(mean_x.abs() < 1e-10 && mean_y.abs() < 1e-10);
    }
}

#[test]
fn test_stage3_lambda_r_positive_expands_outward_regression() {
    let grid = make_grid_graph(4, 4);
    let compact_init = seeded_layout::<f64>(grid.num_nodes(), 0.25, 123);
    let init_rms = rms_radius(&compact_init);

    for backend in [
        TsNetBackend::Exact,
        TsNetBackend::BarnesHut,
        TsNetBackend::FftInterpolation,
        TsNetBackend::LinearFft,
    ] {
        let stages = vec![
            TsNetStage::new()
                .with_iterations(20)
                .with_exaggeration(1.0)
                .with_lambda_kl(0.0)
                .with_lambda_c(0.0)
                .with_lambda_r(1.0),
        ];
        let cfg = TsNetConfig::pure_2hop()
            .with_backend(backend)
            .with_learning_rate(Some(50.0))
            .with_stages(stages);
        let probs = SparseProbabilities::empty(grid.num_nodes());
        let mut solver = TsNetSolver::<Layout2d<f64>>::from_probabilities(probs, cfg).unwrap();
        let mut layout = compact_init.clone();
        solver.run(&mut layout).unwrap();
        let final_rms = rms_radius(&layout);
        assert!(
            final_rms > init_rms * 1.5,
            "{backend:?}: pure entropy failed to expand (init={init_rms}, final={final_rms})"
        );
    }

    for backend in [
        TsNetBackend::Exact,
        TsNetBackend::BarnesHut,
        TsNetBackend::FftInterpolation,
        TsNetBackend::LinearFft,
    ] {
        let cfg_pre = TsNetConfig::pure_2hop()
            .with_backend(backend)
            .with_learning_rate(Some(15.0))
            .with_stages(make_stages(40, 40, 0, 0.0));
        let mut post_stage2 = seeded_layout::<f64>(grid.num_nodes(), 0.5, 99);
        TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&grid, cfg_pre)
            .unwrap()
            .run(&mut post_stage2)
            .unwrap();

        let make_s3 = |lr: f64| {
            vec![
                TsNetStage::new()
                    .with_iterations(40)
                    .with_exaggeration(1.0)
                    .with_lambda_kl(1.0)
                    .with_lambda_c(0.01)
                    .with_lambda_r(lr),
            ]
        };
        let cfg_no_rep = TsNetConfig::pure_2hop()
            .with_backend(backend)
            .with_learning_rate(Some(15.0))
            .with_stages(make_s3(0.0));
        let cfg_with_rep = cfg_no_rep.clone().with_stages(make_s3(0.6));
        let mut l_no_rep = post_stage2.clone();
        let mut l_with_rep = post_stage2;

        TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&grid, cfg_no_rep)
            .unwrap()
            .run(&mut l_no_rep)
            .unwrap();
        TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&grid, cfg_with_rep)
            .unwrap()
            .run(&mut l_with_rep)
            .unwrap();

        let rms_no_rep = rms_radius(&l_no_rep);
        let rms_with_rep = rms_radius(&l_with_rep);
        assert!(
            rms_with_rep > rms_no_rep,
            "{backend:?}: Stage 3 lambda_r=0.6 did not expand relative to lambda_r=0.0 ({rms_with_rep} <= {rms_no_rep})"
        );
    }
}

#[test]
fn test_fit_field_fft_call_counts_and_linear_fft_forces_bilinear() {
    let grid = make_grid_graph(4, 5);
    let init = seeded_layout::<f64>(grid.num_nodes(), 1.0, 55);

    for (backend, expected_order) in [
        (TsNetBackend::FftInterpolation, InterpolationOrder::Cubic),
        (TsNetBackend::LinearFft, InterpolationOrder::Bilinear),
    ] {
        let cfg = TsNetConfig::pure_2hop()
            .with_backend(backend)
            .with_stages(make_stages(2, 1, 2, 0.6))
            .with_ibfft(IbFftConfig::default().with_order(InterpolationOrder::Cubic));
        let mut solver = TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&grid, cfg).unwrap();
        assert_eq!(
            solver
                .state
                .fit_field
                .as_ref()
                .unwrap()
                .config()
                .ibfft
                .order,
            expected_order
        );
        assert!(solver.last_fft_stats().is_none());
        assert_eq!(solver.total_fft_2d_calls(), 0);

        let mut layout = init.clone();
        solver.step(&mut layout).unwrap();
        let s1 = solver.last_fft_stats().unwrap();
        assert_eq!(s1.particle_forward_ffts, 2);
        assert!(s1.total_fft_2d_calls <= 5);
        assert_eq!(s1.total_fft_2d_calls, 5);
        assert_eq!(solver.total_fft_2d_calls(), 5);

        solver.run_steps(&mut layout, 2).unwrap();
        assert_eq!(solver.stage_index(), 2);
        assert_eq!(solver.total_fft_2d_calls(), 15);

        solver.step(&mut layout).unwrap();
        let s3 = solver.last_fft_stats().unwrap();
        assert_eq!(s3.particle_forward_ffts, 2);
        assert!(s3.total_fft_2d_calls <= 8);
        assert_eq!(s3.total_fft_2d_calls, 8);
        assert_eq!(solver.total_fft_2d_calls(), 23);
        assert_eq!(
            solver
                .state
                .fit_field
                .as_ref()
                .unwrap()
                .config()
                .ibfft
                .order,
            expected_order
        );
    }

    for backend in [TsNetBackend::Exact, TsNetBackend::BarnesHut] {
        let cfg = TsNetConfig::pure_2hop()
            .with_backend(backend)
            .with_stages(make_stages(2, 0, 0, 0.0));
        let mut solver = TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&grid, cfg).unwrap();
        let mut layout = init.clone();
        solver.run(&mut layout).unwrap();
        assert!(solver.last_fft_stats().is_none());
        assert_eq!(solver.total_fft_2d_calls(), 0);
    }
}
