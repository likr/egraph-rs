use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, Layout2d, Layout3d, LayoutNd,
    LayoutStorage, MetricSpace, RiemannianLayout, Scalar, SolverStatus,
};
use egraph_neighborhood::SparseProbabilities;

use crate::backends::{Auto3d, BarnesHut, Exact, TsNetBackend3d};

use super::super::{DefaultTsNetBackend, TsNetConfig, TsNetRepulsion, TsNetSolver, TsNetStage};
use super::{make_barbell_graph, make_grid_graph, make_stages};

fn seeded_layout_nd<S: Scalar>(n: usize, dims: usize, scale: f64, seed: u64) -> LayoutNd<S> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut coords = Vec::with_capacity(n * dims);
    for _ in 0..(n * dims) {
        coords.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
    }
    LayoutNd::from_flat(n, dims, &coords).unwrap()
}

fn rms_radius_3d<S: Scalar>(layout: &Layout3d<S>) -> f64 {
    let n = layout.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    for i in 0..n {
        let x = layout.x()[i].to_f64();
        let y = layout.y()[i].to_f64();
        let z = layout.z()[i].to_f64();
        sum_sq += x * x + y * y + z * z;
    }
    (sum_sq / (n as f64)).sqrt()
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
fn test_tsnet_3d_and_nd_step_vs_run_bit_exact_f64_and_f32() {
    let grid = make_grid_graph(4, 4);
    let n = grid.num_nodes();

    for backend in [
        TsNetBackend3d::Exact(Exact::new()),
        TsNetBackend3d::BarnesHut(BarnesHut::default()),
    ] {
        let cfg = TsNetConfig::pure_2hop().with_stages(make_stages(8, 8, 8, 0.6));

        let mut l_run = Layout3d::<f64>::init(n, 7, 1.0).unwrap();
        let mut l_step = l_run.clone();
        let mut s_run =
            TsNetSolver::<Layout3d<f64>, TsNetBackend3d>::from_graph_pure_2hop_with_backend(
                &grid,
                cfg.clone(),
                backend.clone(),
            )
            .unwrap();
        let mut s_step =
            TsNetSolver::<Layout3d<f64>, TsNetBackend3d>::from_graph_pure_2hop_with_backend(
                &grid,
                cfg.clone(),
                backend.clone(),
            )
            .unwrap();

        assert_eq!(s_run.status(), SolverStatus::Initialized);
        assert_eq!(s_run.total_iterations(), Some(24));
        assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);

        assert_eq!(
            s_step.run_steps(&mut l_step, 5).unwrap(),
            SolverStatus::Running
        );
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(s_step.status(), SolverStatus::Completed);
        for i in 0..n {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
            assert_eq!(l_run.z()[i].to_bits(), l_step.z()[i].to_bits());
        }

        let mut l32_run = Layout3d::<f32>::init(n, 7, 1.0).unwrap();
        let mut l32_step = l32_run.clone();
        let mut s32_run =
            TsNetSolver::<Layout3d<f32>, TsNetBackend3d>::from_graph_pure_2hop_with_backend(
                &grid,
                cfg.clone(),
                backend.clone(),
            )
            .unwrap();
        let mut s32_step =
            TsNetSolver::<Layout3d<f32>, TsNetBackend3d>::from_graph_pure_2hop_with_backend(
                &grid, cfg, backend,
            )
            .unwrap();
        assert_eq!(s32_run.run(&mut l32_run).unwrap(), SolverStatus::Completed);
        for _ in 0..24 {
            s32_step.step(&mut l32_step).unwrap();
        }
        for i in 0..n {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
            assert_eq!(l32_run.z()[i].to_bits(), l32_step.z()[i].to_bits());
        }
    }

    for dims in [3usize, 5usize] {
        let cfg = TsNetConfig::pure_2hop().with_stages(make_stages(8, 8, 8, 0.6));

        let mut l_run = seeded_layout_nd::<f64>(n, dims, 0.5, 7);
        let mut l_step = l_run.clone();
        let mut s_run =
            TsNetSolver::<LayoutNd<f64>>::from_graph_pure_2hop(&grid, cfg.clone()).unwrap();
        let mut s_step =
            TsNetSolver::<LayoutNd<f64>>::from_graph_pure_2hop(&grid, cfg.clone()).unwrap();

        assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
        assert_eq!(
            s_step.run_steps(&mut l_step, 5).unwrap(),
            SolverStatus::Running
        );
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        for (a, b) in l_run.coords().iter().zip(l_step.coords().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }

        let mut l32_run = seeded_layout_nd::<f32>(n, dims, 0.5, 7);
        let mut l32_step = l32_run.clone();
        let mut s32_run =
            TsNetSolver::<LayoutNd<f32>>::from_graph_pure_2hop(&grid, cfg.clone()).unwrap();
        let mut s32_step = TsNetSolver::<LayoutNd<f32>>::from_graph_pure_2hop(&grid, cfg).unwrap();
        assert_eq!(s32_run.run(&mut l32_run).unwrap(), SolverStatus::Completed);
        for _ in 0..24 {
            s32_step.step(&mut l32_step).unwrap();
        }
        for (a, b) in l32_run.coords().iter().zip(l32_step.coords().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

#[test]
fn test_tsnet_3d_auto3d_resolves_at_256_and_matches_exact_or_barnes_hut() {
    let g_small = make_grid_graph(16, 16); // N = 256 -> Exact
    let g_large = make_grid_graph(17, 16); // N = 272 -> BarnesHut
    let cfg = TsNetConfig::pure_2hop().with_stages(make_stages(4, 4, 4, 0.4));

    // Layout3d defaults to Auto3d.
    let mut s_small = TsNetSolver::<Layout3d<f64>>::from_graph_row_normalized(
        &g_small,
        0.5,
        Some(16),
        cfg.clone(),
    )
    .unwrap();
    assert!(s_small.backend_state.is_exact());
    let mut s_small_exact = TsNetSolver::<Layout3d<f64>, Exact>::from_graph_row_normalized(
        &g_small,
        0.5,
        Some(16),
        cfg.clone(),
    )
    .unwrap();
    let mut l_s_auto = Layout3d::<f64>::init(256, 19, 1.0).unwrap();
    let mut l_s_ex = l_s_auto.clone();
    s_small.run(&mut l_s_auto).unwrap();
    s_small_exact.run(&mut l_s_ex).unwrap();
    for i in 0..256 {
        assert_eq!(l_s_auto.x()[i].to_bits(), l_s_ex.x()[i].to_bits());
        assert_eq!(l_s_auto.y()[i].to_bits(), l_s_ex.y()[i].to_bits());
        assert_eq!(l_s_auto.z()[i].to_bits(), l_s_ex.z()[i].to_bits());
    }

    let n = g_large.num_nodes();
    let mut s_auto_run = TsNetSolver::<Layout3d<f64>>::from_graph_row_normalized(
        &g_large,
        0.5,
        Some(16),
        cfg.clone(),
    )
    .unwrap();
    assert!(!s_auto_run.backend_state.is_exact());

    let mut l_run = Layout3d::<f64>::init(n, 19, 1.0).unwrap();
    let mut l_bh = l_run.clone();
    assert_eq!(s_auto_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
    assert_eq!(s_auto_run.total_fft_2d_calls(), 0);
    assert!(l_run.is_all_finite());

    let mut s_bh = TsNetSolver::<Layout3d<f64>, BarnesHut>::from_graph_row_normalized(
        &g_large,
        0.5,
        Some(16),
        cfg,
    )
    .unwrap();
    s_bh.run(&mut l_bh).unwrap();
    for i in 0..n {
        assert_eq!(l_run.x()[i].to_bits(), l_bh.x()[i].to_bits());
        assert_eq!(l_run.y()[i].to_bits(), l_bh.y()[i].to_bits());
        assert_eq!(l_run.z()[i].to_bits(), l_bh.z()[i].to_bits());
    }

    // Custom threshold on Auto3d selects BarnesHut even at N=16.
    let g_tiny = make_grid_graph(4, 4);
    let s_tiny_bh = TsNetSolver::<Layout3d<f64>, Auto3d>::new_with_backend(
        &g_tiny,
        TsNetConfig::pure_2hop(),
        Auto3d::new().with_exact_threshold(8).with_theta(0.4),
    )
    .unwrap();
    assert!(!s_tiny_bh.backend_state.is_exact());
    assert_eq!(s_tiny_bh.backend().theta(), 0.4);
}

#[test]
fn test_tsnet_nd_2d_exact_parity_with_layout2d() {
    let grid = make_grid_graph(4, 4);
    let n = grid.num_nodes();
    let cfg = TsNetConfig::pure_2hop().with_stages(make_stages(6, 6, 6, 0.6));

    let mut s2d =
        TsNetSolver::<Layout2d<f64>, Exact>::from_graph_pure_2hop(&grid, cfg.clone()).unwrap();
    let mut snd = TsNetSolver::<LayoutNd<f64>>::from_graph_pure_2hop(&grid, cfg).unwrap();

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

    for i in 0..n {
        assert_eq!(l2d.x()[i].to_bits(), lnd.coords()[2 * i].to_bits());
        assert_eq!(l2d.y()[i].to_bits(), lnd.coords()[2 * i + 1].to_bits());
    }
}

#[test]
fn test_tsnet_3d_barbell_separation_and_entropy_expansion() {
    let barbell = make_barbell_graph(6);
    let n = barbell.num_nodes();

    for backend in [
        TsNetBackend3d::Exact(Exact::new()),
        TsNetBackend3d::BarnesHut(BarnesHut::default()),
    ] {
        let cfg = TsNetConfig::pure_2hop()
            .with_learning_rate(Some(15.0))
            .with_stages(make_stages(80, 80, 100, 0.4));
        let mut layout = Layout3d::<f64>::init(n, 11, 0.4).unwrap();
        let mut solver =
            TsNetSolver::<Layout3d<f64>, TsNetBackend3d>::from_graph_pure_2hop_with_backend(
                &barbell,
                cfg,
                backend.clone(),
            )
            .unwrap();
        assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
        assert!(layout.is_all_finite());

        let mut c0 = [0.0_f64; 3];
        let mut c1 = [0.0_f64; 3];
        for i in 0..6 {
            c0[0] += layout.x()[i] / 6.0;
            c0[1] += layout.y()[i] / 6.0;
            c0[2] += layout.z()[i] / 6.0;
            c1[0] += layout.x()[6 + i] / 6.0;
            c1[1] += layout.y()[6 + i] / 6.0;
            c1[2] += layout.z()[6 + i] / 6.0;
        }
        let centroid_dist =
            ((c0[0] - c1[0]).powi(2) + (c0[1] - c1[1]).powi(2) + (c0[2] - c1[2]).powi(2)).sqrt();
        let mut intra = 0.0_f64;
        for i in 0..6 {
            let d0 = ((layout.x()[i] - c0[0]).powi(2)
                + (layout.y()[i] - c0[1]).powi(2)
                + (layout.z()[i] - c0[2]).powi(2))
            .sqrt();
            let d1 = ((layout.x()[6 + i] - c1[0]).powi(2)
                + (layout.y()[6 + i] - c1[1]).powi(2)
                + (layout.z()[6 + i] - c1[2]).powi(2))
            .sqrt();
            intra += d0 + d1;
        }
        intra /= 12.0;
        assert!(
            centroid_dist > intra * 1.5,
            "{backend:?}: 3D barbell failed to separate (centroid_dist={centroid_dist}, intra={intra})"
        );

        let probs = SparseProbabilities::empty(n);
        let stages = vec![
            TsNetStage::new()
                .with_iterations(20)
                .with_exaggeration(1.0)
                .with_lambda_kl(0.0)
                .with_lambda_c(0.0)
                .with_lambda_r(1.0),
        ];
        let cfg_ent = TsNetConfig::pure_2hop()
            .with_learning_rate(Some(50.0))
            .with_stages(stages);
        let mut s_ent =
            TsNetSolver::<Layout3d<f64>, TsNetBackend3d>::from_probabilities_with_backend(
                probs,
                cfg_ent,
                backend.clone(),
            )
            .unwrap();
        let mut l_ent = Layout3d::<f64>::init(n, 123, 0.5).unwrap();
        let init_rms = rms_radius_3d(&l_ent);
        s_ent.run(&mut l_ent).unwrap();
        let final_rms = rms_radius_3d(&l_ent);
        assert!(
            final_rms > init_rms * 1.5,
            "{backend:?}: 3D pure entropy failed to expand (init={init_rms}, final={final_rms})"
        );
    }

    // Verify 4D LayoutNd entropy expansion as well.
    let probs_nd = SparseProbabilities::empty(n);
    let stages_nd = vec![
        TsNetStage::new()
            .with_iterations(20)
            .with_lambda_kl(0.0)
            .with_lambda_r(1.0),
    ];
    let mut s_nd = TsNetSolver::<LayoutNd<f64>>::from_probabilities(
        probs_nd,
        TsNetConfig::pure_2hop()
            .with_learning_rate(Some(50.0))
            .with_stages(stages_nd),
    )
    .unwrap();
    let mut l_nd = seeded_layout_nd::<f64>(n, 4, 0.25, 123);
    let init_nd = rms_radius_nd(&l_nd);
    s_nd.run(&mut l_nd).unwrap();
    assert!(rms_radius_nd(&l_nd) > init_nd * 1.5);
}

#[test]
fn test_tsnet_nd_dims_lock_and_open_custom_surface_backend() {
    let grid = make_grid_graph(3, 3);

    // Changing LayoutNd dims mid-run is rejected with DimensionMismatch.
    let cfg_exact = TsNetConfig::pure_2hop().with_stages(make_stages(4, 0, 0, 0.0));
    let mut s_dims =
        TsNetSolver::<LayoutNd<f64>>::from_graph_pure_2hop(&grid, cfg_exact.clone()).unwrap();
    let mut l_3d = seeded_layout_nd::<f64>(9, 3, 0.5, 11);
    let mut l_4d = seeded_layout_nd::<f64>(9, 4, 0.5, 11);
    s_dims.step(&mut l_3d).unwrap();
    assert!(matches!(
        s_dims.step(&mut l_4d),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        s_dims.run_steps(&mut l_4d, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Open RiemannianLayout + DefaultTsNetBackend + custom TsNetRepulsion (simulating external
    // `egraph-surface` implementing `GeodesicBarnesHut` without modifying `egraph-layout-neighbor`).
    #[derive(Clone)]
    struct CustomSurfaceMock {
        coords: Vec<[f64; 2]>,
    }
    impl LayoutStorage for CustomSurfaceMock {
        type Scalar = f64;
        fn len(&self) -> usize {
            self.coords.len()
        }
        fn validate_finite(&self) -> CoreResult<()> {
            if self
                .coords
                .iter()
                .all(|p| p[0].is_finite() && p[1].is_finite())
            {
                Ok(())
            } else {
                Err(CoreError::non_finite("CustomSurfaceMock"))
            }
        }
    }
    impl MetricSpace for CustomSurfaceMock {
        fn distance(&self, i: usize, j: usize) -> f64 {
            let dx = self.coords[i][0] - self.coords[j][0];
            let dy = self.coords[i][1] - self.coords[j][1];
            dx.hypot(dy)
        }
    }
    impl RiemannianLayout for CustomSurfaceMock {
        fn tangent_dim(&self) -> usize {
            2
        }
        fn log_map_into(&self, i: usize, j: usize, out: &mut [f64]) -> CoreResult<()> {
            out[0] = self.coords[j][0] - self.coords[i][0];
            out[1] = self.coords[j][1] - self.coords[i][1];
            Ok(())
        }
        fn exp_map(&mut self, i: usize, delta: &[f64]) -> CoreResult<()> {
            self.coords[i][0] += delta[0];
            self.coords[i][1] += delta[1];
            Ok(())
        }
        fn parallel_transport_into(
            &self,
            _i: usize,
            _delta: &[f64],
            _vec: &mut [f64],
        ) -> CoreResult<()> {
            Ok(())
        }
    }
    impl DefaultTsNetBackend for CustomSurfaceMock {
        type Backend = Exact;
    }

    #[derive(Debug, Clone, Default)]
    struct MockGeodesicAccelerator;
    impl TsNetRepulsion<CustomSurfaceMock> for MockGeodesicAccelerator {
        type State = ();
        fn validate_config(&self) -> CoreResult<()> {
            Ok(())
        }
        fn init_state(
            &self,
            _num_nodes: usize,
            _epsilon_r: f64,
            _any_entropy: bool,
            _init_entropy: bool,
        ) -> CoreResult<Self::State> {
            Ok(())
        }
        fn compute_repulsion(
            &self,
            _state: &mut Self::State,
            layout: &CustomSurfaceMock,
            epsilon_r: f64,
            compute_entropy: bool,
            rep_forces: &mut [f64],
            ent_forces: &mut [f64],
        ) -> CoreResult<()> {
            Exact.compute_repulsion(
                &mut Default::default(),
                layout,
                epsilon_r,
                compute_entropy,
                rep_forces,
                ent_forces,
            )
        }
    }

    let mut mock = CustomSurfaceMock {
        coords: (0..9).map(|i| [(i % 3) as f64, (i / 3) as f64]).collect(),
    };
    let mut mock_custom = mock.clone();
    let mut s_mock =
        TsNetSolver::<CustomSurfaceMock>::from_graph_pure_2hop(&grid, cfg_exact.clone()).unwrap();
    let mut s_custom =
        TsNetSolver::<CustomSurfaceMock, MockGeodesicAccelerator>::from_graph_pure_2hop(
            &grid, cfg_exact,
        )
        .unwrap();
    assert_eq!(s_mock.run(&mut mock).unwrap(), SolverStatus::Completed);
    assert_eq!(
        s_custom.run(&mut mock_custom).unwrap(),
        SolverStatus::Completed
    );
    assert!(mock.validate_finite().is_ok());
    assert_eq!(mock.coords, mock_custom.coords);
}
