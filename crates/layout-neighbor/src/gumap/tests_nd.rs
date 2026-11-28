use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Layout3d,
    LayoutNd, LayoutStorage, MetricSpace, RiemannianLayout, Scalar, SolverStatus,
};
use egraph_neighborhood::{DogVaradhanConfig, SparseProbabilities};

use super::{GuMapConfig, GuMapSolver};

fn barbell_graph(clique_size: usize) -> CsrGraph<f64> {
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

fn grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
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

fn seeded_layout_3d<S: Scalar>(n: usize, scale: f64, seed: u64) -> Layout3d<S> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for _ in 0..n {
        x.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
        y.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
        z.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
    }
    Layout3d::from_xyz(x, y, z).unwrap()
}

fn seeded_layout_nd<S: Scalar>(n: usize, dims: usize, scale: f64, seed: u64) -> LayoutNd<S> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut coords = Vec::with_capacity(n * dims);
    for _ in 0..(n * dims) {
        coords.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
    }
    LayoutNd::from_flat(n, dims, &coords).unwrap()
}

#[test]
fn test_gumap_3d_and_nd_step_vs_run_bit_exact_f64_and_f32() {
    let grid = grid_graph(4, 4);
    let n = grid.num_nodes();
    let cfg = GuMapConfig::new().with_iterations(20).with_seed(99);
    let dog_cfg = DogVaradhanConfig::default();

    // Layout3d f64 & f32 bit-exact step == run
    {
        let mut s_run =
            GuMapSolver::<Layout3d<f64>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone()).unwrap();
        let mut s_step =
            GuMapSolver::<Layout3d<f64>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone()).unwrap();
        let mut l_run = seeded_layout_3d::<f64>(n, 1.0, 17);
        let mut l_step = l_run.clone();

        assert_eq!(s_run.total_iterations(), Some(20));
        assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
        assert_eq!(
            s_step.run_steps(&mut l_step, 7).unwrap(),
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

        let mut s32_run =
            GuMapSolver::<Layout3d<f32>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone()).unwrap();
        let mut s32_step =
            GuMapSolver::<Layout3d<f32>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone()).unwrap();
        let mut l32_run = seeded_layout_3d::<f32>(n, 1.0, 17);
        let mut l32_step = l32_run.clone();
        assert_eq!(s32_run.run(&mut l32_run).unwrap(), SolverStatus::Completed);
        for _ in 0..20 {
            s32_step.step(&mut l32_step).unwrap();
        }
        for i in 0..n {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
            assert_eq!(l32_run.z()[i].to_bits(), l32_step.z()[i].to_bits());
        }
    }

    // LayoutNd (3D, 5D) f64 & f32 bit-exact step == run
    for dims in [3usize, 5usize] {
        let mut s_run =
            GuMapSolver::<LayoutNd<f64>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone()).unwrap();
        let mut s_step =
            GuMapSolver::<LayoutNd<f64>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone()).unwrap();
        let mut l_run = seeded_layout_nd::<f64>(n, dims, 1.0, 17);
        let mut l_step = l_run.clone();

        assert_eq!(s_run.status(), SolverStatus::Initialized);
        assert_eq!(s_run.total_iterations(), Some(20));
        assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);

        assert_eq!(
            s_step.run_steps(&mut l_step, 7).unwrap(),
            SolverStatus::Running
        );
        assert_eq!(s_step.iteration(), 7);
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(s_step.status(), SolverStatus::Completed);
        assert_eq!(s_step.iteration(), 20);

        for (a, b) in l_run.coords().iter().zip(l_step.coords().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }

        let mut s32_run =
            GuMapSolver::<LayoutNd<f32>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone()).unwrap();
        let mut s32_step =
            GuMapSolver::<LayoutNd<f32>>::from_graph_dog(&grid, 32, &dog_cfg, cfg.clone()).unwrap();
        let mut l32_run = seeded_layout_nd::<f32>(n, dims, 1.0, 17);
        let mut l32_step = l32_run.clone();

        assert_eq!(s32_run.run(&mut l32_run).unwrap(), SolverStatus::Completed);
        for _ in 0..20 {
            s32_step.step(&mut l32_step).unwrap();
        }
        assert_eq!(s32_step.status(), SolverStatus::Completed);

        for (a, b) in l32_run.coords().iter().zip(l32_step.coords().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

#[test]
fn test_gumap_nd_2d_and_3d_parity_bit_exact() {
    let grid = grid_graph(4, 4);
    let n = grid.num_nodes();
    let cfg = GuMapConfig::new().with_iterations(15).with_seed(77);

    for sym in [true, false] {
        let cfg_sym = cfg.clone().with_symmetric_negative(sym);
        let mut s2d = GuMapSolver::<Layout2d<f64>>::new(&grid, cfg_sym.clone()).unwrap();
        let mut snd = GuMapSolver::<LayoutNd<f64>>::new(&grid, cfg_sym.clone()).unwrap();

        let l_init = seeded_layout_nd::<f64>(n, 2, 1.0, 41);
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

        // 3D Layout3d <-> LayoutNd(3) parity
        let mut s3d = GuMapSolver::<Layout3d<f64>>::new(&grid, cfg_sym.clone()).unwrap();
        let mut snd3 = GuMapSolver::<LayoutNd<f64>>::new(&grid, cfg_sym.clone()).unwrap();
        let l3_init = seeded_layout_3d::<f64>(n, 1.0, 53);
        let mut flat3 = Vec::with_capacity(3 * n);
        for i in 0..n {
            flat3.push(l3_init.x()[i]);
            flat3.push(l3_init.y()[i]);
            flat3.push(l3_init.z()[i]);
        }
        let mut l3d = l3_init;
        let mut lnd3 = LayoutNd::from_flat(n, 3, &flat3).unwrap();
        s3d.run(&mut l3d).unwrap();
        snd3.run(&mut lnd3).unwrap();
        for i in 0..n {
            assert_eq!(l3d.x()[i].to_bits(), lnd3.coords()[3 * i].to_bits());
            assert_eq!(l3d.y()[i].to_bits(), lnd3.coords()[3 * i + 1].to_bits());
            assert_eq!(l3d.z()[i].to_bits(), lnd3.coords()[3 * i + 2].to_bits());
        }

        // All-zero collapsed initialization parity in 2D
        let mut s2d_z = GuMapSolver::<Layout2d<f64>>::new(&grid, cfg_sym.clone()).unwrap();
        let mut snd_z = GuMapSolver::<LayoutNd<f64>>::new(&grid, cfg_sym).unwrap();
        let mut l2d_z = Layout2d::<f64>::zeros(n);
        let mut lnd_z = LayoutNd::<f64>::zeros(n, 2).unwrap();
        s2d_z.run(&mut l2d_z).unwrap();
        snd_z.run(&mut lnd_z).unwrap();
        for i in 0..n {
            assert_eq!(l2d_z.x()[i].to_bits(), lnd_z.coords()[2 * i].to_bits());
            assert_eq!(l2d_z.y()[i].to_bits(), lnd_z.coords()[2 * i + 1].to_bits());
        }
    }
}

#[test]
fn test_gumap_3d_barbell_separation_and_collapsed_init() {
    let barbell = barbell_graph(6);
    let n = barbell.num_nodes();
    let cfg = GuMapConfig::new()
        .with_iterations(150)
        .with_negative_sample_rate(5)
        .with_learning_rate(1.0)
        .with_seed(42);
    let mut solver = GuMapSolver::<Layout3d<f64>>::new(&barbell, cfg).unwrap();
    let mut layout = Layout3d::<f64>::zeros(n);
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
        "expected 3D barbell cluster separation, got centroid_dist={centroid_dist}, intra={intra}"
    );
}

#[test]
fn test_gumap_nd_edge_cases_validation_and_custom_riemannian() {
    let cfg = GuMapConfig::new().with_iterations(10);

    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s_empty = GuMapSolver::<LayoutNd<f64>>::new(&empty, cfg.clone()).unwrap();
    let mut l_empty = LayoutNd::<f64>::zeros(0, 3).unwrap();
    assert_eq!(s_empty.run(&mut l_empty).unwrap(), SolverStatus::Converged);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s_single = GuMapSolver::<LayoutNd<f64>>::new(&single, cfg.clone()).unwrap();
    let mut l_single = LayoutNd::from_flat(1, 3, &[2.0, -3.0, 4.5]).unwrap();
    assert_eq!(
        s_single.run(&mut l_single).unwrap(),
        SolverStatus::Converged
    );
    assert_eq!(l_single.coords(), &[2.0, -3.0, 4.5]);

    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let mut s_edgeless = GuMapSolver::<LayoutNd<f64>>::new(&edgeless, cfg.clone()).unwrap();
    let mut l_edgeless = seeded_layout_nd::<f64>(4, 3, 1.0, 9);
    let before = l_edgeless.clone();
    assert_eq!(
        s_edgeless.run(&mut l_edgeless).unwrap(),
        SolverStatus::Converged
    );
    assert_eq!(l_edgeless.coords(), before.coords());

    // NaN, Inf, node-count mismatch, and mid-run LayoutNd dimension lock.
    let probs =
        SparseProbabilities::from_csr(2, vec![0, 1, 2], vec![1, 0], vec![1.0, 1.0]).unwrap();
    let mut s_err = GuMapSolver::<LayoutNd<f64>>::from_fuzzy_set(probs, cfg.clone()).unwrap();
    let mut l_wrong = LayoutNd::<f64>::zeros(3, 3).unwrap();
    assert!(matches!(
        s_err.step(&mut l_wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));
    let mut l_nan = LayoutNd::from_flat(2, 3, &[0.0, f64::NAN, 0.0, 1.0, 0.0, 0.0]).unwrap();
    assert!(matches!(
        s_err.step(&mut l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let grid = grid_graph(3, 3);
    let mut l_d3 = seeded_layout_nd::<f64>(9, 3, 1.0, 3);
    let mut l_d4 = seeded_layout_nd::<f64>(9, 4, 1.0, 3);
    let mut s_nd = GuMapSolver::<LayoutNd<f64>>::new(&grid, cfg.clone()).unwrap();
    s_nd.step(&mut l_d3).unwrap();
    assert!(matches!(
        s_nd.step(&mut l_d4),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        s_nd.run_steps(&mut l_d4, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Open RiemannianLayout target works directly with GuMapSolver<L>.
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

    let mut mock = CustomSurfaceMock {
        coords: (0..9).map(|i| [(i % 3) as f64, (i / 3) as f64]).collect(),
    };
    let mut s_mock = GuMapSolver::<CustomSurfaceMock>::new(&grid, cfg).unwrap();
    assert_eq!(s_mock.run(&mut mock).unwrap(), SolverStatus::Completed);
    assert!(mock.validate_finite().is_ok());
}
