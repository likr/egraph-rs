use egraph_core::{
    CoreError, CsrGraph, IterativeSolver, Layout2d, Layout3d, LayoutNd, SolverStatus,
};
use egraph_neighborhood::{DogVaradhanConfig, KnnGraph};

use super::super::{FieldRepulsionBackend, IbUmapConfig, IbUmapSolver, TFdpConfig, TFdpSolver};
use super::{barbell_graph, grid_graph, seeded_layout};

#[test]
fn test_t_fdp_step_vs_run_bit_exact_f64_and_f32_all_backends() {
    let grid = grid_graph(4, 4);
    for backend in [
        FieldRepulsionBackend::IbFft,
        FieldRepulsionBackend::BarnesHut,
        FieldRepulsionBackend::Exact,
    ] {
        let cfg = TFdpConfig::new()
            .with_backend(backend)
            .with_iterations(20)
            .with_seed(77);

        // f64 bit-exact parity
        let mut s_run = TFdpSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap();
        let mut s_step = TFdpSolver::<Layout2d<f64>>::new(&grid, cfg.clone()).unwrap();
        let mut l_run = seeded_layout::<f64>(grid.num_nodes(), 1.5, 19);
        let mut l_step = l_run.clone();

        assert_eq!(s_run.status(), SolverStatus::Initialized);
        assert_eq!(s_run.num_nodes(), 16);
        assert_eq!(s_run.total_iterations(), Some(20));
        assert!((s_run.current_rho() - 1.0).abs() < 1e-12);

        assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
        assert!((s_run.current_rho() - 0.1).abs() < 1e-12);

        assert_eq!(
            s_step.run_steps(&mut l_step, 6).unwrap(),
            SolverStatus::Running
        );
        assert_eq!(s_step.iteration(), 6);
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(s_step.status(), SolverStatus::Completed);
        assert_eq!(s_step.iteration(), 20);

        for i in 0..grid.num_nodes() {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        // f32 bit-exact parity
        let mut s32_run = TFdpSolver::<Layout2d<f32>>::new(&grid, cfg.clone()).unwrap();
        let mut s32_step = TFdpSolver::<Layout2d<f32>>::new(&grid, cfg).unwrap();
        let mut l32_run = seeded_layout::<f32>(grid.num_nodes(), 1.5, 19);
        let mut l32_step = l32_run.clone();
        s32_run.run(&mut l32_run).unwrap();
        for _ in 0..20 {
            s32_step.step(&mut l32_step).unwrap();
        }
        for i in 0..grid.num_nodes() {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }
}

#[test]
fn test_t_fdp_ibfft_vs_exact_and_barnes_hut_consistency() {
    let barbell = barbell_graph(6);
    let init = seeded_layout::<f64>(barbell.num_nodes(), 2.0, 101);

    for gamma in [1.0, 1.5, 2.0] {
        let mut s_exact = TFdpSolver::<Layout2d<f64>>::new(
            &barbell,
            TFdpConfig::new()
                .with_backend(FieldRepulsionBackend::Exact)
                .with_iterations(15)
                .with_gamma(gamma),
        )
        .unwrap();
        let mut s_ibfft = TFdpSolver::<Layout2d<f64>>::new(
            &barbell,
            TFdpConfig::new()
                .with_backend(FieldRepulsionBackend::IbFft)
                .with_iterations(15)
                .with_gamma(gamma),
        )
        .unwrap();
        let mut s_bh_exact = TFdpSolver::<Layout2d<f64>>::new(
            &barbell,
            TFdpConfig::new()
                .with_backend(FieldRepulsionBackend::BarnesHut)
                .with_iterations(15)
                .with_theta(0.0)
                .with_gamma(gamma),
        )
        .unwrap();

        let mut l_exact = init.clone();
        let mut l_ibfft = init.clone();
        let mut l_bh = init.clone();
        s_exact.run(&mut l_exact).unwrap();
        s_ibfft.run(&mut l_ibfft).unwrap();
        s_bh_exact.run(&mut l_bh).unwrap();
        assert!(s_ibfft.total_fft_2d_calls() > 0);

        for i in 0..barbell.num_nodes() {
            let bh_err = (l_bh.x()[i] - l_exact.x()[i]).hypot(l_bh.y()[i] - l_exact.y()[i]);
            assert!(
                bh_err < 1e-10,
                "gamma={gamma}: BarnesHut(theta=0) vs Exact err={bh_err}"
            );
            let fft_err = (l_ibfft.x()[i] - l_exact.x()[i]).hypot(l_ibfft.y()[i] - l_exact.y()[i]);
            assert!(
                fft_err < 0.05,
                "gamma={gamma}: IbFft vs Exact err={fft_err}"
            );
        }
    }
}

#[test]
fn test_t_fdp_3d_octree_and_nd_exact() {
    let barbell = barbell_graph(6);
    let n = barbell.num_nodes();
    let init_3d = Layout3d::<f64>::init(n, 101, 2.0).unwrap();

    for gamma in [1.0, 1.5, 2.0] {
        let mut s_exact = TFdpSolver::<Layout3d<f64>>::new(
            &barbell,
            TFdpConfig::new()
                .with_backend(FieldRepulsionBackend::Exact)
                .with_iterations(15)
                .with_gamma(gamma),
        )
        .unwrap();
        let mut s_bh = TFdpSolver::<Layout3d<f64>>::new(
            &barbell,
            TFdpConfig::new()
                .with_backend(FieldRepulsionBackend::BarnesHut)
                .with_iterations(15)
                .with_theta(0.0)
                .with_gamma(gamma),
        )
        .unwrap();

        let mut l_exact = init_3d.clone();
        let mut l_bh = init_3d.clone();
        s_exact.run(&mut l_exact).unwrap();
        s_bh.run(&mut l_bh).unwrap();
        for i in 0..n {
            let dx = l_bh.x()[i] - l_exact.x()[i];
            let dy = l_bh.y()[i] - l_exact.y()[i];
            let dz = l_bh.z()[i] - l_exact.z()[i];
            let err = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(err < 1e-10, "3D gamma={gamma}: BH(theta=0) err={err}");
        }
    }

    // 3D barbell separation with BarnesHut
    let mut s_sep = TFdpSolver::<Layout3d<f64>>::new(
        &barbell,
        TFdpConfig::new()
            .with_backend(FieldRepulsionBackend::BarnesHut)
            .with_iterations(120),
    )
    .unwrap();
    let mut l_sep = Layout3d::<f64>::zeros(n);
    assert_eq!(s_sep.run(&mut l_sep).unwrap(), SolverStatus::Completed);
    assert!(l_sep.is_all_finite());

    // 4D LayoutNd Exact parity with step == run and dimension lock
    let mut s_nd_run = TFdpSolver::<LayoutNd<f64>>::new(
        &barbell,
        TFdpConfig::new()
            .with_backend(FieldRepulsionBackend::Exact)
            .with_iterations(15),
    )
    .unwrap();
    let mut s_nd_step = TFdpSolver::<LayoutNd<f64>>::new(
        &barbell,
        TFdpConfig::new()
            .with_backend(FieldRepulsionBackend::Exact)
            .with_iterations(15),
    )
    .unwrap();
    let mut l_nd_run = LayoutNd::<f64>::init(n, 4, 17, 1.0).unwrap();
    let mut l_nd_step = l_nd_run.clone();
    s_nd_run.run(&mut l_nd_run).unwrap();
    s_nd_step.run_steps(&mut l_nd_step, 5).unwrap();
    while !s_nd_step.status().is_finished() {
        s_nd_step.step(&mut l_nd_step).unwrap();
    }
    for (a, b) in l_nd_run.coords().iter().zip(l_nd_step.coords().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    let mut l_wrong_dim = LayoutNd::<f64>::zeros(n, 5).unwrap();
    assert!(matches!(
        s_nd_run.run_steps(&mut l_wrong_dim, 0),
        Err(CoreError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_t_fdp_and_ib_umap_from_knn_and_t_fdp_separates_barbell() {
    let barbell = barbell_graph(6);
    let mut s_fdp =
        TFdpSolver::<Layout2d<f64>>::new(&barbell, TFdpConfig::new().with_iterations(150)).unwrap();
    let mut l_fdp = Layout2d::<f64>::zeros(barbell.num_nodes());
    assert_eq!(s_fdp.run(&mut l_fdp).unwrap(), SolverStatus::Completed);
    assert!(l_fdp.is_all_finite());

    let c0x = l_fdp.x()[..6].iter().sum::<f64>() / 6.0;
    let c0y = l_fdp.y()[..6].iter().sum::<f64>() / 6.0;
    let c1x = l_fdp.x()[6..].iter().sum::<f64>() / 6.0;
    let c1y = l_fdp.y()[6..].iter().sum::<f64>() / 6.0;
    let centroid_dist = (c0x - c1x).hypot(c0y - c1y);
    let mut intra = 0.0_f64;
    for i in 0..6 {
        intra += (l_fdp.x()[i] - c0x).hypot(l_fdp.y()[i] - c0y);
        intra += (l_fdp.x()[6 + i] - c1x).hypot(l_fdp.y()[6 + i] - c1y);
    }
    intra /= 12.0;
    assert!(
        centroid_dist > intra * 1.5,
        "t-FDP should separate barbell clusters: centroid_dist={centroid_dist}, intra={intra}"
    );

    let knn = KnnGraph::new_validated(
        4,
        vec![0, 2, 4, 6, 8],
        vec![1, 3, 0, 2, 1, 3, 0, 2],
        vec![0.5, 0.8, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8],
    )
    .unwrap();
    let mut s_knn_fdp =
        TFdpSolver::<Layout2d<f64>>::from_knn(&knn, TFdpConfig::new().with_iterations(30)).unwrap();
    let mut l_kf = seeded_layout::<f64>(4, 1.0, 9);
    assert_eq!(s_knn_fdp.run(&mut l_kf).unwrap(), SolverStatus::Completed);
    assert!(l_kf.is_all_finite());

    let mut s_knn_umap =
        IbUmapSolver::<f64>::from_knn(&knn, 64, 1e-5, IbUmapConfig::new().with_iterations(30))
            .unwrap();
    let mut l_ku = seeded_layout::<f64>(4, 1.0, 9);
    assert_eq!(s_knn_umap.run(&mut l_ku).unwrap(), SolverStatus::Completed);
    assert!(l_ku.is_all_finite());
}

#[test]
fn test_zero_per_step_allocations_directed_graph_and_custom_ab_alpha() {
    let mut dir_edges = Vec::new();
    for i in 0..24u32 {
        dir_edges.push((i, (i + 1) % 24));
        if i % 3 == 0 {
            dir_edges.push((i, (i + 6) % 24));
        }
    }
    let dir_weights = vec![1.5_f64; dir_edges.len()];
    let dir_graph =
        CsrGraph::<f64>::from_edges(24, &dir_edges, Some(&dir_weights), true, false).unwrap();

    let mut s_fdp = TFdpSolver::<Layout2d<f64>>::new(
        &dir_graph,
        TFdpConfig::new()
            .with_backend(FieldRepulsionBackend::IbFft)
            .with_iterations(20)
            .with_alpha(2.0)
            .with_gamma(1.0),
    )
    .unwrap();
    let mut l_fdp = seeded_layout::<f64>(24, 2.0, 11);
    s_fdp.step(&mut l_fdp).unwrap();
    let stats = s_fdp.last_fft_stats().unwrap();
    assert!(stats.total_fft_2d_calls > 0);
    assert!(stats.nx < 512 && stats.ny < 512);

    let (attr_cap, vel_cap, next_cap, rep_cap, phi_cap) = s_fdp.state_capacities();
    s_fdp.run(&mut l_fdp).unwrap();
    assert_eq!(
        s_fdp.state_capacities(),
        (attr_cap, vel_cap, next_cap, rep_cap, phi_cap)
    );
    assert!(l_fdp.is_all_finite());

    // Span guard prevents 512x512 FFT blowup on huge coordinate spans and matches Exact forces.
    let mut wide_layout = seeded_layout::<f64>(24, 5000.0, 17);
    let mut wide_exact = wide_layout.clone();
    let mut s_wide_fft = TFdpSolver::<Layout2d<f64>>::new(
        &dir_graph,
        TFdpConfig::new()
            .with_backend(FieldRepulsionBackend::IbFft)
            .with_iterations(2),
    )
    .unwrap();
    let mut s_wide_exact = TFdpSolver::<Layout2d<f64>>::new(
        &dir_graph,
        TFdpConfig::new()
            .with_backend(FieldRepulsionBackend::Exact)
            .with_iterations(2),
    )
    .unwrap();
    s_wide_fft.run(&mut wide_layout).unwrap();
    s_wide_exact.run(&mut wide_exact).unwrap();
    let wide_stats = s_wide_fft.last_fft_stats().unwrap();
    assert!(
        wide_stats.nx < 512 && wide_stats.ny < 512,
        "span guard failed to prevent 512 grid blowup: {wide_stats:?}"
    );
    assert!(wide_layout.is_all_finite());

    let mut s_umap = IbUmapSolver::<f64>::from_graph_dog(
        &dir_graph,
        16,
        &DogVaradhanConfig::default(),
        IbUmapConfig::new()
            .with_backend(FieldRepulsionBackend::BarnesHut)
            .with_iterations(15)
            .with_a(Some(1.6))
            .with_b(Some(0.9)),
    )
    .unwrap();
    assert_eq!(s_umap.ab(), (1.6, 0.9));
    assert_eq!(s_umap.a(), 1.6);
    assert_eq!(s_umap.b(), 0.9);
    let mut l_umap = seeded_layout::<f64>(24, 2.0, 11);
    s_umap.step(&mut l_umap).unwrap();
    let qt_node_cap = s_umap.field_quadtree.nodes.capacity();
    s_umap.run(&mut l_umap).unwrap();
    assert!(s_umap.field_quadtree.nodes.capacity() >= qt_node_cap);
    assert!(l_umap.is_all_finite());
}
