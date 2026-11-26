use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, LayoutNd, Scalar,
    SolverStatus,
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

fn seeded_layout_nd<S: Scalar>(n: usize, dims: usize, scale: f64, seed: u64) -> LayoutNd<S> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut coords = Vec::with_capacity(n * dims);
    for _ in 0..(n * dims) {
        coords.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
    }
    LayoutNd::from_flat(n, dims, coords).unwrap()
}

#[test]
fn test_gumap_nd_step_vs_run_bit_exact_f64_and_f32() {
    let grid = grid_graph(4, 4);
    let n = grid.num_nodes();
    let cfg = GuMapConfig {
        iterations: 20,
        seed: 99,
        ..Default::default()
    };
    let dog_cfg = DogVaradhanConfig::default();

    for dims in [3usize, 5usize] {
        // f64 bit-exact step == run
        let mut s_run = GuMapSolver::<f64>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
        let mut s_step = GuMapSolver::<f64>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
        let mut l_run = seeded_layout_nd::<f64>(n, dims, 1.0, 17);
        let mut l_step = l_run.clone();

        assert_eq!(s_run.status(), SolverStatus::Initialized);
        assert_eq!(s_run.total_iterations(), 20);
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

        // f32 bit-exact step == run
        let mut s32_run = GuMapSolver::<f32>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
        let mut s32_step = GuMapSolver::<f32>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
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
fn test_gumap_nd_2d_parity_bit_exact() {
    let grid = grid_graph(4, 4);
    let n = grid.num_nodes();
    let cfg = GuMapConfig {
        iterations: 15,
        seed: 77,
        ..Default::default()
    };

    for sym in [true, false] {
        let cfg_sym = GuMapConfig {
            symmetric_negative: sym,
            ..cfg
        };
        let mut s2d = GuMapSolver::<f64>::new(&grid, cfg_sym).unwrap();
        let mut snd = GuMapSolver::<f64>::new(&grid, cfg_sym).unwrap();

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

        // All-zero collapsed initialization parity in 2D
        let mut s2d_z = GuMapSolver::<f64>::new(&grid, cfg_sym).unwrap();
        let mut snd_z = GuMapSolver::<f64>::new(&grid, cfg_sym).unwrap();
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
fn test_gumap_nd_barbell_3d_separation_and_collapsed_init() {
    let barbell = barbell_graph(6);
    let n = barbell.num_nodes();
    let cfg = GuMapConfig {
        iterations: 150,
        negative_sample_rate: 5,
        learning_rate: 1.0,
        seed: 42,
        ..Default::default()
    };
    let mut solver = GuMapSolver::<f64>::new(&barbell, cfg).unwrap();
    let mut layout = LayoutNd::<f64>::zeros(n, 3).unwrap();
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
        "expected 3D barbell cluster separation, got centroid_dist={centroid_dist}, intra={intra}"
    );
}

#[test]
fn test_gumap_nd_edge_cases_and_validation() {
    let cfg = GuMapConfig {
        iterations: 10,
        ..Default::default()
    };

    // Empty graph (n = 0)
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut s_empty = GuMapSolver::<f64>::new(&empty, cfg).unwrap();
    let mut l_empty = LayoutNd::<f64>::zeros(0, 3).unwrap();
    assert_eq!(s_empty.run(&mut l_empty).unwrap(), SolverStatus::Converged);

    // Singleton graph (n = 1)
    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut s_single = GuMapSolver::<f64>::new(&single, cfg).unwrap();
    let mut l_single = LayoutNd::from_flat(1, 3, vec![2.0, -3.0, 4.5]).unwrap();
    assert_eq!(
        s_single.run(&mut l_single).unwrap(),
        SolverStatus::Converged
    );
    assert_eq!(l_single.coords(), &[2.0, -3.0, 4.5]);

    // Edgeless graph (n = 4, no edges)
    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let mut s_edgeless = GuMapSolver::<f64>::new(&edgeless, cfg).unwrap();
    let mut l_edgeless = seeded_layout_nd::<f64>(4, 3, 1.0, 9);
    let before = l_edgeless.clone();
    assert_eq!(
        s_edgeless.run(&mut l_edgeless).unwrap(),
        SolverStatus::Converged
    );
    assert_eq!(l_edgeless.coords(), before.coords());

    // Collinear-in-3D and planar-in-3D layouts remain finite
    let grid = grid_graph(3, 3);
    let mut s_col = GuMapSolver::<f64>::new(&grid, cfg).unwrap();
    let mut col_coords = Vec::with_capacity(9 * 3);
    for i in 0..9 {
        col_coords.extend_from_slice(&[i as f64, 0.0, 0.0]);
    }
    let mut l_col = LayoutNd::from_flat(9, 3, col_coords).unwrap();
    assert_eq!(s_col.run(&mut l_col).unwrap(), SolverStatus::Completed);
    assert!(l_col.is_all_finite());

    let mut s_planar = GuMapSolver::<f64>::new(&grid, cfg).unwrap();
    let mut planar_coords = Vec::with_capacity(9 * 3);
    for i in 0..9 {
        planar_coords.extend_from_slice(&[(i % 3) as f64, (i / 3) as f64, 0.0]);
    }
    let mut l_planar = LayoutNd::from_flat(9, 3, planar_coords).unwrap();
    assert_eq!(
        s_planar.run(&mut l_planar).unwrap(),
        SolverStatus::Completed
    );
    assert!(l_planar.is_all_finite());

    // NaN, Inf, and node-count mismatch rejection
    let probs = SparseProbabilities {
        n_nodes: 2,
        indptr: vec![0, 1, 2],
        indices: vec![1, 0],
        values: vec![1.0, 1.0],
    };
    let mut s_err = GuMapSolver::<f64>::from_fuzzy_set(probs, cfg).unwrap();
    let mut l_wrong = LayoutNd::<f64>::zeros(3, 3).unwrap();
    assert!(matches!(
        s_err.step(&mut l_wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let mut l_nan = LayoutNd::from_flat(2, 3, vec![0.0, f64::NAN, 0.0, 1.0, 0.0, 0.0]).unwrap();
    assert!(matches!(
        s_err.step(&mut l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut l_inf =
        LayoutNd::from_flat(2, 3, vec![0.0, 0.0, f64::INFINITY, 1.0, 0.0, 0.0]).unwrap();
    assert!(matches!(
        s_err.step(&mut l_inf),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
