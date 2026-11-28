use egraph_core::{CoreError, CsrGraph, DeterministicRng, Layout2d, LayoutGenerator, LayoutNd};

use super::normalize::next_gaussian_pair;
use super::*;

pub(super) fn isotropic_std(layout: &Layout2d<f64>) -> f64 {
    let n = layout.len() as f64;
    let mx = layout.x().iter().sum::<f64>() / n;
    let my = layout.y().iter().sum::<f64>() / n;
    let sum_sq: f64 = layout
        .x()
        .iter()
        .zip(layout.y().iter())
        .map(|(&x, &y)| (x - mx).powi(2) + (y - my).powi(2))
        .sum();
    (sum_sq / (2.0 * n)).sqrt()
}

#[test]
fn test_normalize_layout_2d_nan_collapsed_singleton_empty() {
    // NaN / Inf sanitization + exact isotropic std == 1.0
    let mut layout = Layout2d::from_xy(
        vec![f64::NAN, 10.0, 20.0, f64::INFINITY],
        vec![-5.0, f64::NEG_INFINITY, 15.0, 25.0],
    )
    .unwrap();
    normalize_layout_2d(&mut layout, 1.0, 42).unwrap();
    assert!(layout.is_all_finite());
    assert!((isotropic_std(&layout) - 1.0).abs() < 1e-12);

    // Collapsed layout falls back deterministically to Gaussian with target_std
    let mut collapsed = Layout2d::from_xy(vec![3.0; 8], vec![7.0; 8]).unwrap();
    normalize_layout_2d(&mut collapsed, 2.5, 7).unwrap();
    assert!(collapsed.is_all_finite());
    assert!((isotropic_std(&collapsed) - 2.5).abs() < 1e-12);

    // Singleton and empty
    let mut single = Layout2d::from_xy(vec![f64::NAN], vec![5.0]).unwrap();
    normalize_layout_2d(&mut single, 1.0, 42).unwrap();
    assert_eq!(single.x(), &[0.0]);
    assert_eq!(single.y(), &[0.0]);

    let mut empty = Layout2d::<f64>::zeros(0);
    normalize_layout_2d(&mut empty, 1.0, 42).unwrap();
}

#[test]
fn test_pivot_mds_std_and_disconnected_safety() {
    // 8x8 grid (64 nodes)
    let mut edges = Vec::new();
    for r in 0..8 {
        for c in 0..8 {
            let u = (r * 8 + c) as u32;
            if c + 1 < 8 {
                edges.push((u, u + 1));
            }
            if r + 1 < 8 {
                edges.push((u, u + 8));
            }
        }
    }
    let grid = CsrGraph::<f64>::from_edges(64, &edges, None, false, false).unwrap();
    let layout: Layout2d<f64> = PivotMds::new(&grid).generate().unwrap();
    assert_eq!(layout.len(), 64);
    assert!(layout.is_all_finite());
    let std_val = isotropic_std(&layout);
    assert!(
        (std_val - 1.0).abs() < 1e-5,
        "PivotMds std expected 1.0, got {std_val}"
    );

    // Disconnected graph: two 4-cycles + 2 isolated vertices
    let disc = CsrGraph::<f64>::from_edges(
        10,
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
        ],
        None,
        false,
        false,
    )
    .unwrap();
    let disc_layout: Layout2d<f64> = PivotMds::new(&disc).generate().unwrap();
    assert!(disc_layout.is_all_finite());
    assert!((isotropic_std(&disc_layout) - 1.0).abs() < 1e-5);

    // Empty and singleton
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let empty_layout: Layout2d<f64> = PivotMds::new(&empty).generate().unwrap();
    assert_eq!(empty_layout.len(), 0);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let s_lay: Layout2d<f64> = PivotMds::new(&single).generate().unwrap();
    assert_eq!(s_lay.x(), &[0.0]);
    assert_eq!(s_lay.y(), &[0.0]);
}

#[test]
fn test_koren_spectral_init_and_rd_and_classical_mds() {
    // Cycle C_20 embeds cleanly on a circle under Koren spectral init
    let edges: Vec<(u32, u32)> = (0..20).map(|i| (i, (i + 1) % 20)).collect();
    let cycle = CsrGraph::<f64>::from_edges(20, &edges, None, false, false).unwrap();

    let koren: Layout2d<f64> = KorenSpectral::new(&cycle)
        .with_target_std(Some(1.0))
        .with_tol(1e-6)
        .with_max_iters(300)
        .with_seed(42)
        .generate()
        .unwrap();
    assert!(koren.is_all_finite());
    assert!((isotropic_std(&koren) - 1.0).abs() < 1e-5);
    // Adjacent cycle vertices should have uniform chord length in 2D spectral init
    let mut step_dists = Vec::new();
    for i in 0..20 {
        let j = (i + 1) % 20;
        let dx = koren.x()[i] - koren.x()[j];
        let dy = koren.y()[i] - koren.y()[j];
        step_dists.push(dx.hypot(dy));
    }
    let mean_d = step_dists.iter().sum::<f64>() / 20.0;
    for &d in &step_dists {
        assert!((d - mean_d).abs() < 1e-4);
    }

    // RdMds on disconnected graph must remain finite (zero-eigenvalue guard)
    let disc = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
        None,
        false,
        false,
    )
    .unwrap();
    let rd: LayoutNd<f64> = RdMds::new(&disc)
        .with_dims(3)
        .with_eig_eps(1e-10)
        .with_tol(1e-6)
        .with_max_iters(200)
        .with_seed(42)
        .generate()
        .unwrap();
    assert!(rd.is_all_finite());
    // First mode corresponds to lambda_1 == 0 across components -> safely zeroed out
    for i in 0..6 {
        assert_eq!(rd.node(i).unwrap()[0], 0.0);
    }

    // Classical MDS recovers exact Euclidean coordinates of a 4-vertex unit square
    let mut d_mat = vec![0.0; 16];
    let pts = [(0.0_f64, 0.0_f64), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    for i in 0..4 {
        for j in 0..4 {
            d_mat[i * 4 + j] = (pts[i].0 - pts[j].0).hypot(pts[i].1 - pts[j].1);
        }
    }
    let cmds: LayoutNd<f64> = ClassicalMds::from_distances(&d_mat, 4)
        .with_dims(2)
        .generate()
        .unwrap();
    assert!(cmds.is_all_finite());
    for i in 0..4 {
        for j in (i + 1)..4 {
            let p_i = cmds.node(i).unwrap();
            let p_j = cmds.node(j).unwrap();
            let emb_d = (p_i[0] - p_j[0]).hypot(p_i[1] - p_j[1]);
            assert!(
                (emb_d - d_mat[i * 4 + j]).abs() < 1e-10,
                "Classical MDS distance mismatch ({i}, {j}): {emb_d} vs {}",
                d_mat[i * 4 + j]
            );
        }
    }

    // Graph-based ClassicalMds on disconnected graph stays finite
    let cmds_disc: LayoutNd<f64> = ClassicalMds::new(&disc).with_dims(2).generate().unwrap();
    assert!(cmds_disc.is_all_finite());

    // Weighted PivotMds + f32 Layout2d output + RdMds builder ergonomics
    let wg = CsrGraph::<f32>::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3), (3, 0)],
        Some(&[1.0_f32, 2.0, 1.0, 2.0]),
        false,
        false,
    )
    .unwrap();
    let lay_f32: Layout2d<f32> = PivotMds::new(&wg).generate().unwrap();
    assert!(lay_f32.is_all_finite());
    let rd_2d: Layout2d<f32> = RdMds::new(&wg)
        .with_dims(2)
        .with_eig_eps(1e-10)
        .with_tol(1e-6)
        .with_max_iters(200)
        .with_seed(42)
        .with_degree_normalize(true)
        .with_target_std(Some(1.0))
        .generate()
        .unwrap();
    assert!(rd_2d.is_all_finite());

    // Edgeless graph under KorenSpectral with target_std=Some(1.0) hits finite fallback
    let iso = CsrGraph::<f64>::from_edges(8, &[], None, false, false).unwrap();
    let k_iso: Layout2d<f64> = KorenSpectral::new(&iso)
        .with_target_std(Some(1.0))
        .with_tol(1e-6)
        .with_max_iters(100)
        .with_seed(42)
        .generate()
        .unwrap();
    assert!(k_iso.is_all_finite());
    assert!((isotropic_std(&k_iso) - 1.0).abs() < 1e-5);
}

#[test]
fn test_robust_normalize_layout_2d_outliers_fallbacks_and_edge_cases() {
    // 1. 96 core points + 4 extreme outliers at (500, 500): core std remains ~1.0
    let mut xs = Vec::with_capacity(100);
    let mut ys = Vec::with_capacity(100);
    let mut rng = DeterministicRng::from_seed_u64(123);
    for _ in 0..96 {
        let (gx, gy) = next_gaussian_pair(&mut rng);
        xs.push(gx);
        ys.push(gy);
    }
    for _ in 0..4 {
        xs.push(500.0);
        ys.push(500.0);
    }
    let mut layout = Layout2d::from_xy(xs, ys).unwrap();
    robust_normalize_layout_2d(&mut layout, 1.0, 0.95, 42).unwrap();
    assert!(layout.is_all_finite());
    let core_sum_sq: f64 = layout.x()[..96]
        .iter()
        .zip(layout.y()[..96].iter())
        .map(|(&x, &y)| x * x + y * y)
        .sum();
    let core_std = (core_sum_sq / (2.0 * 96.0)).sqrt();
    assert!(
        (core_std - 1.0).abs() < 0.35,
        "expected core_std ~ 1.0 under 4% outliers, got {core_std}"
    );

    // 2. >95% duplicate hub core + 2 distinct outer nodes falls back cleanly to isotropic_std
    let mut hub_xs = vec![10.0_f64; 98];
    let mut hub_ys = vec![-5.0_f64; 98];
    hub_xs.extend([12.0, 8.0]);
    hub_ys.extend([-5.0, -5.0]);
    let mut hub_layout = Layout2d::from_xy(hub_xs, hub_ys).unwrap();
    robust_normalize_layout_2d(&mut hub_layout, 1.5, 0.95, 42).unwrap();
    assert!(hub_layout.is_all_finite());
    assert!((isotropic_std(&hub_layout) - 1.5).abs() < 1e-12);

    // 3. All collapsed -> deterministic Gaussian fallback
    let mut collapsed = Layout2d::from_xy(vec![3.0_f32; 8], vec![7.0_f32; 8]).unwrap();
    robust_normalize_layout_2d(&mut collapsed, 2.0, 0.95, 7).unwrap();
    assert!(collapsed.is_all_finite());
    let mut collapsed2 = Layout2d::from_xy(vec![3.0_f32; 8], vec![7.0_f32; 8]).unwrap();
    robust_normalize_layout_2d(&mut collapsed2, 2.0, 0.95, 7).unwrap();
    assert_eq!(collapsed, collapsed2);

    // 4. n = 0, 1, 2, NaN/Inf
    let mut empty = Layout2d::<f64>::zeros(0);
    robust_normalize_layout_2d(&mut empty, 1.0, 0.95, 42).unwrap();

    let mut single = Layout2d::from_xy(vec![f64::NAN], vec![5.0]).unwrap();
    robust_normalize_layout_2d(&mut single, 1.0, 0.95, 42).unwrap();
    assert_eq!(single.x(), &[0.0]);
    assert_eq!(single.y(), &[0.0]);

    let mut two = Layout2d::from_xy(vec![f64::INFINITY, 2.0], vec![0.0, f64::NAN]).unwrap();
    robust_normalize_layout_2d(&mut two, 1.0, 0.95, 42).unwrap();
    assert!(two.is_all_finite());

    // 5. Validation errors
    assert!(matches!(
        robust_normalize_layout_2d(&mut two, 0.0, 0.95, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        robust_normalize_layout_2d(&mut two, f64::NAN, 0.95, 42),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        robust_normalize_layout_2d(&mut two, 1.0, 1.0, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        robust_normalize_layout_2d(&mut two, 1.0, 0.0, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        robust_normalize_layout_2d(&mut two, 1.0, f64::NAN, 42),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_pivot_mds_streaming_matches_all_pivots_classical_mds() {
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)];
    let g = CsrGraph::<f64>::from_edges(5, &edges, None, false, false).unwrap();
    let all_pivots: Vec<u32> = (0..5).collect();
    let pmds: Layout2d<f64> = PivotMds::new(&g)
        .with_pivots(&all_pivots)
        .with_target_std(None)
        .generate()
        .unwrap();
    let cmds: Layout2d<f64> = ClassicalMds::new(&g).generate().unwrap();
    for i in 0..5 {
        for j in (i + 1)..5 {
            let dp = (pmds.x()[i] - pmds.x()[j]).hypot(pmds.y()[i] - pmds.y()[j]);
            let dc = (cmds.x()[i] - cmds.x()[j]).hypot(cmds.y()[i] - cmds.y()[j]);
            assert!((dp - dc).abs() < 1e-10, "pair ({i},{j}): {dp} vs {dc}");
        }
    }
}

#[test]
fn test_pivot_mds_from_pivot_distances_bit_identical_and_validation() {
    use egraph_core::Layout3d;
    use egraph_linalg::BfsWorkspace;

    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 3)];
    let g = CsrGraph::<()>::from_unweighted_edges(6, &edges, false, false).unwrap();
    let pivots = [0u32, 2, 4];
    let n = g.num_nodes();
    let k = pivots.len();

    let mut dists_kn = vec![f64::INFINITY; k * n];
    let mut ws = BfsWorkspace::new(n);
    for (p_idx, &p) in pivots.iter().enumerate() {
        ws.bfs_from(&g, p as usize, &mut dists_kn[p_idx * n..(p_idx + 1) * n])
            .unwrap();
    }
    let mut dists_nk = vec![0.0_f64; n * k];
    for p_idx in 0..k {
        for i in 0..n {
            dists_nk[i * k + p_idx] = dists_kn[p_idx * n + i];
        }
    }

    let from_kn_2d: Layout2d<f64> = PivotMds::from_pivot_distances(&dists_kn, n, k)
        .with_target_std(None)
        .generate()
        .unwrap();
    let from_nk_2d: Layout2d<f64> = PivotMds::from_distances(&dists_nk, n, k)
        .with_target_std(None)
        .generate()
        .unwrap();
    let from_graph_2d: Layout2d<f64> = PivotMds::new(&g)
        .with_pivots(&pivots)
        .with_target_std(None)
        .generate()
        .unwrap();
    assert_eq!(from_kn_2d, from_nk_2d);
    assert_eq!(from_kn_2d, from_graph_2d);

    let from_kn_3d: Layout3d<f64> = PivotMds::from_pivot_distances(&dists_kn, n, k)
        .generate()
        .unwrap();
    let from_nk_3d: Layout3d<f64> = PivotMds::from_distances(&dists_nk, n, k)
        .generate()
        .unwrap();
    assert_eq!(from_kn_3d, from_nk_3d);

    // Disconnected +f64::INFINITY allowed; validation rejects with_pivots, zero pivots, mismatches, NaN, negative
    let disc_kn = vec![0.0, 1.0, f64::INFINITY, f64::INFINITY, f64::INFINITY, 0.0];
    let disc_lay: Layout2d<f64> = PivotMds::from_pivot_distances(&disc_kn, 3, 2)
        .generate()
        .unwrap();
    assert!(disc_lay.is_all_finite());

    assert!(matches!(
        PivotMds::<Layout2d<f64>>::from_pivot_distances(&dists_kn, n, k)
            .with_pivots(&pivots)
            .generate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        PivotMds::<Layout2d<f64>>::from_pivot_distances(&[], n, 0).generate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        PivotMds::<Layout2d<f64>>::from_pivot_distances(&dists_kn[..k * n - 1], n, k).generate(),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        PivotMds::<Layout2d<f64>>::from_pivot_distances(&[0.0, f64::NAN], 2, 1).generate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        PivotMds::<Layout2d<f64>>::from_pivot_distances(&[0.0, -0.5], 2, 1).generate(),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_all_mds_generators_unit_graph_and_rd_mds_from_eigenpairs() {
    use egraph_linalg::{LanczosEigen, smallest_sym_laplacian_eigenpairs_with_config};

    let mut edges = Vec::new();
    for r in 0..10usize {
        for c in 0..10usize {
            let u = (r * 10 + c) as u32;
            if c + 1 < 10 {
                edges.push((u, u + 1));
            }
            if r + 1 < 10 {
                edges.push((u, u + 10));
            }
        }
    }
    let g_unit = CsrGraph::<()>::from_unweighted_edges(100, &edges, false, false).unwrap();
    let g_f64 = CsrGraph::<f64>::from_edges(100, &edges, None, false, false).unwrap();

    // All 5 generators support CsrGraph<()> and match CsrGraph<f64> bit-for-bit
    let small_edges = vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)];
    let s_unit = CsrGraph::<()>::from_unweighted_edges(4, &small_edges, false, false).unwrap();
    let s_f64 = CsrGraph::<f64>::from_edges(4, &small_edges, None, false, false).unwrap();
    assert_eq!(
        ClassicalMds::<Layout2d<f64>, _>::new(&s_unit)
            .generate()
            .unwrap(),
        ClassicalMds::<Layout2d<f64>, _>::new(&s_f64)
            .generate()
            .unwrap()
    );
    assert_eq!(
        PivotMds::<Layout2d<f64>, _>::new(&g_unit)
            .generate()
            .unwrap(),
        PivotMds::<Layout2d<f64>, _>::new(&g_f64)
            .generate()
            .unwrap()
    );
    assert_eq!(
        Hde::<Layout2d<f64>, _>::new(&g_unit).generate().unwrap(),
        Hde::<Layout2d<f64>, _>::new(&g_f64).generate().unwrap()
    );
    assert_eq!(
        KorenSpectral::<Layout2d<f64>, _>::new(&g_unit)
            .generate()
            .unwrap(),
        KorenSpectral::<Layout2d<f64>, _>::new(&g_f64)
            .generate()
            .unwrap()
    );

    // RdMds with subspace_dim matches explicit LanczosEigen + RdMds::from_eigenpairs
    let rd_cfg = RdMdsConfig::new()
        .with_dims(Some(2))
        .with_subspace_dim(Some(24))
        .with_tol(1e-7)
        .with_max_iters(400)
        .with_seed(42)
        .with_target_std(Some(1.5));
    let rd_unit: Layout2d<f64> = RdMds::from_config(&g_unit, rd_cfg.clone())
        .generate()
        .unwrap();
    let rd_f64: Layout2d<f64> = RdMds::from_config(&g_f64, rd_cfg.clone())
        .generate()
        .unwrap();
    assert_eq!(rd_unit, rd_f64);

    let lanczos = LanczosEigen::new(rd_cfg.tol, rd_cfg.max_iters, rd_cfg.seed)
        .with_subspace_dim(rd_cfg.subspace_dim);
    let eig = smallest_sym_laplacian_eigenpairs_with_config(&g_unit, 2, &lanczos).unwrap();
    let rd_from_eig: Layout2d<f64> = RdMds::from_eigenpairs(
        &eig.lambdas,
        &eig.vectors,
        &eig.inv_sqrt_deg,
        eig.num_nodes,
        eig.num_modes,
    )
    .with_config(rd_cfg)
    .generate()
    .unwrap();
    assert_eq!(rd_unit, rd_from_eig);

    // RdMds::from_eigenpairs validation errors
    assert!(matches!(
        RdMds::<Layout2d<f64>>::from_eigenpairs(&[0.5], &eig.vectors, &eig.inv_sqrt_deg, 100, 2)
            .generate(),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        RdMds::<Layout2d<f64>>::from_eigenpairs(
            &eig.lambdas,
            &eig.vectors[..10],
            &eig.inv_sqrt_deg,
            100,
            2
        )
        .generate(),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        RdMds::<Layout2d<f64>>::from_eigenpairs(
            &eig.lambdas,
            &eig.vectors,
            &eig.inv_sqrt_deg[..10],
            100,
            2
        )
        .generate(),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        RdMds::<Layout2d<f64>>::from_eigenpairs(
            &[f64::NAN, 0.5],
            &eig.vectors,
            &eig.inv_sqrt_deg,
            100,
            2
        )
        .generate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
