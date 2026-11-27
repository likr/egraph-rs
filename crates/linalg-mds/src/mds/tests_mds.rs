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
