use egraph_core::{CoreError, CsrGraph, DeterministicRng, Layout2d, Scalar};
use egraph_linalg::{BfsWorkspace, select_maxmin_sp_pivots};

use crate::stress::all_pairs_bfs;

use super::row_stats::{RowScratch, counting_sort_row};
use super::*;

fn brute_force_all_comparable_rta<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    landmark_ids: &[u32],
) -> (f64, usize) {
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut ws = BfsWorkspace::new(n);
    let mut row = vec![u32::MAX; n];
    let mut ok = 0usize;
    let mut eval = 0usize;

    for &lid in landmark_ids {
        ws.bfs_from_u32(graph, lid as usize, &mut row).unwrap();
        let px = xs[lid as usize].to_f64();
        let py = ys[lid as usize].to_f64();
        for u in 0..n {
            let d1 = row[u];
            if !(1..u32::MAX).contains(&d1) {
                continue;
            }
            let r1_sq = (xs[u].to_f64() - px).powi(2) + (ys[u].to_f64() - py).powi(2);
            for v in (u + 1)..n {
                let d2 = row[v];
                if !(1..u32::MAX).contains(&d2) || d1 == d2 {
                    continue;
                }
                let r2_sq = (xs[v].to_f64() - px).powi(2) + (ys[v].to_f64() - py).powi(2);
                eval += 1;
                if (d1 < d2 && r1_sq < r2_sq) || (d1 > d2 && r1_sq > r2_sq) {
                    ok += 1;
                }
            }
        }
    }
    let rta = if eval > 0 {
        ok as f64 / eval as f64
    } else {
        0.0
    };
    (rta, eval)
}

#[test]
fn test_rta_exact_oracle_similarity_invariance_and_stratified_sampling() {
    // 1. Path graph laid out cleanly on x-axis -> RTA == 1.0, exact fallback.
    let path_edges: Vec<(u32, u32)> = (0..11).map(|i| (i, i + 1)).collect();
    let path = CsrGraph::<f64>::from_edges(12, &path_edges, None, false, false).unwrap();
    let path_x: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let path_y = vec![0.0; 12];
    let path_l = Layout2d::from_xy(path_x.clone(), path_y.clone()).unwrap();

    let det_path = landmark_stratified_rta_detail(&path, &path_l, 4, 100_000, 1, 42).unwrap();
    assert!(det_path.is_exact);
    assert!((det_path.rta - 1.0).abs() < 1e-12);

    // Compare directly against O(P_L * N^2) brute-force oracle on a warped path + grid.
    let warped_l = Layout2d::from_xy(
        path_x
            .iter()
            .enumerate()
            .map(|(i, &x)| x + if i % 3 == 0 { 2.5 } else { -0.4 })
            .collect(),
        (0..12).map(|i| ((i * 7) % 5) as f64 * 0.6).collect(),
    )
    .unwrap();
    let lm_ids = select_maxmin_sp_pivots(&path, 4, 42, 1).unwrap();
    let (oracle_rta, oracle_cnt) = brute_force_all_comparable_rta(&path, &warped_l, &lm_ids);
    let got_warped = landmark_stratified_rta_detail(&path, &warped_l, 4, 100_000, 1, 42).unwrap();
    assert!(got_warped.is_exact);
    assert_eq!(got_warped.num_triplets, oracle_cnt);
    assert!((got_warped.rta - oracle_rta).abs() < 1e-12);

    // Similarity transform invariance (scale + rotate + translate).
    let theta = 0.65_f64;
    let (st, ct) = theta.sin_cos();
    let sim_x: Vec<f64> = (0..12)
        .map(|i| 4.2 * (ct * warped_l.x()[i] - st * warped_l.y()[i]) + 15.0)
        .collect();
    let sim_y: Vec<f64> = (0..12)
        .map(|i| 4.2 * (st * warped_l.x()[i] + ct * warped_l.y()[i]) - 9.0)
        .collect();
    let sim_l = Layout2d::from_xy(sim_x, sim_y).unwrap();
    let sim_rta = landmark_stratified_rta(&path, &sim_l, 4, 100_000, 1, 42).unwrap();
    assert!((sim_rta - oracle_rta).abs() < 1e-12);

    // Verify landmark_stratified_rta_from_landmarks parity with graph evaluation.
    let apsp = all_pairs_bfs(&path, 1.0).unwrap();
    let mut flat_u32 = Vec::with_capacity(lm_ids.len() * 12);
    for &lid in &lm_ids {
        let row_off = (lid as usize) * 12;
        for j in 0..12 {
            let d = apsp[row_off + j];
            flat_u32.push(if d.is_finite() { d as u32 } else { u32::MAX });
        }
    }
    let from_tbl =
        landmark_stratified_rta_from_landmarks(&warped_l, &lm_ids, &flat_u32, 100_000, 42).unwrap();
    assert_eq!(from_tbl, got_warped);

    // Stratified sampling branch (total_comparable > num_triplets): grid 8x8 (N=64).
    let mut g_edges = Vec::new();
    let mut gx = Vec::new();
    let mut gy = Vec::new();
    for r in 0..8u32 {
        for c in 0..8u32 {
            let id = r * 8 + c;
            gx.push(c as f64);
            gy.push(r as f64);
            if c + 1 < 8 {
                g_edges.push((id, id + 1));
            }
            if r + 1 < 8 {
                g_edges.push((id, id + 8));
            }
        }
    }
    let grid = CsrGraph::<f64>::from_edges(64, &g_edges, None, false, false).unwrap();
    let grid_l64 = Layout2d::from_xy(gx.clone(), gy.clone()).unwrap();
    let grid_l32 = Layout2d::from_xy(
        gx.iter().map(|&v| v as f32).collect(),
        gy.iter().map(|&v| v as f32).collect(),
    )
    .unwrap();

    let samp_a = landmark_stratified_rta_detail(&grid, &grid_l64, 8, 500, 2, 42).unwrap();
    let samp_b = landmark_stratified_rta_detail(&grid, &grid_l64, 8, 500, 2, 42).unwrap();
    let samp_32 = landmark_stratified_rta_detail(&grid, &grid_l32, 8, 500, 2, 42).unwrap();
    assert!(!samp_a.is_exact);
    assert_eq!(samp_a, samp_b);
    assert!((samp_a.rta - samp_32.rta).abs() < 1e-12);
    assert!(samp_a.rta > 0.85, "grid L-Stratified-RTA={}", samp_a.rta);

    // Random layout on grid should be near ~0.5, collapsed should be 0.0.
    let mut rng = DeterministicRng::from_seed_u64(777);
    let rand_l = Layout2d::from_xy(
        (0..64).map(|_| rng.next_f64()).collect(),
        (0..64).map(|_| rng.next_f64()).collect(),
    )
    .unwrap();
    let rand_rta = landmark_stratified_rta(&grid, &rand_l, 8, 2000, 2, 42).unwrap();
    assert!((0.35..0.65).contains(&rand_rta), "rand_rta={rand_rta}");

    let collapsed = Layout2d::<f64>::zeros(64);
    assert_eq!(
        landmark_stratified_rta(&grid, &collapsed, 8, 500, 2, 42).unwrap(),
        0.0
    );
}

#[test]
fn test_rta_validation_errors() {
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
    let l = Layout2d::from_xy(vec![0.0, 1.0, 2.0, 3.0], vec![0.0; 4]).unwrap();
    assert!(matches!(
        landmark_stratified_rta(&g, &l, 0, 100, 1, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        landmark_stratified_rta(&g, &l, 2, 0, 1, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        landmark_stratified_rta_from_landmarks(&l, &[0, 1], &[0; 5], 100, 42),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        landmark_stratified_rta_from_landmarks(&l, &[4], &[0; 4], 100, 42),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn test_rta_huge_synthetic_distances_no_oom_and_oracle_parity() {
    // Synthetic non-hop integer distances up to u32::MAX - 1 must use the O(N)-RAM sort
    // fallback instead of allocating O(max_d) memory, while preserving exact RTA results.
    let n = 8usize;
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let ys = vec![0.0; n];
    let layout = Layout2d::from_xy(xs, ys).unwrap();
    let landmark_ids = [0_u32];
    let huge_dists = [
        0_u32,
        10_000_000,
        10_000_000,
        250_000_000,
        500_000_000,
        1_000_000_000,
        2_000_000_000,
        u32::MAX - 1,
    ];

    // Exact branch (total_comparable <= num_triplets).
    let exact =
        landmark_stratified_rta_from_landmarks(&layout, &landmark_ids, &huge_dists, 1000, 42)
            .unwrap();
    assert!(exact.is_exact);
    assert!((exact.rta - 1.0).abs() < 1e-12);
    assert_eq!(exact.num_triplets, 20); // C(7,2) - 1 equal pair (nodes 1,2)

    // Stratified sampling branch (total_comparable > num_triplets).
    let sampled =
        landmark_stratified_rta_from_landmarks(&layout, &landmark_ids, &huge_dists, 10, 42)
            .unwrap();
    assert!(!sampled.is_exact);
    assert!((sampled.rta - 1.0).abs() < 1e-12);

    // Verify fallback helper matches counting sort bit-for-bit on small distances too.
    let mut s_cnt = RowScratch::new(n);
    let mut s_sort = RowScratch::new(n);
    let small_row = [0_u32, 3, 1, 3, 2, u32::MAX, 5, 1];
    let oct_cnt = counting_sort_row(&small_row, &mut s_cnt);
    s_sort.sorted_pairs.clear();
    for (u, &d) in small_row.iter().enumerate() {
        if (1..u32::MAX).contains(&d) {
            s_sort.sorted_pairs.push((d, u as u32));
        }
    }
    s_sort.sorted_pairs.sort_unstable();
    let mut exp_nodes = Vec::new();
    let mut exp_dists = Vec::new();
    let mut oct_counts = [0usize; 32];
    for &(d, u) in &s_sort.sorted_pairs {
        exp_dists.push(d);
        exp_nodes.push(u);
        oct_counts[(31 - d.leading_zeros()) as usize] += 1;
    }
    let mut exp_oct = [0usize; 33];
    for b in 0..32 {
        exp_oct[b + 1] = exp_oct[b] + oct_counts[b];
    }
    assert_eq!(oct_cnt, exp_oct);
    assert_eq!(s_cnt.sorted_nodes, exp_nodes);
    assert_eq!(s_cnt.sorted_dists, exp_dists);
}

#[test]
fn test_rta_from_landmarks_f64_infinity_and_validation() {
    let layout = Layout2d::from_xy(
        vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0],
        vec![0.0, 0.0, 0.0, 0.0, 5.0, 5.0],
    )
    .unwrap();
    let lm_ids = [0_u32, 4_u32];
    let dists_u32 = [
        0_u32,
        1,
        2,
        3,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        0,
        1,
    ];
    let dists_f64 = [
        0.0,
        1.0,
        2.0,
        3.0,
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        0.0,
        1.0,
    ];

    let ref_eval =
        landmark_stratified_rta_from_landmarks(&layout, &lm_ids, &dists_u32, 1000, 42).unwrap();
    let f64_eval =
        landmark_stratified_rta_from_landmarks_f64(&layout, &lm_ids, &dists_f64, 1000, 42).unwrap();
    assert_eq!(f64_eval, ref_eval);

    // Reject NaN, negative, -inf, fractional non-integer, and > u32::MAX as f64.
    let mut bad_nan = dists_f64;
    bad_nan[1] = f64::NAN;
    assert!(matches!(
        landmark_stratified_rta_from_landmarks_f64(&layout, &lm_ids, &bad_nan, 1000, 42),
        Err(CoreError::NonFiniteValue { .. })
    ));

    for bad_val in [-1.0, -0.5, f64::NEG_INFINITY, 1.5, (u32::MAX as f64) + 1.0] {
        let mut bad = dists_f64;
        bad[1] = bad_val;
        assert!(
            matches!(
                landmark_stratified_rta_from_landmarks_f64(&layout, &lm_ids, &bad, 1000, 42),
                Err(CoreError::InvalidInput(_))
            ),
            "expected InvalidInput for {bad_val}"
        );
    }
}
