use super::types::dist_sq;
use super::*;
use egraph_core::DeterministicRng;

fn brute_knn<const D: usize>(
    pts: &[[f64; D]],
    q: &[f64; D],
    k: usize,
    exclude: Option<usize>,
) -> Vec<(usize, f64)> {
    let mut all: Vec<(usize, f64)> = pts
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != exclude)
        .map(|(i, p)| (i, dist_sq(q, p)))
        .collect();
    all.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    all.truncate(k);
    all
}

fn brute_within<const D: usize>(pts: &[[f64; D]], q: &[f64; D], r2: f64) -> Vec<(usize, f64)> {
    let mut all: Vec<(usize, f64)> = pts
        .iter()
        .enumerate()
        .map(|(i, p)| (i, dist_sq(q, p)))
        .filter(|&(_, d2)| d2 <= r2)
        .collect();
    all.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    all
}

#[test]
fn test_kdtree_exact_parity_random_collinear_and_duplicates() {
    // 1. Random 2D points (N > LEAF_SIZE so multiple tree levels are exercised).
    let mut rng = DeterministicRng::from_seed_u64(1337);
    let random_pts: Vec<[f64; 2]> = (0..120)
        .map(|_| [rng.next_f64() * 20.0 - 10.0, rng.next_f64() * 20.0 - 10.0])
        .collect();

    // 2. Collinear integer grid with abundant distance ties.
    let mut collinear_pts = Vec::new();
    for i in 0..45 {
        collinear_pts.push([(i % 15) as f64, 3.0]);
    }

    // 3. Heavy duplicates + isolated outliers.
    let mut duplicate_pts = vec![[1.5, -2.5]; 40];
    duplicate_pts.extend(vec![[-3.0, 4.0]; 20]);
    duplicate_pts.push([0.0, 0.0]);

    for pts in [&random_pts, &collinear_pts, &duplicate_pts] {
        let tree = StaticKdTree::<2, f64>::build(pts).unwrap();
        assert_eq!(tree.len(), pts.len());

        for i in 0..pts.len() {
            for &k in &[1usize, 5, 20, 60] {
                let got = tree.k_nearest_excluding_self(i, k).unwrap();
                let expected = brute_knn(pts, &pts[i], k, Some(i));
                assert_eq!(got, expected, "k={k}, i={i}");
            }
            let nearest = tree.nearest_excluding_self(i).unwrap();
            let expected_1 = brute_knn(pts, &pts[i], 1, Some(i)).into_iter().next();
            assert_eq!(nearest, expected_1);
        }

        let queries = [[0.0, 0.0], [1.5, -2.5], [7.0, 3.0], [-12.0, 12.0]];
        for q in &queries {
            assert_eq!(tree.k_nearest(q, 10).unwrap(), brute_knn(pts, q, 10, None));
            assert_eq!(
                tree.within_radius(q, 3.0).unwrap(),
                brute_within(pts, q, 9.0)
            );
            assert_eq!(
                tree.within_radius_sq(q, 0.0).unwrap(),
                brute_within(pts, q, 0.0)
            );
        }
    }
}

#[test]
fn test_kdtree_3d_layout_and_error_cases() {
    let layout3 = LayoutNd::from_flat(
        4,
        3,
        vec![
            0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let tree3 = StaticKdTree::<3, f32>::from_layout_nd(&layout3).unwrap();
    let nn = tree3.nearest_excluding_self(0).unwrap().unwrap();
    assert_eq!(nn, (1, 1.0));

    let empty = StaticKdTree::<2, f64>::build(&[]).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.k_nearest(&[0.0, 0.0], 3).unwrap(), Vec::new());

    let single = StaticKdTree::<2, f64>::build(&[[2.0, 3.0]]).unwrap();
    assert_eq!(single.nearest_excluding_self(0).unwrap(), None);

    assert!(matches!(
        StaticKdTree::<0, f64>::build(&[]),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        StaticKdTree::<2, f64>::build(&[[f64::NAN, 0.0]]),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        single.within_radius(&[0.0, 0.0], -1.0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        single.nearest_excluding_self(1),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn test_kdtree_scratch_reuse_parity() {
    let mut pts = Vec::with_capacity(65);
    for i in 0..65usize {
        let x = ((i * 37) % 19) as f64 * 0.25;
        let y = ((i * 53) % 17) as f64 * 0.25;
        pts.push([x, y]);
    }
    let tree = StaticKdTree::<2, f64>::build(&pts).unwrap();
    let mut scratch = KdQueryScratch::with_capacity(1);
    let mut out_scratch = Vec::new();
    let mut out_direct = Vec::new();

    for &k in &[1usize, 5, 32, 16, 0, 64] {
        for i in 0..pts.len() {
            tree.k_nearest_excluding_self_into(i, k, &mut out_direct)
                .unwrap();
            tree.k_nearest_excluding_self_with_scratch(i, k, &mut scratch, &mut out_scratch)
                .unwrap();
            assert_eq!(out_direct.len(), out_scratch.len());
            for (a, b) in out_direct.iter().zip(out_scratch.iter()) {
                assert_eq!(a.0, b.0);
                assert_eq!(a.1.to_bits(), b.1.to_bits());
            }
        }
        tree.k_nearest_into(&[1.25, 2.0], k, &mut out_direct)
            .unwrap();
        tree.k_nearest_with_scratch(&[1.25, 2.0], k, &mut scratch, &mut out_scratch)
            .unwrap();
        assert_eq!(out_direct.len(), out_scratch.len());
        for (a, b) in out_direct.iter().zip(out_scratch.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1.to_bits(), b.1.to_bits());
        }
    }
}

fn brute_knn_s<const D: usize, S: Scalar>(
    pts: &[[S; D]],
    q: &[S; D],
    k: usize,
    exclude: Option<usize>,
) -> Vec<(usize, f64)> {
    let mut q64 = [0.0_f64; D];
    for d in 0..D {
        q64[d] = q[d].to_f64();
    }
    let mut all: Vec<(usize, f64)> = pts
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != exclude)
        .map(|(i, p)| {
            let mut p64 = [0.0_f64; D];
            for d in 0..D {
                p64[d] = p[d].to_f64();
            }
            (i, dist_sq(&q64, &p64))
        })
        .collect();
    all.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    all.truncate(k);
    all
}

fn brute_within_s<const D: usize, S: Scalar>(
    pts: &[[S; D]],
    q: &[S; D],
    r2: f64,
) -> Vec<(usize, f64)> {
    let mut q64 = [0.0_f64; D];
    for d in 0..D {
        q64[d] = q[d].to_f64();
    }
    let mut all: Vec<(usize, f64)> = pts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let mut p64 = [0.0_f64; D];
            for d in 0..D {
                p64[d] = p[d].to_f64();
            }
            (i, dist_sq(&q64, &p64))
        })
        .filter(|&(_, d2)| d2 <= r2)
        .collect();
    all.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    all
}

#[test]
fn test_kdtree_multiseed_f32_3d_median_split_and_exact_radius_oracles() {
    // 1. Multi-seed 2D f32 & 3D f32/f64 randomized oracles (N > LEAF_SIZE=16).
    for seed in 1..=8u64 {
        let mut rng = DeterministicRng::from_seed_u64(3_000 + seed);
        let pts2_f32: Vec<[f32; 2]> = (0..80)
            .map(|_| {
                [
                    (rng.next_f64() * 10.0 - 5.0) as f32,
                    (rng.next_f64() * 10.0 - 5.0) as f32,
                ]
            })
            .collect();
        let tree2 = StaticKdTree::<2, f32>::build(&pts2_f32).unwrap();
        for i in [0usize, 17, 42, 79] {
            for &k in &[1usize, 5, 25] {
                assert_eq!(
                    tree2.k_nearest_excluding_self(i, k).unwrap(),
                    brute_knn_s(&pts2_f32, &pts2_f32[i], k, Some(i))
                );
            }
        }
        let q2 = [0.5_f32, -1.25_f32];
        assert_eq!(
            tree2.within_radius(&q2, 2.5).unwrap(),
            brute_within_s(&pts2_f32, &q2, 6.25)
        );

        let pts3_f64: Vec<[f64; 3]> = (0..75)
            .map(|_| {
                [
                    rng.next_f64() * 8.0 - 4.0,
                    rng.next_f64() * 8.0 - 4.0,
                    rng.next_f64() * 8.0 - 4.0,
                ]
            })
            .collect();
        let tree3 = StaticKdTree::<3, f64>::build(&pts3_f64).unwrap();
        let pts3_f32: Vec<[f32; 3]> = pts3_f64
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        let tree3_f32 = StaticKdTree::<3, f32>::build(&pts3_f32).unwrap();
        for i in [0usize, 15, 37, 74] {
            assert_eq!(
                tree3.k_nearest_excluding_self(i, 10).unwrap(),
                brute_knn(&pts3_f64, &pts3_f64[i], 10, Some(i))
            );
            assert_eq!(
                tree3_f32.k_nearest_excluding_self(i, 10).unwrap(),
                brute_knn_s(&pts3_f32, &pts3_f32[i], 10, Some(i))
            );
        }
        let q3 = [1.0_f64, -1.0, 0.5];
        assert_eq!(
            tree3.within_radius(&q3, 3.0).unwrap(),
            brute_within(&pts3_f64, &q3, 9.0)
        );
    }

    // 2. Axis-aligned median-split tie-break oracle (many points on exact split plane x=2.0 / y=-1.0).
    let mut tie_pts = Vec::with_capacity(96);
    for i in 0..96usize {
        let x = match i % 3 {
            0 => 2.0,
            1 => (i % 7) as f64,
            _ => -((i % 5) as f64),
        };
        let y = match i % 4 {
            0 => -1.0,
            _ => ((i * 3) % 11) as f64 - 5.0,
        };
        tie_pts.push([x, y]);
    }
    let tie_tree = StaticKdTree::<2, f64>::build(&tie_pts).unwrap();
    for i in 0..tie_pts.len() {
        assert_eq!(
            tie_tree.k_nearest_excluding_self(i, 12).unwrap(),
            brute_knn(&tie_pts, &tie_pts[i], 12, Some(i)),
            "median-split tie-break mismatch at i={i}"
        );
    }

    // 3. Exact boundary d_sq == r_sq inclusion in within_radius_sq and within_radius.
    let mut boundary_pts = Vec::new();
    for x in -4..=4 {
        for y in -4..=4 {
            boundary_pts.push([x as f64, y as f64]);
        }
    }
    let b_tree = StaticKdTree::<2, f64>::build(&boundary_pts).unwrap();
    for &r_sq in &[0.0_f64, 1.0, 2.0, 4.0, 5.0, 9.0, 10.0] {
        let q = [0.0, 0.0];
        assert_eq!(
            b_tree.within_radius_sq(&q, r_sq).unwrap(),
            brute_within(&boundary_pts, &q, r_sq),
            "within_radius_sq boundary r_sq={r_sq}"
        );
    }
    assert_eq!(
        b_tree.within_radius(&[0.0, 0.0], 2.0).unwrap(),
        brute_within(&boundary_pts, &[0.0, 0.0], 4.0)
    );
}
