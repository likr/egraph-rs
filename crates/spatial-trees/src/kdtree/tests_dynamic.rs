use egraph_core::{CoreError, DeterministicRng, LayoutNd, Scalar};

use super::*;

fn brute_knn_nd(
    coords: &[f64],
    dims: usize,
    q: &[f64],
    k: usize,
    exclude: Option<usize>,
) -> Vec<(usize, f64)> {
    let n = coords.len() / dims;
    let mut all: Vec<(usize, f64)> = (0..n)
        .filter(|i| Some(*i) != exclude)
        .map(|i| {
            let mut d2 = 0.0;
            for d in 0..dims {
                let diff = q[d] - coords[i * dims + d];
                d2 += diff * diff;
            }
            (i, d2)
        })
        .collect();
    all.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    all.truncate(k);
    all
}

fn brute_within_nd(coords: &[f64], dims: usize, q: &[f64], r2: f64) -> Vec<(usize, f64)> {
    let n = coords.len() / dims;
    let mut all: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let mut d2 = 0.0;
            for d in 0..dims {
                let diff = q[d] - coords[i * dims + d];
                d2 += diff * diff;
            }
            (i, d2)
        })
        .filter(|&(_, d2)| d2 <= r2)
        .collect();
    all.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    all
}

#[test]
fn test_dynamic_kdtree_brute_force_oracle_across_dims_f64_and_f32() {
    for &dims in &[1usize, 2, 3, 4, 8, 12] {
        let n = 60;
        let mut rng = DeterministicRng::from_seed_u64(500 + dims as u64);
        let flat64: Vec<f64> = (0..n * dims).map(|_| rng.next_f64() * 8.0 - 4.0).collect();
        let layout64 = LayoutNd::from_flat(n, dims, flat64.clone()).unwrap();
        let tree64 = DynamicKdTree::from_layout_nd(&layout64).unwrap();
        assert_eq!(tree64.dims(), dims);
        assert_eq!(tree64.len(), n);
        assert!(!tree64.is_empty());
        assert_eq!(tree64.coords_by_id(), flat64.as_slice());

        let mut scratch = KdQueryScratch::with_capacity(12);
        let mut out = Vec::new();
        for i in [0usize, 13, 29, 59] {
            let nn = tree64
                .nearest_excluding_self_with_scratch(i, &mut scratch)
                .unwrap();
            let exp_1 = brute_knn_nd(&flat64, dims, &flat64[i * dims..(i + 1) * dims], 1, Some(i))
                .into_iter()
                .next();
            assert_eq!(nn, exp_1, "nearest mismatch dims={dims} i={i}");
            assert_eq!(tree64.nearest_excluding_self(i).unwrap(), exp_1);

            for &k in &[1usize, 5, 16, 45] {
                tree64
                    .k_nearest_excluding_self_with_scratch(i, k, &mut scratch, &mut out)
                    .unwrap();
                let exp_k =
                    brute_knn_nd(&flat64, dims, &flat64[i * dims..(i + 1) * dims], k, Some(i));
                assert_eq!(out, exp_k, "knn excluding self dims={dims} k={k} i={i}");
            }
        }

        let q64 = vec![0.25_f64; dims];
        assert_eq!(
            tree64.k_nearest(&q64, 10).unwrap(),
            brute_knn_nd(&flat64, dims, &q64, 10, None)
        );
        let r_sq = 12.0 * (dims as f64);
        tree64.within_radius_sq_into(&q64, r_sq, &mut out).unwrap();
        assert_eq!(out, brute_within_nd(&flat64, dims, &q64, r_sq));

        // Verify f32 parity using exact f64 conversion oracle.
        let flat32: Vec<f32> = flat64.iter().map(|&v| v as f32).collect();
        let flat32_as_64: Vec<f64> = flat32.iter().map(|v| v.to_f64()).collect();
        let tree32 = DynamicKdTree::<f32>::from_flat(&flat32, n, dims).unwrap();
        assert_eq!(tree32.coords_by_id(), flat32_as_64.as_slice());

        for i in [0usize, 17, 41, 58] {
            tree32
                .k_nearest_excluding_self_with_scratch(i, 7, &mut scratch, &mut out)
                .unwrap();
            assert_eq!(
                out,
                brute_knn_nd(
                    &flat32_as_64,
                    dims,
                    &flat32_as_64[i * dims..(i + 1) * dims],
                    7,
                    Some(i)
                )
            );
            assert_eq!(
                tree32
                    .nearest_excluding_self_with_scratch(i, &mut scratch)
                    .unwrap(),
                brute_knn_nd(
                    &flat32_as_64,
                    dims,
                    &flat32_as_64[i * dims..(i + 1) * dims],
                    1,
                    Some(i)
                )
                .into_iter()
                .next()
            );
        }
        let q32 = vec![-0.5_f32; dims];
        let q32_64 = vec![f64::from(-0.5_f32); dims];
        tree32.within_radius_sq_into(&q32, r_sq, &mut out).unwrap();
        assert_eq!(out, brute_within_nd(&flat32_as_64, dims, &q32_64, r_sq));
    }
}

#[test]
fn test_dynamic_kdtree_coincident_ties_empty_single_and_validation() {
    // 1. Coincident & tied points (N > LEAF_SIZE=16 with duplicate clusters in 4D).
    let dims = 4;
    let mut flat = Vec::new();
    for _ in 0..30 {
        flat.extend_from_slice(&[1.5, -2.0, 0.5, 3.0]);
    }
    for _ in 0..20 {
        flat.extend_from_slice(&[-1.0, 4.0, -2.5, 0.0]);
    }
    flat.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
    let n = flat.len() / dims;
    let tree = DynamicKdTree::from_flat(&flat, n, dims).unwrap();
    let mut scratch = KdQueryScratch::with_capacity(10);
    let mut out = Vec::new();

    for i in [0usize, 15, 29, 30, 49, 50] {
        tree.k_nearest_excluding_self_with_scratch(i, 12, &mut scratch, &mut out)
            .unwrap();
        assert_eq!(
            out,
            brute_knn_nd(&flat, dims, &flat[i * dims..(i + 1) * dims], 12, Some(i))
        );
        assert_eq!(
            tree.nearest_excluding_self_with_scratch(i, &mut scratch)
                .unwrap(),
            brute_knn_nd(&flat, dims, &flat[i * dims..(i + 1) * dims], 1, Some(i))
                .into_iter()
                .next()
        );
    }
    tree.within_radius_sq_into(&[1.5, -2.0, 0.5, 3.0], 0.0, &mut out)
        .unwrap();
    assert_eq!(
        out,
        brute_within_nd(&flat, dims, &[1.5, -2.0, 0.5, 3.0], 0.0)
    );

    // 2. Empty (n = 0) and single-point (n = 1).
    let empty = DynamicKdTree::<f64>::from_flat(&[], 0, 5).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.dims(), 5);
    assert_eq!(empty.k_nearest(&[0.0; 5], 3).unwrap(), Vec::new());
    empty
        .within_radius_sq_into(&[0.0; 5], 1.0, &mut out)
        .unwrap();
    assert!(out.is_empty());

    let single = DynamicKdTree::<f64>::from_flat(&[2.0, 3.0, 4.0], 1, 3).unwrap();
    assert_eq!(
        single
            .nearest_excluding_self_with_scratch(0, &mut scratch)
            .unwrap(),
        None
    );
    single
        .k_nearest_excluding_self_with_scratch(0, 5, &mut scratch, &mut out)
        .unwrap();
    assert!(out.is_empty());

    // 3. Invalid construction and queries.
    assert!(matches!(
        DynamicKdTree::<f64>::from_flat(&[], 0, 0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        DynamicKdTree::<f64>::from_flat(&[1.0, 2.0, 3.0], 2, 2),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        DynamicKdTree::<f64>::from_flat(&[1.0, f64::NAN, 3.0], 1, 3),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        single.point(1),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        single.nearest_excluding_self_with_scratch(1, &mut scratch),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        single.k_nearest_excluding_self_with_scratch(1, 1, &mut scratch, &mut out),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        single.k_nearest(&[0.0, 0.0], 1),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        single.k_nearest(&[0.0, f64::INFINITY, 0.0], 1),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        single.within_radius_sq_into(&[0.0, 0.0], 1.0, &mut out),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        single.within_radius_sq_into(&[0.0, 0.0, 0.0], -1.0, &mut out),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        single.within_radius_sq_into(&[0.0, 0.0, 0.0], f64::NAN, &mut out),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
