use egraph_core::{CoreError, DeterministicRng, Layout2d, Layout3d, LayoutNd};

use super::*;

#[test]
fn test_octree_layout3d_and_euclidean_storage_parity() {
    let mut rng = DeterministicRng::from_seed_u64(9001);
    let n = 32usize;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut zs = Vec::with_capacity(n);
    let mut flat = Vec::with_capacity(n * 3);
    let mut masses = Vec::with_capacity(n);
    for i in 0..n {
        let x = rng.next_f64() * 8.0 - 4.0;
        let y = rng.next_f64() * 8.0 - 4.0;
        let z = rng.next_f64() * 8.0 - 4.0;
        xs.push(x);
        ys.push(y);
        zs.push(z);
        flat.extend_from_slice(&[x, y, z]);
        masses.push(0.5 + (i % 5) as f64);
    }

    let layout3d = Layout3d::from_xyz(xs.clone(), ys.clone(), zs.clone()).unwrap();
    let layout_nd = LayoutNd::from_flat(n, 3, &flat).unwrap();

    let mut tree_xyz = ArenaOctree3d::new();
    tree_xyz.build_xyz(&xs, &ys, &zs).unwrap();

    let mut tree_3d = ArenaOctree3d::new();
    tree_3d.build_3d(&layout3d).unwrap();

    let mut tree_euc_3d = ArenaOctree3d::new();
    tree_euc_3d.build_euclidean(&layout3d).unwrap();

    let mut tree_euc_nd = ArenaOctree3d::new();
    tree_euc_nd.build_euclidean(&layout_nd).unwrap();

    let mut ref_kl = vec![0.0_f64; n * 3];
    let z_ref = tree_xyz
        .compute_kl_repulsion_flat(0.5, &mut ref_kl)
        .unwrap();
    for t in [&tree_3d, &tree_euc_3d, &tree_euc_nd] {
        let mut out = vec![0.0_f64; n * 3];
        let z = t.compute_kl_repulsion_flat(0.5, &mut out).unwrap();
        assert_eq!(z, z_ref);
        assert_eq!(out, ref_kl);
    }

    // Weighted parity between build_weighted_xyz, build_weighted_3d, and build_weighted_euclidean.
    let mut w_xyz = ArenaOctree3d::new();
    w_xyz.build_weighted_xyz(&xs, &ys, &zs, &masses).unwrap();
    let mut w_3d = ArenaOctree3d::new();
    w_3d.build_weighted_3d(&layout3d, &masses).unwrap();
    let mut w_euc_3d = ArenaOctree3d::new();
    w_euc_3d
        .build_weighted_euclidean(&layout3d, &masses)
        .unwrap();
    let mut w_euc_nd = ArenaOctree3d::new();
    w_euc_nd
        .build_weighted_euclidean(&layout_nd, &masses)
        .unwrap();

    let mut ref_c = vec![0.0_f64; n * 3];
    w_xyz
        .compute_coulomb_repulsion_flat(0.5, 0.05, &mut ref_c)
        .unwrap();
    for t in [&w_3d, &w_euc_3d, &w_euc_nd] {
        let mut out = vec![0.0_f64; n * 3];
        t.compute_coulomb_repulsion_flat(0.5, 0.05, &mut out)
            .unwrap();
        assert_eq!(out, ref_c);
    }

    // f32 Layout3d support.
    let layout3d_f32 = Layout3d::from_xyz(
        xs.iter().map(|&v| v as f32).collect(),
        ys.iter().map(|&v| v as f32).collect(),
        zs.iter().map(|&v| v as f32).collect(),
    )
    .unwrap();
    let mut tree_f32 = ArenaOctree3d::<f32>::new();
    tree_f32.build_euclidean(&layout3d_f32).unwrap();
    let mut out_f32 = vec![0.0_f32; n * 3];
    tree_f32
        .compute_entropy_repulsion_flat(0.5, 0.05, &mut out_f32)
        .unwrap();
    assert!(out_f32.iter().all(|v| v.is_finite()));
}

#[test]
fn test_octree_euclidean_rejects_non_3d_without_corrupting_state() {
    let mut tree = ArenaOctree3d::new();
    let valid = Layout3d::from_xyz(vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
    tree.build_3d(&valid).unwrap();
    assert_eq!(tree.point_count(), 2);

    let l2 = Layout2d::from_xy(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        tree.build_euclidean(&l2),
        Err(CoreError::DimensionMismatch {
            expected: 3,
            actual: 2,
            ..
        })
    ));
    let l_nd2 = LayoutNd::zeros(2, 2).unwrap();
    assert!(matches!(
        tree.build_euclidean(&l_nd2),
        Err(CoreError::DimensionMismatch {
            expected: 3,
            actual: 2,
            ..
        })
    ));
    let l_nd4 = LayoutNd::zeros(2, 4).unwrap();
    assert!(matches!(
        tree.build_weighted_euclidean(&l_nd4, &[1.0, 1.0]),
        Err(CoreError::DimensionMismatch {
            expected: 3,
            actual: 4,
            ..
        })
    ));
    // Prior tree state remains intact after validation errors.
    assert_eq!(tree.point_count(), 2);
}
