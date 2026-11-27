use egraph_core::{CoreError, DeterministicRng, Layout2d, Layout3d, LayoutNd};

use super::*;

#[test]
fn test_quadtree_euclidean_storage_and_flat_outputs_parity() {
    let mut rng = DeterministicRng::from_seed_u64(7070);
    let n = 28usize;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut flat = Vec::with_capacity(n * 2);
    let mut masses = Vec::with_capacity(n);
    for i in 0..n {
        let x = rng.next_f64() * 8.0 - 4.0;
        let y = rng.next_f64() * 8.0 - 4.0;
        xs.push(x);
        ys.push(y);
        flat.extend_from_slice(&[x, y]);
        masses.push(0.5 + (i % 4) as f64);
    }

    let l2 = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
    let l_nd2 = LayoutNd::from_flat(n, 2, &flat).unwrap();

    let mut t_ref = ArenaQuadtree2d::new();
    t_ref.build(&l2).unwrap();
    let mut t_euc_2d = ArenaQuadtree2d::new();
    t_euc_2d.build_euclidean(&l2).unwrap();
    let mut t_euc_nd = ArenaQuadtree2d::new();
    t_euc_nd.build_euclidean(&l_nd2).unwrap();

    let (mut kx, mut ky) = (vec![0.0; n], vec![0.0; n]);
    let z_ref = t_ref.compute_kl_repulsion(0.5, &mut kx, &mut ky).unwrap();
    let mut k_flat = vec![0.0; n * 2];
    let z_euc = t_euc_2d
        .compute_kl_repulsion_flat(0.5, &mut k_flat)
        .unwrap();
    assert_eq!(z_euc, z_ref);
    for i in 0..n {
        assert_eq!(k_flat[i * 2], kx[i]);
        assert_eq!(k_flat[i * 2 + 1], ky[i]);
    }
    let mut k_flat_nd = vec![0.0; n * 2];
    let z_nd = t_euc_nd
        .compute_kl_repulsion_flat(0.5, &mut k_flat_nd)
        .unwrap();
    assert_eq!(z_nd, z_ref);
    assert_eq!(k_flat_nd, k_flat);

    // Entropy flat parity.
    let (mut ex, mut ey) = (vec![0.0; n], vec![0.0; n]);
    t_ref
        .compute_entropy_repulsion(0.5, 0.05, &mut ex, &mut ey)
        .unwrap();
    let mut e_flat = vec![0.0; n * 2];
    t_euc_nd
        .compute_entropy_repulsion_flat(0.5, 0.05, &mut e_flat)
        .unwrap();
    for i in 0..n {
        assert_eq!(e_flat[i * 2], ex[i]);
        assert_eq!(e_flat[i * 2 + 1], ey[i]);
    }

    // Weighted Coulomb & MaxEnt flat parity.
    let mut w_ref = ArenaQuadtree2d::new();
    w_ref.build_weighted(&l2, &masses).unwrap();
    let mut w_euc_2d = ArenaQuadtree2d::new();
    w_euc_2d.build_weighted_euclidean(&l2, &masses).unwrap();
    let mut w_euc_nd = ArenaQuadtree2d::new();
    w_euc_nd.build_weighted_euclidean(&l_nd2, &masses).unwrap();

    let (mut cx, mut cy) = (vec![0.0; n], vec![0.0; n]);
    w_ref
        .compute_coulomb_repulsion(0.5, 0.04, &mut cx, &mut cy)
        .unwrap();
    let mut c_flat = vec![0.0; n * 2];
    w_euc_nd
        .compute_coulomb_repulsion_flat(0.5, 0.04, &mut c_flat)
        .unwrap();
    for i in 0..n {
        assert_eq!(c_flat[i * 2], cx[i]);
        assert_eq!(c_flat[i * 2 + 1], cy[i]);
    }

    let (mut mx, mut my) = (vec![0.0; n], vec![0.0; n]);
    w_ref
        .compute_maxent_repulsion(0.5, 0.8, 0.04, &mut mx, &mut my)
        .unwrap();
    let mut m_flat = vec![0.0; n * 2];
    w_euc_2d
        .compute_maxent_repulsion_flat(0.5, 0.8, 0.04, &mut m_flat)
        .unwrap();
    for i in 0..n {
        assert_eq!(m_flat[i * 2], mx[i]);
        assert_eq!(m_flat[i * 2 + 1], my[i]);
    }
}

#[test]
fn test_quadtree_euclidean_rejects_non_2d_without_corrupting_state() {
    let mut tree = ArenaQuadtree2d::new();
    let valid = Layout2d::from_xy(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
    tree.build_euclidean(&valid).unwrap();
    assert_eq!(tree.point_count(), 2);

    let l3 = Layout3d::from_xyz(vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        tree.build_euclidean(&l3),
        Err(CoreError::DimensionMismatch {
            expected: 2,
            actual: 3,
            ..
        })
    ));
    let l_nd1 = LayoutNd::zeros(2, 1).unwrap();
    assert!(matches!(
        tree.build_euclidean(&l_nd1),
        Err(CoreError::DimensionMismatch {
            expected: 2,
            actual: 1,
            ..
        })
    ));
    let l_nd3 = LayoutNd::zeros(2, 3).unwrap();
    assert!(matches!(
        tree.build_weighted_euclidean(&l_nd3, &[1.0, 1.0]),
        Err(CoreError::DimensionMismatch {
            expected: 2,
            actual: 3,
            ..
        })
    ));
    assert_eq!(tree.point_count(), 2);
}
