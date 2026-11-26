use egraph_core::{CoreError, Layout2d};

use super::*;

#[test]
fn test_arena_quadtree_exact_when_theta_zero_and_reusable_clear() {
    let layout = Layout2d::from_xy(
        vec![0.0_f64, 0.0, 1.5, -2.1, 0.75, -0.5],
        vec![0.0_f64, 0.0, -0.8, 1.4, 2.2, -1.7],
    )
    .unwrap();
    let n = layout.len();

    let mut tree = ArenaQuadtree2d::new();
    tree.build(&layout).unwrap();
    assert_eq!(tree.point_count(), n);

    // Exact Barnes-Hut (`theta = 0.0`) must match brute-force O(N^2) even with duplicates.
    let mut bh_kl_x = vec![0.0; n];
    let mut bh_kl_y = vec![0.0; n];
    let bh_z = tree
        .compute_kl_repulsion(0.0, &mut bh_kl_x, &mut bh_kl_y)
        .unwrap();

    let mut bh_ent_x = vec![0.0; n];
    let mut bh_ent_y = vec![0.0; n];
    tree.compute_entropy_repulsion(0.0, 0.05, &mut bh_ent_x, &mut bh_ent_y)
        .unwrap();

    let mut ex_z = 0.0;
    let mut ex_kl_x = vec![0.0; n];
    let mut ex_kl_y = vec![0.0; n];
    let mut ex_ent_x = vec![0.0; n];
    let mut ex_ent_y = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = layout.x()[i] - layout.x()[j];
            let dy = layout.y()[i] - layout.y()[j];
            let d_sq = dx * dx + dy * dy;
            let w = 1.0 / (1.0 + d_sq);
            ex_z += w;
            ex_kl_x[i] += w * w * dx;
            ex_kl_y[i] += w * w * dy;

            let d = d_sq.sqrt();
            if d > ENTROPY_MIN_DIST {
                let mult = 1.0 / (d * (d + 0.05));
                ex_ent_x[i] += mult * dx;
                ex_ent_y[i] += mult * dy;
            }
        }
    }
    assert!((bh_z - ex_z).abs() < 1e-12);
    for i in 0..n {
        assert!((bh_kl_x[i] - ex_kl_x[i] / ex_z).abs() < 1e-12);
        assert!((bh_kl_y[i] - ex_kl_y[i] / ex_z).abs() < 1e-12);
        assert!((bh_ent_x[i] - ex_ent_x[i]).abs() < 1e-12);
        assert!((bh_ent_y[i] - ex_ent_y[i]).abs() < 1e-12);
    }

    // Approximation with theta = 0.5 stays close to exact.
    tree.compute_kl_repulsion(0.5, &mut bh_kl_x, &mut bh_kl_y)
        .unwrap();
    for i in 0..n {
        assert!((bh_kl_x[i] - ex_kl_x[i] / ex_z).abs() < 0.02);
        assert!((bh_kl_y[i] - ex_kl_y[i] / ex_z).abs() < 0.02);
    }

    // Reuse capacity across clear/build cycles.
    let cap = tree.nodes.capacity();
    let layout2 = Layout2d::from_xy(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]).unwrap();
    tree.build(&layout2).unwrap();
    assert_eq!(tree.point_count(), 3);
    assert_eq!(tree.nodes.capacity(), cap);
}

#[test]
fn test_arena_quadtree_sub_ulp_f32_and_empty() {
    let mut tree = ArenaQuadtree2d::<f32>::new();
    let empty = Layout2d::<f32>::zeros(0);
    tree.build(&empty).unwrap();
    assert_eq!(tree.point_count(), 0);
    assert_eq!(tree.node_count(), 0);
    assert_eq!(
        tree.compute_kl_repulsion(0.5, &mut [], &mut []).unwrap(),
        1.0
    );

    let sub_ulp = Layout2d::from_xy(
        vec![100.0_f32, 100.00001, 0.0],
        vec![-100.0_f32, -100.0, 0.0],
    )
    .unwrap();
    tree.build(&sub_ulp).unwrap();
    assert_eq!(tree.point_count(), 3);
    let mut fx = vec![0.0_f32; 3];
    let mut fy = vec![0.0_f32; 3];
    let z = tree.compute_kl_repulsion(0.5, &mut fx, &mut fy).unwrap();
    assert!(z.is_finite() && z > 0.0);
    assert!(fx.iter().all(|v| v.is_finite()));
    assert!(fy.iter().all(|v| v.is_finite()));

    let bad = Layout2d::from_xy(vec![f32::INFINITY], vec![0.0]).unwrap();
    assert!(matches!(
        tree.build(&bad),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_arena_quadtree_weighted_coulomb_exact_and_approx_f64() {
    let layout = Layout2d::from_xy(
        vec![0.0_f64, 0.0, 2.5, -3.1, 1.25, -0.75, 4.0],
        vec![0.0_f64, 0.0, -1.2, 2.4, 3.2, -2.7, 0.5],
    )
    .unwrap();
    let masses = vec![1.5, 2.5, 0.0, 4.0, 0.75, 3.25, 2.0];
    let eps_sq = 0.04_f64;
    let n = layout.len();

    let mut tree = ArenaQuadtree2d::new();
    tree.build_weighted(&layout, &masses).unwrap();
    assert_eq!(tree.point_count(), n);

    let mut bh_x = vec![0.0_f64; n];
    let mut bh_y = vec![0.0_f64; n];
    tree.compute_coulomb_repulsion(0.0, eps_sq, &mut bh_x, &mut bh_y)
        .unwrap();

    let mut ex_x = vec![0.0_f64; n];
    let mut ex_y = vec![0.0_f64; n];
    for i in 0..n {
        for (j, &mj) in masses.iter().enumerate() {
            if i == j {
                continue;
            }
            let dx = layout.x()[i] - layout.x()[j];
            let dy = layout.y()[i] - layout.y()[j];
            let mult = mj / (dx * dx + dy * dy + eps_sq);
            ex_x[i] += mult * dx;
            ex_y[i] += mult * dy;
        }
    }

    for i in 0..n {
        assert!(
            (bh_x[i] - ex_x[i]).abs() < 1e-10,
            "x mismatch at {i}: bh={} exact={}",
            bh_x[i],
            ex_x[i]
        );
        assert!(
            (bh_y[i] - ex_y[i]).abs() < 1e-10,
            "y mismatch at {i}: bh={} exact={}",
            bh_y[i],
            ex_y[i]
        );
    }

    // Approximation with theta = 0.5 remains accurate.
    let mut approx_x = vec![0.0_f64; n];
    let mut approx_y = vec![0.0_f64; n];
    tree.compute_coulomb_repulsion(0.5, eps_sq, &mut approx_x, &mut approx_y)
        .unwrap();
    for i in 0..n {
        assert!((approx_x[i] - ex_x[i]).abs() < 0.25);
        assert!((approx_y[i] - ex_y[i]).abs() < 0.25);
    }

    // Unweighted KL/entropy behavior remains identical when masses are unit.
    let unit_masses = vec![1.0_f64; n];
    let mut unweighted = ArenaQuadtree2d::new();
    unweighted.build(&layout).unwrap();
    let mut weighted_unit = ArenaQuadtree2d::new();
    weighted_unit.build_weighted(&layout, &unit_masses).unwrap();

    let (mut kl_ux, mut kl_uy) = (vec![0.0; n], vec![0.0; n]);
    let (mut kl_wx, mut kl_wy) = (vec![0.0; n], vec![0.0; n]);
    let zu = unweighted
        .compute_kl_repulsion(0.5, &mut kl_ux, &mut kl_uy)
        .unwrap();
    let zw = weighted_unit
        .compute_kl_repulsion(0.5, &mut kl_wx, &mut kl_wy)
        .unwrap();
    assert_eq!(zu, zw);
    assert_eq!(kl_ux, kl_wx);
    assert_eq!(kl_uy, kl_wy);
}

#[test]
fn test_arena_quadtree_weighted_coulomb_f32() {
    let layout = Layout2d::from_xy(
        vec![0.0_f32, 1.5, -2.0, 3.25, -1.5, 0.5],
        vec![0.0_f32, -1.0, 2.5, 0.75, -2.25, 1.8],
    )
    .unwrap();
    let masses = vec![1.0_f64, 2.5, 0.5, 4.0, 1.25, 3.0];
    let eps_sq = 0.05_f32;
    let n = layout.len();

    let mut tree = ArenaQuadtree2d::<f32>::new();
    tree.build_weighted(&layout, &masses).unwrap();

    let mut bh_x = vec![0.0_f32; n];
    let mut bh_y = vec![0.0_f32; n];
    tree.compute_coulomb_repulsion(0.0, eps_sq, &mut bh_x, &mut bh_y)
        .unwrap();

    for i in 0..n {
        let mut ex_x = 0.0_f64;
        let mut ex_y = 0.0_f64;
        for (j, &mj) in masses.iter().enumerate() {
            if i == j {
                continue;
            }
            let dx = f64::from(layout.x()[i]) - f64::from(layout.x()[j]);
            let dy = f64::from(layout.y()[i]) - f64::from(layout.y()[j]);
            let mult = mj / (dx * dx + dy * dy + f64::from(eps_sq));
            ex_x += mult * dx;
            ex_y += mult * dy;
        }
        assert!((f64::from(bh_x[i]) - ex_x).abs() < 1e-5);
        assert!((f64::from(bh_y[i]) - ex_y).abs() < 1e-5);
    }

    tree.compute_coulomb_repulsion(0.5, eps_sq, &mut bh_x, &mut bh_y)
        .unwrap();
    assert!(bh_x.iter().all(|v| v.is_finite()));
    assert!(bh_y.iter().all(|v| v.is_finite()));
}

#[test]
fn test_arena_quadtree_weighted_coulomb_validation_and_edge_cases() {
    let mut tree = ArenaQuadtree2d::<f64>::new();

    // Empty and single-point cases.
    tree.build_weighted_xy(&[], &[], &[]).unwrap();
    tree.compute_coulomb_repulsion(0.5, 1e-4, &mut [], &mut [])
        .unwrap();

    tree.build_weighted_xy(&[1.0], &[2.0], &[5.0]).unwrap();
    let mut ox = vec![9.0];
    let mut oy = vec![9.0];
    tree.compute_coulomb_repulsion(0.5, 1e-4, &mut ox, &mut oy)
        .unwrap();
    assert_eq!(ox, vec![0.0]);
    assert_eq!(oy, vec![0.0]);

    // Length mismatches.
    assert!(matches!(
        tree.build_weighted_xy(&[1.0, 2.0], &[1.0], &[1.0, 1.0]),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        tree.build_weighted_xy(&[1.0, 2.0], &[1.0, 2.0], &[1.0]),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Invalid masses (NaN, Inf, negative).
    assert!(matches!(
        tree.build_weighted_xy(&[1.0], &[2.0], &[f64::NAN]),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        tree.build_weighted_xy(&[1.0], &[2.0], &[f64::INFINITY]),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        tree.build_weighted_xy(&[1.0], &[2.0], &[-0.1]),
        Err(CoreError::InvalidInput(_))
    ));

    // compute_coulomb_repulsion input guards.
    tree.build_weighted_xy(&[0.0, 1.0], &[0.0, 1.0], &[1.0, 2.0])
        .unwrap();
    let mut out_x = vec![0.0; 2];
    let mut out_y = vec![0.0; 2];
    assert!(matches!(
        tree.compute_coulomb_repulsion(0.5, 1e-4, &mut out_x[..1], &mut out_y),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        tree.compute_coulomb_repulsion(0.5, 1e-4, &mut out_x, &mut out_y[..1]),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        tree.compute_coulomb_repulsion(-0.1, 1e-4, &mut out_x, &mut out_y),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        tree.compute_coulomb_repulsion(f64::NAN, 1e-4, &mut out_x, &mut out_y),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        tree.compute_coulomb_repulsion(0.5, 0.0, &mut out_x, &mut out_y),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        tree.compute_coulomb_repulsion(0.5, -1.0, &mut out_x, &mut out_y),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        tree.compute_coulomb_repulsion(0.5, f64::INFINITY, &mut out_x, &mut out_y),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_arena_quadtree_maxent_repulsion_oracle_and_validation() {
    let layout = Layout2d::from_xy(
        vec![0.0_f64, 0.0, 2.5, -3.1, 1.25, -0.75, 4.0],
        vec![0.0_f64, 0.0, -1.2, 2.4, 3.2, -2.7, 0.5],
    )
    .unwrap();
    let masses = vec![1.5, 2.5, 0.5, 4.0, 0.75, 3.25, 2.0];
    let eps_sq = 0.04_f64;
    let n = layout.len();

    let mut tree = ArenaQuadtree2d::new();
    tree.build_weighted(&layout, &masses).unwrap();

    for &q in &[0.0_f64, 0.8, -0.5] {
        let mut bh_x = vec![0.0_f64; n];
        let mut bh_y = vec![0.0_f64; n];
        tree.compute_maxent_repulsion(0.0, q, eps_sq, &mut bh_x, &mut bh_y)
            .unwrap();

        let exp = 0.5 * (q + 2.0);
        let mut ex_x = vec![0.0_f64; n];
        let mut ex_y = vec![0.0_f64; n];
        for i in 0..n {
            for (j, &mj) in masses.iter().enumerate() {
                if i == j {
                    continue;
                }
                let dx = layout.x()[i] - layout.x()[j];
                let dy = layout.y()[i] - layout.y()[j];
                let denom = (dx * dx + dy * dy + eps_sq).powf(exp);
                let mult = mj / denom;
                ex_x[i] += mult * dx;
                ex_y[i] += mult * dy;
            }
            assert!(
                (bh_x[i] - ex_x[i]).abs() < 1e-10,
                "q={q} x mismatch at {i}: bh={} exact={}",
                bh_x[i],
                ex_x[i]
            );
            assert!(
                (bh_y[i] - ex_y[i]).abs() < 1e-10,
                "q={q} y mismatch at {i}: bh={} exact={}",
                bh_y[i],
                ex_y[i]
            );
        }

        // Barnes-Hut approximation at theta = 1.2 remains finite and tracks exact sign/magnitude.
        let mut approx_x = vec![0.0_f64; n];
        let mut approx_y = vec![0.0_f64; n];
        tree.compute_maxent_repulsion(1.2, q, eps_sq, &mut approx_x, &mut approx_y)
            .unwrap();
        for i in 0..n {
            assert!(approx_x[i].is_finite() && approx_y[i].is_finite());
            assert!((approx_x[i] - ex_x[i]).abs() < 1.5);
            assert!((approx_y[i] - ex_y[i]).abs() < 1.5);
        }
    }

    // Validation guards.
    let mut out_x = vec![0.0; n];
    let mut out_y = vec![0.0; n];
    assert!(matches!(
        tree.compute_maxent_repulsion(1.2, -2.0, eps_sq, &mut out_x, &mut out_y),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        tree.compute_maxent_repulsion(1.2, -2.5, eps_sq, &mut out_x, &mut out_y),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        tree.compute_maxent_repulsion(1.2, f64::NAN, eps_sq, &mut out_x, &mut out_y),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        tree.compute_maxent_repulsion(1.2, f64::INFINITY, eps_sq, &mut out_x, &mut out_y),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        tree.eval_point_maxent_raw(n, 0.0, 0.8, eps_sq),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        tree.compute_maxent_repulsion(1.2, 0.8, eps_sq, &mut out_x[..1], &mut out_y),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
