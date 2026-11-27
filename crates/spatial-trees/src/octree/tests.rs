use egraph_core::{CoreError, DeterministicRng, LayoutNd};

use super::*;

#[allow(clippy::type_complexity)]
fn brute_coulomb_maxent(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    masses: &[f64],
    exp: f64,
    eps_sq: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = xs.len();
    let mut fx = vec![0.0; n];
    let mut fy = vec![0.0; n];
    let mut fz = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = xs[i] - xs[j];
            let dy = ys[i] - ys[j];
            let dz = zs[i] - zs[j];
            let denom = (dx * dx + dy * dy + dz * dz + eps_sq).powf(exp);
            let mult = masses[j] / denom;
            fx[i] += mult * dx;
            fy[i] += mult * dy;
            fz[i] += mult * dz;
        }
    }
    (fx, fy, fz)
}

#[test]
fn test_octree_kl_entropy_exact_theta_zero_and_approx_f64() {
    let xs = [0.0_f64, 0.0, 1.5, -2.1, 0.75, -0.5, 2.2];
    let ys = [0.0_f64, 0.0, -0.8, 1.4, 2.2, -1.7, 0.4];
    let zs = [0.0_f64, 0.0, 1.1, -1.9, 0.3, 2.6, -0.9];
    let n = xs.len();
    let mut flat = Vec::with_capacity(n * 3);
    for i in 0..n {
        flat.push(xs[i]);
        flat.push(ys[i]);
        flat.push(zs[i]);
    }
    let layout = LayoutNd::from_flat(n, 3, &flat).unwrap();

    let mut tree = ArenaOctree3d::new();
    tree.build(&layout).unwrap();
    assert_eq!(tree.point_count(), n);
    assert!(tree.node_count() >= 1);

    // Exact (theta = 0.0) KL and entropy against O(N^2) brute-force oracle.
    let (mut kl_x, mut kl_y, mut kl_z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let kl_z_sum = tree
        .compute_kl_repulsion(0.0, &mut kl_x, &mut kl_y, &mut kl_z)
        .unwrap();
    let mut kl_flat = vec![0.0; n * 3];
    let kl_z_flat = tree.compute_kl_repulsion_flat(0.0, &mut kl_flat).unwrap();
    assert_eq!(kl_z_sum, kl_z_flat);

    let (mut ent_x, mut ent_y, mut ent_z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    tree.compute_entropy_repulsion(0.0, 0.05, &mut ent_x, &mut ent_y, &mut ent_z)
        .unwrap();
    let mut ent_flat = vec![0.0; n * 3];
    tree.compute_entropy_repulsion_flat(0.0, 0.05, &mut ent_flat)
        .unwrap();

    let mut ex_z = 0.0;
    let (mut ex_kl_x, mut ex_kl_y, mut ex_kl_z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let (mut ex_ent_x, mut ex_ent_y, mut ex_ent_z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let mut zi_exact = 0.0;
        let (mut rkx, mut rky, mut rkz) = (0.0, 0.0, 0.0);
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = xs[i] - xs[j];
            let dy = ys[i] - ys[j];
            let dz = zs[i] - zs[j];
            let d_sq = dx * dx + dy * dy + dz * dz;
            let w = 1.0 / (1.0 + d_sq);
            zi_exact += w;
            rkx += w * w * dx;
            rky += w * w * dy;
            rkz += w * w * dz;

            let d = d_sq.sqrt();
            if d > ENTROPY_MIN_DIST {
                let mult = 1.0 / (d * (d + 0.05));
                ex_ent_x[i] += mult * dx;
                ex_ent_y[i] += mult * dy;
                ex_ent_z[i] += mult * dz;
            }
        }
        let (raw_fx, raw_fy, raw_fz, raw_z) = tree.eval_point_kl_raw(i, 0.0).unwrap();
        assert!((raw_fx - rkx).abs() < 1e-12);
        assert!((raw_fy - rky).abs() < 1e-12);
        assert!((raw_fz - rkz).abs() < 1e-12);
        assert!((raw_z - zi_exact).abs() < 1e-12);

        ex_z += zi_exact;
        ex_kl_x[i] = rkx;
        ex_kl_y[i] = rky;
        ex_kl_z[i] = rkz;
    }

    assert!((kl_z_sum - ex_z).abs() < 1e-12);
    for i in 0..n {
        assert!((kl_x[i] - ex_kl_x[i] / ex_z).abs() < 1e-12);
        assert!((kl_y[i] - ex_kl_y[i] / ex_z).abs() < 1e-12);
        assert!((kl_z[i] - ex_kl_z[i] / ex_z).abs() < 1e-12);
        assert_eq!(kl_flat[i * 3], kl_x[i]);
        assert_eq!(kl_flat[i * 3 + 1], kl_y[i]);
        assert_eq!(kl_flat[i * 3 + 2], kl_z[i]);
        assert!((ent_x[i] - ex_ent_x[i]).abs() < 1e-12);
        assert!((ent_y[i] - ex_ent_y[i]).abs() < 1e-12);
        assert!((ent_z[i] - ex_ent_z[i]).abs() < 1e-12);
        assert_eq!(ent_flat[i * 3], ent_x[i]);
        assert_eq!(ent_flat[i * 3 + 1], ent_y[i]);
        assert_eq!(ent_flat[i * 3 + 2], ent_z[i]);
    }

    // Barnes-Hut approximation at theta = 0.5 tracks exact closely.
    tree.compute_kl_repulsion_flat(0.5, &mut kl_flat).unwrap();
    tree.compute_entropy_repulsion_flat(0.5, 0.05, &mut ent_flat)
        .unwrap();
    for i in 0..n {
        assert!((kl_flat[i * 3] - ex_kl_x[i] / ex_z).abs() < 0.03);
        assert!((kl_flat[i * 3 + 1] - ex_kl_y[i] / ex_z).abs() < 0.03);
        assert!((kl_flat[i * 3 + 2] - ex_kl_z[i] / ex_z).abs() < 0.03);
        assert!((ent_flat[i * 3] - ex_ent_x[i]).abs() < 0.1);
    }

    // Reusable buffer capacity across clear/build cycles.
    let cap = tree.nodes.capacity();
    tree.build_xyz(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0])
        .unwrap();
    assert_eq!(tree.point_count(), 3);
    assert_eq!(tree.nodes.capacity(), cap);
}

#[test]
fn test_octree_coulomb_maxent_power_f64_and_f32() {
    let mut rng = DeterministicRng::from_seed_u64(2026);
    let n = 40;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut zs = Vec::with_capacity(n);
    let mut masses = Vec::with_capacity(n);
    for i in 0..n {
        xs.push(rng.next_f64() * 10.0 - 5.0);
        ys.push(rng.next_f64() * 10.0 - 5.0);
        zs.push(rng.next_f64() * 10.0 - 5.0);
        masses.push(if i % 7 == 0 {
            0.0
        } else {
            0.5 + 3.0 * rng.next_f64()
        });
    }
    let eps_sq = 0.04_f64;

    let mut tree = ArenaOctree3d::new();
    tree.build_weighted_xyz(&xs, &ys, &zs, &masses).unwrap();

    // Coulomb exact & approx.
    let (ex_cx, ex_cy, ex_cz) = brute_coulomb_maxent(&xs, &ys, &zs, &masses, 1.0, eps_sq);
    let (mut cx, mut cy, mut cz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    tree.compute_coulomb_repulsion(0.0, eps_sq, &mut cx, &mut cy, &mut cz)
        .unwrap();
    let mut c_flat = vec![0.0; n * 3];
    tree.compute_coulomb_repulsion_flat(0.0, eps_sq, &mut c_flat)
        .unwrap();
    for i in 0..n {
        assert!((cx[i] - ex_cx[i]).abs() < 1e-10);
        assert!((cy[i] - ex_cy[i]).abs() < 1e-10);
        assert!((cz[i] - ex_cz[i]).abs() < 1e-10);
        assert_eq!(c_flat[i * 3], cx[i]);
        assert_eq!(c_flat[i * 3 + 1], cy[i]);
        assert_eq!(c_flat[i * 3 + 2], cz[i]);
    }
    tree.compute_coulomb_repulsion_flat(0.5, eps_sq, &mut c_flat)
        .unwrap();
    for i in 0..n {
        assert!((c_flat[i * 3] - ex_cx[i]).abs() < 0.35);
        assert!((c_flat[i * 3 + 1] - ex_cy[i]).abs() < 0.35);
        assert!((c_flat[i * 3 + 2] - ex_cz[i]).abs() < 0.35);
    }

    // MaxEnt across q in {0.0, 0.8, -0.5}.
    for &q in &[0.0_f64, 0.8, -0.5] {
        let (ex_mx, ex_my, ex_mz) =
            brute_coulomb_maxent(&xs, &ys, &zs, &masses, 0.5 * (q + 2.0), eps_sq);
        let (mut mx, mut my, mut mz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        tree.compute_maxent_repulsion(0.0, q, eps_sq, &mut mx, &mut my, &mut mz)
            .unwrap();
        let mut m_flat = vec![0.0; n * 3];
        tree.compute_maxent_repulsion_flat(0.0, q, eps_sq, &mut m_flat)
            .unwrap();
        for i in 0..n {
            assert!((mx[i] - ex_mx[i]).abs() < 1e-10);
            assert!((my[i] - ex_my[i]).abs() < 1e-10);
            assert!((mz[i] - ex_mz[i]).abs() < 1e-10);
            assert_eq!(m_flat[i * 3], mx[i]);
            assert_eq!(m_flat[i * 3 + 1], my[i]);
            assert_eq!(m_flat[i * 3 + 2], mz[i]);
        }
        tree.compute_maxent_repulsion_flat(0.5, q, eps_sq, &mut m_flat)
            .unwrap();
        for i in 0..n {
            assert!((m_flat[i * 3] - ex_mx[i]).abs() < 0.5);
        }
    }

    // Power-law repulsion matching LinLog convention: exp_r = -1.0 matches Coulomb; -2.0 and 0.0 match brute-force.
    for &exp_r in &[-1.0_f64, -2.0, -0.5, 0.5] {
        let mut p_flat = vec![0.0; n * 3];
        tree.compute_power_repulsion_flat(0.0, exp_r, eps_sq, &mut p_flat)
            .unwrap();
        let (ex_px, ex_py, ex_pz) =
            brute_coulomb_maxent(&xs, &ys, &zs, &masses, 0.5 * (1.0 - exp_r), eps_sq);
        for i in 0..n {
            assert!((p_flat[i * 3] - ex_px[i]).abs() < 1e-9);
            assert!((p_flat[i * 3 + 1] - ex_py[i]).abs() < 1e-9);
            assert!((p_flat[i * 3 + 2] - ex_pz[i]).abs() < 1e-9);
        }
        if (exp_r - (-1.0)).abs() <= 1e-12 {
            tree.compute_coulomb_repulsion_flat(0.0, eps_sq, &mut c_flat)
                .unwrap();
            assert_eq!(p_flat, c_flat);
        }
        tree.compute_power_repulsion_flat(0.5, exp_r, eps_sq, &mut p_flat)
            .unwrap();
        assert!(p_flat.iter().all(|v| v.is_finite()));
    }

    // f32 precision oracle with f64 accumulation.
    let mut flat32 = Vec::with_capacity(n * 3);
    for i in 0..n {
        flat32.push(xs[i] as f32);
        flat32.push(ys[i] as f32);
        flat32.push(zs[i] as f32);
    }
    let xs32_64: Vec<f64> = (0..n).map(|i| f64::from(flat32[i * 3])).collect();
    let ys32_64: Vec<f64> = (0..n).map(|i| f64::from(flat32[i * 3 + 1])).collect();
    let zs32_64: Vec<f64> = (0..n).map(|i| f64::from(flat32[i * 3 + 2])).collect();
    let layout32 = LayoutNd::from_flat(n, 3, &flat32).unwrap();
    let mut tree32 = ArenaOctree3d::<f32>::new();
    tree32.build_weighted(&layout32, &masses).unwrap();

    let mut f32_out = vec![0.0_f32; n * 3];
    tree32
        .compute_coulomb_repulsion_flat(0.0, eps_sq as f32, &mut f32_out)
        .unwrap();
    let (ex32_x, ex32_y, ex32_z) = brute_coulomb_maxent(
        &xs32_64,
        &ys32_64,
        &zs32_64,
        &masses,
        1.0,
        f64::from(eps_sq as f32),
    );
    for i in 0..n {
        assert!((f64::from(f32_out[i * 3]) - ex32_x[i]).abs() < 1e-4);
        assert!((f64::from(f32_out[i * 3 + 1]) - ex32_y[i]).abs() < 1e-4);
        assert!((f64::from(f32_out[i * 3 + 2]) - ex32_z[i]).abs() < 1e-4);
    }

    let mut kl32 = vec![0.0_f32; n * 3];
    let z32 = tree32.compute_kl_repulsion_flat(0.5, &mut kl32).unwrap();
    assert!(z32.is_finite() && z32 > 0.0);
    assert!(kl32.iter().all(|v| v.is_finite()));
}

#[test]
fn test_octree_degenerate_geometries_and_deep_stack() {
    // 1. 1D-collinear-in-3D (all points on x-axis with y=0, z=0).
    let xs_1d: Vec<f64> = (0..24).map(|i| i as f64 * 0.25).collect();
    let ys_1d = vec![0.0; 24];
    let zs_1d = vec![0.0; 24];
    let mut tree = ArenaOctree3d::new();
    tree.build_xyz(&xs_1d, &ys_1d, &zs_1d).unwrap();
    assert!(tree.nodes[0].half_span > 0.0);
    let mut out = vec![0.0; 24 * 3];
    tree.compute_coulomb_repulsion_flat(0.0, 1e-3, &mut out)
        .unwrap();
    let masses_1d = vec![1.0; 24];
    let (ex_x, ex_y, ex_z) = brute_coulomb_maxent(&xs_1d, &ys_1d, &zs_1d, &masses_1d, 1.0, 1e-3);
    for i in 0..24 {
        assert!((out[i * 3] - ex_x[i]).abs() < 1e-10);
        assert!((out[i * 3 + 1] - ex_y[i]).abs() < 1e-12);
        assert!((out[i * 3 + 2] - ex_z[i]).abs() < 1e-12);
    }

    // 2. 2D-planar-in-3D (z = 3.5 constant).
    let xs_2d: Vec<f64> = (0..25).map(|i| (i % 5) as f64).collect();
    let ys_2d: Vec<f64> = (0..25).map(|i| (i / 5) as f64).collect();
    let zs_2d = vec![3.5; 25];
    tree.build_xyz(&xs_2d, &ys_2d, &zs_2d).unwrap();
    let mut out_2d = vec![0.0; 25 * 3];
    tree.compute_kl_repulsion_flat(0.5, &mut out_2d).unwrap();
    assert!(out_2d.iter().all(|v| v.is_finite()));

    // 3. Deep clustered 3D points reaching depth >= 25 to exercise the 256-entry stack
    //    (7 * 32 + 1 = 225 > 128). Scale by 1024.0 so points at depth 28 remain > COINCIDENT_DIST_SQ.
    let mut deep_x = Vec::new();
    let mut deep_y = Vec::new();
    let mut deep_z = Vec::new();
    for k in 0..=30 {
        let s = 1024.0 * 2.0_f64.powi(-k);
        deep_x.push(s);
        deep_y.push(s);
        deep_z.push(s);
        // Populate sibling octants near the origin at every scale to keep DFS stack full.
        if k < 28 {
            deep_x.push(-s);
            deep_y.push(s);
            deep_z.push(-s);
            deep_x.push(s);
            deep_y.push(-s);
            deep_z.push(s);
        }
    }
    deep_x.push(0.0);
    deep_y.push(0.0);
    deep_z.push(0.0);

    tree.build_xyz(&deep_x, &deep_y, &deep_z).unwrap();
    let root_span = tree.nodes[0].half_span;
    let min_span = tree
        .nodes
        .iter()
        .map(|n| n.half_span)
        .fold(f64::INFINITY, f64::min);
    assert!(
        min_span <= root_span * 2.0_f64.powi(-25),
        "expected depth >= 25, root_span={root_span}, min_span={min_span}"
    );
    let m_deep = deep_x.len();
    let mut deep_out = vec![0.0; m_deep * 3];
    tree.compute_coulomb_repulsion_flat(0.0, 1e-6, &mut deep_out)
        .unwrap();
    tree.compute_coulomb_repulsion_flat(0.5, 1e-6, &mut deep_out)
        .unwrap();
    tree.compute_kl_repulsion_flat(0.5, &mut deep_out).unwrap();
    tree.compute_entropy_repulsion_flat(0.5, 1e-3, &mut deep_out)
        .unwrap();
    assert!(deep_out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_octree_empty_single_coincident_and_validation() {
    let mut tree = ArenaOctree3d::<f64>::new();

    // Empty (n = 0).
    let empty = LayoutNd::<f64>::zeros(0, 3).unwrap();
    tree.build(&empty).unwrap();
    assert_eq!(tree.point_count(), 0);
    assert_eq!(tree.node_count(), 0);
    tree.compute_coulomb_repulsion_flat(0.5, 1e-4, &mut [])
        .unwrap();
    assert_eq!(tree.compute_kl_repulsion_flat(0.5, &mut []).unwrap(), 1.0);
    tree.compute_entropy_repulsion_flat(0.5, 1e-2, &mut [])
        .unwrap();

    // Single point (n = 1).
    tree.build_flat(&[1.0, 2.0, 3.0], 1).unwrap();
    let mut single_out = vec![9.0; 3];
    tree.compute_coulomb_repulsion_flat(0.5, 1e-4, &mut single_out)
        .unwrap();
    assert_eq!(single_out, vec![0.0; 3]);
    assert_eq!(
        tree.compute_kl_repulsion_flat(0.5, &mut single_out)
            .unwrap(),
        1.0
    );
    assert_eq!(single_out, vec![0.0; 3]);

    // All-coincident points.
    tree.build_xyz(&[1.0, 1.0, 1.0], &[2.0, 2.0, 2.0], &[3.0, 3.0, 3.0])
        .unwrap();
    let mut coinc_out = vec![0.0; 9];
    tree.compute_coulomb_repulsion_flat(0.5, 1e-2, &mut coinc_out)
        .unwrap();
    assert_eq!(coinc_out, vec![0.0; 9]);

    // Invalid LayoutNd dims != 3.
    let layout_2d = LayoutNd::<f64>::zeros(4, 2).unwrap();
    assert!(matches!(
        tree.build(&layout_2d),
        Err(CoreError::DimensionMismatch { .. })
    ));
    let layout_4d = LayoutNd::<f64>::zeros(4, 4).unwrap();
    assert!(matches!(
        tree.build_weighted(&layout_4d, &[1.0; 4]),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Invalid flat length, xyz lengths, masses, coords, parameters.
    assert!(matches!(
        tree.build_flat(&[0.0; 5], 2),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        tree.build_xyz(&[0.0, 1.0], &[0.0], &[0.0, 1.0]),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        tree.build_xyz(&[0.0, 1.0], &[0.0, 1.0], &[0.0]),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        tree.build_weighted_xyz(&[0.0], &[0.0], &[0.0], &[-1.0]),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        tree.build_weighted_xyz(&[0.0], &[0.0], &[0.0], &[f64::NAN]),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        tree.build_xyz(&[0.0], &[f64::INFINITY], &[0.0]),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let mut out3 = vec![0.0; 3];
    assert!(matches!(
        tree.compute_power_repulsion_flat(-0.1, -1.0, 1e-4, &mut [0.0; 9]),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        tree.compute_power_repulsion_flat(0.5, f64::NAN, 1e-4, &mut [0.0; 9]),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        tree.compute_power_repulsion_flat(0.5, -1.0, 0.0, &mut [0.0; 9]),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        tree.compute_maxent_repulsion_flat(0.5, -2.0, 1e-4, &mut [0.0; 9]),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        tree.compute_coulomb_repulsion_flat(0.5, 1e-4, &mut out3),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        tree.eval_point_kl_raw(3, 0.0),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn test_octree_near_coincident_chain_and_theta_zero_exactness() {
    let mut tree = ArenaOctree3d::new();
    // Repro A: 3 points with pair inside COINCIDENT_DIST_SQ and third outside.
    let (xs_a, zeros_a) = ([0.0, 0.5e-6, 1.5e-6], [0.0; 3]);
    tree.build_xyz(&xs_a, &zeros_a, &zeros_a).unwrap();
    let (ex_ax, ex_ay, ex_az) =
        brute_coulomb_maxent(&xs_a, &zeros_a, &zeros_a, &[1.0; 3], 1.0, 1e-12);
    let mut out_a = vec![0.0; 9];
    tree.compute_coulomb_repulsion_flat(0.0, 1e-12, &mut out_a)
        .unwrap();
    for i in 0..3 {
        assert!((out_a[i * 3] - ex_ax[i]).abs() <= 1e-9);
        assert!((out_a[i * 3 + 1] - ex_ay[i]).abs() <= 1e-9);
        assert!((out_a[i * 3 + 2] - ex_az[i]).abs() <= 1e-9);
    }

    // Repro B: 200 chained near-coincident points + far point + nearby observer.
    let mut xs_b: Vec<f64> = (0..200).map(|k| k as f64 * 0.8e-6).collect();
    let (mut ys_b, mut zs_b) = (vec![0.0; 200], vec![0.0; 200]);
    xs_b.extend_from_slice(&[10.0, 0.0]);
    ys_b.extend_from_slice(&[0.0, 2e-4]);
    zs_b.extend_from_slice(&[0.0, 0.0]);
    let n_b = xs_b.len();
    tree.build_xyz(&xs_b, &ys_b, &zs_b).unwrap();
    let (ex_bx, ex_by, ex_bz) =
        brute_coulomb_maxent(&xs_b, &ys_b, &zs_b, &vec![1.0; n_b], 1.0, 1e-10);
    let mut out_b = vec![0.0; n_b * 3];
    tree.compute_coulomb_repulsion_flat(0.0, 1e-10, &mut out_b)
        .unwrap();
    for i in 0..n_b {
        for (a, ex) in [ex_bx[i], ex_by[i], ex_bz[i]].into_iter().enumerate() {
            assert!((out_b[i * 3 + a] - ex).abs() <= 1e-8);
        }
    }
    tree.compute_coulomb_repulsion_flat(0.5, 1e-10, &mut out_b)
        .unwrap();
    let obs = n_b - 1;
    let err = ((out_b[obs * 3] - ex_bx[obs]).powi(2)
        + (out_b[obs * 3 + 1] - ex_by[obs]).powi(2)
        + (out_b[obs * 3 + 2] - ex_bz[obs]).powi(2))
    .sqrt();
    let ref_norm = (ex_bx[obs].powi(2) + ex_by[obs].powi(2) + ex_bz[obs].powi(2)).sqrt();
    assert!(err / ref_norm < 1e-2, "rel_err={}", err / ref_norm);
}
