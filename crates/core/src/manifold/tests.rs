use std::f64::consts::{PI, TAU};

use super::*;
use crate::error::CoreError;
use crate::layout::Layout2d;
use crate::scalar::Scalar;

#[test]
fn test_poincare_clamp_distance_and_exp_log_roundtrip() {
    let (cx, cy) = clamp_poincare_point(3.0_f64, 4.0);
    assert!(cx.hypot(cy) < 1.0);
    assert!((cx / cy - 0.75).abs() < 1e-12);

    let origin_dist = poincare_distance(0.0_f64, 0.0, 0.5, 0.0);
    let expected = 2.0 * 0.5_f64.atanh();
    assert!((origin_dist - expected).abs() < 1e-12);

    let x = (0.2_f64, -0.3);
    let y = (-0.4_f64, 0.25);
    let (vx, vy) = poincare_log_map(x.0, x.1, y.0, y.1);
    let d = poincare_distance(x.0, x.1, y.0, y.1);
    assert!((vx.hypot(vy) - d).abs() < 1e-10);
    let (rx, ry) = poincare_exp_map(x.0, x.1, vx, vy);
    assert!((rx - y.0).abs() < 1e-10);
    assert!((ry - y.1).abs() < 1e-10);
}

#[test]
fn test_spherical_projection_distance_and_exp_log() {
    let (lon, colat) = project_spherical_coords(-0.5_f64, -0.25);
    assert!((0.0..TAU).contains(&lon));
    assert!((0.0..=PI).contains(&colat));
    assert!((colat - 0.25).abs() < 1e-12);

    let p1 = (0.2_f64, PI / 2.0);
    let p2 = (0.2 + PI / 2.0, PI / 2.0);
    let d = spherical_distance(p1.0, p1.1, p2.0, p2.1);
    assert!((d - PI / 2.0).abs() < 1e-12);

    let (vx, vy) = spherical_log_map(0.4_f64, 1.1, 1.2, 0.7);
    let d_pair = spherical_distance(0.4_f64, 1.1, 1.2, 0.7);
    assert!((vx.hypot(vy) - d_pair).abs() < 1e-12);
    let (rlon, rcolat) = spherical_exp_map(0.4_f64, 1.1, vx, vy);
    assert!(spherical_distance(rlon, rcolat, 1.2, 0.7) < 1e-11);
}

#[test]
fn test_torus_wrap_delta_and_distance() {
    assert_eq!(wrap_torus_coord(-1.0_f64), 0.0);
    assert_eq!(wrap_torus_coord(-1e-17_f64), 0.0);
    assert!((wrap_torus_coord(-0.125_f64) - 0.875).abs() < 1e-12);
    assert_eq!(wrap_torus_coord(-1e-9_f32), 0.0_f32);

    let (dx, dy) = torus_delta(0.0_f64, 0.75, 0.5, 0.5);
    assert!((dx - (-0.5)).abs() < 1e-12);
    assert!((dy - 0.25).abs() < 1e-12);

    let d_wrap = torus_distance(0.95_f64, 0.05, 0.05, 0.95);
    assert!((d_wrap - (0.1_f64).hypot(0.1)).abs() < 1e-12);
}

#[test]
fn test_manifold_layout_2d_api() {
    let mut m =
        ManifoldLayout2d::from_xy(ManifoldKind::Torus2d, vec![1.2, 0.9], vec![-0.1, 0.2]).unwrap();
    assert_eq!(m.kind(), ManifoldKind::Torus2d);
    assert_eq!(m.len(), 2);
    assert!((m.x()[0] - 0.2).abs() < 1e-12);
    assert!((m.y()[0] - 0.9).abs() < 1e-12);

    let d_before = m.geodesic_distance(0, 1).unwrap();
    let (vx, vy) = m.tangent_delta(0, 1).unwrap();
    assert!((vx.hypot(vy) - d_before).abs() < 1e-12);
    m.apply_tangent_step(0, vx * 0.5, vy * 0.5).unwrap();
    let d_after = m.geodesic_distance(0, 1).unwrap();
    assert!((d_after - d_before * 0.5).abs() < 1e-12);

    assert!(matches!(
        m.set_point(2, 0.0, 0.0),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        ManifoldLayout2d::from_xy(ManifoldKind::Spherical, vec![f64::NAN], vec![0.0]),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_poincare_mobius_centroid_and_safe_init_f64_f32() {
    // Clamp handles huge coordinates without underflow
    let (hx, hy) = clamp_poincare_point(1e300_f64, 2e300_f64);
    assert!(hx.is_finite() && hy.is_finite() && hx.hypot(hy) < 1.0);

    // Mobius addition identity, inverse, and boundary clamping
    let (id_x, id_y) = poincare_mobius_add(0.0_f64, 0.0, 0.3, -0.4);
    assert!((id_x - 0.3).abs() < 1e-12 && (id_y - (-0.4)).abs() < 1e-12);
    let (inv_x, inv_y) = poincare_mobius_add(-0.3_f64, 0.4, 0.3, -0.4);
    assert!(inv_x.hypot(inv_y) < 1e-12);
    let (bx, by) = poincare_mobius_add(0.99_f32, 0.0, 0.99, 0.0);
    assert!(bx.is_finite() && by.is_finite() && (bx * bx + by * by) < 1.0);

    // Einstein midpoint edge cases
    assert_eq!(
        poincare_einstein_midpoint::<f64>(&[], &[]).unwrap(),
        (0.0, 0.0)
    );
    let (sx, sy) = poincare_einstein_midpoint(&[0.25_f64], &[-0.35]).unwrap();
    assert!((sx - 0.25).abs() < 1e-12 && (sy - (-0.35)).abs() < 1e-12);
    let (sym_x, sym_y) = poincare_einstein_midpoint(&[-0.4_f64, 0.4], &[-0.2, 0.2]).unwrap();
    assert!(sym_x.hypot(sym_y) < 1e-12);
    assert!(matches!(
        poincare_einstein_midpoint(&[0.1_f64], &[0.1, 0.2]),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        poincare_einstein_midpoint(&[f64::NAN], &[0.0]),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Centering preserves pairwise geodesic distances isometrically (f64)
    let mut hyp64 = ManifoldLayout2d::from_xy(
        ManifoldKind::HyperbolicPoincare,
        vec![0.6, 0.7, 0.55, 0.85],
        vec![0.4, 0.3, 0.5, -0.1],
    )
    .unwrap();
    let mut dists_before = Vec::new();
    for i in 0..hyp64.len() {
        for j in (i + 1)..hyp64.len() {
            dists_before.push(hyp64.geodesic_distance(i, j).unwrap());
        }
    }
    hyp64.center_poincare_centroid().unwrap();
    let (cx, cy) = hyp64.poincare_centroid().unwrap();
    assert!(
        cx.hypot(cy) < 1e-12,
        "centroid after centering: ({cx}, {cy})"
    );
    let mut idx = 0usize;
    for i in 0..hyp64.len() {
        for j in (i + 1)..hyp64.len() {
            let d_after = hyp64.geodesic_distance(i, j).unwrap();
            assert!((d_after - dists_before[idx]).abs() < 1e-10);
            idx += 1;
        }
    }

    // recenter_poincare_at moves node 2 to (0, 0) isometrically
    hyp64.recenter_poincare_at(2).unwrap();
    assert!(hyp64.x()[2].hypot(hyp64.y()[2]) < 1e-12);
    let mut idx2 = 0usize;
    for i in 0..hyp64.len() {
        for j in (i + 1)..hyp64.len() {
            let d_after = hyp64.geodesic_distance(i, j).unwrap();
            assert!((d_after - dists_before[idx2]).abs() < 1e-10);
            idx2 += 1;
        }
    }

    // f32 parity + near-rim inputs
    let mut hyp32 = ManifoldLayout2d::from_xy(
        ManifoldKind::HyperbolicPoincare,
        vec![0.999_f32, 0.995, 0.99],
        vec![0.0_f32, 0.05, -0.05],
    )
    .unwrap();
    let d32_before = hyp32.geodesic_distance(0, 1).unwrap();
    hyp32.center_poincare_centroid().unwrap();
    let (cx32, cy32) = hyp32.poincare_centroid().unwrap();
    assert!(cx32.hypot(cy32) < 1e-4);
    assert!((hyp32.geodesic_distance(0, 1).unwrap() - d32_before).abs() < 1e-3);

    // random_init across all 3 manifolds
    let r_hyp =
        ManifoldLayout2d::<f64>::random_init(ManifoldKind::HyperbolicPoincare, 16, 0.25, 42)
            .unwrap();
    let r_hyp2 =
        ManifoldLayout2d::<f64>::random_init(ManifoldKind::HyperbolicPoincare, 16, 0.25, 42)
            .unwrap();
    assert_eq!(r_hyp, r_hyp2);
    for (&x, &y) in r_hyp.x().iter().zip(r_hyp.y().iter()) {
        assert!(x.hypot(y) <= 0.25 + 1e-12);
    }
    // Breaking zeros deadlock: tangent_delta between distinct nodes is nonzero
    let (tdx, tdy) = r_hyp.tangent_delta(0, 1).unwrap();
    assert!(tdx.hypot(tdy) > 0.0);

    let r_sph =
        ManifoldLayout2d::<f32>::random_init(ManifoldKind::Spherical, 12, 0.25, 42).unwrap();
    for (&lon, &colat) in r_sph.x().iter().zip(r_sph.y().iter()) {
        assert!((0.0..std::f32::consts::TAU).contains(&lon));
        assert!((0.0..=std::f32::consts::PI).contains(&colat));
    }

    let r_tor = ManifoldLayout2d::<f32>::random_init(ManifoldKind::Torus2d, 12, 0.25, 42).unwrap();
    for (&tx, &ty) in r_tor.x().iter().zip(r_tor.y().iter()) {
        assert!((0.0..1.0).contains(&tx) && (0.0..1.0).contains(&ty));
    }

    // from_layout_scaled avoids rim collapse on large Euclidean layouts
    let big_euclid = Layout2d::from_xy(vec![10.0_f64, 20.0, 30.0], vec![-5.0, 5.0, 15.0]).unwrap();
    let scaled =
        ManifoldLayout2d::from_layout_scaled(ManifoldKind::HyperbolicPoincare, &big_euclid, 0.7)
            .unwrap();
    let max_r = scaled
        .x()
        .iter()
        .zip(scaled.y().iter())
        .map(|(&x, &y)| x.hypot(y))
        .fold(0.0_f64, f64::max);
    assert!((max_r - 0.7).abs() < 1e-12);

    // Non-Poincare manifolds reject Poincare-only operations
    let mut tor = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, 3);
    assert!(matches!(
        tor.poincare_centroid(),
        Err(CoreError::UnsupportedOperation(_))
    ));
    assert!(matches!(
        tor.recenter_poincare(0.1, 0.1),
        Err(CoreError::UnsupportedOperation(_))
    ));
    assert!(matches!(
        tor.recenter_poincare_at(0),
        Err(CoreError::UnsupportedOperation(_))
    ));
    assert!(matches!(
        tor.center_poincare_centroid(),
        Err(CoreError::UnsupportedOperation(_))
    ));
}
