use std::f64::consts::PI;

use super::*;
use crate::error::CoreError;
use crate::geometry::{LayoutStorage, MetricSpace, RiemannianLayout};
use crate::scalar::Scalar;

fn verify_geodesic_reversal_and_isometry<S: Scalar>(
    kind: ManifoldKind,
    p: (f64, f64),
    q: (f64, f64),
    scale: f64,
    aspect: f64,
    tol: f64,
) {
    let mut layout = ManifoldLayout2d::<S>::from_xy(
        kind,
        vec![S::from_f64(p.0), S::from_f64(q.0)],
        vec![S::from_f64(p.1), S::from_f64(q.1)],
    )
    .unwrap();
    if kind == ManifoldKind::Torus2d {
        layout.set_torus_extents(scale * aspect, scale).unwrap();
    } else {
        layout.set_scale(scale).unwrap();
    }

    let d_pq = layout.distance(0, 1);
    let mut delta = [0.0_f64; 2];
    layout.log_map_into(0, 1, &mut delta).unwrap();
    assert!(
        (delta[0].hypot(delta[1]) - d_pq).abs() <= tol * (1.0 + d_pq),
        "{kind:?}: log_map norm {} vs dist {d_pq}",
        delta[0].hypot(delta[1])
    );

    let mut back = [0.0_f64; 2];
    layout.log_map_into(1, 0, &mut back).unwrap();

    // P_{p->q}(log_p(q)) == -log_q(p)
    let mut transported = delta;
    let mut combined_layout = layout.clone();
    let mut combined_vec = delta;
    layout
        .parallel_transport_into(0, &delta, &mut transported)
        .unwrap();
    combined_layout
        .exp_map_and_transport(0, &delta, &mut combined_vec)
        .unwrap();

    assert!(
        (transported[0] - combined_vec[0]).abs() <= 1e-12
            && (transported[1] - combined_vec[1]).abs() <= 1e-12,
        "{kind:?}: exp_map_and_transport mismatch: {transported:?} vs {combined_vec:?}"
    );
    assert!(
        (transported[0] + back[0]).abs() <= tol * (1.0 + d_pq)
            && (transported[1] + back[1]).abs() <= tol * (1.0 + d_pq),
        "{kind:?} p={p:?} q={q:?}: P(log_p(q))={transported:?} vs -log_q(p)={:?}",
        [-back[0], -back[1]]
    );

    // Non-collinear vector norm preservation in all 4 tangent quadrants.
    for &(wx, wy) in &[(1.3, 0.7), (-1.1, 0.9), (-0.8, -1.4), (0.6, -1.2)] {
        let mut w = [wx * scale, wy * scale];
        let norm_before = w[0].hypot(w[1]);
        layout.parallel_transport_into(0, &delta, &mut w).unwrap();
        let norm_after = w[0].hypot(w[1]);
        assert!(
            (norm_after - norm_before).abs() <= 1e-11 * (1.0 + norm_before),
            "{kind:?}: norm drift {norm_before} vs {norm_after}"
        );
    }

    // Verify combined_layout moved vertex 0 onto vertex 1.
    let d_rem = combined_layout.distance(0, 1);
    assert!(
        d_rem <= tol * (1.0 + d_pq),
        "{kind:?} p={p:?} q={q:?}: remaining distance {d_rem} (tol={tol})"
    );
}

#[test]
fn test_riemannian_parallel_transport_all_manifolds_four_quadrants_f64_and_f32() {
    let poincare_pts = [
        (0.0, 0.0),
        (0.35, 0.25),
        (-0.45, 0.30),
        (-0.25, -0.50),
        (0.55, -0.20),
    ];
    for &p in &poincare_pts {
        for &q in &poincare_pts {
            verify_geodesic_reversal_and_isometry::<f64>(
                ManifoldKind::HyperbolicPoincare,
                p,
                q,
                2.5,
                1.0,
                1e-10,
            );
            verify_geodesic_reversal_and_isometry::<f32>(
                ManifoldKind::HyperbolicPoincare,
                p,
                q,
                2.5,
                1.0,
                2e-4,
            );
        }
    }

    let lorentz_pts = [
        (0.0, 0.0),
        (1.4, 0.8),
        (-2.1, 1.3),
        (-1.2, -2.5),
        (3.0, -1.1),
    ];
    for &p in &lorentz_pts {
        for &q in &lorentz_pts {
            verify_geodesic_reversal_and_isometry::<f64>(
                ManifoldKind::HyperbolicLorentz,
                p,
                q,
                1.75,
                1.0,
                1e-9,
            );
            verify_geodesic_reversal_and_isometry::<f32>(
                ManifoldKind::HyperbolicLorentz,
                p,
                q,
                1.75,
                1.0,
                5e-4,
            );
        }
    }

    let sphere_pts = [
        (0.3, 0.4),
        (1.8, 0.9),
        (3.5, 2.1),
        (5.2, 2.6),
        (0.0, 0.0),
        (0.0, PI),
    ];
    for &p in &sphere_pts {
        for &q in &sphere_pts {
            // Skip exact antipodal poles where log_map tangent direction is degenerate.
            if (p.1 == 0.0 && q.1 == PI) || (p.1 == PI && q.1 == 0.0) {
                continue;
            }
            verify_geodesic_reversal_and_isometry::<f64>(
                ManifoldKind::Spherical,
                p,
                q,
                3.0,
                1.0,
                1e-10,
            );
            verify_geodesic_reversal_and_isometry::<f32>(
                ManifoldKind::Spherical,
                p,
                q,
                3.0,
                1.0,
                5e-4,
            );
        }
    }

    let torus_pts = [
        (0.1, 0.1),
        (0.85, 0.2),
        (0.2, 0.88),
        (0.9, 0.92),
        (0.45, 0.55),
    ];
    for &p in &torus_pts {
        for &q in &torus_pts {
            verify_geodesic_reversal_and_isometry::<f64>(
                ManifoldKind::Torus2d,
                p,
                q,
                2.0,
                1.75,
                1e-12,
            );
            verify_geodesic_reversal_and_isometry::<f32>(
                ManifoldKind::Torus2d,
                p,
                q,
                2.0,
                1.75,
                1e-5,
            );
        }
    }
}

#[test]
fn test_lorentz_parallel_transport_extreme_scales_and_ratios() {
    let extreme_cases = [
        ((1e160_f64, 0.0), (1.0, 0.0), (-500.0, 200.0)),
        ((1e200_f64, 1e195), (-1e200, 1e190), (-10.0, 5.0)),
        ((1e-120_f64, -2e-120), (1e20, -3e20), (3.0, -4.0)),
        ((1.5e308_f64, 1.5e308), (1.0, 2.0), (-20.0, 15.0)),
    ];
    for &((px, py), (qx, qy), (dx, dy)) in &extreme_cases {
        let mut layout = ManifoldLayout2d::<f64>::from_xy(
            ManifoldKind::HyperbolicLorentz,
            vec![px, qx],
            vec![py, qy],
        )
        .unwrap();
        let mut w = [1.7, -2.3];
        let norm_before = w[0].hypot(w[1]);
        layout
            .parallel_transport_into(0, &[dx, dy], &mut w)
            .unwrap();
        assert!(w[0].is_finite() && w[1].is_finite());
        assert!((w[0].hypot(w[1]) - norm_before).abs() < 1e-11);

        let mut log_pq = [0.0; 2];
        layout.log_map_into(0, 1, &mut log_pq).unwrap();
        let mut t = log_pq;
        layout.exp_map_and_transport(0, &log_pq, &mut t).unwrap();
        assert!(t[0].is_finite() && t[1].is_finite());
        assert!((t[0].hypot(t[1]) - log_pq[0].hypot(log_pq[1])).abs() < 1e-9);
        layout.validate_finite().unwrap();
    }
}

#[test]
fn test_manifold_move_pair_toward_and_validation_errors() {
    for (kind, xs, ys) in [
        (
            ManifoldKind::HyperbolicPoincare,
            vec![-0.4, 0.4],
            vec![0.1, -0.1],
        ),
        (
            ManifoldKind::HyperbolicLorentz,
            vec![-1.5, 2.0],
            vec![0.5, -1.0],
        ),
        (ManifoldKind::Spherical, vec![0.3, 1.9], vec![1.0, 1.4]),
        (ManifoldKind::Torus2d, vec![0.9, 0.1], vec![0.2, 0.8]),
    ] {
        let mut m = ManifoldLayout2d::<f64>::from_xy(kind, xs, ys).unwrap();
        assert_eq!(LayoutStorage::len(&m), 2);
        let d0 = m.distance(0, 1);
        let target = d0 * 0.4;
        m.move_pair_toward(0, 1, target, 0.25, 0.25).unwrap();
        let d1 = m.distance(0, 1);
        let expected = d0 - 0.5 * (d0 - target);
        assert!(
            (d1 - expected).abs() < 1e-10,
            "{kind:?}: d1={d1} vs expected={expected}"
        );
    }

    let mut m = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Spherical, 2);
    assert!(matches!(
        m.exp_map(2, &[0.0, 0.0]),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        m.exp_map(0, &[0.0]),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        m.parallel_transport_into(0, &[0.0, 0.0], &mut [f64::NAN, 0.0]),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        m.move_pair_toward(0, 1, f64::NAN, 0.1, 0.1),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
