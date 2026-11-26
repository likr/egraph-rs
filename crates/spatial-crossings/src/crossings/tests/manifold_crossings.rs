use std::f64::consts::{FRAC_PI_2, PI};

use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, ManifoldKind, ManifoldLayout2d, spherical_to_cartesian,
};

use super::super::{
    crossing_edges_hyperbolic, crossing_edges_spherical, hyperbolic_crossing_number,
    spherical_crossing_number, torus_crossing_edges, torus_crossing_number,
    torus_layout_edge_segments,
};

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn arc_angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    let c = cross3(a, b);
    dot3(c, c).sqrt().atan2(dot3(a, b).clamp(-1.0, 1.0))
}

fn brute_force_spherical_crossings(
    edges: &[(u32, u32)],
    layout: &ManifoldLayout2d<f64>,
) -> Vec<(usize, usize)> {
    let pts: Vec<[f64; 3]> = (0..layout.len())
        .map(|i| spherical_to_cartesian(layout.x()[i], layout.y()[i]))
        .collect();
    let mut out = Vec::new();
    for i in 0..edges.len() {
        let (u1, v1) = edges[i];
        if u1 == v1 {
            continue;
        }
        let a = pts[u1 as usize];
        let b = pts[v1 as usize];
        let n1 = cross3(a, b);
        if dot3(n1, n1).sqrt() <= 1e-12 {
            continue;
        }
        let d_ab = arc_angle(a, b);
        for (j, &(u2, v2)) in edges.iter().enumerate().skip(i + 1) {
            if u2 == v2 || u1 == u2 || u1 == v2 || v1 == u2 || v1 == v2 {
                continue;
            }
            let c = pts[u2 as usize];
            let d = pts[v2 as usize];
            let n2 = cross3(c, d);
            if dot3(n2, n2).sqrt() <= 1e-12 {
                continue;
            }
            let d_cd = arc_angle(c, d);
            let line = cross3(n1, n2);
            let ln = dot3(line, line).sqrt();
            if ln <= 1e-12 {
                continue;
            }
            let p = [line[0] / ln, line[1] / ln, line[2] / ln];
            let np = [-p[0], -p[1], -p[2]];
            let on_p = (arc_angle(a, p) + arc_angle(p, b) - d_ab).abs() < 1e-9
                && (arc_angle(c, p) + arc_angle(p, d) - d_cd).abs() < 1e-9;
            let on_np = (arc_angle(a, np) + arc_angle(np, b) - d_ab).abs() < 1e-9
                && (arc_angle(c, np) + arc_angle(np, d) - d_cd).abs() < 1e-9;
            if on_p || on_np {
                out.push((i, j));
            }
        }
    }
    out
}

fn mobius_sub_exact(ax: f64, ay: f64, bx: f64, by: f64) -> (f64, f64) {
    // Computes (b - a) / (1 - conj(a) * b) without disk-rim clamping for exact oracle geometry.
    let num_x = bx - ax;
    let num_y = by - ay;
    let den_r = 1.0 - (ax * bx + ay * by);
    let den_i = -(ax * by - ay * bx);
    let den_sq = den_r * den_r + den_i * den_i;
    (
        (num_x * den_r + num_y * den_i) / den_sq,
        (num_y * den_r - num_x * den_i) / den_sq,
    )
}

/// Independent O(M^2) Poincaré oracle using Möbius-translation to map edge `(u1, v1)` onto the
/// real axis `[0, r_B]`, then testing intersection with edge `(u2, v2)` via orthogonal circle.
fn mobius_to_origin_axis_crosses(
    (ax, ay): (f64, f64),
    (bx, by): (f64, f64),
    (cx, cy): (f64, f64),
    (dx, dy): (f64, f64),
) -> bool {
    let (tbx, tby) = mobius_sub_exact(ax, ay, bx, by);
    let rb = tbx.hypot(tby);
    if rb <= 1e-15 {
        return false;
    }
    let cos_r = tbx / rb;
    let sin_r = tby / rb;
    let rot = |x: f64, y: f64| -> (f64, f64) { (cos_r * x + sin_r * y, -sin_r * x + cos_r * y) };
    let (tcx, tcy) = mobius_sub_exact(ax, ay, cx, cy);
    let (tdx, tdy) = mobius_sub_exact(ax, ay, dx, dy);
    let (ucx, ucy) = rot(tcx, tcy);
    let (udx, udy) = rot(tdx, tdy);
    if (ucx - udx).hypot(ucy - udy) <= 1e-15 {
        return false;
    }
    // Since edge 1 is [0, rb] on y = 0, edge 2 crosses [0, rb] iff its endpoints straddle y = 0
    // (or touch it) and the geodesic's x-intercept lies in [0, rb].
    if (ucy > 1e-13 && udy > 1e-13) || (ucy < -1e-13 && udy < -1e-13) {
        return false;
    }
    let det = 2.0 * (ucx * udy - ucy * udx);
    let xi = if det.abs() <= 1e-13 {
        // Diameter geodesic through origin; crosses [0, rb] at x = 0 if opposite sides, or overlaps.
        if ucx * udx <= 1e-15 {
            0.0
        } else {
            return false;
        }
    } else {
        let rhs1 = 1.0 + ucx * ucx + ucy * ucy;
        let rhs2 = 1.0 + udx * udx + udy * udy;
        let circ_cx = (rhs1 * udy - rhs2 * ucy) / det;
        if circ_cx.abs() <= 1.0 {
            return false;
        }
        let sgn = circ_cx.signum();
        circ_cx - sgn * (circ_cx * circ_cx - 1.0).max(0.0).sqrt()
    };
    xi >= -1e-12 && xi <= rb + 1e-12
}

fn brute_force_hyperbolic_crossings(
    edges: &[(u32, u32)],
    layout: &ManifoldLayout2d<f64>,
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..edges.len() {
        let (u1, v1) = edges[i];
        if u1 == v1 {
            continue;
        }
        for (j, &(u2, v2)) in edges.iter().enumerate().skip(i + 1) {
            if u2 == v2 || u1 == u2 || u1 == v2 || v1 == u2 || v1 == v2 {
                continue;
            }
            if mobius_to_origin_axis_crosses(
                (layout.x()[u1 as usize], layout.y()[u1 as usize]),
                (layout.x()[v1 as usize], layout.y()[v1 as usize]),
                (layout.x()[u2 as usize], layout.y()[u2 as usize]),
                (layout.x()[v2 as usize], layout.y()[v2 as usize]),
            ) {
                out.push((i, j));
            }
        }
    }
    out
}

#[test]
fn test_spherical_crossings_oracle_and_degeneracies() {
    // Transverse crossing: equator arc vs meridian arc gives 1 crossing with right angle.
    let g_cross = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let sph_cross = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Spherical,
        vec![0.0, 1.0, 0.5, 0.5],
        vec![FRAC_PI_2, FRAC_PI_2, 1.0, 2.0],
    )
    .unwrap();
    assert_eq!(spherical_crossing_number(&g_cross, &sph_cross).unwrap(), 1);
    let pairs = crossing_edges_spherical(&g_cross, &sph_cross).unwrap();
    assert_eq!(pairs.len(), 1);
    assert!((pairs[0].angle_rad.unwrap() - FRAC_PI_2).abs() < 1e-11);
    assert!(pairs[0].cos_sq < 1e-12);

    // Shared endpoints, antipodal endpoints, coplanar great circles, and coincident vertices.
    let g_shared = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    let sph_shared = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Spherical,
        vec![0.0, 0.5, 1.0],
        vec![1.5; 3],
    )
    .unwrap();
    assert_eq!(
        spherical_crossing_number(&g_shared, &sph_shared).unwrap(),
        0
    );

    let sph_anti = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Spherical,
        vec![0.0, PI, 0.5, 1.5],
        vec![FRAC_PI_2; 4],
    )
    .unwrap();
    assert_eq!(spherical_crossing_number(&g_cross, &sph_anti).unwrap(), 0);

    // Randomized O(M^2) oracle across 20 seeds.
    for seed in 1..=20u64 {
        let mut rng = DeterministicRng::from_seed_u64(71_000 + seed);
        let n = 16usize;
        let sph =
            ManifoldLayout2d::<f64>::random_init(ManifoldKind::Spherical, n, 1.0, seed).unwrap();
        let mut edges = Vec::new();
        for _ in 0..24 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            if u < v && !edges.contains(&(u, v)) {
                edges.push((u, v));
            } else if v < u && !edges.contains(&(v, u)) {
                edges.push((v, u));
            }
        }
        let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
        let mut csr_edges = Vec::new();
        for u in 0..n {
            for &v in g.neighbors(u) {
                if (u as u32) < v {
                    csr_edges.push((u as u32, v));
                }
            }
        }
        let expected = brute_force_spherical_crossings(&csr_edges, &sph);
        let got_pairs = crossing_edges_spherical(&g, &sph).unwrap();
        let got_indices: Vec<(usize, usize)> =
            got_pairs.iter().map(|p| (p.edge_a, p.edge_b)).collect();
        assert_eq!(got_indices, expected, "spherical seed {seed}");
        assert_eq!(spherical_crossing_number(&g, &sph).unwrap(), expected.len());
    }

    // Wrong manifold kind rejected.
    let tor = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, 4);
    assert!(matches!(
        spherical_crossing_number(&g_cross, &tor),
        Err(CoreError::UnsupportedOperation(_))
    ));
}

#[test]
fn test_hyperbolic_crossings_oracle_lorentz_and_angles() {
    // Orthogonal diameters cross at origin with right angle.
    let g_cross = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let hyp = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::HyperbolicPoincare,
        vec![-0.5, 0.5, 0.0, 0.0],
        vec![0.0, 0.0, -0.5, 0.5],
    )
    .unwrap();
    assert_eq!(hyperbolic_crossing_number(&g_cross, &hyp).unwrap(), 1);
    let pairs = crossing_edges_hyperbolic(&g_cross, &hyp).unwrap();
    assert_eq!(pairs.len(), 1);
    assert!((pairs[0].angle_rad.unwrap() - FRAC_PI_2).abs() < 1e-11);

    // HyperbolicLorentz yields identical crossings to its to_poincare_layout() image.
    let lor = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::HyperbolicLorentz,
        vec![-3.0, 3.0, 0.0, 0.0],
        vec![0.0, 0.0, -3.0, 3.0],
    )
    .unwrap();
    assert_eq!(hyperbolic_crossing_number(&g_cross, &lor).unwrap(), 1);

    // Randomized O(M^2) Möbius-circle oracle across 20 seeds (including near-rim points).
    for seed in 1..=20u64 {
        let mut rng = DeterministicRng::from_seed_u64(82_000 + seed);
        let n = 16usize;
        let mut ml =
            ManifoldLayout2d::<f64>::random_init(ManifoldKind::HyperbolicPoincare, n, 0.96, seed)
                .unwrap();
        // Inject boundary-clamped point on vertex 0.
        ml.set_point(0, 0.999999, 0.0).unwrap();
        let mut edges = Vec::new();
        for _ in 0..24 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            if u < v && !edges.contains(&(u, v)) {
                edges.push((u, v));
            } else if v < u && !edges.contains(&(v, u)) {
                edges.push((v, u));
            }
        }
        let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
        let mut csr_edges = Vec::new();
        for u in 0..n {
            for &v in g.neighbors(u) {
                if (u as u32) < v {
                    csr_edges.push((u as u32, v));
                }
            }
        }
        let expected = brute_force_hyperbolic_crossings(&csr_edges, &ml);
        let got_pairs = crossing_edges_hyperbolic(&g, &ml).unwrap();
        let got_indices: Vec<(usize, usize)> =
            got_pairs.iter().map(|p| (p.edge_a, p.edge_b)).collect();
        assert_eq!(got_indices, expected, "hyperbolic seed {seed}");
        assert_eq!(hyperbolic_crossing_number(&g, &ml).unwrap(), expected.len());
    }

    let sph = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Spherical, 4);
    assert!(matches!(
        hyperbolic_crossing_number(&g_cross, &sph),
        Err(CoreError::UnsupportedOperation(_))
    ));

    // Off-center Möbius-shifted orthogonal geodesics preserve conformal right angle (pi/2).
    let (ax, ay) = egraph_core::poincare_mobius_add(0.5_f64, 0.3, -0.4, 0.0);
    let (bx, by) = egraph_core::poincare_mobius_add(0.5_f64, 0.3, 0.4, 0.0);
    let (cx, cy) = egraph_core::poincare_mobius_add(0.5_f64, 0.3, 0.0, -0.4);
    let (dx, dy) = egraph_core::poincare_mobius_add(0.5_f64, 0.3, 0.0, 0.4);
    let off_center = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::HyperbolicPoincare,
        vec![ax, bx, cx, dx],
        vec![ay, by, cy, dy],
    )
    .unwrap();
    let off_pairs = crossing_edges_hyperbolic(&g_cross, &off_center).unwrap();
    assert_eq!(off_pairs.len(), 1);
    assert!(
        (off_pairs[0].angle_rad.unwrap() - FRAC_PI_2).abs() < 1e-9,
        "expected conformal right angle pi/2, got {:?}",
        off_pairs[0].angle_rad
    );
    assert!(off_pairs[0].cos_sq < 1e-12);

    // Empty / singleton / edgeless boundary checks across all three manifolds.
    let g_empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let g_single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    assert_eq!(
        spherical_crossing_number(
            &g_empty,
            &ManifoldLayout2d::<f64>::zeros(ManifoldKind::Spherical, 0)
        )
        .unwrap(),
        0
    );
    assert_eq!(
        hyperbolic_crossing_number(
            &g_single,
            &ManifoldLayout2d::<f64>::zeros(ManifoldKind::HyperbolicLorentz, 1)
        )
        .unwrap(),
        0
    );
    assert_eq!(
        torus_crossing_number(
            &g_empty,
            &ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, 0)
        )
        .unwrap(),
        0
    );
}

#[test]
fn test_torus_manifold_layout_view_and_rectangular_aspect() {
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    // Perpendicular diagonals on [0, 1)^2: (0.2, 0.2)->(0.6, 0.6) and (0.2, 0.6)->(0.6, 0.2).
    let mut tor = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.2, 0.6, 0.2, 0.6],
        vec![0.2, 0.6, 0.6, 0.2],
    )
    .unwrap();
    let c_sq = torus_crossing_edges(&g, &tor).unwrap();
    assert_eq!(c_sq.len(), 1);
    assert_eq!(torus_crossing_number(&g, &tor).unwrap(), 1);
    assert!(c_sq[0].cos_sq < 1e-12);
    assert!((c_sq[0].angle_rad.unwrap() - FRAC_PI_2).abs() < 1e-12);

    // Setting rectangular extents (2.0, 1.0) changes the physical crossing angle to cos_sq = 0.36.
    tor.set_torus_extents(2.0, 1.0).unwrap();
    let c_rect = torus_crossing_edges(&g, &tor).unwrap();
    assert_eq!(c_rect.len(), 1);
    assert!((c_rect[0].cos_sq - 0.36).abs() < 1e-12);
    assert_eq!(c_rect[0].u1, 0);
    assert_eq!(c_rect[0].v1, 1);
    assert_eq!(c_rect[0].u2, 2);
    assert_eq!(c_rect[0].v2, 3);

    // Wrapping fixture segments stay in [0, 1)^2 and split across boundary.
    let tor_wrap = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.9, 0.1, 0.9, 0.1],
        vec![0.1, 0.9, 0.9, 0.1],
    )
    .unwrap();
    let segs = torus_layout_edge_segments(&g, &tor_wrap).unwrap();
    assert!(segs.len() >= 4);
    for seg in segs {
        for coord in seg {
            assert!((0.0..1.0).contains(&coord));
        }
    }

    // Non-torus ManifoldLayout2d is rejected.
    let sph = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Spherical, 4);
    assert!(matches!(
        torus_crossing_edges(&g, &sph),
        Err(CoreError::UnsupportedOperation(_))
    ));
}
