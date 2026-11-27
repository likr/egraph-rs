use super::*;
use egraph_core::DeterministicRng;
use std::collections::BTreeSet;
use std::f64::consts::TAU;

#[test]
fn test_delaunay_square_triangle_collinear_and_coincident() {
    let square = Layout2d::from_xy(vec![0.0_f64, 1.0, 0.0, 1.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    let dt_sq = Delaunay2d::triangulate(&square).unwrap();
    assert_eq!(dt_sq.num_nodes(), 4);
    assert_eq!(dt_sq.edges().len(), 5);
    assert_eq!(dt_sq.triangles().len(), 2);
    let csr = dt_sq.to_csr().unwrap();
    assert_eq!(csr.num_edges(), 5);
    let u_csr = dt_sq.to_unweighted_csr().unwrap();
    assert_eq!(u_csr.num_nodes(), csr.num_nodes());
    assert_eq!(u_csr.num_edges(), csr.num_edges());
    assert_eq!(u_csr.indptr(), csr.indptr());
    assert_eq!(u_csr.indices(), csr.indices());

    let collinear =
        Layout2d::from_xy(vec![0.0_f32, 1.0, 2.0, 3.0], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let dt_col = Delaunay2d::triangulate(&collinear).unwrap();
    assert_eq!(dt_col.edges(), &[(0, 1), (1, 2), (2, 3)]);
    assert!(dt_col.triangles().is_empty());
    let u_col = dt_col.to_unweighted_csr().unwrap();
    assert_eq!(u_col.num_edges(), 3);

    // Coincident points keep smallest vertex IDs deterministically and do not panic.
    let dup = Layout2d::from_xy(vec![0.0_f64, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]).unwrap();
    let dt_dup = Delaunay2d::triangulate(&dup).unwrap();
    assert_eq!(dt_dup.edges(), &[(0, 2), (0, 3), (2, 3)]);
    assert_eq!(dt_dup.triangles(), &[[0, 2, 3]]);
    let u_dup = dt_dup.to_unweighted_csr().unwrap();
    assert_eq!(u_dup.num_nodes(), 4);
    assert_eq!(u_dup.num_edges(), 3);

    let single = Layout2d::from_xy(vec![0.0_f64], vec![0.0]).unwrap();
    let dt_single = Delaunay2d::triangulate(&single).unwrap();
    assert!(dt_single.edges().is_empty());
    assert!(dt_single.triangles().is_empty());
    let u_single = dt_single.to_unweighted_csr().unwrap();
    assert_eq!(u_single.num_nodes(), 1);
    assert_eq!(u_single.num_edges(), 0);
}

#[test]
fn test_delaunay_rejects_nonfinite_and_extreme_coords() {
    let bad = Layout2d::from_xy(vec![0.0, f64::NAN, 1.0], vec![0.0, 1.0, 0.0]).unwrap();
    assert!(matches!(
        Delaunay2d::triangulate(&bad),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let huge = Layout2d::from_xy(vec![0.0, 1e200, 1.0], vec![0.0, 0.0, 1.0]).unwrap();
    assert!(matches!(
        Delaunay2d::triangulate(&huge),
        Err(CoreError::NumericalBreakdown(_))
    ));
}

#[test]
fn test_delaunay_triangulate_edges_into_matches_and_errors() {
    let mut edges = vec![(99, 100)];

    let square = Layout2d::from_xy(vec![0.0_f64, 1.0, 0.0, 1.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    Delaunay2d::triangulate_edges_into(&square, &mut edges).unwrap();
    assert_eq!(edges.len(), 5);
    assert!(edges.windows(2).all(|w| w[0] < w[1]));

    let collinear =
        Layout2d::from_xy(vec![0.0_f32, 1.0, 2.0, 3.0], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    Delaunay2d::triangulate_edges_into(&collinear, &mut edges).unwrap();
    assert_eq!(edges, vec![(0, 1), (1, 2), (2, 3)]);

    let dup = Layout2d::from_xy(vec![0.0_f64, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]).unwrap();
    Delaunay2d::triangulate_edges_into(&dup, &mut edges).unwrap();
    assert_eq!(edges, vec![(0, 2), (0, 3), (2, 3)]);

    let single = Layout2d::from_xy(vec![0.0_f64], vec![0.0]).unwrap();
    Delaunay2d::triangulate_edges_into(&single, &mut edges).unwrap();
    assert!(edges.is_empty());

    let bad = Layout2d::from_xy(vec![0.0, f64::NAN, 1.0], vec![0.0, 1.0, 0.0]).unwrap();
    assert!(matches!(
        Delaunay2d::triangulate_edges_into(&bad, &mut edges),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let huge = Layout2d::from_xy(vec![0.0, 1e200, 1.0], vec![0.0, 0.0, 1.0]).unwrap();
    assert!(matches!(
        Delaunay2d::triangulate_edges_into(&huge, &mut edges),
        Err(CoreError::NumericalBreakdown(_))
    ));
}

fn orient2d_f64(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> f64 {
    (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
}

fn incircle_signed_f64(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> f64 {
    // Translate by (a[0], a[1]) to preserve precision on large-offset coordinates.
    let bdx = b[0] - a[0];
    let bdy = b[1] - a[1];
    let cdx = c[0] - a[0];
    let cdy = c[1] - a[1];
    let ddx = d[0] - a[0];
    let ddy = d[1] - a[1];
    let orient = bdx * cdy - bdy * cdx;

    let adx = -ddx;
    let ady = -ddy;
    let bx_d = bdx - ddx;
    let by_d = bdy - ddy;
    let cx_d = cdx - ddx;
    let cy_d = cdy - ddy;

    let ab = adx * by_d - bx_d * ady;
    let bc = bx_d * cy_d - cx_d * by_d;
    let ca = cx_d * ady - adx * cy_d;
    let det = (adx * adx + ady * ady) * bc
        + (bx_d * bx_d + by_d * by_d) * ca
        + (cx_d * cx_d + cy_d * cy_d) * ab;
    if orient >= 0.0 { det } else { -det }
}

fn assert_delaunay_empty_circumcircle_oracle<S: Scalar>(
    label: &str,
    layout: &Layout2d<S>,
    tol: f64,
    exact_edges_into: bool,
) {
    let dt = Delaunay2d::triangulate(layout).unwrap();
    let mut edges_into = Vec::new();
    Delaunay2d::triangulate_edges_into(layout, &mut edges_into).unwrap();
    if exact_edges_into {
        assert_eq!(
            dt.edges(),
            edges_into.as_slice(),
            "{label}: edges_into mismatch"
        );
    } else {
        // Cocircular 4-tuples admit multiple valid diagonal flips between sequential `insert`
        // and `bulk_load_stable`, both preserving the exact edge count and sorted invariant.
        assert_eq!(
            dt.edges().len(),
            edges_into.len(),
            "{label}: edges_into count mismatch"
        );
        assert!(
            edges_into.windows(2).all(|w| w[0] < w[1]),
            "{label}: edges_into not strictly sorted"
        );
    }
    assert!(
        dt.edges().windows(2).all(|w| w[0] < w[1]),
        "{label}: edges not strictly sorted"
    );

    let edge_set: BTreeSet<(u32, u32)> = dt.edges().iter().copied().collect();
    let xs = layout.x();
    let ys = layout.y();
    let n = layout.len();

    // Collect canonical unique site indices matching Delaunay2d deduplication.
    let mut unique_sites: Vec<(f64, f64, u32)> = (0..n)
        .map(|i| (xs[i].to_f64(), ys[i].to_f64(), i as u32))
        .collect();
    unique_sites.sort_unstable_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.total_cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    unique_sites.dedup_by(|b, a| a.0 == b.0 && a.1 == b.1);

    // Verify every triangle has non-zero area, all 3 edges in edge_set,
    // and no unique site lies strictly inside its circumcircle.
    for &[a, b, c] in dt.triangles() {
        assert!(
            a < b && a < c && b != c,
            "{label}: triangle not canonical [{a},{b},{c}]"
        );
        for (u, v) in [(a, b), (b, c), (a, c)] {
            let key = if u < v { (u, v) } else { (v, u) };
            assert!(
                edge_set.contains(&key),
                "{label}: missing triangle edge {key:?}"
            );
        }
        let ax = xs[a as usize].to_f64();
        let ay = ys[a as usize].to_f64();
        let bx = xs[b as usize].to_f64();
        let by = ys[b as usize].to_f64();
        let cx = xs[c as usize].to_f64();
        let cy = ys[c as usize].to_f64();
        let area2 = orient2d_f64(ax, ay, bx, by, cx, cy).abs();
        assert!(area2 > 0.0, "{label}: degenerate triangle [{a},{b},{c}]");

        let scale = ((bx - ax).hypot(by - ay))
            .max((cx - ax).hypot(cy - ay))
            .max(1.0);
        let scaled_tol = tol * scale.powi(4);

        for &(dx, dy, s_idx) in &unique_sites {
            if s_idx == a || s_idx == b || s_idx == c {
                continue;
            }
            let val = incircle_signed_f64([ax, ay], [bx, by], [cx, cy], [dx, dy]);
            assert!(
                val <= scaled_tol,
                "{label}: site {s_idx} strictly inside circumcircle of [{a},{b},{c}]: val={val}, tol={scaled_tol}"
            );
        }
    }
}

#[test]
fn test_delaunay_brute_force_empty_circumcircle_oracles() {
    // 1. Multi-seed randomized f64 and f32 point sets.
    for seed in 1..=12u64 {
        let mut rng = DeterministicRng::from_seed_u64(8_000 + seed);
        let n = 26usize;
        let xs: Vec<f64> = (0..n).map(|_| rng.next_f64() * 20.0 - 10.0).collect();
        let ys: Vec<f64> = (0..n).map(|_| rng.next_f64() * 20.0 - 10.0).collect();
        let l64 = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
        assert_delaunay_empty_circumcircle_oracle(&format!("rand_f64_{seed}"), &l64, 1e-10, true);

        let l32 = Layout2d::from_xy(
            xs.iter().map(|&v| v as f32).collect(),
            ys.iter().map(|&v| v as f32).collect(),
        )
        .unwrap();
        assert_delaunay_empty_circumcircle_oracle(&format!("rand_f32_{seed}"), &l32, 1e-10, true);
    }

    // 2. Cocircular regular k-gons (k = 4, 5, 8, 12) and 4x4 integer lattice grid.
    for &k in &[4usize, 5, 8, 12] {
        let xs: Vec<f64> = (0..k).map(|i| (i as f64 * TAU / k as f64).cos()).collect();
        let ys: Vec<f64> = (0..k).map(|i| (i as f64 * TAU / k as f64).sin()).collect();
        let poly = Layout2d::from_xy(xs, ys).unwrap();
        assert_delaunay_empty_circumcircle_oracle(&format!("ngon_{k}"), &poly, 1e-10, false);
    }
    let mut gx = Vec::new();
    let mut gy = Vec::new();
    for r in 0..4 {
        for c in 0..4 {
            gx.push(c as f64);
            gy.push(r as f64);
        }
    }
    let grid = Layout2d::from_xy(gx, gy).unwrap();
    assert_delaunay_empty_circumcircle_oracle("grid_4x4", &grid, 1e-10, false);

    // 3. Collinear boundary + interior points.
    let mut cx = vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 4.0, 4.0];
    let mut cy = vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 2.0, 4.0];
    cx.extend([1.5, 2.5, 2.0]);
    cy.extend([1.5, 1.5, 3.0]);
    let col_int = Layout2d::from_xy(cx, cy).unwrap();
    assert_delaunay_empty_circumcircle_oracle("collinear_plus_interior", &col_int, 1e-10, false);

    // 4. Duplicate sites + shifted large-offset coordinates (1e4 + dx).
    let dup_shifted = Layout2d::from_xy(
        vec![
            1e4 + 0.0,
            1e4 + 0.0,
            1e4 + 2.0,
            1e4 + 2.0,
            1e4 + 1.0,
            1e4 + 0.5,
            1e4 + 1.5,
            1e4 + 1.5,
        ],
        vec![
            1e4 + 0.0,
            1e4 + 0.0,
            1e4 + 0.0,
            1e4 + 2.0,
            1e4 + 2.0,
            1e4 + 0.7,
            1e4 + 1.2,
            1e4 + 1.2,
        ],
    )
    .unwrap();
    assert_delaunay_empty_circumcircle_oracle("dup_shifted_f64", &dup_shifted, 1e-10, false);

    let dup_shifted_f32 = Layout2d::from_xy(
        dup_shifted.x().iter().map(|&v| v as f32).collect(),
        dup_shifted.y().iter().map(|&v| v as f32).collect(),
    )
    .unwrap();
    assert_delaunay_empty_circumcircle_oracle("dup_shifted_f32", &dup_shifted_f32, 1e-10, false);
}
