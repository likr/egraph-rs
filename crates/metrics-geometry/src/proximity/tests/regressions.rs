use std::cmp::Ordering;
use std::f64::consts::TAU;

use egraph_core::{CoreError, CsrGraph, DeterministicRng, Layout2d};

use super::super::exact_predicates::{cross_abs_cmp, euclidean_diameter_2d, orient2d};
use super::super::*;
use super::{brute_gg_edges, brute_node_resolution, brute_rng_edges};

#[test]
fn test_regression_rotated_collinear_and_near_collinear_diameter() {
    // Reproduces rotated collinear & near-collinear layouts (n > 64) across angles
    // including `(0.6, 0.8)` at n=90, `angle = 5.0959_f64` at n=80, n=200, and 50 random angles.
    let mut xs_0608 = Vec::with_capacity(90);
    let mut ys_0608 = Vec::with_capacity(90);
    let mut rng0 = DeterministicRng::from_seed_u64(43);
    for _ in 0..90 {
        let t = (rng0.next_f64() - 0.5) * 50.0;
        xs_0608.push(t * 0.6);
        ys_0608.push(t * 0.8);
    }
    let l_0608 = Layout2d::from_xy(xs_0608, ys_0608).unwrap();
    assert!((node_resolution(&l_0608).unwrap() - brute_node_resolution(&l_0608)).abs() < 1e-10);

    let mut angles = vec![5.0959_f64, 0.37_f64, 1.2345_f64, 2.468_f64];
    let mut rng_a = DeterministicRng::from_seed_u64(777);
    for _ in 0..46 {
        angles.push(rng_a.next_f64() * TAU);
    }

    for &eps in &[0.0_f64, 1e-15_f64] {
        for (k, &angle) in angles.iter().enumerate() {
            let (sa, ca) = angle.sin_cos();
            for &n in &[80usize, 200usize] {
                if n == 200 && k >= 6 {
                    continue;
                }
                let mut rng = DeterministicRng::from_seed_u64(5000 + k as u64);
                let mut xs = Vec::with_capacity(n);
                let mut ys = Vec::with_capacity(n);
                for _ in 0..n {
                    let t = (rng.next_f64() - 0.5) * 100.0;
                    let perp = (rng.next_f64() - 0.5) * 2.0 * eps;
                    xs.push(t * ca - perp * sa);
                    ys.push(t * sa + perp * ca);
                }
                let layout = Layout2d::from_xy(xs, ys).unwrap();
                let got_diam = euclidean_diameter_2d(&layout);
                let mut exp_sq = 0.0_f64;
                for i in 0..n {
                    for j in (i + 1)..n {
                        let dx = layout.x()[i] - layout.x()[j];
                        let dy = layout.y()[i] - layout.y()[j];
                        exp_sq = exp_sq.max(dx * dx + dy * dy);
                    }
                }
                let exp_diam = exp_sq.sqrt();
                assert!(
                    (got_diam - exp_diam).abs() <= 1e-11 * exp_diam.max(1.0),
                    "diameter mismatch at angle={angle}, eps={eps}, n={n}: got={got_diam} exp={exp_diam}"
                );
                let got_nr = node_resolution(&layout).unwrap();
                let exp_nr = brute_node_resolution(&layout);
                assert!(
                    (got_nr - exp_nr).abs() <= 1e-9 * exp_nr.max(1.0),
                    "node_resolution mismatch at angle={angle}, eps={eps}, n={n}: got={got_nr} exp={exp_nr}"
                );
            }
        }
    }

    // Also test > 64 hull vertices on a rotated thin convex ellipse (aspect ratio 1e13)
    // to directly exercise the `m > 64` rotating calipers path with near-collinear jitter.
    let (sa, ca) = (5.0959_f64).sin_cos();
    let mut arc_x = Vec::with_capacity(200);
    let mut arc_y = Vec::with_capacity(200);
    for idx in 0..200 {
        let theta = (idx as f64) * TAU / 200.0;
        let lx = 50.0 * theta.cos();
        let ly = 1e-12 * theta.sin();
        arc_x.push(lx * ca - ly * sa);
        arc_y.push(lx * sa + ly * ca);
    }
    let arc_layout = Layout2d::from_xy(arc_x, arc_y).unwrap();
    let got_arc_diam = euclidean_diameter_2d(&arc_layout);
    let mut exp_arc_sq = 0.0_f64;
    for i in 0..200 {
        for j in (i + 1)..200 {
            let dx = arc_layout.x()[i] - arc_layout.x()[j];
            let dy = arc_layout.y()[i] - arc_layout.y()[j];
            exp_arc_sq = exp_arc_sq.max(dx * dx + dy * dy);
        }
    }
    let exp_arc_diam = exp_arc_sq.sqrt();
    assert!((got_arc_diam - exp_arc_diam).abs() <= 1e-11 * exp_arc_diam);
    let got_arc_nr = node_resolution(&arc_layout).unwrap();
    let exp_arc_nr = brute_node_resolution(&arc_layout);
    assert!((got_arc_nr - exp_arc_nr).abs() <= 1e-9 * exp_arc_nr.max(1.0));

    // Direct checks on exact geometric predicates (collinear, parallel, huge/tiny dynamic range).
    assert_eq!(
        orient2d([0.0, 0.0], [7.0, 3.0], [14.0, 6.0]),
        Ordering::Equal
    );
    assert_eq!(
        orient2d([0.0, 0.0], [1e-305, 1e-305], [2e-305, 2e-305]),
        Ordering::Equal
    );
    assert_eq!(
        cross_abs_cmp([0.0, 0.0], [10.0, 0.0], [5.0, 2.0], [8.0, -2.0]),
        Ordering::Equal
    );
    assert_eq!(
        cross_abs_cmp(
            [0.0, 0.0],
            [1e150, 1e150],
            [2e150, 2e150 + 1e135],
            [3e150, 3e150]
        ),
        Ordering::Greater
    );

    // n = 80 at 1e200 must never panic or misbehave.
    let mut rng_huge = DeterministicRng::from_seed_u64(9001);
    let huge_x: Vec<f64> = (0..80)
        .map(|_| (rng_huge.next_f64() - 0.5) * 2e200)
        .collect();
    let huge_y: Vec<f64> = (0..80)
        .map(|_| (rng_huge.next_f64() - 0.5) * 2e200)
        .collect();
    let huge_layout = Layout2d::from_xy(huge_x, huge_y).unwrap();
    let huge_diam = euclidean_diameter_2d(&huge_layout);
    assert!(huge_diam.is_finite() && huge_diam > 0.0);
    assert!(matches!(
        node_resolution(&huge_layout),
        Ok(_) | Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_regression_rng_tiny_witness_sector_pruning_soundness() {
    // Reproduces near_cocircular (N = 7) in [0, 2] x [-1e-14, 1 + 1e-10].
    let l_cocirc = Layout2d::from_xy(
        vec![0.0, 2.0, 1.0, 1.0 + 1e-14, 0.0 + 1e-14, 2.0 - 1e-14, 1.0],
        vec![0.0, 0.0, 1.0 + 1e-10, 1.0, -1e-14, 1e-14, 0.0],
    )
    .unwrap();
    let got_cocirc = relative_neighborhood_edges(&l_cocirc).unwrap();
    let exp_cocirc = brute_rng_edges(&l_cocirc);
    assert_eq!(got_cocirc, exp_cocirc);

    let g_cocirc =
        CsrGraph::<f64>::from_edges(7, &[(0, 1), (1, 2), (2, 0), (3, 6)], None, false, false)
            .unwrap();
    assert_eq!(
        relative_neighborhood_preservation(&g_cocirc, &l_cocirc)
            .unwrap()
            .to_bits(),
        mean_jaccard_graph_vs_edges(&g_cocirc, &exp_cocirc).to_bits()
    );

    // Reproduces 1e-17 tight cluster (17 points) around (0, 0) + distant point (1.0, 0.0).
    let mut rng = DeterministicRng::from_seed_u64(31415);
    let mut cx = Vec::with_capacity(18);
    let mut cy = Vec::with_capacity(18);
    for _ in 0..17 {
        cx.push((rng.next_f64() - 0.5) * 1e-17);
        cy.push((rng.next_f64() - 0.5) * 1e-17);
    }
    cx.push(1.0);
    cy.push(0.0);
    let l_cluster18 = Layout2d::from_xy(cx, cy).unwrap();
    let got_cluster18 = relative_neighborhood_edges(&l_cluster18).unwrap();
    let exp_cluster18 = brute_rng_edges(&l_cluster18);
    assert_eq!(got_cluster18, exp_cluster18);

    // Also test > 128 points in a 1e-17 tight cluster + distant point so k = 128 witnesses
    // stay entirely inside the tight cluster.
    let mut cx2 = Vec::with_capacity(141);
    let mut cy2 = Vec::with_capacity(141);
    for _ in 0..140 {
        cx2.push((rng.next_f64() - 0.5) * 1e-17);
        cy2.push((rng.next_f64() - 0.5) * 1e-17);
    }
    cx2.push(1.0);
    cy2.push(0.0);
    let l_cluster141 = Layout2d::from_xy(cx2, cy2).unwrap();
    assert_eq!(
        relative_neighborhood_edges(&l_cluster141).unwrap(),
        brute_rng_edges(&l_cluster141)
    );
}

#[test]
fn test_regression_gg_large_offset_midpoint_cancellation() {
    // Reproduces 1-ULP-spaced cluster in [1e10, 1e10 + 1e-5]^2 (n = 40 and n = 80).
    for &n in &[40usize, 80usize] {
        let mut rng = DeterministicRng::from_seed_u64(20260 + n as u64);
        let xs: Vec<f64> = (0..n).map(|_| 1e10 + rng.next_f64() * 1e-5).collect();
        let ys: Vec<f64> = (0..n).map(|_| 1e10 + rng.next_f64() * 1e-5).collect();
        let layout = Layout2d::from_xy(xs, ys).unwrap();

        let got_gg = gabriel_graph_edges(&layout).unwrap();
        let exp_gg = brute_gg_edges(&layout);
        assert_eq!(got_gg, exp_gg, "GG edges mismatch at 1e10 offset for n={n}");

        let got_rng = relative_neighborhood_edges(&layout).unwrap();
        let exp_rng = brute_rng_edges(&layout);
        assert_eq!(
            got_rng, exp_rng,
            "RNG edges mismatch at 1e10 offset for n={n}"
        );

        let edges: Vec<(u32, u32)> = (0..(n as u32 - 1)).map(|i| (i, i + 1)).collect();
        let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
        assert_eq!(
            gabriel_graph_preservation(&g, &layout).unwrap().to_bits(),
            mean_jaccard_graph_vs_edges(&g, &exp_gg).to_bits()
        );
        assert_eq!(
            relative_neighborhood_preservation(&g, &layout)
                .unwrap()
                .to_bits(),
            mean_jaccard_graph_vs_edges(&g, &exp_rng).to_bits()
        );
    }
}
