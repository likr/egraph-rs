use std::f64::consts::{FRAC_PI_2, PI, TAU};

use egraph_core::{CoreError, CsrGraph, DeterministicRng, Layout2d, Scalar};

use crate::build_undirected_adjacency;

use super::*;

#[test]
fn test_edge_angle_aspect_ratio_and_angular_resolution() {
    assert!((edge_angle(1.0, 0.0, 0.0, 1.0).unwrap() - FRAC_PI_2).abs() < 1e-12);
    assert!(edge_angle(1.0_f64, 1e-8, 1.0, 1e-8).unwrap().abs() < 1e-12);
    assert!((edge_angle(1.0, 0.0, -1.0, 0.0).unwrap() - PI).abs() < 1e-12);
    assert!(edge_angle(0.0, 0.0, 1.0, 0.0).is_none());
    assert!(edge_angle(f64::NAN, 0.0, 1.0, 0.0).is_none());

    let square = Layout2d::from_xy(vec![0.0, 1.0, 0.0, 1.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    assert!((aspect_ratio(&square).unwrap() - 1.0).abs() < 1e-12);

    let collinear = Layout2d::from_xy(vec![0.0, 1.0, 2.0], vec![0.0, 0.0, 0.0]).unwrap();
    assert_eq!(aspect_ratio(&collinear).unwrap(), 0.0);

    let star =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (0, 2), (0, 3)], None, false, false).unwrap();
    let angles_star = Layout2d::from_xy(
        vec![0.0, 1.0, -0.5, -0.5],
        vec![0.0, 0.0, 0.5 * 3.0_f64.sqrt(), -0.5 * 3.0_f64.sqrt()],
    )
    .unwrap();
    let expected_ang = 3.0 * (-2.0 * PI / 3.0).exp();
    assert!((angular_resolution(&star, &angles_star).unwrap() - expected_ang).abs() < 1e-12);
}

#[test]
fn test_ideal_edge_lengths_unit_weighted_and_distances() {
    let g_unw = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    let l = Layout2d::from_xy(vec![0.0, 2.0, 5.0], vec![0.0, 0.0, 0.0]).unwrap();
    assert!((ideal_edge_lengths(&g_unw, &l).unwrap() - 5.0).abs() < 1e-12);
    assert!((ideal_edge_lengths_with_unit(&g_unw, &l, 2.5).unwrap() - 0.08).abs() < 1e-12);

    let g_w =
        CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], Some(&[2.0, 3.0]), false, false).unwrap();
    assert!(ideal_edge_lengths(&g_w, &l).unwrap().abs() < 1e-12);
    assert!(ideal_edge_lengths_scaled(&g_w, &l, 1.0).unwrap().abs() < 1e-12);

    let l_scaled = Layout2d::from_xy(vec![0.0, 5.0, 12.5], vec![0.0, 0.0, 0.0]).unwrap();
    assert!(
        ideal_edge_lengths_scaled(&g_w, &l_scaled, 2.5)
            .unwrap()
            .abs()
            < 1e-12
    );
    assert!((ideal_edge_lengths_scaled(&g_unw, &l, 2.5).unwrap() - 0.08).abs() < 1e-12);

    let dm = vec![0.0, 2.0, 5.0, 2.0, 0.0, 3.0, 5.0, 3.0, 0.0];
    assert!(
        ideal_edge_lengths_from_distances(&g_unw, &l, &dm)
            .unwrap()
            .abs()
            < 1e-12
    );

    assert!(matches!(
        ideal_edge_lengths_with_unit(&g_unw, &l, 0.0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        ideal_edge_lengths_scaled(&g_w, &l, 0.0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        ideal_edge_lengths_scaled(&g_w, &l, f64::NAN),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_prism_distortion_similarity_invariance_and_perturbation() {
    let orig =
        Layout2d::from_xy(vec![0.0, 1.0, 0.0, 1.0, 0.5], vec![0.0, 0.0, 1.0, 1.0, 1.6]).unwrap();

    // Rigid similarity transform (scale 3.5, rotation 0.7 rad, translation (12, -4)).
    let theta = 0.7_f64;
    let (st, ct) = theta.sin_cos();
    let scale = 3.5_f64;
    let adj_x: Vec<f64> = (0..orig.len())
        .map(|i| scale * (ct * orig.x()[i] - st * orig.y()[i]) + 12.0)
        .collect();
    let adj_y: Vec<f64> = (0..orig.len())
        .map(|i| scale * (st * orig.x()[i] + ct * orig.y()[i]) - 4.0)
        .collect();
    let similar = Layout2d::from_xy(adj_x, adj_y).unwrap();

    let d_sim = prism_distortion(&orig, &similar).unwrap();
    assert!(d_sim.sigma_dist < 1e-12, "sigma_dist={}", d_sim.sigma_dist);
    assert!(d_sim.sigma_disp < 1e-12, "sigma_disp={}", d_sim.sigma_disp);

    // Non-uniform distortion produces positive dispersion & Procrustes disparity.
    let warped =
        Layout2d::from_xy(vec![0.0, 3.0, 0.0, 1.0, 0.5], vec![0.0, 0.0, 1.0, 2.5, 2.0]).unwrap();
    let d_warp = prism_distortion(&orig, &warped).unwrap();
    assert!(d_warp.sigma_dist > 0.05);
    assert!(d_warp.sigma_disp > 0.01 && d_warp.sigma_disp <= 1.0);

    // Collapsed adjusted layout gives finite sigma_dist = 0.0 and sigma_disp = 1.0.
    let collapsed = Layout2d::zeros(orig.len());
    let d_col = prism_distortion(&orig, &collapsed).unwrap();
    assert_eq!(d_col.sigma_dist, 0.0);
    assert_eq!(d_col.sigma_disp, 1.0);
}

fn brute_minimum_angle_oracle<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> (f64, f64) {
    let n = layout.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let adj = build_undirected_adjacency(graph);
    let xs = layout.x();
    let ys = layout.y();
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    for u in 0..n {
        let d = adj[u].len();
        if d < 2 {
            s1 += 1.0;
            s2 += 1.0;
            continue;
        }
        let ux = xs[u].to_f64();
        let uy = ys[u].to_f64();
        let mut has_zero = false;
        let mut vecs = Vec::with_capacity(d);
        for &v in &adj[u] {
            let dx = xs[v as usize].to_f64() - ux;
            let dy = ys[v as usize].to_f64() - uy;
            if dx == 0.0 && dy == 0.0 {
                has_zero = true;
                vecs.push((1.0_f64, 0.0_f64));
            } else {
                vecs.push((dx, dy));
            }
        }
        let mut min_theta = TAU;
        for i in 0..d {
            for j in (i + 1)..d {
                let ang = edge_angle(vecs[i].0, vecs[i].1, vecs[j].0, vecs[j].1).unwrap();
                if ang < min_theta {
                    min_theta = ang;
                }
            }
        }
        let ideal = TAU / (d as f64);
        let score = (1.0 - (ideal - min_theta).abs() / ideal).clamp(0.0, 1.0);
        s1 += score;
        if !has_zero {
            s2 += score;
        }
    }
    (s1 / n as f64, s2 / n as f64)
}

#[test]
fn test_minimum_angle_brute_force_oracle_branch_cut_and_zero_edges() {
    // Empty graph -> (0.0, 0.0).
    let empty_g = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let empty_l = Layout2d::<f64>::zeros(0);
    assert_eq!(
        minimum_angle_summary(&empty_g, &empty_l).unwrap(),
        MinimumAngleSummary { a1: 0.0, a2: 0.0 }
    );
    assert_eq!(minimum_angle(&empty_g, &empty_l).unwrap(), 0.0);

    // Regular 3-star: ideal 120 deg at center, deg 1 leaves -> both A1 and A2 = 1.0.
    let star =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (0, 2), (0, 3)], None, false, false).unwrap();
    let star_l = Layout2d::from_xy(
        vec![0.0, 1.0, -0.5, -0.5],
        vec![0.0, 0.0, 0.5 * 3.0_f64.sqrt(), -0.5 * 3.0_f64.sqrt()],
    )
    .unwrap();
    let s_star = minimum_angle_summary(&star, &star_l).unwrap();
    assert!((s_star.a1 - 1.0).abs() < 1e-12);
    assert!((s_star.a2 - 1.0).abs() < 1e-12);

    // Branch cut around -pi / +pi: center 0 with neighbors near (-1, +eps) and (-1, -eps).
    let branch_g =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (0, 2), (0, 3)], None, false, false).unwrap();
    let branch_l =
        Layout2d::from_xy(vec![0.0, -1.0, -1.0, 1.0], vec![0.0, 1e-4, -1e-4, 0.0]).unwrap();
    let s_branch = minimum_angle_summary(&branch_g, &branch_l).unwrap();
    let (o1, o2) = brute_minimum_angle_oracle(&branch_g, &branch_l);
    assert!((s_branch.a1 - o1).abs() < 1e-12);
    assert!((s_branch.a2 - o2).abs() < 1e-12);

    // Zero-length edge divergence between A1 (atan2(0,0)=0) and A2 (score=0.0).
    let zero_l = Layout2d::from_xy(vec![0.0, 0.0, 0.0, -1.0], vec![0.0, 0.0, 1.0, 0.0]).unwrap();
    let s_zero = minimum_angle_summary(&star, &zero_l).unwrap();
    let (z1, z2) = brute_minimum_angle_oracle(&star, &zero_l);
    assert!((s_zero.a1 - z1).abs() < 1e-12);
    assert!((s_zero.a2 - z2).abs() < 1e-12);
    assert!(s_zero.a1 > s_zero.a2);
    assert_eq!(minimum_angle(&star, &zero_l).unwrap(), s_zero.a2);

    // Multi-seed random graphs (directed + undirected, f32 + f64) vs O(sum d_v^2) oracle.
    for seed in 1..=20u64 {
        let mut rng = DeterministicRng::from_seed_u64(500 + seed);
        let n = 16usize;
        let directed = seed % 2 == 0;
        let mut edges = Vec::new();
        for _ in 0..32 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            edges.push((u, v));
        }
        let g = CsrGraph::<f64>::from_edges(n, &edges, None, directed, true).unwrap();
        let xs: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
        let ys: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
        let l64 = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
        let l32 = Layout2d::from_xy(
            xs.iter().map(|&v| v as f32).collect(),
            ys.iter().map(|&v| v as f32).collect(),
        )
        .unwrap();

        let s64 = minimum_angle_summary(&g, &l64).unwrap();
        let (b64_1, b64_2) = brute_minimum_angle_oracle(&g, &l64);
        assert!((s64.a1 - b64_1).abs() < 1e-12);
        assert!((s64.a2 - b64_2).abs() < 1e-12);

        let s32 = minimum_angle_summary(&g, &l32).unwrap();
        let (b32_1, b32_2) = brute_minimum_angle_oracle(&g, &l32);
        assert!((s32.a1 - b32_1).abs() < 1e-12);
        assert!((s32.a2 - b32_2).abs() < 1e-12);
    }
}

#[test]
fn test_layout_degeneracy_brute_force_oracle_and_edge_cases() {
    // 1. Canonicalization of -0.0 -> +0.0 and f32 vs f64 collision divergence.
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
    // Nodes 0 and 1 are (+0.0, -0.0) and (-0.0, +0.0) -> identical in both f32 and f64.
    // Nodes 2 and 3 differ by 1e-12 -> distinct in f64, identical in f32!
    let l = Layout2d::from_xy(
        vec![0.0_f64, -0.0_f64, 1.0, 1.0 + 1e-12],
        vec![-0.0_f64, 0.0_f64, 1.0, 1.0],
    )
    .unwrap();
    let deg = layout_degeneracy(&g, &l).unwrap();
    assert_eq!(deg.num_nodes, 4);
    assert_eq!(deg.num_edges, 3);
    assert_eq!(deg.duplicate_nodes_f32, 4);
    assert!((deg.duplicate_ratio_f32 - 1.0).abs() < 1e-12);
    assert_eq!(deg.duplicate_nodes_f64, 2);
    assert!((deg.duplicate_ratio_f64 - 0.5).abs() < 1e-12);
    assert_eq!(deg.zero_length_edges, 1);
    assert!((deg.zero_length_edge_ratio - (1.0 / 3.0)).abs() < 1e-12);
    assert!(!deg.is_bbox_degenerate);

    // 2. 1D collinear layout: bbox_height == 0 -> is_bbox_degenerate = true.
    let line = Layout2d::from_xy(vec![0.0, 1.0, 2.0, 3.0], vec![0.0; 4]).unwrap();
    let deg_line = layout_degeneracy(&g, &line).unwrap();
    assert_eq!(deg_line.duplicate_nodes_f32, 0);
    assert_eq!(deg_line.bbox_width, 3.0);
    assert_eq!(deg_line.bbox_height, 0.0);
    assert!(deg_line.is_bbox_degenerate);

    // 3. Empty graph -> zeros, not degenerate.
    let empty_g = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let empty_l = Layout2d::<f64>::zeros(0);
    let deg_empty = layout_degeneracy(&empty_g, &empty_l).unwrap();
    assert_eq!(deg_empty.num_nodes, 0);
    assert_eq!(deg_empty.duplicate_ratio_f32, 0.0);
    assert!(!deg_empty.is_bbox_degenerate);

    // 4. Multi-seed O(N^2) brute-force oracle for duplicate_nodes_f32/f64 and zero_length_edges.
    for seed in 1..=15u64 {
        let mut rng = DeterministicRng::from_seed_u64(600 + seed);
        let n = 25usize;
        let mut edges = Vec::new();
        for _ in 0..40 {
            edges.push((rng.gen_range_usize(n) as u32, rng.gen_range_usize(n) as u32));
        }
        let rg = CsrGraph::<f64>::from_edges(n, &edges, None, false, true).unwrap();
        let xs: Vec<f64> = (0..n)
            .map(|_| {
                let v = rng.gen_range_usize(5) as f64 - 2.0;
                if v == 0.0 && rng.gen_range_usize(2) == 0 {
                    -0.0
                } else {
                    v
                }
            })
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|_| (rng.gen_range_usize(5) as f64) * 0.5)
            .collect();
        let rl = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
        let got = layout_degeneracy(&rg, &rl).unwrap();

        let mut brute_dup64 = 0usize;
        let mut brute_dup32 = 0usize;
        for i in 0..n {
            let has64 = (0..n).any(|j| j != i && xs[i] == xs[j] && ys[i] == ys[j]);
            let has32 = (0..n).any(|j| {
                j != i && (xs[i] as f32) == (xs[j] as f32) && (ys[i] as f32) == (ys[j] as f32)
            });
            if has64 {
                brute_dup64 += 1;
            }
            if has32 {
                brute_dup32 += 1;
            }
        }
        let uedges = extract_unique_edges(&rg);
        let brute_zero = uedges
            .iter()
            .filter(|&&(u, v)| xs[u as usize] == xs[v as usize] && ys[u as usize] == ys[v as usize])
            .count();
        assert_eq!(got.duplicate_nodes_f64, brute_dup64);
        assert_eq!(got.duplicate_nodes_f32, brute_dup32);
        assert_eq!(got.zero_length_edges, brute_zero);
    }
}
