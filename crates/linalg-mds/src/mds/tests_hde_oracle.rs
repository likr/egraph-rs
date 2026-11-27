use egraph_core::{CsrGraph, Layout2d, LayoutGenerator, LayoutNd};
use egraph_linalg::{select_maxmin_sp_pivots, symmetric_eigen_jacobi};

use super::tests_mds::isotropic_std;
use super::*;

pub(super) fn isotropic_std_nd(layout: &LayoutNd<f64>) -> f64 {
    let n = layout.num_nodes();
    let d = layout.dims();
    let coords = layout.coords();
    let mut means = vec![0.0_f64; d];
    for i in 0..n {
        for m in 0..d {
            means[m] += coords[i * d + m];
        }
    }
    for mean in &mut means {
        *mean /= n as f64;
    }
    let mut sum_sq = 0.0_f64;
    for i in 0..n {
        for m in 0..d {
            let diff = coords[i * d + m] - means[m];
            sum_sq += diff * diff;
        }
    }
    (sum_sq / ((d * n) as f64)).sqrt()
}

/// Independent brute-force Floyd-Warshall + O(m^2 n) covariance PCA oracle for HDE.
pub(super) fn hde_naive_oracle(
    graph: &CsrGraph<f64>,
    pivots: &[u32],
    output_dims: usize,
) -> Vec<Vec<f64>> {
    let n = graph.num_nodes();
    let mut apsp = vec![vec![f64::INFINITY; n]; n];
    for (u, apsp_row) in apsp.iter_mut().enumerate() {
        apsp_row[u] = 0.0;
        if let Some(ws) = graph.neighbor_weights(u) {
            for (&v, &w) in graph.neighbors(u).iter().zip(ws.iter()) {
                let vi = v as usize;
                if w < apsp_row[vi] {
                    apsp_row[vi] = w;
                }
            }
        } else {
            for &v in graph.neighbors(u) {
                let vi = v as usize;
                if 1.0 < apsp_row[vi] {
                    apsp_row[vi] = 1.0;
                }
            }
        }
    }
    for k in 0..n {
        for i in 0..n {
            let dik = apsp[i][k];
            if !dik.is_finite() {
                continue;
            }
            for j in 0..n {
                let cand = dik + apsp[k][j];
                if cand < apsp[i][j] {
                    apsp[i][j] = cand;
                }
            }
        }
    }

    // Deduplicate pivots preserving order.
    let mut uniq = Vec::new();
    for &p in pivots {
        if !uniq.contains(&p) {
            uniq.push(p);
        }
    }
    let m = uniq.len();
    let mut x_hat = vec![vec![0.0_f64; n]; m];
    for (i, &p) in uniq.iter().enumerate() {
        let row = &apsp[p as usize];
        let mut max_fin = 0.0_f64;
        let mut has_fin = false;
        let mut has_inf = false;
        for &d in row {
            if d.is_finite() {
                has_fin = true;
                max_fin = max_fin.max(d);
            } else {
                has_inf = true;
            }
        }
        let fill = if has_fin { max_fin + 1.0 } else { 0.0 };
        let mut sum = 0.0;
        for v in 0..n {
            let val = if has_inf && !row[v].is_finite() {
                fill
            } else {
                row[v]
            };
            x_hat[i][v] = val;
            sum += val;
        }
        let mean = sum / (n as f64);
        for val in &mut x_hat[i] {
            *val -= mean;
        }
    }

    let mut cov = vec![0.0_f64; m * m];
    for a in 0..m {
        for b in 0..m {
            let mut s = 0.0;
            for v in 0..n {
                s += x_hat[a][v] * x_hat[b][v];
            }
            cov[a * m + b] = s / (n as f64);
        }
    }
    let (evals, evecs) = symmetric_eigen_jacobi(&cov, m, 200, 1e-14).unwrap();
    let used = output_dims.min(m);
    let lam0 = evals[m - 1];
    let mut y = vec![vec![0.0_f64; output_dims]; n];
    if lam0 > 1e-300 {
        for d in 0..used {
            let idx = m - 1 - d;
            if d >= 1 && evals[idx] <= 1e-12 * lam0 {
                continue;
            }
            for (v, y_row) in y.iter_mut().enumerate() {
                let mut acc = 0.0;
                for p in 0..m {
                    acc += x_hat[p][v] * evecs[p * m + idx];
                }
                y_row[d] = acc;
            }
        }
    }
    y
}

fn axis_cos_abs(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (na * nb)).abs()
    }
}

pub(super) fn pairwise_euclidean_nd(coords: &[Vec<f64>]) -> Vec<f64> {
    let n = coords.len();
    let mut out = Vec::with_capacity(n * (n.saturating_sub(1)) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut max_abs = 0.0_f64;
            for (&a, &b) in coords[i].iter().zip(coords[j].iter()) {
                let d = (a - b).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
            if max_abs == 0.0 || !max_abs.is_finite() {
                out.push(max_abs);
                continue;
            }
            let (pre, inv) = if max_abs < 1e-300 {
                (1e150_f64, 1.0 / (max_abs * 1e150_f64))
            } else if max_abs > 1e300 {
                (1e-150_f64, 1.0 / (max_abs * 1e-150_f64))
            } else {
                (1.0_f64, 1.0 / max_abs)
            };
            let d2: f64 = coords[i]
                .iter()
                .zip(coords[j].iter())
                .map(|(&a, &b)| {
                    let d = ((a - b) * pre) * inv;
                    d * d
                })
                .sum();
            out.push(max_abs * d2.sqrt());
        }
    }
    out
}

#[test]
fn test_hde_dense_naive_oracle() {
    // 1. Asymmetric unweighted 10-node graph with irregular chords
    let asym_edges = vec![
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (0, 3),
        (1, 5),
        (2, 8),
        (4, 9),
        (0, 7),
    ];
    let g_unw = CsrGraph::<f64>::from_edges(10, &asym_edges, None, false, false).unwrap();
    let pivots = vec![0, 4, 7, 2];
    let cfg = HdeConfig::default()
        .with_dimensions(4)
        .with_output_dims(Some(2))
        .with_seed(42)
        .with_tol(1e-12)
        .with_max_sweeps(200)
        .with_normalize(false);
    let lay: Layout2d<f64> = Hde::from_config(&g_unw, cfg)
        .with_pivots(&pivots)
        .generate()
        .unwrap();
    let oracle = hde_naive_oracle(&g_unw, &pivots, 2);
    let ox: Vec<f64> = oracle.iter().map(|r| r[0]).collect();
    let oy: Vec<f64> = oracle.iter().map(|r| r[1]).collect();
    assert!(axis_cos_abs(lay.x(), &ox) > 1.0 - 1e-12);
    assert!(axis_cos_abs(lay.y(), &oy) > 1.0 - 1e-12);
    let lay_coords: Vec<Vec<f64>> = (0..10).map(|i| vec![lay.x()[i], lay.y()[i]]).collect();
    for (a, b) in pairwise_euclidean_nd(&lay_coords)
        .iter()
        .zip(pairwise_euclidean_nd(&oracle).iter())
    {
        assert!((a - b).abs() < 1e-10, "pairwise mismatch: {a} vs {b}");
    }

    // 2. Asymmetric positive weighted graph
    let w_edges = vec![
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (0, 4),
        (1, 6),
        (2, 5),
        (3, 7),
    ];
    let weights = vec![0.5, 1.3, 2.7, 0.9, 3.4, 1.1, 2.2, 4.0, 1.7, 0.8, 2.9];
    let g_w = CsrGraph::<f64>::from_edges(8, &w_edges, Some(&weights), false, false).unwrap();
    let w_pivots = select_maxmin_sp_pivots(&g_w, 5, 11, 1).unwrap();
    let cfg_w = HdeConfig::default()
        .with_dimensions(5)
        .with_seed(11)
        .with_tol(1e-12)
        .with_max_sweeps(200)
        .with_normalize(false);
    let lay_w: Layout2d<f64> = Hde::from_config(&g_w, cfg_w.clone()).generate().unwrap();
    let oracle_w = hde_naive_oracle(&g_w, &w_pivots, 2);
    let lay_w_coords: Vec<Vec<f64>> = (0..8).map(|i| vec![lay_w.x()[i], lay_w.y()[i]]).collect();
    for (a, b) in pairwise_euclidean_nd(&lay_w_coords)
        .iter()
        .zip(pairwise_euclidean_nd(&oracle_w).iter())
    {
        assert!((a - b).abs() < 1e-10, "weighted mismatch: {a} vs {b}");
    }

    // Normalized variant retains finite std=1.0
    let lay_w_norm: Layout2d<f64> = Hde::from_config(&g_w, cfg_w.with_normalize(true))
        .generate()
        .unwrap();
    assert!((isotropic_std(&lay_w_norm) - 1.0).abs() < 1e-12);

    // 3. Symmetric 4x5 grid (pairwise distance invariant under possible basis rotation)
    let mut grid_edges = Vec::new();
    for r in 0..4 {
        for c in 0..5 {
            let u = (r * 5 + c) as u32;
            if c + 1 < 5 {
                grid_edges.push((u, u + 1));
            }
            if r + 1 < 4 {
                grid_edges.push((u, u + 5));
            }
        }
    }
    let grid = CsrGraph::<f64>::from_edges(20, &grid_edges, None, false, false).unwrap();
    let gp = select_maxmin_sp_pivots(&grid, 8, 42, 1).unwrap();
    let grid_cfg = HdeConfig::default()
        .with_dimensions(8)
        .with_tol(1e-12)
        .with_max_sweeps(200)
        .with_normalize(false);
    let grid_lay: Layout2d<f64> = Hde::from_config(&grid, grid_cfg).generate().unwrap();
    let grid_oracle = hde_naive_oracle(&grid, &gp, 2);
    let grid_coords: Vec<Vec<f64>> = (0..20)
        .map(|i| vec![grid_lay.x()[i], grid_lay.y()[i]])
        .collect();
    for (a, b) in pairwise_euclidean_nd(&grid_coords)
        .iter()
        .zip(pairwise_euclidean_nd(&grid_oracle).iter())
    {
        assert!((a - b).abs() < 1e-10, "grid mismatch: {a} vs {b}");
    }
}

#[test]
fn test_hde_topologies_path_tree_grid_cycle() {
    // 1. Path P_8: rank-1 -> y identically 0.0 for both normalize=false and normalize=true
    let p8_edges: Vec<(u32, u32)> = (0..7).map(|i| (i, i + 1)).collect();
    let p8 = CsrGraph::<f64>::from_edges(8, &p8_edges, None, false, false).unwrap();
    for normalize in [false, true] {
        let lay: Layout2d<f64> = Hde::new(&p8)
            .with_pivots(&[0, 7])
            .with_normalize(normalize)
            .generate()
            .unwrap();
        assert!(lay.y().iter().all(|&y| y == 0.0));
        // Strictly monotonic along X
        let increasing = lay.x().windows(2).all(|w| w[0] < w[1]);
        let decreasing = lay.x().windows(2).all(|w| w[0] > w[1]);
        assert!(increasing || decreasing);
        if normalize {
            assert!((isotropic_std(&lay) - 1.0).abs() < 1e-12);
        }
    }
    let p8_auto: Layout2d<f64> = Hde::new(&p8).generate().unwrap();
    assert!(p8_auto.is_all_finite());
    assert!((isotropic_std(&p8_auto) - 1.0).abs() < 1e-12);
    // With interior pivots (dimensions >= 8), |v - p| creates a U-arch along X; verify X ordering.
    let inc_auto = p8_auto.x().windows(2).all(|w| w[0] < w[1]);
    let dec_auto = p8_auto.x().windows(2).all(|w| w[0] > w[1]);
    assert!(inc_auto || dec_auto);

    // 2. Tree with pivot-free subtree (Harel & Koren §4 Fig. 8):
    // Edges: 0-1, 1-2, 1-3, 0-4, 4-5; pivots=[0, 4] -> leaves 2 and 3 collapse identically.
    let tree = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (1, 3), (0, 4), (4, 5)],
        None,
        false,
        false,
    )
    .unwrap();
    let tree_lay: Layout2d<f64> = Hde::new(&tree).with_pivots(&[0, 4]).generate().unwrap();
    assert!((tree_lay.x()[2] - tree_lay.x()[3]).abs() < 1e-12);
    assert!((tree_lay.y()[2] - tree_lay.y()[3]).abs() < 1e-12);

    // 3. Cycle C_12
    let c12_edges: Vec<(u32, u32)> = (0..12).map(|i| (i, (i + 1) % 12)).collect();
    let c12 = CsrGraph::<f64>::from_edges(12, &c12_edges, None, false, false).unwrap();
    let c12_lay: Layout2d<f64> = Hde::new(&c12).generate().unwrap();
    assert!(c12_lay.is_all_finite());
    assert!((isotropic_std(&c12_lay) - 1.0).abs() < 1e-12);
    let c12_unnorm: Layout2d<f64> = Hde::new(&c12)
        .with_tol(1e-12)
        .with_max_sweeps(200)
        .with_normalize(false)
        .generate()
        .unwrap();
    let c12_pivots = select_maxmin_sp_pivots(&c12, 50, 42, 1).unwrap();
    let c12_oracle = hde_naive_oracle(&c12, &c12_pivots, 2);
    let c12_coords: Vec<Vec<f64>> = (0..12)
        .map(|i| vec![c12_unnorm.x()[i], c12_unnorm.y()[i]])
        .collect();
    for (a, b) in pairwise_euclidean_nd(&c12_coords)
        .iter()
        .zip(pairwise_euclidean_nd(&c12_oracle).iter())
    {
        assert!((a - b).abs() < 1e-10);
    }

    // 4. 8x8 regular grid
    let mut grid_edges = Vec::new();
    for r in 0..8 {
        for c in 0..8 {
            let u = (r * 8 + c) as u32;
            if c + 1 < 8 {
                grid_edges.push((u, u + 1));
            }
            if r + 1 < 8 {
                grid_edges.push((u, u + 8));
            }
        }
    }
    let grid8 = CsrGraph::<f64>::from_edges(64, &grid_edges, None, false, false).unwrap();
    let g8_lay: Layout2d<f64> = Hde::new(&grid8).generate().unwrap();
    assert_eq!(g8_lay.len(), 64);
    assert!(g8_lay.is_all_finite());
    assert!((isotropic_std(&g8_lay) - 1.0).abs() < 1e-12);
    let mean_edge_len: f64 = grid_edges
        .iter()
        .map(|&(u, v)| {
            let ui = u as usize;
            let vi = v as usize;
            (g8_lay.x()[ui] - g8_lay.x()[vi]).hypot(g8_lay.y()[ui] - g8_lay.y()[vi])
        })
        .sum::<f64>()
        / (grid_edges.len() as f64);
    assert!(
        mean_edge_len < 0.7,
        "expected short grid edges, got {mean_edge_len}"
    );
}
