use egraph_core::CsrGraph;

use super::*;
use crate::amg_pcg::AmgPcgConfig;

/// Builds a finite `{3, 7}` hyperbolic triangulation ball of `layers` rings around vertex `0`.
/// For `layers = 5`, this produces `N = 617` vertices and `M = 1463` edges.
fn make_hyperbolic_3_7_tessellation(layers: usize) -> CsrGraph<f64> {
    if layers == 0 {
        return CsrGraph::from_edges(1, &[], None, false, false).unwrap();
    }
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut next_id = 1u32;
    let mut ring: Vec<(u32, usize)> = Vec::with_capacity(7);
    for _ in 0..7 {
        let v = next_id;
        next_id += 1;
        edges.push((0, v));
        ring.push((v, 1));
    }
    for i in 0..ring.len() {
        edges.push((ring[i].0, ring[(i + 1) % ring.len()].0));
    }

    for _layer in 2..=layers {
        let m = ring.len();
        let mut next_ring: Vec<(u32, usize)> = Vec::new();
        for i in 0..m {
            let (u, parents) = ring[i];
            let (u_next, _) = ring[(i + 1) % m];
            let interior = if parents == 1 { 2 } else { 1 };
            for _ in 0..interior {
                let c = next_id;
                next_id += 1;
                edges.push((u, c));
                next_ring.push((c, 1));
            }
            let shared = next_id;
            next_id += 1;
            edges.push((u, shared));
            edges.push((u_next, shared));
            next_ring.push((shared, 2));
        }
        for i in 0..next_ring.len() {
            edges.push((next_ring[i].0, next_ring[(i + 1) % next_ring.len()].0));
        }
        ring = next_ring;
    }

    CsrGraph::from_edges(next_id as usize, &edges, None, false, true).unwrap()
}

/// Builds a hyperbolic fan mesh with `sectors` branches of `branching`-ary expansion over `depth`
/// concentric ring layers (`N = 1 + sectors * (branching^depth - 1) / (branching - 1)`,
/// `M = 2 * N - 2`).
fn make_hyperbolic_fan_mesh(sectors: usize, branching: usize, depth: usize) -> CsrGraph<f64> {
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut next_id = 1u32;
    let mut layer: Vec<u32> = Vec::with_capacity(sectors);
    for _ in 0..sectors {
        let v = next_id;
        next_id += 1;
        edges.push((0, v));
        layer.push(v);
    }
    for i in 0..layer.len() {
        edges.push((layer[i], layer[(i + 1) % layer.len()]));
    }

    for _ in 1..depth {
        let mut next_layer = Vec::with_capacity(layer.len() * branching);
        for &u in &layer {
            for _ in 0..branching {
                let c = next_id;
                next_id += 1;
                edges.push((u, c));
                next_layer.push(c);
            }
        }
        for i in 0..next_layer.len() {
            edges.push((next_layer[i], next_layer[(i + 1) % next_layer.len()]));
        }
        layer = next_layer;
    }

    CsrGraph::from_edges(next_id as usize, &edges, None, false, true).unwrap()
}

fn make_open_grid(rows: usize, cols: usize, triangulated: bool) -> CsrGraph<f64> {
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            if c + 1 < cols {
                edges.push((u, u + 1));
            }
            if r + 1 < rows {
                edges.push((u, u + cols as u32));
            }
            if triangulated && r + 1 < rows && c + 1 < cols {
                edges.push((u, u + cols as u32 + 1));
            }
        }
    }
    CsrGraph::from_edges(rows * cols, &edges, None, false, false).unwrap()
}

#[test]
fn test_hyperbolic_3_7_tessellation_and_fans_across_seeds() {
    let default_cfg = ResistanceCurvatureConfig::default();

    let tess = make_hyperbolic_3_7_tessellation(5);
    assert_eq!(tess.num_nodes(), 617);
    assert_eq!(tess.num_edges(), 1463);

    // Raw JL effective-resistance diagnostics remain un-normalized and accurate.
    let raw_er = compute_effective_resistances(&tess, &default_cfg.resistance).unwrap();
    assert!(!raw_er.used_exact);
    assert!(
        raw_er.foster_rel_err < 0.05,
        "raw JL foster_rel_err too large on tess_3_7: {}",
        raw_er.foster_rel_err
    );

    for seed in 40..=55u64 {
        let cfg = default_cfg
            .clone()
            .with_resistance(default_cfg.resistance.clone().with_seed(seed));
        let res = resistance_curvature_flow(&tess, &cfg).unwrap();
        assert!(!res.used_exact);
        assert!(res.foster_rel_err < 1e-11);
        assert_eq!(
            res.suggested_geometry,
            SuggestedGeometry::HyperbolicPoincare,
            "tess_3_7 N=617 seed={seed}: mean={}, std={}",
            res.mean_link_curvature,
            res.std_link_curvature
        );
    }

    let fans = [
        ("fan_316_b2", make_hyperbolic_fan_mesh(5, 2, 6), 316, 630),
        ("fan_379_b2", make_hyperbolic_fan_mesh(6, 2, 6), 379, 756),
        ("fan_485_b3", make_hyperbolic_fan_mesh(4, 3, 5), 485, 968),
        ("fan_727_b3", make_hyperbolic_fan_mesh(6, 3, 5), 727, 1452),
    ];
    for (name, graph, exp_n, exp_m) in fans {
        assert_eq!(graph.num_nodes(), exp_n, "{name} node count");
        assert_eq!(graph.num_edges(), exp_m, "{name} edge count");
        for seed in 40..=55u64 {
            let cfg = default_cfg
                .clone()
                .with_resistance(default_cfg.resistance.clone().with_seed(seed));
            let res = resistance_curvature_flow(&graph, &cfg).unwrap();
            assert!(!res.used_exact);
            assert!(res.foster_rel_err < 1e-11);
            assert_eq!(
                res.suggested_geometry,
                SuggestedGeometry::HyperbolicPoincare,
                "{name} seed={seed}: mean={}, std={}",
                res.mean_link_curvature,
                res.std_link_curvature
            );
        }
    }
}

#[test]
fn test_open_elongated_grid_and_loose_rel_tol_jacobi_cv() {
    let default_cfg = ResistanceCurvatureConfig::default();
    let open_10x40_tri = make_open_grid(10, 40, true);
    assert_eq!(open_10x40_tri.num_nodes(), 400);
    assert_eq!(open_10x40_tri.num_edges(), 1101);

    for seed in 40..=55u64 {
        let cfg = default_cfg
            .clone()
            .with_resistance(default_cfg.resistance.clone().with_seed(seed));
        let res = resistance_curvature_flow(&open_10x40_tri, &cfg).unwrap();
        assert!(!res.used_exact);
        assert!(res.foster_rel_err < 1e-11);
        assert_eq!(
            res.suggested_geometry,
            SuggestedGeometry::Euclidean,
            "open 10x40 tri seed={seed}: mean={}, std={}",
            res.mean_link_curvature,
            res.std_link_curvature
        );
    }

    // Loose rel_tol (> 1e-4) triggers the explicit L*Z SpMM control-variate branch.
    let grid_16x16 = make_open_grid(16, 16, false);
    let loose_cfg = EffectiveResistanceConfig {
        exact: Some(false),
        jl_dim: 64,
        amg_config: AmgPcgConfig {
            rel_tol: 1e-2,
            ..Default::default()
        },
        ..Default::default()
    };
    let loose_er = compute_effective_resistances(&grid_16x16, &loose_cfg).unwrap();
    assert!(!loose_er.used_exact);
    assert!(loose_er.foster_rel_err < 0.05);
    assert!(
        loose_er
            .effective_resistances
            .iter()
            .all(|&r| r.is_finite() && r >= 0.0)
    );
}

#[test]
fn test_open_24x24_quad_uv_sphere_and_per_component_foster_normalization() {
    let default_cfg = ResistanceCurvatureConfig::default();

    for (rows, cols, tri) in [(10usize, 40usize, false), (24, 24, false), (24, 24, true)] {
        let grid = make_open_grid(rows, cols, tri);
        for seed in 40..=55u64 {
            let cfg = default_cfg
                .clone()
                .with_resistance(default_cfg.resistance.clone().with_seed(seed));
            let res = resistance_curvature_flow(&grid, &cfg).unwrap();
            assert!(!res.used_exact);
            assert_eq!(
                res.suggested_geometry,
                SuggestedGeometry::Euclidean,
                "open {rows}x{cols} tri={tri} seed={seed}: mean={}, std={}",
                res.mean_link_curvature,
                res.std_link_curvature
            );
        }
    }

    // Quad UV sphere (18 rings x 18 meridians + 2 poles => N=326, M=666): must not be Torus2d.
    let (lat_rings, lon_segs) = (18usize, 18usize);
    let mut uv_edges = Vec::new();
    let north = 0u32;
    let south = (1 + lat_rings * lon_segs) as u32;
    for r in 0..lat_rings {
        for c in 0..lon_segs {
            let u = (1 + r * lon_segs + c) as u32;
            let next_c = (1 + r * lon_segs + (c + 1) % lon_segs) as u32;
            uv_edges.push((u, next_c));
            if r + 1 < lat_rings {
                let down = (1 + (r + 1) * lon_segs + c) as u32;
                uv_edges.push((u, down));
            }
            if r == 0 {
                uv_edges.push((north, u));
            }
            if r + 1 == lat_rings {
                uv_edges.push((u, south));
            }
        }
    }
    let uv_sphere =
        CsrGraph::<f64>::from_edges(2 + lat_rings * lon_segs, &uv_edges, None, false, false)
            .unwrap();
    assert_eq!(uv_sphere.num_nodes(), 326);
    assert_eq!(uv_sphere.num_edges(), 666);
    for seed in 40..=47u64 {
        let cfg = default_cfg
            .clone()
            .with_resistance(default_cfg.resistance.clone().with_seed(seed));
        let res = resistance_curvature_flow(&uv_sphere, &cfg).unwrap();
        assert_ne!(res.suggested_geometry, SuggestedGeometry::Torus2d);
    }

    // Multi-component graph (16x16 grid + K_30 + 3 isolated vertices = 289 nodes, 5 components):
    // compute_effective_resistances keeps raw JL Foster diagnostics, while resistance_curvature_flow
    // normalizes each non-trivial component to |V_k| - 1 and preserves sum(p) == N - foster_sum.
    let mut multi_edges = Vec::new();
    for r in 0..16u32 {
        for c in 0..16u32 {
            let u = r * 16 + c;
            if c + 1 < 16 {
                multi_edges.push((u, u + 1));
            }
            if r + 1 < 16 {
                multi_edges.push((u, u + 16));
            }
        }
    }
    for u in 256..286u32 {
        for v in (u + 1)..286u32 {
            multi_edges.push((u, v));
        }
    }
    let g_multi = CsrGraph::<f64>::from_edges(289, &multi_edges, None, false, false).unwrap();
    let raw_er = compute_effective_resistances(&g_multi, &default_cfg.resistance).unwrap();
    assert!(!raw_er.used_exact);
    assert_eq!(raw_er.num_components, 5);
    assert!((raw_er.expected_foster - 284.0).abs() < 1e-12);
    assert!(raw_er.foster_rel_err < 0.05);

    let flow = resistance_curvature_flow(&g_multi, &default_cfg).unwrap();
    assert!(!flow.used_exact);
    assert_eq!(flow.num_components, 5);
    assert!(flow.foster_rel_err < 1e-11);
    let comp1_foster: f64 = flow
        .edge_u
        .iter()
        .enumerate()
        .filter(|&(_, &u)| u < 256)
        .map(|(e, _)| flow.conductances[e] * flow.effective_resistances[e])
        .sum();
    let comp2_foster: f64 = flow
        .edge_u
        .iter()
        .enumerate()
        .filter(|&(_, &u)| (256..286).contains(&u))
        .map(|(e, _)| flow.conductances[e] * flow.effective_resistances[e])
        .sum();
    assert!((comp1_foster - 255.0).abs() < 1e-10);
    assert!((comp2_foster - 29.0).abs() < 1e-10);
    let sum_p: f64 = flow.vertex_curvatures.iter().sum();
    assert!((sum_p - (289.0 - flow.foster_sum)).abs() < 1e-10);
}

#[test]
fn test_non_converged_tight_rel_tol_and_loose_abs_tol_use_explicit_lz_spmm() {
    use egraph_core::DeterministicRng;

    let grid_16x16 = make_open_grid(16, 16, false);
    let max_iter_cfg = EffectiveResistanceConfig {
        exact: Some(false),
        jl_dim: 32,
        amg_config: AmgPcgConfig {
            rel_tol: 1e-10,
            abs_tol: 0.0,
            max_iterations: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let er = compute_effective_resistances(&grid_16x16, &max_iter_cfg).unwrap();
    assert!(!er.used_exact);
    assert!(!er.amg_all_converged);
    assert!(er.amg_max_relative_residual > 1e-4);
    assert!(
        er.effective_resistances
            .iter()
            .all(|&r| r.is_finite() && r >= 0.0)
    );

    // Independently reconstruct Z after 1 PCG step and verify explicit L * Z SpMM was used
    // (not the `&b_rhs` shortcut, which would miscompute D^-1 L Z when PCG has not converged).
    let l_mat = super::graph_ops::build_combinatorial_laplacian_from_edges(
        er.num_nodes,
        &er.edge_u,
        &er.edge_v,
        &er.conductances,
    )
    .unwrap();
    let m = er.num_edges;
    let k_jl = max_iter_cfg.jl_dim;
    let inv_sqrt_k = 1.0 / (k_jl as f64).sqrt();
    let total = er.num_nodes * k_jl;
    let mut b_rhs = vec![0.0_f64; total];
    let mut rng = DeterministicRng::from_seed_u64(max_iter_cfg.seed);
    let mut bits = 0u64;
    let mut bit_rem = 0u32;
    for e in 0..m {
        let u_off = (er.edge_u[e] as usize) * k_jl;
        let v_off = (er.edge_v[e] as usize) * k_jl;
        let s = er.conductances[e].sqrt() * inv_sqrt_k;
        for c in 0..k_jl {
            if bit_rem == 0 {
                bits = rng.next_u64();
                bit_rem = 64;
            }
            let signed = if (bits & 1) == 0 { s } else { -s };
            bits >>= 1;
            bit_rem -= 1;
            b_rhs[u_off + c] += signed;
            b_rhs[v_off + c] -= signed;
        }
    }
    let solver =
        crate::AmgPcgSolver::with_near_nullspace(&l_mat, None, max_iter_cfg.amg_config).unwrap();
    let mut ws = solver.new_workspace();
    let mut z_sol = vec![0.0_f64; total];
    let batch = solver
        .solve_batch(&b_rhs, &mut z_sol, k_jl, &mut ws)
        .unwrap();
    assert!(!batch.all_converged);
    assert!(batch.max_relative_residual > 1e-4);

    let mut l_mat_z = vec![0.0_f64; total];
    l_mat.spmm(&z_sol, &mut l_mat_z, k_jl).unwrap();
    let mut diag = vec![0.0_f64; er.num_nodes];
    for e in 0..m {
        diag[er.edge_u[e] as usize] += er.conductances[e];
        diag[er.edge_v[e] as usize] += er.conductances[e];
    }
    for (i, (&d, z_row)) in diag.iter().zip(z_sol.chunks_exact_mut(k_jl)).enumerate() {
        let inv_d = 1.0 / d;
        let lz_row = &l_mat_z[i * k_jl..(i + 1) * k_jl];
        for c in 0..k_jl {
            z_row[c] -= lz_row[c] * inv_d;
        }
    }
    for e in 0..m {
        let u = er.edge_u[e] as usize;
        let v = er.edge_v[e] as usize;
        let r_det = (1.0 / diag[u] + 1.0 / diag[v]
            - 2.0 * er.conductances[e] / (diag[u] * diag[v]))
            .max(0.0);
        let r_res: f64 = (0..k_jl)
            .map(|c| {
                let diff = z_sol[u * k_jl + c] - z_sol[v * k_jl + c];
                diff * diff
            })
            .sum();
        assert!((er.effective_resistances[e] - (r_det + r_res)).abs() < 1e-12);
    }

    // Loose abs_tol with tight rel_tol also converges early with max_relative_residual > 1e-4,
    // which must likewise trigger the explicit L * Z SpMM fallback cleanly.
    let loose_abs_cfg = EffectiveResistanceConfig {
        exact: Some(false),
        jl_dim: 32,
        amg_config: AmgPcgConfig {
            rel_tol: 1e-10,
            abs_tol: 1e2,
            ..Default::default()
        },
        ..Default::default()
    };
    let er_abs = compute_effective_resistances(&grid_16x16, &loose_abs_cfg).unwrap();
    assert!(er_abs.amg_all_converged);
    assert!(er_abs.amg_max_relative_residual > 1e-4);
    assert!(
        er_abs
            .effective_resistances
            .iter()
            .all(|&r| r.is_finite() && r >= 0.0)
    );
}
