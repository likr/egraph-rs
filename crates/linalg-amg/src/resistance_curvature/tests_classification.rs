use egraph_core::CsrGraph;

use super::*;

fn make_torus_grid(rows: usize, cols: usize, triangulated: bool) -> CsrGraph<f64> {
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            let r_next = (r * cols + ((c + 1) % cols)) as u32;
            let d_next = (((r + 1) % rows) * cols + c) as u32;
            edges.push((u, r_next));
            edges.push((u, d_next));
            if triangulated {
                let diag = (((r + 1) % rows) * cols + ((c + 1) % cols)) as u32;
                edges.push((u, diag));
            }
        }
    }
    CsrGraph::from_edges(rows * cols, &edges, None, false, true).unwrap()
}

fn make_subdivided_octahedron(subdivisions: usize) -> CsrGraph<f64> {
    // Start with a regular octahedron: 6 vertices, 12 edges, 8 triangular faces.
    let mut tris: Vec<[u32; 3]> = vec![
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        [5, 2, 1],
        [5, 3, 2],
        [5, 4, 3],
        [5, 1, 4],
    ];
    let mut num_nodes = 6u32;

    for _ in 0..subdivisions {
        let mut edge_midpoints: Vec<((u32, u32), u32)> = Vec::new();
        let mut get_midpoint = |a: u32, b: u32| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&(_, mid)) = edge_midpoints.iter().find(|&&(k, _)| k == key) {
                return mid;
            }
            let mid = num_nodes;
            num_nodes += 1;
            edge_midpoints.push((key, mid));
            mid
        };

        let mut next_tris = Vec::with_capacity(tris.len() * 4);
        for &[a, b, c] in &tris {
            let ab = get_midpoint(a, b);
            let bc = get_midpoint(b, c);
            let ca = get_midpoint(c, a);
            next_tris.push([a, ab, ca]);
            next_tris.push([b, bc, ab]);
            next_tris.push([c, ca, bc]);
            next_tris.push([ab, bc, ca]);
        }
        tris = next_tris;
    }

    let mut edges = Vec::with_capacity(tris.len() * 3);
    for &[a, b, c] in &tris {
        edges.push((a, b));
        edges.push((b, c));
        edges.push((c, a));
    }
    CsrGraph::from_edges(num_nodes as usize, &edges, None, false, true).unwrap()
}

#[test]
fn test_default_config_and_degree_cv_threshold() {
    let eff_cfg = EffectiveResistanceConfig::default();
    assert_eq!(eff_cfg.exact_max_nodes, 256);
    assert_eq!(eff_cfg.jl_dim, 128);
    assert!((TORUS_DEG_CV_THRESHOLD - 0.02).abs() < 1e-12);

    // deg_cv = 0.03 is rejected under tightened 0.02 threshold even when std_link_curvature == 0.
    assert_eq!(
        classify_geometry_from_curvature(100, 0.02, 0.0, 0.0, 0.03, -0.05, 0.2, 0.05),
        SuggestedGeometry::Euclidean
    );
    assert_eq!(
        classify_geometry_from_curvature(100, 0.02, 0.0, 0.0, 0.01, -0.05, 0.2, 0.05),
        SuggestedGeometry::Torus2d
    );
}

#[test]
fn test_torus_n_gt_64_exact_and_n_gt_256_jl() {
    // N = 144 (64 < N <= 256): default uses exact pseudoinverse -> Torus2d.
    let t12 = make_torus_grid(12, 12, false);
    let res_144 = resistance_curvature_flow(
        &t12,
        &ResistanceCurvatureConfig {
            steps: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(res_144.used_exact);
    assert_eq!(res_144.suggested_geometry, SuggestedGeometry::Torus2d);

    // N = 324 (N > 256): default uses JL (jl_dim=128); JL-calibrated threshold keeps Torus2d.
    let t18 = make_torus_grid(18, 18, false);
    let res_324 = resistance_curvature_flow(
        &t18,
        &ResistanceCurvatureConfig {
            steps: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!res_324.used_exact);
    assert!(
        res_324.std_link_curvature > 0.05,
        "expected JL curvature std > 0.05, got {}",
        res_324.std_link_curvature
    );
    assert_eq!(res_324.suggested_geometry, SuggestedGeometry::Torus2d);

    // Triangulated torus T_{18x18} (Euler V - E + T = 0) must still classify as Torus2d, NOT Spherical.
    let t18_tri = make_torus_grid(18, 18, true);
    let res_tri = resistance_curvature_flow(
        &t18_tri,
        &ResistanceCurvatureConfig {
            steps: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!res_tri.used_exact);
    assert_eq!(
        res_tri.suggested_geometry,
        SuggestedGeometry::Torus2d,
        "triangulated torus N=324: mean={}, std={}, foster_err={}",
        res_tri.mean_link_curvature,
        res_tri.std_link_curvature,
        res_tri.foster_rel_err
    );
}

#[test]
fn test_closed_spherical_surface_vs_open_planar_mesh() {
    // Subdivided octahedron level 3: N = 258 > 256, closed genus-0 triangulated sphere (Euler = 2).
    // Even though N is large and mean_link_curvature is well below 0.2, closed-surface topology
    // classifies it as Spherical.
    let sphere_258 = make_subdivided_octahedron(3);
    assert_eq!(sphere_258.num_nodes(), 258);
    let res_sph = resistance_curvature_flow(
        &sphere_258,
        &ResistanceCurvatureConfig {
            steps: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        res_sph.mean_link_curvature < 0.2,
        "expected large sphere mean_link_curvature < 0.2, got {}",
        res_sph.mean_link_curvature
    );
    assert_eq!(res_sph.suggested_geometry, SuggestedGeometry::Spherical);

    // Closed triangulated sphere with chordal cross-braces (heterogeneous degrees like can_96,
    // every edge in >= 2 triangles and positive Euler characteristic) -> Spherical even if
    // mean_link_curvature drops below hyperbolic_threshold.
    let mut sphere_66 = make_subdivided_octahedron(2); // 66 nodes, 192 edges
    let mut chord_edges: Vec<(u32, u32)> = Vec::new();
    for u in 0..sphere_66.num_nodes() {
        for &v in sphere_66.neighbors(u) {
            if (u as u32) < v {
                chord_edges.push((u as u32, v));
            }
        }
    }
    // Add a closed ring of internal triangle-forming chords on a face patch (like can_96 4-tri edges).
    let n0 = sphere_66.neighbors(0).to_vec();
    for i in 0..n0.len() {
        let a = n0[i];
        let b = n0[(i + 2) % n0.len()];
        if a != b {
            chord_edges.push((a, b));
        }
    }
    sphere_66 = CsrGraph::from_edges(66, &chord_edges, None, false, true).unwrap();
    let res_can_like = resistance_curvature_flow(
        &sphere_66,
        &ResistanceCurvatureConfig {
            steps: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        res_can_like.suggested_geometry,
        SuggestedGeometry::Spherical
    );

    // Open planar triangulated mesh (has boundary edges belonging to only 1 triangle) -> Euclidean.
    let mut open_edges = Vec::new();
    let rows = 8usize;
    let cols = 8usize;
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            if c + 1 < cols {
                open_edges.push((u, u + 1));
            }
            if r + 1 < rows {
                open_edges.push((u, u + cols as u32));
            }
            if r + 1 < rows && c + 1 < cols {
                open_edges.push((u, u + cols as u32 + 1));
            }
        }
    }
    let open_mesh =
        CsrGraph::<f64>::from_edges(rows * cols, &open_edges, None, false, true).unwrap();
    let res_open = resistance_curvature_flow(
        &open_mesh,
        &ResistanceCurvatureConfig {
            steps: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_ne!(res_open.suggested_geometry, SuggestedGeometry::Spherical);
    assert_ne!(res_open.suggested_geometry, SuggestedGeometry::Torus2d);
}

#[test]
fn test_default_flow_tori_triangulated_and_quad_across_seeds() {
    let default_cfg = ResistanceCurvatureConfig::default();
    assert_eq!(default_cfg.steps, 2);
    assert_eq!(default_cfg.resistance.exact_max_nodes, 256);
    assert_eq!(default_cfg.resistance.jl_dim, 128);

    for &(rows, cols) in &[(18usize, 18usize), (20, 20), (24, 24)] {
        for triangulated in [false, true] {
            let torus = make_torus_grid(rows, cols, triangulated);
            for seed in 42..=49u64 {
                let cfg = default_cfg
                    .clone()
                    .with_resistance(default_cfg.resistance.clone().with_seed(seed));
                let res = resistance_curvature_flow(&torus, &cfg).unwrap();
                assert!(!res.used_exact);
                assert_eq!(res.step_history.len(), 3);
                assert_eq!(
                    res.suggested_geometry,
                    SuggestedGeometry::Torus2d,
                    "T_{rows}x{cols} tri={triangulated} seed={seed}: mean={}, std={}, foster={}",
                    res.mean_link_curvature,
                    res.std_link_curvature,
                    res.foster_rel_err
                );
            }
        }
    }
}

#[test]
fn test_default_flow_spheres_open_planar_hyperbolic_and_irregular() {
    let default_cfg = ResistanceCurvatureConfig::default();

    // Closed spherical meshes (N >= 98): N=258 and N=1026 subdivided octahedra via default JL,
    // plus forced JL on N=258 / N=1026.
    for sub in [3usize, 4usize] {
        let sphere = make_subdivided_octahedron(sub);
        assert!(sphere.num_nodes() >= 98);
        let res = resistance_curvature_flow(&sphere, &default_cfg).unwrap();
        assert!(!res.used_exact);
        assert_eq!(res.suggested_geometry, SuggestedGeometry::Spherical);
    }

    // Open planar 18x18 grids (quad and triangulated, N=324 > 256) must classify as Euclidean.
    for tri in [false, true] {
        let mut open_edges = Vec::new();
        let rows = 18usize;
        let cols = 18usize;
        for r in 0..rows {
            for c in 0..cols {
                let u = (r * cols + c) as u32;
                if c + 1 < cols {
                    open_edges.push((u, u + 1));
                }
                if r + 1 < rows {
                    open_edges.push((u, u + cols as u32));
                }
                if tri && r + 1 < rows && c + 1 < cols {
                    open_edges.push((u, u + cols as u32 + 1));
                }
            }
        }
        let open_grid =
            CsrGraph::<f64>::from_edges(rows * cols, &open_edges, None, false, false).unwrap();
        let res = resistance_curvature_flow(&open_grid, &default_cfg).unwrap();
        assert!(!res.used_exact);
        assert_eq!(
            res.suggested_geometry,
            SuggestedGeometry::Euclidean,
            "open 18x18 tri={tri} default seed=42: mean={}, std={}",
            res.mean_link_curvature,
            res.std_link_curvature
        );
    }

    // Hyperbolic 3-ary tree of depth 5 (N=364 > 256) classifies as HyperbolicPoincare.
    let mut tree_edges = Vec::new();
    let mut next_child = 1u32;
    for parent in 0..121u32 {
        for _ in 0..3 {
            tree_edges.push((parent, next_child));
            next_child += 1;
        }
    }
    let htree = CsrGraph::<f64>::from_edges(364, &tree_edges, None, false, false).unwrap();
    let res_hyp = resistance_curvature_flow(&htree, &default_cfg).unwrap();
    assert!(!res_hyp.used_exact);
    assert_eq!(
        res_hyp.suggested_geometry,
        SuggestedGeometry::HyperbolicPoincare
    );

    // Irregular graph (N=300, deg_cv >= 0.02: 18x16 grid plus a degree-16 hub chord cluster) -> not Torus2d.
    let mut irreg_edges = Vec::new();
    for r in 0..15usize {
        for c in 0..20usize {
            let u = (r * 20 + c) as u32;
            if c + 1 < 20 {
                irreg_edges.push((u, u + 1));
            }
            if r + 1 < 15 {
                irreg_edges.push((u, u + 20));
            }
        }
    }
    for v in 2..30u32 {
        irreg_edges.push((0, v));
    }
    let irreg = CsrGraph::<f64>::from_edges(300, &irreg_edges, None, false, true).unwrap();
    let res_irreg = resistance_curvature_flow(&irreg, &default_cfg).unwrap();
    assert_ne!(res_irreg.suggested_geometry, SuggestedGeometry::Torus2d);

    // Direct JL sampling classifier boundary checks:
    // Regular non-closed-spherical graph with flat mean curvature and residual JL spread
    // std <= torus_std_threshold * (1 + 8 * sqrt(64/128)) (~0.3328) -> Torus2d.
    assert_eq!(
        classify_geometry_from_curvature_with_sampling(
            972, 0.03, 0.25, 0.0, 0.0, -0.05, 0.2, 0.05, false, 128, false
        ),
        SuggestedGeometry::Torus2d
    );
    // Excessive std (> 0.3328) falls back to Euclidean.
    assert_eq!(
        classify_geometry_from_curvature_with_sampling(
            972, 0.03, 0.40, 0.0, 0.0, -0.05, 0.2, 0.05, false, 128, false
        ),
        SuggestedGeometry::Euclidean
    );
    // Sharp thresholds (-0.05 / 0.2) apply identically in JL mode without mean widening.
    assert_eq!(
        classify_geometry_from_curvature_with_sampling(
            972, 0.21, 0.15, 0.0, 0.0, -0.05, 0.2, 0.05, false, 128, false
        ),
        SuggestedGeometry::Spherical
    );
    assert_eq!(
        classify_geometry_from_curvature_with_sampling(
            972, -0.06, 0.15, 0.0, 0.0, -0.05, 0.2, 0.05, false, 128, false
        ),
        SuggestedGeometry::HyperbolicPoincare
    );
    // Non-regular graph keeps sharp -0.05 / 0.2 boundaries in both JL and exact modes.
    assert_eq!(
        classify_geometry_from_curvature_with_sampling(
            612, -0.03, 0.30, 0.0, 0.15, -0.05, 0.2, 0.05, false, 128, false
        ),
        SuggestedGeometry::Euclidean
    );
    assert_eq!(
        classify_geometry_from_curvature_with_sampling(
            612, 0.21, 0.30, 0.0, 0.15, -0.05, 0.2, 0.05, false, 128, false
        ),
        SuggestedGeometry::Spherical
    );
    assert_eq!(
        classify_geometry_from_curvature_with_sampling(
            612, -0.06, 0.30, 0.0, 0.15, -0.05, 0.2, 0.05, false, 128, false
        ),
        SuggestedGeometry::HyperbolicPoincare
    );
    assert_eq!(
        classify_geometry_from_curvature_with_sampling(
            612, -0.12, 0.25, 0.0, 0.15, -0.05, 0.2, 0.05, true, 128, false
        ),
        SuggestedGeometry::HyperbolicPoincare
    );
}

#[test]
fn test_default_flow_open_planar_grids_euclidean_across_seeds() {
    let default_cfg = ResistanceCurvatureConfig::default();
    for &(rows, cols) in &[(18usize, 18usize), (20, 20)] {
        for tri in [false, true] {
            let mut open_edges = Vec::new();
            for r in 0..rows {
                for c in 0..cols {
                    let u = (r * cols + c) as u32;
                    if c + 1 < cols {
                        open_edges.push((u, u + 1));
                    }
                    if r + 1 < rows {
                        open_edges.push((u, u + cols as u32));
                    }
                    if tri && r + 1 < rows && c + 1 < cols {
                        open_edges.push((u, u + cols as u32 + 1));
                    }
                }
            }
            let open_grid =
                CsrGraph::<f64>::from_edges(rows * cols, &open_edges, None, false, false).unwrap();
            for seed in 40..=55u64 {
                let cfg = default_cfg
                    .clone()
                    .with_resistance(default_cfg.resistance.clone().with_seed(seed));
                let res = resistance_curvature_flow(&open_grid, &cfg).unwrap();
                assert!(!res.used_exact);
                assert_eq!(
                    res.suggested_geometry,
                    SuggestedGeometry::Euclidean,
                    "open {rows}x{cols} tri={tri} seed={seed}: mean={}, std={}, foster={}",
                    res.mean_link_curvature,
                    res.std_link_curvature,
                    res.foster_rel_err
                );
            }
        }
    }
}
