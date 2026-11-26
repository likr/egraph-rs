use egraph_core::{CoreError, CsrGraph, DeterministicRng, Layout2d, LayoutNd};

use crate::extract_unique_edges;
use crate::nd_index::{build_nd_spatial_index, dist_sq_f64, euclidean_diameter_nd, slice_dist_nd};

use super::*;

fn brute_gg_edges_nd(layout: &LayoutNd<f64>) -> Vec<(u32, u32)> {
    let n = layout.num_nodes();
    let dims = layout.dims();
    let coords = layout.coords();
    let mut out = Vec::new();
    for i in 0..n {
        let pi = &coords[i * dims..(i + 1) * dims];
        for j in (i + 1)..n {
            let pj = &coords[j * dims..(j + 1) * dims];
            let d_ij_sq = dist_sq_f64(pi, pj);
            if d_ij_sq <= 0.0 {
                continue;
            }
            let mut ok = true;
            for r in 0..n {
                if r == i || r == j {
                    continue;
                }
                let pr = &coords[r * dims..(r + 1) * dims];
                let d_ir_sq = dist_sq_f64(pi, pr);
                let d_jr_sq = dist_sq_f64(pj, pr);
                if d_ir_sq + d_jr_sq <= d_ij_sq * (1.0 + 1e-12) {
                    ok = false;
                    break;
                }
            }
            if ok {
                out.push((i as u32, j as u32));
            }
        }
    }
    out
}

fn brute_rng_edges_nd(layout: &LayoutNd<f64>) -> Vec<(u32, u32)> {
    let n = layout.num_nodes();
    let dims = layout.dims();
    let coords = layout.coords();
    let mut out = Vec::new();
    for i in 0..n {
        let pi = &coords[i * dims..(i + 1) * dims];
        for j in (i + 1)..n {
            let pj = &coords[j * dims..(j + 1) * dims];
            let d_ij_sq = dist_sq_f64(pi, pj);
            if d_ij_sq <= 0.0 {
                continue;
            }
            let thresh = d_ij_sq * (1.0 - 1e-12);
            let mut ok = true;
            for r in 0..n {
                if r == i || r == j {
                    continue;
                }
                let pr = &coords[r * dims..(r + 1) * dims];
                let d_ir_sq = dist_sq_f64(pi, pr);
                let d_jr_sq = dist_sq_f64(pj, pr);
                if d_ir_sq < thresh && d_jr_sq < thresh {
                    ok = false;
                    break;
                }
            }
            if ok {
                out.push((i as u32, j as u32));
            }
        }
    }
    out
}

fn brute_gabriel_property_nd<W>(graph: &CsrGraph<W>, layout: &LayoutNd<f64>) -> f64 {
    let n = layout.num_nodes();
    let dims = layout.dims();
    let coords = layout.coords();
    let mut mid = vec![0.0_f64; dims];
    let mut sum = 0.0_f64;
    for (u, v) in extract_unique_edges(graph) {
        let ui = u as usize;
        let vi = v as usize;
        let pu = &coords[ui * dims..(ui + 1) * dims];
        let pv = &coords[vi * dims..(vi + 1) * dims];
        let r = 0.5 * slice_dist_nd(pu, pv);
        if r <= 0.0 {
            continue;
        }
        for d in 0..dims {
            mid[d] = 0.5 * (pu[d] + pv[d]);
        }
        for i in 0..n {
            if i == ui || i == vi {
                continue;
            }
            let pi = &coords[i * dims..(i + 1) * dims];
            let d = slice_dist_nd(pi, &mid);
            if d < r {
                let viol = r - d;
                sum += viol * viol;
            }
        }
    }
    sum
}

fn brute_node_resolution_nd(layout: &LayoutNd<f64>) -> f64 {
    let n = layout.num_nodes();
    if n <= 1 {
        return 0.0;
    }
    let dims = layout.dims();
    let coords = layout.coords();
    let d_max = euclidean_diameter_nd(layout).unwrap();
    if d_max <= 0.0 {
        return (n as f64) * ((n - 1) as f64) * 0.5;
    }
    let thresh = d_max / (n as f64).sqrt();
    let mut sum = 0.0_f64;
    for i in 0..n {
        let pi = &coords[i * dims..(i + 1) * dims];
        for j in (i + 1)..n {
            let pj = &coords[j * dims..(j + 1) * dims];
            let d = slice_dist_nd(pi, pj);
            if d < thresh {
                let hinge = 1.0 - d / thresh;
                sum += hinge * hinge;
            }
        }
    }
    sum
}

#[test]
fn test_proximity_nd_2d_bit_exact_parity() {
    let g = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2)],
        None,
        false,
        false,
    )
    .unwrap();
    let l2 = Layout2d::from_xy(
        vec![0.0, 2.0, 1.0, 1.1, 3.0, 4.0],
        vec![0.0, 0.0, 1.5, 0.2, 0.5, 0.0],
    )
    .unwrap();
    let lnd = LayoutNd::from_flat(6, 2, l2.to_interleaved()).unwrap();
    let idx = build_nd_spatial_index(&lnd).unwrap();

    assert_eq!(
        gabriel_graph_property(&g, &l2).unwrap().to_bits(),
        gabriel_graph_property_nd(&g, &lnd).unwrap().to_bits()
    );
    assert_eq!(
        gabriel_graph_property_nd(&g, &lnd).unwrap().to_bits(),
        gabriel_graph_property_nd_with_index(&g, &lnd, &idx)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        node_resolution(&l2).unwrap().to_bits(),
        node_resolution_nd(&lnd).unwrap().to_bits()
    );
    assert_eq!(
        node_resolution_nd(&lnd).unwrap().to_bits(),
        node_resolution_nd_with_index(&lnd, &idx).unwrap().to_bits()
    );
    assert_eq!(
        relative_neighborhood_edges(&l2).unwrap(),
        relative_neighborhood_edges_nd(&lnd).unwrap()
    );
    assert_eq!(
        gabriel_graph_edges(&l2).unwrap(),
        gabriel_graph_edges_nd(&lnd).unwrap()
    );
    assert_eq!(
        relative_neighborhood_preservation(&g, &l2)
            .unwrap()
            .to_bits(),
        relative_neighborhood_preservation_nd(&g, &lnd)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        gabriel_graph_preservation(&g, &l2).unwrap().to_bits(),
        gabriel_graph_preservation_nd(&g, &lnd).unwrap().to_bits()
    );
}

#[test]
fn test_proximity_nd_brute_oracle_across_dims_ties_and_coincident() {
    for dims in [1usize, 3, 4, 8, 10] {
        let mut rng = DeterministicRng::from_seed_u64(300 + dims as u64);
        let n = 40usize;
        let mut coords = Vec::with_capacity(n * dims);
        for i in 0..n {
            for _d in 0..dims {
                let scale = if i < 32 { 4.0 } else { 80.0 };
                coords.push((rng.next_f64() - 0.5) * scale);
            }
        }
        // Inject exact coincident duplicate and cocircular/hypercube tie points.
        for d in 0..dims {
            coords[d] = 0.0;
            coords[dims + d] = 0.0;
            coords[2 * dims + d] = if d == 0 { 1.0 } else { 0.0 };
            coords[3 * dims + d] = if d == dims - 1 { 1.0 } else { 0.0 };
        }

        let layout = LayoutNd::from_flat(n, dims, coords).unwrap();
        let idx = build_nd_spatial_index(&layout).unwrap();

        let gg = gabriel_graph_edges_nd(&layout).unwrap();
        let rng_edges = relative_neighborhood_edges_nd(&layout).unwrap();
        assert_eq!(gg, brute_gg_edges_nd(&layout));
        assert_eq!(rng_edges, brute_rng_edges_nd(&layout));
        assert_eq!(
            gg,
            gabriel_graph_edges_nd_with_index(&layout, &idx).unwrap()
        );
        assert_eq!(
            rng_edges,
            relative_neighborhood_edges_nd_with_index(&layout, &idx).unwrap()
        );

        let nr = node_resolution_nd(&layout).unwrap();
        let nr_exp = brute_node_resolution_nd(&layout);
        assert!((nr - nr_exp).abs() < 1e-10);
        assert_eq!(
            nr.to_bits(),
            node_resolution_nd_with_index(&layout, &idx)
                .unwrap()
                .to_bits()
        );

        let g =
            CsrGraph::<f64>::from_edges(n, &gg[..gg.len().min(30)], None, false, false).unwrap();
        let gp = gabriel_graph_property_nd(&g, &layout).unwrap();
        let gp_exp = brute_gabriel_property_nd(&g, &layout);
        assert!((gp - gp_exp).abs() < 1e-10);
        assert_eq!(
            gp.to_bits(),
            gabriel_graph_property_nd_with_index(&g, &layout, &idx)
                .unwrap()
                .to_bits()
        );

        let s_rng = relative_neighborhood_preservation_nd(&g, &layout).unwrap();
        let s_gg = gabriel_graph_preservation_nd(&g, &layout).unwrap();
        assert!((0.0..=1.0).contains(&s_rng));
        assert!((0.0..=1.0).contains(&s_gg));
        assert_eq!(
            s_rng.to_bits(),
            relative_neighborhood_preservation_nd_with_index(&g, &layout, &idx)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            s_gg.to_bits(),
            gabriel_graph_preservation_nd_with_index(&g, &layout, &idx)
                .unwrap()
                .to_bits()
        );
    }

    // Exact 3D equilateral cube lattice edges: 3D unit cube has GG == RNG == 12 cube edges.
    let mut cube_coords = Vec::new();
    for z in [0.0, 1.0] {
        for y in [0.0, 1.0] {
            for x in [0.0, 1.0] {
                cube_coords.extend_from_slice(&[x, y, z]);
            }
        }
    }
    let cube = LayoutNd::from_flat(8, 3, cube_coords).unwrap();
    let cube_gg = gabriel_graph_edges_nd(&cube).unwrap();
    let cube_rng = relative_neighborhood_edges_nd(&cube).unwrap();
    assert_eq!(cube_gg.len(), 12);
    assert_eq!(cube_rng.len(), 12);
    let g_cube = CsrGraph::<f64>::from_edges(8, &cube_gg, None, false, false).unwrap();
    assert!((gabriel_graph_preservation_nd(&g_cube, &cube).unwrap() - 1.0).abs() < 1e-12);
    assert!((relative_neighborhood_preservation_nd(&g_cube, &cube).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn test_proximity_nd_boundaries_f32_and_errors() {
    for n in [0usize, 1] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let l = LayoutNd::<f64>::zeros(n, 3).unwrap();
        assert_eq!(gabriel_graph_property_nd(&g, &l).unwrap(), 0.0);
        assert_eq!(node_resolution_nd(&l).unwrap(), 0.0);
        assert!(relative_neighborhood_edges_nd(&l).unwrap().is_empty());
        assert!(gabriel_graph_edges_nd(&l).unwrap().is_empty());
        let expected_pres = if n == 0 { 0.0 } else { 1.0 };
        assert_eq!(
            relative_neighborhood_preservation_nd(&g, &l).unwrap(),
            expected_pres
        );
        assert_eq!(
            gabriel_graph_preservation_nd(&g, &l).unwrap(),
            expected_pres
        );
    }

    // Collapsed 3D layout.
    let collapsed = LayoutNd::<f64>::zeros(5, 3).unwrap();
    assert_eq!(node_resolution_nd(&collapsed).unwrap(), 10.0);
    assert!(gabriel_graph_edges_nd(&collapsed).unwrap().is_empty());
    assert!(
        relative_neighborhood_edges_nd(&collapsed)
            .unwrap()
            .is_empty()
    );

    // f32 layout check in 3D: right-angled corner has 3 GG/RNG edges; regular tetrahedron has 6.
    let l32_corner = LayoutNd::from_flat(
        4,
        3,
        vec![
            0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
    .unwrap();
    let g32 =
        CsrGraph::<f32>::from_edges(4, &[(0, 1), (0, 2), (0, 3)], None, false, false).unwrap();
    assert_eq!(gabriel_graph_edges_nd(&l32_corner).unwrap().len(), 3);
    assert_eq!(
        relative_neighborhood_edges_nd(&l32_corner).unwrap().len(),
        3
    );
    assert!((gabriel_graph_preservation_nd(&g32, &l32_corner).unwrap() - 1.0).abs() < 1e-12);

    let l32_tet = LayoutNd::from_flat(
        4,
        3,
        vec![
            1.0_f32, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
        ],
    )
    .unwrap();
    assert_eq!(gabriel_graph_edges_nd(&l32_tet).unwrap().len(), 6);
    assert_eq!(relative_neighborhood_edges_nd(&l32_tet).unwrap().len(), 6);
    assert!((node_resolution_nd(&l32_tet).unwrap() - 0.0).abs() < 1e-12);
    assert!(gabriel_graph_preservation_nd(&g32, &l32_tet).unwrap() > 0.0);

    // Non-finite and dimension mismatch checks.
    let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let l_nan = LayoutNd::from_flat(2, 3, vec![0.0, 0.0, f64::NAN, 1.0, 0.0, 0.0]).unwrap();
    assert!(matches!(
        node_resolution_nd(&l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        gabriel_graph_edges_nd(&l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        relative_neighborhood_edges_nd(&l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        gabriel_graph_property_nd(&g2, &l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let l4 = LayoutNd::<f64>::zeros(4, 3).unwrap();
    assert!(matches!(
        gabriel_graph_preservation_nd(&g2, &l4),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
