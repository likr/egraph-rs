use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, Layout3d, LayoutNd, LayoutStorage,
    ManifoldKind, ManifoldLayout2d, MetricSpace,
};
use egraph_spatial_trees::StaticKdTree;

use crate::extract_unique_edges;
use crate::spatial_index::{build_spatial_index, dist_sq_f64, euclidean_diameter};

use super::*;

struct CustomMetricCloud3d {
    pts: Vec<[f64; 3]>,
}

impl LayoutStorage for CustomMetricCloud3d {
    type Scalar = f64;

    fn len(&self) -> usize {
        self.pts.len()
    }

    fn validate_finite(&self) -> CoreResult<()> {
        if self
            .pts
            .iter()
            .all(|p| p.iter().all(|coord| coord.is_finite()))
        {
            Ok(())
        } else {
            Err(CoreError::non_finite("CustomMetricCloud3d"))
        }
    }
}

impl MetricSpace for CustomMetricCloud3d {
    fn distance(&self, i: usize, j: usize) -> f64 {
        let a = self.pts[i];
        let b = self.pts[j];
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

fn brute_gg_edges_flat(layout: &LayoutNd<f64>) -> Vec<(u32, u32)> {
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

fn brute_rng_edges_flat(layout: &LayoutNd<f64>) -> Vec<(u32, u32)> {
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

fn brute_gabriel_property_flat<W>(graph: &CsrGraph<W>, layout: &LayoutNd<f64>) -> f64 {
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
        let r = 0.5 * layout.distance(ui, vi);
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
            let d = dist_sq_f64(pi, &mid).sqrt();
            if d < r {
                let viol = r - d;
                sum += viol * viol;
            }
        }
    }
    sum
}

fn brute_node_resolution_flat(layout: &LayoutNd<f64>) -> f64 {
    let n = layout.num_nodes();
    if n <= 1 {
        return 0.0;
    }
    let d_max = euclidean_diameter(layout).unwrap();
    if d_max <= 0.0 {
        return (n as f64) * ((n - 1) as f64) * 0.5;
    }
    let thresh = d_max / (n as f64).sqrt();
    let mut sum = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = layout.distance(i, j);
            if d < thresh {
                let hinge = 1.0 - d / thresh;
                sum += hinge * hinge;
            }
        }
    }
    sum
}

#[test]
fn test_proximity_2d_3d_nd_custom_metric_parity() {
    let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2)];
    let g = CsrGraph::<f64>::from_edges(6, &edges, None, false, false).unwrap();
    let g_unit = CsrGraph::<()>::from_unweighted_edges(6, &edges, false, false).unwrap();
    let l2 = Layout2d::from_xy(
        vec![0.0, 2.0, 1.0, 1.1, 3.0, 4.0],
        vec![0.0, 0.0, 1.5, 0.2, 0.5, 0.0],
    )
    .unwrap();
    let lnd2 = LayoutNd::from_flat(6, 2, &l2.to_interleaved()).unwrap();
    let idx2 = build_spatial_index(&lnd2).unwrap();

    assert!(
        (gabriel_graph_property(&g, &l2).unwrap()
            - gabriel_graph_property(&g_unit, &lnd2).unwrap())
        .abs()
            < 1e-13
    );
    assert_eq!(
        gabriel_graph_property(&g, &lnd2).unwrap().to_bits(),
        gabriel_graph_property_with_index(&g, &lnd2, &idx2)
            .unwrap()
            .to_bits()
    );
    assert!((node_resolution(&l2).unwrap() - node_resolution(&lnd2).unwrap()).abs() < 1e-13);
    assert_eq!(
        node_resolution(&lnd2).unwrap().to_bits(),
        node_resolution_with_index(&lnd2, &idx2).unwrap().to_bits()
    );
    assert_eq!(
        relative_neighborhood_edges(&l2).unwrap(),
        relative_neighborhood_edges(&lnd2).unwrap()
    );
    assert_eq!(
        gabriel_graph_edges(&l2).unwrap(),
        gabriel_graph_edges(&lnd2).unwrap()
    );
    assert_eq!(
        relative_neighborhood_preservation(&g, &l2)
            .unwrap()
            .to_bits(),
        relative_neighborhood_preservation(&g_unit, &lnd2)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        gabriel_graph_preservation(&g, &l2).unwrap().to_bits(),
        gabriel_graph_preservation(&g_unit, &lnd2)
            .unwrap()
            .to_bits()
    );

    // Layout3d vs LayoutNd(dims=3) vs CustomMetricCloud3d parity.
    let l3 = Layout3d::from_xyz(
        vec![0.0, 1.2, 0.4, 2.1, -0.5, 1.8],
        vec![0.0, 0.3, 1.6, 0.8, 1.1, -0.7],
        vec![0.0, 0.9, -0.4, 1.3, 0.5, 0.2],
    )
    .unwrap();
    let mut flat3 = Vec::with_capacity(18);
    let mut pts3 = Vec::with_capacity(6);
    for i in 0..6 {
        flat3.extend_from_slice(&[l3.x()[i], l3.y()[i], l3.z()[i]]);
        pts3.push([l3.x()[i], l3.y()[i], l3.z()[i]]);
    }
    let lnd3 = LayoutNd::from_flat(6, 3, &flat3).unwrap();
    let cust = CustomMetricCloud3d { pts: pts3 };
    let idx3 = StaticKdTree::<3, f64>::from_layout_3d(&l3).unwrap();

    assert_eq!(
        gabriel_graph_edges(&l3).unwrap(),
        gabriel_graph_edges(&lnd3).unwrap()
    );
    assert_eq!(
        gabriel_graph_edges(&l3).unwrap(),
        gabriel_graph_edges(&cust).unwrap()
    );
    assert_eq!(
        gabriel_graph_edges(&l3).unwrap(),
        gabriel_graph_edges_with_index(&l3, &idx3).unwrap()
    );
    assert_eq!(
        relative_neighborhood_edges(&l3).unwrap(),
        relative_neighborhood_edges(&cust).unwrap()
    );
    assert!(
        (gabriel_graph_property(&g, &l3).unwrap() - gabriel_graph_property(&g, &cust).unwrap())
            .abs()
            < 1e-12
    );
    assert!((node_resolution(&l3).unwrap() - node_resolution(&cust).unwrap()).abs() < 1e-12);
}

#[test]
fn test_proximity_brute_oracle_across_dims_ties_and_coincident() {
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
        for d in 0..dims {
            coords[d] = 0.0;
            coords[dims + d] = 0.0;
            coords[2 * dims + d] = if d == 0 { 1.0 } else { 0.0 };
            coords[3 * dims + d] = if d == dims - 1 { 1.0 } else { 0.0 };
        }

        let layout = LayoutNd::from_flat(n, dims, &coords).unwrap();
        let idx = build_spatial_index(&layout).unwrap();

        let gg = gabriel_graph_edges(&layout).unwrap();
        let rng_edges = relative_neighborhood_edges(&layout).unwrap();
        assert_eq!(gg, brute_gg_edges_flat(&layout));
        assert_eq!(rng_edges, brute_rng_edges_flat(&layout));
        assert_eq!(gg, gabriel_graph_edges_with_index(&layout, &idx).unwrap());
        assert_eq!(
            rng_edges,
            relative_neighborhood_edges_with_index(&layout, &idx).unwrap()
        );

        let nr = node_resolution(&layout).unwrap();
        let nr_exp = brute_node_resolution_flat(&layout);
        assert!((nr - nr_exp).abs() < 1e-10);
        assert_eq!(
            nr.to_bits(),
            node_resolution_with_index(&layout, &idx).unwrap().to_bits()
        );

        let g =
            CsrGraph::<f64>::from_edges(n, &gg[..gg.len().min(30)], None, false, false).unwrap();
        let gp = gabriel_graph_property(&g, &layout).unwrap();
        let gp_exp = brute_gabriel_property_flat(&g, &layout);
        assert!((gp - gp_exp).abs() < 1e-10);
        assert_eq!(
            gp.to_bits(),
            gabriel_graph_property_with_index(&g, &layout, &idx)
                .unwrap()
                .to_bits()
        );

        let s_rng = relative_neighborhood_preservation(&g, &layout).unwrap();
        let s_gg = gabriel_graph_preservation(&g, &layout).unwrap();
        assert!((0.0..=1.0).contains(&s_rng));
        assert!((0.0..=1.0).contains(&s_gg));
        assert_eq!(
            s_rng.to_bits(),
            relative_neighborhood_preservation_with_index(&g, &layout, &idx)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            s_gg.to_bits(),
            gabriel_graph_preservation_with_index(&g, &layout, &idx)
                .unwrap()
                .to_bits()
        );
    }

    let mut cube_coords = Vec::new();
    for z in [0.0, 1.0] {
        for y in [0.0, 1.0] {
            for x in [0.0, 1.0] {
                cube_coords.extend_from_slice(&[x, y, z]);
            }
        }
    }
    let cube = LayoutNd::from_flat(8, 3, &cube_coords).unwrap();
    let cube_gg = gabriel_graph_edges(&cube).unwrap();
    let cube_rng = relative_neighborhood_edges(&cube).unwrap();
    assert_eq!(cube_gg.len(), 12);
    assert_eq!(cube_rng.len(), 12);
    let g_cube = CsrGraph::<f64>::from_edges(8, &cube_gg, None, false, false).unwrap();
    assert!((gabriel_graph_preservation(&g_cube, &cube).unwrap() - 1.0).abs() < 1e-12);
    assert!((relative_neighborhood_preservation(&g_cube, &cube).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn test_proximity_3d_f32_and_manifold_kinds() {
    let l32_corner = Layout3d::from_xyz(
        vec![0.0_f32, 1.0, 0.0, 0.0],
        vec![0.0_f32, 0.0, 1.0, 0.0],
        vec![0.0_f32, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let g32 =
        CsrGraph::<f32>::from_edges(4, &[(0, 1), (0, 2), (0, 3)], None, false, false).unwrap();
    assert_eq!(gabriel_graph_edges(&l32_corner).unwrap().len(), 3);
    assert_eq!(relative_neighborhood_edges(&l32_corner).unwrap().len(), 3);
    assert!((gabriel_graph_preservation(&g32, &l32_corner).unwrap() - 1.0).abs() < 1e-12);

    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        let mut ml = ManifoldLayout2d::<f64>::init(kind, 14, 77, 0.7).unwrap();
        if kind == ManifoldKind::Torus2d {
            ml.set_torus_extents(2.5, 0.8).unwrap();
        } else {
            ml.set_scale(2.2).unwrap();
        }
        let rng = relative_neighborhood_edges(&ml).unwrap();
        let gg = gabriel_graph_edges(&ml).unwrap();
        assert!(!rng.is_empty());
        assert!(!gg.is_empty());
        let g = CsrGraph::<f64>::from_edges(14, &gg, None, false, false).unwrap();
        assert!((gabriel_graph_preservation(&g, &ml).unwrap() - 1.0).abs() < 1e-12);
        assert!(gabriel_graph_property(&g, &ml).unwrap() < 1e-12);
        let srng = relative_neighborhood_preservation(&g, &ml).unwrap();
        assert!((0.0..=1.0).contains(&srng));
    }

    // Collinear triple on Torus2d -> Gabriel violation = 0.15^2
    let g_col = CsrGraph::<f64>::from_edges(3, &[(0, 2)], None, false, false).unwrap();
    let torus = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.1, 0.25, 0.4],
        vec![0.5, 0.5, 0.5],
    )
    .unwrap();
    let viol = gabriel_graph_property(&g_col, &torus).unwrap();
    assert!((viol - 0.15_f64.powi(2)).abs() < 1e-12, "got {viol}");

    // Collapsed sphere north pole -> node_resolution == 6.0
    let collapsed = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Spherical,
        vec![0.1, 0.5, 1.0, 2.0],
        vec![0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    assert_eq!(node_resolution(&collapsed).unwrap(), 6.0);
}
