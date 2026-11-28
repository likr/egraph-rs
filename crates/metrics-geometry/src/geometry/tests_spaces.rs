use egraph_core::{
    CoreError, CoreResult, CsrGraph, Layout2d, Layout3d, LayoutNd, LayoutStorage, ManifoldKind,
    ManifoldLayout2d, MetricSpace, RiemannianLayout,
};

use crate::spatial_index::build_spatial_index;

use super::*;

struct CustomRiemannianSpace {
    pts: Vec<[f64; 3]>,
}

impl LayoutStorage for CustomRiemannianSpace {
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
            Err(CoreError::non_finite("CustomRiemannianSpace"))
        }
    }
}

impl MetricSpace for CustomRiemannianSpace {
    fn distance(&self, i: usize, j: usize) -> f64 {
        let a = self.pts[i];
        let b = self.pts[j];
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl RiemannianLayout for CustomRiemannianSpace {
    fn tangent_dim(&self) -> usize {
        3
    }

    fn log_map_into(&self, i: usize, j: usize, out: &mut [f64]) -> CoreResult<()> {
        for d in 0..3 {
            out[d] = self.pts[j][d] - self.pts[i][d];
        }
        Ok(())
    }

    fn exp_map(&mut self, i: usize, delta: &[f64]) -> CoreResult<()> {
        for d in 0..3 {
            self.pts[i][d] += delta[d];
        }
        Ok(())
    }

    fn parallel_transport_into(
        &self,
        _i: usize,
        _delta: &[f64],
        _vec: &mut [f64],
    ) -> CoreResult<()> {
        Ok(())
    }
}

#[test]
fn test_geometry_2d_3d_nd_custom_riemannian_parity_and_unit_graph() {
    let g_unw =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], None, false, false)
            .unwrap();
    let g_unit =
        CsrGraph::<()>::from_unweighted_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], false, false)
            .unwrap();
    let g_w = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3), (3, 0)],
        Some(&[1.0, 2.0, 1.5, 2.5]),
        false,
        false,
    )
    .unwrap();
    let l2 = Layout2d::from_xy(vec![0.0, 1.0, 1.0, 0.0], vec![-0.0, 0.0, 2.0, 2.0]).unwrap();
    let flat = l2.to_interleaved();
    let lnd2 = LayoutNd::from_flat(4, 2, &flat).unwrap();
    let idx2 = build_spatial_index(&lnd2).unwrap();

    assert_eq!(
        ideal_edge_lengths(&g_unw, &l2).unwrap().to_bits(),
        ideal_edge_lengths(&g_unw, &lnd2).unwrap().to_bits()
    );
    assert_eq!(
        ideal_edge_lengths(&g_w, &l2).unwrap().to_bits(),
        ideal_edge_lengths(&g_w, &lnd2).unwrap().to_bits()
    );
    assert_eq!(
        ideal_edge_lengths_with_unit(&g_unit, &l2, 2.5)
            .unwrap()
            .to_bits(),
        ideal_edge_lengths_with_unit(&g_unit, &lnd2, 2.5)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        ideal_edge_lengths_scaled(&g_w, &l2, 1.5).unwrap().to_bits(),
        ideal_edge_lengths_scaled(&g_w, &lnd2, 1.5)
            .unwrap()
            .to_bits()
    );

    let dists = vec![
        0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0,
    ];
    assert_eq!(
        ideal_edge_lengths_from_distances(&g_unit, &l2, &dists)
            .unwrap()
            .to_bits(),
        ideal_edge_lengths_from_distances(&g_unit, &lnd2, &dists)
            .unwrap()
            .to_bits()
    );

    assert_eq!(
        angular_resolution(&g_unw, &l2).unwrap().to_bits(),
        angular_resolution(&g_unit, &lnd2).unwrap().to_bits()
    );

    let deg_2 = layout_degeneracy(&g_unw, &l2).unwrap();
    let deg_flat = layout_degeneracy(&g_unit, &lnd2).unwrap();
    let deg_idx = layout_degeneracy_with_index(&g_unit, &lnd2, &idx2).unwrap();
    assert_eq!(deg_2, deg_flat);
    assert_eq!(deg_flat, deg_idx);

    // Layout3d vs CustomRiemannianSpace parity.
    let l3_ortho = Layout3d::from_xyz(
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let custom = CustomRiemannianSpace {
        pts: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    };
    let g_star =
        CsrGraph::<()>::from_unweighted_edges(4, &[(0, 1), (0, 2), (0, 3)], false, false).unwrap();
    assert!(ideal_edge_lengths_with_unit(&g_star, &l3_ortho, 1.0).unwrap() < 1e-12);
    assert!(ideal_edge_lengths_with_unit(&g_star, &custom, 1.0).unwrap() < 1e-12);
    assert!(
        (angular_resolution(&g_star, &l3_ortho).unwrap()
            - angular_resolution(&g_star, &custom).unwrap())
        .abs()
            < 1e-12
    );
    let deg_cust = layout_degeneracy(&g_star, &custom).unwrap();
    assert_eq!(deg_cust.duplicate_nodes_f64, 0);
    assert_eq!(deg_cust.zero_length_edges, 0);
    assert!(!deg_cust.is_bbox_degenerate);
}

#[test]
fn test_geometry_multi_dims_degeneracy_and_ideal_lengths() {
    let g3 = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    let l1d = LayoutNd::from_flat(3, 1, &[0.0, 2.0, 5.0]).unwrap();
    let deg1 = layout_degeneracy(&g3, &l1d).unwrap();
    assert_eq!(deg1.bbox_width, 5.0);
    assert_eq!(deg1.bbox_height, 0.0);
    assert!(!deg1.is_bbox_degenerate);
    assert_eq!(deg1.near_collision_ratio, 0.0);

    let l3d_flat_z = Layout3d::from_xyz(
        vec![0.0, 10.0, 0.0, 10.0],
        vec![0.0, 0.0, 10.0, 10.0],
        vec![0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let g4 = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
    let deg_flat_z = layout_degeneracy(&g4, &l3d_flat_z).unwrap();
    assert!(deg_flat_z.is_bbox_degenerate);
    assert_eq!(deg_flat_z.bbox_width, 10.0);
    assert_eq!(deg_flat_z.bbox_height, 10.0);

    let dims = 10usize;
    let mut coords = vec![0.0_f64; 5 * dims];
    coords[0] = 1.0;
    coords[dims] = 1.0 + 1e-12;
    coords[2 * dims] = 1.0;
    coords[2 * dims + 1] = -0.0;
    for d in 0..dims {
        coords[3 * dims + d] = 2.0;
    }
    coords[4 * dims + 9] = 3.0;

    let g5 = CsrGraph::<f64>::from_edges(5, &[(0, 2), (0, 3), (3, 4)], None, false, false).unwrap();
    let l10 = LayoutNd::from_flat(5, dims, &coords).unwrap();
    let deg10 = layout_degeneracy(&g5, &l10).unwrap();
    assert_eq!(deg10.duplicate_nodes_f64, 2);
    assert_eq!(deg10.duplicate_nodes_f32, 3);
    assert!((deg10.excess_duplicate_ratio - 1.0 / 5.0).abs() < 1e-12);
    assert!((deg10.near_collision_ratio - 3.0 / 5.0).abs() < 1e-12);
    assert_eq!(deg10.zero_length_edges, 1);
    assert!(!deg10.is_bbox_degenerate);
}

#[test]
fn geometry_manifold_ideal_edge_lengths_and_degeneracy() {
    let g_unw = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    let mut torus = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.0, 0.25, 0.5],
        vec![0.0, 0.0, 0.0],
    )
    .unwrap();
    torus.set_torus_extents(4.0, 1.0).unwrap();
    assert!(ideal_edge_lengths(&g_unw, &torus).unwrap() < 1e-12);
    assert!(ideal_edge_lengths_with_unit(&g_unw, &torus, 1.0).unwrap() < 1e-12);
    assert!(ideal_edge_lengths_scaled(&g_unw, &torus, 1.0).unwrap() < 1e-12);

    let g_w =
        CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], Some(&[2.0, 2.0]), false, false).unwrap();
    assert!((ideal_edge_lengths(&g_w, &torus).unwrap() - 0.5).abs() < 1e-12);
    assert!((ideal_edge_lengths_scaled(&g_w, &torus, 0.5).unwrap()).abs() < 1e-12);

    let dists = vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
    assert!(ideal_edge_lengths_from_distances(&g_unw, &torus, &dists).unwrap() < 1e-12);

    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
    let sph = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Spherical,
        vec![0.1, 1.7, 0.5, 0.5],
        vec![0.0, 0.0, 1.0, 1.0],
    )
    .unwrap();
    let deg = layout_degeneracy(&g, &sph).unwrap();
    assert_eq!(deg.duplicate_nodes_f64, 2);
    assert!((deg.excess_duplicate_ratio - 0.25).abs() < 1e-12);
    assert_eq!(deg.zero_length_edges, 2);
    assert!((deg.zero_length_edge_ratio - 2.0 / 3.0).abs() < 1e-12);
    assert!((deg.near_collision_ratio - 1.0).abs() < 1e-12);

    let g2 = CsrGraph::<f64>::from_edges(3, &[(0, 2)], None, false, false).unwrap();
    let torus_wrap = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.0, 1.0 - 1e-9, 0.5],
        vec![0.5, 0.5, 0.5],
    )
    .unwrap();
    let deg_t = layout_degeneracy(&g2, &torus_wrap).unwrap();
    assert_eq!(deg_t.duplicate_nodes_f64, 0);
    assert!((deg_t.near_collision_ratio - 2.0 / 3.0).abs() < 1e-12);

    let lor = ManifoldLayout2d::<f32>::init(ManifoldKind::HyperbolicLorentz, 4, 42, 1.0).unwrap();
    let deg_l = layout_degeneracy(&g, &lor).unwrap();
    assert_eq!(deg_l.duplicate_nodes_f64, 0);
    assert_eq!(deg_l.zero_length_edges, 0);
    assert_eq!(deg_l.near_collision_ratio, 0.0);
    assert!(angular_resolution(&g, &lor).unwrap().is_finite());
}
