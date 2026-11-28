use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, Layout3d, LayoutNd, LayoutStorage,
    ManifoldKind, ManifoldLayout2d, MetricSpace,
};
use egraph_spatial_trees::StaticKdTree;

use crate::spatial_index::build_spatial_index;

use super::*;

struct CustomMetricPoints {
    pts: Vec<[f64; 3]>,
}

impl LayoutStorage for CustomMetricPoints {
    type Scalar = f64;

    fn len(&self) -> usize {
        self.pts.len()
    }

    fn validate_finite(&self) -> CoreResult<()> {
        if self.pts.iter().all(|p| p.iter().all(|v| v.is_finite())) {
            Ok(())
        } else {
            Err(CoreError::non_finite("CustomMetricPoints"))
        }
    }
}

impl MetricSpace for CustomMetricPoints {
    fn distance(&self, i: usize, j: usize) -> f64 {
        let a = self.pts[i];
        let b = self.pts[j];
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

#[test]
fn test_neighborhood_2d_3d_nd_custom_metric_parity_and_unit_graph() {
    let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2), (3, 5)];
    let g = CsrGraph::<f64>::from_edges(6, &edges, None, false, false).unwrap();
    let g_unit = CsrGraph::<()>::from_unweighted_edges(6, &edges, false, false).unwrap();
    let l2 = Layout2d::from_xy(
        vec![0.0, 1.0, 2.1, 3.3, 4.4, 5.6],
        vec![0.0, 0.5, -0.3, 0.2, 0.8, 0.1],
    )
    .unwrap();
    let flat = l2.to_interleaved();
    let lnd2 = LayoutNd::from_flat(6, 2, &flat).unwrap();
    let idx = build_spatial_index(&lnd2).unwrap();
    let static2 = StaticKdTree::<2, f64>::from_layout(&l2).unwrap();

    assert_eq!(
        neighborhood_preservation_1hop(&g, &l2).unwrap().to_bits(),
        neighborhood_preservation_1hop(&g_unit, &lnd2)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        neighborhood_preservation_1hop(&g, &l2).unwrap().to_bits(),
        neighborhood_preservation_1hop_with_index(&g, &l2, &static2)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        neighborhood_preservation_1hop(&g, &lnd2).unwrap().to_bits(),
        neighborhood_preservation_1hop_with_index(&g, &lnd2, &idx)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        neighborhood_preservation_2hop(&g, &l2).unwrap().to_bits(),
        neighborhood_preservation_2hop(&g, &lnd2).unwrap().to_bits()
    );
    assert_eq!(
        neighborhood_preservation_2hop(&g, &lnd2).unwrap().to_bits(),
        neighborhood_preservation_2hop_with_index(&g, &lnd2, &idx)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        sampled_neighborhood_preservation_1hop(&g, &l2, 4, 42)
            .unwrap()
            .to_bits(),
        sampled_neighborhood_preservation_1hop(&g, &lnd2, 4, 42)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        sampled_neighborhood_preservation_2hop(&g, &l2, 4, 42)
            .unwrap()
            .to_bits(),
        sampled_neighborhood_preservation_2hop(&g, &lnd2, 4, 42)
            .unwrap()
            .to_bits()
    );

    // Layout3d vs LayoutNd(dims=3) vs CustomMetricPoints parity.
    let l3 = Layout3d::from_xyz(
        vec![0.0, 1.0, 2.2, 3.1, 4.5, 5.2],
        vec![0.1, -0.4, 0.8, 0.2, -0.5, 0.3],
        vec![0.0, 0.3, -0.2, 0.7, 0.1, -0.4],
    )
    .unwrap();
    let mut flat3 = Vec::with_capacity(18);
    let mut cust_pts = Vec::with_capacity(6);
    for i in 0..6 {
        flat3.extend_from_slice(&[l3.x()[i], l3.y()[i], l3.z()[i]]);
        cust_pts.push([l3.x()[i], l3.y()[i], l3.z()[i]]);
    }
    let lnd3 = LayoutNd::from_flat(6, 3, &flat3).unwrap();
    let cust = CustomMetricPoints { pts: cust_pts };
    let idx3 = StaticKdTree::<3, f64>::from_layout_3d(&l3).unwrap();

    assert_eq!(
        neighborhood_preservation_1hop(&g, &l3).unwrap().to_bits(),
        neighborhood_preservation_1hop_with_index(&g, &l3, &idx3)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        neighborhood_preservation_1hop(&g, &l3).unwrap().to_bits(),
        neighborhood_preservation_1hop(&g, &lnd3).unwrap().to_bits()
    );
    assert_eq!(
        neighborhood_preservation_1hop(&g, &l3).unwrap().to_bits(),
        neighborhood_preservation_1hop(&g_unit, &cust)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        neighborhood_preservation_2hop(&g, &l3).unwrap().to_bits(),
        neighborhood_preservation_2hop(&g_unit, &cust)
            .unwrap()
            .to_bits()
    );

    let g32 = CsrGraph::<f32>::from_edges(6, &edges, None, false, false).unwrap();
    let lnd32 =
        LayoutNd::from_flat(6, 2, &flat.iter().map(|&v| v as f32).collect::<Vec<_>>()).unwrap();
    assert!(
        (neighborhood_preservation_1hop(&g32, &lnd32).unwrap()
            - neighborhood_preservation_1hop(&g, &lnd2).unwrap())
        .abs()
            < 1e-12
    );
}

#[test]
fn test_neighborhood_multi_dims_and_brute_force_oracle() {
    let edges = [
        (0u32, 1u32),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (0, 3),
        (2, 5),
    ];
    let g = CsrGraph::<f64>::from_edges(8, &edges, None, false, false).unwrap();

    let g_p8 = CsrGraph::<f64>::from_edges(
        8,
        &(0..7).map(|i| (i, i + 1)).collect::<Vec<_>>(),
        None,
        false,
        false,
    )
    .unwrap();
    let l1d = LayoutNd::from_flat(8, 1, &(0..8).map(|i| i as f64).collect::<Vec<_>>()).unwrap();
    assert!((neighborhood_preservation_1hop(&g_p8, &l1d).unwrap() - 1.0).abs() < 1e-12);
    assert!((neighborhood_preservation_2hop(&g_p8, &l1d).unwrap() - 1.0).abs() < 1e-12);

    let mut rng = DeterministicRng::from_seed_u64(515);
    for dims in [1usize, 3, 4, 8, 10] {
        let coords: Vec<f64> = (0..8 * dims).map(|_| rng.next_f64() * 5.0).collect();
        let layout = LayoutNd::from_flat(8, dims, &coords).unwrap();
        let idx = build_spatial_index(&layout).unwrap();

        let np1 = neighborhood_preservation_1hop(&g, &layout).unwrap();
        let np2 = neighborhood_preservation_2hop(&g, &layout).unwrap();
        assert!((0.0..=1.0).contains(&np1));
        assert!((0.0..=1.0).contains(&np2));
        assert_eq!(
            np1.to_bits(),
            neighborhood_preservation_1hop_with_index(&g, &layout, &idx)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            np2.to_bits(),
            neighborhood_preservation_2hop_with_index(&g, &layout, &idx)
                .unwrap()
                .to_bits()
        );

        let s1 = sampled_neighborhood_preservation_1hop(&g, &layout, 5, 17).unwrap();
        let s2 = sampled_neighborhood_preservation_2hop(&g, &layout, 5, 17).unwrap();
        assert_eq!(
            s1.to_bits(),
            sampled_neighborhood_preservation_1hop_with_index(&g, &layout, &idx, 5, 17)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            s2.to_bits(),
            sampled_neighborhood_preservation_2hop_with_index(&g, &layout, &idx, 5, 17)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            sampled_neighborhood_preservation_1hop(&g, &layout, 10, 17)
                .unwrap()
                .to_bits(),
            np1.to_bits()
        );
        assert_eq!(
            sampled_neighborhood_preservation_2hop(&g, &layout, 10, 17)
                .unwrap()
                .to_bits(),
            np2.to_bits()
        );
    }
}

#[test]
fn test_neighborhood_spaces_boundaries_and_errors() {
    for n in [0usize, 1] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let l = LayoutNd::<f64>::zeros(n, 3).unwrap();
        assert_eq!(neighborhood_preservation_1hop(&g, &l).unwrap(), 0.0);
        assert_eq!(
            sampled_neighborhood_preservation_1hop(&g, &l, 2, 1).unwrap(),
            0.0
        );
        let expected_np2 = if n == 0 { 0.0 } else { 1.0 };
        assert_eq!(
            neighborhood_preservation_2hop(&g, &l).unwrap(),
            expected_np2
        );
        assert_eq!(
            sampled_neighborhood_preservation_2hop(&g, &l, 2, 1).unwrap(),
            expected_np2
        );
    }

    let g_edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let l4 = LayoutNd::<f64>::init(4, 10, 9, 1.0).unwrap();
    assert_eq!(
        neighborhood_preservation_1hop(&g_edgeless, &l4).unwrap(),
        0.0
    );
    assert_eq!(
        neighborhood_preservation_2hop(&g_edgeless, &l4).unwrap(),
        1.0
    );

    let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let l_nan = LayoutNd::from_flat(2, 3, &[0.0, f64::NAN, 0.0, 1.0, 0.0, 0.0]).unwrap();
    assert!(matches!(
        neighborhood_preservation_1hop(&g2, &l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        neighborhood_preservation_2hop(&g2, &l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        neighborhood_preservation_1hop(&g2, &l4),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let l4_d3 = LayoutNd::<f64>::zeros(4, 3).unwrap();
    let idx_d10 = build_spatial_index(&l4).unwrap();
    assert!(matches!(
        neighborhood_preservation_1hop_with_index(&g_edgeless, &l4_d3, &idx_d10),
        Err(CoreError::DimensionMismatch { .. })
    ));
}

#[test]
fn neighborhood_manifold_torus_aspect_and_all_kinds() {
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let mut torus = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.1, 0.1, 0.25, 0.25],
        vec![0.1, 0.35, 0.1, 0.35],
    )
    .unwrap();
    assert_eq!(neighborhood_preservation_1hop(&g, &torus).unwrap(), 0.0);

    torus.set_torus_extents(4.0, 1.0).unwrap();
    assert_eq!(neighborhood_preservation_1hop(&g, &torus).unwrap(), 1.0);
    assert_eq!(neighborhood_preservation_2hop(&g, &torus).unwrap(), 1.0);
    assert_eq!(
        sampled_neighborhood_preservation_1hop(&g, &torus, 4, 42).unwrap(),
        1.0
    );
    assert_eq!(
        sampled_neighborhood_preservation_2hop(&g, &torus, 10, 42).unwrap(),
        1.0
    );

    let gp = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        None,
        false,
        false,
    )
    .unwrap();
    for kind in [
        ManifoldKind::HyperbolicPoincare,
        ManifoldKind::HyperbolicLorentz,
        ManifoldKind::Spherical,
        ManifoldKind::Torus2d,
    ] {
        let mut ml = ManifoldLayout2d::<f64>::init(kind, 6, 42, 0.6).unwrap();
        let np1 = neighborhood_preservation_1hop(&gp, &ml).unwrap();
        let np2 = neighborhood_preservation_2hop(&gp, &ml).unwrap();
        assert!((0.0..=1.0).contains(&np1));
        assert!((0.0..=1.0).contains(&np2));

        ml.set_scale(3.0).unwrap();
        assert_eq!(neighborhood_preservation_1hop(&gp, &ml).unwrap(), np1);
        assert_eq!(neighborhood_preservation_2hop(&gp, &ml).unwrap(), np2);

        let s1 = sampled_neighborhood_preservation_1hop(&gp, &ml, 6, 7).unwrap();
        let s2 = sampled_neighborhood_preservation_2hop(&gp, &ml, 6, 7).unwrap();
        assert_eq!(s1, np1);
        assert_eq!(s2, np2);
    }

    // Deterministic tie-breaking on tied distances.
    let sph = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Spherical,
        vec![0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let mut buf = Vec::new();
    super::k_nearest_metric_excluding_self_into(&sph, 0, 1, &mut buf);
    assert_eq!(buf.len(), 1);
    assert_eq!(buf[0].0, 1);
}
