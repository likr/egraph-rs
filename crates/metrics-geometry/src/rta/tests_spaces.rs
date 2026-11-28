use egraph_core::{
    CoreError, CoreResult, CsrGraph, Layout2d, Layout3d, LayoutNd, LayoutStorage, ManifoldKind,
    ManifoldLayout2d, MetricSpace,
};

use super::*;

struct CustomMetricLine {
    coords: Vec<f64>,
}

impl LayoutStorage for CustomMetricLine {
    type Scalar = f64;

    fn len(&self) -> usize {
        self.coords.len()
    }

    fn validate_finite(&self) -> CoreResult<()> {
        if self.coords.iter().all(|v| v.is_finite()) {
            Ok(())
        } else {
            Err(CoreError::non_finite("CustomMetricLine"))
        }
    }
}

impl MetricSpace for CustomMetricLine {
    fn distance(&self, i: usize, j: usize) -> f64 {
        (self.coords[i] - self.coords[j]).abs()
    }
}

#[test]
fn test_rta_2d_3d_nd_custom_metric_parity_and_unit_graph() {
    let edges: Vec<(u32, u32)> = (0..11).map(|i| (i, i + 1)).collect();
    let g = CsrGraph::<f64>::from_edges(12, &edges, None, false, false).unwrap();
    let g_unit = CsrGraph::<()>::from_unweighted_edges(12, &edges, false, false).unwrap();
    let l2 = Layout2d::<f64>::init(12, 42, 10.0).unwrap();
    let flat = l2.to_interleaved();
    let lnd = LayoutNd::from_flat(12, 2, &flat).unwrap();

    assert_eq!(
        landmark_stratified_rta(&g, &l2, 4, 200, 1, 7)
            .unwrap()
            .to_bits(),
        landmark_stratified_rta(&g_unit, &lnd, 4, 200, 1, 7)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        landmark_stratified_rta_detail(&g, &l2, 4, 200, 1, 7).unwrap(),
        landmark_stratified_rta_detail(&g, &lnd, 4, 200, 1, 7).unwrap()
    );

    let landmark_ids = vec![0u32, 11];
    let dists_u32: Vec<u32> = (0..12)
        .map(|i| i as u32)
        .chain((0..12).map(|i| (11 - i) as u32))
        .collect();
    let dists_f64: Vec<f64> = dists_u32.iter().map(|&d| d as f64).collect();
    assert_eq!(
        landmark_stratified_rta_from_landmarks(&l2, &landmark_ids, &dists_u32, 100, 5).unwrap(),
        landmark_stratified_rta_from_landmarks(&lnd, &landmark_ids, &dists_u32, 100, 5).unwrap()
    );
    assert_eq!(
        landmark_stratified_rta_from_landmarks_f64(&l2, &landmark_ids, &dists_f64, 100, 5).unwrap(),
        landmark_stratified_rta_from_landmarks_f64(&lnd, &landmark_ids, &dists_f64, 100, 5)
            .unwrap()
    );

    // Layout3d vs LayoutNd(dims=3) parity.
    let l3 = Layout3d::<f64>::init(12, 7, 3.0).unwrap();
    let mut flat3 = Vec::with_capacity(36);
    for i in 0..12 {
        flat3.extend_from_slice(&[l3.x()[i], l3.y()[i], l3.z()[i]]);
    }
    let lnd3 = LayoutNd::from_flat(12, 3, &flat3).unwrap();
    assert!(
        (landmark_stratified_rta(&g, &l3, 4, 200, 1, 7).unwrap()
            - landmark_stratified_rta(&g, &lnd3, 4, 200, 1, 7).unwrap())
        .abs()
            < 1e-14
    );

    // Custom MetricSpace line embedding achieves RTA == 1.0.
    let custom = CustomMetricLine {
        coords: (0..12).map(|i| i as f64).collect(),
    };
    let ev_custom = landmark_stratified_rta_detail(&g_unit, &custom, 3, 500, 1, 13).unwrap();
    assert!((ev_custom.rta - 1.0).abs() < 1e-12);
    assert!(ev_custom.is_exact);

    let g32 = CsrGraph::<f32>::from_edges(12, &edges, None, false, false).unwrap();
    let lnd32 =
        LayoutNd::from_flat(12, 2, &flat.iter().map(|&v| v as f32).collect::<Vec<_>>()).unwrap();
    let rta32 = landmark_stratified_rta(&g32, &lnd32, 4, 200, 1, 7).unwrap();
    assert!((rta32 - landmark_stratified_rta(&g, &lnd, 4, 200, 1, 7).unwrap()).abs() < 1e-6);
}

#[test]
fn test_rta_multi_dims_exact_path_and_from_landmarks() {
    let n = 10usize;
    let edges: Vec<(u32, u32)> = (0..n as u32 - 1).map(|i| (i, i + 1)).collect();
    let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();

    for dims in [1usize, 3, 4, 8, 10] {
        let mut coords = vec![0.0_f64; n * dims];
        for i in 0..n {
            coords[i * dims] = i as f64;
        }
        let layout = LayoutNd::from_flat(n, dims, &coords).unwrap();
        let ev = landmark_stratified_rta_detail(&g, &layout, 3, 500, 1, 13).unwrap();
        assert!((ev.rta - 1.0).abs() < 1e-12);
        assert!(ev.is_exact);

        let ev_sampled = landmark_stratified_rta_detail(&g, &layout, 3, 12, 1, 13).unwrap();
        assert!((ev_sampled.rta - 1.0).abs() < 1e-12);
        assert!(!ev_sampled.is_exact);
    }

    let layout3d = Layout3d::from_xyz(
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 2.0, 0.0],
        vec![0.0, 0.0, 0.0, 5.0],
    )
    .unwrap();
    let eval_inf = landmark_stratified_rta_from_landmarks_f64(
        &layout3d,
        &[0],
        &[0.0, 1.0, 2.0, f64::INFINITY],
        10,
        1,
    )
    .unwrap();
    assert!((eval_inf.rta - 1.0).abs() < 1e-12);
    assert_eq!(eval_inf.num_triplets, 1);
}

#[test]
fn test_rta_spaces_boundaries_and_errors() {
    for n in [0usize, 1, 2] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let l = LayoutNd::<f64>::zeros(n, 3).unwrap();
        assert_eq!(landmark_stratified_rta(&g, &l, 2, 10, 1, 1).unwrap(), 0.0);
    }

    let g_edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let l4 = LayoutNd::<f64>::init(4, 3, 5, 1.0).unwrap();
    assert_eq!(
        landmark_stratified_rta(&g_edgeless, &l4, 2, 10, 1, 1).unwrap(),
        0.0
    );

    let g4 = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
    let l_col = LayoutNd::<f64>::zeros(4, 4).unwrap();
    assert_eq!(
        landmark_stratified_rta(&g4, &l_col, 2, 10, 1, 1).unwrap(),
        0.0
    );

    let l_nan = LayoutNd::from_flat(4, 2, &[0.0, f64::NAN, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
    assert!(matches!(
        landmark_stratified_rta(&g4, &l_nan, 2, 10, 1, 1),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        landmark_stratified_rta(&g4, &l4, 0, 10, 1, 1),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        landmark_stratified_rta(&g4, &l4, 2, 0, 1, 1),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        landmark_stratified_rta(&g4, &LayoutNd::<f64>::zeros(3, 3).unwrap(), 2, 10, 1, 1),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        landmark_stratified_rta_from_landmarks_f64(&l4, &[0], &[0.0, -1.0, 2.0, 3.0], 10, 1),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        landmark_stratified_rta_from_landmarks_f64(&l4, &[4], &[0.0, 1.0, 2.0, 3.0], 10, 1),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn rta_manifold_all_kinds_and_collapsed() {
    let g = CsrGraph::<f64>::from_edges(
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
        let mut ml = ManifoldLayout2d::<f64>::init(kind, 6, 42, 0.5).unwrap();
        let ev1 = landmark_stratified_rta_detail(&g, &ml, 4, 1000, 1, 42).unwrap();
        assert!(ev1.is_exact);
        assert!((0.0..=1.0).contains(&ev1.rta));

        ml.set_scale(3.0).unwrap();
        let rta2 = landmark_stratified_rta(&g, &ml, 4, 1000, 1, 42).unwrap();
        assert!((ev1.rta - rta2).abs() < 1e-12);
    }

    let lor = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::HyperbolicLorentz,
        vec![0.0, 0.5, 1.5, 3.0, 6.0, 10.0],
        vec![0.0; 6],
    )
    .unwrap();
    let ev = landmark_stratified_rta_detail(&g, &lor, 2, 1000, 1, 1).unwrap();
    assert!((ev.rta - 1.0).abs() < 1e-12);

    let g4 = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
    let north_pole = ManifoldLayout2d::<f32>::from_xy(
        ManifoldKind::Spherical,
        vec![0.0, 1.0, 2.0, 3.0],
        vec![0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    assert_eq!(
        landmark_stratified_rta(&g4, &north_pole, 2, 100, 1, 42).unwrap(),
        0.0
    );
}
