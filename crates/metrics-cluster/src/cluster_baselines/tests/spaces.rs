use super::super::space_dispatch::build_spatial_index;
use super::*;
use egraph_core::{
    CoreError, CoreResult, CsrGraph, Layout3d, LayoutNd, LayoutStorage, MetricSpace, NodeExtents2d,
};
use egraph_spatial_trees::{DynamicKdTree, StaticKdTree};

struct CustomMetricCloud3d {
    pts: Vec<[f64; 3]>,
}

impl LayoutStorage for CustomMetricCloud3d {
    type Scalar = f64;

    fn len(&self) -> usize {
        self.pts.len()
    }

    fn validate_finite(&self) -> CoreResult<()> {
        if self.pts.iter().all(|p| p.iter().all(|v| v.is_finite())) {
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

fn layout_nd_from_2d(l2: &Layout2d<f64>) -> LayoutNd<f64> {
    let n = l2.len();
    let mut flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        flat.push(l2.x()[i]);
        flat.push(l2.y()[i]);
    }
    LayoutNd::from_flat(n, 2, &flat).unwrap()
}

fn layout_nd_from_3d(l3: &Layout3d<f64>) -> LayoutNd<f64> {
    let n = l3.len();
    let mut flat = Vec::with_capacity(n * 3);
    for i in 0..n {
        flat.push(l3.x()[i]);
        flat.push(l3.y()[i]);
        flat.push(l3.z()[i]);
    }
    LayoutNd::from_flat(n, 3, &flat).unwrap()
}

#[test]
fn test_layout2d_layoutnd_and_layout3d_exact_parity() {
    let x_b = vec![
        -1.1897, -2.8445, 0.9574, -0.4679, -0.4015, 1.0801, -0.0962, 0.2414, -0.6056, -0.1957,
        -0.7159, -0.3484, 0.9628, -1.0698, -1.845, -1.7543, 1.2509, -1.3283, -0.7785, 0.7598,
    ];
    let y_b = vec![
        0.3609, 2.0937, -0.4381, 0.4558, -0.3389, 0.7721, -0.1282, -0.921, 0.8224, -2.0616, 0.9849,
        -0.2231, 2.7369, 2.0223, 0.2625, 2.0272, 1.7066, 1.0268, -0.6861, 1.3151,
    ];
    let z_b: Vec<f64> = (0..20)
        .map(|i| ((i * 7) % 11) as f64 * 0.35 - 1.5)
        .collect();
    let l2 = Layout2d::from_xy(x_b.clone(), y_b.clone()).unwrap();
    let lnd2 = layout_nd_from_2d(&l2);
    let l3 = Layout3d::from_xyz(x_b, y_b, z_b).unwrap();
    let lnd3 = layout_nd_from_3d(&l3);
    let labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3];

    assert_close(
        distance_consistency(&lnd2, &labels).unwrap(),
        distance_consistency(&l2, &labels).unwrap(),
        1e-12,
        "dim2.dsc",
    );
    assert_close(
        distribution_consistency(&lnd2, &labels, 16).unwrap(),
        distribution_consistency(&l2, &labels, 16).unwrap(),
        1e-12,
        "dim2.dc",
    );
    assert_close(
        silhouette_score(&lnd2, &labels).unwrap(),
        silhouette_score(&l2, &labels).unwrap(),
        1e-12,
        "dim2.sil",
    );
    assert_close(
        calinski_harabasz_score(&lnd2, &labels).unwrap(),
        calinski_harabasz_score(&l2, &labels).unwrap(),
        1e-12,
        "dim2.ch",
    );
    assert_close(
        davies_bouldin_score(&lnd2, &labels).unwrap(),
        davies_bouldin_score(&l2, &labels).unwrap(),
        1e-12,
        "dim2.db",
    );
    assert_close(
        pairwise_distance_overlap_error(&lnd2, &labels, 10).unwrap(),
        pairwise_distance_overlap_error(&l2, &labels, 10).unwrap(),
        1e-12,
        "dim2.pdoe",
    );
    let km2 = cq_kmeans(&l2, &labels, 5, 100, 42).unwrap();
    let km_nd = cq_kmeans(&lnd2, &labels, 5, 100, 42).unwrap();
    assert_close(km_nd.ari, km2.ari, 1e-12, "dim2.cq_kmeans.ari");
    assert_close(km_nd.ami, km2.ami, 1e-12, "dim2.cq_kmeans.ami");

    let hac2 = cq_hac(&l2, &labels).unwrap();
    let hac_nd = cq_hac(&lnd2, &labels).unwrap();
    assert_close(hac_nd.ari, hac2.ari, 1e-12, "dim2.cq_hac.ari");
    assert_close(hac_nd.ami, hac2.ami, 1e-12, "dim2.cq_hac.ami");

    assert_close(
        sepme_knng(&lnd2, &labels, 2).unwrap(),
        sepme_knng(&l2, &labels, 2).unwrap(),
        1e-12,
        "dim2.knng",
    );
    assert_close(
        sepme_rng(&lnd2, &labels).unwrap(),
        sepme_rng(&l2, &labels).unwrap(),
        1e-12,
        "dim2.rng",
    );
    assert_close(
        sepme_gab(&lnd2, &labels).unwrap(),
        sepme_gab(&l2, &labels).unwrap(),
        1e-12,
        "dim2.gab",
    );
    for gamma in [0.0, 0.35, 0.75] {
        assert_close(
            sepme_gong(&lnd2, &labels, gamma).unwrap(),
            sepme_gong(&l2, &labels, gamma).unwrap(),
            1e-12,
            &format!("dim2.gong({gamma})"),
        );
    }
    assert_close(
        neighborhood_hit(&lnd2, &labels, 7).unwrap(),
        neighborhood_hit(&l2, &labels, 7).unwrap(),
        1e-12,
        "dim2.m_nh",
    );

    // 3D SoA Layout3d vs 3D FlatNd parity across all 13 metrics + with_index.
    assert_close(
        distance_consistency(&l3, &labels).unwrap(),
        distance_consistency(&lnd3, &labels).unwrap(),
        1e-12,
        "dim3.dsc",
    );
    assert_close(
        distribution_consistency(&l3, &labels, 10).unwrap(),
        distribution_consistency(&lnd3, &labels, 10).unwrap(),
        1e-12,
        "dim3.dc",
    );
    assert_close(
        silhouette_score(&l3, &labels).unwrap(),
        silhouette_score(&lnd3, &labels).unwrap(),
        1e-12,
        "dim3.sil",
    );
    assert_close(
        calinski_harabasz_score(&l3, &labels).unwrap(),
        calinski_harabasz_score(&lnd3, &labels).unwrap(),
        1e-12,
        "dim3.ch",
    );
    assert_close(
        davies_bouldin_score(&l3, &labels).unwrap(),
        davies_bouldin_score(&lnd3, &labels).unwrap(),
        1e-12,
        "dim3.db",
    );
    assert_close(
        pairwise_distance_overlap_error(&l3, &labels, 10).unwrap(),
        pairwise_distance_overlap_error(&lnd3, &labels, 10).unwrap(),
        1e-12,
        "dim3.pdoe",
    );
    assert_close(
        cq_kmeans(&l3, &labels, 5, 100, 42).unwrap().ari,
        cq_kmeans(&lnd3, &labels, 5, 100, 42).unwrap().ari,
        1e-12,
        "dim3.cq_kmeans",
    );
    assert_close(
        cq_hac(&l3, &labels).unwrap().ari,
        cq_hac(&lnd3, &labels).unwrap().ari,
        1e-12,
        "dim3.cq_hac",
    );

    let idx3 = StaticKdTree::<3, f64>::from_euclidean(&l3).unwrap();
    let dyn3 = build_spatial_index(&lnd3).unwrap();
    assert_eq!(
        sepme_knng(&l3, &labels, 3).unwrap().to_bits(),
        sepme_knng_with_index(&l3, &labels, 3, &idx3)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        sepme_rng(&l3, &labels).unwrap().to_bits(),
        sepme_rng_with_index(&l3, &labels, &idx3).unwrap().to_bits()
    );
    assert_eq!(
        sepme_gab(&l3, &labels).unwrap().to_bits(),
        sepme_gab_with_index(&l3, &labels, &idx3).unwrap().to_bits()
    );
    assert_eq!(
        sepme_gong(&l3, &labels, 0.35).unwrap().to_bits(),
        sepme_gong_with_index(&l3, &labels, 0.35, &idx3)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        neighborhood_hit(&l3, &labels, 5).unwrap().to_bits(),
        neighborhood_hit_with_index(&l3, &labels, 5, &idx3)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        sepme_knng(&lnd3, &labels, 3).unwrap().to_bits(),
        sepme_knng_with_index(&lnd3, &labels, 3, dyn3.as_ref())
            .unwrap()
            .to_bits()
    );
}

#[test]
fn test_custom_metric_space_and_unit_graph_nor() {
    let pts = vec![
        [0.0, 0.0, 0.0],
        [0.2, 0.1, 0.0],
        [0.0, 0.2, 0.1],
        [5.0, 5.0, 5.0],
        [5.2, 5.1, 5.0],
        [5.0, 5.2, 5.1],
    ];
    let labels = [0_u32, 0, 0, 1, 1, 1];
    let l3 = Layout3d::from_xyz(
        pts.iter().map(|p| p[0]).collect(),
        pts.iter().map(|p| p[1]).collect(),
        pts.iter().map(|p| p[2]).collect(),
    )
    .unwrap();
    let custom = CustomMetricCloud3d { pts };

    assert_close(
        silhouette_score(&custom, &labels).unwrap(),
        silhouette_score(&l3, &labels).unwrap(),
        1e-12,
        "custom.sil",
    );
    assert_close(
        pairwise_distance_overlap_error(&custom, &labels, 10).unwrap(),
        pairwise_distance_overlap_error(&l3, &labels, 10).unwrap(),
        1e-12,
        "custom.pdoe",
    );
    assert_eq!(
        cq_hac(&custom, &labels).unwrap(),
        cq_hac(&l3, &labels).unwrap()
    );
    assert_close(
        neighborhood_hit(&custom, &labels, 2).unwrap(),
        neighborhood_hit(&l3, &labels, 2).unwrap(),
        1e-12,
        "custom.nh",
    );
    assert_close(
        sepme_knng(&custom, &labels, 2).unwrap(),
        sepme_knng(&l3, &labels, 2).unwrap(),
        1e-12,
        "custom.knng",
    );
    assert_close(
        sepme_rng(&custom, &labels).unwrap(),
        sepme_rng(&l3, &labels).unwrap(),
        1e-12,
        "custom.rng",
    );
    assert_close(
        sepme_gab(&custom, &labels).unwrap(),
        sepme_gab(&l3, &labels).unwrap(),
        1e-12,
        "custom.gab",
    );
    for gamma in [0.0, 0.35, 0.75] {
        assert_close(
            sepme_gong(&custom, &labels, gamma).unwrap(),
            sepme_gong(&l3, &labels, gamma).unwrap(),
            1e-12,
            &format!("custom.gong({gamma})"),
        );
    }

    for res in [
        distance_consistency(&custom, &labels),
        distribution_consistency(&custom, &labels, 8),
        calinski_harabasz_score(&custom, &labels),
        davies_bouldin_score(&custom, &labels),
    ] {
        assert!(matches!(res, Err(CoreError::UnsupportedOperation(_))));
    }
    assert!(matches!(
        cq_kmeans(&custom, &labels, 2, 10, 42),
        Err(CoreError::UnsupportedOperation(_))
    ));

    let g_unit =
        CsrGraph::<()>::from_unweighted_edges(4, &[(0, 1), (1, 2), (2, 3)], false, false).unwrap();
    let orig = layout_from_pairs(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
    let adj = layout_from_pairs(&[(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0)]);
    let nor = nor_metrics(
        &g_unit,
        &orig,
        &adj,
        NodeExtents2d::uniform(4, 0.2, 0.2).unwrap(),
    )
    .unwrap();
    assert_close(nor.el_rsd, 0.0, 1e-12, "unit_graph.nor.el_rsd");

    let l_dim4 = LayoutNd::from_flat(2, 4, &[0.0; 8]).unwrap();
    let idx_dim4 = DynamicKdTree::<f64>::from_euclidean(&l_dim4).unwrap();
    let l_dim3 = Layout3d::<f64>::zeros(2);
    assert!(matches!(
        sepme_knng_with_index(&l_dim3, &[0, 1], 1, &idx_dim4),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
