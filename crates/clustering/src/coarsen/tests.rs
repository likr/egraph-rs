use egraph_core::{CoreError, CsrGraph, Layout2d};

use super::*;

#[test]
fn test_coarsen_and_project_roundtrip() {
    // Two triangles {0, 1, 2} and {3, 4, 5} linked by two edges (0, 3) and (2, 4).
    let g = CsrGraph::<f64>::from_edges(
        6,
        &[
            (0, 1),
            (1, 2),
            (0, 2),
            (3, 4),
            (4, 5),
            (3, 5),
            (0, 3),
            (2, 4),
        ],
        None,
        false,
        false,
    )
    .unwrap();

    // Non-contiguous large labels verify canonical compaction without HashMap.
    let labels = [100, 100, 100, u32::MAX - 5, u32::MAX - 5, u32::MAX - 5];
    let coarse = coarsen_graph(&g, &labels).unwrap();
    assert_eq!(coarse.num_clusters, 2);
    assert_eq!(coarse.node_to_coarse, vec![0, 0, 0, 1, 1, 1]);
    assert_eq!(coarse.coarse_sizes, vec![3, 3]);
    assert_eq!(coarse.graph.num_nodes(), 2);
    assert_eq!(coarse.graph.num_edges(), 1);
    assert_eq!(coarse.graph.neighbors(0), &[1]);
    assert_eq!(coarse.graph.neighbor_weights(0), Some(&[2.0][..]));

    let coarse_layout =
        Layout2d::from_xy(vec![-5.0_f64, 5.0_f64], vec![1.0_f64, -1.0_f64]).unwrap();
    let fine1 = project_clustered_layout(&coarse_layout, &coarse.node_to_coarse, 0.5, 42).unwrap();
    let fine2 = project_clustered_layout(&coarse_layout, &coarse.node_to_coarse, 0.5, 42).unwrap();
    assert_eq!(fine1, fine2);
    assert_eq!(fine1.len(), 6);

    for (i, &cid_u32) in coarse.node_to_coarse.iter().enumerate() {
        let cid = cid_u32 as usize;
        let dx = fine1.x()[i] - coarse_layout.x()[cid];
        let dy = fine1.y()[i] - coarse_layout.y()[cid];
        assert!(dx.hypot(dy) <= 0.5 + 1e-12);
    }

    // Zero spread radius recovers exact cluster centers.
    let exact = project_clustered_layout(&coarse_layout, &coarse.node_to_coarse, 0.0, 42).unwrap();
    assert_eq!(exact.x(), &[-5.0, -5.0, -5.0, 5.0, 5.0, 5.0]);
    assert_eq!(exact.y(), &[1.0, 1.0, 1.0, -1.0, -1.0, -1.0]);

    // Weighted f32 coarsening + singleton cluster preservation in f32 projection.
    let wg = CsrGraph::from_edges(
        4,
        &[(0, 1), (0, 2), (1, 2), (2, 3)],
        Some(&[2.0_f32, 3.0, 4.0, 1.5]),
        false,
        false,
    )
    .unwrap();
    let wc = coarsen_graph(&wg, &[7, 7, 9, 7]).unwrap();
    assert_eq!(wc.num_clusters, 2);
    assert_eq!(wc.node_to_coarse, vec![0, 0, 1, 0]);
    assert_eq!(wc.coarse_sizes, vec![3, 1]);
    assert_eq!(wc.graph.neighbor_weights(0), Some(&[8.5_f64][..]));

    let cl_f32 = Layout2d::from_xy(vec![1.0_f32, 4.0_f32], vec![2.0_f32, -3.0_f32]).unwrap();
    let fine_f32 = project_clustered_layout(&cl_f32, &wc.node_to_coarse, 0.25_f32, 99).unwrap();
    assert!(fine_f32.is_all_finite());
    // Singleton cluster (node 2 -> coarse 1) sits directly at coarse center.
    assert_eq!(fine_f32.x()[2], 4.0_f32);
    assert_eq!(fine_f32.y()[2], -3.0_f32);
}

#[test]
fn test_coarsen_and_project_errors_and_empty() {
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let ce = coarsen_graph(&empty, &[]).unwrap();
    assert_eq!(ce.num_clusters, 0);
    assert_eq!(ce.graph.num_nodes(), 0);
    let pe = project_clustered_layout(&Layout2d::<f32>::zeros(0), &[], 0.1_f32, 1).unwrap();
    assert!(pe.is_empty());

    let directed = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, true, false).unwrap();
    assert!(matches!(
        coarsen_graph(&directed, &[0, 1]),
        Err(CoreError::InvalidInput(_))
    ));

    let g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    assert!(matches!(
        coarsen_graph(&g, &[0]),
        Err(CoreError::DimensionMismatch { .. })
    ));

    let cl = Layout2d::from_xy(vec![0.0_f64, 1.0], vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        project_clustered_layout(&cl, &[0, 2], 0.5, 42),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        project_clustered_layout(&cl, &[0, 1], -0.1, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        project_clustered_layout(&cl, &[0, 1], f64::NAN, 42),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let bad_cl = Layout2d::from_xy(vec![f64::NAN, 1.0], vec![0.0, 1.0]).unwrap();
    assert!(matches!(
        project_clustered_layout(&bad_cl, &[0, 1], 0.5, 42),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_louvain_coarsen_hierarchy_and_disconnected_floor() {
    // Connected barbell of 4 cliques of size 5 (20 nodes) contracted to <= 6 nodes.
    let mut edges = Vec::new();
    for c in 0..4 {
        let off = c * 5;
        for i in 0..5 {
            for j in (i + 1)..5 {
                edges.push(((off + i) as u32, (off + j) as u32));
            }
        }
    }
    edges.extend([(4, 5), (9, 10), (14, 15)]);
    let g = CsrGraph::<f64>::from_edges(20, &edges, None, false, false).unwrap();

    let cfg = CoarseLouvainConfig::new()
        .with_max_coarse_nodes(3)
        .with_resolution(1.0)
        .with_resolution_decay(0.5)
        .with_max_passes(50)
        .with_min_modularity_gain(1e-7)
        .with_max_outer_rounds(16)
        .with_seed(42);
    let h1 = louvain_coarsen(&g, &cfg).unwrap();
    let h2 = louvain_coarsen(&g, &cfg).unwrap();
    assert_eq!(h1, h2);
    assert!(h1.coarsest.num_clusters <= 3);
    assert_eq!(h1.coarsest.node_to_coarse.len(), 20);
    let sum_sizes: u32 = h1.coarsest.coarse_sizes.iter().sum();
    assert_eq!(sum_sizes, 20);

    // Disconnected graph with 4 isolated triangles (> max_coarse_nodes = 2):
    // must stop cleanly at the connected-component floor (4 clusters) without looping.
    let mut disc_edges = Vec::new();
    for c in 0..4 {
        let off = (c * 3) as u32;
        disc_edges.extend([(off, off + 1), (off + 1, off + 2), (off, off + 2)]);
    }
    let disc = CsrGraph::<f64>::from_edges(12, &disc_edges, None, false, false).unwrap();
    let disc_cfg = CoarseLouvainConfig {
        max_coarse_nodes: 2,
        ..CoarseLouvainConfig::default()
    };
    let hd = louvain_coarsen(&disc, &disc_cfg).unwrap();
    assert_eq!(hd.coarsest.num_clusters, 4);
    assert_eq!(hd.coarsest.graph.num_edges(), 0);

    // Empty and edgeless edge cases.
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    assert_eq!(
        louvain_coarsen(&empty, &cfg).unwrap().coarsest.num_clusters,
        0
    );
    let edgeless = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
    assert_eq!(
        louvain_coarsen(&edgeless, &cfg)
            .unwrap()
            .coarsest
            .num_clusters,
        5
    );
}

#[test]
fn test_project_with_radii_and_centroid_alignment_preserves_filament() {
    // Pitfall #7 oracle: cluster 0 is an elongated horizontal filament (aspect ratio 10:1),
    // cluster 1 is a singleton. Rigid translation must shift cluster 0 centroid to (5.0, -3.0)
    // while preserving its > 2.0 aspect ratio identically.
    let coarse = Layout2d::from_xy(vec![5.0_f64, -5.0], vec![-3.0, 2.0]).unwrap();
    let n2c = [0u32, 0, 0, 0, 0, 1];

    let mut fine = Layout2d::from_xy(
        vec![0.0, 2.5, 5.0, 7.5, 10.0, 0.0],
        vec![0.0, 0.25, 0.0, -0.25, 0.0, 0.0],
    )
    .unwrap();
    align_clustered_centroids(&mut fine, &coarse, &n2c, 1.0).unwrap();

    let mean_x: f64 = fine.x()[..5].iter().sum::<f64>() / 5.0;
    let mean_y: f64 = fine.y()[..5].iter().sum::<f64>() / 5.0;
    assert!((mean_x - 5.0).abs() < 1e-12);
    assert!((mean_y - (-3.0)).abs() < 1e-12);
    assert!((fine.x()[5] - (-5.0)).abs() < 1e-12);
    assert!((fine.y()[5] - 2.0).abs() < 1e-12);

    let span_x = fine.x()[..5]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        - fine.x()[..5].iter().copied().fold(f64::INFINITY, f64::min);
    let span_y = fine.y()[..5]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        - fine.y()[..5].iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        span_x / span_y > 10.0,
        "filament distorted: span_x={span_x}, span_y={span_y}"
    );

    // Radius-scaled projection respects per-cluster radius bounds.
    let proj = project_clustered_layout_with_radii(&coarse, &n2c, &[2.0, 4.0], 0.5, 7).unwrap();
    for i in 0..5 {
        let dist = (proj.x()[i] - 5.0).hypot(proj.y()[i] - (-3.0));
        assert!(dist <= 1.0 + 1e-12);
    }
    assert_eq!(proj.x()[5], -5.0);
    assert_eq!(proj.y()[5], 2.0);
}
