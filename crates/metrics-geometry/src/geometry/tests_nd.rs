use egraph_core::{CoreError, CsrGraph, Layout2d, LayoutNd};

use crate::nd_index::build_nd_spatial_index;

use super::*;

#[test]
fn test_geometry_nd_2d_bit_exact_parity_and_f32() {
    let g_unw =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], None, false, false)
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
    let lnd = LayoutNd::from_flat(4, 2, flat.clone()).unwrap();
    let idx = build_nd_spatial_index(&lnd).unwrap();

    assert_eq!(
        ideal_edge_lengths(&g_unw, &l2).unwrap().to_bits(),
        ideal_edge_lengths_nd(&g_unw, &lnd).unwrap().to_bits()
    );
    assert_eq!(
        ideal_edge_lengths(&g_w, &l2).unwrap().to_bits(),
        ideal_edge_lengths_nd(&g_w, &lnd).unwrap().to_bits()
    );
    assert_eq!(
        ideal_edge_lengths_with_unit(&g_unw, &l2, 2.5)
            .unwrap()
            .to_bits(),
        ideal_edge_lengths_with_unit_nd(&g_unw, &lnd, 2.5)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        ideal_edge_lengths_scaled(&g_w, &l2, 1.5).unwrap().to_bits(),
        ideal_edge_lengths_scaled_nd(&g_w, &lnd, 1.5)
            .unwrap()
            .to_bits()
    );

    let dists = vec![
        0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0,
    ];
    assert_eq!(
        ideal_edge_lengths_from_distances(&g_unw, &l2, &dists)
            .unwrap()
            .to_bits(),
        ideal_edge_lengths_from_distances_nd(&g_unw, &lnd, &dists)
            .unwrap()
            .to_bits()
    );

    let deg_2 = layout_degeneracy(&g_unw, &l2).unwrap();
    let deg_nd = layout_degeneracy_nd(&g_unw, &lnd).unwrap();
    let deg_idx = layout_degeneracy_nd_with_index(&g_unw, &lnd, &idx).unwrap();
    assert_eq!(deg_2, deg_nd);
    assert_eq!(deg_nd, deg_idx);

    let g32 = CsrGraph::<f32>::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3), (3, 0)],
        Some(&[1.0_f32, 2.0, 1.5, 2.5]),
        false,
        false,
    )
    .unwrap();
    let lnd32 = LayoutNd::from_flat(4, 2, flat.iter().map(|&v| v as f32).collect()).unwrap();
    assert!(
        (ideal_edge_lengths_nd(&g32, &lnd32).unwrap() - ideal_edge_lengths_nd(&g_w, &lnd).unwrap())
            .abs()
            < 1e-12
    );
}

#[test]
fn test_geometry_nd_multi_dims_degeneracy_and_ideal_lengths() {
    // 1D layout: bbox_height == 0.0, diag == span_0, and is_bbox_degenerate == false if span_0 > 1e-12!
    let g3 = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    let l1d = LayoutNd::from_flat(3, 1, vec![0.0, 2.0, 5.0]).unwrap();
    let deg1 = layout_degeneracy_nd(&g3, &l1d).unwrap();
    assert_eq!(deg1.bbox_width, 5.0);
    assert_eq!(deg1.bbox_height, 0.0);
    assert!(!deg1.is_bbox_degenerate);
    assert_eq!(deg1.near_collision_ratio, 0.0);

    // 3D layout where x, y vary widely (10.0), z is flat (0.0) -> is_bbox_degenerate == true!
    let l3d_flat_z = LayoutNd::from_flat(
        4,
        3,
        vec![
            0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0,
        ],
    )
    .unwrap();
    let g4 = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
    let deg_flat_z = layout_degeneracy_nd(&g4, &l3d_flat_z).unwrap();
    assert!(deg_flat_z.is_bbox_degenerate);
    assert_eq!(deg_flat_z.bbox_width, 10.0);
    assert_eq!(deg_flat_z.bbox_height, 10.0);

    // 10D layout with f32-only duplicate, exact f64 duplicate, near-collision, and zero-length edge.
    let dims = 10usize;
    let mut coords = vec![0.0_f64; 5 * dims];
    // Node 0: (1.0, 0, ...)
    coords[0] = 1.0;
    // Node 1: differs at 1e-12 from 1.0 (rounds to 1.0 in f32, distinct in f64, near collision!)
    coords[dims] = 1.0 + 1e-12;
    // Node 2: exact duplicate of Node 0 (-0.0 canonicalized)
    coords[2 * dims] = 1.0;
    coords[2 * dims + 1] = -0.0;
    // Node 3: far away (2.0 in all dims)
    for d in 0..dims {
        coords[3 * dims + d] = 2.0;
    }
    // Node 4: distinct (3.0 in dim 9)
    coords[4 * dims + 9] = 3.0;

    let g5 = CsrGraph::<f64>::from_edges(5, &[(0, 2), (0, 3), (3, 4)], None, false, false).unwrap();
    let l10 = LayoutNd::from_flat(5, dims, coords).unwrap();
    let deg10 = layout_degeneracy_nd(&g5, &l10).unwrap();
    assert_eq!(deg10.duplicate_nodes_f64, 2); // nodes 0 and 2
    assert_eq!(deg10.duplicate_nodes_f32, 3); // nodes 0, 1, 2
    assert!((deg10.excess_duplicate_ratio - 1.0 / 5.0).abs() < 1e-12);
    assert!((deg10.near_collision_ratio - 3.0 / 5.0).abs() < 1e-12);
    assert_eq!(deg10.zero_length_edges, 1); // edge (0, 2)
    assert!(!deg10.is_bbox_degenerate);

    // Unit-edge ideal lengths on 3D orthogonal unit axes.
    let l3_ortho = LayoutNd::from_flat(
        4,
        3,
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let g_star =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (0, 2), (0, 3)], None, false, false).unwrap();
    assert!(ideal_edge_lengths_nd(&g_star, &l3_ortho).unwrap() < 1e-12);
    assert!(
        (ideal_edge_lengths_with_unit_nd(&g_star, &l3_ortho, 2.0).unwrap() - 0.75).abs() < 1e-12
    );
}

#[test]
fn test_geometry_nd_boundaries_and_errors() {
    for n in [0usize, 1] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let l = LayoutNd::<f64>::zeros(n, 3).unwrap();
        assert_eq!(ideal_edge_lengths_nd(&g, &l).unwrap(), 0.0);
        assert_eq!(ideal_edge_lengths_with_unit_nd(&g, &l, 1.0).unwrap(), 0.0);
        assert_eq!(ideal_edge_lengths_scaled_nd(&g, &l, 1.0).unwrap(), 0.0);
        let d = vec![1.0_f64; n * n];
        assert_eq!(
            ideal_edge_lengths_from_distances_nd(&g, &l, &d).unwrap(),
            0.0
        );
        let deg = layout_degeneracy_nd(&g, &l).unwrap();
        assert_eq!(deg.num_nodes, n);
        assert_eq!(deg.near_collision_ratio, 0.0);
    }

    // Collapsed N = 3 layout.
    let g3 = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    let l_col = LayoutNd::<f64>::zeros(3, 4).unwrap();
    let deg_col = layout_degeneracy_nd(&g3, &l_col).unwrap();
    assert_eq!(deg_col.near_collision_ratio, 1.0);
    assert_eq!(deg_col.zero_length_edge_ratio, 1.0);
    assert!(deg_col.is_bbox_degenerate);

    // Non-finite, non-positive unit length, and dimension mismatches.
    let l_nan = LayoutNd::from_flat(3, 2, vec![0.0, 0.0, f64::NAN, 1.0, 0.0, 0.0]).unwrap();
    assert!(matches!(
        ideal_edge_lengths_nd(&g3, &l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        layout_degeneracy_nd(&g3, &l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let l3 = LayoutNd::<f64>::zeros(3, 3).unwrap();
    assert!(matches!(
        ideal_edge_lengths_with_unit_nd(&g3, &l3, 0.0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        ideal_edge_lengths_scaled_nd(&g3, &l3, -1.0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        ideal_edge_lengths_from_distances_nd(&g3, &l3, &[0.0; 4]),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        layout_degeneracy_nd(&g3, &LayoutNd::<f64>::zeros(2, 3).unwrap()),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
