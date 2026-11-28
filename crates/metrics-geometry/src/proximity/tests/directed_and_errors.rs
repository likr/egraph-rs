use egraph_core::{CoreError, CsrGraph, DeterministicRng, Layout2d, Scalar};
use egraph_spatial_trees::StaticKdTree;

use super::super::exact_predicates::euclidean_diameter_2d;
use super::super::*;
use super::{
    assert_proximity_determinism_and_tree_parity, brute_gg_edges, brute_node_resolution,
    brute_rng_edges,
};

#[test]
fn test_signed_zero_diameter_colocated_pass2b_and_gabriel_property_colocation() {
    // 1. Regression for signed-zero (-0.0 / +0.0) in euclidean_diameter_2d (n > 64 branch).
    let mut xs_y = vec![-0.0_f64, -0.0_f64];
    let mut ys_y = vec![0.0_f64, 1.0_f64];
    xs_y.extend(std::iter::repeat_n(0.0_f64, 63));
    ys_y.extend(std::iter::repeat_n(0.0_f64, 63));
    let layout_sz_y = Layout2d::from_xy(xs_y, ys_y).unwrap();
    assert_eq!(layout_sz_y.len(), 65);
    assert!((euclidean_diameter_2d(&layout_sz_y) - 1.0).abs() < 1e-12);
    let nr_sz_y = node_resolution(&layout_sz_y).unwrap();
    assert!((nr_sz_y - brute_node_resolution(&layout_sz_y)).abs() < 1e-10);
    // Must NOT trigger collapsed-layout branch (65 * 64 / 2 = 2080.0).
    assert!(nr_sz_y < 2080.0);

    let mut xs_x = vec![0.0_f64, 1.0_f64];
    let mut ys_x = vec![-0.0_f64, -0.0_f64];
    xs_x.extend(std::iter::repeat_n(0.0_f64, 63));
    ys_x.extend(std::iter::repeat_n(0.0_f64, 63));
    let layout_sz_x = Layout2d::from_xy(xs_x, ys_x).unwrap();
    assert!((euclidean_diameter_2d(&layout_sz_x) - 1.0).abs() < 1e-12);
    assert!(
        (node_resolution(&layout_sz_x).unwrap() - brute_node_resolution(&layout_sz_x)).abs()
            < 1e-10
    );

    // 2. > 128 co-located points (n = 160) exercising Pass 2b `uncovered` fallback.
    let mut rng = DeterministicRng::from_seed_u64(2026);
    let mut xs_160 = vec![0.0_f64; 135];
    let mut ys_160 = vec![0.0_f64; 135];
    for _ in 0..25 {
        xs_160.push((rng.next_f64() - 0.5) * 20.0 + 5.0);
        ys_160.push((rng.next_f64() - 0.5) * 20.0 + 5.0);
    }
    let layout_160 = Layout2d::from_xy(xs_160, ys_160).unwrap();
    assert_eq!(
        gabriel_graph_edges(&layout_160).unwrap(),
        brute_gg_edges(&layout_160)
    );
    assert_eq!(
        relative_neighborhood_edges(&layout_160).unwrap(),
        brute_rng_edges(&layout_160)
    );

    // 3. gabriel_graph_property with zero-length edge and co-located 3rd vertex.
    let g_zero_edge = CsrGraph::<f64>::from_edges(3, &[(0, 1)], None, false, false).unwrap();
    let l_zero_edge = Layout2d::from_xy(vec![1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0]).unwrap();
    assert_eq!(
        gabriel_graph_property(&g_zero_edge, &l_zero_edge).unwrap(),
        0.0
    );

    // Non-zero edge (0, 1) from (0,0) to (4,0): radius r = 2.0, center (2,0).
    let l_endpoint_coloc = Layout2d::from_xy(vec![0.0, 4.0, 0.0], vec![0.0, 0.0, 0.0]).unwrap();
    assert_eq!(
        gabriel_graph_property(&g_zero_edge, &l_endpoint_coloc).unwrap(),
        0.0
    );

    // Vertex 2 at midpoint (2,0) -> dist == 0.0 < r -> violation = r^2 = 4.0.
    let l_midpoint = Layout2d::from_xy(vec![0.0, 4.0, 2.0], vec![0.0, 0.0, 0.0]).unwrap();
    assert!((gabriel_graph_property(&g_zero_edge, &l_midpoint).unwrap() - 4.0).abs() < 1e-12);
}

#[test]
fn test_directed_reciprocal_and_sanitized_self_loop_parallel_edges() {
    let base_edges = [(0_u32, 1_u32), (1, 2), (2, 3), (3, 0), (0, 2)];
    let g_undir = CsrGraph::<f64>::from_edges(4, &base_edges, None, false, false).unwrap();

    let mut recip_edges = Vec::new();
    for &(u, v) in &base_edges {
        recip_edges.push((u, v));
        recip_edges.push((v, u));
    }
    let g_dir_recip = CsrGraph::<f64>::from_edges(4, &recip_edges, None, true, false).unwrap();
    let g_dir_oneway = CsrGraph::<f64>::from_edges(4, &base_edges, None, true, false).unwrap();

    let layout = Layout2d::from_xy(vec![0.0, 4.0, 2.0, 1.9], vec![0.0, 0.0, 1.0, 0.2]).unwrap();

    // S_RNG and S_GG deduplicate directed edges via undirected symmetrization.
    assert_eq!(
        relative_neighborhood_preservation(&g_undir, &layout)
            .unwrap()
            .to_bits(),
        relative_neighborhood_preservation(&g_dir_recip, &layout)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        relative_neighborhood_preservation(&g_undir, &layout)
            .unwrap()
            .to_bits(),
        relative_neighborhood_preservation(&g_dir_oneway, &layout)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        gabriel_graph_preservation(&g_undir, &layout)
            .unwrap()
            .to_bits(),
        gabriel_graph_preservation(&g_dir_recip, &layout)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        gabriel_graph_preservation(&g_undir, &layout)
            .unwrap()
            .to_bits(),
        gabriel_graph_preservation(&g_dir_oneway, &layout)
            .unwrap()
            .to_bits()
    );

    // gabriel_graph_property evaluates each directed edge yielded by extract_unique_edges.
    let prop_undir = gabriel_graph_property(&g_undir, &layout).unwrap();
    let prop_dir_recip = gabriel_graph_property(&g_dir_recip, &layout).unwrap();
    let prop_dir_oneway = gabriel_graph_property(&g_dir_oneway, &layout).unwrap();
    assert!(prop_undir > 0.0);
    assert!((prop_dir_recip - 2.0 * prop_undir).abs() < 1e-12);
    assert!((prop_dir_oneway - prop_undir).abs() < 1e-12);

    // Sanitized construction with self-loops and parallel/reverse duplicates.
    let dirty_edges = [
        (0_u32, 0_u32),
        (1, 1),
        (0, 1),
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 2),
        (2, 0),
    ];
    let g_sanitized_undir =
        CsrGraph::<f64>::from_edges(4, &dirty_edges, None, false, true).unwrap();
    let g_sanitized_dir = CsrGraph::<f64>::from_edges(4, &dirty_edges, None, true, true).unwrap();

    assert_eq!(
        relative_neighborhood_preservation(&g_sanitized_undir, &layout)
            .unwrap()
            .to_bits(),
        relative_neighborhood_preservation(&g_undir, &layout)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        gabriel_graph_preservation(&g_sanitized_undir, &layout)
            .unwrap()
            .to_bits(),
        gabriel_graph_preservation(&g_undir, &layout)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        gabriel_graph_property(&g_sanitized_undir, &layout)
            .unwrap()
            .to_bits(),
        prop_undir.to_bits()
    );
    assert_eq!(
        relative_neighborhood_preservation(&g_sanitized_dir, &layout)
            .unwrap()
            .to_bits(),
        relative_neighborhood_preservation(&g_undir, &layout)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        gabriel_graph_preservation(&g_sanitized_dir, &layout)
            .unwrap()
            .to_bits(),
        gabriel_graph_preservation(&g_undir, &layout)
            .unwrap()
            .to_bits()
    );
    assert_proximity_determinism_and_tree_parity(&g_dir_recip, &layout);
}

fn assert_all_12_reject_non_finite<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    bad_layout: &Layout2d<S>,
    valid_tree_same_len: &StaticKdTree<2, f64>,
) {
    assert!(matches!(
        gabriel_graph_property(graph, bad_layout),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        gabriel_graph_property_with_index(graph, bad_layout, valid_tree_same_len),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        node_resolution(bad_layout),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        node_resolution_with_index(bad_layout, valid_tree_same_len),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        relative_neighborhood_edges(bad_layout),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        relative_neighborhood_edges_with_index(bad_layout, valid_tree_same_len),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        gabriel_graph_edges(bad_layout),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        gabriel_graph_edges_with_index(bad_layout, valid_tree_same_len),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        relative_neighborhood_preservation(graph, bad_layout),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        relative_neighborhood_preservation_with_index(graph, bad_layout, valid_tree_same_len),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        gabriel_graph_preservation(graph, bad_layout),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        gabriel_graph_preservation_with_index(graph, bad_layout, valid_tree_same_len),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_non_finite_extreme_overflow_and_dimension_mismatch_all_12() {
    // 1. Non-finite (NaN, +Inf, -Inf) at N = 1 and N = 2 for f64 and f32 across all 12 functions.
    let g1_64 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let t1 = StaticKdTree::<2, f64>::from_layout(&Layout2d::<f64>::zeros(1)).unwrap();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let lx = Layout2d::from_xy(vec![bad], vec![0.0]).unwrap();
        let ly = Layout2d::from_xy(vec![0.0], vec![bad]).unwrap();
        assert_all_12_reject_non_finite(&g1_64, &lx, &t1);
        assert_all_12_reject_non_finite(&g1_64, &ly, &t1);
    }

    let g2_64 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let t2 = StaticKdTree::<2, f64>::from_layout(&Layout2d::<f64>::zeros(2)).unwrap();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let l = Layout2d::from_xy(vec![0.0, bad], vec![0.0, 1.0]).unwrap();
        assert_all_12_reject_non_finite(&g2_64, &l, &t2);
    }

    let g1_32 = CsrGraph::<f32>::from_edges(1, &[], None, false, false).unwrap();
    let g2_32 = CsrGraph::<f32>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let l1 = Layout2d::from_xy(vec![bad], vec![0.0_f32]).unwrap();
        let l2 = Layout2d::from_xy(vec![0.0_f32, 1.0], vec![0.0_f32, bad]).unwrap();
        assert_all_12_reject_non_finite(&g1_32, &l1, &t1);
        assert_all_12_reject_non_finite(&g2_32, &l2, &t2);
    }

    // 2. Extreme finite coordinates (1e200): must never panic or hang across all 12 functions.
    let g3_64 =
        CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2), (2, 0)], None, false, false).unwrap();
    let l_huge = Layout2d::from_xy(vec![0.0, 1e200, -1e200], vec![0.0, 1e200, 0.0]).unwrap();
    let t_huge = StaticKdTree::<2, f64>::from_layout(&l_huge).unwrap();
    for res in [
        gabriel_graph_property(&g3_64, &l_huge),
        gabriel_graph_property_with_index(&g3_64, &l_huge, &t_huge),
        node_resolution(&l_huge),
        node_resolution_with_index(&l_huge, &t_huge),
        relative_neighborhood_preservation(&g3_64, &l_huge),
        relative_neighborhood_preservation_with_index(&g3_64, &l_huge, &t_huge),
        gabriel_graph_preservation(&g3_64, &l_huge),
        gabriel_graph_preservation_with_index(&g3_64, &l_huge, &t_huge),
    ] {
        assert!(matches!(res, Ok(_) | Err(CoreError::NonFiniteValue { .. })));
    }
    assert!(matches!(
        relative_neighborhood_edges(&l_huge),
        Ok(_) | Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        relative_neighborhood_edges_with_index(&l_huge, &t_huge),
        Ok(_) | Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        gabriel_graph_edges(&l_huge),
        Ok(_) | Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        gabriel_graph_edges_with_index(&l_huge, &t_huge),
        Ok(_) | Err(CoreError::NonFiniteValue { .. })
    ));

    // 3. DimensionMismatch: graph/layout mismatch and tree/layout mismatch across all applicable functions.
    let l4 = Layout2d::from_xy(vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    let t4 = StaticKdTree::<2, f64>::from_layout(&l4).unwrap();
    let t3 = StaticKdTree::<2, f64>::from_layout(&Layout2d::<f64>::zeros(3)).unwrap();

    // Graph (3) vs layout (4).
    assert!(matches!(
        gabriel_graph_property(&g3_64, &l4),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        gabriel_graph_property_with_index(&g3_64, &l4, &t4),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        relative_neighborhood_preservation(&g3_64, &l4),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        relative_neighborhood_preservation_with_index(&g3_64, &l4, &t4),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        gabriel_graph_preservation(&g3_64, &l4),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        gabriel_graph_preservation_with_index(&g3_64, &l4, &t4),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Graph (4) & layout (4) vs tree (3).
    let g4_64 = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2)], None, false, false).unwrap();
    assert!(matches!(
        gabriel_graph_property_with_index(&g4_64, &l4, &t3),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        node_resolution_with_index(&l4, &t3),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        relative_neighborhood_edges_with_index(&l4, &t3),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        gabriel_graph_edges_with_index(&l4, &t3),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        relative_neighborhood_preservation_with_index(&g4_64, &l4, &t3),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        gabriel_graph_preservation_with_index(&g4_64, &l4, &t3),
        Err(CoreError::DimensionMismatch { .. })
    ));
}
