//! Opaque-box E2E tests for `CsrGraph::permute_nodes` and `Layout{2d,3d,Nd}::{permute, unpermute}`.

use egraph_core::{CoreError, CsrGraph, Layout2d, Layout3d, LayoutNd, MetricSpace};

#[test]
fn csr_permute_nodes_undirected_weighted_and_unweighted_preserves_sorted_rows() {
    let edges = [(0, 1), (1, 2), (2, 3), (0, 3), (1, 3)];
    let weights = [1.5_f64, 2.5, 3.5, 4.5, 5.5];
    let gw = CsrGraph::<f64>::from_edges(4, &edges, Some(&weights), false, false).unwrap();
    let gu = CsrGraph::<()>::from_unweighted_edges(4, &edges, false, false).unwrap();

    let perm = [2_u32, 0, 3, 1];
    let gw_p = gw.permute_nodes(&perm).unwrap();
    let gu_p = gu.permute_nodes(&perm).unwrap();

    assert_eq!(gw_p.num_nodes(), 4);
    assert_eq!(gw_p.num_edges(), gw.num_edges());
    assert_eq!(gw_p.nnz(), gw.nnz());
    assert!(!gw_p.is_directed());
    assert!(gw_p.is_weighted());
    assert!(!gu_p.is_weighted());

    for u in 0..4 {
        let pu = perm[u] as usize;
        let nbrs = gw_p.neighbors(pu);
        assert!(nbrs.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(nbrs, gu_p.neighbors(pu));
        for v in 0..4 {
            let pv = perm[v] as usize;
            assert_eq!(gw_p.has_edge(pu, pv), gw.has_edge(u, v));
            assert_eq!(gw_p.edge_weight(pu, pv), gw.edge_weight(u, v));
            assert_eq!(gu_p.has_edge(pu, pv), gu.has_edge(u, v));
        }
    }
}

#[test]
fn csr_permute_nodes_directed_preserves_asymmetric_edges_and_weights() {
    let edges = [(0, 1), (2, 0), (2, 3)];
    let weights = [1.5_f32, 3.5, 7.25];
    let dg = CsrGraph::<f32>::from_edges(4, &edges, Some(&weights), true, false).unwrap();
    let perm = [3_u32, 2, 0, 1];
    let dg_p = dg.permute_nodes(&perm).unwrap();

    assert!(dg_p.is_directed());
    assert_eq!(dg_p.num_edges(), 3);
    assert_eq!(dg_p.edge_weight(3, 2), Some(1.5));
    assert_eq!(dg_p.edge_weight(2, 3), None);
    assert_eq!(dg_p.edge_weight(0, 3), Some(3.5));
    assert_eq!(dg_p.edge_weight(3, 0), None);
    assert_eq!(dg_p.edge_weight(0, 1), Some(7.25));
    assert_eq!(dg_p.edge_weight(1, 0), None);
}

#[test]
fn layout_2d_3d_nd_permute_unpermute_roundtrip_bit_exact() {
    let perm = [3_u32, 0, 4, 1, 2];

    let l2_64 = Layout2d::<f64>::init(5, 11, 5.0).unwrap();
    let l2_p = l2_64.permute(&perm).unwrap();
    for (old_i, &new_i) in perm.iter().enumerate() {
        assert_eq!(l2_p.get(new_i as usize).unwrap(), l2_64.get(old_i).unwrap());
    }
    assert_eq!(l2_p.unpermute(&perm).unwrap(), l2_64);
    assert_eq!(
        l2_64.unpermute(&perm).unwrap().permute(&perm).unwrap(),
        l2_64
    );

    let l2_32 = Layout2d::<f32>::init(5, 12, 5.0).unwrap();
    assert_eq!(
        l2_32.permute(&perm).unwrap().unpermute(&perm).unwrap(),
        l2_32
    );

    let l3_64 = Layout3d::<f64>::init(5, 13, 5.0).unwrap();
    let l3_p = l3_64.permute(&perm).unwrap();
    for (old_i, &new_i) in perm.iter().enumerate() {
        assert_eq!(l3_p.get(new_i as usize).unwrap(), l3_64.get(old_i).unwrap());
    }
    assert_eq!(l3_p.unpermute(&perm).unwrap(), l3_64);

    let l3_32 = Layout3d::<f32>::init(5, 14, 5.0).unwrap();
    assert_eq!(
        l3_32.permute(&perm).unwrap().unpermute(&perm).unwrap(),
        l3_32
    );

    let lnd_64 = LayoutNd::<f64>::init(5, 4, 15, 5.0).unwrap();
    let lnd_p = lnd_64.permute(&perm).unwrap();
    assert_eq!(lnd_p.dims(), 4);
    for (old_i, &new_i) in perm.iter().enumerate() {
        assert_eq!(
            lnd_p.node(new_i as usize).unwrap(),
            lnd_64.node(old_i).unwrap()
        );
    }
    assert_eq!(lnd_p.unpermute(&perm).unwrap(), lnd_64);

    let lnd_32 = LayoutNd::<f32>::init(5, 4, 16, 5.0).unwrap();
    assert_eq!(
        lnd_32.permute(&perm).unwrap().unpermute(&perm).unwrap(),
        lnd_32
    );
}

#[test]
fn permutation_preserves_pairwise_distances_and_weighted_adjacency() {
    let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)];
    let weights = [1.0, 2.0, 1.5, 2.5, 3.0, 4.0];
    let g = CsrGraph::<f64>::from_edges(5, &edges, Some(&weights), false, false).unwrap();
    let l2 = Layout2d::<f64>::init(5, 29, 10.0).unwrap();
    let perm = [4_u32, 2, 0, 3, 1];

    let gp = g.permute_nodes(&perm).unwrap();
    let lp = l2.permute(&perm).unwrap();
    for i in 0..5 {
        for j in (i + 1)..5 {
            let pi = perm[i] as usize;
            let pj = perm[j] as usize;
            assert_eq!(l2.distance(i, j).to_bits(), lp.distance(pi, pj).to_bits());
            assert_eq!(g.edge_weight(i, j), gp.edge_weight(pi, pj));
        }
    }
}

#[test]
fn identity_and_cycle_permutations_restore_original_state() {
    let g =
        CsrGraph::<()>::from_unweighted_edges(4, &[(0, 1), (1, 2), (2, 3)], false, false).unwrap();
    let id = [0_u32, 1, 2, 3];
    assert_eq!(g.permute_nodes(&id).unwrap(), g);

    let l2 = Layout2d::<f64>::init(4, 3, 2.0).unwrap();
    assert_eq!(l2.permute(&id).unwrap(), l2);
    assert_eq!(l2.unpermute(&id).unwrap(), l2);

    let cycle = [1_u32, 2, 3, 0];
    let mut g_cur = g.clone();
    let mut l_cur = l2.clone();
    for _ in 0..4 {
        g_cur = g_cur.permute_nodes(&cycle).unwrap();
        l_cur = l_cur.permute(&cycle).unwrap();
    }
    assert_eq!(g_cur, g);
    assert_eq!(l_cur, l2);
}

#[test]
fn permute_empty_and_singleton_graphs_and_layouts() {
    let g_empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    assert_eq!(g_empty.permute_nodes(&[]).unwrap(), g_empty);

    let l2 = Layout2d::<f64>::zeros(0);
    assert_eq!(l2.permute(&[]).unwrap(), l2);
    assert_eq!(l2.unpermute(&[]).unwrap(), l2);

    let l3 = Layout3d::<f64>::zeros(0);
    assert_eq!(l3.permute(&[]).unwrap(), l3);
    assert_eq!(l3.unpermute(&[]).unwrap(), l3);

    let lnd = LayoutNd::<f64>::zeros(0, 3).unwrap();
    assert_eq!(lnd.permute(&[]).unwrap(), lnd);
    assert_eq!(lnd.unpermute(&[]).unwrap(), lnd);

    // n = 1 singleton graph and layouts
    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    assert_eq!(g1.permute_nodes(&[0]).unwrap(), g1);
    assert!(matches!(
        g1.permute_nodes(&[1]),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        g1.permute_nodes(&[]),
        Err(CoreError::DimensionMismatch {
            expected: 1,
            actual: 0,
            ..
        })
    ));

    let l2_1 = Layout2d::<f64>::from_xy(vec![3.25], vec![-1.5]).unwrap();
    assert_eq!(l2_1.permute(&[0]).unwrap(), l2_1);
    assert_eq!(l2_1.unpermute(&[0]).unwrap(), l2_1);
    assert!(matches!(
        l2_1.permute(&[1]),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        l2_1.unpermute(&[]),
        Err(CoreError::DimensionMismatch {
            expected: 1,
            actual: 0,
            ..
        })
    ));

    let l3_1 = Layout3d::<f64>::from_xyz(vec![1.0], vec![2.0], vec![3.0]).unwrap();
    assert_eq!(l3_1.permute(&[0]).unwrap(), l3_1);
    assert!(matches!(
        l3_1.permute(&[1]),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        l3_1.permute(&[]),
        Err(CoreError::DimensionMismatch {
            expected: 1,
            actual: 0,
            ..
        })
    ));

    let lnd_1 = LayoutNd::<f64>::from_flat(1, 3, &[4.0, 5.0, 6.0]).unwrap();
    assert_eq!(lnd_1.permute(&[0]).unwrap(), lnd_1);
    assert!(matches!(
        lnd_1.unpermute(&[1]),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        lnd_1.unpermute(&[]),
        Err(CoreError::DimensionMismatch {
            expected: 1,
            actual: 0,
            ..
        })
    ));
}

#[test]
fn permute_length_mismatch_returns_dimension_mismatch() {
    let g = CsrGraph::<f64>::from_edges(3, &[(0, 1)], None, false, false).unwrap();
    assert!(matches!(
        g.permute_nodes(&[0, 1]),
        Err(CoreError::DimensionMismatch {
            expected: 3,
            actual: 2,
            ..
        })
    ));
    assert!(matches!(
        g.permute_nodes(&[0, 1, 2, 3]),
        Err(CoreError::DimensionMismatch {
            expected: 3,
            actual: 4,
            ..
        })
    ));

    let l2 = Layout2d::<f64>::zeros(3);
    let l3 = Layout3d::<f64>::zeros(3);
    let lnd = LayoutNd::<f64>::zeros(3, 2).unwrap();
    for res in [
        l2.permute(&[0, 1]).map(|_| ()),
        l2.unpermute(&[0, 1, 2, 3]).map(|_| ()),
        l3.permute(&[0, 1]).map(|_| ()),
        l3.unpermute(&[0, 1, 2, 3]).map(|_| ()),
        lnd.permute(&[0, 1]).map(|_| ()),
        lnd.unpermute(&[0, 1, 2, 3]).map(|_| ()),
    ] {
        assert!(matches!(res, Err(CoreError::DimensionMismatch { .. })));
    }
}

#[test]
fn permute_duplicate_index_returns_invalid_input() {
    let dup = [1_u32, 1, 2];
    let g = CsrGraph::<f64>::from_edges(3, &[(0, 1)], None, false, false).unwrap();
    assert!(matches!(
        g.permute_nodes(&dup),
        Err(CoreError::InvalidInput(_))
    ));

    let l2 = Layout2d::<f64>::zeros(3);
    let l3 = Layout3d::<f64>::zeros(3);
    let lnd = LayoutNd::<f64>::zeros(3, 4).unwrap();
    assert!(matches!(l2.permute(&dup), Err(CoreError::InvalidInput(_))));
    assert!(matches!(
        l2.unpermute(&dup),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(l3.permute(&dup), Err(CoreError::InvalidInput(_))));
    assert!(matches!(
        l3.unpermute(&dup),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(lnd.permute(&dup), Err(CoreError::InvalidInput(_))));
    assert!(matches!(
        lnd.unpermute(&dup),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn permute_out_of_bounds_index_returns_invalid_input() {
    let g = CsrGraph::<f64>::from_edges(3, &[(0, 1)], None, false, false).unwrap();
    let l2 = Layout2d::<f64>::zeros(3);
    let l3 = Layout3d::<f64>::zeros(3);
    let lnd = LayoutNd::<f64>::zeros(3, 2).unwrap();

    for bad in [[0_u32, 1, 3], [0_u32, 1, u32::MAX]] {
        assert!(matches!(
            g.permute_nodes(&bad),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(l2.permute(&bad), Err(CoreError::InvalidInput(_))));
        assert!(matches!(
            l2.unpermute(&bad),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(l3.permute(&bad), Err(CoreError::InvalidInput(_))));
        assert!(matches!(
            l3.unpermute(&bad),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(lnd.permute(&bad), Err(CoreError::InvalidInput(_))));
        assert!(matches!(
            lnd.unpermute(&bad),
            Err(CoreError::InvalidInput(_))
        ));
    }
}

#[test]
fn permute_preserves_non_finite_coordinates_bit_exact() {
    let perm = [2_u32, 0, 1];
    let l2 = Layout2d::<f64>::from_xy(
        vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
        vec![1.0, 2.0, 3.0],
    )
    .unwrap();
    let l2_p = l2.permute(&perm).unwrap();
    assert_eq!(l2_p.x()[2].to_bits(), l2.x()[0].to_bits());
    assert_eq!(l2_p.x()[0].to_bits(), f64::INFINITY.to_bits());
    assert_eq!(l2_p.x()[1].to_bits(), f64::NEG_INFINITY.to_bits());
    let l2_rt = l2_p.unpermute(&perm).unwrap();
    for idx in 0..3 {
        assert_eq!(l2_rt.x()[idx].to_bits(), l2.x()[idx].to_bits());
        assert_eq!(l2_rt.y()[idx].to_bits(), l2.y()[idx].to_bits());
    }

    let l3 = Layout3d::<f64>::from_xyz(
        vec![1.0, f64::NAN, 3.0],
        vec![f64::NEG_INFINITY, 2.0, f64::INFINITY],
        vec![0.0, 1.0, 2.0],
    )
    .unwrap();
    let l3_p = l3.permute(&perm).unwrap();
    assert_eq!(l3_p.x()[0].to_bits(), l3.x()[1].to_bits());
    assert_eq!(l3_p.y()[2].to_bits(), f64::NEG_INFINITY.to_bits());
    let l3_rt = l3_p.unpermute(&perm).unwrap();
    for idx in 0..3 {
        assert_eq!(l3_rt.x()[idx].to_bits(), l3.x()[idx].to_bits());
        assert_eq!(l3_rt.y()[idx].to_bits(), l3.y()[idx].to_bits());
        assert_eq!(l3_rt.z()[idx].to_bits(), l3.z()[idx].to_bits());
    }

    let lnd = LayoutNd::<f64>::from_flat(
        3,
        2,
        &[f64::NAN, 1.0, f64::INFINITY, 2.0, f64::NEG_INFINITY, 3.0],
    )
    .unwrap();
    let lnd_p = lnd.permute(&perm).unwrap();
    assert_eq!(lnd_p.coords()[4].to_bits(), lnd.coords()[0].to_bits());
    let lnd_rt = lnd_p.unpermute(&perm).unwrap();
    for (a, b) in lnd_rt.coords().iter().zip(lnd.coords().iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn permute_isolated_vertices_and_arbitrary_nd_dims() {
    // Nodes 1 and 3 are isolated in a 5-node graph.
    let g =
        CsrGraph::<f64>::from_edges(5, &[(0, 2), (2, 4)], Some(&[2.0, 3.0]), false, false).unwrap();
    let perm = [4_u32, 0, 3, 2, 1];
    let gp = g.permute_nodes(&perm).unwrap();
    assert_eq!(gp.neighbors(0), &[]);
    assert_eq!(gp.neighbors(2), &[]);
    assert_eq!(gp.degree(perm[1] as usize), 0);
    assert_eq!(gp.degree(perm[3] as usize), 0);
    assert_eq!(gp.edge_weight(4, 3), Some(2.0));
    assert_eq!(gp.edge_weight(3, 1), Some(3.0));

    let lnd1 = LayoutNd::<f64>::init(5, 1, 99, 2.0).unwrap();
    let lnd7 = LayoutNd::<f64>::init(5, 7, 100, 2.0).unwrap();
    assert_eq!(lnd1.permute(&perm).unwrap().dims(), 1);
    assert_eq!(lnd7.permute(&perm).unwrap().dims(), 7);
    assert_eq!(lnd1.permute(&perm).unwrap().unpermute(&perm).unwrap(), lnd1);
    assert_eq!(lnd7.permute(&perm).unwrap().unpermute(&perm).unwrap(), lnd7);
}
