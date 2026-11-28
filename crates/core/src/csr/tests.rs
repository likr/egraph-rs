use super::*;
use crate::error::CoreError;

#[test]
fn test_csr_from_edges_undirected_and_directed() {
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
    assert_eq!(g.num_nodes(), 4);
    assert_eq!(g.num_edges(), 3);
    assert_eq!(g.nnz(), 6);
    assert!(!g.is_directed());
    assert!(!g.is_weighted());
    assert_eq!(g.indptr(), &[0, 1, 3, 5, 6]);
    assert_eq!(g.indices(), &[1, 0, 2, 1, 3, 2]);
    assert_eq!(g.neighbors(1), &[0, 2]);
    assert_eq!(g.degree(1), 2);
    assert!(g.weights().is_none());
    assert!(g.neighbor_weights(1).is_none());

    let dg =
        CsrGraph::<f32>::from_edges(3, &[(0, 1), (0, 2)], Some(&[1.5_f32, 2.5_f32]), true, false)
            .unwrap();
    assert!(dg.is_directed());
    assert!(dg.is_weighted());
    assert_eq!(dg.num_edges(), 2);
    assert_eq!(dg.neighbors(0), &[1, 2]);
    assert_eq!(dg.neighbor_weights(0), Some(&[1.5_f32, 2.5_f32][..]));
    assert_eq!(dg.degree(1), 0);
}

#[test]
fn test_csr_sanitize_vs_strict_validation() {
    let indptr = [0, 4, 5, 5];
    let indices = [0, 2, 1, 2, 0];
    let weights = [9.0, 2.0, 1.5, 4.0, 1.5];

    let err = CsrGraph::from_csr(&indptr, &indices, Some(&weights), false, false).unwrap_err();
    assert!(matches!(err, CoreError::InvalidCsr(_)));

    let g = CsrGraph::from_csr(&indptr, &indices, Some(&weights), false, true).unwrap();
    assert_eq!(g.num_nodes(), 3);
    assert_eq!(g.num_edges(), 2);
    assert_eq!(g.neighbors(0), &[1, 2]);
    assert_eq!(g.neighbors(1), &[0]);
    assert_eq!(g.neighbors(2), &[0]);
    assert_eq!(g.neighbor_weights(0), Some(&[1.5, 4.0][..]));
    assert_eq!(g.neighbor_weights(1), Some(&[1.5][..]));
    assert_eq!(g.neighbor_weights(2), Some(&[4.0][..]));
}

#[test]
fn test_csr_strict_validation_errors() {
    assert!(matches!(
        CsrGraph::<f64>::from_csr(&[], &[], None, false, false),
        Err(CoreError::InvalidCsr(_))
    ));
    assert!(matches!(
        CsrGraph::<f64>::from_csr(&[1, 1], &[], None, false, false),
        Err(CoreError::InvalidCsr(_))
    ));
    assert!(matches!(
        CsrGraph::<f64>::from_csr(&[0, 2, 1], &[0], None, false, false),
        Err(CoreError::InvalidCsr(_))
    ));
    assert!(matches!(
        CsrGraph::<f64>::from_csr(&[0, 1, 1], &[5], None, false, false),
        Err(CoreError::IndexOutOfBounds {
            index: 5,
            len: 2,
            ..
        })
    ));
    assert!(matches!(
        CsrGraph::<f64>::from_csr(&[0, 1, 2], &[1, 0], Some(&[f64::NAN, 1.0]), false, true),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        CsrGraph::<f64>::from_csr(&[0, 1, 2, 3], &[1, 2, 0], None, false, false),
        Err(CoreError::InvalidCsr(_))
    ));
    assert!(matches!(
        CsrGraph::<f64>::from_edges(2, &[(0, 1), (1, 0)], None, false, false),
        Err(CoreError::InvalidCsr(_))
    ));
    assert!(matches!(
        CsrGraph::<f64>::from_flat_edges(3, &[0, 1, 2], None, false, true),
        Err(CoreError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_csr_largest_connected_component_and_subgraph() {
    let g = CsrGraph::from_edges(
        6,
        &[(0, 2), (1, 3), (3, 4), (4, 5)],
        Some(&[1.0, 2.0, 3.0, 4.0]),
        false,
        true,
    )
    .unwrap();
    let (lcc, orig_ids) = g.largest_connected_component();
    assert_eq!(orig_ids, vec![1, 3, 4, 5]);
    assert_eq!(lcc.num_nodes(), 4);
    assert_eq!(lcc.num_edges(), 3);
    assert_eq!(lcc.neighbors(0), &[1]);
    assert_eq!(lcc.neighbor_weights(0), Some(&[2.0][..]));
    assert_eq!(lcc.neighbors(1), &[0, 2]);
    assert_eq!(lcc.neighbor_weights(1), Some(&[2.0, 3.0][..]));

    let dg = CsrGraph::<f64>::from_edges(4, &[(1, 0), (1, 2)], None, true, false).unwrap();
    let (dlcc, d_orig) = dg.largest_connected_component();
    assert_eq!(d_orig, vec![0, 1, 2]);
    assert_eq!(dlcc.num_nodes(), 3);
    assert_eq!(dlcc.neighbors(1), &[0, 2]);

    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let (elcc, e_orig) = empty.largest_connected_component();
    assert_eq!(elcc.num_nodes(), 0);
    assert!(e_orig.is_empty());
}

#[test]
fn test_csr_edges_has_edge_weight_and_unsorted_rows() {
    let indptr = [0, 2, 4, 6];
    let indices = [2, 1, 2, 0, 1, 0];
    let weights = [10.0_f64, 5.0, 20.0, 5.0, 20.0, 10.0];
    let g = CsrGraph::from_csr(&indptr, &indices, Some(&weights), false, false).unwrap();

    assert_eq!(g.edges(), vec![(0, 2), (0, 1), (1, 2)]);
    assert_eq!(g.canonical_undirected_edges(), vec![(0, 1), (0, 2), (1, 2)]);
    assert_eq!(g.edges().len(), g.num_edges());
    assert!(g.has_edge(0, 1));
    assert!(g.has_edge(0, 2));
    assert!(g.has_edge(1, 0));
    assert!(!g.has_edge(0, 0));
    assert!(!g.has_edge(9, 0));
    assert!(!g.has_edge(0, 9));
    assert_eq!(g.edge_weight(0, 2), Some(10.0));
    assert_eq!(g.edge_weight(0, 1), Some(5.0));
    assert_eq!(g.edge_weight(2, 1), Some(20.0));
    assert_eq!(g.edge_weight(0, 0), None);
    assert_eq!(g.edge_weight(9, 0), None);

    assert!(g.try_has_edge(1, 2).unwrap());
    assert_eq!(g.try_edge_weight(1, 2).unwrap(), Some(20.0));
    assert!(matches!(
        g.try_has_edge(3, 0),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        g.try_has_edge(0, 3),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        g.try_edge_weight(3, 0),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        g.try_edge_weight(0, 3),
        Err(CoreError::IndexOutOfBounds { .. })
    ));

    let dg = CsrGraph::<f32>::from_edges(3, &[(0, 1), (1, 0), (1, 2)], None, true, false).unwrap();
    assert_eq!(dg.edges(), vec![(0, 1), (1, 0), (1, 2)]);
    assert_eq!(dg.edges().len(), dg.num_edges());
    assert!(dg.has_edge(0, 1));
    assert!(dg.has_edge(1, 0));
    assert!(!dg.has_edge(2, 1));
    assert_eq!(dg.edge_weight(0, 1), None);
    assert_eq!(dg.try_edge_weight(0, 1).unwrap(), None);
}

#[test]
fn test_csr_is_connected_edge_cases() {
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    assert!(empty.is_connected());

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    assert!(single.is_connected());

    let edgeless_2 = CsrGraph::<f64>::from_edges(2, &[], None, false, false).unwrap();
    assert!(!edgeless_2.is_connected());

    let connected =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
    assert!(connected.is_connected());

    let disconnected =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    assert!(!disconnected.is_connected());

    let weak = CsrGraph::<f64>::from_edges(3, &[(1, 0), (1, 2)], None, true, false).unwrap();
    assert!(weak.is_connected());

    let weak_disc = CsrGraph::<f64>::from_edges(4, &[(1, 0), (1, 2)], None, true, false).unwrap();
    assert!(!weak_disc.is_connected());
}

#[test]
fn test_csr_unweighted_unit_type_builder_and_undirected_simple() {
    let g_edges =
        CsrGraph::<()>::from_unweighted_edges(3, &[(0, 1), (1, 2)], false, false).unwrap();
    let g_flat =
        CsrGraph::<()>::from_unweighted_flat_edges(3, &[0, 1, 1, 2], false, false).unwrap();
    let g_csr =
        CsrGraph::<()>::from_unweighted_csr(&[0, 1, 3, 4], &[1, 0, 2, 1], false, false).unwrap();
    assert_eq!(g_edges, g_flat);
    assert_eq!(g_edges, g_csr);
    assert!(!g_edges.is_weighted());

    // Directed graph with reciprocal edges, duplicates, self-loops -> canonical simple undirected.
    let dir = CsrGraph::<f64>::from_csr(
        &[0, 3, 5, 5],
        &[0, 1, 2, 0, 2],
        Some(&[1.0, 2.0, 3.0, 4.0, 5.0]),
        true,
        true,
    )
    .unwrap();
    assert_eq!(
        dir.canonical_undirected_edges(),
        vec![(0, 1), (0, 2), (1, 2)]
    );
    let simple = dir.to_undirected_simple().unwrap();
    assert!(!simple.is_directed());
    assert!(!simple.is_weighted());
    assert_eq!(simple.num_nodes(), 3);
    assert_eq!(simple.num_edges(), 3);
    assert_eq!(simple.neighbors(0), &[1, 2]);
    assert_eq!(simple.neighbors(1), &[0, 2]);
    assert_eq!(simple.neighbors(2), &[0, 1]);

    // CsrGraphBuilder unweighted & weighted workflows.
    let mut ub = CsrGraphBuilder::<()>::new(false);
    let n0 = ub.add_node().unwrap();
    let n1 = ub.add_node().unwrap();
    ub.add_nodes(1).unwrap();
    assert_eq!(ub.num_nodes(), 3);
    ub.add_edge(n0, n1).unwrap();
    ub.add_edge(n1, 2).unwrap();
    assert_eq!(ub.num_edges(), 2);
    assert!(!ub.is_directed());
    assert_eq!(ub.build_unweighted(false).unwrap(), g_edges);

    let mut wb = CsrGraphBuilder::<f64>::with_num_nodes(3, true).unwrap();
    wb.add_weighted_edge(0, 1, 2.5).unwrap();
    wb.add_weighted_edge(0, 1, 4.0).unwrap();
    assert!(matches!(wb.add_edge(1, 2), Err(CoreError::InvalidInput(_))));
    assert!(matches!(
        wb.build_unweighted(false),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        wb.add_weighted_edge(3, 0, 1.0),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        wb.add_weighted_edge(0, 1, f64::NAN),
        Err(CoreError::NonFiniteValue { .. })
    ));
    let wg = wb.build(true).unwrap();
    assert_eq!(wg.edge_weight(0, 1), Some(4.0));
}

#[test]
fn test_csr_permute_nodes_exhaustive_and_errors() {
    let assert_edge_oracle = |orig: &CsrGraph<f64>, permuted: &CsrGraph<f64>, perm: &[u32]| {
        let n = orig.num_nodes();
        assert_eq!(permuted.num_nodes(), n);
        assert_eq!(permuted.num_edges(), orig.num_edges());
        assert_eq!(permuted.is_directed(), orig.is_directed());
        assert_eq!(permuted.is_weighted(), orig.is_weighted());
        for new_u in 0..n {
            assert!(permuted.neighbors(new_u).windows(2).all(|w| w[0] < w[1]));
        }
        for u in 0..n {
            for v in 0..n {
                assert_eq!(
                    orig.has_edge(u, v),
                    permuted.has_edge(perm[u] as usize, perm[v] as usize)
                );
                assert_eq!(
                    orig.edge_weight(u, v),
                    permuted.edge_weight(perm[u] as usize, perm[v] as usize)
                );
            }
        }
    };

    // Undirected unweighted CsrGraph<()> with isolated vertex (node 4).
    let uu =
        CsrGraph::<()>::from_unweighted_edges(5, &[(0, 2), (2, 3), (1, 3)], false, true).unwrap();
    let perm5 = [3, 0, 4, 1, 2];
    let uu_p = uu.permute_nodes(&perm5).unwrap();
    assert!(!uu_p.is_weighted());
    assert!(!uu_p.is_directed());
    assert_eq!(uu_p.num_edges(), uu.num_edges());
    for new_u in 0..5 {
        assert!(uu_p.neighbors(new_u).windows(2).all(|w| w[0] < w[1]));
    }
    for u in 0..5 {
        for v in 0..5 {
            assert_eq!(
                uu.has_edge(u, v),
                uu_p.has_edge(perm5[u] as usize, perm5[v] as usize)
            );
        }
    }
    let mut inv5 = vec![0u32; 5];
    for (old_i, &new_i) in perm5.iter().enumerate() {
        inv5[new_i as usize] = old_i as u32;
    }
    assert_eq!(uu_p.permute_nodes(&inv5).unwrap(), uu);

    // Undirected weighted f64 with unsorted rows (sanitize = false) -> output rows sorted!
    let unsorted = CsrGraph::<f64>::from_csr(
        &[0, 2, 4, 6],
        &[2, 1, 2, 0, 1, 0],
        Some(&[10.0, 5.0, 20.0, 5.0, 20.0, 10.0]),
        false,
        false,
    )
    .unwrap();
    let perm3 = [2, 0, 1];
    let unsorted_p = unsorted.permute_nodes(&perm3).unwrap();
    assert_edge_oracle(&unsorted, &unsorted_p, &perm3);

    // Directed weighted f64 graph + identity + reversal permutations.
    let dir = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 3), (0, 1), (2, 0), (3, 1)],
        Some(&[1.25, 2.5, 3.75, 4.5]),
        true,
        true,
    )
    .unwrap();
    assert_eq!(dir.permute_nodes(&[0, 1, 2, 3]).unwrap(), dir);
    let rev4 = [3, 2, 1, 0];
    let dir_rev = dir.permute_nodes(&rev4).unwrap();
    assert_edge_oracle(&dir, &dir_rev, &rev4);
    assert_eq!(dir_rev.permute_nodes(&rev4).unwrap(), dir);

    // Weighted f32.
    let wf32 = CsrGraph::<f32>::from_edges(3, &[(0, 2)], Some(&[7.5_f32]), false, true).unwrap();
    let wf32_p = wf32.permute_nodes(&[1, 2, 0]).unwrap();
    assert_eq!(wf32_p.edge_weight(1, 0), Some(7.5_f32));

    // Empty (n == 0) weighted & unweighted, and singleton (n == 1).
    let empty_w = CsrGraph::<f64>::from_edges(0, &[], Some(&[]), false, false).unwrap();
    let empty_wp = empty_w.permute_nodes(&[]).unwrap();
    assert_eq!(empty_wp, empty_w);
    assert!(empty_wp.is_weighted());

    let empty_u = CsrGraph::<()>::from_unweighted_edges(0, &[], false, false).unwrap();
    let empty_up = empty_u.permute_nodes(&[]).unwrap();
    assert_eq!(empty_up, empty_u);
    assert!(!empty_up.is_weighted());

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    assert_eq!(single.permute_nodes(&[0]).unwrap(), single);

    // Error contracts: DimensionMismatch on length mismatch, InvalidInput on duplicate or >= n.
    assert!(matches!(
        dir.permute_nodes(&[]),
        Err(CoreError::DimensionMismatch {
            expected: 4,
            actual: 0,
            ..
        })
    ));
    assert!(matches!(
        empty_u.permute_nodes(&[0]),
        Err(CoreError::DimensionMismatch {
            expected: 0,
            actual: 1,
            ..
        })
    ));
    assert!(matches!(
        dir.permute_nodes(&[0, 0, 2, 3]),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        dir.permute_nodes(&[0, 1, 2, 4]),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        dir.permute_nodes(&[0, 1, 2, 99]),
        Err(CoreError::InvalidInput(_))
    ));
}
