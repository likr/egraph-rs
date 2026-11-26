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
    // Unsorted symmetric + self-loop + duplicate + missing reverse edge:
    // 0 -> [0, 2, 1, 2], 1 -> [0], 2 -> []
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
    // Max weight kept on duplicate (0, 2); symmetric (0, 1) does NOT double weight.
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
    // Asymmetric undirected CSR rejected when sanitize=false.
    assert!(matches!(
        CsrGraph::<f64>::from_csr(&[0, 1, 2, 3], &[1, 2, 0], None, false, false),
        Err(CoreError::InvalidCsr(_))
    ));
    // Duplicate undirected edge via (0,1) and (1,0) rejected when sanitize=false.
    assert!(matches!(
        CsrGraph::<f64>::from_edges(2, &[(0, 1), (1, 0)], None, false, false),
        Err(CoreError::InvalidCsr(_))
    ));
    // Odd flat edge length rejected.
    assert!(matches!(
        CsrGraph::<f64>::from_flat_edges(3, &[0, 1, 2], None, false, true),
        Err(CoreError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_csr_largest_connected_component_and_subgraph() {
    // Component A: {0, 2} (1 edge); Component B: {1, 3, 4, 5} (3 edges, size 4).
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
    assert_eq!(lcc.neighbors(0), &[1]); // old 1 -> old 3 (new 1)
    assert_eq!(lcc.neighbor_weights(0), Some(&[2.0][..]));
    assert_eq!(lcc.neighbors(1), &[0, 2]); // old 3 -> old 1, 4 (new 0, 2)
    assert_eq!(lcc.neighbor_weights(1), Some(&[2.0, 3.0][..]));

    // Directed weak component test: 0 <- 1 -> 2, isolated 3.
    let dg = CsrGraph::<f64>::from_edges(4, &[(1, 0), (1, 2)], None, true, false).unwrap();
    let (dlcc, d_orig) = dg.largest_connected_component();
    assert_eq!(d_orig, vec![0, 1, 2]);
    assert_eq!(dlcc.num_nodes(), 3);
    assert_eq!(dlcc.neighbors(1), &[0, 2]);

    // Empty graph LCC.
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let (elcc, e_orig) = empty.largest_connected_component();
    assert_eq!(elcc.num_nodes(), 0);
    assert!(e_orig.is_empty());
}

#[test]
fn test_csr_edges_has_edge_weight_and_unsorted_rows() {
    // Unsorted duplicate-free rows permitted by `sanitize = false`.
    // 0 -> [2, 1], 1 -> [2, 0], 2 -> [1, 0]
    let indptr = [0, 2, 4, 6];
    let indices = [2, 1, 2, 0, 1, 0];
    let weights = [10.0_f64, 5.0, 20.0, 5.0, 20.0, 10.0];
    let g = CsrGraph::from_csr(&indptr, &indices, Some(&weights), false, false).unwrap();

    assert_eq!(g.edges(), vec![(0, 2), (0, 1), (1, 2)]);
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
        Err(CoreError::IndexOutOfBounds {
            index: 3,
            len: 3,
            ..
        })
    ));
    assert!(matches!(
        g.try_has_edge(0, 3),
        Err(CoreError::IndexOutOfBounds {
            index: 3,
            len: 3,
            ..
        })
    ));
    assert!(matches!(
        g.try_edge_weight(3, 0),
        Err(CoreError::IndexOutOfBounds {
            index: 3,
            len: 3,
            ..
        })
    ));
    assert!(matches!(
        g.try_edge_weight(0, 3),
        Err(CoreError::IndexOutOfBounds {
            index: 3,
            len: 3,
            ..
        })
    ));

    // Unweighted directed graph with reciprocal edges (0, 1) and (1, 0).
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

    // Directed weak connectivity: 1 -> 0, 1 -> 2 is weakly connected.
    let weak = CsrGraph::<f64>::from_edges(3, &[(1, 0), (1, 2)], None, true, false).unwrap();
    assert!(weak.is_connected());

    // Directed with isolated vertex 3 is disconnected.
    let weak_disc = CsrGraph::<f64>::from_edges(4, &[(1, 0), (1, 2)], None, true, false).unwrap();
    assert!(!weak_disc.is_connected());
}
