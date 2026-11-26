use super::*;

#[test]
fn test_bfs_and_dijkstra_workspaces_reuse_and_disconnected() {
    // Component A: 0 - 1 - 2 (weights 1.0, 5.0), shortcut 0 - 2 (weight 10.0)
    // Component B: isolated node 3
    let g = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 2), (0, 2)],
        Some(&[1.0, 5.0, 10.0]),
        false,
        false,
    )
    .unwrap();

    let mut bfs = BfsWorkspace::new(4);
    let mut dist = vec![0.0; 4];
    bfs.bfs_from(&g, 0, &mut dist).unwrap();
    assert_eq!(dist, vec![0.0, 1.0, 1.0, f64::INFINITY]);

    // Reuse workspace on smaller graph and verify unreachable nodes stay Inf
    let path = CsrGraph::<f64>::from_edges(3, &[(0, 1)], None, false, false).unwrap();
    let mut dist_small = vec![123.0; 3];
    bfs.bfs_from(&path, 0, &mut dist_small).unwrap();
    assert_eq!(dist_small, vec![0.0, 1.0, f64::INFINITY]);

    let mut dijk = DijkstraWorkspace::new(4);
    dijk.dijkstra_from(&g, 0, &mut dist).unwrap();
    assert_eq!(dist, vec![0.0, 1.0, 6.0, f64::INFINITY]);

    dijk.dijkstra_from(&g, 2, &mut dist).unwrap();
    assert_eq!(dist, vec![6.0, 5.0, 0.0, f64::INFINITY]);

    // Reject negative edge weights even on disconnected edges
    let neg_g = CsrGraph::<f64>::from_edges(3, &[(1, 2)], Some(&[-0.5]), false, false).unwrap();
    assert!(matches!(
        dijk.dijkstra_from(&neg_g, 0, &mut dist_small),
        Err(CoreError::InvalidInput(_))
    ));

    // Reject dimension mismatch and out-of-bounds source
    assert!(matches!(
        bfs.bfs_from(&g, 0, &mut dist_small),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        bfs.bfs_from(&g, 4, &mut dist),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn test_maxmin_sp_pivots_core_first_and_fallback() {
    // Core path 0-1-2-3-4 with leaves attached at 0 and 4, plus isolated node 9
    let mut edges = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
    edges.push((0, 5));
    edges.push((0, 6));
    edges.push((4, 7));
    edges.push((4, 8));
    let g = CsrGraph::<f64>::from_edges(10, &edges, None, false, false).unwrap();

    let p1 = select_maxmin_sp_pivots(&g, 5, 42, 2).unwrap();
    let p2 = select_maxmin_sp_pivots(&g, 5, 42, 2).unwrap();
    assert_eq!(p1, p2);
    assert_eq!(p1.len(), 5);
    // All 5 core nodes (0..=4) have degree >= 2, so leaves (5..=8) and isolated (9) are deferred
    assert!(p1.iter().all(|&u| u <= 4));

    // Requesting 9 pivots exhausts core nodes, picks non-isolated leaves before isolated node 9
    let p9 = select_maxmin_sp_pivots(&g, 9, 42, 2).unwrap();
    assert_eq!(p9.len(), 9);
    assert!(!p9.contains(&9), "isolated node should be picked last");

    let p_all = select_maxmin_sp_pivots(&g, 12, 42, 2).unwrap();
    assert_eq!(p_all.len(), 10);

    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    assert!(
        select_maxmin_sp_pivots(&empty, 5, 42, 2)
            .unwrap()
            .is_empty()
    );

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    assert_eq!(select_maxmin_sp_pivots(&single, 5, 42, 2).unwrap(), vec![0]);
}

#[test]
fn test_dijkstra_single_source_and_all_pairs_dijkstra() {
    // Weighted graph: 0-1 (1.5), 1-2 (2.5), 0-2 (10.0), isolated node 3.
    let g = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 2), (0, 2)],
        Some(&[1.5, 2.5, 10.0]),
        false,
        false,
    )
    .unwrap();
    let d0 = dijkstra_single_source(&g, 0).unwrap();
    assert_eq!(d0, vec![0.0, 1.5, 4.0, f64::INFINITY]);

    let apsp = all_pairs_dijkstra(&g).unwrap();
    assert_eq!(
        apsp,
        vec![
            0.0,
            1.5,
            4.0,
            f64::INFINITY,
            1.5,
            0.0,
            2.5,
            f64::INFINITY,
            4.0,
            2.5,
            0.0,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            0.0,
        ]
    );

    // Directed graph asymmetry + unweighted default weight 1.0.
    let dg = CsrGraph::<f32>::from_edges(3, &[(0, 1), (1, 2)], None, true, false).unwrap();
    let apsp_dir = all_pairs_dijkstra(&dg).unwrap();
    assert_eq!(
        apsp_dir,
        vec![
            0.0,
            1.0,
            2.0,
            f64::INFINITY,
            0.0,
            1.0,
            f64::INFINITY,
            f64::INFINITY,
            0.0,
        ]
    );

    // Empty and singleton graphs.
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    assert!(all_pairs_dijkstra(&empty).unwrap().is_empty());
    assert!(matches!(
        dijkstra_single_source(&empty, 0),
        Err(CoreError::IndexOutOfBounds { .. })
    ));

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    assert_eq!(dijkstra_single_source(&single, 0).unwrap(), vec![0.0]);
    assert_eq!(all_pairs_dijkstra(&single).unwrap(), vec![0.0]);

    // Reject negative and out-of-bounds source.
    let neg = CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[-1.0]), false, false).unwrap();
    assert!(matches!(
        dijkstra_single_source(&neg, 0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        all_pairs_dijkstra(&neg),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        dijkstra_single_source(&g, 4),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
}
