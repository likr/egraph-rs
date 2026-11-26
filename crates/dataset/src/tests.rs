use super::*;

#[test]
fn test_parse_csr_csv_valid_and_errors() {
    let csv = "4,3\n0,1\r\n1,2\n2,3\n";
    let g = parse_csr_csv::<f64>(csv, false, false).unwrap();
    assert_eq!(g.num_nodes(), 4);
    assert_eq!(g.num_edges(), 3);
    assert!(!g.is_directed());
    assert_eq!(g.neighbors(1), &[0, 2]);

    let dg = parse_csr_csv::<f64>("3,2\n0,1\n1,0\n", true, false).unwrap();
    assert!(dg.is_directed());
    assert_eq!(dg.num_edges(), 2);

    assert!(matches!(
        parse_csr_csv::<f64>("", false, false),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        parse_csr_csv::<f64>("3,2\n0,1\n", false, false),
        Err(CoreError::DimensionMismatch { .. })
    ));
    let wg = parse_csr_csv::<f64>("3,2\n0,1,2.5\n1, 2, 1.25\n", false, false).unwrap();
    assert!(wg.is_weighted());
    assert_eq!(wg.num_edges(), 2);
    assert_eq!(wg.weights().unwrap(), &[2.5, 2.5, 1.25, 1.25]);

    assert!(matches!(
        parse_csr_csv::<f64>("3,1\n0,1,2,3\n", false, false),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        parse_csr_csv::<f64>("3,2\n0,1,2.0\n1,2\n", false, false),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        parse_csr_csv::<f64>("3,1\n0,1,nan\n", false, false),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        parse_csr_csv::<f64>("3,1\n0,1,inf\n", false, false),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        parse_csr_csv::<f64>("3,1\n0,1,abc\n", false, false),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        parse_csr_csv::<f64>("3,1\n0,5\n", false, false),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        parse_csr_csv::<f64>("3,1\n0,0\n", false, false),
        Err(CoreError::InvalidCsr(_))
    ));
}

#[test]
fn test_parse_csr_csv_f32() {
    let csv = "4,3\n0,1\n1,2\n2,3\n";
    let g: CsrGraph<f32> = parse_csr_csv::<f32>(csv, false, false).unwrap();
    assert_eq!(g.num_nodes(), 4);
    assert_eq!(g.num_edges(), 3);
    assert!(!g.is_directed());
    assert!(g.weights().is_none());
    assert_eq!(g.neighbors(1), &[0, 2]);

    let wg = parse_csr_csv::<f32>("3,2\n0,1,1.5\n1,2,3.25\n", false, false).unwrap();
    assert!(wg.is_weighted());
    assert_eq!(wg.weights().unwrap(), &[1.5_f32, 1.5, 3.25, 3.25]);
    assert!(matches!(
        parse_csr_csv::<f32>("3,1\n0,1,1e300\n", false, false),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Sanitize deduplicates undirected edges and removes self-loops.
    let dirty = "3,3\n0,1\n1,0\n0,0\n";
    let clean = parse_csr_csv::<f32>(dirty, false, true).unwrap();
    assert_eq!(clean.num_nodes(), 3);
    assert_eq!(clean.num_edges(), 1);
    assert_eq!(clean.neighbors(0), &[1]);

    let dg = parse_csr_csv::<f32>("3,2\n0,1\n1,0\n", true, false).unwrap();
    assert!(dg.is_directed());
    assert_eq!(dg.num_edges(), 2);

    assert!(matches!(
        parse_csr_csv::<f32>("3,2\n0,1\n", false, false),
        Err(CoreError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_catalog_list_datasets_and_metadata() {
    let names = list_datasets();
    assert_eq!(names.len(), 9);
    assert!(names.contains(&"karate"));
    assert!(names.contains(&"3elt"));
    assert_eq!(dataset_catalog().len(), names.len());
    let karate = dataset_info("karate").unwrap();
    assert_eq!(karate.num_nodes, 34);
    assert_eq!(karate.num_edges, 78);
    assert!(dataset_info("nonexistent").is_none());
}

#[test]
fn test_synthetic_generators_deterministic_and_valid() {
    // Weighted graph
    let w1 = generate_weighted_graph(20, 4, 0.5, 3.0, 42).unwrap();
    let w2 = generate_weighted_graph(20, 4, 0.5, 3.0, 42).unwrap();
    assert_eq!(w1, w2);
    assert!(w1.is_weighted());
    assert!(w1.is_connected());
    assert!(
        w1.weights()
            .unwrap()
            .iter()
            .all(|&w| (0.5..=3.0).contains(&w))
    );
    assert!(generate_weighted_graph(5, 4, -1.0, 2.0, 42).is_err());

    // Layered DAG
    let dag = generate_layered_dag(4, 5, 0.5, 42).unwrap();
    assert!(dag.is_directed());
    assert_eq!(dag.num_nodes(), 20);
    for u in 0..dag.num_nodes() {
        for &v in dag.neighbors(u) {
            assert!(
                u < v as usize,
                "DAG edge {u} -> {v} violates topological order"
            );
        }
    }
    assert!(generate_layered_dag(3, 3, 1.5, 42).is_err());

    // SBM with ground-truth labels
    let (sbm1, labels1) = generate_sbm_graph(&[10, 12, 8], 0.8, 0.05, 7).unwrap();
    let (sbm2, labels2) = generate_sbm_graph(&[10, 12, 8], 0.8, 0.05, 7).unwrap();
    assert_eq!(sbm1, sbm2);
    assert_eq!(labels1, labels2);
    assert_eq!(sbm1.num_nodes(), 30);
    assert_eq!(labels1.len(), 30);
    assert_eq!(labels1[0], 0);
    assert_eq!(labels1[10], 1);
    assert_eq!(labels1[22], 2);
    assert!(generate_sbm_graph(&[5], -0.1, 0.1, 1).is_err());

    // Manifold meshes: torus grid (4-regular and 6-regular), icosphere, UV sphere, hyperbolic tree
    let tor4 = generate_torus_grid(6, 8, false).unwrap();
    assert_eq!(tor4.num_nodes(), 48);
    assert_eq!(tor4.num_edges(), 96);
    assert!((0..48).all(|u| tor4.degree(u) == 4));

    let tor6 = generate_torus_grid(6, 8, true).unwrap();
    assert_eq!(tor6.num_nodes(), 48);
    assert_eq!(tor6.num_edges(), 144);
    assert!((0..48).all(|u| tor6.degree(u) == 6));
    assert!(generate_torus_grid(2, 5, false).is_err());

    let ico0 = generate_spherical_icosphere(0).unwrap();
    assert_eq!(ico0.num_nodes(), 12);
    assert_eq!(ico0.num_edges(), 30);
    let ico2 = generate_spherical_icosphere(2).unwrap();
    assert_eq!(ico2.num_nodes(), 162);
    assert_eq!(ico2.num_edges(), 480);
    assert!(ico2.is_connected());
    assert!(generate_spherical_icosphere(8).is_err());

    let uv = generate_spherical_grid(4, 8).unwrap();
    assert_eq!(uv.num_nodes(), 34);
    assert!(uv.is_connected());

    let htree = generate_hyperbolic_tree(3, 3).unwrap();
    assert_eq!(htree.num_nodes(), 40);
    assert_eq!(htree.num_edges(), 39);
    assert!(htree.is_connected());
    assert_eq!(generate_hyperbolic_tree(3, 0).unwrap().num_nodes(), 1);
}

#[cfg(all(
    feature = "petgraph",
    feature = "1138_bus",
    feature = "3elt",
    feature = "dwt_1005",
    feature = "dwt_2680",
    feature = "poli",
    feature = "qh882",
    feature = "USpowerGrid",
    feature = "karate",
    feature = "lesmis"
))]
#[test]
fn test_all_embedded_datasets_match_petgraph_counts() {
    let cases: Vec<(UnGraph<(), ()>, CsrGraph<f64>)> = vec![
        (dataset_1138_bus(), dataset_1138_bus_csr()),
        (dataset_3_elt(), dataset_3_elt_csr()),
        (dataset_dwt_1005(), dataset_dwt_1005_csr()),
        (dataset_dwt_2680(), dataset_dwt_2680_csr()),
        (dataset_poli(), dataset_poli_csr()),
        (dataset_qh882(), dataset_qh882_csr()),
        (dataset_uspower_grid(), dataset_uspower_grid_csr()),
        (dataset_karate(), dataset_karate_csr()),
        (dataset_lesmis(), dataset_lesmis_csr()),
    ];

    for (pg, csr) in cases {
        assert_eq!(csr.num_nodes(), pg.node_count());
        assert_eq!(csr.num_edges(), pg.edge_count());
        assert!(!csr.is_directed());
        for u in 0..csr.num_nodes() {
            assert_eq!(csr.degree(u), pg.neighbors(NodeIndex::new(u)).count());
        }
    }
}
