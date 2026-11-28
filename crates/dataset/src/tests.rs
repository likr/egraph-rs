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

#[test]
fn test_generate_weighted_graph_canonical_wrap_edge_and_seeds() {
    // Reproduces wrap-edge (num_nodes - 1, 0) colliding with random chord (0, num_nodes - 1).
    let g_repro = generate_weighted_graph(40, 6, 0.2, 5.0, 99).unwrap();
    assert_eq!(g_repro.num_nodes(), 40);
    assert!(g_repro.is_weighted());
    assert!(g_repro.is_connected());
    assert!(
        g_repro
            .weights()
            .unwrap()
            .iter()
            .all(|&w| (0.2..=5.0).contains(&w))
    );

    for &n in &[0usize, 1, 2, 3, 4, 10, 40, 80] {
        for &avg_deg in &[0usize, 1, 2, 6, 12] {
            for seed in 0..15u64 {
                let g = generate_weighted_graph(n, avg_deg, 0.1, 2.5, seed).unwrap();
                assert_eq!(g.num_nodes(), n);
                if n <= 1 || avg_deg == 0 {
                    assert_eq!(g.num_edges(), 0);
                } else {
                    assert!(g.is_weighted());
                    assert!(g.is_connected());
                    assert!(
                        g.weights()
                            .unwrap()
                            .iter()
                            .all(|&w| (0.1..=2.5).contains(&w))
                    );
                }
            }
        }
    }
}

#[test]
fn test_parse_unweighted_csr_csv_unit_and_errors() {
    let csv = "4,3\n0,1\r\n1,2\n2,3\n";
    let g: CsrGraph<()> = parse_unweighted_csr_csv(csv, false, false).unwrap();
    assert_eq!(g.num_nodes(), 4);
    assert_eq!(g.num_edges(), 3);
    assert!(!g.is_directed());
    assert!(!g.is_weighted());
    assert!(g.weights().is_none());
    assert_eq!(g.neighbors(1), &[0, 2]);

    let empty = parse_unweighted_csr_csv::<()>("0,0\n", false, false).unwrap();
    assert_eq!(empty.num_nodes(), 0);
    assert_eq!(empty.num_edges(), 0);

    let dg: CsrGraph<()> = parse_unweighted_csr_csv("3,2\n0,1\n1,0\n", true, false).unwrap();
    assert!(dg.is_directed());
    assert_eq!(dg.num_edges(), 2);

    let dirty = "3,3\n0,1\n1,0\n0,0\n";
    let clean: CsrGraph<()> = parse_unweighted_csr_csv(dirty, false, true).unwrap();
    assert_eq!(clean.num_nodes(), 3);
    assert_eq!(clean.num_edges(), 1);
    assert_eq!(clean.neighbors(0), &[1]);

    assert!(matches!(
        parse_unweighted_csr_csv::<()>("", false, false),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        parse_unweighted_csr_csv::<()>("3,2\n0,1\n", false, false),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        parse_unweighted_csr_csv::<()>("3,1\n0,1,2.0\n", false, false),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        parse_unweighted_csr_csv::<()>("3,1\n0\n", false, false),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        parse_unweighted_csr_csv::<()>("3,1\n0,abc\n", false, false),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        parse_unweighted_csr_csv::<()>("3,1\n0,4294967296\n", false, false),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        parse_unweighted_csr_csv::<()>("3,1\n0,5\n", false, false),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        parse_unweighted_csr_csv::<()>("3,1\n0,0\n", false, false),
        Err(CoreError::InvalidCsr(_))
    ));
}

#[test]
fn test_dataset_info_clone_and_catalog_consistency() {
    let info = dataset_info("karate").unwrap().clone();
    assert_eq!(info.name, "karate");
    assert_eq!(info.num_nodes, 34);
    assert_eq!(info.num_edges, 78);
    assert_eq!(dataset_catalog().len(), list_datasets().len());
    for entry in dataset_catalog() {
        assert!(list_datasets().contains(&entry.name));
        assert_eq!(dataset_info(entry.name), Some(entry));
    }
}

#[test]
fn test_unweighted_synthetic_generators_have_no_weights() {
    let dag = generate_layered_dag(3, 4, 0.4, 11).unwrap();
    assert!(!dag.is_weighted());
    assert!(dag.weights().is_none());
    let empty_dag = generate_layered_dag(0, 3, 0.5, 11).unwrap();
    assert_eq!(empty_dag.num_nodes(), 0);
    assert_eq!(empty_dag.num_edges(), 0);

    let (sbm, labels) = generate_sbm_graph(&[6, 7], 0.7, 0.1, 13).unwrap();
    assert!(!sbm.is_weighted());
    assert!(sbm.weights().is_none());
    assert_eq!(labels.len(), 13);
    let (empty_sbm, empty_labels) = generate_sbm_graph(&[], 0.5, 0.1, 13).unwrap();
    assert_eq!(empty_sbm.num_nodes(), 0);
    assert!(empty_labels.is_empty());

    let torus = generate_torus_grid(4, 5, false).unwrap();
    assert!(!torus.is_weighted());
    assert!(torus.weights().is_none());

    let ico = generate_spherical_icosphere(1).unwrap();
    assert!(!ico.is_weighted());
    assert!(ico.weights().is_none());
    assert_eq!(ico.num_nodes(), 42);
    assert_eq!(ico.num_edges(), 120);

    let grid = generate_spherical_grid(3, 6).unwrap();
    assert!(!grid.is_weighted());
    assert!(grid.weights().is_none());

    let htree = generate_hyperbolic_tree(2, 3).unwrap();
    assert!(!htree.is_weighted());
    assert!(htree.weights().is_none());
    assert_eq!(htree.num_nodes(), 15);
    assert_eq!(htree.num_edges(), 14);
}

#[cfg(all(
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
fn test_all_embedded_datasets_match_catalog_counts() {
    let cases: [(&str, CsrGraph<f64>); 9] = [
        ("1138_bus", dataset_1138_bus()),
        ("3elt", dataset_3_elt()),
        ("dwt_1005", dataset_dwt_1005()),
        ("dwt_2680", dataset_dwt_2680()),
        ("poli", dataset_poli()),
        ("qh882", dataset_qh882()),
        ("USpowerGrid", dataset_uspower_grid()),
        ("karate", dataset_karate()),
        ("lesmis", dataset_lesmis()),
    ];

    for (name, csr) in cases {
        let info = dataset_info(name).expect("embedded dataset must appear in catalog");
        assert_eq!(csr.num_nodes(), info.num_nodes, "{name} num_nodes");
        assert_eq!(csr.num_edges(), info.num_edges, "{name} num_edges");
        assert!(!csr.is_directed(), "{name} must be undirected");
        assert!(!csr.is_weighted(), "{name} must be unweighted");
        assert!(csr.weights().is_none(), "{name} must have no weights");
        let degree_sum: usize = (0..csr.num_nodes()).map(|u| csr.degree(u)).sum();
        assert_eq!(
            degree_sum,
            2 * csr.num_edges(),
            "{name} handshaking invariant"
        );
    }

    let karate = dataset_karate();
    assert_eq!(karate.degree(0), 16);
    assert_eq!(karate.degree(33), 17);
}
