use egraph_dataset::dataset_1138_bus;
use petgraph::prelude::*;
use petgraph_layout_mds::{ClassicalMds, PivotMds};

/// Test that reproduces the NaN issue with ClassicalMds when using dimensions higher than needed
#[test]
fn test_classical_mds_high_dimension() {
    // Create a simple line graph (low-dimensional structure)
    let mut graph = Graph::new_undirected();
    let n1 = graph.add_node(());
    let n2 = graph.add_node(());
    let n3 = graph.add_node(());
    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());

    // Run ClassicalMds with dimensions higher than the number of nodes
    let mds = ClassicalMds::<_, f32>::new(&graph, |_| 1.0);
    let drawing = mds.run(5); // 5 dimensions (more than the 3 nodes)

    // Check that coordinates are not NaN
    for u in graph.node_indices() {
        for d in 0..5 {
            let value = drawing.get(u, d).unwrap();
            assert!(
                !value.is_nan(),
                "Coordinate is NaN at node {:?}, dimension {}",
                u,
                d
            );
        }
    }
}

/// Test ClassicalMds with various dimensions to ensure robustness
#[test]
fn test_classical_mds_various_dimensions() {
    // Create a more complex graph
    let mut graph = Graph::new_undirected();
    let nodes: Vec<_> = (0..10).map(|_| graph.add_node(())).collect();

    // Add some edges
    for i in 0..9 {
        graph.add_edge(nodes[i], nodes[i + 1], ());
    }
    graph.add_edge(nodes[0], nodes[5], ());
    graph.add_edge(nodes[2], nodes[7], ());

    let mds = ClassicalMds::<_, f32>::new(&graph, |_| 1.0);

    // Test with various dimensions
    for dim in 1..15 {
        let drawing = mds.run(dim);

        // Check that coordinates are not NaN
        for u in graph.node_indices() {
            for d in 0..dim {
                let value = drawing.get(u, d).unwrap();
                assert!(
                    !value.is_nan(),
                    "Coordinate is NaN at node {:?}, dimension {} (total dim: {})",
                    u,
                    d,
                    dim
                );
            }
        }
    }
}

#[test]
fn test_classical_mds_2d() {
    let graph: UnGraph<(), ()> = dataset_1138_bus();
    let mds = ClassicalMds::<_, f32>::new(&graph, |_| 1.);
    let drawing = mds.run_2d();
    for u in graph.node_indices() {
        assert!(drawing.x(u).unwrap().is_finite());
        assert!(drawing.y(u).unwrap().is_finite());
    }
}

#[test]
fn test_classical_mds_3d() {
    let graph: UnGraph<(), ()> = dataset_1138_bus();
    let mds = ClassicalMds::<_, f32>::new(&graph, |_| 1.);
    let drawing = mds.run(3);
    for u in graph.node_indices() {
        assert!(drawing.get(u, 0).unwrap().is_finite());
        assert!(drawing.get(u, 1).unwrap().is_finite());
        assert!(drawing.get(u, 2).unwrap().is_finite());
    }
}

#[test]
fn test_pivot_mds_2d() {
    let graph: UnGraph<(), ()> = dataset_1138_bus();
    let pivot = graph.node_indices().take(50).collect::<Vec<_>>();
    let mds = PivotMds::<_, f32>::new(&graph, |_| 1., &pivot);
    let drawing = mds.run_2d();
    for u in graph.node_indices() {
        assert!(drawing.x(u).unwrap().is_finite());
        assert!(drawing.y(u).unwrap().is_finite());
    }
}

#[test]
fn test_pivot_mds_3d() {
    let graph: UnGraph<(), ()> = dataset_1138_bus();
    let pivot = graph.node_indices().take(50).collect::<Vec<_>>();
    let mds = PivotMds::<_, f32>::new(&graph, |_| 1., &pivot);
    let drawing = mds.run(3);
    for u in graph.node_indices() {
        assert!(drawing.get(u, 0).unwrap().is_finite());
        assert!(drawing.get(u, 1).unwrap().is_finite());
        assert!(drawing.get(u, 2).unwrap().is_finite());
    }
}

/// Test that reproduces the NaN issue with PivotMds when using dimensions higher than needed
#[test]
fn test_pivot_mds_high_dimension() {
    // Create a simple line graph (low-dimensional structure)
    let mut graph = Graph::new_undirected();
    let n1 = graph.add_node(());
    let n2 = graph.add_node(());
    let n3 = graph.add_node(());
    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());

    // Use n1 and n3 as pivot nodes
    let pivot_nodes = vec![n1, n3];

    // Run PivotMds with dimensions higher than the number of nodes
    let mds = PivotMds::<_, f32>::new(&graph, |_| 1.0, &pivot_nodes);
    let drawing = mds.run(5); // 5 dimensions (more than the 3 nodes)

    // Check that coordinates are not NaN
    for u in graph.node_indices() {
        for d in 0..5 {
            let value = drawing.get(u, d).unwrap();
            assert!(
                !value.is_nan(),
                "Coordinate is NaN at node {:?}, dimension {}",
                u,
                d
            );
        }
    }
}

/// Test PivotMds with various dimensions to ensure robustness
#[test]
fn test_pivot_mds_various_dimensions() {
    // Create a more complex graph
    let mut graph = Graph::new_undirected();
    let nodes: Vec<_> = (0..10).map(|_| graph.add_node(())).collect();

    // Add some edges
    for i in 0..9 {
        graph.add_edge(nodes[i], nodes[i + 1], ());
    }
    graph.add_edge(nodes[0], nodes[5], ());
    graph.add_edge(nodes[2], nodes[7], ());

    // Use a subset of nodes as pivots
    let pivot_nodes = vec![nodes[0], nodes[3], nodes[6], nodes[9]];

    let mds = PivotMds::<_, f32>::new(&graph, |_| 1.0, &pivot_nodes);

    // Test with various dimensions
    for dim in 1..15 {
        let drawing = mds.run(dim);

        // Check that coordinates are not NaN
        for u in graph.node_indices() {
            for d in 0..dim {
                let value = drawing.get(u, d).unwrap();
                assert!(
                    !value.is_nan(),
                    "Coordinate is NaN at node {:?}, dimension {} (total dim: {})",
                    u,
                    d,
                    dim
                );
            }
        }
    }
}
