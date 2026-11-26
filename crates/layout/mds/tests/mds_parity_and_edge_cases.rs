use egraph_dataset::dataset_1138_bus;
use petgraph::prelude::*;
use petgraph_drawing::Drawing;
use petgraph_layout_mds::{ClassicalMds, PivotMds};

fn build_grid_graph(rows: usize, cols: usize) -> (UnGraph<(), ()>, Vec<NodeIndex>) {
    let mut graph = Graph::new_undirected();
    let nodes: Vec<_> = (0..rows * cols).map(|_| graph.add_node(())).collect();
    for r in 0..rows {
        for c in 0..cols {
            let u = r * cols + c;
            if c + 1 < cols {
                graph.add_edge(nodes[u], nodes[u + 1], ());
            }
            if r + 1 < rows {
                graph.add_edge(nodes[u], nodes[u + cols], ());
            }
        }
    }
    (graph, nodes)
}

fn assert_embeddings_match(a: &[Vec<f64>], b: &[Vec<f64>], tol: f64, label: &str) {
    let n = a.len();
    assert_eq!(n, b.len());
    let d = a[0].len();

    let rms = |pts: &[Vec<f64>]| -> f64 {
        let sum_sq: f64 = pts
            .iter()
            .map(|p| p.iter().map(|&v| v * v).sum::<f64>())
            .sum();
        (sum_sq / n as f64).sqrt()
    };
    let rms_a = rms(a);
    let rms_b = rms(b);
    let rms_rel = (rms_a - rms_b).abs() / rms_a.max(rms_b).max(1.0);
    assert!(
        rms_rel < tol,
        "{label}: RMS mismatch: {rms_a} vs {rms_b} (rel={rms_rel})"
    );

    let max_gram = a
        .iter()
        .map(|p| p.iter().map(|&v| v * v).sum::<f64>())
        .fold(1.0f64, f64::max);
    for i in 0..n {
        for j in 0..n {
            let gram_a: f64 = (0..d).map(|k| a[i][k] * a[j][k]).sum();
            let gram_b: f64 = (0..d).map(|k| b[i][k] * b[j][k]).sum();
            assert!(
                (gram_a - gram_b).abs() / max_gram < tol,
                "{label}: Gram[{i},{j}] mismatch: {gram_a} vs {gram_b} (scale={max_gram})"
            );

            let dist_a: f64 = (0..d)
                .map(|k| (a[i][k] - a[j][k]).powi(2))
                .sum::<f64>()
                .sqrt();
            let dist_b: f64 = (0..d)
                .map(|k| (b[i][k] - b[j][k]).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                (dist_a - dist_b).abs() / dist_a.max(dist_b).max(1.0) < tol,
                "{label}: dist[{i},{j}] mismatch: {dist_a} vs {dist_b}"
            );
        }
    }
}

#[test]
fn test_pivot_mds_matches_classical_mds_all_pivots_2d_and_3d() {
    use petgraph_linalg_kernel::{EmbeddingKernelBuilder, KernelDistanceBuilder};

    // 1. 2D shortest-path distances on a 3x8 unweighted grid (all 24 nodes as pivots)
    let (graph, nodes) = build_grid_graph(3, 8);
    let mut cmds = ClassicalMds::<_, f64>::new(&graph, |_| 1.0);
    cmds.eps = 1e-8;
    let mut pmds = PivotMds::<_, f64>::new(&graph, |_| 1.0, &nodes);
    pmds.eps = 1e-8;

    let c_2d = cmds.run_2d();
    let p_2d = pmds.run_2d();
    let c_pts_2d: Vec<Vec<f64>> = nodes
        .iter()
        .map(|&u| vec![c_2d.x(u).unwrap(), c_2d.y(u).unwrap()])
        .collect();
    let p_pts_2d: Vec<Vec<f64>> = nodes
        .iter()
        .map(|&u| vec![p_2d.x(u).unwrap(), p_2d.y(u).unwrap()])
        .collect();
    assert_embeddings_match(&c_pts_2d, &p_pts_2d, 1e-3, "grid_3x8_2d");

    // 2. 2D and 3D exact Euclidean distances on a 2x3x4 3D lattice via KernelDistance
    // (PSD Gram matrix with 3 well-separated positive eigenvalues so run(3) is unambiguous)
    let mut g3d: UnGraph<(), ()> = Graph::new_undirected();
    let nodes3d: Vec<_> = (0..24).map(|_| g3d.add_node(())).collect();
    for i in 0..23 {
        g3d.add_edge(nodes3d[i], nodes3d[i + 1], ());
    }
    let mut emb = ndarray::Array2::<f64>::zeros((24, 3));
    for x in 0..2 {
        for y in 0..3 {
            for z in 0..4 {
                let idx = (x * 3 + y) * 4 + z;
                emb[[idx, 0]] = (x as f64 - 0.5) * 1.0;
                emb[[idx, 1]] = (y as f64 - 1.0) * 2.5;
                emb[[idx, 2]] = (z as f64 - 1.5) * 4.0;
            }
        }
    }
    let kernel = EmbeddingKernelBuilder::new().build(&g3d, emb).unwrap();
    let distance = KernelDistanceBuilder::new().build(kernel).unwrap();

    let mut cmds_3d = ClassicalMds::<NodeIndex, f64>::new_with_distance_matrix(&distance);
    cmds_3d.eps = 1e-8;
    let mut pmds_3d = PivotMds::<NodeIndex, f64>::new_with_distance_matrix(&distance);
    pmds_3d.eps = 1e-8;

    let c3d_2 = cmds_3d.run_2d();
    let p3d_2 = pmds_3d.run_2d();
    let c3d_pts_2: Vec<Vec<f64>> = nodes3d
        .iter()
        .map(|&u| vec![c3d_2.x(u).unwrap(), c3d_2.y(u).unwrap()])
        .collect();
    let p3d_pts_2: Vec<Vec<f64>> = nodes3d
        .iter()
        .map(|&u| vec![p3d_2.x(u).unwrap(), p3d_2.y(u).unwrap()])
        .collect();
    assert_embeddings_match(&c3d_pts_2, &p3d_pts_2, 1e-3, "box_2x3x4_2d");

    let c_3d = cmds_3d.run(3);
    let p_3d = pmds_3d.run(3);
    let c_pts_3d: Vec<Vec<f64>> = nodes3d
        .iter()
        .map(|&u| {
            (0..3)
                .map(|dim| c_3d.get(u, dim).unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    let p_pts_3d: Vec<Vec<f64>> = nodes3d
        .iter()
        .map(|&u| {
            (0..3)
                .map(|dim| p_3d.get(u, dim).unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_embeddings_match(&c_pts_3d, &p_pts_3d, 1e-3, "box_2x3x4_3d");
}

#[test]
fn test_pivot_mds_subsampled_scale_1138_bus() {
    let graph: UnGraph<(), ()> = dataset_1138_bus();
    let step = (graph.node_count() / 50).max(1);
    let pivots: Vec<_> = graph.node_indices().step_by(step).take(50).collect();

    let cmds = ClassicalMds::<_, f64>::new(&graph, |_| 1.0);
    let pmds = PivotMds::<_, f64>::new(&graph, |_| 1.0, &pivots);

    let c_2d = cmds.run_2d();
    let p_2d = pmds.run_2d();

    let n = graph.node_count() as f64;
    let mut sum_sq_c = 0.0;
    let mut sum_sq_p = 0.0;
    for u in graph.node_indices() {
        let cx = c_2d.x(u).unwrap();
        let cy = c_2d.y(u).unwrap();
        let px = p_2d.x(u).unwrap();
        let py = p_2d.y(u).unwrap();
        assert!(cx.is_finite() && cy.is_finite());
        assert!(px.is_finite() && py.is_finite());
        sum_sq_c += cx * cx + cy * cy;
        sum_sq_p += px * px + py * py;
    }

    let rms_c = (sum_sq_c / n).sqrt();
    let rms_p = (sum_sq_p / n).sqrt();
    let ratio = rms_p / rms_c;
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "1138_bus RMS scale mismatch: rms_pivot={rms_p}, rms_cmds={rms_c}, ratio={ratio}"
    );
}

#[test]
fn test_pivot_mds_edge_cases() {
    use petgraph_linalg_kernel::{EmbeddingKernelBuilder, KernelDistanceBuilder};

    // Empty graph (n = 0, k = 0)
    let empty_graph: UnGraph<(), ()> = Graph::new_undirected();
    let mds_empty = PivotMds::<NodeIndex, f64>::new(&empty_graph, |_| 1.0, &[]);
    let d2 = mds_empty.run_2d();
    assert_eq!(d2.len(), 0);
    let d3 = mds_empty.run(3);
    assert_eq!(d3.len(), 0);
    let d0 = mds_empty.run(0);
    assert_eq!(d0.len(), 0);

    // Non-empty graph with empty pivots (n = 4, k = 0)
    let mut g4 = Graph::new_undirected();
    let n4: Vec<_> = (0..4).map(|_| g4.add_node(())).collect();
    g4.add_edge(n4[0], n4[1], ());
    let mds_no_pivots = PivotMds::<NodeIndex, f64>::new(&g4, |_| 1.0, &[]);
    let d2_no_pivots = mds_no_pivots.run_2d();
    let d3_no_pivots = mds_no_pivots.run(3);
    let d0_no_pivots = mds_no_pivots.run(0);
    assert_eq!(d0_no_pivots.len(), 4);
    for &u in &n4 {
        assert_eq!(d2_no_pivots.x(u).unwrap(), 0.0);
        assert_eq!(d2_no_pivots.y(u).unwrap(), 0.0);
        for dim in 0..3 {
            assert_eq!(d3_no_pivots.get(u, dim).unwrap(), 0.0);
        }
    }

    // Single node (n = 1, k = 1)
    let mut g1: UnGraph<(), ()> = Graph::new_undirected();
    let u0 = g1.add_node(());
    let mds_single = PivotMds::<NodeIndex, f64>::new(&g1, |_| 1.0, &[u0]);
    let d2_single = mds_single.run_2d();
    let d3_single = mds_single.run(3);
    assert!(d2_single.x(u0).unwrap().is_finite());
    assert!(d2_single.y(u0).unwrap().is_finite());
    for dim in 0..3 {
        assert!(d3_single.get(u0, dim).unwrap().is_finite());
    }

    // Zero-distance KernelDistance (all eigenvalues == 0) + run(0)
    let mut g3 = Graph::new_undirected();
    let n3: Vec<_> = (0..3).map(|_| g3.add_node(())).collect();
    g3.add_edge(n3[0], n3[1], ());
    g3.add_edge(n3[1], n3[2], ());
    let embedding = ndarray::Array2::<f64>::zeros((3, 2));
    let kernel = EmbeddingKernelBuilder::new().build(&g3, embedding).unwrap();
    let distance = KernelDistanceBuilder::new().build(kernel).unwrap();
    let mds_zero = PivotMds::<NodeIndex, f64>::new_with_distance_matrix(&distance);
    let d2_zero = mds_zero.run_2d();
    let d3_zero = mds_zero.run(3);
    let d0_zero = mds_zero.run(0);
    assert_eq!(d0_zero.len(), 3);
    for &u in &n3 {
        assert_eq!(d2_zero.x(u).unwrap(), 0.0);
        assert_eq!(d2_zero.y(u).unwrap(), 0.0);
        for dim in 0..3 {
            assert_eq!(d3_zero.get(u, dim).unwrap(), 0.0);
        }
    }
}

#[test]
fn test_pivot_mds_disconnected_graph_and_k1() {
    let mut g = Graph::new_undirected();
    let n0 = g.add_node(());
    let n1 = g.add_node(());
    let n2 = g.add_node(());
    let n3 = g.add_node(());
    g.add_edge(n0, n1, ());
    g.add_edge(n2, n3, ());

    // Single pivot (k=1) on connected 5-node path must be finite
    let mut gp = Graph::new_undirected();
    let pn: Vec<_> = (0..5).map(|_| gp.add_node(())).collect();
    for i in 0..4 {
        gp.add_edge(pn[i], pn[i + 1], ());
    }
    let mds_k1 = PivotMds::<_, f64>::new(&gp, |_| 1.0, &[pn[0]]);
    let d_k1 = mds_k1.run_2d();
    for &u in &pn {
        assert!(d_k1.x(u).unwrap().is_finite());
        assert!(d_k1.y(u).unwrap().is_finite());
    }

    // Disconnected graph must produce finite coordinates
    let mds_disc = PivotMds::<_, f64>::new(&g, |_| 1.0, &[n0, n2]);
    let d_disc = mds_disc.run_2d();
    for &u in &[n0, n1, n2, n3] {
        assert!(d_disc.x(u).unwrap().is_finite());
        assert!(d_disc.y(u).unwrap().is_finite());
    }

    // ClassicalMds on disconnected graph must also produce finite coordinates
    let cmds_disc = ClassicalMds::<_, f64>::new(&g, |_| 1.0);
    let dc_disc = cmds_disc.run_2d();
    for &u in &[n0, n1, n2, n3] {
        assert!(dc_disc.x(u).unwrap().is_finite());
        assert!(dc_disc.y(u).unwrap().is_finite());
    }
}
