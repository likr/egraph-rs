use petgraph::Graph;
use petgraph_drawing::DrawingEuclidean2d;
use petgraph_layout_ts_net::{
    compute_knn_probabilities, compute_partial_bfs_probabilities, BhTsNetBuilder, FitTsNetBuilder,
    LTsNetBuilder, SparseP,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn assert_sparse_p_symmetric_and_normalized(sparse_p: &SparseP<f64>, expected_sum: f64) {
    let n = sparse_p.len();
    let mut total = 0.0;
    for (i, row) in sparse_p.iter().enumerate() {
        let mut prev = None;
        for &(j, p_ij) in row {
            assert_ne!(i, j, "self-loop found in sparse_p[{i}]");
            assert!(j < n, "out-of-range neighbor {j} in sparse_p[{i}]");
            assert!(p_ij.is_finite() && p_ij > 0.0);
            if let Some(p) = prev {
                assert!(p < j, "duplicate or unsorted neighbor {j} in sparse_p[{i}]");
            }
            prev = Some(j);
            let p_ji = sparse_p[j]
                .iter()
                .find(|&&(col, _)| col == i)
                .map(|&(_, v)| v)
                .expect("missing symmetric reverse entry");
            assert!(
                (p_ij - p_ji).abs() < 1e-12,
                "asymmetry at ({i}, {j}): {p_ij} vs {p_ji}"
            );
            total += p_ij;
        }
    }
    assert!(
        (total - expected_sum).abs() < 1e-6,
        "expected probability sum {expected_sum}, got {total}"
    );
}

#[test]
fn test_compute_knn_probabilities_directed_symmetric_sum() {
    // Asymmetric directed k-NN graph: node 0 -> (1, 2), node 1 -> (2, 3), etc.
    let knn: Vec<Vec<(usize, f64)>> = vec![
        vec![(1, 0.2), (2, 1.0), (3, 2.5)],
        vec![(2, 0.3), (3, 1.1), (0, 2.0)],
        vec![(3, 0.4), (4, 0.9), (1, 1.8)],
        vec![(4, 0.25), (5, 1.2), (2, 2.1)],
        vec![(5, 0.35), (0, 1.5), (3, 2.2)],
        vec![(0, 0.45), (1, 1.4), (4, 2.4)],
    ];
    let sparse_p = compute_knn_probabilities(6, &knn, 2.0);
    assert_sparse_p_symmetric_and_normalized(&sparse_p, 1.0);
}

#[test]
fn test_compute_knn_probabilities_filters_and_edge_cases() {
    // Empty and tiny n
    assert!(compute_knn_probabilities::<f64>(0, &[], 2.0).is_empty());
    assert_eq!(
        compute_knn_probabilities::<f64>(1, &[vec![(0, 1.0)]], 2.0),
        vec![Vec::<(usize, f64)>::new()]
    );

    // Self-loops, out-of-range, duplicates (keeping min distance), non-finite, and empty row
    let knn: Vec<Vec<(usize, f64)>> = vec![
        vec![
            (0, 0.0),           // self-loop -> filtered
            (99, 0.1),          // out of range -> filtered
            (1, 5.0),           // duplicate neighbor 1 -> min(5.0, 0.5) = 0.5
            (1, 0.5),           // duplicate with smaller distance
            (2, f64::NAN),      // non-finite -> filtered
            (3, f64::INFINITY), // non-finite -> filtered
            (2, 1.5),
        ],
        vec![(0, 0.5), (2, 1.0)],
        vec![(1, 1.0), (0, 1.5)],
        vec![], // empty row
    ];
    let sparse_p = compute_knn_probabilities(4, &knn, 1.5);
    // 3 out of 4 rows have neighbors => sum = 3 / 4 = 0.75
    assert_sparse_p_symmetric_and_normalized(&sparse_p, 0.75);

    // Verify duplicate neighbor 1 used distance 0.5 (so p_{1|0} > p_{2|0})
    let p01 = sparse_p[0].iter().find(|&&(j, _)| j == 1).unwrap().1;
    let p02 = sparse_p[0].iter().find(|&&(j, _)| j == 2).unwrap().1;
    assert!(p01 > p02);

    // Extreme tiny scale (1e-10) should still converge cleanly without underflow
    let tiny_knn: Vec<Vec<(usize, f64)>> = vec![
        vec![(1, 1e-10), (2, 1e-9), (3, 1e-8)],
        vec![(0, 1e-10), (2, 1e-9), (3, 1e-8)],
        vec![(0, 1e-9), (1, 1e-9), (3, 1e-10)],
        vec![(2, 1e-10), (1, 1e-8), (0, 1e-8)],
    ];
    let tiny_p = compute_knn_probabilities(4, &tiny_knn, 2.0);
    assert_sparse_p_symmetric_and_normalized(&tiny_p, 1.0);
}

#[test]
fn test_partial_bfs_matches_knn_probabilities() {
    let mut graph = Graph::new_undirected();
    let nodes: Vec<_> = (0..6).map(|_| graph.add_node(())).collect();
    for i in 0..6 {
        graph.add_edge(nodes[i], nodes[(i + 1) % 6], ());
    }
    // k = 4 visits hop-1 (2 nodes) and hop-2 (2 nodes) with zero ties on C_6
    let mut rng = StdRng::seed_from_u64(42);
    let bfs_p = compute_partial_bfs_probabilities(&graph, 2.5f64, 4, 2.0, 50, 1e-5, &mut rng);

    let exact_knn: Vec<Vec<(usize, f64)>> = (0..6)
        .map(|i| {
            vec![
                ((i + 1) % 6, 1.0),
                ((i + 5) % 6, 1.0),
                ((i + 2) % 6, 2.0),
                ((i + 4) % 6, 2.0),
            ]
        })
        .collect();
    let knn_p = compute_knn_probabilities(6, &exact_knn, 2.5f64);

    assert_sparse_p_symmetric_and_normalized(&bfs_p, 1.0);
    assert_eq!(bfs_p.len(), knn_p.len());
    for i in 0..6 {
        assert_eq!(bfs_p[i].len(), knn_p[i].len());
        for (&(j_a, p_a), &(j_b, p_b)) in bfs_p[i].iter().zip(knn_p[i].iter()) {
            assert_eq!(j_a, j_b);
            assert!((p_a - p_b).abs() < 1e-12);
        }
    }
}

type ClusterKnnFixture = (
    petgraph::Graph<(), (), petgraph::Undirected>,
    Vec<petgraph::graph::NodeIndex>,
    Vec<Vec<(usize, f32)>>,
);

fn build_two_cluster_knn(n_per_cluster: usize) -> ClusterKnnFixture {
    let n = 2 * n_per_cluster;
    let mut graph = Graph::new_undirected();
    let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
    let mut knn = vec![Vec::new(); n];
    for (i, row) in knn.iter_mut().enumerate() {
        let same_start = if i < n_per_cluster { 0 } else { n_per_cluster };
        for j in 0..n {
            if i == j {
                continue;
            }
            let in_same = j >= same_start && j < same_start + n_per_cluster;
            let dist = if in_same { 0.1f32 } else { 5.0f32 };
            row.push((j, dist));
        }
    }
    (graph, nodes, knn)
}

#[test]
fn test_bh_fit_l_ts_net_run_with_knn() {
    let (graph, nodes, knn) = build_two_cluster_knn(4);

    let bh = BhTsNetBuilder::new()
        .perplexity(2.5f32)
        .learning_rate(5.0f32)
        .iterations_stage1(15)
        .iterations_stage2(15)
        .iterations_stage3(15)
        .build()
        .unwrap();
    let mut bh_drawing = DrawingEuclidean2d::initial_placement(&graph);
    bh.run_with_knn(&mut bh_drawing, &knn);
    for &u in &nodes {
        assert!(bh_drawing.x(u).unwrap().is_finite());
        assert!(bh_drawing.y(u).unwrap().is_finite());
    }

    let fit = FitTsNetBuilder::new()
        .intervals(10)
        .interpolation_points(3)
        .perplexity(2.5f32)
        .learning_rate(5.0f32)
        .iterations_stage1(15)
        .iterations_stage2(15)
        .iterations_stage3(15)
        .build()
        .unwrap();
    let mut fit_drawing = DrawingEuclidean2d::initial_placement(&graph);
    fit.run_with_knn(&mut fit_drawing, &knn);
    for &u in &nodes {
        assert!(fit_drawing.x(u).unwrap().is_finite());
        assert!(fit_drawing.y(u).unwrap().is_finite());
    }

    let l = LTsNetBuilder::new()
        .intervals(10)
        .interpolation_points(3)
        .perplexity(2.5f32)
        .learning_rate(5.0f32)
        .iterations_stage1(15)
        .iterations_stage2(15)
        .iterations_stage3(15)
        .build()
        .unwrap();
    let mut l_drawing = DrawingEuclidean2d::initial_placement(&graph);
    l.run_with_knn(&mut l_drawing, &knn);
    for &u in &nodes {
        assert!(l_drawing.x(u).unwrap().is_finite());
        assert!(l_drawing.y(u).unwrap().is_finite());
    }
}

#[test]
fn test_run_with_knn_matches_run_on_cycle() {
    let mut graph = Graph::new_undirected();
    let nodes: Vec<_> = (0..6).map(|_| graph.add_node(())).collect();
    for i in 0..6 {
        graph.add_edge(nodes[i], nodes[(i + 1) % 6], ());
    }
    let knn: Vec<Vec<(usize, f32)>> = (0..6)
        .map(|i| {
            vec![
                ((i + 1) % 6, 1.0),
                ((i + 5) % 6, 1.0),
                ((i + 2) % 6, 2.0),
                ((i + 4) % 6, 2.0),
            ]
        })
        .collect();

    let bh = BhTsNetBuilder::new()
        .perplexity(2.5f32)
        .k(4)
        .learning_rate(2.0f32)
        .iterations_stage1(10)
        .iterations_stage2(10)
        .iterations_stage3(10)
        .build()
        .unwrap();
    let mut d_run = DrawingEuclidean2d::initial_placement(&graph);
    let mut d_knn = DrawingEuclidean2d::initial_placement(&graph);
    let mut rng = StdRng::seed_from_u64(42);
    bh.run(&mut d_run, &graph, &mut rng);
    bh.run_with_knn(&mut d_knn, &knn);

    for &u in &nodes {
        assert!((d_run.x(u).unwrap() - d_knn.x(u).unwrap()).abs() < 1e-4);
        assert!((d_run.y(u).unwrap() - d_knn.y(u).unwrap()).abs() < 1e-4);
    }

    let fit = FitTsNetBuilder::new()
        .intervals(10)
        .interpolation_points(3)
        .perplexity(2.5f32)
        .k(4)
        .learning_rate(2.0f32)
        .iterations_stage1(10)
        .iterations_stage2(10)
        .iterations_stage3(10)
        .build()
        .unwrap();
    let mut fit_run = DrawingEuclidean2d::initial_placement(&graph);
    let mut fit_knn = DrawingEuclidean2d::initial_placement(&graph);
    let mut rng_fit = StdRng::seed_from_u64(42);
    fit.run(&mut fit_run, &graph, &mut rng_fit);
    fit.run_with_knn(&mut fit_knn, &knn);

    for &u in &nodes {
        assert!((fit_run.x(u).unwrap() - fit_knn.x(u).unwrap()).abs() < 1e-4);
        assert!((fit_run.y(u).unwrap() - fit_knn.y(u).unwrap()).abs() < 1e-4);
    }

    let l = LTsNetBuilder::new()
        .intervals(10)
        .interpolation_points(3)
        .perplexity(2.5f32)
        .k(4)
        .learning_rate(2.0f32)
        .iterations_stage1(10)
        .iterations_stage2(10)
        .iterations_stage3(10)
        .build()
        .unwrap();
    let mut l_run = DrawingEuclidean2d::initial_placement(&graph);
    let mut l_knn = DrawingEuclidean2d::initial_placement(&graph);
    let mut rng_l = StdRng::seed_from_u64(42);
    l.run(&mut l_run, &graph, &mut rng_l);
    l.run_with_knn(&mut l_knn, &knn);

    for &u in &nodes {
        assert!((l_run.x(u).unwrap() - l_knn.x(u).unwrap()).abs() < 1e-4);
        assert!((l_run.y(u).unwrap() - l_knn.y(u).unwrap()).abs() < 1e-4);
    }

    // Single-node graph edge case should be a clean no-op
    let mut g1: Graph<(), (), petgraph::Undirected> = Graph::new_undirected();
    let n0 = g1.add_node(());
    let mut d1 = DrawingEuclidean2d::<_, f32>::initial_placement(&g1);
    bh.run_with_knn(&mut d1, &[vec![]]);
    fit.run_with_knn(&mut d1, &[vec![]]);
    l.run_with_knn(&mut d1, &[vec![]]);
    assert!(d1.x(n0).unwrap().is_finite());
    assert!(d1.y(n0).unwrap().is_finite());
}

#[test]
fn test_fit_and_l_ts_net_default_250_iters_bounded() {
    let (graph, nodes, knn) = build_two_cluster_knn(5);

    let fit = FitTsNetBuilder::new().perplexity(3.0f32).build().unwrap();
    assert_eq!(fit.iterations_stage1, 250);
    assert_eq!(fit.iterations_stage2, 250);
    assert_eq!(fit.iterations_stage3, 250);

    let mut fit_drawing = DrawingEuclidean2d::initial_placement(&graph);
    fit.run_with_knn(&mut fit_drawing, &knn);
    for &u in &nodes {
        let x = fit_drawing.x(u).unwrap();
        let y = fit_drawing.y(u).unwrap();
        assert!(x.is_finite() && x.abs() < 1e3, "FitTsNet unbounded x={x}");
        assert!(y.is_finite() && y.abs() < 1e3, "FitTsNet unbounded y={y}");
    }

    let l = LTsNetBuilder::new().perplexity(3.0f32).build().unwrap();
    let mut l_drawing = DrawingEuclidean2d::initial_placement(&graph);
    l.run_with_knn(&mut l_drawing, &knn);
    for &u in &nodes {
        let x = l_drawing.x(u).unwrap();
        let y = l_drawing.y(u).unwrap();
        assert!(x.is_finite() && x.abs() < 1e3, "LTsNet unbounded x={x}");
        assert!(y.is_finite() && y.abs() < 1e3, "LTsNet unbounded y={y}");
    }
}
