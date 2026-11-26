use petgraph::Graph;
use petgraph_algorithm_shortest_path::all_sources_dijkstra;
use petgraph_drawing::DrawingEuclidean2d;
use petgraph_layout_ts_net::{BhTsNetBuilder, FitTsNetBuilder, LTsNetBuilder, TsNet, TsNetBuilder};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn test_ts_net_basic() {
    let mut graph = Graph::new_undirected();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    graph.add_edge(a, b, ());
    graph.add_edge(b, c, ());
    graph.add_edge(c, a, ());

    let mut coordinates = DrawingEuclidean2d::initial_placement(&graph);
    let distance_matrix = all_sources_dijkstra(&graph, |_| 1.0f32);

    let ts_net = TsNetBuilder::new()
        .learning_rate(2.0f32)
        .n_epochs(30)
        .build()
        .expect("valid builder configuration");

    ts_net.run(&mut coordinates, &distance_matrix);

    for &u in &[a, b, c] {
        let x = coordinates.x(u).unwrap();
        let y = coordinates.y(u).unwrap();
        assert!(x.is_finite());
        assert!(y.is_finite());
    }
}

#[test]
fn test_bh_ts_net_basic() {
    let mut graph = Graph::new_undirected();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    graph.add_edge(a, b, ());
    graph.add_edge(b, c, ());
    graph.add_edge(c, d, ());
    graph.add_edge(d, a, ());

    let mut coordinates = DrawingEuclidean2d::initial_placement(&graph);
    let mut rng = StdRng::seed_from_u64(42);

    let bh_ts_net = BhTsNetBuilder::new()
        .perplexity(2.0f32)
        .k(3)
        .learning_rate(2.0f32)
        .iterations_stage1(10)
        .iterations_stage2(10)
        .iterations_stage3(10)
        .build()
        .expect("valid builder configuration");

    bh_ts_net.run(&mut coordinates, &graph, &mut rng);

    for &u in &[a, b, c, d] {
        let x = coordinates.x(u).unwrap();
        let y = coordinates.y(u).unwrap();
        assert!(x.is_finite());
        assert!(y.is_finite());
    }
}

#[test]
fn test_fit_ts_net_basic() {
    let mut graph = Graph::new_undirected();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    graph.add_edge(a, b, ());
    graph.add_edge(b, c, ());
    graph.add_edge(c, d, ());
    graph.add_edge(d, a, ());

    let mut coordinates = DrawingEuclidean2d::initial_placement(&graph);
    let mut rng = StdRng::seed_from_u64(42);

    let fit_ts_net = FitTsNetBuilder::new()
        .intervals(10)
        .interpolation_points(3)
        .perplexity(2.0f32)
        .k(3)
        .learning_rate(2.0f32)
        .iterations_stage1(10)
        .iterations_stage2(10)
        .iterations_stage3(10)
        .build()
        .expect("valid builder configuration");

    fit_ts_net.run(&mut coordinates, &graph, &mut rng);

    for &u in &[a, b, c, d] {
        let x = coordinates.x(u).unwrap();
        let y = coordinates.y(u).unwrap();
        assert!(x.is_finite());
        assert!(y.is_finite());
    }
}

#[test]
fn test_l_ts_net_basic() {
    let mut graph = Graph::new_undirected();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    graph.add_edge(a, b, ());
    graph.add_edge(b, c, ());
    graph.add_edge(c, d, ());
    graph.add_edge(d, a, ());

    let mut coordinates = DrawingEuclidean2d::initial_placement(&graph);
    let mut rng = StdRng::seed_from_u64(42);

    let l_ts_net = LTsNetBuilder::new()
        .intervals(10)
        .interpolation_points(3)
        .perplexity(2.0f32)
        .k(3)
        .learning_rate(2.0f32)
        .iterations_stage1(10)
        .iterations_stage2(10)
        .iterations_stage3(10)
        .build()
        .expect("valid builder configuration");

    l_ts_net.run(&mut coordinates, &graph, &mut rng);

    for &u in &[a, b, c, d] {
        let x = coordinates.x(u).unwrap();
        let y = coordinates.y(u).unwrap();
        assert!(x.is_finite());
        assert!(y.is_finite());
    }
}

#[test]
fn test_ts_net_via_default_constructor() {
    let mut graph = Graph::new_undirected();
    let a = graph.add_node(());
    let b = graph.add_node(());
    graph.add_edge(a, b, ());

    let mut coordinates = DrawingEuclidean2d::initial_placement(&graph);
    let distance_matrix = all_sources_dijkstra(&graph, |_| 1.0f32);

    let ts_net = TsNet::<f32>::new();
    ts_net.run(&mut coordinates, &distance_matrix);

    for &u in &[a, b] {
        let x = coordinates.x(u).unwrap();
        let y = coordinates.y(u).unwrap();
        assert!(x.is_finite());
        assert!(y.is_finite());
    }
}

fn evaluate_layout_quality(
    drawing: &DrawingEuclidean2d<petgraph::graph::NodeIndex, f64>,
    nodes: &[petgraph::graph::NodeIndex],
    edges: &[(usize, usize)],
) -> (f64, f64, f64) {
    let n = nodes.len();
    let mut max_abs = 0.0f64;
    let pts: Vec<[f64; 2]> = nodes
        .iter()
        .map(|&u| {
            let x = drawing.x(u).unwrap();
            let y = drawing.y(u).unwrap();
            assert!(x.is_finite() && y.is_finite(), "non-finite coordinate");
            max_abs = max_abs.max(x.abs()).max(y.abs());
            [x, y]
        })
        .collect();
    let mean_edge = edges
        .iter()
        .map(|&(u, v)| {
            let dx = pts[u][0] - pts[v][0];
            let dy = pts[u][1] - pts[v][1];
            (dx * dx + dy * dy).sqrt()
        })
        .sum::<f64>()
        / edges.len() as f64;
    let mut total_all = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let dx = pts[i][0] - pts[j][0];
                let dy = pts[i][1] - pts[j][1];
                total_all += (dx * dx + dy * dy).sqrt();
            }
        }
    }
    let mean_all = total_all / (n * (n - 1)) as f64;
    (max_abs, mean_edge, mean_all)
}

#[test]
fn test_bh_fit_l_ts_net_grid_5x5_and_cycle_30_compact_and_coherent() {
    // grid_5x5 (n=25)
    let mut grid_graph = Graph::new_undirected();
    let grid_nodes: Vec<_> = (0..25).map(|_| grid_graph.add_node(())).collect();
    let mut grid_edges = Vec::new();
    for r in 0..5 {
        for c in 0..5 {
            let u = r * 5 + c;
            if c + 1 < 5 {
                grid_graph.add_edge(grid_nodes[u], grid_nodes[u + 1], ());
                grid_edges.push((u, u + 1));
            }
            if r + 1 < 5 {
                grid_graph.add_edge(grid_nodes[u], grid_nodes[u + 5], ());
                grid_edges.push((u, u + 5));
            }
        }
    }

    // cycle_30 (n=30)
    let mut cycle_graph = Graph::new_undirected();
    let cycle_nodes: Vec<_> = (0..30).map(|_| cycle_graph.add_node(())).collect();
    let mut cycle_edges = Vec::new();
    for i in 0..30 {
        let j = (i + 1) % 30;
        cycle_graph.add_edge(cycle_nodes[i], cycle_nodes[j], ());
        cycle_edges.push((i, j));
    }

    for (label, graph, nodes, edges, perp, large_init) in [
        (
            "grid_5x5",
            &grid_graph,
            &grid_nodes,
            &grid_edges,
            8.0f64,
            false,
        ),
        (
            "cycle_30",
            &cycle_graph,
            &cycle_nodes,
            &cycle_edges,
            10.0f64,
            false,
        ),
        (
            "cycle_30_large_init",
            &cycle_graph,
            &cycle_nodes,
            &cycle_edges,
            10.0f64,
            true,
        ),
    ] {
        let init_drawing = |g: &Graph<(), (), petgraph::Undirected>| {
            let mut d = DrawingEuclidean2d::<_, f64>::initial_placement(g);
            if large_init {
                for (idx, &u) in nodes.iter().enumerate() {
                    d.set_x(u, (idx as f64 - 14.5) * 1000.0);
                    d.set_y(u, ((idx % 5) as f64 - 2.0) * 500.0);
                }
            }
            d
        };

        let bh = BhTsNetBuilder::new().perplexity(perp).build().unwrap();
        let mut d_bh = init_drawing(graph);
        bh.run(&mut d_bh, graph, &mut StdRng::seed_from_u64(42));
        let (max_abs, me, ma) = evaluate_layout_quality(&d_bh, nodes, edges);
        assert!(
            max_abs < 100.0 && me < 0.6 * ma,
            "{label} BhTsNet failed: max_abs={max_abs:.2}, edge={me:.2}, all={ma:.2}"
        );

        let fit = FitTsNetBuilder::new().perplexity(perp).build().unwrap();
        let mut d_fit = init_drawing(graph);
        fit.run(&mut d_fit, graph, &mut StdRng::seed_from_u64(42));
        let (max_abs, me, ma) = evaluate_layout_quality(&d_fit, nodes, edges);
        assert!(
            max_abs < 100.0 && me < 0.6 * ma,
            "{label} FitTsNet failed: max_abs={max_abs:.2}, edge={me:.2}, all={ma:.2}"
        );

        let l = LTsNetBuilder::new().perplexity(perp).build().unwrap();
        let mut d_l = init_drawing(graph);
        l.run(&mut d_l, graph, &mut StdRng::seed_from_u64(42));
        let (max_abs, me, ma) = evaluate_layout_quality(&d_l, nodes, edges);
        assert!(
            max_abs < 100.0 && me < 0.6 * ma,
            "{label} LTsNet failed: max_abs={max_abs:.2}, edge={me:.2}, all={ma:.2}"
        );
    }
}

#[test]
fn test_l_ts_net_entropy_repels_two_points_like_bh_ts_net() {
    let mut graph: Graph<(), (), petgraph::Undirected> = Graph::new_undirected();
    let n0 = graph.add_node(());
    let n1 = graph.add_node(());
    let knn: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 2];

    let mut d_l = DrawingEuclidean2d::<_, f64>::initial_placement(&graph);
    d_l.set_x(n0, -2.0);
    d_l.set_y(n0, 0.0);
    d_l.set_x(n1, 2.0);
    d_l.set_y(n1, 0.0);

    let mut d_bh = DrawingEuclidean2d::<_, f64>::initial_placement(&graph);
    d_bh.set_x(n0, -2.0);
    d_bh.set_y(n0, 0.0);
    d_bh.set_x(n1, 2.0);
    d_bh.set_y(n1, 0.0);

    let l_ts_net = LTsNetBuilder::new()
        .perplexity(1.0f64)
        .intervals(25)
        .interpolation_points(3)
        .iterations_stage1(0)
        .iterations_stage2(0)
        .iterations_stage3(5)
        .lambda_c_stage3(0.0)
        .lambda_r_stage3(10.0)
        .learning_rate(2.0)
        .momentum(0.0)
        .build()
        .unwrap();
    l_ts_net.run_with_knn(&mut d_l, &knn);

    let bh_ts_net = BhTsNetBuilder::new()
        .perplexity(1.0f64)
        .theta(0.5)
        .iterations_stage1(0)
        .iterations_stage2(0)
        .iterations_stage3(5)
        .lambda_c_stage3(0.0)
        .lambda_r_stage3(10.0)
        .learning_rate(2.0)
        .momentum(0.0)
        .build()
        .unwrap();
    bh_ts_net.run_with_knn(&mut d_bh, &knn);

    let l_sep = d_l.x(n1).unwrap() - d_l.x(n0).unwrap();
    let bh_sep = d_bh.x(n1).unwrap() - d_bh.x(n0).unwrap();
    assert!(
        l_sep > 4.0,
        "LTsNet entropy should repel points apart from 4.0, got {l_sep}"
    );
    assert!(
        bh_sep > 4.0,
        "BhTsNet entropy should repel points apart from 4.0, got {bh_sep}"
    );
    let rel_diff = (l_sep - bh_sep).abs() / bh_sep;
    assert!(
        rel_diff < 0.05,
        "LTsNet separation ({l_sep}) diverged from BhTsNet ({bh_sep}), rel_diff={rel_diff}"
    );
}
