use petgraph::Graph;
use petgraph_drawing::DrawingEuclidean2d;
use petgraph_layout_ts_net::{BhTsNetBuilder, FitTsNetBuilder, LTsNetBuilder};
use rand::rngs::StdRng;
use rand::SeedableRng;

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

fn knn_to_csr<S: Copy>(knn: &[Vec<(usize, S)>]) -> (Vec<usize>, Vec<usize>, Vec<S>) {
    let mut indptr = Vec::with_capacity(knn.len() + 1);
    let mut indices = Vec::new();
    let mut distances = Vec::new();
    indptr.push(0);
    for row in knn {
        for &(j, d) in row {
            indices.push(j);
            distances.push(d);
        }
        indptr.push(indices.len());
    }
    (indptr, indices, distances)
}

#[test]
fn test_run_with_knn_adaptive_matches_run_with_knn_when_uniform() {
    let (graph, nodes, knn) = build_two_cluster_knn(4);
    let n = nodes.len();
    let perp = 2.5f32;
    let perplexities = vec![perp; n];
    let (indptr, indices, distances) = knn_to_csr(&knn);

    let bh = BhTsNetBuilder::new()
        .perplexity(perp)
        .learning_rate(5.0f32)
        .iterations_stage1(15)
        .iterations_stage2(15)
        .iterations_stage3(15)
        .build()
        .unwrap();
    let fit = FitTsNetBuilder::new()
        .intervals(10)
        .interpolation_points(3)
        .perplexity(perp)
        .learning_rate(5.0f32)
        .iterations_stage1(15)
        .iterations_stage2(15)
        .iterations_stage3(15)
        .build()
        .unwrap();
    let l = LTsNetBuilder::new()
        .intervals(10)
        .interpolation_points(3)
        .perplexity(perp)
        .learning_rate(5.0f32)
        .iterations_stage1(15)
        .iterations_stage2(15)
        .iterations_stage3(15)
        .build()
        .unwrap();

    let assert_same =
        |d_a: &DrawingEuclidean2d<_, f32>, d_b: &DrawingEuclidean2d<_, f32>, label: &str| {
            for &u in &nodes {
                assert_eq!(
                    d_a.x(u).unwrap().to_bits(),
                    d_b.x(u).unwrap().to_bits(),
                    "{label} x mismatch"
                );
                assert_eq!(
                    d_a.y(u).unwrap().to_bits(),
                    d_b.y(u).unwrap().to_bits(),
                    "{label} y mismatch"
                );
            }
        };

    let mut bh_ref = DrawingEuclidean2d::initial_placement(&graph);
    let mut bh_adap = DrawingEuclidean2d::initial_placement(&graph);
    let mut bh_try = DrawingEuclidean2d::initial_placement(&graph);
    let mut bh_csr = DrawingEuclidean2d::initial_placement(&graph);
    bh.run_with_knn(&mut bh_ref, &knn);
    bh.run_with_knn_adaptive(&mut bh_adap, &knn, &perplexities);
    bh.try_run_with_knn_adaptive(&mut bh_try, &knn, &perplexities)
        .unwrap();
    bh.try_run_with_csr_knn_adaptive(&mut bh_csr, &indptr, &indices, &distances, &perplexities)
        .unwrap();
    assert_same(&bh_adap, &bh_ref, "BhTsNet run_with_knn_adaptive");
    assert_same(&bh_try, &bh_ref, "BhTsNet try_run_with_knn_adaptive");
    assert_same(&bh_csr, &bh_ref, "BhTsNet try_run_with_csr_knn_adaptive");

    let mut fit_ref = DrawingEuclidean2d::initial_placement(&graph);
    let mut fit_adap = DrawingEuclidean2d::initial_placement(&graph);
    let mut fit_try = DrawingEuclidean2d::initial_placement(&graph);
    let mut fit_csr = DrawingEuclidean2d::initial_placement(&graph);
    fit.run_with_knn(&mut fit_ref, &knn);
    fit.run_with_knn_adaptive(&mut fit_adap, &knn, &perplexities);
    fit.try_run_with_knn_adaptive(&mut fit_try, &knn, &perplexities)
        .unwrap();
    fit.try_run_with_csr_knn_adaptive(&mut fit_csr, &indptr, &indices, &distances, &perplexities)
        .unwrap();
    assert_same(&fit_adap, &fit_ref, "FitTsNet run_with_knn_adaptive");
    assert_same(&fit_try, &fit_ref, "FitTsNet try_run_with_knn_adaptive");
    assert_same(&fit_csr, &fit_ref, "FitTsNet try_run_with_csr_knn_adaptive");

    let mut l_ref = DrawingEuclidean2d::initial_placement(&graph);
    let mut l_adap = DrawingEuclidean2d::initial_placement(&graph);
    let mut l_try = DrawingEuclidean2d::initial_placement(&graph);
    let mut l_csr = DrawingEuclidean2d::initial_placement(&graph);
    l.run_with_knn(&mut l_ref, &knn);
    l.run_with_knn_adaptive(&mut l_adap, &knn, &perplexities);
    l.try_run_with_knn_adaptive(&mut l_try, &knn, &perplexities)
        .unwrap();
    l.try_run_with_csr_knn_adaptive(&mut l_csr, &indptr, &indices, &distances, &perplexities)
        .unwrap();
    assert_same(&l_adap, &l_ref, "LTsNet run_with_knn_adaptive");
    assert_same(&l_try, &l_ref, "LTsNet try_run_with_knn_adaptive");
    assert_same(&l_csr, &l_ref, "LTsNet try_run_with_csr_knn_adaptive");
}

#[test]
fn test_run_with_knn_adaptive_variable_rows_and_heterogeneous_perplexities() {
    let mut graph: Graph<(), (), petgraph::Undirected> = Graph::new_undirected();
    let nodes: Vec<_> = (0..7).map(|_| graph.add_node(())).collect();
    // Variable row degrees: hub m=5, medium m=3, leaf m=1, isolated m=0
    let knn: Vec<Vec<(usize, f64)>> = vec![
        vec![(1, 0.2), (2, 0.4), (3, 0.8), (4, 1.2), (5, 2.0)],
        vec![(0, 0.2), (2, 0.5), (3, 1.0)],
        vec![(0, 0.4), (1, 0.5), (4, 1.1)],
        vec![(0, 0.8), (1, 1.0)],
        vec![(0, 1.2)],
        vec![(0, 2.0)],
        vec![],
    ];
    let perplexities = vec![3.0, 2.0, 1.5, 50.0, 0.5, 1.0, 2.0];
    let (indptr, indices, distances) = knn_to_csr(&knn);

    let bh = BhTsNetBuilder::new()
        .perplexity(2.0f64)
        .learning_rate(5.0)
        .iterations_stage1(10)
        .iterations_stage2(10)
        .iterations_stage3(10)
        .build()
        .unwrap();
    let fit = FitTsNetBuilder::new()
        .intervals(10)
        .interpolation_points(3)
        .perplexity(2.0f64)
        .learning_rate(5.0)
        .iterations_stage1(10)
        .iterations_stage2(10)
        .iterations_stage3(10)
        .build()
        .unwrap();
    let l = LTsNetBuilder::new()
        .intervals(10)
        .interpolation_points(3)
        .perplexity(2.0f64)
        .learning_rate(5.0)
        .iterations_stage1(10)
        .iterations_stage2(10)
        .iterations_stage3(10)
        .build()
        .unwrap();

    let mut d_bh = DrawingEuclidean2d::initial_placement(&graph);
    let mut d_fit = DrawingEuclidean2d::initial_placement(&graph);
    let mut d_l = DrawingEuclidean2d::initial_placement(&graph);
    bh.try_run_with_csr_knn_adaptive(&mut d_bh, &indptr, &indices, &distances, &perplexities)
        .unwrap();
    fit.try_run_with_csr_knn_adaptive(&mut d_fit, &indptr, &indices, &distances, &perplexities)
        .unwrap();
    l.try_run_with_csr_knn_adaptive(&mut d_l, &indptr, &indices, &distances, &perplexities)
        .unwrap();

    for d in [&d_bh, &d_fit, &d_l] {
        let mut mean_x = 0.0;
        let mut mean_y = 0.0;
        for &u in &nodes {
            let x = d.x(u).unwrap();
            let y = d.y(u).unwrap();
            assert!(x.is_finite() && x.abs() < 1e3);
            assert!(y.is_finite() && y.abs() < 1e3);
            mean_x += x;
            mean_y += y;
        }
        assert!((mean_x / nodes.len() as f64).abs() < 1e-6);
        assert!((mean_y / nodes.len() as f64).abs() < 1e-6);
    }
}

#[test]
fn test_try_run_with_knn_adaptive_validation_and_edge_cases() {
    let bh = BhTsNetBuilder::new().perplexity(2.0f64).build().unwrap();
    let fit = FitTsNetBuilder::new().perplexity(2.0f64).build().unwrap();
    let l = LTsNetBuilder::new().perplexity(2.0f64).build().unwrap();

    // n = 0 valid & invalid
    let g0: Graph<(), (), petgraph::Undirected> = Graph::new_undirected();
    let mut d0 = DrawingEuclidean2d::<petgraph::graph::NodeIndex, f64>::initial_placement(&g0);
    assert!(bh
        .try_run_with_csr_knn_adaptive(&mut d0, &[0], &[], &[], &[])
        .is_ok());
    assert!(fit
        .try_run_with_csr_knn_adaptive(&mut d0, &[0], &[], &[], &[])
        .is_ok());
    assert!(l
        .try_run_with_csr_knn_adaptive(&mut d0, &[0], &[], &[], &[])
        .is_ok());
    assert!(bh
        .try_run_with_csr_knn_adaptive(&mut d0, &[], &[], &[], &[])
        .is_err());

    // n = 1 valid & invalid perplexity
    let mut g1: Graph<(), (), petgraph::Undirected> = Graph::new_undirected();
    g1.add_node(());
    let mut d1 = DrawingEuclidean2d::<petgraph::graph::NodeIndex, f64>::initial_placement(&g1);
    assert!(bh
        .try_run_with_csr_knn_adaptive(&mut d1, &[0, 0], &[], &[], &[1.0])
        .is_ok());
    assert!(fit
        .try_run_with_csr_knn_adaptive(&mut d1, &[0, 0], &[], &[], &[0.0])
        .is_err());
    assert!(l
        .try_run_with_knn_adaptive(&mut d1, &[vec![]], &[f64::NAN])
        .is_err());

    // n = 3 invalid CSR / perplexity checks across all 3 variants
    let mut g3: Graph<(), (), petgraph::Undirected> = Graph::new_undirected();
    for _ in 0..3 {
        g3.add_node(());
    }
    let mut d3 = DrawingEuclidean2d::<petgraph::graph::NodeIndex, f64>::initial_placement(&g3);
    assert!(bh
        .try_run_with_csr_knn_adaptive(&mut d3, &[0, 1, 1], &[1], &[1.0], &[1.0, 1.0, 1.0])
        .is_err());
    assert!(fit
        .try_run_with_csr_knn_adaptive(&mut d3, &[0, 1, 1, 1], &[3], &[1.0], &[1.0, 1.0, 1.0])
        .is_err());
    assert!(l
        .try_run_with_csr_knn_adaptive(&mut d3, &[0, 1, 1, 1], &[1], &[-0.1], &[1.0, 1.0, 1.0])
        .is_err());
    assert!(bh
        .try_run_with_knn_adaptive(&mut d3, &[vec![], vec![]], &[1.0, 1.0, 1.0])
        .is_err());
}

#[test]
fn test_tsnet_try_run_with_knn_adaptive_rejects_bad_indices_and_distances() {
    let mut g: Graph<(), (), petgraph::Undirected> = Graph::new_undirected();
    for _ in 0..3 {
        g.add_node(());
    }
    let mut d = DrawingEuclidean2d::<petgraph::graph::NodeIndex, f64>::initial_placement(&g);
    let fit = FitTsNetBuilder::new().perplexity(2.0f64).build().unwrap();
    let perps = [1.5, 1.5, 1.5];
    // Out-of-bounds neighbor index
    assert!(fit
        .try_run_with_knn_adaptive(&mut d, &[vec![(3, 1.0)], vec![], vec![]], &perps)
        .is_err());
    // Negative / NaN distance
    assert!(fit
        .try_run_with_knn_adaptive(&mut d, &[vec![(1, -0.1)], vec![], vec![]], &perps)
        .is_err());
    assert!(fit
        .try_run_with_knn_adaptive(&mut d, &[vec![(1, f64::NAN)], vec![], vec![]], &perps)
        .is_err());
    // Non-finite drawing coordinate rejected cleanly
    {
        use petgraph_drawing::Drawing;
        d.raw_entry_mut(0).0 = f64::INFINITY;
    }
    let mut rng = StdRng::seed_from_u64(42);
    assert!(fit.try_run(&mut d, &g, &mut rng).is_err());
}
