use super::*;
use ndarray::prelude::*;
use petgraph_algorithm_shortest_path::FullDistanceMatrix;
use petgraph_drawing::{Drawing, DrawingEuclidean2d};

#[test]
fn test_conjugate_gradient() {
    let a = arr2(&[[3., 1.], [1., 2.]]);
    let b = arr1(&[6., 7.]);
    let mut x = arr1(&[2., 1.]);
    let epsilon = 1e-4;
    conjugate_gradient(&a, &b, &mut x, epsilon);
    let x_exact = [1., 3.];
    let mut d = 0.;
    for i in 0..x.len() {
        let dx = x[i] - x_exact[i];
        d += dx * dx;
    }
    assert!(d < epsilon);
}

#[test]
fn test_stress_majorization() {
    use petgraph::Graph;

    let n = 10;
    let mut graph = Graph::new_undirected();
    let nodes = (0..n).map(|_| graph.add_node(())).collect::<Vec<_>>();
    for j in 1..n {
        for i in 0..j {
            graph.add_edge(nodes[i], nodes[j], ());
        }
    }
    let mut coordinates = DrawingEuclidean2d::initial_placement(&graph);

    for &u in &nodes {
        println!("{:?}", coordinates.position(u));
    }

    let mut stress_majorization = StressMajorization::new(&graph, &coordinates, &mut |_| 1.);
    stress_majorization.run(&mut coordinates);

    for &u in &nodes {
        println!("{:?}", coordinates.position(u));
    }
}

#[test]
fn test_stress_majorization_parameters() {
    use petgraph::Graph;

    // Create a simple graph
    let mut graph = Graph::new_undirected();
    let n1 = graph.add_node(());
    let n2 = graph.add_node(());
    let n3 = graph.add_node(());
    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());

    let coordinates = DrawingEuclidean2d::initial_placement(&graph);

    // Default parameters
    let mut stress_majorization = StressMajorization::new(&graph, &coordinates, &mut |_| 1.0);

    // Check default values
    assert_eq!(stress_majorization.epsilon, 1e-4);
    assert_eq!(stress_majorization.max_iterations, 100);

    // Update parameters
    stress_majorization.epsilon = 1e-6;
    stress_majorization.max_iterations = 200;

    // Check updated values
    assert_eq!(stress_majorization.epsilon, 1e-6);
    assert_eq!(stress_majorization.max_iterations, 200);
}

#[test]
fn test_stress_majorization_update_weight_from_array() {
    use petgraph::Graph;

    let mut graph = Graph::new_undirected();
    let n0 = graph.add_node(());
    let n1 = graph.add_node(());
    let n2 = graph.add_node(());
    let n3 = graph.add_node(());
    graph.add_edge(n0, n1, ());
    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());
    graph.add_edge(n3, n0, ());

    let mut coords_fn = DrawingEuclidean2d::initial_placement(&graph);
    let mut coords_arr = coords_fn.clone();

    let mut sm_fn = StressMajorization::new(&graph, &coords_fn, |_| 1.0f64);
    let mut sm_arr = StressMajorization::new(&graph, &coords_arr, |_| 1.0f64);

    let weights = arr2(&[
        [0.0, 2.0, 0.5, 1.5],
        [2.0, 0.0, 3.0, 0.25],
        [0.5, 3.0, 0.0, 4.0],
        [1.5, 0.25, 4.0, 0.0],
    ]);

    sm_fn.update_weight(|i, j, _, _| weights[[i, j]]);
    sm_arr.update_weight_from_array(weights.view());

    assert_eq!(sm_fn.w, sm_arr.w);
    assert_eq!(sm_fn.l_w, sm_arr.l_w);
    assert!((sm_fn.stress - sm_arr.stress).abs() < 1e-12);

    let diff_fn = sm_fn.apply(&mut coords_fn);
    let diff_arr = sm_arr.apply(&mut coords_arr);
    assert!((diff_fn - diff_arr).abs() < 1e-12);

    for u in [n0, n1, n2, n3] {
        assert!((coords_fn.x(u).unwrap() - coords_arr.x(u).unwrap()).abs() < 1e-12);
        assert!((coords_fn.y(u).unwrap() - coords_arr.y(u).unwrap()).abs() < 1e-12);
    }
}

#[test]
fn test_stress_majorization_empty_and_singleton_graphs() {
    use petgraph::graph::UnGraph;

    // n == 0: no 0usize - 1 underflow, n() == 0, apply/run/update_weight are clean no-ops.
    let g0: UnGraph<(), ()> = UnGraph::new_undirected();
    let mut d0: DrawingEuclidean2d<_, f64> = DrawingEuclidean2d::initial_placement(&g0);
    let dm0 = FullDistanceMatrix::from_array(&g0, Array2::zeros((0, 0)));
    let mut sm0 = StressMajorization::try_new(&g0, &d0, |_| 1.0).unwrap();
    assert_eq!(sm0.n(), 0);
    assert_eq!(sm0.try_apply(&mut d0).unwrap(), 0.0);
    sm0.try_run(&mut d0).unwrap();
    sm0.try_update_weight::<&str, _>(|_, _, _, w| Ok(w))
        .unwrap();
    let mut sm0_dm = StressMajorization::try_new_with_distance_matrix(&d0, &dm0).unwrap();
    assert_eq!(sm0_dm.n(), 0);
    sm0_dm.try_run(&mut d0).unwrap();

    // n == 1: singleton graph works cleanly with n() == 1.
    let mut g1: UnGraph<(), ()> = UnGraph::new_undirected();
    let u0 = g1.add_node(());
    let mut d1: DrawingEuclidean2d<_, f64> = DrawingEuclidean2d::initial_placement(&g1);
    let orig_pos = (d1.x(u0).unwrap(), d1.y(u0).unwrap());
    let dm1 = FullDistanceMatrix::from_array(&g1, Array2::zeros((1, 1)));
    let mut sm1 = StressMajorization::try_new(&g1, &d1, |_| 1.0).unwrap();
    assert_eq!(sm1.n(), 1);
    assert_eq!(sm1.try_apply(&mut d1).unwrap(), 0.0);
    sm1.try_run(&mut d1).unwrap();
    assert_eq!((d1.x(u0).unwrap(), d1.y(u0).unwrap()), orig_pos);
    let mut sm1_dm = StressMajorization::try_new_with_distance_matrix(&d1, &dm1).unwrap();
    assert_eq!(sm1_dm.n(), 1);
    sm1_dm.try_run(&mut d1).unwrap();
}

#[test]
fn test_stress_majorization_fallible_validation_and_atomic_update() {
    use petgraph::graph::UnGraph;

    let mut g2: UnGraph<(), ()> = UnGraph::new_undirected();
    let a = g2.add_node(());
    let b = g2.add_node(());
    g2.add_edge(a, b, ());
    let mut d2: DrawingEuclidean2d<_, f64> = DrawingEuclidean2d::initial_placement(&g2);
    let dm2 = FullDistanceMatrix::from_array(&g2, arr2(&[[0.0, 1.0], [1.0, 0.0]]));

    let mut g3: UnGraph<(), ()> = UnGraph::new_undirected();
    let n0 = g3.add_node(());
    let n1 = g3.add_node(());
    let n2 = g3.add_node(());
    g3.add_edge(n0, n1, ());
    g3.add_edge(n1, n2, ());
    let mut d3: DrawingEuclidean2d<_, f64> = DrawingEuclidean2d::initial_placement(&g3);

    // Mismatched graph/drawing/matrix sizes return Err without panicking.
    assert!(StressMajorization::try_new(&g3, &d2, |_| 1.0).is_err());
    assert!(StressMajorization::try_new_with_distance_matrix(&d3, &dm2).is_err());

    let mut sm3 = StressMajorization::try_new(&g3, &d3, |_| 1.0).unwrap();
    assert!(sm3.try_apply(&mut d2).is_err());
    assert!(sm3.try_run(&mut d2).is_err());

    // Non-finite drawing coordinates are rejected.
    let mut d3_nan = d3.clone();
    d3_nan.raw_entry_mut(0).0 = f64::NAN;
    assert!(StressMajorization::try_new(&g3, &d3_nan, |_| 1.0).is_err());
    assert!(sm3.try_apply(&mut d3_nan).is_err());
    assert!(sm3.try_run(&mut d3_nan).is_err());

    // try_update_weight leaves weights unchanged when callback errors partway through.
    let before_w = sm3.w.clone();
    let mut calls = 0usize;
    let err = sm3.try_update_weight(|_, _, _, _| {
        calls += 1;
        if calls == 2 {
            Err("boom")
        } else {
            Ok(42.0)
        }
    });
    assert_eq!(err, Err("boom"));
    assert_eq!(sm3.w, before_w);

    // try_update_weight_from_array rejects bad shape, NaN, and negative weights atomically.
    assert!(sm3
        .try_update_weight_from_array(Array2::zeros((2, 2)).view())
        .is_err());
    assert!(sm3
        .try_update_weight_from_array(
            arr2(&[[0.0, -1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]).view()
        )
        .is_err());
    assert!(sm3
        .try_update_weight_from_array(
            arr2(&[[0.0, f64::NAN, 1.0], [f64::NAN, 0.0, 1.0], [1.0, 1.0, 0.0]]).view()
        )
        .is_err());
    assert_eq!(sm3.w, before_w);

    // Disconnected graph remains finite across apply/run (no 0.0 * inf = NaN).
    let mut g_disc: UnGraph<(), ()> = UnGraph::new_undirected();
    let c0 = g_disc.add_node(());
    let c1 = g_disc.add_node(());
    let c2 = g_disc.add_node(());
    let c3 = g_disc.add_node(());
    g_disc.add_edge(c0, c1, ());
    g_disc.add_edge(c2, c3, ());
    let mut d_disc: DrawingEuclidean2d<_, f64> = DrawingEuclidean2d::initial_placement(&g_disc);
    let mut sm_disc = StressMajorization::try_new(&g_disc, &d_disc, |_| 1.0).unwrap();
    let diff = sm_disc.try_apply(&mut d_disc).unwrap();
    assert!(diff.is_finite());
    sm_disc.try_run(&mut d_disc).unwrap();
    for u in [c0, c1, c2, c3] {
        assert!(d_disc.x(u).unwrap().is_finite());
        assert!(d_disc.y(u).unwrap().is_finite());
    }
    let _ = &mut d3;
}
