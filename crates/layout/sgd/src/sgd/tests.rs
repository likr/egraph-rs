use super::*;
use ndarray::arr2;
use petgraph::graph::UnGraph;
use petgraph_drawing::DrawingEuclidean2d;

#[test]
fn test_sgd_from_node_pairs_and_bulk_updates() {
    let pairs = vec![(0, 1, 1.0, 1.0, 1.0, 1.0), (1, 2, 2.0, 2.0, 0.25, 0.25)];
    let mut sgd = Sgd::from_node_pairs(pairs);
    assert_eq!(sgd.node_pairs().len(), 2);

    let w = arr2(&[[0.0, 2.0, 0.0], [3.0, 0.0, 4.0], [0.0, 5.0, 0.0]]);
    sgd.update_weight_from_array(w.view());
    assert_eq!(sgd.node_pairs()[0], (0, 1, 1.0, 1.0, 2.0, 3.0));
    assert_eq!(sgd.node_pairs()[1], (1, 2, 2.0, 2.0, 4.0, 5.0));

    let d = arr2(&[[0.0, 10.0, 0.0], [11.0, 0.0, 12.0], [0.0, 13.0, 0.0]]);
    sgd.update_distance_from_array(d.view(), None);
    assert_eq!(sgd.node_pairs()[0], (0, 1, 10.0, 11.0, 2.0, 3.0));
    assert_eq!(sgd.node_pairs()[1], (1, 2, 12.0, 13.0, 4.0, 5.0));

    let w2 = arr2(&[[0.0, 0.5, 0.0], [0.6, 0.0, 0.7], [0.0, 0.8, 0.0]]);
    sgd.update_distance_from_array(d.view(), Some(w2.view()));
    assert_eq!(sgd.node_pairs()[0], (0, 1, 10.0, 11.0, 0.5, 0.6));
    assert_eq!(sgd.node_pairs()[1], (1, 2, 12.0, 13.0, 0.7, 0.8));
}

#[test]
fn test_sgd_hinge_zero_force_when_separated_and_repels_when_too_close() {
    let mut g = UnGraph::<(), ()>::new_undirected();
    let u = g.add_node(());
    let v = g.add_node(());

    // Case 1: ||x_0 - x_1|| = 3.0 >= d_target = 2.0 -> exact zero displacement
    let mut drawing_far = DrawingEuclidean2d::<_, f64>::initial_placement(&g);
    drawing_far.raw_entry_mut(0).0 = 0.0;
    drawing_far.raw_entry_mut(0).1 = 0.0;
    drawing_far.raw_entry_mut(1).0 = 3.0;
    drawing_far.raw_entry_mut(1).1 = 0.0;
    let sgd = Sgd::new_with_hinge_pairs(vec![], vec![(0, 1, 2.0, 2.0, 1.0, 1.0)]);
    assert_eq!(sgd.total_pair_count(), 1);
    assert!(sgd.node_pairs().is_empty());
    assert_eq!(sgd.hinge_pairs().len(), 1);
    sgd.apply(&mut drawing_far, 0.5);
    assert_eq!(drawing_far.x(u).unwrap().to_bits(), 0.0_f64.to_bits());
    assert_eq!(drawing_far.y(u).unwrap().to_bits(), 0.0_f64.to_bits());
    assert_eq!(drawing_far.x(v).unwrap().to_bits(), 3.0_f64.to_bits());
    assert_eq!(drawing_far.y(v).unwrap().to_bits(), 0.0_f64.to_bits());

    // Case 2: Exact boundary ||x_0 - x_1|| == d_target -> exact zero displacement
    drawing_far.raw_entry_mut(1).0 = 2.0;
    sgd.apply(&mut drawing_far, 0.5);
    assert_eq!(drawing_far.x(u).unwrap().to_bits(), 0.0_f64.to_bits());
    assert_eq!(drawing_far.x(v).unwrap().to_bits(), 2.0_f64.to_bits());

    // Case 3: ||x_0 - x_1|| = 0.5 < d_target = 2.0 -> repels symmetrically outward
    let mut drawing_close = DrawingEuclidean2d::<_, f64>::initial_placement(&g);
    drawing_close.raw_entry_mut(0).0 = 0.0;
    drawing_close.raw_entry_mut(0).1 = 0.0;
    drawing_close.raw_entry_mut(1).0 = 0.5;
    drawing_close.raw_entry_mut(1).1 = 0.0;
    sgd.apply(&mut drawing_close, 0.5);
    assert!(drawing_close.x(u).unwrap() < 0.0);
    assert!(drawing_close.x(v).unwrap() > 0.5);
    assert_eq!(drawing_close.y(u).unwrap().to_bits(), 0.0_f64.to_bits());
    assert_eq!(drawing_close.y(v).unwrap().to_bits(), 0.0_f64.to_bits());

    // Case 4: Asymmetric hinge (wij = 0, wji = 1) moves only j
    let mut drawing_asym = DrawingEuclidean2d::<_, f64>::initial_placement(&g);
    drawing_asym.raw_entry_mut(0).0 = 0.0;
    drawing_asym.raw_entry_mut(0).1 = 0.0;
    drawing_asym.raw_entry_mut(1).0 = 0.5;
    drawing_asym.raw_entry_mut(1).1 = 0.0;
    let sgd_asym = Sgd::new_with_hinge_pairs(vec![], vec![(0, 1, 2.0, 2.0, 0.0, 1.0)]);
    sgd_asym.apply(&mut drawing_asym, 0.5);
    assert_eq!(drawing_asym.x(u).unwrap().to_bits(), 0.0_f64.to_bits());
    assert!(drawing_asym.x(v).unwrap() > 0.5);
}

#[test]
fn test_sgd_mixed_spring_and_hinge_behavior_and_eta_bounds() {
    let mut g = UnGraph::<(), ()>::new_undirected();
    let n0 = g.add_node(());
    let n1 = g.add_node(());
    let n2 = g.add_node(());

    let mut drawing = DrawingEuclidean2d::<_, f64>::initial_placement(&g);
    // (0, 1) is stretched spring (dist 4.0 > target 1.0 -> pulls inward)
    // (0, 2) is separated hinge (dist 5.0 > target 2.0 -> exact zero force on node 2)
    drawing.raw_entry_mut(0).0 = 0.0;
    drawing.raw_entry_mut(0).1 = 0.0;
    drawing.raw_entry_mut(1).0 = 4.0;
    drawing.raw_entry_mut(1).1 = 0.0;
    drawing.raw_entry_mut(2).0 = 0.0;
    drawing.raw_entry_mut(2).1 = 5.0;

    let mut sgd = Sgd::new(vec![(0, 1, 1.0, 1.0, 1.0, 1.0)]);
    sgd.add_hinge_pairs(vec![(0, 2, 2.0, 2.0, 0.25, 0.25)]);
    assert_eq!(sgd.total_pair_count(), 2);

    let (eta_min, eta_max) = sgd.calculate_eta_bounds(0.01_f64);
    assert!((eta_min - 0.01_f64).abs() < 1e-12);
    assert!((eta_max - 4.0_f64).abs() < 1e-12);

    let mut rng = StdRng::seed_from_u64(42);
    sgd.shuffle(&mut rng);
    sgd.apply(&mut drawing, 0.5);

    assert!(drawing.x(n0).unwrap() > 0.0);
    assert!(drawing.x(n1).unwrap() < 4.0);
    assert_eq!(drawing.x(n2).unwrap().to_bits(), 0.0_f64.to_bits());
    assert_eq!(drawing.y(n2).unwrap().to_bits(), 5.0_f64.to_bits());

    // Guard when no positive finite weights exist
    let empty_sgd = Sgd::<f64>::new(vec![]);
    assert_eq!(empty_sgd.calculate_eta_bounds(0.01), (0.0, 0.0));
    let zero_w = Sgd::new(vec![(0, 1, 1.0_f64, 1.0, 0.0, 0.0)]);
    assert_eq!(zero_w.calculate_eta_bounds(0.01), (0.0, 0.0));
}

#[test]
fn test_sgd_schedulers_empty_pairs_and_small_tmax_finite() {
    use crate::{
        Scheduler, SchedulerConstant, SchedulerExponential, SchedulerLinear, SchedulerQuadratic,
        SchedulerReciprocal,
    };

    let empty_sgd = Sgd::<f64>::new(vec![]);
    for t_max in [0usize, 1, 5] {
        let mut c: SchedulerConstant<f64> = empty_sgd.scheduler(t_max, 0.01);
        let mut l: SchedulerLinear<f64> = empty_sgd.scheduler(t_max, 0.01);
        let mut q: SchedulerQuadratic<f64> = empty_sgd.scheduler(t_max, 0.01);
        let mut e: SchedulerExponential<f64> = empty_sgd.scheduler(t_max, 0.01);
        let mut r: SchedulerReciprocal<f64> = empty_sgd.scheduler(t_max, 0.01);
        c.run(&mut |eta| assert!(eta.is_finite()));
        l.run(&mut |eta| assert!(eta.is_finite()));
        q.run(&mut |eta| assert!(eta.is_finite()));
        e.run(&mut |eta| assert!(eta.is_finite()));
        r.run(&mut |eta| assert!(eta.is_finite()));
    }

    // t_max=1 on non-empty Sgd with SchedulerLinear must not emit NaN
    let sgd = Sgd::new(vec![(0, 1, 1.0_f64, 1.0, 1.0, 1.0)]);
    let mut lin1: SchedulerLinear<f64> = sgd.scheduler(1, 0.01);
    let mut seen = Vec::new();
    lin1.run(&mut |eta| seen.push(eta));
    assert_eq!(seen.len(), 1);
    assert!(seen[0].is_finite());

    // apply(drawing, NaN) must be a no-op
    let mut g = UnGraph::<(), ()>::new_undirected();
    let u = g.add_node(());
    let v = g.add_node(());
    let mut d = DrawingEuclidean2d::<_, f64>::initial_placement(&g);
    d.raw_entry_mut(0).0 = 0.0;
    d.raw_entry_mut(1).0 = 3.0;
    sgd.apply(&mut d, f64::NAN);
    assert_eq!(d.x(u).unwrap(), 0.0);
    assert_eq!(d.x(v).unwrap(), 3.0);
}
