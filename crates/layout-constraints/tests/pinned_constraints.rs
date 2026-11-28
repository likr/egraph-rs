use egraph_core::{
    CoreError, DeterministicRng, IterativeSolver, Layout2d, NodeExtents2d, NodeShape2d, PinMask,
};
use egraph_layout_constraints::{
    ConstraintProjector, PrismConfig, PrismOverlapRemoval, SeparationConstraint,
    project_clustered_rectangle_no_overlap_2d, project_rectangle_no_overlap_2d,
};
use egraph_spatial_trees::RectangleOverlapSweep;

#[test]
fn test_separation_constraint_builders_and_validation() {
    let sc = SeparationConstraint::new(1, 2, 3.5_f64)
        .unwrap()
        .with_left(0)
        .with_right(4)
        .with_gap(2.25)
        .unwrap();
    assert_eq!(sc.left(), 0);
    assert_eq!(sc.right(), 4);
    assert_eq!(sc.gap(), 2.25);

    assert!(matches!(
        SeparationConstraint::new(0, 1, f64::NAN),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        SeparationConstraint::new(0, 1, f64::INFINITY),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        sc.with_gap(f64::NEG_INFINITY),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_project_1d_with_pins_fixed_nodes_and_qp_optimality() {
    // 1. Pinned node stays exact while unpinned neighbors separate around it.
    let mut proj = ConstraintProjector::<f64>::new(3);
    let mut coords = [5.0_f64, 4.0, 2.0];
    let pins = PinMask::from_indices(3, &[0]).unwrap();
    let c = [
        SeparationConstraint::new(0, 1, 2.0).unwrap(),
        SeparationConstraint::new(1, 2, 1.5).unwrap(),
    ];
    proj.project_1d_with_pins(&mut coords, &c, &pins).unwrap();
    assert_eq!(coords[0], 5.0);
    assert!((coords[1] - 7.0).abs() < 1e-9);
    assert!((coords[2] - 8.5).abs() < 1e-9);

    // 2. Two pinned blocks bridging via an unpinned pair (exact QP optimum: [-5, 15]).
    // Node 0 pinned at 0, Node 3 pinned at 10; Node 1 desired=10 (x1 <= -2), Node 2 desired=0 (x2 >= 12),
    // bridging constraint x1 + 20 <= x2 => optimal x1 = -5, x2 = 15.
    let mut proj4 = ConstraintProjector::<f64>::new(4);
    let mut x4 = [0.0_f64, 10.0, 0.0, 10.0];
    let pins4 = PinMask::from_indices(4, &[0, 3]).unwrap();
    let c4 = [
        SeparationConstraint::new(1, 0, 2.0).unwrap(),
        SeparationConstraint::new(3, 2, 2.0).unwrap(),
        SeparationConstraint::new(1, 2, 20.0).unwrap(),
    ];
    proj4.project_1d_with_pins(&mut x4, &c4, &pins4).unwrap();
    assert_eq!(x4[0], 0.0);
    assert_eq!(x4[3], 10.0);
    assert!((x4[1] - (-5.0)).abs() < 1e-8, "x4[1]={}", x4[1]);
    assert!((x4[2] - 15.0).abs() < 1e-8, "x4[2]={}", x4[2]);

    // 3. Same with an additional lower bound x1 >= -4 forcing x1 = -4, x2 = 16.
    let mut x4_lb = [0.0_f64, 10.0, 0.0, 10.0];
    let c4_lb = [
        SeparationConstraint::new(1, 0, 2.0).unwrap(),
        SeparationConstraint::new(0, 1, -4.0).unwrap(),
        SeparationConstraint::new(3, 2, 2.0).unwrap(),
        SeparationConstraint::new(1, 2, 20.0).unwrap(),
    ];
    proj4
        .project_1d_with_pins(&mut x4_lb, &c4_lb, &pins4)
        .unwrap();
    assert_eq!(x4_lb[0], 0.0);
    assert_eq!(x4_lb[3], 10.0);
    assert!((x4_lb[1] - (-4.0)).abs() < 1e-8, "x4_lb[1]={}", x4_lb[1]);
    assert!((x4_lb[2] - 16.0).abs() < 1e-8, "x4_lb[2]={}", x4_lb[2]);

    // 4. Layout axis helper with pins in f32.
    let mut layout32 = Layout2d::from_xy(vec![1.0_f32, 0.0, 3.0], vec![0.0_f32; 3]).unwrap();
    let mut proj32 = ConstraintProjector::<f32>::new(3);
    let pins32 = PinMask::from_bools(&[true, false, false]);
    proj32
        .project_layout_axis_with_pins(
            &mut layout32,
            0,
            &[SeparationConstraint::new(1, 0, 2.0_f32).unwrap()],
            &pins32,
        )
        .unwrap();
    assert_eq!(layout32.x()[0], 1.0_f32);
    assert!(layout32.x()[1] <= -1.0_f32 + 1e-6);
    assert_eq!(layout32.x()[2], 3.0_f32);
}

#[test]
fn test_project_1d_with_pins_rejects_conflicts_and_mismatches_without_mutation() {
    let mut proj = ConstraintProjector::<f64>::new(3);

    // 1. Direct pinned-pinned conflict leaves coords untouched.
    let mut coords = [5.0_f64, 0.0, 6.0];
    let orig = coords;
    let pins = PinMask::from_indices(3, &[0, 2]).unwrap();
    let direct_conflict = [SeparationConstraint::new(0, 2, 2.0).unwrap()];
    assert!(matches!(
        proj.project_1d_with_pins(&mut coords, &direct_conflict, &pins),
        Err(CoreError::InvalidInput(_))
    ));
    assert_eq!(coords, orig);

    // 2. Indirect pinned-pinned chain conflict (`0 -> 1 -> 2` requires gap 4 > 1) leaves coords untouched.
    let chain_conflict = [
        SeparationConstraint::new(0, 1, 2.0).unwrap(),
        SeparationConstraint::new(1, 2, 2.0).unwrap(),
    ];
    assert!(matches!(
        proj.project_1d_with_pins(&mut coords, &chain_conflict, &pins),
        Err(CoreError::InvalidInput(_))
    ));
    assert_eq!(coords, orig);

    // 3. Mismatched pin mask length rejected.
    let bad_pins = PinMask::unpinned(2);
    assert!(matches!(
        proj.project_1d_with_pins(&mut coords, &[], &bad_pins),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // 4. Mismatched coords length rejected.
    let mut short_coords = [0.0_f64; 2];
    assert!(matches!(
        proj.project_1d_with_pins(&mut short_coords, &[], &pins),
        Err(CoreError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_project_1d_with_pins_randomized_feasibility() {
    let mut rng = DeterministicRng::from_seed_u64(0x9191);
    let n = 16usize;
    let mut proj = ConstraintProjector::<f64>::new(n);

    for trial in 0..20 {
        // Construct a feasible ground truth and pin a subset at their feasible positions.
        let truth: Vec<f64> = (0..n).map(|i| (i as f64) * 2.0).collect();
        let mut constraints = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.next_f64() < 0.35 {
                    let max_gap = truth[j] - truth[i];
                    let gap = (max_gap * rng.next_f64()).max(0.1);
                    constraints.push(SeparationConstraint::new(i as u32, j as u32, gap).unwrap());
                }
            }
        }
        let mut pin_bools = vec![false; n];
        for p in &mut pin_bools {
            *p = rng.next_f64() < 0.3;
        }
        let pins = PinMask::from_bools(&pin_bools);

        let mut coords: Vec<f64> = (0..n)
            .map(|i| {
                if pins.is_pinned(i) {
                    truth[i]
                } else {
                    (rng.next_f64() - 0.5) * 20.0
                }
            })
            .collect();
        let orig = coords.clone();

        proj.project_1d_with_pins(&mut coords, &constraints, &pins)
            .unwrap_or_else(|e| panic!("trial {trial} failed unexpectedly: {e}"));

        for i in 0..n {
            if pins.is_pinned(i) {
                assert_eq!(coords[i], orig[i], "pinned node {i} moved in trial {trial}");
            }
        }
        for c in &constraints {
            assert!(
                coords[c.left() as usize] + c.gap() <= coords[c.right() as usize] + 1e-7,
                "constraint {:?} violated in trial {trial}",
                c
            );
        }
    }
}

#[test]
fn test_node_extents_uniform_and_shapes_variants() {
    let mut l_uni = Layout2d::from_xy(vec![0.0_f64, 0.2, 0.4], vec![0.0, 0.1, -0.1]).unwrap();
    let uni = NodeExtents2d::uniform(3, 0.5_f64, 0.4).unwrap();
    project_rectangle_no_overlap_2d(&mut l_uni, uni).unwrap();
    assert!(!RectangleOverlapSweep::has_layout_extents_overlaps(&l_uni, &uni).unwrap());

    let shapes = [
        NodeShape2d::rectangle(0.6_f64, 0.3).unwrap(),
        NodeShape2d::circle(0.5_f64).unwrap(),
        NodeShape2d::rectangle(0.4_f64, 0.7).unwrap(),
        NodeShape2d::circle(0.45_f64).unwrap(),
    ];
    let shape_ext = NodeExtents2d::from_shapes(&shapes).unwrap();
    let mut l_shapes =
        Layout2d::from_xy(vec![0.0, 0.3, 1.0, 1.2], vec![0.0, 0.2, 0.0, 0.1]).unwrap();
    let mut prism = PrismOverlapRemoval::new(shape_ext, PrismConfig::default()).unwrap();
    prism.run(&mut l_shapes).unwrap();
    assert!(!RectangleOverlapSweep::has_layout_extents_overlaps(&l_shapes, &shape_ext).unwrap());

    let mut l_clust =
        Layout2d::from_xy(vec![0.0, 0.4, 0.3, 0.7], vec![0.0, 0.0, 0.1, 0.1]).unwrap();
    project_clustered_rectangle_no_overlap_2d(&mut l_clust, &[0, 0, 1, 1], shape_ext).unwrap();
}
