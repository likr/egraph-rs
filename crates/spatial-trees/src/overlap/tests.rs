use super::*;
use egraph_core::{DeterministicRng, Layout3d, LayoutNd, NodeExtents2d, NodeShape2d};

fn brute_overlaps(rects: &[Rect2d<f64>]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..rects.len() {
        for j in (i + 1)..rects.len() {
            if rects[i].overlaps(&rects[j]) {
                out.push((i, j));
            }
        }
    }
    out
}

#[test]
fn test_rectangle_overlap_sweep_matches_brute_force_and_edge_cases() {
    // Disjoint horizontal strips sharing identical x intervals (checks O(N log N) tree).
    let mut strips = Vec::new();
    for k in 0..100 {
        strips.push(Rect2d::new(0.0, 10.0, k as f64, k as f64 + 1.0).unwrap());
    }
    assert!(!RectangleOverlapSweep::has_overlaps(&strips).unwrap());
    assert_eq!(
        RectangleOverlapSweep::find_overlaps(&strips).unwrap(),
        vec![]
    );

    // Random rectangles including touching boundaries and zero-width boxes.
    let mut rng = DeterministicRng::from_seed_u64(2026);
    let mut rects = Vec::new();
    for _ in 0..80 {
        let x0 = (rng.gen_range_usize(20) as f64) * 0.5;
        let y0 = (rng.gen_range_usize(20) as f64) * 0.5;
        let w = (rng.gen_range_usize(5) as f64) * 0.5;
        let h = (rng.gen_range_usize(5) as f64) * 0.5;
        rects.push(Rect2d::new(x0, x0 + w, y0, y0 + h).unwrap());
    }
    let expected = brute_overlaps(&rects);
    assert_eq!(
        RectangleOverlapSweep::find_overlaps(&rects).unwrap(),
        expected
    );
    assert_eq!(
        RectangleOverlapSweep::has_overlaps(&rects).unwrap(),
        !expected.is_empty()
    );
}

#[test]
fn test_rectangle_overlap_layout_helper_and_validation() {
    let layout = Layout2d::from_xy(vec![0.0_f32, 0.9, 5.0], vec![0.0, 0.0, 0.0]).unwrap();
    let hw = [0.5_f32, 0.5, 0.5];
    let hh = [0.5_f32, 0.5, 0.5];
    assert!(RectangleOverlapSweep::has_layout_overlaps(&layout, &hw, &hh).unwrap());
    assert_eq!(
        RectangleOverlapSweep::find_layout_overlaps(&layout, &hw, &hh).unwrap(),
        vec![(0, 1)]
    );

    assert!(matches!(
        Rect2d::new(2.0, 1.0, 0.0, 1.0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        Rect2d::from_center_size(0.0, 0.0, f64::NAN, 1.0),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // Rect2d::from_center_extents and with_* builders.
    let r = Rect2d::from_center_extents(1.0_f64, 2.0, 0.5, 1.0)
        .unwrap()
        .with_x_min(0.0)
        .unwrap()
        .with_x_max(3.0)
        .unwrap()
        .with_y_min(0.5)
        .unwrap()
        .with_y_max(4.0)
        .unwrap();
    assert_eq!(r, Rect2d::new(0.0, 3.0, 0.5, 4.0).unwrap());
    assert!(matches!(
        r.clone().with_x_min(4.0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        Rect2d::from_center_extents(0.0, 0.0, -0.1, 1.0),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_rectangle_overlap_node_extents_and_euclidean_layouts() {
    let l2 = Layout2d::from_xy(vec![0.0_f64, 0.9, 5.0], vec![0.0, 0.0, 0.0]).unwrap();
    let l_nd2 = LayoutNd::from_flat(3, 2, &[0.0_f64, 0.0, 0.9, 0.0, 5.0, 0.0]).unwrap();

    let hw = [0.5_f64, 0.5, 0.5];
    let hh = [0.5_f64, 0.5, 0.5];
    let split = NodeExtents2d::new(&hw, &hh).unwrap();
    let radii = NodeExtents2d::from_radii(&[0.5_f64, 0.5, 0.5]).unwrap();
    let uniform = NodeExtents2d::uniform(3, 0.5_f64, 0.5).unwrap();
    let shapes_vec = vec![
        NodeShape2d::rectangle(0.5_f64, 0.5).unwrap(),
        NodeShape2d::circle(0.5_f64).unwrap(),
        NodeShape2d::rectangle(0.5_f64, 0.5).unwrap(),
    ];
    let shapes = NodeExtents2d::from_shapes(&shapes_vec).unwrap();

    for ext in [&split, &radii, &uniform, &shapes] {
        assert_eq!(
            RectangleOverlapSweep::find_layout_extents_overlaps(&l2, ext).unwrap(),
            vec![(0, 1)]
        );
        assert!(RectangleOverlapSweep::has_layout_extents_overlaps(&l2, ext).unwrap());
        assert_eq!(
            RectangleOverlapSweep::find_layout_extents_overlaps(&l_nd2, ext).unwrap(),
            vec![(0, 1)]
        );
        assert!(RectangleOverlapSweep::has_layout_extents_overlaps(&l_nd2, ext).unwrap());
    }

    let disjoint_uni = NodeExtents2d::uniform(3, 0.4_f64, 0.4).unwrap();
    assert!(!RectangleOverlapSweep::has_layout_extents_overlaps(&l2, &disjoint_uni).unwrap());
    assert!(
        RectangleOverlapSweep::find_layout_extents_overlaps(&l_nd2, &disjoint_uni)
            .unwrap()
            .is_empty()
    );

    // Non-2D layout and mismatched extents count errors.
    let l3 = Layout3d::from_xyz(vec![0.0; 3], vec![0.0; 3], vec![0.0; 3]).unwrap();
    assert!(matches!(
        RectangleOverlapSweep::find_layout_extents_overlaps(&l3, &uniform),
        Err(CoreError::DimensionMismatch {
            expected: 2,
            actual: 3,
            ..
        })
    ));
    let l_nd3 = LayoutNd::zeros(3, 3).unwrap();
    assert!(matches!(
        RectangleOverlapSweep::has_layout_extents_overlaps(&l_nd3, &uniform),
        Err(CoreError::DimensionMismatch {
            expected: 2,
            actual: 3,
            ..
        })
    ));
    let wrong_len = NodeExtents2d::uniform(2, 0.5_f64, 0.5).unwrap();
    assert!(matches!(
        RectangleOverlapSweep::find_layout_extents_overlaps(&l2, &wrong_len),
        Err(CoreError::DimensionMismatch {
            expected: 3,
            actual: 2,
            ..
        })
    ));
}
