use super::*;
use egraph_core::DeterministicRng;

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
}
