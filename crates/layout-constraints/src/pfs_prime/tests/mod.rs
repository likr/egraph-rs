mod basic;
mod heap_ulp;
mod multiscale_fuzz;
mod offsets;

use egraph_core::{Layout2d, Scalar};
use egraph_spatial_trees::RectangleOverlapSweep;

pub(super) fn brute_f64_exact_overlaps<S: Scalar>(
    layout: &Layout2d<S>,
    hw: &[S],
    hh: &[S],
    padding: f64,
) -> Vec<(usize, usize)> {
    let n = layout.len();
    let pad_half = 0.5 * padding;
    let xs = layout.x();
    let ys = layout.y();
    let mut out = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let req_x = (hw[i].to_f64() + pad_half) + (hw[j].to_f64() + pad_half);
            let req_y = (hh[i].to_f64() + pad_half) + (hh[j].to_f64() + pad_half);
            if req_x > 0.0
                && req_y > 0.0
                && (xs[i].to_f64() - xs[j].to_f64()).abs() < req_x
                && (ys[i].to_f64() - ys[j].to_f64()).abs() < req_y
            {
                out.push((i, j));
            }
        }
    }
    out
}

pub(super) fn assert_no_overlaps_all<S: Scalar>(
    layout: &Layout2d<S>,
    hw: &[S],
    hh: &[S],
    padding: f64,
) {
    let pad_half = 0.5 * padding;
    let eff_hw_s: Vec<S> = hw
        .iter()
        .map(|&v| S::from_f64(v.to_f64() + pad_half))
        .collect();
    let eff_hh_s: Vec<S> = hh
        .iter()
        .map(|&v| S::from_f64(v.to_f64() + pad_half))
        .collect();
    let overlaps =
        RectangleOverlapSweep::find_layout_overlaps(layout, &eff_hw_s, &eff_hh_s).unwrap();
    assert!(
        overlaps.is_empty(),
        "expected 0 RectangleOverlapSweep overlaps, found {}: {:?}",
        overlaps.len(),
        &overlaps[..overlaps.len().min(5)]
    );
    let brute_f64 = brute_f64_exact_overlaps(layout, hw, hh, padding);
    assert!(
        brute_f64.is_empty(),
        "expected 0 f64-exact brute overlaps, found {}: {:?}",
        brute_f64.len(),
        &brute_f64[..brute_f64.len().min(5)]
    );
    // Also verify f64-exact against S-casted half-extents.
    let brute_s_f64 = brute_f64_exact_overlaps(layout, &eff_hw_s, &eff_hh_s, 0.0);
    assert!(
        brute_s_f64.is_empty(),
        "expected 0 S-extent f64-exact overlaps, found {}: {:?}",
        brute_s_f64.len(),
        &brute_s_f64[..brute_s_f64.len().min(5)]
    );
}

pub(super) fn assert_no_overlaps<S: Scalar>(layout: &Layout2d<S>, hw: &[S], hh: &[S]) {
    assert_no_overlaps_all(layout, hw, hh, 0.0);
}

pub(super) fn count_orthogonal_inversions(
    orig_x: &[f64],
    orig_y: &[f64],
    new_x: &[f64],
    new_y: &[f64],
) -> usize {
    let n = orig_x.len();
    let mut inversions = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if (orig_x[i] < orig_x[j] && new_x[i] > new_x[j])
                || (orig_x[i] > orig_x[j] && new_x[i] < new_x[j])
            {
                inversions += 1;
            }
            if (orig_y[i] < orig_y[j] && new_y[i] > new_y[j])
                || (orig_y[i] > orig_y[j] && new_y[i] < new_y[j])
            {
                inversions += 1;
            }
        }
    }
    inversions
}

pub(super) fn count_orthogonal_inversions_scalar<S: Scalar>(
    orig_x: &[S],
    orig_y: &[S],
    new_x: &[S],
    new_y: &[S],
) -> usize {
    let ox: Vec<f64> = orig_x.iter().map(|v| v.to_f64()).collect();
    let oy: Vec<f64> = orig_y.iter().map(|v| v.to_f64()).collect();
    let nx: Vec<f64> = new_x.iter().map(|v| v.to_f64()).collect();
    let ny: Vec<f64> = new_y.iter().map(|v| v.to_f64()).collect();
    count_orthogonal_inversions(&ox, &oy, &nx, &ny)
}
