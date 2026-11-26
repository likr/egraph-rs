use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};

use super::overlap_2d::project_rectangle_no_overlap_2d;

/// Performs a one-shot post-processing projection that removes overlaps between axis-aligned
/// cluster bounding boxes via VPSC while rigidly translating each cluster's nodes.
///
/// Note: This rigid AABB VPSC post-processing pass is distinct from Dwyer et al. (2006)
/// iterative constrained stress minimization (`IPSep-CoLa`).
pub fn project_clustered_rectangle_no_overlap_2d<S: Scalar>(
    layout: &mut Layout2d<S>,
    cluster_ids: &[u32],
    half_widths: &[S],
    half_heights: &[S],
) -> CoreResult<()> {
    let n = layout.len();
    if cluster_ids.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            cluster_ids.len(),
            "project_clustered_rectangle_no_overlap_2d::cluster_ids",
        ));
    }
    if half_widths.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            half_widths.len(),
            "project_clustered_rectangle_no_overlap_2d::half_widths",
        ));
    }
    if half_heights.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            half_heights.len(),
            "project_clustered_rectangle_no_overlap_2d::half_heights",
        ));
    }
    layout.validate_finite()?;
    for i in 0..n {
        if !half_widths[i].is_finite() || !half_heights[i].is_finite() {
            return Err(CoreError::non_finite(
                "project_clustered_rectangle_no_overlap_2d::extents",
            ));
        }
        if half_widths[i] < S::ZERO || half_heights[i] < S::ZERO {
            return Err(CoreError::invalid_input(
                "half_widths and half_heights must be >= 0",
            ));
        }
    }

    if n < 2 {
        return Ok(());
    }

    let mut members: Vec<(u32, u32)> = cluster_ids
        .iter()
        .enumerate()
        .map(|(i, &cid)| (cid, i as u32))
        .collect();
    members.sort_unstable();

    let mut cluster_cx = Vec::new();
    let mut cluster_cy = Vec::new();
    let mut cluster_hw = Vec::new();
    let mut cluster_hh = Vec::new();
    let mut cluster_ranges = Vec::new();

    {
        let xs = layout.x();
        let ys = layout.y();
        let mut start = 0usize;
        while start < n {
            let cid = members[start].0;
            let mut end = start + 1;
            while end < n && members[end].0 == cid {
                end += 1;
            }
            let mut min_x = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            for &(_, node_u32) in &members[start..end] {
                let idx = node_u32 as usize;
                let x = xs[idx].to_f64();
                let y = ys[idx].to_f64();
                let hw = half_widths[idx].to_f64();
                let hh = half_heights[idx].to_f64();
                min_x = min_x.min(x - hw);
                max_x = max_x.max(x + hw);
                min_y = min_y.min(y - hh);
                max_y = max_y.max(y + hh);
            }
            cluster_cx.push(S::from_f64(0.5 * (min_x + max_x)));
            cluster_cy.push(S::from_f64(0.5 * (min_y + max_y)));
            cluster_hw.push(S::from_f64(0.5 * (max_x - min_x)));
            cluster_hh.push(S::from_f64(0.5 * (max_y - min_y)));
            cluster_ranges.push((start, end));
            start = end;
        }
    }

    if cluster_ranges.len() < 2 {
        return Ok(());
    }

    let old_cx = cluster_cx.clone();
    let old_cy = cluster_cy.clone();
    let mut cluster_layout = Layout2d::from_xy(cluster_cx, cluster_cy)?;
    project_rectangle_no_overlap_2d(&mut cluster_layout, &cluster_hw, &cluster_hh)?;

    let (xs_mut, ys_mut) = layout.xy_mut();
    let new_cx = cluster_layout.x();
    let new_cy = cluster_layout.y();
    for (c, &(start, end)) in cluster_ranges.iter().enumerate() {
        let dx = new_cx[c].to_f64() - old_cx[c].to_f64();
        let dy = new_cy[c].to_f64() - old_cy[c].to_f64();
        for &(_, node_u32) in &members[start..end] {
            let idx = node_u32 as usize;
            xs_mut[idx] = S::from_f64(xs_mut[idx].to_f64() + dx);
            ys_mut[idx] = S::from_f64(ys_mut[idx].to_f64() + dy);
        }
    }

    Ok(())
}
