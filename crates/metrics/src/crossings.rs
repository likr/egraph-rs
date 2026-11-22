//! Edge-crossing metrics built on `egraph-spatial`'s sweep-line and sampling primitives.

use egraph_core::{CoreResult, CsrGraph, DeterministicRng, Layout2d, Scalar};
use egraph_spatial::{
    SweepLineCrossings, sampled_crossing_number_edges, torus_crossing_edges_from_edges,
};

pub use egraph_spatial::{CrossingPair, CrossingSummary, TorusCrossing};

use crate::{extract_unique_edges, validate_graph_layout};

/// Counts the exact number of non-adjacent crossing edge pairs in `graph` under `layout`.
pub fn crossing_number<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<usize> {
    validate_graph_layout(graph, layout, "crossing_number")?;
    SweepLineCrossings::count(graph, layout)
}

/// Computes the legacy crossing-angle penalty $\sum \cos^2(\theta)$ over all crossing edge pairs.
pub fn crossing_angle<W, S: Scalar>(graph: &CsrGraph<W>, layout: &Layout2d<S>) -> CoreResult<f64> {
    Ok(crossing_summary(graph, layout)?.angle_cos_sq_sum)
}

/// Computes both exact crossing count and crossing-angle statistics in a single sweep pass.
pub fn crossing_summary<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<CrossingSummary> {
    validate_graph_layout(graph, layout, "crossing_summary")?;
    SweepLineCrossings::summarize(graph, layout)
}

/// Returns all non-adjacent crossing edge pairs in `graph` under `layout`.
pub fn crossing_edges<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<Vec<CrossingPair>> {
    validate_graph_layout(graph, layout, "crossing_edges")?;
    SweepLineCrossings::find_crossings(graph, layout)
}

/// Computes crossing count directly from precomputed [`CrossingPair`]s.
#[inline]
pub fn crossing_number_from_pairs(pairs: &[CrossingPair]) -> usize {
    pairs.len()
}

/// Computes the crossing-angle sum $\sum \cos^2(\theta)$ directly from precomputed [`CrossingPair`]s.
#[inline]
pub fn crossing_angle_from_pairs(pairs: &[CrossingPair]) -> f64 {
    pairs.iter().map(|p| p.cos_sq).sum()
}

/// Finds all crossing edge sub-segment pairs for `graph` laid out on the unit torus `[0, 1)^2`.
pub fn crossing_edges_torus<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<Vec<TorusCrossing>> {
    validate_graph_layout(graph, layout, "crossing_edges_torus")?;
    let edges = extract_unique_edges(graph);
    torus_crossing_edges_from_edges(&edges, layout)
}

/// Finds all crossing edge sub-segment pairs for an explicit edge slice on `[0, 1)^2`.
pub fn crossing_edges_torus_from_edges<S: Scalar>(
    edges: &[(u32, u32)],
    layout: &Layout2d<S>,
) -> CoreResult<Vec<TorusCrossing>> {
    torus_crossing_edges_from_edges(edges, layout)
}

/// Estimates total edge crossings in $O(\text{num\_samples})$ time using deterministic sampling.
pub fn sampled_crossing_number<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    sampled_crossing_number_with_rng(graph, layout, num_samples, &mut rng)
}

/// RNG-injecting variant of [`sampled_crossing_number`].
pub fn sampled_crossing_number_with_rng<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    num_samples: usize,
    rng: &mut DeterministicRng,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "sampled_crossing_number")?;
    let edges = extract_unique_edges(graph);
    sampled_crossing_number_edges(&edges, layout, num_samples, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_core::CoreError;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn test_crossings_exact_summary_torus_and_sampled() {
        // Right-angle crossing between (0, 1) and (2, 3).
        let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
        let layout =
            Layout2d::from_xy(vec![-1.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, -1.0, 1.0]).unwrap();
        assert_eq!(crossing_number(&g, &layout).unwrap(), 1);
        assert!(crossing_angle(&g, &layout).unwrap().abs() < 1e-12);
        let pairs = crossing_edges(&g, &layout).unwrap();
        assert_eq!(crossing_number_from_pairs(&pairs), 1);
        assert!(crossing_angle_from_pairs(&pairs).abs() < 1e-12);
        assert!((pairs[0].angle_rad.unwrap() - FRAC_PI_2).abs() < 1e-12);

        // K5 pentagram has 5 crossings.
        let mut k5_edges = Vec::new();
        for u in 0..5u32 {
            for v in (u + 1)..5u32 {
                k5_edges.push((u, v));
            }
        }
        let k5 = CsrGraph::<f64>::from_edges(5, &k5_edges, None, false, false).unwrap();
        let k5_x: Vec<f64> = (0..5)
            .map(|i| (i as f64 * std::f64::consts::TAU / 5.0).cos())
            .collect();
        let k5_y: Vec<f64> = (0..5)
            .map(|i| (i as f64 * std::f64::consts::TAU / 5.0).sin())
            .collect();
        let k5_layout = Layout2d::from_xy(k5_x, k5_y).unwrap();
        assert_eq!(crossing_number(&k5, &k5_layout).unwrap(), 5);
        assert_eq!(
            sampled_crossing_number(&k5, &k5_layout, 100, 42).unwrap(),
            5.0
        );

        // Torus wrap crossing.
        let torus_layout =
            Layout2d::from_xy(vec![0.9, 0.1, 0.9, 0.1], vec![0.1, 0.9, 0.9, 0.1]).unwrap();
        let torus_crossings = crossing_edges_torus(&g, &torus_layout).unwrap();
        assert!(!torus_crossings.is_empty());
    }

    #[test]
    fn test_crossings_validation_isolated_mismatch_and_nan() {
        // Node 4 is isolated, so SweepLineCrossings edge bounds alone would miss len mismatch.
        let g = CsrGraph::<f64>::from_edges(5, &[(0, 1), (2, 3)], None, false, false).unwrap();
        let short_layout = Layout2d::<f64>::zeros(4);
        assert!(matches!(
            crossing_number(&g, &short_layout),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            crossing_edges_torus(&g, &short_layout),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            sampled_crossing_number(&g, &short_layout, 10, 42),
            Err(CoreError::DimensionMismatch { .. })
        ));

        let bad_layout = Layout2d::from_xy(
            vec![0.0, 1.0, 0.0, 1.0, f64::NAN],
            vec![0.0, 0.0, 1.0, 1.0, 0.0],
        )
        .unwrap();
        assert!(matches!(
            crossing_angle(&g, &bad_layout),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }
}
