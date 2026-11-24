//! Edge-crossing metrics built on `egraph-spatial`'s sweep-line and sampling primitives.

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, Scalar};
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

pub(crate) fn max_crossing_pairs_from_edges(num_nodes: usize, edges: &[(u32, u32)]) -> u128 {
    let mut inc = vec![0_u128; num_nodes];
    let mut keys = Vec::with_capacity(edges.len());
    let mut m = 0_u128;
    for &(u, v) in edges {
        if u == v {
            continue;
        }
        let ui = u as usize;
        let vi = v as usize;
        if ui >= num_nodes || vi >= num_nodes {
            continue;
        }
        m += 1;
        inc[ui] += 1;
        inc[vi] += 1;
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        keys.push(((a as u64) << 32) | (b as u64));
    }
    if m < 2 {
        return 0;
    }
    let total_pairs = (m * (m - 1)) / 2;
    let adj_pairs: u128 = inc
        .into_iter()
        .map(|d| if d >= 2 { (d * (d - 1)) / 2 } else { 0 })
        .sum();
    keys.sort_unstable();
    let mut shared_both = 0_u128;
    let mut run = 1_u128;
    for i in 1..keys.len() {
        if keys[i] == keys[i - 1] {
            run += 1;
        } else {
            if run >= 2 {
                shared_both += (run * (run - 1)) / 2;
            }
            run = 1;
        }
    }
    if run >= 2 {
        shared_both += (run * (run - 1)) / 2;
    }
    (total_pairs + shared_both).saturating_sub(adj_pairs)
}

/// Computes the exact maximum number of non-adjacent edge pairs $c_{\max}$ in `graph`
/// using `u128` combinatorics (safe on 32-bit `wasm32` and accounting for reciprocal/parallel
/// edges while ignoring self-loops).
pub fn max_crossing_pairs<W>(graph: &CsrGraph<W>) -> u128 {
    let edges = extract_unique_edges(graph);
    max_crossing_pairs_from_edges(graph.num_nodes(), &edges)
}

#[inline]
pub(crate) fn crosslessness_from_cmax(crossings: f64, c_max: u128) -> f64 {
    if c_max == 0 {
        return 1.0;
    }
    let ratio = (crossings / (c_max as f64)).clamp(0.0, 1.0);
    (1.0 - ratio.sqrt()).clamp(0.0, 1.0)
}

/// Computes DRGraph Eq. (17) crosslessness $1 - \sqrt{c / c_{\max}} \in [0.0, 1.0]$ from a
/// precomputed crossing count `crossings` without re-running sweep-line detection.
#[inline]
pub fn crosslessness_from_count<W>(graph: &CsrGraph<W>, crossings: f64) -> CoreResult<f64> {
    if !crossings.is_finite() {
        return Err(CoreError::non_finite("crosslessness_from_count::crossings"));
    }
    if crossings < 0.0 {
        return Err(CoreError::invalid_input("crossings must be >= 0.0"));
    }
    let c_max = max_crossing_pairs(graph);
    Ok(crosslessness_from_cmax(crossings, c_max))
}

/// Computes exact DRGraph Eq. (17) crosslessness $1 - \sqrt{c / c_{\max}} \in [0.0, 1.0]$.
///
/// Returns `1.0` when $c_{\max} = 0$ (e.g., graphs with $< 2$ edges, stars, or triangles).
pub fn crosslessness<W, S: Scalar>(graph: &CsrGraph<W>, layout: &Layout2d<S>) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "crosslessness")?;
    let c_max = max_crossing_pairs(graph);
    if c_max == 0 {
        return Ok(1.0);
    }
    let count = SweepLineCrossings::count(graph, layout)?;
    Ok(crosslessness_from_cmax(count as f64, c_max))
}

/// Estimates DRGraph Eq. (17) crosslessness $1 - \sqrt{\hat{c} / c_{\max}} \in [0.0, 1.0]$
/// using deterministic edge-pair sampling (`sampled_crossing_number`).
pub fn sampled_crosslessness<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "sampled_crosslessness")?;
    if num_samples == 0 {
        return Err(CoreError::invalid_input("num_samples must be >= 1"));
    }
    let edges = extract_unique_edges(graph);
    let c_max = max_crossing_pairs_from_edges(graph.num_nodes(), &edges);
    if c_max == 0 {
        return Ok(1.0);
    }
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let est = sampled_crossing_number_edges(&edges, layout, num_samples, &mut rng)?;
    Ok(crosslessness_from_cmax(est, c_max))
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
        assert!(matches!(
            crosslessness(&g, &bad_layout),
            Err(CoreError::NonFiniteValue { .. })
        ));
        let ok_layout = Layout2d::<f64>::zeros(5);
        assert!(matches!(
            sampled_crosslessness(&g, &ok_layout, 0, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            crosslessness_from_count(&g, -1.0),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            crosslessness_from_count(&g, f64::NAN),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }

    fn brute_cmax_and_crossings<W, S: Scalar>(
        graph: &CsrGraph<W>,
        layout: &Layout2d<S>,
    ) -> (u128, usize) {
        let edges = extract_unique_edges(graph);
        let m = edges.len();
        let mut c_max = 0_u128;
        for i in 0..m {
            let (u1, v1) = edges[i];
            if u1 == v1 {
                continue;
            }
            for &(u2, v2) in &edges[(i + 1)..] {
                if u2 == v2 {
                    continue;
                }
                if u1 != u2 && u1 != v2 && v1 != u2 && v1 != v2 {
                    c_max += 1;
                }
            }
        }
        let exact_crossings = crossing_number(graph, layout).unwrap();
        (c_max, exact_crossings)
    }

    #[test]
    fn test_crosslessness_brute_force_oracle_and_degeneracies() {
        // 1. Star K_{1,5} and triangle K3 have c_max == 0 -> crosslessness == 1.0.
        let star = CsrGraph::<f64>::from_edges(
            6,
            &[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
            None,
            false,
            false,
        )
        .unwrap();
        let star_layout = Layout2d::from_xy(
            vec![0.0, 1.0, 0.0, -1.0, 0.5, -0.5],
            vec![0.0, 0.0, 1.0, 0.0, -1.0, -1.0],
        )
        .unwrap();
        assert_eq!(max_crossing_pairs(&star), 0);
        assert_eq!(crosslessness(&star, &star_layout).unwrap(), 1.0);
        assert_eq!(
            sampled_crosslessness(&star, &star_layout, 16, 42).unwrap(),
            1.0
        );

        let k3 =
            CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2), (2, 0)], None, false, false).unwrap();
        let k3_layout = Layout2d::from_xy(vec![0.0, 1.0, 0.5], vec![0.0, 0.0, 1.0]).unwrap();
        assert_eq!(max_crossing_pairs(&k3), 0);
        assert_eq!(crosslessness(&k3, &k3_layout).unwrap(), 1.0);

        // 2. K4 square with crossing diagonals: c = 1, c_max = C(6,2) - 4*C(3,2) = 15 - 12 = 3.
        let k4 = CsrGraph::<f64>::from_edges(
            4,
            &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)],
            None,
            false,
            false,
        )
        .unwrap();
        let sq = Layout2d::from_xy(vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        assert_eq!(max_crossing_pairs(&k4), 3);
        let expected_k4 = 1.0 - (1.0_f64 / 3.0).sqrt();
        assert!((crosslessness(&k4, &sq).unwrap() - expected_k4).abs() < 1e-12);
        assert!((sampled_crosslessness(&k4, &sq, 100, 42).unwrap() - expected_k4).abs() < 1e-12);
        assert!((crosslessness_from_count(&k4, 1.0).unwrap() - expected_k4).abs() < 1e-12);

        // 3. Directed graph with reciprocal pair (0,1)/(1,0) and disjoint (2,3) (plus self-loop stripped via sanitize=true).
        let dir =
            CsrGraph::<f64>::from_edges(4, &[(0, 0), (0, 1), (1, 0), (2, 3)], None, true, true)
                .unwrap();
        let (brute_cmax, brute_c) = brute_cmax_and_crossings(&dir, &sq);
        assert_eq!(brute_cmax, 2);
        assert_eq!(max_crossing_pairs(&dir), brute_cmax);
        assert_eq!(brute_c, 0);
        assert_eq!(crosslessness(&dir, &sq).unwrap(), 1.0);

        // 4. Multi-seed random graphs (undirected & directed, f32 & f64) vs O(M^2) oracle.
        for seed in 1..=15u64 {
            let mut rng = DeterministicRng::from_seed_u64(400 + seed);
            let n = 12usize;
            let directed = seed.is_multiple_of(3);
            let mut edges = Vec::new();
            for _ in 0..24 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                if u != v {
                    edges.push((u, v));
                }
            }
            let g = CsrGraph::<f64>::from_edges(n, &edges, None, directed, true).unwrap();
            let xs: Vec<f64> = (0..n).map(|_| rng.next_f64() * 10.0 - 5.0).collect();
            let ys: Vec<f64> = (0..n).map(|_| rng.next_f64() * 10.0 - 5.0).collect();
            let l64 = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
            let l32 = Layout2d::from_xy(
                xs.iter().map(|&v| v as f32).collect(),
                ys.iter().map(|&v| v as f32).collect(),
            )
            .unwrap();

            let (oracle_cmax64, oracle_c64) = brute_cmax_and_crossings(&g, &l64);
            assert_eq!(max_crossing_pairs(&g), oracle_cmax64);
            let expected64 = if oracle_cmax64 == 0 {
                1.0
            } else {
                1.0 - (oracle_c64 as f64 / oracle_cmax64 as f64).sqrt()
            };
            assert!((crosslessness(&g, &l64).unwrap() - expected64).abs() < 1e-12);
            // Total pairs <= 1000 so sampled_crosslessness hits exact branch.
            assert!(
                (sampled_crosslessness(&g, &l64, 1000, 42).unwrap() - expected64).abs() < 1e-12
            );

            let (oracle_cmax32, oracle_c32) = brute_cmax_and_crossings(&g, &l32);
            let expected32 = if oracle_cmax32 == 0 {
                1.0
            } else {
                1.0 - (oracle_c32 as f64 / oracle_cmax32 as f64).sqrt()
            };
            assert!((crosslessness(&g, &l32).unwrap() - expected32).abs() < 1e-12);
        }
    }

    #[test]
    fn test_max_crossing_pairs_directed_reciprocal_and_parallel_oracles() {
        // 1. Single reciprocal pair: c_max == 0, crosslessness == 1.0.
        let recip_pair =
            CsrGraph::<f64>::from_edges(2, &[(0, 1), (1, 0)], None, true, true).unwrap();
        let l2 = Layout2d::from_xy(vec![0.0, 1.0], vec![0.0, 0.0]).unwrap();
        assert_eq!(max_crossing_pairs(&recip_pair), 0);
        assert_eq!(crosslessness(&recip_pair, &l2).unwrap(), 1.0);
        assert_eq!(crosslessness_from_count(&recip_pair, 0.0).unwrap(), 1.0);

        // 2. Reciprocal 3-leaf star: adj_pairs > total_pairs, yet c_max == 0.
        let recip_star = CsrGraph::<f64>::from_edges(
            4,
            &[(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)],
            None,
            true,
            true,
        )
        .unwrap();
        let l4 = Layout2d::from_xy(vec![0.0, 1.0, 0.0, -1.0], vec![0.0, 0.0, 1.0, 0.0]).unwrap();
        assert_eq!(max_crossing_pairs(&recip_star), 0);
        assert_eq!(crosslessness(&recip_star, &l4).unwrap(), 1.0);
        assert_eq!(
            sampled_crosslessness(&recip_star, &l4, 64, 42).unwrap(),
            1.0
        );
        assert_eq!(crosslessness_from_count(&recip_star, 1.0).unwrap(), 1.0);

        // 3. 4-node 6-edge simple directed graph with adj_pairs (16) > total_pairs (15):
        //    only {(0,1), (3,2)} are vertex-disjoint -> c_max = 1, c = 1, crosslessness = 0.0.
        let dir_4 = CsrGraph::<f64>::from_edges(
            4,
            &[(0, 1), (0, 3), (1, 3), (3, 0), (3, 1), (3, 2)],
            None,
            true,
            true,
        )
        .unwrap();
        let l_cross =
            Layout2d::from_xy(vec![0.0, 2.0, 0.0, 2.0], vec![0.0, 2.0, 2.0, 0.0]).unwrap();
        let (brute_cmax, brute_c) = brute_cmax_and_crossings(&dir_4, &l_cross);
        assert_eq!(brute_cmax, 1);
        assert_eq!(brute_c, 1);
        assert_eq!(max_crossing_pairs(&dir_4), 1);
        assert_eq!(crosslessness(&dir_4, &l_cross).unwrap(), 0.0);
        assert_eq!(
            sampled_crosslessness(&dir_4, &l_cross, 100, 42).unwrap(),
            0.0
        );

        // 4. Raw edge list with parallel multi-edges, reciprocal edges, and self-loops vs O(M^2).
        let raw_edges = [
            (0_u32, 1),
            (0, 1),
            (1, 0),
            (0, 2),
            (2, 0),
            (2, 3),
            (2, 3),
            (1, 1),
            (3, 3),
        ];
        let mut expected_raw = 0_u128;
        for i in 0..raw_edges.len() {
            let (u1, v1) = raw_edges[i];
            if u1 == v1 {
                continue;
            }
            for &(u2, v2) in &raw_edges[(i + 1)..] {
                if u2 == v2 {
                    continue;
                }
                if u1 != u2 && u1 != v2 && v1 != u2 && v1 != v2 {
                    expected_raw += 1;
                }
            }
        }
        assert_eq!(max_crossing_pairs_from_edges(4, &raw_edges), expected_raw);
    }
}
