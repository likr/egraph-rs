use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, Scalar};

use super::geometry::{Seg, build_segments, extract_csr_edges, segments_cross};

/// Estimates total edge crossings in $O(M)$ time via deterministic edge-pair sampling.
///
/// When the total number of distinct edge pairs $\binom{|E|}{2}$ is at most `num_samples`,
/// evaluates all pairs deterministically and returns the exact crossing count in $O(|E|^2) \le O(M)$.
pub fn sampled_crossing_number<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    num_samples: usize,
    rng: &mut DeterministicRng,
) -> CoreResult<f64> {
    let edges = extract_csr_edges(graph);
    sampled_crossing_number_edges(&edges, layout, num_samples, rng)
}

/// Edge-slice variant of [`sampled_crossing_number`].
pub fn sampled_crossing_number_edges<S: Scalar>(
    edges: &[(u32, u32)],
    layout: &Layout2d<S>,
    num_samples: usize,
    rng: &mut DeterministicRng,
) -> CoreResult<f64> {
    if num_samples == 0 {
        return Err(CoreError::invalid_input("num_samples must be >= 1"));
    }
    let segs = build_segments(edges, layout)?;
    let non_loops: Vec<&Seg> = segs.iter().filter(|s| s.u != s.v).collect();
    let m = non_loops.len();
    if m < 2 {
        return Ok(0.0);
    }

    let total_pairs = (m as u128 * (m as u128 - 1)) / 2;
    if total_pairs <= num_samples as u128 {
        let mut exact = 0usize;
        for i in 0..m {
            for j in (i + 1)..m {
                if segments_cross(non_loops[i], non_loops[j]) {
                    exact += 1;
                }
            }
        }
        return Ok(exact as f64);
    }

    let mut hits = 0usize;
    for _ in 0..num_samples {
        let i = rng.gen_range_usize(m);
        let mut j = rng.gen_range_usize(m - 1);
        if j >= i {
            j += 1;
        }
        if segments_cross(non_loops[i], non_loops[j]) {
            hits += 1;
        }
    }

    Ok((hits as f64 / num_samples as f64) * (total_pairs as f64))
}
