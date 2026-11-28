use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};
use egraph_linalg::{BfsWorkspace, DijkstraWorkspace};

use super::config::StressMajorizationConfig;

pub(super) fn apsp_from_weighted_graph<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &StressMajorizationConfig,
) -> CoreResult<(usize, Vec<f64>)> {
    config.validate()?;
    if let Some(ws) = graph.weights() {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite(
                    "StressMajorizationSolver::from_graph::weights",
                ));
            }
            if wf < 0.0 {
                return Err(CoreError::invalid_input(
                    "graph edge weights must be non-negative",
                ));
            }
        }
    }

    let n = graph.num_nodes();
    let total = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("APSP distance matrix overflow"))?;
    let mut dists_flat = vec![f64::INFINITY; total];

    if n > 0 {
        let mut bfs = (!graph.is_weighted()).then(|| BfsWorkspace::new(n));
        let mut dijk = graph.is_weighted().then(|| DijkstraWorkspace::new(n));
        for u in 0..n {
            let row = &mut dists_flat[u * n..(u + 1) * n];
            if let Some(ws) = bfs.as_mut() {
                ws.bfs_from(graph, u, row)?;
            } else if let Some(ws) = dijk.as_mut() {
                ws.dijkstra_from(graph, u, row)?;
            }
        }
    }
    Ok((n, dists_flat))
}

pub(super) fn apsp_from_unweighted_graph<W>(
    graph: &CsrGraph<W>,
    config: &StressMajorizationConfig,
) -> CoreResult<(usize, Vec<f64>)> {
    config.validate()?;
    let n = graph.num_nodes();
    let total = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("APSP distance matrix overflow"))?;
    let mut dists_flat = vec![f64::INFINITY; total];

    if n > 0 {
        let mut bfs = BfsWorkspace::new(n);
        for u in 0..n {
            let row = &mut dists_flat[u * n..(u + 1) * n];
            bfs.bfs_from(graph, u, row)?;
        }
    }
    Ok((n, dists_flat))
}

pub(super) fn build_dists_and_weights(
    n: usize,
    dists_flat: &[f64],
) -> CoreResult<(Vec<f64>, Vec<f64>)> {
    let expected = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("distance matrix dimension overflow"))?;
    if dists_flat.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            dists_flat.len(),
            "StressMajorizationSolver::from_distance_matrix",
        ));
    }

    for &d in dists_flat {
        if d.is_nan() {
            return Err(CoreError::non_finite(
                "StressMajorizationSolver::from_distance_matrix",
            ));
        }
        if d < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "distance matrix entries must be >= 0.0, got {d}"
            )));
        }
    }

    let mut dists = vec![0.0_f64; expected];
    let mut weights = vec![0.0_f64; expected];

    for i in 0..n {
        for j in (i + 1)..n {
            let dij_raw = dists_flat[i * n + j];
            let dji_raw = dists_flat[j * n + i];
            let ok_ij = dij_raw.is_finite() && dij_raw > 0.0;
            let ok_ji = dji_raw.is_finite() && dji_raw > 0.0;
            let d = match (ok_ij, ok_ji) {
                (true, true) => 0.5 * (dij_raw + dji_raw),
                (true, false) => dij_raw,
                (false, true) => dji_raw,
                (false, false) => f64::INFINITY,
            };
            if d.is_finite() && d > 0.0 {
                let w = 1.0 / (d * d);
                if w.is_finite() && w > 0.0 {
                    dists[i * n + j] = d;
                    dists[j * n + i] = d;
                    weights[i * n + j] = w;
                    weights[j * n + i] = w;
                }
            }
        }
    }

    Ok((dists, weights))
}
