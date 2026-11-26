//! GuMap 1+2-hop priority CSR [`KnnGraph`] builder enforcing strict 1-hop distance separation.

use std::cmp::Ordering;

use egraph_core::{CoreError, CoreResult};

use super::graph::KnnGraph;

/// Builds a sorted symmetric CSR [`KnnGraph`] from 1-hop and 2-hop pair distances while strictly
/// enforcing $d_1 < \mathrm{tier2\_margin} \cdot d_2$ and $d_1 < d_2$ over incident 1-hop maxima.
#[allow(clippy::too_many_arguments)]
pub fn build_gumap_12hop_priority_csr(
    n_nodes: usize,
    u1: &[u32],
    v1: &[u32],
    d1: &[f64],
    u2: &[u32],
    v2: &[u32],
    d2: &[f64],
    tier2_margin: f64,
) -> CoreResult<KnnGraph> {
    if u32::try_from(n_nodes).is_err() {
        return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
    }
    if !tier2_margin.is_finite() {
        return Err(CoreError::non_finite(
            "build_gumap_12hop_priority_csr::tier2_margin",
        ));
    }
    if tier2_margin <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "tier2_margin must be > 0, got {tier2_margin}"
        )));
    }
    if u1.len() != v1.len() || u1.len() != d1.len() {
        return Err(CoreError::dimension_mismatch(
            u1.len(),
            v1.len().min(d1.len()),
            "build_gumap_12hop_priority_csr::tier1",
        ));
    }
    if u2.len() != v2.len() || u2.len() != d2.len() {
        return Err(CoreError::dimension_mismatch(
            u2.len(),
            v2.len().min(d2.len()),
            "build_gumap_12hop_priority_csr::tier2",
        ));
    }

    for (&u, (&v, &d)) in u1.iter().zip(v1.iter().zip(d1.iter())) {
        if (u as usize) >= n_nodes {
            return Err(CoreError::index_out_of_bounds(
                u as usize,
                n_nodes,
                "build_gumap_12hop_priority_csr::u1",
            ));
        }
        if (v as usize) >= n_nodes {
            return Err(CoreError::index_out_of_bounds(
                v as usize,
                n_nodes,
                "build_gumap_12hop_priority_csr::v1",
            ));
        }
        if !d.is_finite() {
            return Err(CoreError::non_finite("build_gumap_12hop_priority_csr::d1"));
        }
        if d < 0.0 {
            return Err(CoreError::invalid_input("d1 distances must be >= 0.0"));
        }
    }
    for (&u, (&v, &d)) in u2.iter().zip(v2.iter().zip(d2.iter())) {
        if (u as usize) >= n_nodes {
            return Err(CoreError::index_out_of_bounds(
                u as usize,
                n_nodes,
                "build_gumap_12hop_priority_csr::u2",
            ));
        }
        if (v as usize) >= n_nodes {
            return Err(CoreError::index_out_of_bounds(
                v as usize,
                n_nodes,
                "build_gumap_12hop_priority_csr::v2",
            ));
        }
        if !d.is_finite() {
            return Err(CoreError::non_finite("build_gumap_12hop_priority_csr::d2"));
        }
        if d < 0.0 {
            return Err(CoreError::invalid_input("d2 distances must be >= 0.0"));
        }
    }

    if n_nodes <= 1 || (u1.is_empty() && u2.is_empty()) {
        return KnnGraph::empty(n_nodes);
    }

    let mut d1_max = vec![0.0_f64; n_nodes];
    let mut deg = vec![0usize; n_nodes];

    for (&u, &v) in u1.iter().zip(v1.iter()) {
        if u != v {
            deg[u as usize] += 1;
            deg[v as usize] += 1;
        }
    }
    for (&u, &v) in u2.iter().zip(v2.iter()) {
        if u != v {
            deg[u as usize] += 1;
            deg[v as usize] += 1;
        }
    }

    let mut raw_indptr = Vec::with_capacity(n_nodes + 1);
    raw_indptr.push(0usize);
    let mut raw_total = 0usize;
    for &d in &deg {
        raw_total += d;
        raw_indptr.push(raw_total);
    }
    drop(deg);
    if raw_total == 0 {
        return KnnGraph::empty(n_nodes);
    }

    let mut raw_indices = vec![0u32; raw_total];
    let mut raw_dists = vec![0.0_f64; raw_total];
    let mut cursor = raw_indptr[..n_nodes].to_vec();

    for (&u, (&v, &d_raw)) in u1.iter().zip(v1.iter().zip(d1.iter())) {
        if u == v {
            continue;
        }
        let d = d_raw.max(1e-12);
        let ui = u as usize;
        let vi = v as usize;
        d1_max[ui] = d1_max[ui].max(d);
        d1_max[vi] = d1_max[vi].max(d);
        let pu = cursor[ui];
        raw_indices[pu] = v;
        raw_dists[pu] = d;
        cursor[ui] = pu + 1;
        let pv = cursor[vi];
        raw_indices[pv] = u;
        raw_dists[pv] = d;
        cursor[vi] = pv + 1;
    }

    for (&u, (&v, &d_raw)) in u2.iter().zip(v2.iter().zip(d2.iter())) {
        if u == v {
            continue;
        }
        let d = d_raw.max(1e-12);
        let ui = u as usize;
        let vi = v as usize;
        let d1_pair = d1_max[ui].max(d1_max[vi]);
        // Guarantee both d1 < tier2_margin * d2 and strict 1-hop priority d1 < d2 even when
        // tier2_margin <= 1.0 or incident 1-hop edges saturate floating-point precision.
        let d2_floor = if d1_pair > 0.0 {
            if tier2_margin >= 1.0 {
                (tier2_margin * d1_pair).max(d1_pair.next_up())
            } else {
                (d1_pair / tier2_margin).next_up()
            }
        } else {
            1e-12
        };
        let d_adj = d.max(d2_floor);
        let pu = cursor[ui];
        raw_indices[pu] = v;
        raw_dists[pu] = d_adj;
        cursor[ui] = pu + 1;
        let pv = cursor[vi];
        raw_indices[pv] = u;
        raw_dists[pv] = d_adj;
        cursor[vi] = pv + 1;
    }
    drop(cursor);
    drop(d1_max);

    let mut indptr = Vec::with_capacity(n_nodes + 1);
    indptr.push(0usize);
    let mut row_buf: Vec<(u32, f64)> = Vec::new();
    let mut write_pos = 0usize;

    for u in 0..n_nodes {
        let start = raw_indptr[u];
        let end = raw_indptr[u + 1];
        if start < end {
            row_buf.clear();
            for pos in start..end {
                row_buf.push((raw_indices[pos], raw_dists[pos]));
            }
            row_buf.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
            let mut prev_col: Option<u32> = None;
            for &(col, dist) in &row_buf {
                if prev_col == Some(col) {
                    continue;
                }
                prev_col = Some(col);
                raw_indices[write_pos] = col;
                raw_dists[write_pos] = dist;
                write_pos += 1;
            }
        }
        indptr.push(write_pos);
    }
    drop(row_buf);
    drop(raw_indptr);
    raw_indices.truncate(write_pos);
    raw_dists.truncate(write_pos);
    if write_pos * 8 < raw_total * 7 {
        raw_indices.shrink_to_fit();
        raw_dists.shrink_to_fit();
    }

    KnnGraph::new_validated(n_nodes, indptr, raw_indices, raw_dists)
}

#[inline]
#[allow(dead_code)]
pub(crate) fn cmp_dist_then_id(a: &(f64, u32), b: &(f64, u32)) -> Ordering {
    a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1))
}
