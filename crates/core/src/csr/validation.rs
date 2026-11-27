use crate::error::{CoreError, CoreResult};
use crate::scalar::Scalar;
use crate::visit_stamp::VisitStamp;

pub(super) fn validate_csr_topology(indptr: &[u32], indices: &[u32]) -> CoreResult<usize> {
    if indptr.is_empty() {
        return Err(CoreError::invalid_csr(
            "indptr must contain at least 1 element (num_nodes + 1)",
        ));
    }
    let num_nodes = indptr.len() - 1;
    if u32::try_from(num_nodes).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if indptr[0] != 0 {
        return Err(CoreError::invalid_csr(format!(
            "indptr[0] must be 0, got {}",
            indptr[0]
        )));
    }
    for (row, win) in indptr.windows(2).enumerate() {
        if win[0] > win[1] {
            return Err(CoreError::invalid_csr(format!(
                "indptr is not monotonic at row {row}: {} > {}",
                win[0], win[1]
            )));
        }
    }
    let last_ptr = *indptr.last().unwrap() as usize;
    if last_ptr != indices.len() {
        return Err(CoreError::invalid_csr(format!(
            "indptr[num_nodes] ({last_ptr}) != indices.len() ({})",
            indices.len()
        )));
    }
    for &v in indices {
        if (v as usize) >= num_nodes {
            return Err(CoreError::index_out_of_bounds(
                v as usize,
                num_nodes,
                "CsrGraph::indices",
            ));
        }
    }
    Ok(num_nodes)
}

pub(super) fn validate_csr_structure<W: Scalar>(
    indptr: &[u32],
    indices: &[u32],
    weights: Option<&[W]>,
) -> CoreResult<usize> {
    let num_nodes = validate_csr_topology(indptr, indices)?;
    if let Some(w) = weights {
        if w.len() != indices.len() {
            return Err(CoreError::dimension_mismatch(
                indices.len(),
                w.len(),
                "CsrGraph::weights",
            ));
        }
        if w.iter().any(|val| !val.is_finite()) {
            return Err(CoreError::non_finite("CsrGraph::weights"));
        }
    }
    Ok(num_nodes)
}

pub(super) fn validate_edge_inputs<W: Scalar>(
    num_nodes: usize,
    num_edges: usize,
    weights: Option<&[W]>,
) -> CoreResult<()> {
    if u32::try_from(num_nodes).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if let Some(w) = weights {
        if w.len() != num_edges {
            return Err(CoreError::dimension_mismatch(
                num_edges,
                w.len(),
                "CsrGraph::weights",
            ));
        }
        if w.iter().any(|val| !val.is_finite()) {
            return Err(CoreError::non_finite("CsrGraph::weights"));
        }
    }
    Ok(())
}

pub(super) fn validate_clean_csr_topology(
    num_nodes: usize,
    indptr: &[u32],
    indices: &[u32],
    directed: bool,
) -> CoreResult<()> {
    validate_clean_csr::<f64>(num_nodes, indptr, indices, None, directed)
}

pub(super) fn validate_clean_csr<W: Scalar>(
    num_nodes: usize,
    indptr: &[u32],
    indices: &[u32],
    weights: Option<&[W]>,
    directed: bool,
) -> CoreResult<()> {
    let mut seen = VisitStamp::new(num_nodes);
    let mut all_sorted = true;
    for u in 0..num_nodes {
        seen.clear();
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        let row = &indices[start..end];
        if !row.windows(2).all(|w| w[0] < w[1]) {
            all_sorted = false;
        }
        for &v in row {
            if v as usize == u {
                return Err(CoreError::invalid_csr(format!(
                    "self-loop ({u}, {u}) not permitted when sanitize=false"
                )));
            }
            if !seen.visit(v as usize) {
                return Err(CoreError::invalid_csr(format!(
                    "duplicate edge ({u}, {v}) not permitted when sanitize=false"
                )));
            }
        }
    }

    if !directed {
        if !indices.len().is_multiple_of(2) {
            return Err(CoreError::invalid_csr(
                "undirected CSR must have an even number of directed entries",
            ));
        }
        if all_sorted {
            for u in 0..num_nodes {
                let start = indptr[u] as usize;
                let end = indptr[u + 1] as usize;
                for pos in start..end {
                    let v = indices[pos] as usize;
                    let v_start = indptr[v] as usize;
                    let v_end = indptr[v + 1] as usize;
                    let Ok(offset) = indices[v_start..v_end].binary_search(&(u as u32)) else {
                        return Err(CoreError::invalid_csr(format!(
                            "undirected CSR missing reverse edge ({v}, {u})"
                        )));
                    };
                    if let Some(w) = weights
                        && w[pos] != w[v_start + offset]
                    {
                        return Err(CoreError::invalid_csr(format!(
                            "undirected CSR asymmetric weight for edge ({u}, {v})"
                        )));
                    }
                }
            }
        } else {
            // Fallback for unsorted but duplicate-free rows: sort canonical endpoint pairs and check 2-by-2 pairing.
            let mut pairs: Vec<(u32, u32, W)> = Vec::with_capacity(indices.len());
            for u in 0..num_nodes {
                let u32_u = u as u32;
                let start = indptr[u] as usize;
                let end = indptr[u + 1] as usize;
                for pos in start..end {
                    let v = indices[pos];
                    let (a, b) = if u32_u < v { (u32_u, v) } else { (v, u32_u) };
                    let w = weights.map_or(W::ONE, |ws| ws[pos]);
                    pairs.push((a, b, w));
                }
            }
            pairs.sort_unstable_by_key(|x| (x.0, x.1));
            for [p0, p1] in pairs.as_chunks::<2>().0 {
                if p0.0 != p1.0 || p0.1 != p1.1 {
                    return Err(CoreError::invalid_csr(format!(
                        "undirected CSR missing reverse edge for ({}, {})",
                        p0.0, p0.1
                    )));
                }
                if weights.is_some() && p0.2 != p1.2 {
                    return Err(CoreError::invalid_csr(format!(
                        "undirected CSR asymmetric weight for edge ({}, {})",
                        p0.0, p0.1
                    )));
                }
            }
        }
    }

    Ok(())
}
