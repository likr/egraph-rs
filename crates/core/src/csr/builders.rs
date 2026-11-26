use super::CsrGraph;
use crate::error::{CoreError, CoreResult};
use crate::scalar::Scalar;

pub(super) fn sanitize_directed_csr<W: Scalar>(
    num_nodes: usize,
    indptr: &[u32],
    indices: &[u32],
    weights: Option<&[W]>,
) -> CoreResult<CsrGraph<W>> {
    let mut out_indptr = Vec::with_capacity(num_nodes + 1);
    out_indptr.push(0u32);
    let mut out_indices = Vec::new();
    let mut out_weights = weights.map(|_| Vec::new());

    if let Some(w_src) = weights {
        let mut row: Vec<(u32, W)> = Vec::new();
        let w_dst = out_weights.as_mut().unwrap();
        for u in 0..num_nodes {
            row.clear();
            let start = indptr[u] as usize;
            let end = indptr[u + 1] as usize;
            for pos in start..end {
                let v = indices[pos];
                if v as usize != u {
                    row.push((v, w_src[pos]));
                }
            }
            row.sort_unstable_by_key(|&(v, _)| v);
            for (v, w) in row.drain(..) {
                if out_indices.len() > *out_indptr.last().unwrap() as usize
                    && *out_indices.last().unwrap() == v
                {
                    let last_w: &mut W = w_dst.last_mut().unwrap();
                    *last_w = (*last_w).max(w);
                } else {
                    out_indices.push(v);
                    w_dst.push(w);
                }
            }
            let ptr = u32::try_from(out_indices.len())
                .map_err(|_| CoreError::invalid_csr("edge count exceeds u32::MAX"))?;
            out_indptr.push(ptr);
        }
    } else {
        let mut row: Vec<u32> = Vec::new();
        for u in 0..num_nodes {
            row.clear();
            let start = indptr[u] as usize;
            let end = indptr[u + 1] as usize;
            for &v in &indices[start..end] {
                if v as usize != u {
                    row.push(v);
                }
            }
            row.sort_unstable();
            row.dedup();
            out_indices.extend_from_slice(&row);
            let ptr = u32::try_from(out_indices.len())
                .map_err(|_| CoreError::invalid_csr("edge count exceeds u32::MAX"))?;
            out_indptr.push(ptr);
        }
    }

    Ok(CsrGraph {
        num_nodes,
        directed: true,
        indptr: out_indptr,
        indices: out_indices,
        weights: out_weights,
    })
}

pub(super) fn build_directed_from_pairs<W: Scalar>(
    num_nodes: usize,
    pairs: impl Iterator<Item = (u32, u32, Option<W>)>,
    has_weights: bool,
    sanitize: bool,
) -> CoreResult<CsrGraph<W>> {
    if has_weights {
        let mut buckets: Vec<Vec<(u32, W)>> = vec![Vec::new(); num_nodes];
        for (u, v, w) in pairs {
            if u == v {
                if !sanitize {
                    return Err(CoreError::invalid_csr(format!(
                        "self-loop ({u}, {u}) not permitted when sanitize=false"
                    )));
                }
                continue;
            }
            buckets[u as usize].push((v, w.unwrap_or(W::ONE)));
        }

        let mut out_indptr = Vec::with_capacity(num_nodes + 1);
        out_indptr.push(0u32);
        let mut out_indices = Vec::new();
        let mut out_weights = Vec::new();

        for (u, row) in buckets.iter_mut().enumerate() {
            row.sort_unstable_by_key(|&(v, _)| v);
            let row_start = out_indices.len();
            for &(v, w) in row.iter() {
                if out_indices.len() > row_start && *out_indices.last().unwrap() == v {
                    if !sanitize {
                        return Err(CoreError::invalid_csr(format!(
                            "duplicate edge ({u}, {v}) not permitted when sanitize=false"
                        )));
                    }
                    let last_w: &mut W = out_weights.last_mut().unwrap();
                    *last_w = (*last_w).max(w);
                } else {
                    out_indices.push(v);
                    out_weights.push(w);
                }
            }
            let ptr = u32::try_from(out_indices.len())
                .map_err(|_| CoreError::invalid_csr("edge count exceeds u32::MAX"))?;
            out_indptr.push(ptr);
        }

        Ok(CsrGraph {
            num_nodes,
            directed: true,
            indptr: out_indptr,
            indices: out_indices,
            weights: Some(out_weights),
        })
    } else {
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_nodes];
        for (u, v, _) in pairs {
            if u == v {
                if !sanitize {
                    return Err(CoreError::invalid_csr(format!(
                        "self-loop ({u}, {u}) not permitted when sanitize=false"
                    )));
                }
                continue;
            }
            buckets[u as usize].push(v);
        }

        let mut out_indptr = Vec::with_capacity(num_nodes + 1);
        out_indptr.push(0u32);
        let mut out_indices = Vec::new();

        for (u, row) in buckets.iter_mut().enumerate() {
            row.sort_unstable();
            let prev_len = row.len();
            row.dedup();
            if !sanitize && row.len() != prev_len {
                return Err(CoreError::invalid_csr(format!(
                    "duplicate directed edge from node {u} not permitted when sanitize=false"
                )));
            }
            out_indices.extend_from_slice(row);
            let ptr = u32::try_from(out_indices.len())
                .map_err(|_| CoreError::invalid_csr("edge count exceeds u32::MAX"))?;
            out_indptr.push(ptr);
        }

        Ok(CsrGraph {
            num_nodes,
            directed: true,
            indptr: out_indptr,
            indices: out_indices,
            weights: None,
        })
    }
}

pub(super) fn build_undirected_from_pairs<W: Scalar>(
    num_nodes: usize,
    pairs: impl Iterator<Item = (u32, u32, Option<W>)>,
    has_weights: bool,
    sanitize: bool,
) -> CoreResult<CsrGraph<W>> {
    if has_weights {
        let mut half_buckets: Vec<Vec<(u32, W)>> = vec![Vec::new(); num_nodes];
        for (u, v, w) in pairs {
            if u == v {
                if !sanitize {
                    return Err(CoreError::invalid_csr(format!(
                        "self-loop ({u}, {u}) not permitted when sanitize=false"
                    )));
                }
                continue;
            }
            let (a, b) = if u < v { (u, v) } else { (v, u) };
            half_buckets[a as usize].push((b, w.unwrap_or(W::ONE)));
        }

        let mut deg = vec![0u32; num_nodes];
        for (a, row) in half_buckets.iter_mut().enumerate() {
            row.sort_unstable_by_key(|&(b, _)| b);
            let mut write = 0usize;
            for read in 0..row.len() {
                let (b, w) = row[read];
                if write > 0 && row[write - 1].0 == b {
                    if !sanitize {
                        return Err(CoreError::invalid_csr(format!(
                            "duplicate undirected edge ({a}, {b}) not permitted when sanitize=false"
                        )));
                    }
                    row[write - 1].1 = row[write - 1].1.max(w);
                } else {
                    row[write] = (b, w);
                    write += 1;
                    deg[a] += 1;
                    deg[b as usize] += 1;
                }
            }
            row.truncate(write);
        }

        let (out_indptr, total_nnz) = prefix_sum_degrees(&deg)?;
        let mut cursor = out_indptr[..num_nodes].to_vec();
        let mut out_indices = vec![0u32; total_nnz];
        let mut out_weights = vec![W::ZERO; total_nnz];

        // Iterating ascending `a` with ascending `b > a` populates every CSR row in strictly sorted order.
        for (a, row) in half_buckets.iter().enumerate() {
            let a_u32 = a as u32;
            for &(b, w) in row {
                let pa = cursor[a] as usize;
                out_indices[pa] = b;
                out_weights[pa] = w;
                cursor[a] += 1;

                let pb = cursor[b as usize] as usize;
                out_indices[pb] = a_u32;
                out_weights[pb] = w;
                cursor[b as usize] += 1;
            }
        }

        Ok(CsrGraph {
            num_nodes,
            directed: false,
            indptr: out_indptr,
            indices: out_indices,
            weights: Some(out_weights),
        })
    } else {
        let mut half_buckets: Vec<Vec<u32>> = vec![Vec::new(); num_nodes];
        for (u, v, _) in pairs {
            if u == v {
                if !sanitize {
                    return Err(CoreError::invalid_csr(format!(
                        "self-loop ({u}, {u}) not permitted when sanitize=false"
                    )));
                }
                continue;
            }
            let (a, b) = if u < v { (u, v) } else { (v, u) };
            half_buckets[a as usize].push(b);
        }

        let mut deg = vec![0u32; num_nodes];
        for (a, row) in half_buckets.iter_mut().enumerate() {
            row.sort_unstable();
            let prev_len = row.len();
            row.dedup();
            if !sanitize && row.len() != prev_len {
                return Err(CoreError::invalid_csr(format!(
                    "duplicate undirected edge incident to node {a} not permitted when sanitize=false"
                )));
            }
            deg[a] += row.len() as u32;
            for &b in row.iter() {
                deg[b as usize] += 1;
            }
        }

        let (out_indptr, total_nnz) = prefix_sum_degrees(&deg)?;
        let mut cursor = out_indptr[..num_nodes].to_vec();
        let mut out_indices = vec![0u32; total_nnz];

        for (a, row) in half_buckets.iter().enumerate() {
            let a_u32 = a as u32;
            for &b in row {
                let pa = cursor[a] as usize;
                out_indices[pa] = b;
                cursor[a] += 1;

                let pb = cursor[b as usize] as usize;
                out_indices[pb] = a_u32;
                cursor[b as usize] += 1;
            }
        }

        Ok(CsrGraph {
            num_nodes,
            directed: false,
            indptr: out_indptr,
            indices: out_indices,
            weights: None,
        })
    }
}

fn prefix_sum_degrees(deg: &[u32]) -> CoreResult<(Vec<u32>, usize)> {
    let mut out_indptr = Vec::with_capacity(deg.len() + 1);
    out_indptr.push(0u32);
    let mut acc = 0usize;
    for &d in deg {
        acc += d as usize;
        let ptr =
            u32::try_from(acc).map_err(|_| CoreError::invalid_csr("CSR nnz exceeds u32::MAX"))?;
        out_indptr.push(ptr);
    }
    Ok((out_indptr, acc))
}
