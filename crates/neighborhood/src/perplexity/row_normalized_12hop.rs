//! Directly row-normalized 1+2-hop joint probabilities with optional hub-safe capping and
//! symmetric degree decay.

use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};

use crate::knn::{
    canonical_unweighted_1hop, extract_1hop_2hop_csr, extract_sparse_1hop_2hop_pairs,
};

use super::config::SparseProbabilities;
use super::symmetrize::scatter_symmetric_pairs;

/// Computes directly row-normalized 1+2-hop joint probabilities
/// $P = (D_{\mathrm{row}}^{-1} A + A^T D_{\mathrm{row}}^{-1}) / (2 N)$ for $A = A_1 + w_2 A_2$.
///
/// When `max_2hop_per_node` is `None`, exact uncapped 1+2-hop neighborhoods from
/// [`extract_1hop_2hop_csr`] are used; when `Some(cap)`, 2-hop pairs are hub-safely
/// capped via [`extract_sparse_1hop_2hop_pairs`] using `seed`.
pub fn graph_12hop_row_normalized_probabilities<W: Scalar>(
    graph: &CsrGraph<W>,
    w2: f64,
    max_2hop_per_node: Option<usize>,
    seed: u64,
) -> CoreResult<SparseProbabilities> {
    graph_12hop_row_normalized_probabilities_with_degree_decay(
        graph,
        w2,
        max_2hop_per_node,
        0.0,
        seed,
    )
}

/// Computes directly row-normalized 1+2-hop joint probabilities with symmetric degree decay
/// $(d_i d_j)^{-\beta}$ applied to edge weights before row normalization and symmetrization:
/// $A_{ij} = c_{ij} \cdot (\max(\deg_1(i), 1) \cdot \max(\deg_1(j), 1))^{-\beta}$, where
/// $c_{ij} = 1$ for 1-hop edges and $c_{ij} = w_2$ for pure 2-hop edges.
pub fn graph_12hop_row_normalized_probabilities_with_degree_decay<W: Scalar>(
    graph: &CsrGraph<W>,
    w2: f64,
    max_2hop_per_node: Option<usize>,
    degree_beta: f64,
    seed: u64,
) -> CoreResult<SparseProbabilities> {
    if !w2.is_finite() {
        return Err(CoreError::non_finite(
            "graph_12hop_row_normalized_probabilities::w2",
        ));
    }
    if w2 < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "w2 must be >= 0.0, got {w2}"
        )));
    }
    if !degree_beta.is_finite() {
        return Err(CoreError::non_finite(
            "graph_12hop_row_normalized_probabilities::degree_beta",
        ));
    }
    if degree_beta < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "degree_beta must be >= 0.0, got {degree_beta}"
        )));
    }
    let n = graph.num_nodes();
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if n < 2 {
        return Ok(SparseProbabilities::empty(n));
    }

    if w2 == 0.0 || max_2hop_per_node == Some(0) {
        let adj1 = canonical_unweighted_1hop(graph)?;
        if adj1.nnz() == 0 {
            return Ok(SparseProbabilities::empty(n));
        }
        let mut pairs: Vec<(u32, u32, f64)> = Vec::with_capacity(adj1.num_edges());
        if degree_beta == 0.0 {
            let mut row_sum = vec![0.0_f64; n];
            for (u, sum_out) in row_sum.iter_mut().enumerate() {
                *sum_out = adj1.degree(u) as f64;
            }
            for u in 0..n {
                let u_u32 = u as u32;
                let inv_u = if row_sum[u] > 0.0 {
                    1.0 / row_sum[u]
                } else {
                    0.0
                };
                for &v in adj1.neighbors(u) {
                    if u_u32 < v {
                        let vi = v as usize;
                        let inv_v = if row_sum[vi] > 0.0 {
                            1.0 / row_sum[vi]
                        } else {
                            0.0
                        };
                        pairs.push((u_u32, v, inv_u + inv_v));
                    }
                }
            }
        } else {
            let deg_scale: Vec<f64> = (0..n)
                .map(|u| (adj1.degree(u).max(1) as f64).powf(-degree_beta))
                .collect();
            let mut row_sum = vec![0.0_f64; n];
            for u in 0..n {
                let u_u32 = u as u32;
                let su = deg_scale[u];
                for &v in adj1.neighbors(u) {
                    if u_u32 < v {
                        let vi = v as usize;
                        let w_uv = su * deg_scale[vi];
                        row_sum[u] += w_uv;
                        row_sum[vi] += w_uv;
                    }
                }
            }
            for u in 0..n {
                let u_u32 = u as u32;
                let su = deg_scale[u];
                let ru = row_sum[u];
                for &v in adj1.neighbors(u) {
                    if u_u32 < v {
                        let vi = v as usize;
                        let w_uv = su * deg_scale[vi];
                        if w_uv > 0.0 {
                            let p_uv = if ru > 0.0 { w_uv / ru } else { 0.0 };
                            let p_vu = if row_sum[vi] > 0.0 {
                                w_uv / row_sum[vi]
                            } else {
                                0.0
                            };
                            let val = p_uv + p_vu;
                            if val.is_finite() && val > 0.0 {
                                pairs.push((u_u32, v, val));
                            }
                        }
                    }
                }
            }
        }
        return Ok(scatter_symmetric_pairs(n, pairs, true));
    }

    match max_2hop_per_node {
        None => {
            let (adj1, adj2) = extract_1hop_2hop_csr(graph)?;
            if adj1.nnz() == 0 && adj2.nnz() == 0 {
                return Ok(SparseProbabilities::empty(n));
            }
            let mut pairs: Vec<(u32, u32, f64)> =
                Vec::with_capacity(adj1.num_edges() + adj2.num_edges());
            if degree_beta == 0.0 {
                let mut row_sum = vec![0.0_f64; n];
                for (u, sum_out) in row_sum.iter_mut().enumerate() {
                    *sum_out = (adj1.degree(u) as f64) + w2 * (adj2.degree(u) as f64);
                }
                for u in 0..n {
                    let u_u32 = u as u32;
                    let inv_u = if row_sum[u] > 0.0 {
                        1.0 / row_sum[u]
                    } else {
                        0.0
                    };
                    for &v in adj1.neighbors(u) {
                        if u_u32 < v {
                            let vi = v as usize;
                            let inv_v = if row_sum[vi] > 0.0 {
                                1.0 / row_sum[vi]
                            } else {
                                0.0
                            };
                            pairs.push((u_u32, v, inv_u + inv_v));
                        }
                    }
                    for &v in adj2.neighbors(u) {
                        if u_u32 < v {
                            let vi = v as usize;
                            let inv_v = if row_sum[vi] > 0.0 {
                                1.0 / row_sum[vi]
                            } else {
                                0.0
                            };
                            pairs.push((u_u32, v, w2 * inv_u + w2 * inv_v));
                        }
                    }
                }
            } else {
                let deg_scale: Vec<f64> = (0..n)
                    .map(|u| (adj1.degree(u).max(1) as f64).powf(-degree_beta))
                    .collect();
                let mut row_sum = vec![0.0_f64; n];
                for u in 0..n {
                    let u_u32 = u as u32;
                    let su = deg_scale[u];
                    for &v in adj1.neighbors(u) {
                        if u_u32 < v {
                            let vi = v as usize;
                            let w_uv = su * deg_scale[vi];
                            row_sum[u] += w_uv;
                            row_sum[vi] += w_uv;
                        }
                    }
                    for &v in adj2.neighbors(u) {
                        if u_u32 < v {
                            let vi = v as usize;
                            let w_uv = w2 * su * deg_scale[vi];
                            row_sum[u] += w_uv;
                            row_sum[vi] += w_uv;
                        }
                    }
                }
                for u in 0..n {
                    let u_u32 = u as u32;
                    let su = deg_scale[u];
                    let ru = row_sum[u];
                    for &v in adj1.neighbors(u) {
                        if u_u32 < v {
                            let vi = v as usize;
                            let w_uv = su * deg_scale[vi];
                            if w_uv > 0.0 {
                                let p_uv = if ru > 0.0 { w_uv / ru } else { 0.0 };
                                let p_vu = if row_sum[vi] > 0.0 {
                                    w_uv / row_sum[vi]
                                } else {
                                    0.0
                                };
                                let val = p_uv + p_vu;
                                if val.is_finite() && val > 0.0 {
                                    pairs.push((u_u32, v, val));
                                }
                            }
                        }
                    }
                    for &v in adj2.neighbors(u) {
                        if u_u32 < v {
                            let vi = v as usize;
                            let w_uv = w2 * su * deg_scale[vi];
                            if w_uv > 0.0 {
                                let p_uv = if ru > 0.0 { w_uv / ru } else { 0.0 };
                                let p_vu = if row_sum[vi] > 0.0 {
                                    w_uv / row_sum[vi]
                                } else {
                                    0.0
                                };
                                let val = p_uv + p_vu;
                                if val.is_finite() && val > 0.0 {
                                    pairs.push((u_u32, v, val));
                                }
                            }
                        }
                    }
                }
            }
            Ok(scatter_symmetric_pairs(n, pairs, true))
        }
        Some(cap) => {
            let sp = extract_sparse_1hop_2hop_pairs(graph, cap, seed)?;
            if sp.u1.is_empty() && sp.u2.is_empty() {
                return Ok(SparseProbabilities::empty(n));
            }
            let mut deg1 = vec![0usize; n];
            for (&u, &v) in sp.u1.iter().zip(sp.v1.iter()) {
                deg1[u as usize] += 1;
                deg1[v as usize] += 1;
            }
            let mut pairs: Vec<(u32, u32, f64)> = Vec::with_capacity(sp.u1.len() + sp.u2.len());
            if degree_beta == 0.0 {
                let mut deg2 = vec![0usize; n];
                for (&u, &v) in sp.u2.iter().zip(sp.v2.iter()) {
                    deg2[u as usize] += 1;
                    deg2[v as usize] += 1;
                }
                let mut row_sum = vec![0.0_f64; n];
                for u in 0..n {
                    row_sum[u] = (deg1[u] as f64) + w2 * (deg2[u] as f64);
                }
                for (&u, &v) in sp.u1.iter().zip(sp.v1.iter()) {
                    let ui = u as usize;
                    let vi = v as usize;
                    let inv_u = if row_sum[ui] > 0.0 {
                        1.0 / row_sum[ui]
                    } else {
                        0.0
                    };
                    let inv_v = if row_sum[vi] > 0.0 {
                        1.0 / row_sum[vi]
                    } else {
                        0.0
                    };
                    pairs.push((u, v, inv_u + inv_v));
                }
                for (&u, &v) in sp.u2.iter().zip(sp.v2.iter()) {
                    let ui = u as usize;
                    let vi = v as usize;
                    let inv_u = if row_sum[ui] > 0.0 {
                        w2 / row_sum[ui]
                    } else {
                        0.0
                    };
                    let inv_v = if row_sum[vi] > 0.0 {
                        w2 / row_sum[vi]
                    } else {
                        0.0
                    };
                    pairs.push((u, v, inv_u + inv_v));
                }
            } else {
                let deg_scale: Vec<f64> = (0..n)
                    .map(|u| (deg1[u].max(1) as f64).powf(-degree_beta))
                    .collect();
                let mut row_sum = vec![0.0_f64; n];
                for (&u, &v) in sp.u1.iter().zip(sp.v1.iter()) {
                    let ui = u as usize;
                    let vi = v as usize;
                    let w_uv = deg_scale[ui] * deg_scale[vi];
                    row_sum[ui] += w_uv;
                    row_sum[vi] += w_uv;
                }
                for (&u, &v) in sp.u2.iter().zip(sp.v2.iter()) {
                    let ui = u as usize;
                    let vi = v as usize;
                    let w_uv = w2 * deg_scale[ui] * deg_scale[vi];
                    row_sum[ui] += w_uv;
                    row_sum[vi] += w_uv;
                }
                for (&u, &v) in sp.u1.iter().zip(sp.v1.iter()) {
                    let ui = u as usize;
                    let vi = v as usize;
                    let w_uv = deg_scale[ui] * deg_scale[vi];
                    if w_uv > 0.0 {
                        let p_uv = if row_sum[ui] > 0.0 {
                            w_uv / row_sum[ui]
                        } else {
                            0.0
                        };
                        let p_vu = if row_sum[vi] > 0.0 {
                            w_uv / row_sum[vi]
                        } else {
                            0.0
                        };
                        let val = p_uv + p_vu;
                        if val.is_finite() && val > 0.0 {
                            pairs.push((u, v, val));
                        }
                    }
                }
                for (&u, &v) in sp.u2.iter().zip(sp.v2.iter()) {
                    let ui = u as usize;
                    let vi = v as usize;
                    let w_uv = w2 * deg_scale[ui] * deg_scale[vi];
                    if w_uv > 0.0 {
                        let p_uv = if row_sum[ui] > 0.0 {
                            w_uv / row_sum[ui]
                        } else {
                            0.0
                        };
                        let p_vu = if row_sum[vi] > 0.0 {
                            w_uv / row_sum[vi]
                        } else {
                            0.0
                        };
                        let val = p_uv + p_vu;
                        if val.is_finite() && val > 0.0 {
                            pairs.push((u, v, val));
                        }
                    }
                }
            }
            Ok(scatter_symmetric_pairs(n, pairs, true))
        }
    }
}
