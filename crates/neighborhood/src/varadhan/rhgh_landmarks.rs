//! Pivoted Resolvent Heat-Geodesic Hybrid (`RHGH`) landmark distance computation via
//! `AmgPcgSolver` and Spielman-Srivastava JL diagonal probes.

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar};
// Keep AMG-PCG imports isolated on a dedicated line for clean downstream crate splitting.
use egraph_linalg::{
    CsrMatrix, select_maxmin_sp_pivots, spectral_moment_stats, symmetric_normalized_laplacian,
};
use egraph_linalg_amg::{AmgPcgConfig, AmgPcgSolver};

use crate::perplexity::compute_2hop_neighborhood_sizes;

use super::config::{RhghLandmarkConfig, RhghLandmarkStats};
use super::{DogVaradhanLandmarks, RhghLandmarks};

fn median_f64_slice(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(f64::total_cmp);
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        vals[mid]
    } else {
        0.5 * (vals[mid - 1] + vals[mid])
    }
}

struct RhghAutoParams {
    deg_cv: f64,
    leaf_fraction: f64,
    mean_degree: f64,
    exp2: f64,
    s_sw: f64,
    gamma: f64,
    p_exp: f64,
}

fn compute_pivoted_rhgh_landmarks_core(
    n: usize,
    l_sym: &CsrMatrix,
    inv_sqrt_deg: &[f64],
    landmarks: Vec<u32>,
    params: RhghAutoParams,
    config: &RhghLandmarkConfig,
) -> CoreResult<(RhghLandmarks, RhghLandmarkStats)> {
    let RhghAutoParams {
        deg_cv,
        leaf_fraction,
        mean_degree,
        exp2,
        s_sw,
        gamma,
        p_exp,
    } = params;
    let denom_gamma = (1.0 + 1.0 / gamma).ln();

    if n == 0 {
        return Ok((
            RhghLandmarks::from_validated_parts(0, Vec::new(), Vec::new()),
            RhghLandmarkStats {
                gamma,
                p_exp,
                deg_cv,
                leaf_fraction,
                mean_degree,
                exp2,
                s_sw,
                n_landmarks: 0,
                num_probes: config.num_probes,
                max_pcg_iters: 0,
                total_pcg_iters: 0,
                all_converged: true,
                edge_med_scale: 1.0,
            },
        ));
    }

    if n == 1 {
        return Ok((
            RhghLandmarks::from_validated_parts(1, vec![0], vec![0.0]),
            RhghLandmarkStats {
                gamma,
                p_exp,
                deg_cv,
                leaf_fraction,
                mean_degree,
                exp2,
                s_sw,
                n_landmarks: 1,
                num_probes: config.num_probes,
                max_pcg_iters: 0,
                total_pcg_iters: 0,
                all_converged: true,
                edge_med_scale: 1.0,
            },
        ));
    }

    let mut upper_edges: Vec<(u32, u32, f64, f64)> = Vec::new();
    for u in 0..n {
        let inv_u = inv_sqrt_deg[u];
        if inv_u <= 0.0 {
            continue;
        }
        let start = l_sym.indptr()[u] as usize;
        let end = l_sym.indptr()[u + 1] as usize;
        for pos in start..end {
            let v = l_sym.indices()[pos] as usize;
            if v <= u {
                continue;
            }
            let val = l_sym.values()[pos];
            if val < 0.0 {
                let inv_v = inv_sqrt_deg[v];
                if inv_v > 0.0 {
                    let b_u = ((-val) * inv_u / inv_v).max(0.0).sqrt();
                    let b_v = -((-val) * inv_v / inv_u).max(0.0).sqrt();
                    upper_edges.push((u as u32, v as u32, b_u, b_v));
                }
            }
        }
    }

    let m_eff = upper_edges.len();
    let h = landmarks.len();
    if m_eff == 0 {
        let fill_val = ((-1e-15_f64.ln()) / denom_gamma).max(0.0).powf(p_exp);
        let mut d_rhgh = vec![fill_val; n * h];
        for (col, &lm) in landmarks.iter().enumerate() {
            d_rhgh[(lm as usize) * h + col] = 0.0;
        }
        return Ok((
            RhghLandmarks::from_validated_parts(n, landmarks, d_rhgh),
            RhghLandmarkStats {
                gamma,
                p_exp,
                deg_cv,
                leaf_fraction,
                mean_degree,
                exp2,
                s_sw,
                n_landmarks: h,
                num_probes: config.num_probes,
                max_pcg_iters: 0,
                total_pcg_iters: 0,
                all_converged: true,
                edge_med_scale: 1.0,
            },
        ));
    }

    let m_probes = config.num_probes;
    let ncols = h + m_probes;
    let total_rhs = n
        .checked_mul(ncols)
        .ok_or_else(|| CoreError::invalid_input("RHGH RHS matrix size overflow"))?;
    let mut rhs = vec![0.0_f64; total_rhs];
    for (col, &lm) in landmarks.iter().enumerate() {
        rhs[(lm as usize) * ncols + col] = 1.0;
    }

    let mut rng = DeterministicRng::from_seed_u64(config.seed);
    let inv_sqrt_m = 1.0 / (m_probes as f64).sqrt();
    let sqrt_gamma = gamma.sqrt();
    for i in 0..n {
        let row_off = i * ncols + h;
        for k in 0..m_probes {
            let sign = if (rng.next_u64() & 1) == 0 { 1.0 } else { -1.0 };
            rhs[row_off + k] = sign * inv_sqrt_m;
        }
    }
    for &(u, v, b_u, b_v) in &upper_edges {
        let u_off = (u as usize) * ncols + h;
        let v_off = (v as usize) * ncols + h;
        let cu = sqrt_gamma * inv_sqrt_m * b_u;
        let cv = sqrt_gamma * inv_sqrt_m * b_v;
        for k in 0..m_probes {
            let sign = if (rng.next_u64() & 1) == 0 { 1.0 } else { -1.0 };
            rhs[u_off + k] += cu * sign;
            rhs[v_off + k] += cv * sign;
        }
    }

    let a_reg = l_sym.scale_and_shift(gamma, 1.0)?;
    let near_nullspace: Vec<f64> = inv_sqrt_deg
        .iter()
        .map(|&inv| if inv > 0.0 { 1.0 / inv } else { 1.0 })
        .collect();
    let amg_cfg = AmgPcgConfig::new()
        .with_rel_tol(config.rel_tol)
        .with_max_iterations(config.max_iterations)
        .with_seed(config.seed);
    let solver = AmgPcgSolver::with_near_nullspace(&a_reg, Some(&near_nullspace), amg_cfg)?;
    let mut ws = solver.new_workspace();
    let mut sol = vec![0.0_f64; total_rhs];
    let batch_res = solver.solve_batch(&rhs, &mut sol, ncols, &mut ws)?;

    let diag_lo = 1.0 / (1.0 + gamma);
    let mut diag_est = vec![1.0_f64; n];
    for i in 0..n {
        if inv_sqrt_deg[i] > 0.0 {
            let y_row = &sol[i * ncols + h..i * ncols + ncols];
            let sum_sq: f64 = y_row.iter().map(|&y| y * y).sum();
            diag_est[i] = sum_sq.clamp(diag_lo, 1.0);
        }
    }
    for (col, &lm) in landmarks.iter().enumerate() {
        let lmi = lm as usize;
        if inv_sqrt_deg[lmi] > 0.0 {
            diag_est[lmi] = sol[lmi * ncols + col].clamp(diag_lo, 1.0);
        }
    }

    let mut d_dog = vec![0.0_f64; n * h];
    for i in 0..n {
        let di = diag_est[i];
        let sol_row = &sol[i * ncols..i * ncols + h];
        let dst_row = &mut d_dog[i * h..(i + 1) * h];
        for (col, &lm) in landmarks.iter().enumerate() {
            let dp = diag_est[lm as usize];
            let denom = (di * dp).max(1e-30).sqrt();
            let psi = (sol_row[col] / denom).clamp(1e-15, 1.0);
            let log_psi = (-psi.ln() / denom_gamma).max(0.0);
            let val = log_psi.powf(p_exp);
            dst_row[col] = if val.is_finite() { val } else { 0.0 };
        }
    }
    for (col, &lm) in landmarks.iter().enumerate() {
        d_dog[(lm as usize) * h + col] = 0.0;
    }

    let mut edge_med_scale = 1.0_f64;
    if config.normalize_1hop && m_eff > 0 && h > 0 {
        let mut d_edge = Vec::with_capacity(m_eff);
        for &(u, v, _, _) in &upper_edges {
            let ru = &d_dog[(u as usize) * h..(u as usize + 1) * h];
            let rv = &d_dog[(v as usize) * h..(v as usize + 1) * h];
            let mut best = 0.0_f64;
            for (&a, &b) in ru.iter().zip(rv.iter()) {
                let diff = (a - b).abs();
                if diff > best {
                    best = diff;
                }
            }
            d_edge.push(best);
        }
        let med = median_f64_slice(&mut d_edge);
        if med.is_finite() && med > 1e-12 {
            edge_med_scale = med;
            let inv_med = 1.0 / med;
            for val in &mut d_dog {
                *val *= inv_med;
            }
        }
    }

    Ok((
        RhghLandmarks::from_validated_parts(n, landmarks, d_dog),
        RhghLandmarkStats {
            gamma,
            p_exp,
            deg_cv,
            leaf_fraction,
            mean_degree,
            exp2,
            s_sw,
            n_landmarks: h,
            num_probes: m_probes,
            max_pcg_iters: batch_res.max_iterations_used,
            total_pcg_iters: batch_res.total_iterations,
            all_converged: batch_res.all_converged,
            edge_med_scale,
        },
    ))
}

/// Computes pure-Rust $O((h + m)|E|)$ pivoted Resolvent Heat-Geodesic Hybrid (`RHGH`)
/// landmark distance tables using `AmgPcgSolver` + Spielman-Srivastava JL diagonal probes
/// and $O(|E|)$ topology-adaptive auto-$(\gamma, p)$ selection, returning the dedicated
/// [`RhghLandmarks`] result alongside [`RhghLandmarkStats`].
pub fn compute_pivoted_rhgh_landmarks_result<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &RhghLandmarkConfig,
) -> CoreResult<(RhghLandmarks, RhghLandmarkStats)> {
    config.validate()?;
    if let Some(ws) = graph.weights() {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite(
                    "compute_pivoted_rhgh_landmarks::graph::weights",
                ));
            }
            if wf < 0.0 {
                return Err(CoreError::invalid_input(
                    "compute_pivoted_rhgh_landmarks edge weights must be non-negative",
                ));
            }
        }
    }

    let n = graph.num_nodes();
    let (deg_cv, leaf_fraction, mean_degree, exp2, s_sw, gamma_auto, p_auto) =
        if n <= 1 || graph.nnz() == 0 {
            (0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.9)
        } else {
            let st = spectral_moment_stats(graph);
            let k2 = compute_2hop_neighborhood_sizes(graph);
            let mut k2_f64: Vec<f64> = k2.into_iter().map(f64::from).collect();
            let med_k2 = median_f64_slice(&mut k2_f64);
            let exp2_val = med_k2 / st.mean_degree.max(1.0);
            let c_cv = ((st.deg_cv - 0.25) / 0.95).clamp(0.0, 1.0);
            let c_exp = ((exp2_val - 3.0) / 9.0).clamp(0.0, 1.0);
            let c_leaf = (st.leaf_fraction / 0.25).clamp(0.0, 1.0);
            let s_sw_val = (0.65 * c_cv.max(c_exp) + 0.25 * c_cv + 0.10 * c_leaf).clamp(0.0, 1.0);
            let g_auto = 0.5 * 64.0_f64.powf(s_sw_val);
            let p_a = 0.90 - 0.30 * s_sw_val;
            (
                st.deg_cv,
                st.leaf_fraction,
                st.mean_degree,
                exp2_val,
                s_sw_val,
                g_auto,
                p_a,
            )
        };

    let gamma = config.gamma.unwrap_or(gamma_auto);
    let p_exp = config.p_exp.unwrap_or(p_auto);
    let (l_sym, inv_sqrt_deg) = symmetric_normalized_laplacian(graph);
    let k_lm = config.n_landmarks.min(n);
    let landmarks = if n > 1 && graph.nnz() > 0 {
        select_maxmin_sp_pivots(graph, k_lm, config.seed, config.min_degree)?
    } else if n > 0 {
        (0..k_lm as u32).collect()
    } else {
        Vec::new()
    };

    compute_pivoted_rhgh_landmarks_core(
        n,
        &l_sym,
        &inv_sqrt_deg,
        landmarks,
        RhghAutoParams {
            deg_cv,
            leaf_fraction,
            mean_degree,
            exp2,
            s_sw,
            gamma,
            p_exp,
        },
        config,
    )
}

/// Computes pure-Rust $O((h + m)|E|)$ pivoted Resolvent Heat-Geodesic Hybrid (`RHGH`)
/// landmark distance tables using `AmgPcgSolver` + Spielman-Srivastava JL diagonal probes
/// and $O(|E|)$ topology-adaptive auto-$(\gamma, p)$ selection.
///
/// Returns `(DogVaradhanLandmarks, RhghLandmarkStats)` via [`From<RhghLandmarks>`] so the
/// resulting landmark table plugs directly into [`DogVaradhanLandmarks::alt_pair_distances`],
/// `SgdSolver::sparse_barrier_from_landmarks`, and `SgdSolver::pacmap_graph_with_landmarks`.
pub fn compute_pivoted_rhgh_landmarks<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &RhghLandmarkConfig,
) -> CoreResult<(DogVaradhanLandmarks, RhghLandmarkStats)> {
    let (rhgh, stats) = compute_pivoted_rhgh_landmarks_result(graph, config)?;
    Ok((rhgh.into_dog_landmarks(), stats))
}
