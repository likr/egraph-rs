use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar};
use egraph_linalg::eigen::symmetric_eigen_jacobi;

use crate::amg_pcg::AmgPcgSolver;

use super::config::{EffectiveResistanceConfig, EffectiveResistanceResult};
use super::graph_ops::{
    build_combinatorial_laplacian_from_edges, count_components, extract_canonical_edges,
};
use super::{EXACT_JACOBI_MAX_SWEEPS, EXACT_JACOBI_TOL, LINK_RESISTANCE_EPS, MAX_EXACT_NODES};

pub(super) struct ResistanceEval {
    pub(super) effective_resistances: Vec<f64>,
    pub(super) foster_sum: f64,
    pub(super) expected_foster: f64,
    pub(super) foster_rel_err: f64,
    pub(super) used_exact: bool,
    pub(super) amg_total_iterations: usize,
    pub(super) amg_max_relative_residual: f64,
    pub(super) amg_all_converged: bool,
}

pub(super) struct CurvatureEval {
    pub(super) vertex_curvatures: Vec<f64>,
    pub(super) link_curvatures: Vec<f64>,
    pub(super) mean_link: f64,
    pub(super) std_link: f64,
    pub(super) min_link: f64,
    pub(super) max_link: f64,
    pub(super) mean_vertex: f64,
    pub(super) min_vertex: f64,
    pub(super) max_vertex: f64,
}

/// Computes effective edge resistances $R_e$ and Foster's theorem diagnostics for `graph`.
pub fn compute_effective_resistances<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &EffectiveResistanceConfig,
) -> CoreResult<EffectiveResistanceResult> {
    config.validate()?;
    let n = graph.num_nodes();
    let use_exact = resolve_exact_flag(n, config)?;
    let canonical = extract_canonical_edges(graph, config.weights_are_lengths)?;
    let num_components = count_components(n, &canonical.edge_u, &canonical.edge_v);

    let eval = evaluate_effective_resistances(
        n,
        num_components,
        &canonical.edge_u,
        &canonical.edge_v,
        &canonical.conductances,
        config,
        use_exact,
        config.seed,
    )?;

    Ok(EffectiveResistanceResult {
        num_nodes: n,
        num_edges: canonical.edge_u.len(),
        num_components,
        edge_u: canonical.edge_u,
        edge_v: canonical.edge_v,
        edge_lengths: canonical.edge_lengths,
        conductances: canonical.conductances,
        effective_resistances: eval.effective_resistances,
        foster_sum: eval.foster_sum,
        expected_foster: eval.expected_foster,
        foster_rel_err: eval.foster_rel_err,
        used_exact: eval.used_exact,
        amg_total_iterations: eval.amg_total_iterations,
        amg_max_relative_residual: eval.amg_max_relative_residual,
        amg_all_converged: eval.amg_all_converged,
    })
}

pub(super) fn resolve_exact_flag(
    num_nodes: usize,
    config: &EffectiveResistanceConfig,
) -> CoreResult<bool> {
    match config.exact {
        Some(true) => {
            if num_nodes > MAX_EXACT_NODES {
                return Err(CoreError::invalid_input(format!(
                    "exact=true is capped at {MAX_EXACT_NODES} nodes, got {num_nodes}"
                )));
            }
            Ok(true)
        }
        Some(false) => Ok(false),
        None => Ok(num_nodes <= config.exact_max_nodes),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_effective_resistances(
    num_nodes: usize,
    num_components: usize,
    edge_u: &[u32],
    edge_v: &[u32],
    conductances: &[f64],
    config: &EffectiveResistanceConfig,
    use_exact: bool,
    seed: u64,
) -> CoreResult<ResistanceEval> {
    let m = edge_u.len();
    let expected_foster = if num_nodes == 0 {
        0.0
    } else {
        (num_nodes.saturating_sub(num_components)) as f64
    };

    if num_nodes == 0 || m == 0 {
        return Ok(ResistanceEval {
            effective_resistances: Vec::new(),
            foster_sum: 0.0,
            expected_foster,
            foster_rel_err: 0.0,
            used_exact: use_exact,
            amg_total_iterations: 0,
            amg_max_relative_residual: 0.0,
            amg_all_converged: true,
        });
    }

    let l_mat = build_combinatorial_laplacian_from_edges(num_nodes, edge_u, edge_v, conductances)?;
    let mut effective_resistances = vec![0.0_f64; m];
    let mut amg_total_iterations = 0usize;
    let mut amg_max_relative_residual = 0.0_f64;
    let mut amg_all_converged = true;

    if use_exact {
        let dense = l_mat.to_dense()?;
        let (lambdas, vectors) =
            symmetric_eigen_jacobi(&dense, num_nodes, EXACT_JACOBI_MAX_SWEEPS, EXACT_JACOBI_TOL)?;
        let lambda_max = lambdas.iter().copied().fold(0.0_f64, f64::max);
        let cutoff = if lambda_max > 0.0 {
            config.eig_eps * lambda_max
        } else {
            0.0
        };
        let mut active_modes: Vec<(usize, f64)> = Vec::new();
        for (k, &lam) in lambdas.iter().enumerate() {
            if lambda_max > 0.0 && lam > cutoff {
                let inv = 1.0 / lam;
                if inv.is_finite() {
                    active_modes.push((k, inv));
                }
            }
        }
        for e in 0..m {
            let u_off = (edge_u[e] as usize) * num_nodes;
            let v_off = (edge_v[e] as usize) * num_nodes;
            let mut r_uv = 0.0_f64;
            for &(k, inv_l) in &active_modes {
                let diff = vectors[u_off + k] - vectors[v_off + k];
                r_uv += inv_l * diff * diff;
            }
            effective_resistances[e] = r_uv.max(0.0);
        }
    } else {
        let mut diag = vec![0.0_f64; num_nodes];
        for e in 0..m {
            let w = conductances[e];
            diag[edge_u[e] as usize] += w;
            diag[edge_v[e] as usize] += w;
        }

        let k_jl = config.jl_dim;
        let inv_sqrt_k = 1.0 / (k_jl as f64).sqrt();
        let total = num_nodes
            .checked_mul(k_jl)
            .ok_or_else(|| CoreError::invalid_input("JL projection matrix size overflow"))?;
        let mut b_rhs = vec![0.0_f64; total];
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut bits = 0u64;
        let mut bit_rem = 0u32;

        for e in 0..m {
            let u_off = (edge_u[e] as usize) * k_jl;
            let v_off = (edge_v[e] as usize) * k_jl;
            let s = conductances[e].sqrt() * inv_sqrt_k;
            for c in 0..k_jl {
                if bit_rem == 0 {
                    bits = rng.next_u64();
                    bit_rem = 64;
                }
                let signed = if (bits & 1) == 0 { s } else { -s };
                bits >>= 1;
                bit_rem -= 1;
                b_rhs[u_off + c] += signed;
                b_rhs[v_off + c] -= signed;
            }
        }

        let solver = AmgPcgSolver::with_near_nullspace(&l_mat, None, config.amg_config)?;
        let mut ws = solver.new_workspace();
        let mut z_sol = vec![0.0_f64; total];
        let batch = solver.solve_batch(&b_rhs, &mut z_sol, k_jl, &mut ws)?;
        amg_total_iterations = batch.total_iterations;
        amg_max_relative_residual = batch.max_relative_residual;
        amg_all_converged = batch.all_converged;

        // Exact diagonal Jacobi control variate (M^-1 = D^-1):
        // R_uv = (1/d_u + 1/d_v - 2*w_uv/(d_u*d_v)) + ||(I - D^-1 L) Z (e_u - e_v)||^2.
        // Only when every PCG column actually converged within 1e-4 relative residual does
        // L Z == b_rhs hold accurately enough to reuse D^-1 b_rhs with zero extra SpMVs;
        // loose tolerances or non-converged iteration caps fall back to explicit L Z SpMM.
        let mut l_mat_z = Vec::new();
        let residual_rhs: &[f64] = if batch.all_converged && batch.max_relative_residual <= 1e-4 {
            &b_rhs
        } else {
            l_mat_z.resize(total, 0.0);
            l_mat.spmm(&z_sol, &mut l_mat_z, k_jl)?;
            &l_mat_z
        };
        for ((&d, z_row), rhs_row) in diag
            .iter()
            .zip(z_sol.chunks_exact_mut(k_jl))
            .zip(residual_rhs.chunks_exact(k_jl))
        {
            if d > 0.0 {
                let inv_d = 1.0 / d;
                for c in 0..k_jl {
                    z_row[c] -= rhs_row[c] * inv_d;
                }
            }
        }

        for e in 0..m {
            let u = edge_u[e] as usize;
            let v = edge_v[e] as usize;
            let du = diag[u];
            let dv = diag[v];
            let r_det = (1.0 / du + 1.0 / dv - 2.0 * conductances[e] / (du * dv)).max(0.0);
            let u_row = &z_sol[u * k_jl..(u + 1) * k_jl];
            let v_row = &z_sol[v * k_jl..(v + 1) * k_jl];
            let mut r_res = 0.0_f64;
            for c in 0..k_jl {
                let diff = u_row[c] - v_row[c];
                r_res += diff * diff;
            }
            effective_resistances[e] = r_det + r_res;
        }
    }

    let mut foster_sum = 0.0_f64;
    for e in 0..m {
        foster_sum += conductances[e] * effective_resistances[e];
    }
    let foster_rel_err = if expected_foster > 0.0 {
        (foster_sum - expected_foster).abs() / expected_foster
    } else {
        foster_sum.abs()
    };

    Ok(ResistanceEval {
        effective_resistances,
        foster_sum,
        expected_foster,
        foster_rel_err,
        used_exact: use_exact,
        amg_total_iterations,
        amg_max_relative_residual,
        amg_all_converged,
    })
}

/// Rescales JL effective resistances per connected component so each non-trivial component
/// satisfies Foster's identity $\sum_{e \in E_k} w_e R_e = |V_k| - 1$, returning the updated
/// `(foster_sum, foster_rel_err)`.
pub(super) fn foster_normalize_per_component(
    num_components: usize,
    comp_id: &[usize],
    comp_sizes: &[usize],
    edge_u: &[u32],
    conductances: &[f64],
    expected_foster: f64,
    effective_resistances: &mut [f64],
) -> (f64, f64) {
    if effective_resistances.is_empty() || num_components == 0 {
        return (0.0, 0.0);
    }

    let mut comp_obs = vec![0.0_f64; num_components];
    for e in 0..edge_u.len() {
        let cid = comp_id[edge_u[e] as usize];
        comp_obs[cid] += conductances[e] * effective_resistances[e];
    }

    let mut comp_scale = vec![1.0_f64; num_components];
    for cid in 0..num_components {
        let sz = comp_sizes[cid];
        let obs = comp_obs[cid];
        if sz > 1 && obs > 0.0 {
            comp_scale[cid] = ((sz - 1) as f64) / obs;
        }
    }

    let mut foster_sum = 0.0_f64;
    for e in 0..edge_u.len() {
        let cid = comp_id[edge_u[e] as usize];
        effective_resistances[e] *= comp_scale[cid];
        foster_sum += conductances[e] * effective_resistances[e];
    }
    let foster_rel_err = if expected_foster > 0.0 {
        (foster_sum - expected_foster).abs() / expected_foster
    } else {
        foster_sum.abs()
    };
    (foster_sum, foster_rel_err)
}

pub(super) fn compute_curvatures(
    num_nodes: usize,
    edge_u: &[u32],
    edge_v: &[u32],
    conductances: &[f64],
    effective_resistances: &[f64],
) -> CurvatureEval {
    let mut vertex_curvatures = vec![1.0_f64; num_nodes];
    let m = edge_u.len();
    for e in 0..m {
        let u = edge_u[e] as usize;
        let v = edge_v[e] as usize;
        let contrib = 0.5 * conductances[e] * effective_resistances[e];
        vertex_curvatures[u] -= contrib;
        vertex_curvatures[v] -= contrib;
    }

    let mut link_curvatures = Vec::with_capacity(m);
    for e in 0..m {
        let r_uv = effective_resistances[e];
        if r_uv > LINK_RESISTANCE_EPS {
            let u = edge_u[e] as usize;
            let v = edge_v[e] as usize;
            link_curvatures.push(2.0 * (vertex_curvatures[u] + vertex_curvatures[v]) / r_uv);
        } else {
            link_curvatures.push(0.0);
        }
    }

    let (mean_link, std_link, min_link, max_link) = summarize_slice(&link_curvatures);
    let (mean_vertex, _, min_vertex, max_vertex) = summarize_slice(&vertex_curvatures);

    CurvatureEval {
        vertex_curvatures,
        link_curvatures,
        mean_link,
        std_link,
        min_link,
        max_link,
        mean_vertex,
        min_vertex,
        max_vertex,
    }
}

fn summarize_slice(vals: &[f64]) -> (f64, f64, f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut sum = 0.0_f64;
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in vals {
        sum += v;
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    let mean = sum / (vals.len() as f64);
    let mut var = 0.0_f64;
    for &v in vals {
        let d = v - mean;
        var += d * d;
    }
    let std = (var / (vals.len() as f64)).sqrt();
    (mean, std, min_v, max_v)
}
