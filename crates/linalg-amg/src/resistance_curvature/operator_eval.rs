use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar};
use egraph_linalg::LinearOperator;

use crate::amg_pcg::AmgPcgSolver;

use super::config::{EffectiveResistanceConfig, EffectiveResistanceResult};
use super::evaluation::ResistanceEval;
use super::graph_ops::{count_components, extract_canonical_edges};

/// Computes effective edge resistances $R_e$ and Foster's theorem diagnostics for `graph`
/// via the Spielman-Srivastava JL projection backed by a pre-built [`AmgPcgSolver`] and
/// matrix-free [`LinearOperator<f64>`] `op` representing the graph Laplacian.
pub fn compute_effective_resistances_with_operator<W: Scalar, O: LinearOperator<f64> + ?Sized>(
    graph: &CsrGraph<W>,
    solver: &AmgPcgSolver,
    op: &mut O,
    config: &EffectiveResistanceConfig,
) -> CoreResult<EffectiveResistanceResult> {
    config.validate()?;
    let n = graph.num_nodes();
    if solver.num_nodes() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            solver.num_nodes(),
            "compute_effective_resistances_with_operator::solver",
        ));
    }
    if op.dim() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            op.dim(),
            "compute_effective_resistances_with_operator::op",
        ));
    }
    if config.exact == Some(true) {
        return Err(CoreError::unsupported(
            "compute_effective_resistances_with_operator requires JL operator mode (set exact=Some(false) or None)",
        ));
    }

    let canonical = extract_canonical_edges(graph, config.weights_are_lengths)?;
    let num_components = count_components(n, &canonical.edge_u, &canonical.edge_v);

    let eval = evaluate_effective_resistances_with_operator(
        n,
        num_components,
        &canonical.edge_u,
        &canonical.edge_v,
        &canonical.conductances,
        solver,
        op,
        config,
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

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_effective_resistances_with_operator<O: LinearOperator<f64> + ?Sized>(
    num_nodes: usize,
    num_components: usize,
    edge_u: &[u32],
    edge_v: &[u32],
    conductances: &[f64],
    solver: &AmgPcgSolver,
    op: &mut O,
    config: &EffectiveResistanceConfig,
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
            used_exact: false,
            amg_total_iterations: 0,
            amg_max_relative_residual: 0.0,
            amg_all_converged: true,
        });
    }

    let k_jl = config.jl_dim;
    let (diag, b_rhs) = build_jl_rhs(num_nodes, edge_u, edge_v, conductances, k_jl, seed)?;
    let total = b_rhs.len();

    let mut ws = solver.new_workspace();
    let mut z_sol = vec![0.0_f64; total];
    let batch = solver.solve_batch_with_operator(op, &b_rhs, &mut z_sol, k_jl, &mut ws)?;

    let mut op_z = Vec::new();
    let residual_rhs: &[f64] = if batch.all_converged && batch.max_relative_residual <= 1e-4 {
        &b_rhs
    } else {
        op_z.resize(total, 0.0);
        let mut col_in = vec![0.0_f64; num_nodes];
        let mut col_out = vec![0.0_f64; num_nodes];
        for c in 0..k_jl {
            for r in 0..num_nodes {
                col_in[r] = z_sol[r * k_jl + c];
            }
            op.apply(&col_in, &mut col_out)?;
            for r in 0..num_nodes {
                op_z[r * k_jl + c] = col_out[r];
            }
        }
        &op_z
    };

    let mut effective_resistances = vec![0.0_f64; m];
    apply_jacobi_control_variate(
        edge_u,
        edge_v,
        conductances,
        &diag,
        k_jl,
        &mut z_sol,
        residual_rhs,
        &mut effective_resistances,
    );

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
        used_exact: false,
        amg_total_iterations: batch.total_iterations,
        amg_max_relative_residual: batch.max_relative_residual,
        amg_all_converged: batch.all_converged,
    })
}

pub(super) fn build_jl_rhs(
    num_nodes: usize,
    edge_u: &[u32],
    edge_v: &[u32],
    conductances: &[f64],
    k_jl: usize,
    seed: u64,
) -> CoreResult<(Vec<f64>, Vec<f64>)> {
    let m = edge_u.len();
    let mut diag = vec![0.0_f64; num_nodes];
    for e in 0..m {
        let w = conductances[e];
        diag[edge_u[e] as usize] += w;
        diag[edge_v[e] as usize] += w;
    }

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
    Ok((diag, b_rhs))
}

/// Applies the exact diagonal Jacobi control variate ($M^{-1} = D^{-1}$):
/// $R_{uv} = (1/d_u + 1/d_v - 2 w_{uv}/(d_u d_v)) + \|(I - D^{-1} L) Z (e_u - e_v)\|^2$.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_jacobi_control_variate(
    edge_u: &[u32],
    edge_v: &[u32],
    conductances: &[f64],
    diag: &[f64],
    k_jl: usize,
    z_sol: &mut [f64],
    residual_rhs: &[f64],
    effective_resistances: &mut [f64],
) {
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

    for e in 0..edge_u.len() {
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
