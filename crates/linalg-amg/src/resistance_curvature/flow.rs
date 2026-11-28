use egraph_core::{CoreError, CoreResult, CsrGraph, EdgeWeight};

use super::classify::classify_geometry_from_curvature_with_sampling;
use super::config::{
    ResistanceCurvatureConfig, ResistanceCurvatureFlowResult, ResistanceCurvatureStep,
};
use super::evaluation::{
    compute_curvatures, evaluate_effective_resistances, foster_normalize_per_component,
    resolve_exact_flag,
};
use super::graph_ops::{
    compute_components, degree_regularity_stats, extract_canonical_edges,
    is_closed_spherical_surface,
};

/// Runs Devriendt-Lambiotte resistance curvature flow on `graph` and classifies geometry.
pub fn resistance_curvature_flow<W: EdgeWeight>(
    graph: &CsrGraph<W>,
    config: &ResistanceCurvatureConfig,
) -> CoreResult<ResistanceCurvatureFlowResult> {
    config.validate()?;
    let n = graph.num_nodes();
    let use_exact = resolve_exact_flag(n, &config.resistance)?;
    let canonical = extract_canonical_edges(graph, config.resistance.weights_are_lengths)?;
    let m = canonical.edge_u.len();
    let (num_components, comp_id, comp_sizes) =
        compute_components(n, &canonical.edge_u, &canonical.edge_v);
    let (leaf_fraction, deg_cv) = degree_regularity_stats(n, &canonical.edge_u, &canonical.edge_v);
    let closed_spherical =
        is_closed_spherical_surface(n, &canonical.edge_u, &canonical.edge_v, leaf_fraction);

    let mut edge_lengths = canonical.edge_lengths;
    let mut conductances = canonical.conductances;
    let mut step_history = Vec::with_capacity(config.steps + 1);

    for step_idx in 0..config.steps {
        let step_seed = config.resistance.seed.wrapping_add(step_idx as u64);
        let mut eval = evaluate_effective_resistances(
            n,
            num_components,
            &canonical.edge_u,
            &canonical.edge_v,
            &conductances,
            &config.resistance,
            use_exact,
            step_seed,
        )?;
        if !eval.used_exact {
            let (foster_sum, foster_rel_err) = foster_normalize_per_component(
                num_components,
                &comp_id,
                &comp_sizes,
                &canonical.edge_u,
                &conductances,
                eval.expected_foster,
                &mut eval.effective_resistances,
            );
            eval.foster_sum = foster_sum;
            eval.foster_rel_err = foster_rel_err;
        }
        let curv = compute_curvatures(
            n,
            &canonical.edge_u,
            &canonical.edge_v,
            &conductances,
            &eval.effective_resistances,
        );
        step_history.push(ResistanceCurvatureStep {
            step: step_idx,
            foster_sum: eval.foster_sum,
            expected_foster: eval.expected_foster,
            foster_rel_err: eval.foster_rel_err,
            mean_link_curvature: curv.mean_link,
            std_link_curvature: curv.std_link,
            min_link_curvature: curv.min_link,
            max_link_curvature: curv.max_link,
            mean_vertex_curvature: curv.mean_vertex,
            min_vertex_curvature: curv.min_vertex,
            max_vertex_curvature: curv.max_vertex,
            used_exact: eval.used_exact,
            amg_total_iterations: eval.amg_total_iterations,
            amg_max_relative_residual: eval.amg_max_relative_residual,
        });

        if m > 0 {
            let mut sum_len = 0.0_f64;
            for e in 0..m {
                let kappa_clamped =
                    curv.link_curvatures[e].clamp(-config.kappa_max, config.kappa_max);
                let raw_len = (edge_lengths[e] * (-config.eta * kappa_clamped).exp())
                    .max(config.min_edge_length);
                edge_lengths[e] = raw_len;
                sum_len += raw_len;
            }
            let mean_len = sum_len / (m as f64);
            if !mean_len.is_finite() || mean_len <= 0.0 {
                return Err(CoreError::invalid_input(
                    "resistance_curvature_flow produced non-positive mean edge length",
                ));
            }
            for e in 0..m {
                let norm_len = edge_lengths[e] / mean_len;
                let cond = 1.0 / norm_len;
                if !norm_len.is_finite() || norm_len <= 0.0 || !cond.is_finite() || cond <= 0.0 {
                    return Err(CoreError::invalid_input(
                        "resistance_curvature_flow edge length normalization overflow",
                    ));
                }
                edge_lengths[e] = norm_len;
                conductances[e] = cond;
            }
        }
    }

    let final_seed = config.resistance.seed.wrapping_add(config.steps as u64);
    let mut final_eval = evaluate_effective_resistances(
        n,
        num_components,
        &canonical.edge_u,
        &canonical.edge_v,
        &conductances,
        &config.resistance,
        use_exact,
        final_seed,
    )?;
    if !final_eval.used_exact {
        let (foster_sum, foster_rel_err) = foster_normalize_per_component(
            num_components,
            &comp_id,
            &comp_sizes,
            &canonical.edge_u,
            &conductances,
            final_eval.expected_foster,
            &mut final_eval.effective_resistances,
        );
        final_eval.foster_sum = foster_sum;
        final_eval.foster_rel_err = foster_rel_err;
    }
    let final_curv = compute_curvatures(
        n,
        &canonical.edge_u,
        &canonical.edge_v,
        &conductances,
        &final_eval.effective_resistances,
    );
    step_history.push(ResistanceCurvatureStep {
        step: config.steps,
        foster_sum: final_eval.foster_sum,
        expected_foster: final_eval.expected_foster,
        foster_rel_err: final_eval.foster_rel_err,
        mean_link_curvature: final_curv.mean_link,
        std_link_curvature: final_curv.std_link,
        min_link_curvature: final_curv.min_link,
        max_link_curvature: final_curv.max_link,
        mean_vertex_curvature: final_curv.mean_vertex,
        min_vertex_curvature: final_curv.min_vertex,
        max_vertex_curvature: final_curv.max_vertex,
        used_exact: final_eval.used_exact,
        amg_total_iterations: final_eval.amg_total_iterations,
        amg_max_relative_residual: final_eval.amg_max_relative_residual,
    });

    let suggested_geometry = classify_geometry_from_curvature_with_sampling(
        m,
        final_curv.mean_link,
        final_curv.std_link,
        leaf_fraction,
        deg_cv,
        config.hyperbolic_threshold,
        config.spherical_threshold,
        config.torus_std_threshold,
        final_eval.used_exact,
        config.resistance.jl_dim,
        closed_spherical,
    );

    Ok(ResistanceCurvatureFlowResult {
        num_nodes: n,
        num_edges: m,
        num_components,
        edge_u: canonical.edge_u,
        edge_v: canonical.edge_v,
        edge_lengths,
        conductances,
        effective_resistances: final_eval.effective_resistances,
        vertex_curvatures: final_curv.vertex_curvatures,
        link_curvatures: final_curv.link_curvatures,
        foster_sum: final_eval.foster_sum,
        expected_foster: final_eval.expected_foster,
        foster_rel_err: final_eval.foster_rel_err,
        mean_link_curvature: final_curv.mean_link,
        std_link_curvature: final_curv.std_link,
        min_link_curvature: final_curv.min_link,
        max_link_curvature: final_curv.max_link,
        mean_vertex_curvature: final_curv.mean_vertex,
        min_vertex_curvature: final_curv.min_vertex,
        max_vertex_curvature: final_curv.max_vertex,
        used_exact: final_eval.used_exact,
        amg_total_iterations: final_eval.amg_total_iterations,
        amg_max_relative_residual: final_eval.amg_max_relative_residual,
        amg_all_converged: final_eval.amg_all_converged,
        suggested_geometry,
        step_history,
    })
}
