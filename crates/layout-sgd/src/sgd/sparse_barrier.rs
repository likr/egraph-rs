use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar, VisitStamp};
use egraph_linalg::select_maxmin_sp_pivots;
use egraph_neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, DogVaradhanStats, apply_relative_distance_floor,
    compute_horvitz_thompson_gamma, compute_regularized_dog_varadhan_landmarks,
    extract_sparse_1hop_2hop_pairs,
};

use super::config::{BarrierSymmetry, SparseBarrierConfig, SparseBarrierStats};
use super::graph_helpers::{
    build_symmetric_csr_from_sorted_pairs, compute_adaptive_perplexity_from_pairs,
    sample_non2hop_pairs,
};
use super::pairs::{PackedSymPair, validate_barrier_config};

pub(super) struct BuiltSparseBarrier<S: Scalar> {
    pub(super) pairs: Vec<PackedSymPair<S>>,
    pub(super) n_two_sided: usize,
    pub(super) n_hinge: usize,
    pub(super) stats: SparseBarrierStats,
}

pub(super) fn build_sparse_barrier_from_landmarks<W, S: Scalar>(
    graph: &CsrGraph<W>,
    landmarks: &DogVaradhanLandmarks,
    pivots: Option<&[u32]>,
    config: &SparseBarrierConfig,
) -> CoreResult<BuiltSparseBarrier<S>> {
    validate_barrier_config(config, pivots.is_some())?;
    let n = graph.num_nodes();
    if landmarks.num_nodes() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            landmarks.num_nodes(),
            "SgdSolver::sparse_barrier_from_landmarks::landmarks",
        ));
    }
    let k_lm = landmarks.num_landmarks();
    if n > 0 && k_lm == 0 {
        return Err(CoreError::invalid_input(
            "landmarks must have at least 1 landmark when num_nodes > 0",
        ));
    }
    let expected = n
        .checked_mul(k_lm)
        .ok_or_else(|| CoreError::invalid_input("landmark table size overflow"))?;
    if landmarks.d_dog().len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            landmarks.d_dog().len(),
            "DogVaradhanLandmarks::d_dog",
        ));
    }
    if landmarks.d_dog().iter().any(|v| !v.is_finite()) {
        return Err(CoreError::non_finite("DogVaradhanLandmarks::d_dog"));
    }
    if let Some(dc) = landmarks.d_coarse() {
        if dc.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                dc.len(),
                "DogVaradhanLandmarks::d_coarse",
            ));
        }
        if dc.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite("DogVaradhanLandmarks::d_coarse"));
        }
    }
    for &lm in landmarks.landmarks() {
        if (lm as usize) >= n {
            return Err(CoreError::index_out_of_bounds(
                lm as usize,
                n,
                "DogVaradhanLandmarks::landmarks",
            ));
        }
    }

    let unique_pivots: Vec<u32> = match pivots {
        Some(supplied) => {
            let mut seen = VisitStamp::new(n);
            let mut out = Vec::with_capacity(supplied.len());
            for &p in supplied {
                let pi = p as usize;
                if pi >= n {
                    return Err(CoreError::index_out_of_bounds(
                        pi,
                        n,
                        "SgdSolver::sparse_barrier_from_landmarks::pivots",
                    ));
                }
                if seen.visit(pi) {
                    out.push(p);
                }
            }
            out
        }
        None => {
            let min_deg = if config.core_pivots { 2 } else { 1 };
            select_maxmin_sp_pivots(graph, config.num_pivots, config.seed, min_deg)?
        }
    };

    if n <= 1 {
        let (perplexity_val, per_node_neg_used, perplexity_source) = match config.neg_per_node {
            Some(k) => (k as f64, k, "explicit"),
            None => (1.0, 1, "adaptive_2hop"),
        };
        let gamma_eff = if config.ht_repulsion {
            compute_horvitz_thompson_gamma(config.gamma_non2, n, per_node_neg_used)?
        } else {
            config.gamma_non2
        };
        let stats = SparseBarrierStats {
            n_pairs_1hop: 0,
            n_pairs_2hop: 0,
            n_pairs_non2: 0,
            n_pairs_hinge: 0,
            n_pairs_two_sided: 0,
            perplexity_val,
            per_node_neg_used,
            perplexity_source,
            d_floor: config.eps_dist,
            gamma_eff,
        };
        return Ok(BuiltSparseBarrier {
            pairs: Vec::new(),
            n_two_sided: 0,
            n_hinge: 0,
            stats,
        });
    }

    let pairs = extract_sparse_1hop_2hop_pairs(graph, config.max_2hop_per_node, config.seed)?;
    let (perplexity_val, per_node_neg_used, perplexity_source) = match config.neg_per_node {
        Some(k) => (k as f64, k, "explicit"),
        None => {
            let p_val = compute_adaptive_perplexity_from_pairs(n, &pairs);
            let neg = if n > 3 {
                (p_val.round_ties_even().max(0.0) as usize).max(1)
            } else {
                1
            };
            (p_val, neg, "adaptive_2hop")
        }
    };

    let (u3, v3) = if n > 2 && per_node_neg_used > 0 {
        let (adj_indptr, adj_indices) =
            build_symmetric_csr_from_sorted_pairs(n, pairs.u1(), pairs.v1());
        sample_non2hop_pairs(
            n,
            &adj_indptr,
            &adj_indices,
            &unique_pivots,
            &pairs,
            per_node_neg_used,
            config.seed,
        )
    } else {
        (Vec::new(), Vec::new())
    };

    let mut d1 = landmarks.alt_pair_distances(pairs.u1(), pairs.v1())?;
    let mut d2 = landmarks.alt_pair_distances(pairs.u2(), pairs.v2())?;
    let mut d3 = landmarks.alt_pair_distances(&u3, &v3)?;
    let d_floor = apply_relative_distance_floor(
        &mut d1,
        &mut d2,
        &mut d3,
        config.eps_dist,
        config.min_ratio,
    )?;

    let gamma_eff = if config.ht_repulsion {
        compute_horvitz_thompson_gamma(config.gamma_non2, n, per_node_neg_used)?
    } else {
        config.gamma_non2
    };

    let deg_scale: Vec<f64> = if config.degree_weight_beta > 0.0 {
        let mut degs = vec![0usize; n];
        for (&u, &v) in pairs.u1().iter().zip(pairs.v1().iter()) {
            degs[u as usize] += 1;
            degs[v as usize] += 1;
        }
        let beta = config.degree_weight_beta;
        degs.into_iter()
            .map(|d| (d.max(1) as f64).powf(-beta))
            .collect()
    } else {
        Vec::new()
    };

    let mut r2 = vec![0.0_f64; n];
    let mut packed_pairs = Vec::with_capacity(pairs.num_1hop() + pairs.num_2hop() + u3.len());
    let mut n_pairs_two_sided = 0usize;
    let mut n_pairs_hinge = 0usize;

    for (pairs_u, pairs_v, dists, omega) in [
        (pairs.u1(), pairs.v1(), &d1, config.omega_1),
        (pairs.u2(), pairs.v2(), &d2, config.omega_2),
    ] {
        for idx in 0..pairs_u.len() {
            let u = pairs_u[idx];
            let v = pairs_v[idx];
            let d = dists[idx];
            r2[u as usize] = r2[u as usize].max(d);
            r2[v as usize] = r2[v as usize].max(d);
            if omega > 0.0 {
                let deg_factor = if deg_scale.is_empty() {
                    1.0
                } else {
                    deg_scale[u as usize] * deg_scale[v as usize]
                };
                let d_s = S::from_f64(d);
                let w_s = S::from_f64((omega * deg_factor) / (d * d));
                if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
                    packed_pairs.push(PackedSymPair::new(u, v, false, d_s, w_s));
                    n_pairs_two_sided += 1;
                }
            }
        }
    }

    let barrier_mult = 1.0 + config.delta_rel;
    for idx in 0..u3.len() {
        let u = u3[idx];
        let v = v3[idx];
        let r_pair = match config.barrier_sym {
            BarrierSymmetry::SymMax => r2[u as usize].max(r2[v as usize]),
            BarrierSymmetry::SymAvg => 0.5 * (r2[u as usize] + r2[v as usize]),
        };
        let d3_bar = d3[idx].max(barrier_mult * r_pair).max(config.eps_dist);
        if gamma_eff > 0.0 {
            let d_s = S::from_f64(d3_bar);
            let w_s = S::from_f64(gamma_eff / (d3_bar * d3_bar));
            if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
                packed_pairs.push(PackedSymPair::new(u, v, config.hinge_repulsion, d_s, w_s));
                if config.hinge_repulsion {
                    n_pairs_hinge += 1;
                } else {
                    n_pairs_two_sided += 1;
                }
            }
        }
    }

    let stats = SparseBarrierStats {
        n_pairs_1hop: pairs.num_1hop(),
        n_pairs_2hop: pairs.num_2hop(),
        n_pairs_non2: u3.len(),
        n_pairs_hinge,
        n_pairs_two_sided,
        perplexity_val,
        per_node_neg_used,
        perplexity_source,
        d_floor,
        gamma_eff,
    };
    Ok(BuiltSparseBarrier {
        pairs: packed_pairs,
        n_two_sided: n_pairs_two_sided,
        n_hinge: n_pairs_hinge,
        stats,
    })
}

pub(super) fn build_sparse_barrier_dog<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    dog_config: &DogVaradhanConfig,
    barrier_config: &SparseBarrierConfig,
) -> CoreResult<(BuiltSparseBarrier<S>, DogVaradhanStats)> {
    validate_barrier_config(barrier_config, false)?;
    let (landmarks, dog_stats) = compute_regularized_dog_varadhan_landmarks(graph, dog_config)?;
    let built = build_sparse_barrier_from_landmarks(graph, &landmarks, None, barrier_config)?;
    Ok((built, dog_stats))
}
