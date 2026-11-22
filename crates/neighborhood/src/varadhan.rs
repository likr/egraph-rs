//! R1/R2 regularized DoG Varadhan landmark tables, R3 Horvitz-Thompson repulsion scaling,
//! coarse diffusive scaffold tables, landmark ALT pair distances, and relative distance floors.

use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};
use egraph_linalg::{
    ChebyshevWorkspace, SpectralMomentStats, adaptive_chebyshev_degree, build_dog_terms,
    compute_chebyshev_bandpass_coefficients, project_psd_spectral_filter, select_maxmin_sp_pivots,
    spectral_moment_stats, symmetric_normalized_laplacian,
};

/// Computes R3 parameter-free Horvitz-Thompson repulsion density scaling:
/// $\gamma_{\mathrm{eff}} = \gamma_{\mathrm{non2}} \sqrt{N / \max(k_{\mathrm{neg}}, 1)}$.
pub fn compute_horvitz_thompson_gamma(
    gamma_non2: f64,
    num_nodes: usize,
    neg_per_node: usize,
) -> CoreResult<f64> {
    if !gamma_non2.is_finite() {
        return Err(CoreError::non_finite(
            "compute_horvitz_thompson_gamma::gamma_non2",
        ));
    }
    if gamma_non2 < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "gamma_non2 must be >= 0, got {gamma_non2}"
        )));
    }
    if num_nodes == 0 {
        return Ok(0.0);
    }
    let k_neg = neg_per_node.max(1) as f64;
    Ok(gamma_non2 * ((num_nodes as f64) / k_neg).sqrt())
}

/// Configuration for [`compute_regularized_dog_varadhan_landmarks`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DogVaradhanConfig {
    /// Optional short-range diffusion time $t_1$ (`None` uses R1 $1 / (1 + \sigma_\lambda)$).
    pub t1: Option<f64>,
    /// Optional long-range diffusion time $t_2$ (`None` uses R1 $4 / (1 + \sigma_\lambda)$).
    pub t2: Option<f64>,
    /// Optional DoG attenuation $\alpha$ (`None` uses R1 $\max(0, 1 - \sigma_\lambda)$).
    pub alpha: Option<f64>,
    /// Optional explicit rank-1 positivity floor $\beta \in [0, 1)$ (`None` uses `beta_auto`).
    pub beta_floor: Option<f64>,
    /// Optional Chebyshev polynomial degree (`None` uses [`adaptive_chebyshev_degree`]).
    pub degree: Option<usize>,
    /// Number of Core-First MaxMin shortest-path landmarks (`default: 64`).
    pub n_landmarks: usize,
    /// Minimum degree threshold for Core-First MaxMin landmark selection (`default: 2`).
    pub min_degree: usize,
    /// Johnson-Lindenstrauss embedding dimension for positive kernel diagonal estimation (`default: 64`).
    pub jl_dim: usize,
    /// Minimum spectral floor $\epsilon_{\mathrm{psd}}$ enforced during R2 PSD projection (`default: 1e-4`).
    pub eps_psd: f64,
    /// Spectral upper bound $\lambda_{\max}$ for $L_{\mathrm{sym}}$ (`default: 2.0`).
    pub lambda_max: f64,
    /// Whether Varadhan similarities normalize by $\sqrt{K_{uu} K_{pp}}$ (`default: true`).
    pub degree_normalize: bool,
    /// Power exponent $p$ applied to $[-4 t \ln(\mathrm{sim})]^p$ (`default: 0.5`).
    pub p_exp: f64,
    /// Minimum similarity clamp $\epsilon_{\mathrm{sim}} \in (0, 1]$ (`default: 1e-12`).
    pub eps_sim: f64,
    /// Coarse diffusive scaffold time $t_{\mathrm{coarse}}$ evaluated when $\max(t_1, t_2) < 32$ (`default: 48.0`, `0.0` disables).
    pub t_coarse: f64,
    /// Deterministic explicit seed (`default: 42`).
    pub seed: u64,
}

impl Default for DogVaradhanConfig {
    fn default() -> Self {
        Self {
            t1: None,
            t2: None,
            alpha: None,
            beta_floor: None,
            degree: None,
            n_landmarks: 64,
            min_degree: 2,
            jl_dim: 64,
            eps_psd: 1e-4,
            lambda_max: 2.0,
            degree_normalize: true,
            p_exp: 0.5,
            eps_sim: 1e-12,
            t_coarse: 48.0,
            seed: 42,
        }
    }
}

/// Summary diagnostics emitted by [`compute_regularized_dog_varadhan_landmarks`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DogVaradhanStats {
    pub moment_stats: SpectralMomentStats,
    pub t1: f64,
    pub t2: f64,
    pub alpha: f64,
    pub degree: usize,
    pub delta_psd: f64,
    pub h_min: f64,
    pub k_min: f64,
    pub k_med: f64,
    pub beta_auto: f64,
    pub beta_eff: f64,
    pub n_landmarks: usize,
    pub t_coarse: f64,
    pub used_coarse_scaffold: bool,
}

/// Precomputed row-major `(num_nodes, num_landmarks)` R1/R2 DoG Varadhan landmark tables
/// (plus optional coarse diffusive scaffold table) supporting fast ALT lower-bound distance queries.
#[derive(Debug, Clone, PartialEq)]
pub struct DogVaradhanLandmarks {
    pub num_nodes: usize,
    pub landmarks: Vec<u32>,
    pub d_dog: Vec<f64>,
    pub d_coarse: Option<Vec<f64>>,
}

impl DogVaradhanLandmarks {
    /// Returns the number of vertices.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of landmarks.
    #[inline]
    pub fn num_landmarks(&self) -> usize {
        self.landmarks.len()
    }

    /// Returns the selected landmark vertex IDs.
    #[inline]
    pub fn landmarks(&self) -> &[u32] {
        &self.landmarks
    }

    /// Returns the row-major `(num_nodes, num_landmarks)` DoG Varadhan table.
    #[inline]
    pub fn d_dog(&self) -> &[f64] {
        &self.d_dog
    }

    /// Returns the optional coarse diffusive Varadhan scaffold table.
    #[inline]
    pub fn d_coarse(&self) -> Option<&[f64]> {
        self.d_coarse.as_deref()
    }

    /// Evaluates landmark ALT lower-bound distances
    /// $\max_p |D_{\mathrm{dog}}[u, p] - D_{\mathrm{dog}}[v, p]|$ (and max over `d_coarse` when present)
    /// for all pairs `(u[i], v[i])`.
    pub fn alt_pair_distances(&self, u: &[u32], v: &[u32]) -> CoreResult<Vec<f64>> {
        if u.len() != v.len() {
            return Err(CoreError::dimension_mismatch(
                u.len(),
                v.len(),
                "DogVaradhanLandmarks::alt_pair_distances",
            ));
        }
        let k_lm = self.landmarks.len();
        let expected = self.num_nodes.saturating_mul(k_lm);
        if self.d_dog.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                self.d_dog.len(),
                "DogVaradhanLandmarks::d_dog",
            ));
        }
        if let Some(dc) = &self.d_coarse
            && dc.len() != expected
        {
            return Err(CoreError::dimension_mismatch(
                expected,
                dc.len(),
                "DogVaradhanLandmarks::d_coarse",
            ));
        }

        let mut out = Vec::with_capacity(u.len());
        for (&ui, &vi) in u.iter().zip(v.iter()) {
            let u_idx = ui as usize;
            let v_idx = vi as usize;
            if u_idx >= self.num_nodes {
                return Err(CoreError::index_out_of_bounds(
                    u_idx,
                    self.num_nodes,
                    "DogVaradhanLandmarks::alt_pair_distances::u",
                ));
            }
            if v_idx >= self.num_nodes {
                return Err(CoreError::index_out_of_bounds(
                    v_idx,
                    self.num_nodes,
                    "DogVaradhanLandmarks::alt_pair_distances::v",
                ));
            }
            if k_lm == 0 || self.num_nodes <= 1 {
                out.push(0.0);
                continue;
            }
            let ru = &self.d_dog[u_idx * k_lm..(u_idx + 1) * k_lm];
            let rv = &self.d_dog[v_idx * k_lm..(v_idx + 1) * k_lm];
            let mut best = 0.0_f64;
            for (&a, &b) in ru.iter().zip(rv.iter()) {
                let diff = (a - b).abs();
                if diff > best {
                    best = diff;
                }
            }
            if let Some(dc) = &self.d_coarse {
                let cu = &dc[u_idx * k_lm..(u_idx + 1) * k_lm];
                let cv = &dc[v_idx * k_lm..(v_idx + 1) * k_lm];
                for (&a, &b) in cu.iter().zip(cv.iter()) {
                    let diff = (a - b).abs();
                    if diff > best {
                        best = diff;
                    }
                }
            }
            out.push(best);
        }
        Ok(out)
    }
}

fn build_varadhan_table(
    k_lm_mat: &[f64],
    k_u: &[f64],
    k_p: &[f64],
    t_eff: f64,
    config: &DogVaradhanConfig,
) -> Vec<f64> {
    let n = k_u.len();
    let k_lm = k_p.len();
    if n == 0 || k_lm == 0 {
        return Vec::new();
    }
    let max_ku = k_u.iter().fold(1e-12_f64, |acc, &v| acc.max(v));
    let max_kp = k_p.iter().fold(1e-12_f64, |acc, &v| acc.max(v));
    let k_scale = max_ku.max(max_kp);

    let mut out = vec![0.0_f64; n * k_lm];
    for (i, &ku_raw) in k_u.iter().enumerate() {
        let ku = ku_raw.max(1e-12);
        let row_off = i * k_lm;
        for (col, &kp_raw) in k_p.iter().enumerate() {
            let denom = if config.degree_normalize {
                (ku * kp_raw.max(1e-12)).sqrt().max(1e-12)
            } else {
                k_scale
            };
            let sim = (k_lm_mat[row_off + col] / denom).clamp(config.eps_sim, 1.0);
            let log_val = (-4.0 * t_eff * sim.ln()).max(0.0);
            out[row_off + col] = log_val.powf(config.p_exp);
        }
    }
    out
}

/// Computes pure $O(|E| + \mathrm{deg}\cdot k \cdot |E|)$ regularized DoG Varadhan landmark
/// tables combining `egraph-linalg`'s R1/R2 PSD-projected DoG filter, single-pass dual tiled
/// Chebyshev SpMM + JL diagonal estimation, `beta_auto` positivity shift, and an optional
/// coarse heat scaffold at `config.t_coarse` (`48.0`).
pub fn compute_regularized_dog_varadhan_landmarks<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &DogVaradhanConfig,
) -> CoreResult<(DogVaradhanLandmarks, DogVaradhanStats)> {
    if !config.t_coarse.is_finite() {
        return Err(CoreError::non_finite("DogVaradhanConfig::t_coarse"));
    }
    if config.t_coarse < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "t_coarse must be >= 0, got {}",
            config.t_coarse
        )));
    }
    if !config.p_exp.is_finite() {
        return Err(CoreError::non_finite("DogVaradhanConfig::p_exp"));
    }
    if config.p_exp <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "p_exp must be > 0, got {}",
            config.p_exp
        )));
    }
    if !config.eps_sim.is_finite() {
        return Err(CoreError::non_finite("DogVaradhanConfig::eps_sim"));
    }
    if config.eps_sim <= 0.0 || config.eps_sim > 1.0 {
        return Err(CoreError::invalid_input(format!(
            "eps_sim must be in (0, 1], got {}",
            config.eps_sim
        )));
    }
    if config.n_landmarks == 0 {
        return Err(CoreError::invalid_input("n_landmarks must be >= 1"));
    }
    if config.jl_dim == 0 {
        return Err(CoreError::invalid_input("jl_dim must be >= 1"));
    }
    if !config.eps_psd.is_finite() || config.eps_psd < 0.0 {
        return Err(CoreError::invalid_input("eps_psd must be finite and >= 0"));
    }
    if !config.lambda_max.is_finite() || config.lambda_max <= 0.0 {
        return Err(CoreError::invalid_input(
            "lambda_max must be finite and > 0",
        ));
    }
    if let Some(bf) = config.beta_floor
        && (!bf.is_finite() || !(0.0..1.0).contains(&bf))
    {
        return Err(CoreError::invalid_input(
            "beta_floor must be finite and in [0.0, 1.0)",
        ));
    }
    if let Some(deg) = config.degree
        && deg == 0
    {
        return Err(CoreError::invalid_input("degree must be >= 1"));
    }

    let moment_stats = spectral_moment_stats(graph);
    let s = moment_stats.sigma_lambda;
    let t1 = config.t1.unwrap_or(1.0 / (1.0 + s));
    let t2 = config.t2.unwrap_or(4.0 / (1.0 + s));
    let alpha = config.alpha.unwrap_or((1.0 - s).max(0.0));
    let base_terms = build_dog_terms(t1, t2, alpha, 0.0)?;

    let deg_fine = config
        .degree
        .unwrap_or_else(|| adaptive_chebyshev_degree(t1.max(t2)));
    let (terms_psd, delta_psd, h_min) =
        project_psd_spectral_filter(&base_terms, config.lambda_max, deg_fine, config.eps_psd)?;
    let coeffs_fine =
        compute_chebyshev_bandpass_coefficients(&terms_psd, config.lambda_max, deg_fine)?;

    let n = graph.num_nodes();
    if n == 0 {
        let beta_auto = 1e-4;
        let beta_eff = config.beta_floor.unwrap_or(beta_auto);
        let stats = DogVaradhanStats {
            moment_stats,
            t1,
            t2,
            alpha,
            degree: deg_fine,
            delta_psd,
            h_min,
            k_min: 0.0,
            k_med: 0.0,
            beta_auto,
            beta_eff,
            n_landmarks: 0,
            t_coarse: config.t_coarse,
            used_coarse_scaffold: false,
        };
        return Ok((
            DogVaradhanLandmarks {
                num_nodes: 0,
                landmarks: Vec::new(),
                d_dog: Vec::new(),
                d_coarse: None,
            },
            stats,
        ));
    }

    if n == 1 {
        let beta_auto = 1e-4;
        let beta_eff = config.beta_floor.unwrap_or(beta_auto);
        let stats = DogVaradhanStats {
            moment_stats,
            t1,
            t2,
            alpha,
            degree: deg_fine,
            delta_psd,
            h_min,
            k_min: 1.0,
            k_med: 1.0,
            beta_auto,
            beta_eff,
            n_landmarks: 1,
            t_coarse: config.t_coarse,
            used_coarse_scaffold: false,
        };
        return Ok((
            DogVaradhanLandmarks {
                num_nodes: 1,
                landmarks: vec![0],
                d_dog: vec![0.0],
                d_coarse: None,
            },
            stats,
        ));
    }

    let k_lm = config.n_landmarks.min(n);
    let landmarks = select_maxmin_sp_pivots(graph, k_lm, config.seed, config.min_degree)?;
    let k_lm_actual = landmarks.len();

    let (l_sym, _) = symmetric_normalized_laplacian(graph);
    let l_tilde = l_sym.scale_and_shift(2.0 / config.lambda_max, -1.0)?;

    let use_coarse = t1.max(t2) < 32.0 && config.t_coarse > 0.0;
    let deg_c = match config.degree {
        Some(d) => d.max(75),
        None => deg_fine.max(adaptive_chebyshev_degree(config.t_coarse)),
    };
    let coarse_terms = [(1.0, config.t_coarse)];
    let coeffs_c_opt = if use_coarse {
        Some(compute_chebyshev_bandpass_coefficients(
            &coarse_terms,
            config.lambda_max,
            deg_c,
        )?)
    } else {
        None
    };

    let mut e_lm = vec![0.0_f64; n * k_lm_actual];
    for (col, &lm) in landmarks.iter().enumerate() {
        e_lm[(lm as usize) * k_lm_actual + col] = 1.0;
    }

    let mut ws = ChebyshevWorkspace::with_capacity(n * k_lm_actual.min(8));
    let mut k_plus_lm = vec![0.0_f64; n * k_lm_actual];
    let mut k_c_opt = if use_coarse {
        Some(vec![0.0_f64; n * k_lm_actual])
    } else {
        None
    };
    ws.apply_batch_dual(
        &l_tilde,
        &coeffs_fine,
        coeffs_c_opt.as_deref(),
        &e_lm,
        &mut k_plus_lm,
        k_c_opt.as_deref_mut(),
        k_lm_actual,
    )?;

    let n_probes = config.jl_dim.max(k_lm_actual);
    let (mut k_u_plus, k_u_c_opt) = ws.compute_dual_jl_squared_norms_shifted(
        &l_tilde,
        &terms_psd,
        deg_fine,
        use_coarse.then_some((&coarse_terms[..], deg_c)),
        config.lambda_max,
        n_probes,
        config.seed,
    )?;

    let mut k_min = f64::INFINITY;
    for &val in &k_plus_lm {
        if val < k_min {
            k_min = val;
        }
    }
    let mut k_p_plus = Vec::with_capacity(k_lm_actual);
    for (col, &lm) in landmarks.iter().enumerate() {
        k_p_plus.push(k_plus_lm[(lm as usize) * k_lm_actual + col]);
    }
    k_p_plus.sort_by(f64::total_cmp);
    let k_med = if k_p_plus.is_empty() {
        0.0
    } else if k_p_plus.len() % 2 == 1 {
        k_p_plus[k_p_plus.len() / 2]
    } else {
        let mid = k_p_plus.len() / 2;
        0.5 * (k_p_plus[mid - 1] + k_p_plus[mid])
    };

    let denom_beta = (1.0 - k_min).max(1e-12);
    let beta_auto = ((0.02 * k_med - k_min).max(0.0) / denom_beta).clamp(1e-4, 0.5);
    let beta_eff = config.beta_floor.unwrap_or(beta_auto);

    let one_minus_beta = 1.0 - beta_eff;
    for val in &mut k_plus_lm {
        *val = one_minus_beta * (*val) + beta_eff;
    }
    for val in &mut k_u_plus {
        *val = one_minus_beta * (*val) + beta_eff;
    }
    let mut k_p_eff = Vec::with_capacity(k_lm_actual);
    for (col, &lm) in landmarks.iter().enumerate() {
        k_p_eff.push(k_plus_lm[(lm as usize) * k_lm_actual + col]);
    }
    let d_dog = build_varadhan_table(&k_plus_lm, &k_u_plus, &k_p_eff, t1, config);

    let d_coarse = match (k_c_opt, k_u_c_opt) {
        (Some(k_c), Some(k_u_c)) => {
            let mut k_p_c = Vec::with_capacity(k_lm_actual);
            for (col, &lm) in landmarks.iter().enumerate() {
                k_p_c.push(k_c[(lm as usize) * k_lm_actual + col]);
            }
            Some(build_varadhan_table(
                &k_c,
                &k_u_c,
                &k_p_c,
                config.t_coarse,
                config,
            ))
        }
        _ => None,
    };

    let stats = DogVaradhanStats {
        moment_stats,
        t1,
        t2,
        alpha,
        degree: deg_fine,
        delta_psd,
        h_min,
        k_min,
        k_med,
        beta_auto,
        beta_eff,
        n_landmarks: k_lm_actual,
        t_coarse: config.t_coarse,
        used_coarse_scaffold: d_coarse.is_some(),
    };

    Ok((
        DogVaradhanLandmarks {
            num_nodes: n,
            landmarks,
            d_dog,
            d_coarse,
        },
        stats,
    ))
}

/// Clamps pair distances `d1`, `d2`, and `d3` in-place using a relative floor derived from
/// positive 1-hop medians (`max(eps_dist, min_ratio * median_pos)`), returning the applied floor.
pub fn apply_relative_distance_floor(
    d1: &mut [f64],
    d2: &mut [f64],
    d3: &mut [f64],
    eps_dist: f64,
    min_ratio: f64,
) -> CoreResult<f64> {
    if !eps_dist.is_finite() {
        return Err(CoreError::non_finite(
            "apply_relative_distance_floor::eps_dist",
        ));
    }
    if eps_dist <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "eps_dist must be > 0, got {eps_dist}"
        )));
    }
    if !min_ratio.is_finite() {
        return Err(CoreError::non_finite(
            "apply_relative_distance_floor::min_ratio",
        ));
    }
    if min_ratio < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "min_ratio must be >= 0, got {min_ratio}"
        )));
    }
    for (ctx, slice) in [
        ("apply_relative_distance_floor::d1", &*d1),
        ("apply_relative_distance_floor::d2", &*d2),
        ("apply_relative_distance_floor::d3", &*d3),
    ] {
        if slice.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite(ctx));
        }
    }

    let mut pos: Vec<f64> = d1.iter().copied().filter(|&v| v > 0.0).collect();
    if pos.is_empty() {
        pos = d2.iter().copied().filter(|&v| v > 0.0).collect();
    }
    if pos.is_empty() {
        pos = d3.iter().copied().filter(|&v| v > 0.0).collect();
    }

    let ref_med = if pos.is_empty() {
        1.0
    } else {
        pos.sort_by(f64::total_cmp);
        let mid = pos.len() / 2;
        if pos.len() % 2 == 1 {
            pos[mid]
        } else {
            0.5 * (pos[mid - 1] + pos[mid])
        }
    };

    let d_floor = eps_dist.max(min_ratio * ref_med);
    for slice in [d1, d2, d3] {
        for v in slice.iter_mut() {
            if *v < d_floor {
                *v = d_floor;
            }
        }
    }
    Ok(d_floor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knn::extract_sparse_1hop_2hop_pairs;

    #[test]
    fn test_horvitz_thompson_gamma_r3() {
        let g = compute_horvitz_thompson_gamma(1.0, 1600, 16).unwrap();
        assert!((g - 10.0).abs() < 1e-12);
        assert_eq!(compute_horvitz_thompson_gamma(1.0, 0, 16).unwrap(), 0.0);
        assert_eq!(compute_horvitz_thompson_gamma(0.5, 9, 0).unwrap(), 1.5);
        assert!(compute_horvitz_thompson_gamma(-0.1, 10, 4).is_err());
        assert!(compute_horvitz_thompson_gamma(f64::NAN, 10, 4).is_err());
    }

    #[test]
    fn test_regularized_dog_varadhan_landmarks_alt_and_floor() {
        let edges: Vec<(u32, u32)> = (0..16)
            .map(|i| (i, (i + 1) % 16))
            .chain([(0, 8), (4, 12)])
            .collect();
        let g = CsrGraph::<f64>::from_edges(16, &edges, None, false, false).unwrap();

        let cfg = DogVaradhanConfig::default();
        let (lm1, st1) = compute_regularized_dog_varadhan_landmarks(&g, &cfg).unwrap();
        let (lm2, st2) = compute_regularized_dog_varadhan_landmarks(&g, &cfg).unwrap();
        assert_eq!(lm1, lm2);
        assert_eq!(st1, st2);
        assert!(st1.used_coarse_scaffold);
        assert!(st1.beta_eff >= 1e-4 && st1.beta_eff <= 0.5);
        assert!(lm1.d_dog().iter().all(|&v| v.is_finite() && v >= 0.0));
        assert!(
            lm1.d_coarse()
                .unwrap()
                .iter()
                .all(|&v| v.is_finite() && v >= 0.0)
        );

        let pairs = extract_sparse_1hop_2hop_pairs(&g, 8, 42).unwrap();
        let mut d1 = lm1.alt_pair_distances(&pairs.u1, &pairs.v1).unwrap();
        let mut d2 = lm1.alt_pair_distances(&pairs.u2, &pairs.v2).unwrap();
        let mut d3 = lm1.alt_pair_distances(&[0, 1], &[8, 9]).unwrap();
        let floor = apply_relative_distance_floor(&mut d1, &mut d2, &mut d3, 1e-6, 0.5).unwrap();
        assert!(floor >= 1e-6);
        assert!(
            d1.iter()
                .chain(d2.iter())
                .chain(d3.iter())
                .all(|&d| d >= floor)
        );

        // Empty and singleton graphs
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let (le, se) = compute_regularized_dog_varadhan_landmarks(&empty, &cfg).unwrap();
        assert_eq!(le.num_nodes(), 0);
        assert!(!se.used_coarse_scaffold);
        assert!(le.alt_pair_distances(&[], &[]).unwrap().is_empty());
        assert!(le.alt_pair_distances(&[0], &[0]).is_err());

        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let (ls, ss) = compute_regularized_dog_varadhan_landmarks(&single, &cfg).unwrap();
        assert_eq!(ls.num_nodes(), 1);
        assert_eq!(ss.n_landmarks, 1);
        assert_eq!(ls.alt_pair_distances(&[0], &[0]).unwrap(), vec![0.0]);

        // Disconnected graph and NaN config rejection
        let disc =
            CsrGraph::<f64>::from_edges(6, &[(0, 1), (1, 2), (3, 4), (4, 5)], None, false, false)
                .unwrap();
        let (ld, _sd) = compute_regularized_dog_varadhan_landmarks(&disc, &cfg).unwrap();
        let d_disc = ld.alt_pair_distances(&[0, 3], &[2, 5]).unwrap();
        assert!(d_disc.iter().all(|v| v.is_finite() && *v >= 0.0));

        let mut bad_cfg = cfg;
        bad_cfg.t1 = Some(f64::NAN);
        assert!(compute_regularized_dog_varadhan_landmarks(&g, &bad_cfg).is_err());

        let mut bad_cfg2 = cfg;
        bad_cfg2.t_coarse = f64::NAN;
        assert!(compute_regularized_dog_varadhan_landmarks(&g, &bad_cfg2).is_err());

        // Fine DoG table and beta_auto must match identically whether coarse scaffold is enabled or disabled.
        let cfg_fine_only = DogVaradhanConfig {
            degree: Some(25),
            t_coarse: 0.0,
            ..cfg
        };
        let cfg_dual = DogVaradhanConfig {
            degree: Some(25),
            t_coarse: 48.0,
            ..cfg
        };
        let (lm_f, st_f) = compute_regularized_dog_varadhan_landmarks(&g, &cfg_fine_only).unwrap();
        let (lm_d, st_d) = compute_regularized_dog_varadhan_landmarks(&g, &cfg_dual).unwrap();
        assert_eq!(lm_f.landmarks(), lm_d.landmarks());
        assert!((st_f.beta_auto - st_d.beta_auto).abs() < 1e-14);
        for (&a, &b) in lm_f.d_dog().iter().zip(lm_d.d_dog().iter()) {
            assert!((a - b).abs() < 1e-13);
        }
        assert!(lm_f.d_coarse().is_none());
        assert!(lm_d.d_coarse().is_some());
    }
}
