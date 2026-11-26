use egraph_core::{CoreResult, CsrGraph, Layout2d, Scalar};
use egraph_linalg::{DijkstraWorkspace, select_maxmin_sp_pivots};
use egraph_linalg_amg::{
    EffectiveResistanceConfig, ResistanceCurvatureConfig, resistance_curvature_flow,
};
use egraph_linalg_mds::{pivot_mds_from_distances, robust_normalize_layout_2d};
use egraph_neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, RhghLandmarkConfig, compute_pivoted_rhgh_landmarks,
    compute_regularized_dog_varadhan_landmarks,
};

use super::config::{CoarseBackboneConfig, CoarseDistanceBackend, CoarseEdgeLengthMode};

pub(super) struct CoarseMetaEdges {
    pub edge_u: Vec<u32>,
    pub edge_v: Vec<u32>,
    pub edge_w: Vec<f64>,
    pub l_edge: Vec<f64>,
    pub median_l_edge: f64,
}

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

pub(super) fn build_coarse_meta_edges<S: Scalar>(
    cg: &CsrGraph<f64>,
    n_l: usize,
    radii: &[f64],
    config: &CoarseBackboneConfig,
) -> CoreResult<CoarseMetaEdges> {
    // Extract canonical coarse meta-edges and compute base conductance/curvature lengths.
    let mut edge_u = Vec::new();
    let mut edge_v = Vec::new();
    let mut edge_w = Vec::new();
    let w_opt = cg.weights();
    for u in 0..n_l {
        let start = cg.indptr()[u] as usize;
        let end = cg.indptr()[u + 1] as usize;
        for pos in start..end {
            let v = cg.indices()[pos] as usize;
            if v > u {
                let w = match w_opt {
                    Some(ws) => ws[pos].to_f64(),
                    None => 1.0,
                };
                edge_u.push(u as u32);
                edge_v.push(v as u32);
                edge_w.push(w);
            }
        }
    }
    let m_l = edge_u.len();

    let l_cond: Vec<f64> = if m_l == 0 {
        Vec::new()
    } else {
        match config.edge_length_mode {
            CoarseEdgeLengthMode::ConductanceInverse => {
                let mut raw: Vec<f64> = edge_w.iter().map(|&w| 1.0 / w.max(1e-12)).collect();
                let mean_raw = raw.iter().sum::<f64>() / (m_l as f64);
                let inv_mean = if mean_raw.is_finite() && mean_raw > 1e-12 {
                    1.0 / mean_raw
                } else {
                    1.0
                };
                for val in &mut raw {
                    *val *= inv_mean;
                }
                raw
            }
            CoarseEdgeLengthMode::ResistanceCurvatureFlow => {
                // CoarsenedGraph::graph stores aggregated conductances W_{C1, C2}, NOT lengths.
                // Filter non-positive edge weights safely so explicit 0.0 edges do not fail RCF.
                let mut pos_pairs = Vec::with_capacity(m_l);
                let mut pos_weights: Vec<S> = Vec::with_capacity(m_l);
                let mut pos_edge_indices = Vec::with_capacity(m_l);
                for e in 0..m_l {
                    let w = edge_w[e];
                    if w.is_finite() && w > 0.0 {
                        let ws = S::from_f64(w.min(1e30));
                        if ws.is_finite() && ws > S::ZERO {
                            pos_pairs.push((edge_u[e], edge_v[e]));
                            pos_weights.push(ws);
                            pos_edge_indices.push(e);
                        }
                    }
                }
                if pos_pairs.is_empty() {
                    vec![1.0; m_l]
                } else {
                    let cg_rcf = CsrGraph::<S>::from_edges(
                        n_l,
                        &pos_pairs,
                        Some(&pos_weights),
                        false,
                        false,
                    )?;
                    let rcf_cfg = ResistanceCurvatureConfig {
                        steps: config.rcf_steps,
                        eta: config.rcf_eta,
                        resistance: EffectiveResistanceConfig {
                            weights_are_lengths: false,
                            seed: config.seed,
                            ..EffectiveResistanceConfig::default()
                        },
                        ..ResistanceCurvatureConfig::default()
                    };
                    let rcf = resistance_curvature_flow(&cg_rcf, &rcf_cfg)?;
                    if pos_edge_indices.len() == m_l {
                        rcf.edge_lengths
                    } else {
                        let mut max_pos_len = 0.0_f64;
                        for &len in &rcf.edge_lengths {
                            if len.is_finite() && len > max_pos_len {
                                max_pos_len = len;
                            }
                        }
                        let zero_fallback = max_pos_len.max(1.0) * 1e6;
                        let mut out = vec![zero_fallback; m_l];
                        for (&orig_idx, len) in pos_edge_indices.iter().zip(rcf.edge_lengths) {
                            out[orig_idx] = len;
                        }
                        out
                    }
                }
            }
        }
    };

    // Edge-length-first meta-edge augmentation before coarse geodesics.
    let mut l_edge = Vec::with_capacity(m_l);
    for e in 0..m_l {
        let u = edge_u[e] as usize;
        let v = edge_v[e] as usize;
        let len = (l_cond[e] + config.kappa * (radii[u] + radii[v])).max(1e-12);
        l_edge.push(len);
    }

    let mut l_edge_sorted = l_edge.clone();
    let median_l_edge = median_f64_slice(&mut l_edge_sorted);

    Ok(CoarseMetaEdges {
        edge_u,
        edge_v,
        edge_w,
        l_edge,
        median_l_edge,
    })
}

fn build_sp_landmarks<S: Scalar>(
    length_graph: &CsrGraph<S>,
    num_pivots: usize,
    seed: u64,
) -> CoreResult<DogVaradhanLandmarks> {
    let n = length_graph.num_nodes();
    if n == 0 {
        return Ok(DogVaradhanLandmarks {
            num_nodes: 0,
            landmarks: Vec::new(),
            d_dog: Vec::new(),
            d_coarse: None,
        });
    }
    let k_lm = num_pivots.min(n).max(1);
    let landmarks = select_maxmin_sp_pivots(length_graph, k_lm, seed, 1)?;
    let k_actual = landmarks.len();
    let mut d_table = vec![f64::INFINITY; n * k_actual];
    let mut row_col = vec![f64::INFINITY; n];
    let mut ws = DijkstraWorkspace::new(n);
    let mut max_finite = 0.0_f64;

    for (col, &lm) in landmarks.iter().enumerate() {
        ws.dijkstra_from(length_graph, lm as usize, &mut row_col)?;
        for i in 0..n {
            let d = row_col[i];
            d_table[i * k_actual + col] = d;
            if d.is_finite() && d > max_finite {
                max_finite = d;
            }
        }
    }

    let fallback = if max_finite > 0.0 {
        2.0 * max_finite
    } else {
        1.0
    };
    for val in &mut d_table {
        if !val.is_finite() {
            *val = fallback;
        }
    }

    Ok(DogVaradhanLandmarks {
        num_nodes: n,
        landmarks,
        d_dog: d_table,
        d_coarse: None,
    })
}

fn scale_landmarks_to_l_edge(
    landmarks: &mut DogVaradhanLandmarks,
    edge_u: &[u32],
    edge_v: &[u32],
    med_l_edge: f64,
) -> CoreResult<()> {
    if edge_u.is_empty() || med_l_edge <= 0.0 {
        return Ok(());
    }
    let mut edge_alts = landmarks.alt_pair_distances(edge_u, edge_v)?;
    let med_alt = median_f64_slice(&mut edge_alts);
    if med_alt.is_finite() && med_alt > 1e-12 {
        let scale = med_l_edge / med_alt;
        for val in &mut landmarks.d_dog {
            *val *= scale;
        }
        if let Some(dc) = &mut landmarks.d_coarse {
            for val in dc {
                *val *= scale;
            }
        }
    }
    Ok(())
}

pub(super) fn solve_coarse_mds_layout<S: Scalar>(
    n_l: usize,
    meta: &CoarseMetaEdges,
    radii: &[f64],
    max_radius: f64,
    config: &CoarseBackboneConfig,
) -> CoreResult<(DogVaradhanLandmarks, Layout2d<S>)> {
    // Build coarse ideal distance landmark table and PivotMDS initialization.
    let m_l = meta.edge_u.len();
    let pairs: Vec<(u32, u32)> = meta
        .edge_u
        .iter()
        .copied()
        .zip(meta.edge_v.iter().copied())
        .collect();
    let k_lm = config.coarse_pivots.min(n_l).max(1);
    let mut landmarks = if m_l == 0 {
        DogVaradhanLandmarks {
            num_nodes: n_l,
            landmarks: (0..k_lm as u32).collect(),
            d_dog: vec![1.0 + 2.0 * max_radius; n_l * k_lm],
            d_coarse: None,
        }
    } else {
        match config.distance_backend {
            CoarseDistanceBackend::Rhgh => {
                let cond_edge_s: Vec<S> = meta
                    .l_edge
                    .iter()
                    .map(|&len| S::from_f64((1.0 / len).clamp(1e-30, 1e30)))
                    .collect();
                let coarse_conductance_graph =
                    CsrGraph::<S>::from_edges(n_l, &pairs, Some(&cond_edge_s), false, false)?;
                let rhgh_cfg = RhghLandmarkConfig {
                    gamma: config.rhgh_gamma,
                    p_exp: config.rhgh_p,
                    n_landmarks: k_lm,
                    min_degree: 1,
                    normalize_1hop: false,
                    seed: config.seed,
                    ..RhghLandmarkConfig::default()
                };
                let (mut lm, _) =
                    compute_pivoted_rhgh_landmarks(&coarse_conductance_graph, &rhgh_cfg)?;
                scale_landmarks_to_l_edge(&mut lm, &meta.edge_u, &meta.edge_v, meta.median_l_edge)?;
                lm
            }
            CoarseDistanceBackend::ChebyshevDog => {
                let cond_edge_s: Vec<S> = meta
                    .l_edge
                    .iter()
                    .map(|&len| S::from_f64((1.0 / len).clamp(1e-30, 1e30)))
                    .collect();
                let coarse_conductance_graph =
                    CsrGraph::<S>::from_edges(n_l, &pairs, Some(&cond_edge_s), false, false)?;
                let dog_cfg = DogVaradhanConfig {
                    n_landmarks: k_lm,
                    min_degree: 1,
                    seed: config.seed,
                    ..DogVaradhanConfig::default()
                };
                let (mut lm, _) = compute_regularized_dog_varadhan_landmarks(
                    &coarse_conductance_graph,
                    &dog_cfg,
                )?;
                scale_landmarks_to_l_edge(&mut lm, &meta.edge_u, &meta.edge_v, meta.median_l_edge)?;
                lm
            }
            CoarseDistanceBackend::ShortestPath => {
                let l_edge_s: Vec<S> = meta
                    .l_edge
                    .iter()
                    .map(|&len| S::from_f64(len.clamp(1e-12, 1e30)))
                    .collect();
                let coarse_length_graph =
                    CsrGraph::<S>::from_edges(n_l, &pairs, Some(&l_edge_s), false, false)?;
                build_sp_landmarks(&coarse_length_graph, k_lm, config.seed)?
            }
        }
    };

    if m_l == 0 {
        for (col, &lm_id) in landmarks.landmarks.iter().enumerate() {
            landmarks.d_dog[(lm_id as usize) * k_lm + col] = 0.0;
        }
    }

    let mut coarse_layout = pivot_mds_from_distances::<S>(
        landmarks.d_dog(),
        n_l,
        landmarks.num_landmarks(),
        None,
        config.seed,
    )?;

    // Ensure non-degenerate initial coordinates aligned with median_l_edge scale.
    let mut init_spread = 0.0_f64;
    if m_l > 0 {
        let mut edge_2d = Vec::with_capacity(m_l);
        let xs = coarse_layout.x();
        let ys = coarse_layout.y();
        for e in 0..m_l {
            let u = meta.edge_u[e] as usize;
            let v = meta.edge_v[e] as usize;
            edge_2d.push((xs[u].to_f64() - xs[v].to_f64()).hypot(ys[u].to_f64() - ys[v].to_f64()));
        }
        let med_2d = median_f64_slice(&mut edge_2d);
        if med_2d.is_finite() && med_2d > 1e-12 && meta.median_l_edge > 0.0 {
            let sc = meta.median_l_edge / med_2d;
            let (xs_m, ys_m) = coarse_layout.xy_mut();
            for i in 0..n_l {
                xs_m[i] = S::from_f64(xs_m[i].to_f64() * sc);
                ys_m[i] = S::from_f64(ys_m[i].to_f64() * sc);
            }
            init_spread = meta.median_l_edge;
        }
    }
    if init_spread <= 0.0 {
        let area_r = radii
            .iter()
            .map(|&r| {
                let hr = r + 0.5 * config.m_gap;
                hr * hr
            })
            .sum::<f64>()
            .sqrt();
        let target = (1.0 + 2.0 * max_radius).max(2.0 * area_r).max(1.0);
        robust_normalize_layout_2d(
            &mut coarse_layout,
            target,
            config.robust_quantile,
            config.seed,
        )?;
    }

    Ok((landmarks, coarse_layout))
}
